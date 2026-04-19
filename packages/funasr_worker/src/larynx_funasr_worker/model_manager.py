"""Fun-ASR backend lifecycle.

Two backends, mirroring the VoxCPM worker:

* **FunASRBackendReal** — loads both Fun-ASR-Nano-2512 and Fun-ASR-MLT-Nano-2512
  on a single GPU via the ``FunASRNano`` wrapper from ``yuekaizhang/Fun-ASR-vllm``.
  Each checkpoint is a torch model whose encoder runs on GPU and whose
  decoder is handed to a vLLM instance (``m.vllm = LLM(...)``). Both models
  are loaded eagerly at startup — PRD §6 says "dual STT models always loaded"
  (~6 GB combined, easy fit on a 48 GB RTX Pro 6000).

* **MockFunASRBackend** — deterministic, CPU-only. Returns a fixed string
  whose content depends on the audio hash so tests can assert round-trips
  through the IPC without loading the models. Implements ``tokenizer`` as
  a char-level stub so the drop-last-5 helper can be exercised end-to-end.

Mode selection: ``LARYNX_STT_MODE`` env var ("mock" | "funasr").
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

from larynx_funasr_worker.language_router import FunASRModel
from larynx_funasr_worker.streaming_utils import drop_last_n_tokens, tokenizer_from_kwargs

log = structlog.get_logger(__name__)


NANO_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
NANO_VLLM = "yuekai/Fun-ASR-Nano-2512-vllm"
MLT_MODEL = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
MLT_VLLM = "yuekai/Fun-ASR-MLT-Nano-2512-vllm"


class STTMode(StrEnum):
    MOCK = "mock"
    FUNASR = "funasr"


@dataclass(frozen=True)
class TranscribeResult:
    text: str
    language: str
    model_used: FunASRModel


class FunASRBackend(ABC):
    mode: STTMode

    @abstractmethod
    async def transcribe(
        self,
        *,
        audio: NDArray[np.float32],
        sample_rate: int,
        model: FunASRModel,
        funasr_language: str | None,
        hotwords: list[str],
        itn: bool,
        iso_language: str | None,
    ) -> TranscribeResult: ...

    @abstractmethod
    async def transcribe_rolling(
        self,
        *,
        audio: NDArray[np.float32],
        sample_rate: int,
        model: FunASRModel,
        funasr_language: str | None,
        hotwords: list[str],
        itn: bool,
        prev_text: str,
        is_final: bool,
        drop_tail_tokens: int,
        iso_language: str | None,
    ) -> TranscribeResult: ...

    async def close(self) -> None:  # noqa: B027 — intentional no-op default
        """Release GPU / vLLM handles. Override in subclasses that hold resources."""


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class _CharTokenizer:
    """Character-level tokenizer — exactly enough for the drop-last-5 test.

    The real Fun-ASR tokenizer is a Qwen3 BPE vocabulary; we stub with
    chars so mock tests can still assert the shape of the streaming
    helper. Unit tests that need BPE semantics should use the real
    tokenizer on the GPU box (opt-in via RUN_REAL_MODEL=1).
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class MockFunASRBackend(FunASRBackend):
    mode = STTMode.MOCK

    _CANNED: dict[FunASRModel, str] = {
        FunASRModel.NANO: "hello from the mock fun-asr nano backend",
        FunASRModel.MLT: "olá do mock fun-asr mlt backend",
    }

    def __init__(self) -> None:
        self.tokenizer = _CharTokenizer()

    async def transcribe(
        self,
        *,
        audio: NDArray[np.float32],
        sample_rate: int,
        model: FunASRModel,
        funasr_language: str | None,
        hotwords: list[str],
        itn: bool,
        iso_language: str | None,
    ) -> TranscribeResult:
        _ = sample_rate, funasr_language, itn  # intentionally unused in mock
        base = self._CANNED[model]
        # Stir in the audio hash + hotwords so identical inputs produce
        # identical outputs but different audio hashes diverge — enough
        # for integration tests to distinguish cached vs fresh paths.
        digest = hashlib.sha256(audio.tobytes()).hexdigest()[:8]
        tail = f" [{digest}]"
        if hotwords:
            tail += " hotwords=" + ",".join(hotwords)
        return TranscribeResult(
            text=base + tail,
            language=iso_language or ("en" if model is FunASRModel.NANO else "pt"),
            model_used=model,
        )

    async def transcribe_rolling(
        self,
        *,
        audio: NDArray[np.float32],
        sample_rate: int,
        model: FunASRModel,
        funasr_language: str | None,
        hotwords: list[str],
        itn: bool,
        prev_text: str,
        is_final: bool,
        drop_tail_tokens: int,
        iso_language: str | None,
    ) -> TranscribeResult:
        base = await self.transcribe(
            audio=audio,
            sample_rate=sample_rate,
            model=model,
            funasr_language=funasr_language,
            hotwords=hotwords,
            itn=itn,
            iso_language=iso_language,
        )
        text = prev_text + base.text if prev_text else base.text
        if not is_final:
            text = drop_last_n_tokens(text, self.tokenizer, drop_tail_tokens)
        return TranscribeResult(text=text, language=base.language, model_used=base.model_used)


# ---------------------------------------------------------------------------
# Real backend — Fun-ASR-vllm
# ---------------------------------------------------------------------------


@dataclass
class _LoadedModel:
    """One Fun-ASR checkpoint + its attached vLLM instance."""

    handle: Any  # FunASRNano instance
    kwargs: dict[str, Any]  # carries tokenizer, frontend, model_path, device, ...
    tokenizer: Any  # == kwargs["tokenizer"], pre-extracted for hot-path access


class FunASRBackendReal(FunASRBackend):
    """Loads Nano + MLT-Nano together and routes each request by model.

    Both checkpoints share the same vLLM process (separate ``LLM``
    instances on the same device); the torch encoder side stays on the
    Python process calling ``m.inference``. Each inference call is
    executed in a thread via ``asyncio.to_thread`` because FunASRNano is
    sync.
    """

    mode = STTMode.FUNASR

    def __init__(
        self,
        *,
        gpu: int = 1,
        gpu_memory_utilization: float = 0.4,
        sampling_max_tokens: int = 500,
        sampling_top_p: float = 0.001,
    ) -> None:
        import asyncio

        self._gpu = gpu
        self._gpu_mem = gpu_memory_utilization
        self._max_tokens = sampling_max_tokens
        self._top_p = sampling_top_p
        self._models: dict[FunASRModel, _LoadedModel] = {}
        self._sampling_params: Any | None = None
        # Model-level lock. FunASRNano wraps a shared vLLM ``LLM``
        # instance that's not safe to call from multiple threads at once
        # — under 4-way ``asyncio.to_thread`` contention the underlying
        # ``.inference()`` hangs inside vLLM with no recovery, starving
        # every session after the first. Mirrors the pattern the sibling
        # VAD worker already uses for its own fsmn-vad wrapper; see
        # ``FunasrStreamingVad._model_lock`` in
        # ``packages/vad_punc_worker/.../streaming_vad.py`` and
        # ``bugs/001_concurrent_stt.md`` § 2.2 for the full trace.
        self._model_lock = asyncio.Lock()

    async def load(self) -> None:
        import asyncio

        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            from model import FunASRNano  # provided by Fun-ASR-vllm repo on sys.path
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise RuntimeError(
                "LARYNX_STT_MODE=funasr requires Fun-ASR + Fun-ASR-vllm installed. "
                "On the GPU box: uv sync --extra gpu, then ensure the Fun-ASR-vllm "
                "checkout is on PYTHONPATH (LARYNX_FUNASR_VLLM_DIR)."
            ) from e

        device = f"cuda:{self._gpu}"
        log.info("funasr.loading", device=device, nano=NANO_MODEL, mlt=MLT_MODEL)

        # vLLM spawns a child subprocess for its GPU worker. The child only
        # sees the devices listed in CUDA_VISIBLE_DEVICES, and vLLM ignores
        # the torch device hint on the parent-side FunASRNano call. We
        # temporarily pin CUDA_VISIBLE_DEVICES to the target GPU around each
        # LLM() init so the worker lands on GPU 1 and doesn't fight VoxCPM
        # on GPU 0 for the KV-cache reservation.
        import contextlib

        @contextlib.contextmanager
        def _pinned_visible_devices(gpu: int) -> Any:
            prev = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            try:
                yield
            finally:
                if prev is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev

        nano, nano_kwargs = FunASRNano.from_pretrained(model=NANO_MODEL, device=device)
        nano.eval()
        with _pinned_visible_devices(self._gpu):
            nano_vllm = LLM(
                model=NANO_VLLM,
                enable_prompt_embeds=True,
                gpu_memory_utilization=self._gpu_mem,
                dtype="bfloat16",
            )
        nano.vllm = nano_vllm
        nano.vllm_sampling_params = SamplingParams(top_p=self._top_p, max_tokens=self._max_tokens)
        self._models[FunASRModel.NANO] = _LoadedModel(
            handle=nano,
            kwargs=nano_kwargs,
            tokenizer=tokenizer_from_kwargs(nano_kwargs),
        )
        log.info("funasr.loaded", checkpoint="nano")

        mlt, mlt_kwargs = FunASRNano.from_pretrained(model=MLT_MODEL, device=device)
        mlt.eval()
        with _pinned_visible_devices(self._gpu):
            mlt_vllm = LLM(
                model=MLT_VLLM,
                enable_prompt_embeds=True,
                gpu_memory_utilization=self._gpu_mem,
                dtype="bfloat16",
            )
        mlt.vllm = mlt_vllm
        mlt.vllm_sampling_params = SamplingParams(top_p=self._top_p, max_tokens=self._max_tokens)
        self._models[FunASRModel.MLT] = _LoadedModel(
            handle=mlt,
            kwargs=mlt_kwargs,
            tokenizer=tokenizer_from_kwargs(mlt_kwargs),
        )
        log.info("funasr.loaded", checkpoint="mlt")

    def tokenizer_for(self, model: FunASRModel) -> Any:
        """Exposed so test code and the streaming helper can grab the
        actual tokenizer for a given checkpoint (never a module-level one)."""
        loaded = self._models.get(model)
        if loaded is None:
            raise RuntimeError(f"model {model} not loaded")
        return loaded.tokenizer

    async def transcribe(
        self,
        *,
        audio: NDArray[np.float32],
        sample_rate: int,
        model: FunASRModel,
        funasr_language: str | None,
        hotwords: list[str],
        itn: bool,
        iso_language: str | None,
    ) -> TranscribeResult:
        import asyncio

        import torch

        loaded = self._require(model)
        audio_t = torch.from_numpy(np.ascontiguousarray(audio)).float()

        inf_kwargs: dict[str, Any] = dict(loaded.kwargs)
        inf_kwargs["hotwords"] = list(hotwords)
        inf_kwargs["itn"] = itn
        if funasr_language is not None:
            inf_kwargs["language"] = funasr_language

        def _run() -> str:
            res = loaded.handle.inference(data_in=[audio_t], **inf_kwargs)
            # res shape: (results_list, meta_data); results_list[0] is a dict
            # with a "text" key (see FunASRNano.inference in Fun-ASR-vllm).
            return res[0][0]["text"]

        async with self._model_lock:
            text = await asyncio.to_thread(_run)
        return TranscribeResult(
            text=text,
            language=iso_language or _default_iso_for(model),
            model_used=model,
        )

    async def transcribe_rolling(
        self,
        *,
        audio: NDArray[np.float32],
        sample_rate: int,
        model: FunASRModel,
        funasr_language: str | None,
        hotwords: list[str],
        itn: bool,
        prev_text: str,
        is_final: bool,
        drop_tail_tokens: int,
        iso_language: str | None,
    ) -> TranscribeResult:
        import asyncio

        import torch

        loaded = self._require(model)
        audio_t = torch.from_numpy(np.ascontiguousarray(audio)).float()

        inf_kwargs: dict[str, Any] = dict(loaded.kwargs)
        inf_kwargs["hotwords"] = list(hotwords)
        inf_kwargs["itn"] = itn
        inf_kwargs["prev_text"] = prev_text
        if funasr_language is not None:
            inf_kwargs["language"] = funasr_language

        def _run() -> str:
            res = loaded.handle.inference(data_in=[audio_t], **inf_kwargs)
            return res[0][0]["text"]

        async with self._model_lock:
            text = await asyncio.to_thread(_run)
        if not is_final:
            text = drop_last_n_tokens(text, loaded.tokenizer, drop_tail_tokens)
        return TranscribeResult(
            text=text,
            language=iso_language or _default_iso_for(model),
            model_used=model,
        )

    def _require(self, model: FunASRModel) -> _LoadedModel:
        loaded = self._models.get(model)
        if loaded is None:
            raise RuntimeError(f"fun-asr {model} checkpoint not loaded")
        return loaded

    async def close(self) -> None:
        # vLLM doesn't expose a clean teardown hook; dropping the refs lets
        # Python GC the torch side. The CUDA context exits with the process.
        self._models.clear()


def _default_iso_for(model: FunASRModel) -> str:
    """Best-effort ISO when the caller didn't provide one.

    Nano defaults to "en" because its auto-detect happily switches to
    zh/ja on its own; MLT's input is only reached with an explicit
    language param, so this fallback only fires as a defensive default.
    """
    return "en" if model is FunASRModel.NANO else "pt"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class FunASRModelManager:
    def __init__(self, backend: FunASRBackend) -> None:
        self.backend = backend
        self.mode = backend.mode

    @classmethod
    async def from_env(cls) -> FunASRModelManager:
        raw = os.environ.get("LARYNX_STT_MODE", "mock").lower()
        try:
            mode = STTMode(raw)
        except ValueError as e:
            raise RuntimeError(f"LARYNX_STT_MODE must be 'mock' or 'funasr', got {raw!r}") from e

        if mode is STTMode.MOCK:
            log.info("funasr.mode", mode="mock")
            return cls(MockFunASRBackend())

        gpu = int(os.environ.get("LARYNX_FUNASR_GPU", "1"))
        gpu_mem = float(os.environ.get("LARYNX_FUNASR_GPU_MEM_UTIL", "0.4"))
        backend = FunASRBackendReal(gpu=gpu, gpu_memory_utilization=gpu_mem)
        await backend.load()
        log.info("funasr.mode", mode="funasr", gpu=gpu)
        return cls(backend)

    async def close(self) -> None:
        await self.backend.close()
