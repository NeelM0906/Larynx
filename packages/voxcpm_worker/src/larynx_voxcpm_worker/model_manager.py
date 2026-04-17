"""Manages the VoxCPM2 backend lifecycle.

Two backends:

* ``VoxCPMBackendReal`` — adapter around
  ``nanovllm_voxcpm.models.voxcpm2.server.AsyncVoxCPM2ServerPool``. The
  upstream library spawns its own CUDA subprocess (via torch.multiprocessing),
  so this class only owns the async handle + our encode / generate contract.

* ``MockVoxCPMBackend`` — deterministic sine-wave generator that honours the
  full voice-cloning contract (encode_reference + ref_audio_latents vary the
  pitch so different voices sound different). Lets every gateway/IPC code
  path be exercised on a CPU-only box.

Mode selection: ``LARYNX_TTS_MODE`` env var (see .env.example).
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import struct
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import librosa
import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger(__name__)


def _env_bool(name: str, *, default: bool) -> bool:
    """Parse an env var as a bool; accepts 1/0, true/false, yes/no (any case)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Values read out of a loaded VoxCPM2 via get_model_info(); the mock
# backend mirrors them so callers don't branch on mode. The encoder takes
# 16 kHz audio and the decoder emits 48 kHz; we expose these explicitly
# so the gateway can decide where to resample.
DEFAULT_ENCODER_SR = 16000
DEFAULT_OUTPUT_SR = 48000

# Each latent row is float32 × feat_dim; VoxCPM2 uses feat_dim=64. The
# engine requires num_frames % patch_size == 0 with patch_size=4.
MOCK_FEAT_DIM = 64
MOCK_PATCH_SIZE = 4


class ModelMode(StrEnum):
    MOCK = "mock"
    VOXCPM = "voxcpm"


@dataclass(frozen=True)
class ModelInfo:
    """Shape information exposed by the loaded model."""

    encoder_sample_rate: int
    output_sample_rate: int
    feat_dim: int


class VoxCPMBackend(ABC):
    """Async interface every backend implements.

    Kept async even in mock mode so the real backend (which genuinely awaits
    AsyncVoxCPM2ServerPool.generate) doesn't need a separate interface.
    """

    mode: ModelMode

    @abstractmethod
    async def get_info(self) -> ModelInfo: ...

    @abstractmethod
    async def encode_reference(self, audio: bytes, wav_format: str = "wav") -> bytes: ...

    @abstractmethod
    async def synthesize(
        self,
        *,
        text: str,
        ref_audio_latents: bytes | None = None,
        prompt_audio_latents: bytes | None = None,
        prompt_text: str = "",
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        max_generate_length: int = 2000,
        lora_name: str | None = None,
    ) -> NDArray[np.float32]: ...

    @abstractmethod
    def synthesize_stream(
        self,
        *,
        text: str,
        ref_audio_latents: bytes | None = None,
        prompt_audio_latents: bytes | None = None,
        prompt_text: str = "",
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        max_generate_length: int = 2000,
        lora_name: str | None = None,
    ) -> AsyncIterator[NDArray[np.float32]]:
        """Async generator yielding chunks of float32 mono audio as they're
        produced at the model's native output sample rate.

        Callers (worker server) resample each chunk at emission time so the
        WebSocket can begin flushing audio without waiting for the full
        utterance. The final chunk is whatever the model produces last — no
        explicit terminator is yielded; iteration just ends.
        """
        ...

    # -- LoRA hot-swap --------------------------------------------------------
    #
    # Mirrors ``AsyncVoxCPM2ServerPool.{register,unregister,list}_lora``.
    # Registration is CPU-resident for the life of the worker; the engine
    # manages GPU-slot LRU internally. ``synthesize(..., lora_name=...)``
    # selects per-request. See ORCHESTRATION-M7.md §3.

    @abstractmethod
    async def load_lora(self, name: str, path: str) -> None:
        """Register a LoRA by directory path. Duplicate name -> ValueError."""

    @abstractmethod
    async def unload_lora(self, name: str) -> None:
        """Unregister a LoRA. Unknown name -> ValueError."""

    @abstractmethod
    async def list_loras(self) -> list[str]:
        """Names of currently-registered LoRAs."""

    async def close(self) -> None:  # noqa: B027 — intentional no-op default
        """Release resources. Override in subclasses that hold GPU memory."""


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockVoxCPMBackend(VoxCPMBackend):
    """Deterministic sine-wave generator honoring the voice-cloning contract.

    - ``encode_reference(audio)``: decodes WAV, reduces to a 64-float digest
      that depends on the audio content; returns it as raw float32 bytes that
      round-trip through the cache layer just like real VoxCPM2 latents.
    - ``synthesize(text, ref_audio_latents=...)``: text hash controls base
      pitch; when ``ref_audio_latents`` is given, the latents' first sample
      tilts the pitch further so identical ref + identical text -> identical
      audio, but different refs produce audibly different output.

    This is enough for integration tests to prove caching, voice lookup,
    and request plumbing end-to-end without loading the 4.7GB model.
    """

    MS_PER_CHAR = 60
    MIN_DURATION_MS = 400
    mode = ModelMode.MOCK

    def __init__(
        self,
        encoder_sample_rate: int = DEFAULT_ENCODER_SR,
        output_sample_rate: int = DEFAULT_OUTPUT_SR,
        feat_dim: int = MOCK_FEAT_DIM,
    ) -> None:
        self._encoder_sr = encoder_sample_rate
        self._output_sr = output_sample_rate
        self._feat_dim = feat_dim
        # LoRA registry: name -> path (path never validated here; the real
        # backend validates at nanovllm register_lora time — mock stays dumb
        # so unit tests don't have to ship a real LoRA artifact).
        self._loras: dict[str, str] = {}

    async def get_info(self) -> ModelInfo:
        return ModelInfo(
            encoder_sample_rate=self._encoder_sr,
            output_sample_rate=self._output_sr,
            feat_dim=self._feat_dim,
        )

    async def encode_reference(self, audio: bytes, wav_format: str = "wav") -> bytes:
        if not audio:
            raise ValueError("audio must not be empty")
        # Decode to mono float32 at encoder_sr so the resulting digest
        # reflects the ACTUAL audio content, not just the input bytes
        # (otherwise identical WAV written with/without metadata would
        # produce different latents).
        samples, _ = librosa.load(io.BytesIO(audio), sr=self._encoder_sr, mono=True)
        # Project the signal onto `feat_dim` fixed-frequency basis functions
        # to get a content-dependent embedding, then tile across enough
        # frames to match what VoxCPM2 produces (~25 frames per second).
        digest = hashlib.sha256(samples.tobytes()).digest()
        seed = int.from_bytes(digest[:4], "little")
        rng = np.random.default_rng(seed)
        num_frames = max(32, len(samples) // (self._encoder_sr // 25))
        latents = rng.standard_normal((num_frames, self._feat_dim)).astype(np.float32) * 0.1
        # Patch size alignment — VoxCPM2 requires (num_frames % patch_size == 0).
        pad = (-num_frames) % MOCK_PATCH_SIZE
        if pad:
            latents = np.concatenate([latents, np.tile(latents[-1:], (pad, 1))], axis=0)
        return latents.tobytes()

    async def synthesize(
        self,
        *,
        text: str,
        ref_audio_latents: bytes | None = None,
        prompt_audio_latents: bytes | None = None,
        prompt_text: str = "",
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        max_generate_length: int = 2000,
        lora_name: str | None = None,
    ) -> NDArray[np.float32]:
        if not text:
            raise ValueError("text must not be empty")
        if lora_name is not None and lora_name not in self._loras:
            raise ValueError(f"LoRA {lora_name!r} is not registered")

        duration_ms = max(self.MIN_DURATION_MS, self.MS_PER_CHAR * len(text))
        num_samples = int(self._output_sr * duration_ms / 1000)

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        base_freq = 120.0 + (digest[0] / 255.0) * 400.0  # 120 - 520 Hz

        # Voice conditioning: a voice's latents shift the pitch by up to
        # one octave so the same text with different voices sounds different.
        voice_shift = 0.0
        if ref_audio_latents is not None and len(ref_audio_latents) >= 4:
            (first_float,) = struct.unpack("<f", ref_audio_latents[:4])
            voice_shift = float(np.clip(first_float, -1.0, 1.0)) * 60.0
        if prompt_audio_latents is not None and len(prompt_audio_latents) >= 4:
            (first_float,) = struct.unpack("<f", prompt_audio_latents[:4])
            voice_shift += float(np.clip(first_float, -1.0, 1.0)) * 30.0
        # LoRA conditioning: hash the name into a deterministic offset.
        # ±80 Hz range lets a LoRA clearly differ from the base text-hash
        # pitch without saturating against voice-latent shifts.
        if lora_name is not None:
            lora_digest = hashlib.sha256(lora_name.encode("utf-8")).digest()
            lora_byte = (lora_digest[0] / 255.0) * 2.0 - 1.0
            voice_shift += lora_byte * 80.0
        freq_hz = base_freq + voice_shift

        t = np.arange(num_samples, dtype=np.float32) / self._output_sr
        signal = 0.25 * np.sin(2.0 * math.pi * freq_hz * t, dtype=np.float32)
        fade_samples = min(int(self._output_sr * 0.01), num_samples // 4)
        if fade_samples > 0:
            ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            signal[:fade_samples] *= ramp
            signal[-fade_samples:] *= ramp[::-1]
        return signal

    async def synthesize_stream(
        self,
        *,
        text: str,
        ref_audio_latents: bytes | None = None,
        prompt_audio_latents: bytes | None = None,
        prompt_text: str = "",
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        max_generate_length: int = 2000,
        lora_name: str | None = None,
    ) -> AsyncIterator[NDArray[np.float32]]:
        # Delegate to the one-shot path and slice the result into chunks so
        # the streaming code path is exercised in mock mode. ~120ms chunks
        # roughly match what nano-vllm-voxcpm emits in real mode.
        import asyncio

        full = await self.synthesize(
            text=text,
            ref_audio_latents=ref_audio_latents,
            prompt_audio_latents=prompt_audio_latents,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            temperature=temperature,
            max_generate_length=max_generate_length,
            lora_name=lora_name,
        )
        chunk_samples = int(self._output_sr * 0.12)
        if chunk_samples <= 0:
            yield full
            return
        for start in range(0, len(full), chunk_samples):
            yield full[start : start + chunk_samples]
            # Yield to the loop so the worker can interleave cancels / other
            # sessions; real backend awaits the GPU between chunks naturally.
            await asyncio.sleep(0)

    # -- LoRA hot-swap (mock) -----------------------------------------------

    async def load_lora(self, name: str, path: str) -> None:
        if name in self._loras:
            raise ValueError(f"LoRA {name!r} is already registered")
        self._loras[name] = path

    async def unload_lora(self, name: str) -> None:
        if name not in self._loras:
            raise ValueError(f"LoRA {name!r} is not registered")
        del self._loras[name]

    async def list_loras(self) -> list[str]:
        return sorted(self._loras)


# ---------------------------------------------------------------------------
# Real backend — nano-vllm-voxcpm adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRAInitConfig:
    """Init-time LoRA constraints for nano-vllm-voxcpm.

    These are fixed once per worker boot — ``max_loras`` / ``max_lora_rank``
    control GPU slot pool shape, ``enable_*`` and ``target_modules_*`` must
    be supersets of every trained LoRA we'll serve. See ORCHESTRATION-M7.md
    §3.2 for the reasoning behind the defaults.
    """

    max_loras: int = 8
    max_lora_rank: int = 32
    enable_lm: bool = True
    enable_dit: bool = True
    enable_proj: bool = False
    target_modules_lm: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    target_modules_dit: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    target_proj_modules: tuple[str, ...] = (
        "enc_to_lm_proj",
        "lm_to_dit_proj",
        "res_to_dit_proj",
    )


class VoxCPMBackendReal(VoxCPMBackend):
    """Adapter around ``nanovllm_voxcpm``'s ``AsyncVoxCPM2ServerPool``.

    The upstream server spawns a torch.multiprocessing child process per
    device and owns the CUDA context. We only hold the async handle; all
    inference happens in the child.
    """

    mode = ModelMode.VOXCPM

    def __init__(
        self,
        model: str = "openbmb/VoxCPM2",
        gpu: int = 0,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 8192,
        max_num_seqs: int = 16,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.80,
        lora_config: LoRAInitConfig | None = None,
    ) -> None:
        self._model_ref = model
        self._gpu = gpu
        self._inference_timesteps = inference_timesteps
        self._max_num_batched_tokens = max_num_batched_tokens
        self._max_num_seqs = max_num_seqs
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._lora_config = lora_config
        self._pool: Any | None = None
        self._info: ModelInfo | None = None

    async def load(self) -> None:
        try:
            from nanovllm_voxcpm import VoxCPM
        except ImportError as e:
            raise RuntimeError(
                "LARYNX_TTS_MODE=voxcpm requires nano-vllm-voxcpm. Install "
                "with `uv sync --extra gpu` on the GPU box."
            ) from e

        lora_runtime_config = None
        if self._lora_config is not None:
            # Import lazily so the mock-mode CI box doesn't need nano-vllm
            # installed at all. The config type lives in the upstream
            # package; we marshal our frozen dataclass into it.
            from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

            lora_runtime_config = LoRAConfig(
                enable_lm=self._lora_config.enable_lm,
                enable_dit=self._lora_config.enable_dit,
                enable_proj=self._lora_config.enable_proj,
                max_loras=self._lora_config.max_loras,
                max_lora_rank=self._lora_config.max_lora_rank,
                target_modules_lm=list(self._lora_config.target_modules_lm),
                target_modules_dit=list(self._lora_config.target_modules_dit),
                target_proj_modules=list(self._lora_config.target_proj_modules),
            )

        log.info(
            "voxcpm.loading",
            gpu=self._gpu,
            model=self._model_ref,
            lora_enabled=lora_runtime_config is not None,
        )
        self._pool = VoxCPM.from_pretrained(
            model=self._model_ref,
            inference_timesteps=self._inference_timesteps,
            max_num_batched_tokens=self._max_num_batched_tokens,
            max_num_seqs=self._max_num_seqs,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_memory_utilization,
            enforce_eager=False,
            devices=[self._gpu],
            lora_config=lora_runtime_config,
        )
        # AsyncVoxCPM2ServerPool.wait_for_ready awaits model init in the
        # child process; model_info has the real feat_dim and sample rates.
        await self._pool.wait_for_ready()
        info = await self._pool.get_model_info()
        self._info = ModelInfo(
            encoder_sample_rate=int(info["encoder_sample_rate"]),
            output_sample_rate=int(info["output_sample_rate"]),
            feat_dim=int(info["feat_dim"]),
        )
        log.info(
            "voxcpm.loaded",
            gpu=self._gpu,
            encoder_sr=self._info.encoder_sample_rate,
            output_sr=self._info.output_sample_rate,
            feat_dim=self._info.feat_dim,
        )

    async def get_info(self) -> ModelInfo:
        if self._info is None:
            raise RuntimeError("backend not loaded; call load() first")
        return self._info

    async def encode_reference(self, audio: bytes, wav_format: str = "wav") -> bytes:
        if self._pool is None:
            raise RuntimeError("backend not loaded")
        return await self._pool.encode_latents(audio, wav_format)

    async def synthesize(
        self,
        *,
        text: str,
        ref_audio_latents: bytes | None = None,
        prompt_audio_latents: bytes | None = None,
        prompt_text: str = "",
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        max_generate_length: int = 2000,
        lora_name: str | None = None,
    ) -> NDArray[np.float32]:
        chunks: list[NDArray[np.float32]] = []
        async for chunk in self.synthesize_stream(
            text=text,
            ref_audio_latents=ref_audio_latents,
            prompt_audio_latents=prompt_audio_latents,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            temperature=temperature,
            max_generate_length=max_generate_length,
            lora_name=lora_name,
        ):
            chunks.append(chunk)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    async def synthesize_stream(
        self,
        *,
        text: str,
        ref_audio_latents: bytes | None = None,
        prompt_audio_latents: bytes | None = None,
        prompt_text: str = "",
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        max_generate_length: int = 2000,
        lora_name: str | None = None,
    ) -> AsyncIterator[NDArray[np.float32]]:
        if self._pool is None:
            raise RuntimeError("backend not loaded")
        if not text:
            raise ValueError("text must not be empty")

        async for chunk in self._pool.generate(
            target_text=text,
            prompt_latents=prompt_audio_latents,
            prompt_text=prompt_text,
            prompt_id=None,
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
            ref_audio_latents=ref_audio_latents,
            lora_name=lora_name,
        ):
            yield np.asarray(chunk, dtype=np.float32).reshape(-1)

    # -- LoRA hot-swap (real) -----------------------------------------------

    async def load_lora(self, name: str, path: str) -> None:
        if self._pool is None:
            raise RuntimeError("backend not loaded")
        if self._lora_config is None:
            raise RuntimeError(
                "LoRA is not enabled on this backend; pass lora_config to "
                "VoxCPMBackendReal() at init or set LARYNX_VOXCPM_LORA_* env."
            )
        await self._pool.register_lora(name, path)

    async def unload_lora(self, name: str) -> None:
        if self._pool is None:
            raise RuntimeError("backend not loaded")
        await self._pool.unregister_lora(name)

    async def list_loras(self) -> list[str]:
        if self._pool is None:
            raise RuntimeError("backend not loaded")
        entries = await self._pool.list_loras()
        # Upstream returns LoRAInfo objects with a `.name` attr.
        return sorted(entry.name for entry in entries)

    async def close(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is not None:
            await pool.stop()


# ---------------------------------------------------------------------------
# Manager — picks mode + resamples worker output to the caller's target SR
# ---------------------------------------------------------------------------


class VoxCPMModelManager:
    """Orchestrates the active backend and the sample-rate bridge to callers.

    Worker output is at the model's native SR (24 kHz for VoxCPM2). Callers
    request arbitrary sample rates; we resample via librosa at the edge.
    """

    def __init__(self, backend: VoxCPMBackend) -> None:
        self.backend = backend
        self.mode = backend.mode

    @classmethod
    async def from_env(cls) -> VoxCPMModelManager:
        raw = os.environ.get("LARYNX_TTS_MODE", "mock").lower()
        try:
            mode = ModelMode(raw)
        except ValueError as e:
            raise RuntimeError(f"LARYNX_TTS_MODE must be 'mock' or 'voxcpm', got {raw!r}") from e

        if mode is ModelMode.MOCK:
            log.info("voxcpm.mode", mode="mock")
            return cls(MockVoxCPMBackend())

        gpu = int(os.environ.get("LARYNX_VOXCPM_GPU", "0"))
        model_ref = os.environ.get("LARYNX_VOXCPM_MODEL", "openbmb/VoxCPM2")

        # LoRA config is opt-in via LARYNX_VOXCPM_LORA_ENABLED so non-M7
        # deployments don't pay the GPU-slot preallocation cost. Defaults
        # match ORCHESTRATION-M7.md §3.2.
        lora_config: LoRAInitConfig | None = None
        if _env_bool("LARYNX_VOXCPM_LORA_ENABLED", default=True):
            lora_config = LoRAInitConfig(
                max_loras=int(os.environ.get("LARYNX_VOXCPM_LORA_MAX_LORAS", "8")),
                max_lora_rank=int(os.environ.get("LARYNX_VOXCPM_LORA_MAX_RANK", "32")),
                enable_lm=_env_bool("LARYNX_VOXCPM_LORA_ENABLE_LM", default=True),
                enable_dit=_env_bool("LARYNX_VOXCPM_LORA_ENABLE_DIT", default=True),
                enable_proj=_env_bool("LARYNX_VOXCPM_LORA_ENABLE_PROJ", default=False),
            )

        backend = VoxCPMBackendReal(
            model=model_ref,
            gpu=gpu,
            inference_timesteps=int(os.environ.get("LARYNX_VOXCPM_INFERENCE_TIMESTEPS", "10")),
            max_num_batched_tokens=int(os.environ.get("LARYNX_VOXCPM_MAX_BATCHED_TOKENS", "8192")),
            max_num_seqs=int(os.environ.get("LARYNX_VOXCPM_MAX_NUM_SEQS", "16")),
            max_model_len=int(os.environ.get("LARYNX_VOXCPM_MAX_MODEL_LEN", "4096")),
            gpu_memory_utilization=float(os.environ.get("LARYNX_VOXCPM_GPU_MEM_UTIL", "0.80")),
            lora_config=lora_config,
        )
        await backend.load()
        log.info("voxcpm.mode", mode="voxcpm", gpu=gpu, lora=lora_config is not None)
        return cls(backend)

    async def close(self) -> None:
        await self.backend.close()

    @staticmethod
    def resample(
        samples: NDArray[np.float32], source_sr: int, target_sr: int
    ) -> NDArray[np.float32]:
        if source_sr == target_sr or samples.size == 0:
            return samples
        return librosa.resample(
            samples.astype(np.float32), orig_sr=source_sr, target_sr=target_sr
        ).astype(np.float32)
