"""VAD + Punctuation backend lifecycle.

Two backends:

* **VadPuncBackendReal** — loads ``funasr.AutoModel(model="fsmn-vad")``
  and ``funasr.AutoModel(model="ct-punc")`` on CPU. ``ct-punc`` is the
  canonical FunASR punctuation checkpoint (alias for
  ``iic/punc_ct-transformer_cn-en-common-vocab471067-large``). We
  deliberately pick ``ct-punc`` over ``ct-punc-c`` because (a) it's the
  default in the FunASR README's combined-pipeline example and (b) the
  ``-c`` (controllable) variant needs a ``vad_pre_num`` context arg that
  would plumb from the streaming path we don't wire until M4. If FunASR
  later retires ``ct-punc`` we can flip this single import.

* **MockVadPuncBackend** — deterministic, no dependencies. Segmentation
  returns a single speech span spanning the whole clip; punctuation
  appends a period + capitalises the first letter. Enough to exercise
  the gateway's silence-trim path + the "punctuate=True/False"
  branching without pulling FunASR into the default install.

Mode selection: ``LARYNX_VAD_PUNC_MODE`` env var ("mock" | "real").
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger(__name__)


# Fun-ASR's itn=True output is already punctuated for every MLT language
# except the CJK three; ct-punc is bilingual zh/en. Listing the codes it
# helps with keeps the short-circuit explicit in one place.
CT_PUNC_LANGUAGES: frozenset[str] = frozenset({"zh", "en"})


class VadPuncMode(StrEnum):
    MOCK = "mock"
    REAL = "real"


@dataclass(frozen=True)
class VadSegment:
    start_ms: int
    end_ms: int
    is_speech: bool = True


class VadPuncBackend(ABC):
    mode: VadPuncMode

    @abstractmethod
    async def segment(self, audio: NDArray[np.float32]) -> list[VadSegment]: ...

    @abstractmethod
    async def punctuate(self, text: str, language: str | None) -> tuple[str, bool]: ...

    async def close(self) -> None:  # noqa: B027 — intentional no-op default
        """Release model handles. Override in subclasses that hold resources."""


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockVadPuncBackend(VadPuncBackend):
    mode = VadPuncMode.MOCK

    SAMPLE_RATE = 16000
    # Anything louder than this RMS threshold over a 30 ms window counts
    # as "speech" in the mock. Picked empirically so silence from soundfile
    # zeros (rms=0) clearly differentiates from real-looking noise.
    RMS_THRESHOLD = 0.005

    async def segment(self, audio: NDArray[np.float32]) -> list[VadSegment]:
        if audio.size == 0:
            return []
        # Approximate VAD by looking for the first and last windows above
        # the RMS floor. Fine for mock + for trimming silence in tests.
        win = max(1, self.SAMPLE_RATE // 100)  # 10 ms frames
        n = len(audio) // win
        if n == 0:
            rms_full = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            if rms_full < self.RMS_THRESHOLD:
                return []
            return [VadSegment(0, int(1000 * len(audio) / self.SAMPLE_RATE), True)]

        frames = audio[: n * win].reshape(n, win)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        voiced = rms > self.RMS_THRESHOLD
        if not voiced.any():
            return []
        start = int(np.argmax(voiced))
        end = int(n - np.argmax(voiced[::-1]))
        ms_per_frame = 1000 * win / self.SAMPLE_RATE
        return [
            VadSegment(
                start_ms=int(start * ms_per_frame),
                end_ms=int(end * ms_per_frame),
                is_speech=True,
            )
        ]

    async def punctuate(self, text: str, language: str | None) -> tuple[str, bool]:
        code = (language or "").split("-")[0].split("_")[0].lower()
        if code and code not in CT_PUNC_LANGUAGES:
            # MLT languages: Fun-ASR's itn=True already punctuated inline.
            return text, False
        stripped = text.strip()
        if not stripped:
            return text, False
        punctuated = stripped[0].upper() + stripped[1:]
        if punctuated[-1] not in ".!?":
            punctuated += "."
        return punctuated, True


# ---------------------------------------------------------------------------
# Real backend
# ---------------------------------------------------------------------------


class VadPuncBackendReal(VadPuncBackend):
    mode = VadPuncMode.REAL

    def __init__(
        self,
        *,
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
    ) -> None:
        self._vad_name = vad_model
        self._punc_name = punc_model
        self._vad: Any | None = None
        self._punc: Any | None = None

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise RuntimeError(
                "LARYNX_VAD_PUNC_MODE=real requires funasr. Install with "
                "`uv sync --extra real` on the target box."
            ) from e

        log.info("vad_punc.loading", vad=self._vad_name, punc=self._punc_name)
        self._vad = AutoModel(model=self._vad_name, disable_update=True)
        self._punc = AutoModel(model=self._punc_name, disable_update=True)
        log.info("vad_punc.loaded")

    async def segment(self, audio: NDArray[np.float32]) -> list[VadSegment]:
        if self._vad is None:
            raise RuntimeError("vad model not loaded")
        vad = self._vad

        def _run() -> list[VadSegment]:
            res = vad.generate(input=audio)
            # fsmn-vad offline returns [{"key": ..., "value": [[start_ms, end_ms], ...]}]
            if not res:
                return []
            pairs = res[0].get("value") or []
            out: list[VadSegment] = []
            for pair in pairs:
                if not pair or len(pair) != 2:
                    continue
                start, end = int(pair[0]), int(pair[1])
                if start < 0 or end < 0:
                    # Streaming sentinel values; offline path shouldn't
                    # produce these but skip defensively.
                    continue
                out.append(VadSegment(start_ms=start, end_ms=end, is_speech=True))
            return out

        return await asyncio.to_thread(_run)

    async def punctuate(self, text: str, language: str | None) -> tuple[str, bool]:
        if not text.strip():
            return text, False
        code = (language or "").split("-")[0].split("_")[0].lower()
        if code and code not in CT_PUNC_LANGUAGES:
            # ct-punc is bilingual zh/en. For MLT languages Fun-ASR's
            # itn=True output already contains inline punctuation (the
            # LLM decoder emits it), so re-running ct-punc would fight
            # the existing punctuation. Document that choice here.
            return text, False
        if self._punc is None:
            raise RuntimeError("punc model not loaded")
        punc = self._punc

        def _run() -> str:
            res = punc.generate(input=text)
            if not res:
                return text
            return res[0].get("text", text) or text

        return await asyncio.to_thread(_run), True

    async def close(self) -> None:
        self._vad = None
        self._punc = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class VadPuncModelManager:
    def __init__(self, backend: VadPuncBackend) -> None:
        self.backend = backend
        self.mode = backend.mode

    @classmethod
    async def from_env(cls) -> VadPuncModelManager:
        raw = os.environ.get("LARYNX_VAD_PUNC_MODE", "mock").lower()
        try:
            mode = VadPuncMode(raw)
        except ValueError as e:
            raise RuntimeError(
                f"LARYNX_VAD_PUNC_MODE must be 'mock' or 'real', got {raw!r}"
            ) from e

        if mode is VadPuncMode.MOCK:
            log.info("vad_punc.mode", mode="mock")
            return cls(MockVadPuncBackend())

        vad_model = os.environ.get("LARYNX_VAD_MODEL", "fsmn-vad")
        punc_model = os.environ.get("LARYNX_PUNC_MODEL", "ct-punc")
        backend = VadPuncBackendReal(vad_model=vad_model, punc_model=punc_model)
        await backend.load()
        log.info("vad_punc.mode", mode="real")
        return cls(backend)

    async def close(self) -> None:
        await self.backend.close()
