"""Manages the VoxCPM2 backend lifecycle.

Two backends:

* ``VoxCPMBackend`` — wraps ``nano-vllm-voxcpm``. Loaded at process start,
  held in VRAM, queried synchronously per request. Real TTS.

* ``MockVoxCPMBackend`` — generates a deterministic sine wave whose pitch
  depends on the input text (so identical inputs produce identical audio and
  different inputs differ audibly). Lets the gateway, IPC, WAV packaging and
  smoke test all be verified on a box without a GPU.

The mode is selected via the ``LARYNX_TTS_MODE`` env var (see .env.example)
and read by ``VoxCPMModelManager.from_env``.
"""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger(__name__)


class ModelMode(StrEnum):
    MOCK = "mock"
    VOXCPM = "voxcpm"


class VoxCPMBackend(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        sample_rate: int,
        cfg_value: float = 2.0,
    ) -> NDArray[np.float32]: ...

    def close(self) -> None:  # noqa: B027 — intentional no-op default
        """Release resources. Override in subclasses that hold GPU memory."""


class MockVoxCPMBackend(VoxCPMBackend):
    """Deterministic sine-wave generator. ~60ms per character of text, so the
    output length scales with input like a real TTS would, which is enough
    realism for integration tests and smoke checks."""

    MS_PER_CHAR = 60
    MIN_DURATION_MS = 400

    def synthesize(
        self,
        text: str,
        sample_rate: int,
        cfg_value: float = 2.0,
    ) -> NDArray[np.float32]:
        if not text:
            raise ValueError("text must not be empty")

        duration_ms = max(self.MIN_DURATION_MS, self.MS_PER_CHAR * len(text))
        num_samples = int(sample_rate * duration_ms / 1000)

        # Frequency derived from text hash so identical text -> identical audio.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        freq_hz = 120.0 + (digest[0] / 255.0) * 400.0  # 120 - 520 Hz

        t = np.arange(num_samples, dtype=np.float32) / sample_rate
        signal = 0.25 * np.sin(2.0 * math.pi * freq_hz * t, dtype=np.float32)
        # 10ms fade in/out to avoid end-point clicks when played back.
        fade_samples = min(int(sample_rate * 0.01), num_samples // 4)
        if fade_samples > 0:
            ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            signal[:fade_samples] *= ramp
            signal[-fade_samples:] *= ramp[::-1]
        return signal


class VoxCPMBackendReal(VoxCPMBackend):
    """Thin adapter around nano-vllm-voxcpm.

    The actual nano-vllm-voxcpm API surface is wired up at load time below.
    It lives in the worker's own process, so importing the (heavy) module is
    only paid in ``voxcpm`` mode.
    """

    def __init__(self, gpu: int = 0) -> None:
        self._gpu = gpu
        self._engine = None  # set in _load
        self._load()

    def _load(self) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self._gpu))
        try:
            import nano_vllm_voxcpm  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "LARYNX_TTS_MODE=voxcpm requires nano-vllm-voxcpm to be installed. "
                "Install it on the GPU box, or switch to LARYNX_TTS_MODE=mock for "
                "GPU-less development."
            ) from e

        log.info("voxcpm.loading", gpu=self._gpu)
        # The exact constructor name varies across nano-vllm-voxcpm revisions.
        # This lookup keeps the worker importable while letting us swap as the
        # upstream API solidifies.
        for attr in ("VoxCPMEngine", "Engine", "LLMEngine"):
            cls = getattr(nano_vllm_voxcpm, attr, None)
            if cls is not None:
                self._engine = cls()
                break
        else:  # no break
            raise RuntimeError(
                "nano-vllm-voxcpm is installed but doesn't expose a known engine class"
            )
        log.info("voxcpm.loaded", gpu=self._gpu)

    def synthesize(
        self,
        text: str,
        sample_rate: int,
        cfg_value: float = 2.0,
    ) -> NDArray[np.float32]:
        # M1 placeholder signature. Task #11 replaces this class entirely with
        # an async adapter around AsyncVoxCPM2ServerPool that supports
        # reference encoding and voice-cloning parameters.
        raise NotImplementedError("VoxCPMBackendReal is rewritten in M2 task #11")

    def close(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is not None and hasattr(engine, "close"):
            engine.close()


class VoxCPMModelManager:
    """One instance per worker process. Owns the active backend."""

    def __init__(self, backend: VoxCPMBackend, mode: ModelMode) -> None:
        self.backend = backend
        self.mode = mode

    @classmethod
    def from_env(cls) -> VoxCPMModelManager:
        raw = os.environ.get("LARYNX_TTS_MODE", "mock").lower()
        try:
            mode = ModelMode(raw)
        except ValueError as e:
            raise RuntimeError(f"LARYNX_TTS_MODE must be 'mock' or 'voxcpm', got {raw!r}") from e

        if mode is ModelMode.MOCK:
            log.info("voxcpm.mode", mode="mock")
            return cls(MockVoxCPMBackend(), mode)

        gpu = int(os.environ.get("LARYNX_VOXCPM_GPU", "0"))
        log.info("voxcpm.mode", mode="voxcpm", gpu=gpu)
        return cls(VoxCPMBackendReal(gpu=gpu), mode)

    def close(self) -> None:
        self.backend.close()
