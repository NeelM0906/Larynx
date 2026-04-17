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
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import librosa
import numpy as np
import structlog
from numpy.typing import NDArray

log = structlog.get_logger(__name__)


# VoxCPM2 operates at 24 kHz end-to-end (both encoder in and decoder out).
# We expose the actual values via get_model_info() at load time; the mock
# backend mirrors them so callers don't branch on mode.
DEFAULT_ENCODER_SR = 24000
DEFAULT_OUTPUT_SR = 24000

# Each latent row is float32 × feat_dim; VoxCPM2 uses feat_dim=64 (confirmed
# by reading VoxCPM2Engine.feat_dim at load time). The mock backend emits
# the same shape so cache layers can't tell the difference.
MOCK_FEAT_DIM = 64


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
    ) -> NDArray[np.float32]: ...

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
        # We use patch_size=2, so round up.
        if num_frames % 2:
            latents = np.concatenate([latents, latents[-1:]], axis=0)
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
    ) -> NDArray[np.float32]:
        if not text:
            raise ValueError("text must not be empty")

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
        freq_hz = base_freq + voice_shift

        t = np.arange(num_samples, dtype=np.float32) / self._output_sr
        signal = 0.25 * np.sin(2.0 * math.pi * freq_hz * t, dtype=np.float32)
        fade_samples = min(int(self._output_sr * 0.01), num_samples // 4)
        if fade_samples > 0:
            ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            signal[:fade_samples] *= ramp
            signal[-fade_samples:] *= ramp[::-1]
        return signal


# ---------------------------------------------------------------------------
# Real backend — nano-vllm-voxcpm adapter
# ---------------------------------------------------------------------------


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
    ) -> None:
        self._model_ref = model
        self._gpu = gpu
        self._inference_timesteps = inference_timesteps
        self._max_num_batched_tokens = max_num_batched_tokens
        self._max_num_seqs = max_num_seqs
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
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

        log.info("voxcpm.loading", gpu=self._gpu, model=self._model_ref)
        self._pool = VoxCPM.from_pretrained(
            model=self._model_ref,
            inference_timesteps=self._inference_timesteps,
            max_num_batched_tokens=self._max_num_batched_tokens,
            max_num_seqs=self._max_num_seqs,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_memory_utilization,
            enforce_eager=False,
            devices=[self._gpu],
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
    ) -> NDArray[np.float32]:
        if self._pool is None:
            raise RuntimeError("backend not loaded")
        if not text:
            raise ValueError("text must not be empty")

        chunks: list[NDArray[np.float32]] = []
        async for chunk in self._pool.generate(
            target_text=text,
            prompt_latents=prompt_audio_latents,
            prompt_text=prompt_text,
            prompt_id=None,
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
            ref_audio_latents=ref_audio_latents,
        ):
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            chunks.append(arr)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0)

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
        backend = VoxCPMBackendReal(
            model=model_ref,
            gpu=gpu,
            inference_timesteps=int(os.environ.get("LARYNX_VOXCPM_INFERENCE_TIMESTEPS", "10")),
            max_num_batched_tokens=int(os.environ.get("LARYNX_VOXCPM_MAX_BATCHED_TOKENS", "8192")),
            max_num_seqs=int(os.environ.get("LARYNX_VOXCPM_MAX_NUM_SEQS", "16")),
            max_model_len=int(os.environ.get("LARYNX_VOXCPM_MAX_MODEL_LEN", "4096")),
            gpu_memory_utilization=float(os.environ.get("LARYNX_VOXCPM_GPU_MEM_UTIL", "0.80")),
        )
        await backend.load()
        log.info("voxcpm.mode", mode="voxcpm", gpu=gpu)
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
