"""Audio helpers for the Fun-ASR worker.

Fun-ASR expects 16 kHz mono float32 tensors (or a file path). The gateway
already resamples arriving audio, so the worker side mostly converts
int16-PCM bytes -> float32 numpy and validates the sample rate.
"""

from __future__ import annotations

import io

import numpy as np
from larynx_shared.audio import int16_to_float32
from numpy.typing import NDArray

FUNASR_SAMPLE_RATE = 16000


def pcm_to_float32(pcm_s16le: bytes, sample_rate: int) -> NDArray[np.float32]:
    """int16 little-endian mono PCM bytes -> float32 numpy in [-1, 1].

    Raises if the sample rate doesn't match Fun-ASR's expected 16 kHz —
    the gateway is responsible for resampling before the call so that the
    worker never silently drops quality.
    """
    if sample_rate != FUNASR_SAMPLE_RATE:
        raise ValueError(
            f"fun-asr expects {FUNASR_SAMPLE_RATE} Hz, got {sample_rate} Hz; "
            "resample at the gateway before sending"
        )
    if len(pcm_s16le) % 2 != 0:
        raise ValueError("pcm_s16le byte length must be even (int16 stride)")
    samples = np.frombuffer(pcm_s16le, dtype=np.int16)
    return int16_to_float32(samples)


def decode_any_to_float32(
    audio_bytes: bytes, target_sr: int = FUNASR_SAMPLE_RATE
) -> NDArray[np.float32]:
    """Decode an arbitrary audio container (wav/mp3/flac/...) to 16 kHz mono.

    Used by the smoke script / real-model tests that pass a file blob
    directly to the worker instead of pre-resampled PCM.
    """
    import librosa  # deferred — pulled in only when the helper is called

    samples, _ = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
    return samples.astype(np.float32)
