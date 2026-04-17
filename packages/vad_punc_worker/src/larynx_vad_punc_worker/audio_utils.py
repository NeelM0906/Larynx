"""Audio helpers for the VAD+punctuation worker.

fsmn-vad expects a float32 numpy array at 16 kHz mono. The gateway
always resamples upstream, so this module just converts int16 PCM bytes
-> float32.
"""

from __future__ import annotations

import numpy as np
from larynx_shared.audio import int16_to_float32
from numpy.typing import NDArray

VAD_SAMPLE_RATE = 16000


def pcm_to_float32(pcm_s16le: bytes, sample_rate: int) -> NDArray[np.float32]:
    if sample_rate != VAD_SAMPLE_RATE:
        raise ValueError(
            f"fsmn-vad expects {VAD_SAMPLE_RATE} Hz, got {sample_rate} Hz; "
            "resample at the gateway before sending"
        )
    if len(pcm_s16le) % 2 != 0:
        raise ValueError("pcm_s16le byte length must be even (int16 stride)")
    samples = np.frombuffer(pcm_s16le, dtype=np.int16)
    return int16_to_float32(samples)
