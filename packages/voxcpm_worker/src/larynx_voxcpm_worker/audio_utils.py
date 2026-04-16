"""Audio helpers specific to the VoxCPM2 worker.

M1 only needs float32 -> int16 packing; crossfade + resample helpers land in
M4 for streaming TTS.
"""

from __future__ import annotations

import numpy as np
from larynx_shared.audio import float32_to_int16
from numpy.typing import NDArray


def pcm_from_float(samples: NDArray[np.float32]) -> bytes:
    """float32 mono samples -> int16 little-endian PCM bytes."""
    return float32_to_int16(np.asarray(samples, dtype=np.float32)).tobytes()
