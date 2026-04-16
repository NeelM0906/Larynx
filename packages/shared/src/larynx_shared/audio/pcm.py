"""PCM sample-format conversions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def float32_to_int16(samples: NDArray[np.float32]) -> NDArray[np.int16]:
    # Clip before scaling so out-of-range model output doesn't wrap around.
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def int16_to_float32(samples: NDArray[np.int16]) -> NDArray[np.float32]:
    return samples.astype(np.float32) / 32768.0
