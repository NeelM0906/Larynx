"""PCM sample-format conversions and stream-boundary helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def float32_to_int16(samples: NDArray[np.float32]) -> NDArray[np.int16]:
    # Clip before scaling so out-of-range model output doesn't wrap around.
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def int16_to_float32(samples: NDArray[np.int16]) -> NDArray[np.float32]:
    return samples.astype(np.float32) / 32768.0


def crossfade_chunks(
    tail: bytes,
    head: bytes,
    sample_rate: int,
    overlap_ms: float = 10.0,
) -> tuple[bytes, bytes]:
    """Apply a linear equal-gain crossfade across a chunk boundary.

    Returns ``(tail_out, head_out)`` — both PCM16 little-endian — where the
    last ``overlap_ms`` of ``tail`` and the first ``overlap_ms`` of ``head``
    have been blended with complementary linear ramps. Streaming callers flush
    ``tail_out`` and keep ``head_out`` as the new tail for the next boundary.

    Why linear / equal-gain: VoxCPM chunk boundaries don't guarantee phase
    alignment across runs, so a short linear fade is safer than overlap-add.
    10 ms is short enough to be inaudible but long enough to mask the
    discontinuity at 24 kHz / 48 kHz decoder output.

    If either chunk is shorter than the overlap window the function silently
    shrinks the window to the available samples.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    overlap = int(round(sample_rate * overlap_ms / 1000.0))
    if overlap <= 0 or not tail or not head:
        return tail, head

    tail_arr = np.frombuffer(tail, dtype=np.int16)
    head_arr = np.frombuffer(head, dtype=np.int16)
    n = min(overlap, len(tail_arr), len(head_arr))
    if n <= 0:
        return tail, head

    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    fade_in = 1.0 - fade_out

    tail_tail = tail_arr[-n:].astype(np.float32)
    head_head = head_arr[:n].astype(np.float32)
    mixed = tail_tail * fade_out + head_head * fade_in
    mixed_i16 = np.clip(mixed, -32768.0, 32767.0).astype(np.int16)

    # Write the mixed samples back into the tail (so the emitted tail ends
    # smoothly) and strip them from the head (so subsequent emit doesn't
    # double up).
    new_tail = np.concatenate([tail_arr[:-n], mixed_i16]).tobytes()
    new_head = head_arr[n:].tobytes()
    return new_tail, new_head
