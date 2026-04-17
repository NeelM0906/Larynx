"""Tests for PCM helpers, especially the streaming crossfade."""

from __future__ import annotations

import numpy as np
from larynx_shared.audio import crossfade_chunks


def _pcm(samples: np.ndarray) -> bytes:
    return samples.astype(np.int16).tobytes()


def test_crossfade_is_monotonic_linear_at_boundary() -> None:
    # Flat +10000 tail meets flat -10000 head. With linear equal-gain
    # crossfade, the mixed window should interpolate smoothly from +10000
    # toward -10000 across N samples (no step).
    sr = 24000
    overlap_ms = 10.0
    overlap = int(sr * overlap_ms / 1000)
    tail = _pcm(np.full(overlap * 2, 10000, dtype=np.int16))
    head = _pcm(np.full(overlap * 2, -10000, dtype=np.int16))

    new_tail, new_head = crossfade_chunks(tail, head, sr, overlap_ms=overlap_ms)

    tail_arr = np.frombuffer(new_tail, dtype=np.int16)
    # Unchanged prefix:
    assert np.all(tail_arr[:overlap] == 10000)
    # Crossfade region should be strictly decreasing from ~+10000 → ~-10000.
    mix = tail_arr[overlap:]
    assert mix[0] > mix[-1]
    diffs = np.diff(mix.astype(np.int32))
    assert (diffs < 0).all(), "linear ramp must be monotonically decreasing"

    # The head now starts from the post-overlap region only.
    head_arr = np.frombuffer(new_head, dtype=np.int16)
    assert len(head_arr) == overlap
    assert np.all(head_arr == -10000)


def test_crossfade_noop_for_empty_inputs() -> None:
    tail, head = crossfade_chunks(b"", b"\x00\x00", 24000)
    assert tail == b""
    assert head == b"\x00\x00"


def test_crossfade_shrinks_window_when_chunks_short() -> None:
    sr = 16000
    # Both chunks are 3 samples = well below a 10ms window (160 samples).
    tail = _pcm(np.array([8000, 8000, 8000], dtype=np.int16))
    head = _pcm(np.array([-8000, -8000, -8000], dtype=np.int16))
    new_tail, new_head = crossfade_chunks(tail, head, sr, overlap_ms=10.0)
    assert len(new_tail) == len(tail)
    assert len(new_head) == 0  # whole head consumed into the crossfade


def test_crossfade_preserves_total_frames_across_boundary() -> None:
    sr = 48000
    tail = _pcm(np.random.default_rng(0).integers(-1000, 1000, 800, dtype=np.int16))
    head = _pcm(np.random.default_rng(1).integers(-1000, 1000, 800, dtype=np.int16))

    new_tail, new_head = crossfade_chunks(tail, head, sr, overlap_ms=10.0)
    # The number of samples actually emitted (tail + head) should match the
    # original total minus one overlap window (the blended samples replace
    # both the old tail's end and the new head's start).
    overlap = int(sr * 10.0 / 1000)
    assert len(new_tail) + len(new_head) == len(tail) + len(head) - 2 * overlap
