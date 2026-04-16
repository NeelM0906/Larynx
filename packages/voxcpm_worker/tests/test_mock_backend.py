"""Mock backend determinism + shape checks."""

from __future__ import annotations

import numpy as np
import pytest
from larynx_voxcpm_worker.model_manager import MockVoxCPMBackend


def test_mock_produces_float32_in_range() -> None:
    be = MockVoxCPMBackend()
    audio = be.synthesize("hello world", sample_rate=16000)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.max(np.abs(audio)) <= 1.0
    assert len(audio) > 0


def test_mock_is_deterministic() -> None:
    be = MockVoxCPMBackend()
    a = be.synthesize("the same text", sample_rate=16000)
    b = be.synthesize("the same text", sample_rate=16000)
    assert np.array_equal(a, b)


def test_mock_differs_by_text() -> None:
    be = MockVoxCPMBackend()
    a = be.synthesize("alpha", sample_rate=16000)
    b = be.synthesize("bravo", sample_rate=16000)
    assert not np.array_equal(a, b)


def test_mock_length_scales_with_text() -> None:
    be = MockVoxCPMBackend()
    short = be.synthesize("hi", sample_rate=16000)
    long = be.synthesize("a" * 200, sample_rate=16000)
    assert len(long) > len(short)


def test_mock_rejects_empty() -> None:
    be = MockVoxCPMBackend()
    with pytest.raises(ValueError):
        be.synthesize("", sample_rate=16000)
