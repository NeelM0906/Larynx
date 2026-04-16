"""Round-trip tests for the WAV packer/parser."""

from __future__ import annotations

import numpy as np
from larynx_shared.audio import float32_to_int16, pack_wav, parse_wav_header


def test_pack_parse_roundtrip() -> None:
    sr = 16000
    samples = np.sin(np.linspace(0, 2 * np.pi, sr // 2, dtype=np.float32)) * 0.5
    pcm = float32_to_int16(samples).tobytes()
    wav = pack_wav(pcm, sample_rate=sr)

    assert wav.startswith(b"RIFF")
    header = parse_wav_header(wav)
    assert header.num_channels == 1
    assert header.sample_rate == sr
    assert header.bits_per_sample == 16
    assert header.num_frames == sr // 2


def test_parse_wav_rejects_non_riff() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_wav_header(b"NOT-A-WAV")


def test_float_to_int16_clips_out_of_range() -> None:
    samples = np.array([2.0, -2.0, 0.0, 0.5], dtype=np.float32)
    out = float32_to_int16(samples)
    assert out[0] == 32767
    assert out[1] == -32767
    assert out[2] == 0
    assert out[3] == int(0.5 * 32767)
