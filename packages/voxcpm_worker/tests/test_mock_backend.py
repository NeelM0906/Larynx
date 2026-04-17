"""Mock backend determinism + voice-cloning contract."""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest
import soundfile as sf
from larynx_voxcpm_worker.model_manager import MockVoxCPMBackend, ModelMode


def _make_wav(seed: int, duration_s: float = 1.0, sr: int = 24000) -> bytes:
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(int(sr * duration_s)).astype(np.float32) * 0.1
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_mock_produces_float32_in_range() -> None:
    be = MockVoxCPMBackend()
    audio = await be.synthesize(text="hello world")
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.max(np.abs(audio)) <= 1.0
    assert len(audio) > 0


@pytest.mark.asyncio
async def test_mock_is_deterministic() -> None:
    be = MockVoxCPMBackend()
    a = await be.synthesize(text="the same text")
    b = await be.synthesize(text="the same text")
    assert np.array_equal(a, b)


@pytest.mark.asyncio
async def test_mock_differs_by_text() -> None:
    be = MockVoxCPMBackend()
    a = await be.synthesize(text="alpha")
    b = await be.synthesize(text="bravo")
    assert not np.array_equal(a, b)


@pytest.mark.asyncio
async def test_mock_length_scales_with_text() -> None:
    be = MockVoxCPMBackend()
    short = await be.synthesize(text="hi")
    long = await be.synthesize(text="a" * 200)
    assert len(long) > len(short)


@pytest.mark.asyncio
async def test_mock_rejects_empty() -> None:
    be = MockVoxCPMBackend()
    with pytest.raises(ValueError):
        await be.synthesize(text="")


@pytest.mark.asyncio
async def test_mock_mode_is_mock() -> None:
    be = MockVoxCPMBackend()
    assert be.mode is ModelMode.MOCK
    info = await be.get_info()
    assert info.encoder_sample_rate == 24000
    assert info.output_sample_rate == 24000
    assert info.feat_dim == 64


@pytest.mark.asyncio
async def test_encode_reference_deterministic_and_content_dependent() -> None:
    be = MockVoxCPMBackend()
    wav_a1 = _make_wav(seed=1)
    wav_a2 = _make_wav(seed=1)
    wav_b = _make_wav(seed=2)

    l_a1 = await be.encode_reference(wav_a1)
    l_a2 = await be.encode_reference(wav_a2)
    l_b = await be.encode_reference(wav_b)

    assert l_a1 == l_a2  # same audio content -> same latents
    assert l_a1 != l_b  # different content -> different latents
    # Shape: bytes are float32 × feat_dim frames.
    assert len(l_a1) % (4 * 64) == 0
    num_frames = len(l_a1) // (4 * 64)
    assert num_frames >= 32


@pytest.mark.asyncio
async def test_encode_reference_rejects_empty_audio() -> None:
    be = MockVoxCPMBackend()
    with pytest.raises(ValueError):
        await be.encode_reference(b"")


@pytest.mark.asyncio
async def test_ref_audio_latents_change_output() -> None:
    be = MockVoxCPMBackend()
    wav = _make_wav(seed=42)
    latents = await be.encode_reference(wav)

    a = await be.synthesize(text="same prompt")
    b = await be.synthesize(text="same prompt", ref_audio_latents=latents)
    assert not np.array_equal(a, b)

    # Known-shape invariant: the mock reads the first float of the latents
    # to derive the pitch shift, so prefixing bytes should change output.
    shifted = struct.pack("<f", 0.5) + latents[4:]
    c = await be.synthesize(text="same prompt", ref_audio_latents=shifted)
    assert not np.array_equal(b, c)
