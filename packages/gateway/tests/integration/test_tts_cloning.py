"""/v1/tts voice-cloning path tests (mock TTS).

Covers:
  - voice_id lookup with cached latents -> audio varies vs no-voice synthesis
  - voice_id=missing -> 404
  - voice_id + reference_audio multipart -> 400 (mutually exclusive)
  - reference_audio multipart upload (one-off clone)
  - prompt_audio without prompt_text -> 400 (ultimate cloning requires text)
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient


def _wav_bytes(seed: int = 1, duration_s: float = 1.0, sr: int = 24000) -> bytes:
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(int(sr * duration_s)).astype(np.float32) * 0.1
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_tts_with_voice_id(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    # Upload a voice first.
    up = await client.post(
        "/v1/voices",
        headers=auth_headers,
        files={"audio": ("ref.wav", _wav_bytes(seed=42), "audio/wav")},
        data={"name": "clone-subject"},
    )
    assert up.status_code == 201
    vid = up.json()["id"]

    # Baseline — synthesize with no voice
    r_base = await client.post(
        "/v1/tts",
        headers=auth_headers,
        json={"text": "the same text", "sample_rate": 24000},
    )
    assert r_base.status_code == 200

    # Cloned — same text, with voice_id
    r_clone = await client.post(
        "/v1/tts",
        headers=auth_headers,
        json={"text": "the same text", "sample_rate": 24000, "voice_id": vid},
    )
    assert r_clone.status_code == 200
    assert r_clone.headers["x-voice-id"] == vid
    # The mock backend shifts pitch from the latents' first float, so output
    # bytes MUST differ between baseline and cloned output.
    assert r_base.content != r_clone.content


@pytest.mark.asyncio
async def test_tts_unknown_voice_id_returns_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/tts",
        headers=auth_headers,
        json={"text": "anyone there?", "voice_id": "no-such-voice"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_tts_multipart_reference_audio(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/tts",
        headers=auth_headers,
        files={"reference_audio": ("ref.wav", _wav_bytes(seed=9), "audio/wav")},
        data={"text": "ad-hoc clone", "sample_rate": "24000"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")


@pytest.mark.asyncio
async def test_tts_voice_id_and_reference_audio_conflict(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    up = await client.post(
        "/v1/voices",
        headers=auth_headers,
        files={"audio": ("ref.wav", _wav_bytes(seed=21), "audio/wav")},
        data={"name": "conflict-subject"},
    )
    vid = up.json()["id"]

    r = await client.post(
        "/v1/tts",
        headers=auth_headers,
        files={"reference_audio": ("r.wav", _wav_bytes(seed=22), "audio/wav")},
        data={"text": "nope", "voice_id": vid},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_tts_prompt_audio_without_text_rejected(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/tts",
        headers=auth_headers,
        files={"prompt_audio": ("p.wav", _wav_bytes(seed=3), "audio/wav")},
        data={"text": "some target"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_tts_with_prompt_audio_and_text_ok(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/tts",
        headers=auth_headers,
        files={"prompt_audio": ("p.wav", _wav_bytes(seed=15), "audio/wav")},
        data={
            "text": "target text to synthesize",
            "prompt_text": "prompt transcript",
        },
    )
    assert r.status_code == 200
    assert r.content.startswith(b"RIFF")
