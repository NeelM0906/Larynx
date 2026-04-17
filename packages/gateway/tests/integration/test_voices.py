"""Integration tests for /v1/voices — real Postgres, real Redis, mock TTS.

These exercise the full upload -> cache -> list -> get -> delete pipeline
via HTTP. Real-model voice cloning numbers are in test_real_model.py.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient


def _wav_bytes(seed: int = 1, duration_s: float = 1.2, sr: int = 24000) -> bytes:
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(int(sr * duration_s)).astype(np.float32) * 0.1
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_upload_voice_returns_metadata(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    audio = _wav_bytes(seed=1)
    r = await client.post(
        "/v1/voices",
        headers=auth_headers,
        files={"audio": ("reference.wav", audio, "audio/wav")},
        data={"name": "alice", "description": "warm, clear"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "alice"
    assert body["description"] == "warm, clear"
    assert body["source"] == "uploaded"
    assert body["sample_rate_hz"] == 24000
    assert body["duration_ms"] > 0
    assert "id" in body


@pytest.mark.asyncio
async def test_upload_requires_auth(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/voices",
        files={"audio": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"name": "bob"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_duplicate_name_returns_409(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    audio = _wav_bytes(seed=2)
    r1 = await client.post(
        "/v1/voices",
        headers=auth_headers,
        files={"audio": ("a.wav", audio, "audio/wav")},
        data={"name": "dup"},
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/v1/voices",
        headers=auth_headers,
        files={"audio": ("b.wav", audio, "audio/wav")},
        data={"name": "dup"},
    )
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_list_and_get(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    for i in range(3):
        await client.post(
            "/v1/voices",
            headers=auth_headers,
            files={"audio": ("a.wav", _wav_bytes(seed=i + 10), "audio/wav")},
            data={"name": f"voice-{i}"},
        )

    r = await client.get("/v1/voices", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert len(body["voices"]) == 3

    vid = body["voices"][0]["id"]
    r_get = await client.get(f"/v1/voices/{vid}", headers=auth_headers)
    assert r_get.status_code == 200
    assert r_get.json()["id"] == vid


@pytest.mark.asyncio
async def test_get_audio_returns_wav(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    audio = _wav_bytes(seed=5)
    r = await client.post(
        "/v1/voices",
        headers=auth_headers,
        files={"audio": ("reference.wav", audio, "audio/wav")},
        data={"name": "charlie"},
    )
    vid = r.json()["id"]

    r_audio = await client.get(f"/v1/voices/{vid}/audio", headers=auth_headers)
    assert r_audio.status_code == 200
    assert r_audio.headers["content-type"] == "audio/wav"
    assert r_audio.content.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_get_unknown_returns_404(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    r = await client.get("/v1/voices/nonexistent", headers=auth_headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_voice(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    audio = _wav_bytes(seed=7)
    r = await client.post(
        "/v1/voices",
        headers=auth_headers,
        files={"audio": ("a.wav", audio, "audio/wav")},
        data={"name": "delme"},
    )
    vid = r.json()["id"]

    r_del = await client.delete(f"/v1/voices/{vid}", headers=auth_headers)
    assert r_del.status_code == 204

    r_get = await client.get(f"/v1/voices/{vid}", headers=auth_headers)
    assert r_get.status_code == 404

    r_del2 = await client.delete(f"/v1/voices/{vid}", headers=auth_headers)
    assert r_del2.status_code == 404


@pytest.mark.asyncio
async def test_design_preview_and_save(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    r = await client.post(
        "/v1/voices/design",
        headers=auth_headers,
        json={
            "name": "designed-voice",
            "description": "designed from prompt",
            "design_prompt": "warm, middle-aged female",
            "preview_text": "Hello, this is a test.",
        },
    )
    assert r.status_code == 200, r.text
    preview = r.json()
    assert preview["name"] == "designed-voice"
    assert preview["duration_ms"] > 0

    # Preview audio is downloadable
    r_audio = await client.get(
        f"/v1/voices/design/{preview['preview_id']}/audio", headers=auth_headers
    )
    assert r_audio.status_code == 200
    assert r_audio.content.startswith(b"RIFF")

    # Save -> creates a permanent voice with source=designed
    r_save = await client.post(
        f"/v1/voices/design/{preview['preview_id']}/save",
        headers=auth_headers,
        json={},
    )
    assert r_save.status_code == 201
    saved = r_save.json()
    assert saved["source"] == "designed"
    assert saved["name"] == "designed-voice"


@pytest.mark.asyncio
async def test_unreadable_audio_returns_400(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/voices",
        headers=auth_headers,
        files={"audio": ("bad.wav", b"not a real wav file", "audio/wav")},
        data={"name": "broken"},
    )
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unreadable_audio"
