"""Integration tests for /health + POST /v1/tts.

The gateway boots with ``LARYNX_TTS_MODE=mock`` (see conftest.py), so every
request exercises the real code path end-to-end: schema validation, auth,
in-process worker IPC, WAV packaging. The only thing swapped out is the
VoxCPM2 model itself — which is the layer that M2 integration tests will
cover on the GPU box.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from larynx_shared.audio import parse_wav_header


@pytest.mark.asyncio
async def test_health_no_auth_required(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ready_reports_worker_started(client: AsyncClient) -> None:
    r = await client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["worker"] == "ready"


@pytest.mark.asyncio
async def test_tts_without_auth_returns_401(client: AsyncClient) -> None:
    r = await client.post("/v1/tts", json={"text": "hello"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_tts_with_bad_token_returns_401(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/tts",
        json={"text": "hello"},
        headers={"Authorization": "Bearer totally-wrong"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_tts_returns_valid_wav(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    r = await client.post(
        "/v1/tts",
        headers=auth_headers,
        json={"text": "Hello from Larynx.", "sample_rate": 24000},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    assert int(r.headers["x-generation-time-ms"]) >= 0
    assert int(r.headers["x-sample-rate"]) == 24000

    audio = r.content
    assert audio.startswith(b"RIFF")
    header = parse_wav_header(audio)
    assert header.num_channels == 1
    assert header.bits_per_sample == 16
    assert header.sample_rate == 24000
    # Mock TTS is ~60ms/char, 18-char prompt -> ≥400ms floor still applies.
    assert header.duration_ms >= 400


@pytest.mark.asyncio
async def test_tts_empty_text_returns_422(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """pydantic's Field(min_length=1) fires before the worker sees the request."""
    r = await client.post("/v1/tts", headers=auth_headers, json={"text": ""})
    assert r.status_code == 422
    body = r.json()
    assert "detail" in body


@pytest.mark.asyncio
async def test_tts_missing_text_returns_422(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post("/v1/tts", headers=auth_headers, json={})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_tts_bad_sample_rate_returns_422(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/tts",
        headers=auth_headers,
        json={"text": "hi", "sample_rate": 999999},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_tts_pcm16_output_format(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    r = await client.post(
        "/v1/tts",
        headers=auth_headers,
        json={"text": "raw pcm", "output_format": "pcm16", "sample_rate": 16000},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/L16"
    # no RIFF header when output_format=pcm16
    assert not r.content.startswith(b"RIFF")
    # 16-bit mono, so length is always even
    assert len(r.content) % 2 == 0
