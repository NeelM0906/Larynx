"""Integration tests for POST /v1/stt and POST /v1/audio/transcriptions.

Uses the mock Fun-ASR backend (see ``MockFunASRBackend``) so every path
exercises schema validation, multipart parsing, auth, the language
router, the VAD silence-trim path, and the punctuation routing — the
only thing faked is the transcription itself.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from httpx import AsyncClient
from larynx_shared.audio import float32_to_int16


def _wav_bytes(seconds: float = 0.5, sr: int = 16000, silent: bool = False) -> bytes:
    n = int(sr * seconds)
    if silent:
        samples = np.zeros(n, dtype=np.float32)
    else:
        t = np.arange(n, dtype=np.float32) / sr
        samples = 0.3 * np.sin(2 * np.pi * 220 * t, dtype=np.float32)
    pcm = float32_to_int16(samples).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_stt_without_auth_returns_401(client: AsyncClient) -> None:
    r = await client.post("/v1/stt", files={"file": ("t.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_stt_happy_path_english(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"language": "en"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["language"] == "en"
    assert body["model_used"] == "nano"
    assert isinstance(body["text"], str) and len(body["text"]) > 0
    assert body["duration_ms"] >= 400
    assert body["processing_ms"] >= 0
    assert body["punctuated"] is True  # mock capitalises + adds period


@pytest.mark.asyncio
async def test_stt_portuguese_routes_to_mlt(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"language": "pt"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_used"] == "mlt"
    assert body["language"] == "pt"
    # ct-punc is zh/en only — MLT languages should report punctuated=False.
    assert body["punctuated"] is False


@pytest.mark.asyncio
async def test_stt_auto_detect_when_no_language(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["model_used"] == "nano"


@pytest.mark.asyncio
async def test_stt_unsupported_language_returns_400(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"language": "es"},  # Spanish — not in Fun-ASR coverage
    )
    assert r.status_code == 400
    assert "spanish" in r.text.lower() or "not covered" in r.text.lower() or "es" in r.text


@pytest.mark.asyncio
async def test_stt_hotwords_passthrough(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"language": "en", "hotwords": "Larynx, VoxCPM , Fun-ASR"},
    )
    assert r.status_code == 200
    body = r.json()
    # Mock backend echoes hotwords into the transcript; asserts split +
    # whitespace trim worked.
    assert "Larynx" in body["text"]
    assert "VoxCPM" in body["text"]
    assert "Fun-ASR" in body["text"]


@pytest.mark.asyncio
async def test_stt_punctuate_false_returns_raw(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"language": "en", "punctuate": "false"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["punctuated"] is False
    assert not body["text"].endswith(".")


@pytest.mark.asyncio
async def test_stt_empty_upload_returns_400(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_stt_silent_audio_returns_empty_text(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("silence.wav", _wav_bytes(silent=True), "audio/wav")},
        data={"language": "en"},
    )
    assert r.status_code == 200
    body = r.json()
    # Silence is trimmed by VAD; mock returns empty text.
    assert body["text"] == ""
    assert body["punctuated"] is False


@pytest.mark.asyncio
async def test_stt_trim_silence_false_keeps_audio(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("silence.wav", _wav_bytes(silent=True), "audio/wav")},
        data={"language": "en", "trim_silence": "false"},
    )
    assert r.status_code == 200
    # With trim disabled, silence still flows through to Fun-ASR (mock
    # returns its canned string).
    assert len(r.json()["text"]) > 0


@pytest.mark.asyncio
async def test_openai_transcriptions_json(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"model": "whisper-1", "language": "en"},
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"text", "language"}
    assert body["language"] == "en"


@pytest.mark.asyncio
async def test_openai_transcriptions_text_format(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"model": "whisper-1", "language": "en", "response_format": "text"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert len(r.text) > 0


@pytest.mark.asyncio
async def test_openai_transcriptions_prompt_maps_to_hotwords(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={
            "model": "whisper-1",
            "language": "en",
            "prompt": "Larynx, VoxCPM",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "Larynx" in body["text"]
    assert "VoxCPM" in body["text"]


@pytest.mark.asyncio
async def test_openai_transcriptions_invalid_format_returns_400(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    r = await client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"model": "whisper-1", "response_format": "verbose_json"},
    )
    # verbose_json isn't supported in M3 (Fun-ASR has no timestamps yet)
    # — FastAPI's form validation returns 422 on the Literal constraint.
    assert r.status_code in {400, 422}
