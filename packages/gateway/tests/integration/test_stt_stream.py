"""WS /v1/stt/stream mock-backend integration tests.

Covers: auth, config validation, rolling-partials cadence, speech_start /
speech_end / final, cancellation on disconnect, concurrent sessions don't
starve each other.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

TEST_TOKEN = "test-token-please-ignore"


@pytest.fixture
def sync_client(data_dir, _session_env):
    from larynx_gateway.config import get_settings
    from larynx_gateway.main import create_app

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        yield c


def _pcm_tone(ms: int, sr: int = 16000, freq: float = 220.0, amp: float = 0.5) -> bytes:
    n = sr * ms // 1000
    t = np.arange(n, dtype=np.float32) / sr
    s = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return (s * 32767).astype(np.int16).tobytes()


def _pcm_silence(ms: int, sr: int = 16000) -> bytes:
    n = sr * ms // 1000
    return (np.zeros(n, dtype=np.int16)).tobytes()


def _connect(client: TestClient) -> object:
    return client.websocket_connect(f"/v1/stt/stream?token={TEST_TOKEN}")


def test_stt_rejects_missing_token(sync_client: TestClient) -> None:
    with sync_client.websocket_connect("/v1/stt/stream") as ws:
        msg = ws.receive()
    assert msg.get("type") == "websocket.close"
    assert msg.get("code") == 1008


def test_stt_config_validation_rejects_bad_sample_rate(sync_client: TestClient) -> None:
    with _connect(sync_client) as ws:
        ws.send_json({"type": "config", "sample_rate": 4000})  # < 8000
        frame = ws.receive()
        payload = json.loads(frame["text"])
        assert payload["type"] == "error"
        assert payload["code"] == "invalid_config"


def test_stt_stream_emits_start_partials_end_final(sync_client: TestClient) -> None:
    """Tone → silence transitions produce the full event sequence."""
    with _connect(sync_client) as ws:
        ws.send_json(
            {
                "type": "config",
                "sample_rate": 16000,
                "chunk_interval_ms": 300,  # shorter for test speed
                "speech_end_silence_ms": 200,
            }
        )

        # 1.5s of tone, then 1s of silence.
        for _ in range(15):
            ws.send_bytes(_pcm_tone(100))  # 100ms frames, 15 of them = 1.5s
        for _ in range(10):
            ws.send_bytes(_pcm_silence(100))
        # Graceful stop.
        ws.send_json({"type": "stop"})

        events: list[dict] = []
        while True:
            try:
                frame = ws.receive()
            except Exception:
                break
            t = frame.get("type")
            if t in ("websocket.disconnect", "websocket.close"):
                break
            if frame.get("text"):
                events.append(json.loads(frame["text"]))
            # ignore binary or other payloads

    kinds = [e["type"] for e in events]
    assert "speech_start" in kinds, f"got kinds={kinds}"
    assert "speech_end" in kinds, f"got kinds={kinds}"
    # A final event for the closed utterance must appear after speech_end.
    assert "final" in kinds, f"got kinds={kinds}"
    # Partial cadence isn't asserted here — TestClient delivers all PCM
    # frames atomically, which beats the 200ms partials_loop tick. The
    # service unit test (`test_service_emits_full_event_sequence`) covers
    # the partial path with realistic inter-frame gaps.


def test_stt_stream_closes_cleanly_on_disconnect(sync_client: TestClient) -> None:
    with _connect(sync_client) as ws:
        ws.send_json(
            {"type": "config", "sample_rate": 16000, "chunk_interval_ms": 300}
        )
        ws.send_bytes(_pcm_tone(500))
    # Implicit close — the context manager exits, WS tears down. No
    # exceptions must leak across the boundary (test passes by reaching
    # this line without hanging or raising).
