"""WS /v1/tts/stream — mock-backend integration tests.

Exercises the full WS path: auth, config validation, chunk streaming,
crossfade, done frame, cancellation-on-disconnect. Backend is mock so
these run on CPU only.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

TEST_TOKEN = "test-token-please-ignore"


@pytest.fixture
def sync_client(data_dir, _session_env):
    """Starlette TestClient wrapping the gateway app with the lifespan active.

    Separate from the ``client`` fixture because ``TestClient`` is synchronous
    and manages its own lifespan. We clear the settings cache so env var
    overrides take effect per test.
    """
    from larynx_gateway.config import get_settings
    from larynx_gateway.main import create_app

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        yield c


def _connect(
    client: TestClient, token: str = TEST_TOKEN, path: str = "/v1/tts/stream"
) -> WebSocketTestSession:
    return client.websocket_connect(f"{path}?token={token}")


def test_ws_rejects_missing_token(sync_client: TestClient) -> None:
    # Server accepts the handshake, then closes with policy-violation after
    # checking the (missing) token. Starlette's TestClient surfaces the
    # close in the message dict rather than raising — we assert on the
    # close code we set in ws_auth.py.
    with sync_client.websocket_connect("/v1/tts/stream") as ws:
        msg = ws.receive()
    assert msg.get("type") == "websocket.close"
    assert msg.get("code") == 1008


def test_ws_streams_chunks_then_done(sync_client: TestClient) -> None:
    with _connect(sync_client) as ws:
        ws.send_json(
            {
                "type": "synthesize",
                "text": "streaming websocket test",
                "sample_rate": 24000,
            }
        )
        chunks: list[bytes] = []
        done: dict | None = None
        while True:
            msg = ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                chunks.append(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                payload = json.loads(msg["text"])
                if payload.get("type") == "done":
                    done = payload
                    break
                if payload.get("type") == "error":
                    pytest.fail(f"received error frame: {payload}")
            else:
                break

    assert chunks, "expected at least one PCM chunk"
    assert done is not None
    assert done["chunk_count"] == len(chunks)
    assert done["sample_rate"] == 24000
    assert done["total_duration_ms"] > 0
    # PCM16 → even byte count, no odd-sized frames.
    assert all(len(c) % 2 == 0 for c in chunks)
    # Gateway applies 10ms crossfade across each boundary, so the emitted
    # byte count is the model's audio duration minus (chunks - 1) × 10ms.
    total_bytes = sum(len(c) for c in chunks)
    emitted_ms = int(1000 * (total_bytes / 2) / 24000)
    expected_ms = done["total_duration_ms"] - (len(chunks) - 1) * 10
    assert abs(emitted_ms - expected_ms) <= 2


def test_ws_invalid_config_closes_with_error(sync_client: TestClient) -> None:
    with _connect(sync_client) as ws:
        ws.send_json({"type": "synthesize"})  # missing `text`
        frame = ws.receive()
        assert "text" in frame
        payload = json.loads(frame["text"])
        assert payload["type"] == "error"
        assert payload["code"] == "invalid_config"


def test_ws_disconnect_cancels_generation(sync_client: TestClient) -> None:
    """After early client close, the gateway must not hang the worker.

    We send the config, read one chunk, then break out of the with block.
    The session ends cleanly (no exceptions) — which proves the cancel
    path inside ``synthesize_text_stream.__aexit__`` fires.
    """
    with _connect(sync_client) as ws:
        ws.send_json(
            {
                "type": "synthesize",
                # A long text so many chunks are queued up.
                "text": "x " * 500,
                "sample_rate": 24000,
            }
        )
        msg = ws.receive()
        assert "bytes" in msg and msg["bytes"] is not None
    # Context manager cleanly closed — no assertions beyond "didn't hang".
