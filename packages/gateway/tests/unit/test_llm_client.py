"""Unit tests for the OpenRouter streaming client.

Uses httpx.MockTransport so tests are deterministic + offline. The real
integration path is exercised by the opt-in real-model conversation test.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from larynx_gateway.services.llm_client import (
    ChatMessage,
    LLMClient,
    LLMHTTPError,
    LLMTimeoutError,
)


def _sse_bytes(events: list[dict | str]) -> bytes:
    """Encode a list of events into OpenAI-style SSE bytes.

    dict → ``data: <json>\\n\\n``. str → passed through (for emitting
    ``[DONE]`` sentinels or malformed payloads).
    """
    out: list[str] = []
    for ev in events:
        if isinstance(ev, dict):
            out.append(f"data: {json.dumps(ev)}\n\n")
        else:
            out.append(ev)
    return "".join(out).encode("utf-8")


def _make_client_with_transport(handler) -> LLMClient:
    client = LLMClient(api_key="sk-test")
    client._set_transport_for_tests(httpx.MockTransport(handler))
    return client


@pytest.mark.asyncio
async def test_stream_chat_yields_content_deltas_in_order() -> None:
    events = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": ", "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
        {"choices": [{"delta": {"content": "!"}}]},
        "data: [DONE]\n\n",
    ]
    body = _sse_bytes(events)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/chat/completions")
        sent = json.loads(request.content)
        assert sent["stream"] is True
        assert sent["model"] == "anthropic/claude-haiku-4.5"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
        )

    client = _make_client_with_transport(handler)
    out: list[str] = []
    async for delta in client.stream_chat(
        [ChatMessage("user", "hi")], model="anthropic/claude-haiku-4.5"
    ):
        out.append(delta)
    assert out == ["Hello", ", ", "world", "!"]


@pytest.mark.asyncio
async def test_stream_chat_ignores_keepalive_and_empty_deltas() -> None:
    events = [
        ": ping\n\n",  # comment keepalive
        {"choices": [{"delta": {}}]},  # empty delta
        {"choices": [{"delta": {"role": "assistant"}}]},  # role-only, no content
        {"choices": [{"delta": {"content": "ok"}}]},  # the one we want
        "event: ping\n\n",  # non-data event
        "data: [DONE]\n\n",
    ]
    client = _make_client_with_transport(
        lambda req: httpx.Response(200, content=_sse_bytes(events))
    )
    out = [d async for d in client.stream_chat([ChatMessage("user", "x")], model="m")]
    assert out == ["ok"]


@pytest.mark.asyncio
async def test_stream_chat_skips_malformed_json_but_continues() -> None:
    """A corrupt line must not kill the stream — we log and move on."""
    events = [
        {"choices": [{"delta": {"content": "before"}}]},
        "data: {not valid json\n\n",
        {"choices": [{"delta": {"content": "after"}}]},
        "data: [DONE]\n\n",
    ]
    client = _make_client_with_transport(
        lambda req: httpx.Response(200, content=_sse_bytes(events))
    )
    out = [d async for d in client.stream_chat([ChatMessage("user", "x")], model="m")]
    assert out == ["before", "after"]


@pytest.mark.asyncio
async def test_stream_chat_raises_on_non_2xx_status() -> None:
    client = _make_client_with_transport(
        lambda req: httpx.Response(401, content=b'{"error":{"message":"bad key"}}')
    )
    with pytest.raises(LLMHTTPError) as ei:
        async for _ in client.stream_chat([ChatMessage("user", "x")], model="m"):
            pass
    assert ei.value.status == 401
    assert "bad key" in ei.value.message


import contextlib
import socket
import threading

import uvicorn


def _sse_app(chunks: list[bytes | float]):
    """Build a tiny ASGI app that streams ``chunks`` as SSE.

    A ``float`` entry sleeps for that many seconds between chunks — used
    to simulate mid-stream idle for the timeout / cancellation tests.
    Real HTTP streaming (via a live uvicorn server) is required because
    httpx's in-process transports (MockTransport, ASGITransport) buffer
    the full body before delivering it to the client.
    """

    async def app(scope, receive, send):
        assert scope["type"] == "http"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for chunk in chunks:
            if isinstance(chunk, float):
                await asyncio.sleep(chunk)
                continue
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    return app


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextlib.asynccontextmanager
async def _live_server(app):
    """Run a uvicorn server in a background thread; yield its base URL.

    Background thread rather than asyncio task because uvicorn wants to
    own an event loop for its signal handlers and lifespan; sharing the
    test loop leads to double-installed signal handlers. Cross-thread
    shutdown uses server.should_exit.
    """
    port = _pick_free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    # Uvicorn installs signal handlers by default; disable because we're
    # in a background thread.
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Poll for readiness.
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started, "uvicorn did not start within 2s"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def _client_with_base(base_url: str) -> LLMClient:
    return LLMClient(api_key="sk-test", base_url=base_url)


@pytest.mark.asyncio
async def test_stream_chat_cancellation_closes_connection_cleanly() -> None:
    """Cancelling the consumer task must exit the stream cleanly.

    Load-bearing barge-in property: when ConversationSession does
    pending_llm_task.cancel(), the consumer inside stream_chat stops
    iterating, the async-with exits, and httpx closes the TCP connection.
    OpenRouter stops billing at that point.
    """

    def app_factory(path: str):  # noqa: ARG001
        return _sse_app(
            [
                _sse_bytes([{"choices": [{"delta": {"content": "tok1"}}]}]),
                5.0,  # long stall — we'll cancel before this resolves
                _sse_bytes([{"choices": [{"delta": {"content": "tok2"}}]}]),
                b"data: [DONE]\n\n",
            ]
        )

    async with _live_server(app_factory("/")) as base_url:
        client = _client_with_base(f"{base_url}/api/v1")
        received: list[str] = []

        async def consumer() -> None:
            async for delta in client.stream_chat([ChatMessage("user", "x")], model="m"):
                received.append(delta)

        task = asyncio.create_task(consumer())
        for _ in range(200):
            if received:
                break
            await asyncio.sleep(0.01)
        assert received == ["tok1"], f"expected first token captured, got {received}"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The stall prevented tok2 from ever being sent — the stream was cut.
        assert received == ["tok1"]


@pytest.mark.asyncio
async def test_stream_chat_times_out_on_idle_stream() -> None:
    """If no chunk arrives within read_timeout, raise LLMTimeoutError."""
    app = _sse_app(
        [
            _sse_bytes([{"choices": [{"delta": {"content": "first"}}]}]),
            5.0,  # stall longer than read_timeout
        ]
    )
    async with _live_server(app) as base_url:
        client = _client_with_base(f"{base_url}/api/v1")
        with pytest.raises(LLMTimeoutError):
            async for _ in client.stream_chat(
                [ChatMessage("user", "x")],
                model="m",
                read_timeout=0.3,
                connect_timeout=0.3,
            ):
                pass
