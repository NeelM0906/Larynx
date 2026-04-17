"""OpenRouter streaming chat client.

OpenRouter exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint
that returns Server-Sent Events when ``stream=true``. This module wraps it
in a tiny async iterator that yields content deltas as plain strings. That
is all the orchestrator needs — sentence-boundary detection, history
management, and token accounting all live upstream.

Cancellation: ``stream_chat`` is a regular async generator. If the
consumer stops iterating (e.g. because its owning task got ``.cancel()``-d
on barge-in), the ``async with`` context manager for the httpx stream
exits, which closes the underlying TCP connection. OpenRouter stops
billing at that point. We do not need to ``await`` any explicit cancel
RPC — HTTP streaming semantics are the cancellation protocol.

Timeouts: one knob, ``read_timeout``. This is the *inter-chunk* idle
window — if no SSE event arrives for that long the stream aborts with
``LLMTimeoutError``. There is deliberately no total-duration timeout;
long responses are the caller's business. Default 15s matches
ORCHESTRATION.md v2 §5 E2.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import structlog

log = structlog.get_logger(__name__)


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_READ_TIMEOUT_S = 15.0
DEFAULT_CONNECT_TIMEOUT_S = 5.0


class LLMError(Exception):
    """Base class for LLM client errors. ``code`` is stable for routing."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class LLMTimeoutError(LLMError):
    """Raised when no SSE chunk arrives within ``read_timeout``."""

    def __init__(self, message: str = "no chunk within read timeout") -> None:
        super().__init__("llm_timeout", message)


class LLMHTTPError(LLMError):
    """Upstream returned a non-2xx status (auth, rate limit, etc)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__("llm_http_error", message)
        self.status = status


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient:
    """Thin OpenRouter chat-completions streaming client.

    Single instance is safe for concurrent sessions — httpx's
    ``AsyncClient`` owns a connection pool. Construct at gateway startup,
    reuse across sessions, ``aclose()`` at shutdown.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        http_referer: str | None = None,
        x_title: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # OpenRouter asks senders to identify themselves via these headers
        # so they can attribute traffic to the right app. Not required
        # but polite; configure at gateway boot.
        self._default_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if http_referer:
            self._default_headers["HTTP-Referer"] = http_referer
        if x_title:
            self._default_headers["X-Title"] = x_title
        # Transport is settable for tests; default is the real one.
        self._transport: httpx.AsyncBaseTransport | None = None

    def _set_transport_for_tests(self, transport: httpx.AsyncBaseTransport) -> None:
        """Inject a mock transport. Tests only — no public API guarantee."""
        self._transport = transport

    @asynccontextmanager
    async def _client(self, connect_timeout: float, read_timeout: float) -> AsyncIterator[httpx.AsyncClient]:
        # New client per call so timeouts don't leak across sessions with
        # different knob settings. Connection pooling inside a single
        # stream_chat call is what matters; across calls we can afford the
        # handshake (~1 RTT over TLS).
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )
        kwargs: dict[str, object] = {"timeout": timeout, "headers": self._default_headers}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        async with httpx.AsyncClient(**kwargs) as client:  # type: ignore[arg-type]
            yield client

    async def stream_chat(
        self,
        messages: Iterable[ChatMessage | Mapping[str, str]],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        read_timeout: float = DEFAULT_READ_TIMEOUT_S,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_S,
        extra_body: Mapping[str, object] | None = None,
    ) -> AsyncIterator[str]:
        """Stream content deltas from OpenRouter chat/completions.

        Yields each non-empty ``choices[0].delta.content`` as a string.
        Completion role / tool call deltas are ignored — this client only
        surfaces user-visible text.

        Raises:
            LLMTimeoutError: inter-chunk idle exceeded ``read_timeout``.
            LLMHTTPError: upstream non-2xx.
            LLMError: malformed SSE, transport failures.
        """
        body: dict[str, object] = {
            "model": model,
            "stream": True,
            "messages": [
                m.to_dict() if isinstance(m, ChatMessage) else dict(m)
                for m in messages
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if extra_body:
            body.update(extra_body)

        url = f"{self._base_url}/chat/completions"
        async with self._client(connect_timeout=connect_timeout, read_timeout=read_timeout) as client:
            try:
                async with client.stream("POST", url, json=body) as resp:
                    if resp.status_code // 100 != 2:
                        text = (await resp.aread()).decode("utf-8", errors="replace")
                        raise LLMHTTPError(resp.status_code, text[:500])
                    async for delta in _iter_sse_deltas(resp):
                        if delta:
                            yield delta
            except httpx.ReadTimeout as e:
                raise LLMTimeoutError(str(e)) from e
            except httpx.HTTPError as e:
                raise LLMError("llm_transport_error", str(e)) from e

    async def aclose(self) -> None:
        """Lifecycle hook. Nothing to close right now — each ``stream_chat``
        call owns its own httpx client. Exists so callers can treat the
        lifetime symmetrically to other gateway clients."""


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


async def _iter_sse_deltas(response: httpx.Response) -> AsyncIterator[str]:
    """Parse OpenAI-shaped SSE and yield ``delta.content`` strings.

    SSE lines are CRLF- or LF-terminated. Events are delimited by blank
    lines. Each event we care about starts ``data: <json>``. A terminal
    ``data: [DONE]`` may or may not arrive — either way, stream close is
    authoritative and we just return.

    OpenRouter (and some upstreams it proxies) emit ``: comment`` keepalive
    lines periodically. We skip anything that isn't a ``data:`` event.
    """
    async for line in response.aiter_lines():
        if not line:
            continue
        if line.startswith(":"):
            # SSE comment / keepalive. Ignore.
            continue
        if not line.startswith("data:"):
            # Some providers emit ``event:`` / ``id:`` lines we don't use.
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            log.warning("llm.sse_parse_error", line=payload[:200])
            continue
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield content


__all__ = [
    "ChatMessage",
    "LLMClient",
    "LLMError",
    "LLMHTTPError",
    "LLMTimeoutError",
]
