"""Tests for the /metrics/workers proxy.

Covers the §3.1 contract: concatenate two sidecar scrapes, and stay
200 when one upstream is unreachable (operability > purity — a single
worker being down must not blind the operator to the other one).

Uses :class:`httpx.MockTransport` via FastAPI dependency override so
we never actually touch :9101/:9102 — CI boxes don't have the workers
running as standalone processes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from httpx import AsyncClient, MockTransport, Request, Response
from larynx_gateway.routes.metrics import get_scrape_client


def _both_ok(request: Request) -> Response:
    url = str(request.url)
    if ":9101" in url:
        return Response(
            200,
            text=(
                "# HELP larynx_stt_rtfx STT rtfx.\n"
                "# TYPE larynx_stt_rtfx histogram\n"
                'larynx_stt_rtfx_count{model="nano"} 7\n'
            ),
            headers={"content-type": "text/plain; version=0.0.4"},
        )
    if ":9102" in url:
        return Response(
            200,
            text=(
                "# HELP larynx_request_duration_seconds VAD+Punc duration.\n"
                "# TYPE larynx_request_duration_seconds histogram\n"
                'larynx_request_duration_seconds_bucket{method="segment",le="0.05"} 3\n'
            ),
            headers={"content-type": "text/plain; version=0.0.4"},
        )
    return Response(404)


def _vad_down(request: Request) -> Response:
    url = str(request.url)
    if ":9101" in url:
        return Response(
            200,
            text=('# HELP larynx_stt_rtfx STT rtfx.\nlarynx_stt_rtfx_count{model="nano"} 4\n'),
            headers={"content-type": "text/plain"},
        )
    # vad_punc: simulate a connection refused.
    raise httpx.ConnectError("connection refused (simulated)")


def _both_down(_request: Request) -> Response:
    raise httpx.ConnectError("connection refused (simulated)")


def _override_with(handler):
    """Build a FastAPI dependency replacement for :func:`get_scrape_client`.

    The returned coroutine yields an httpx client backed by the given
    ``MockTransport`` handler — matches the original's async-generator
    shape so FastAPI's Depends machinery doesn't notice the swap.
    """

    async def _override() -> AsyncIterator[httpx.AsyncClient]:
        async with AsyncClient(transport=MockTransport(handler)) as c:
            yield c

    return _override


def _app_of(client: AsyncClient):
    # conftest's ``client`` fixture stashes the ASGI app on the transport —
    # reach through so we can register dependency overrides per test.
    return client._transport.app  # type: ignore[attr-defined]


async def test_metrics_workers_concatenates_both_upstreams(client: AsyncClient) -> None:
    app = _app_of(client)
    app.dependency_overrides[get_scrape_client] = _override_with(_both_ok)
    try:
        resp = await client.get("/metrics/workers")
    finally:
        app.dependency_overrides.pop(get_scrape_client, None)

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "# source: funasr" in body
    assert "larynx_stt_rtfx_count" in body
    assert "# source: vad_punc" in body
    assert "larynx_request_duration_seconds_bucket" in body


async def test_metrics_workers_returns_200_when_one_upstream_down(client: AsyncClient) -> None:
    app = _app_of(client)
    app.dependency_overrides[get_scrape_client] = _override_with(_vad_down)
    try:
        resp = await client.get("/metrics/workers")
    finally:
        app.dependency_overrides.pop(get_scrape_client, None)

    assert resp.status_code == 200, resp.text
    body = resp.text
    # Healthy upstream still contributes.
    assert "# source: funasr" in body
    assert "larynx_stt_rtfx_count" in body
    # Unhealthy upstream marked via comment; no stack trace leaked.
    assert "# vad_punc unreachable: ConnectError" in body


async def test_metrics_workers_still_200_when_all_upstreams_down(
    client: AsyncClient,
) -> None:
    app = _app_of(client)
    app.dependency_overrides[get_scrape_client] = _override_with(_both_down)
    try:
        resp = await client.get("/metrics/workers")
    finally:
        app.dependency_overrides.pop(get_scrape_client, None)

    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "# funasr unreachable" in body
    assert "# vad_punc unreachable" in body
