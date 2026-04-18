"""Hardening surface tests — /ready, /metrics, body-size limits."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


async def test_ready_reports_worker_state(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """/ready returns 200 with structured workers + queue state when ready."""
    resp = await client.get("/ready")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ready"
    assert body["worker"] == "ready"
    assert body["version"].count(".") == 2
    # All three in-process worker clients should register.
    for name in ("voxcpm", "funasr", "vad_punc"):
        assert body["workers"][name]["state"] == "ready", body


async def test_metrics_exposes_prometheus_format(client: AsyncClient) -> None:
    """/metrics responds in the Prometheus text format with our histograms."""
    # Warm up the histogram with one real request.
    await client.get("/health")

    resp = await client.get("/metrics")
    assert resp.status_code == 200
    ct = resp.headers["content-type"]
    assert "text/plain" in ct, ct
    body = resp.text
    # Middleware must have published its counters + histograms.
    assert "larynx_request_duration_seconds" in body
    # The warm-up /health GET should show up as a 2xx observation.
    assert 'route="/health"' in body


async def test_body_size_limit_rejects_oversized_json(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """POST to /v1/batch with an oversized body is 413'd at the middleware.

    Body-size middleware peeks ``Content-Length`` and short-circuits
    before any route-level parsing runs.
    """
    # 3 MB > 2 MB /v1/batch limit
    oversize = "A" * (3 * 1024 * 1024)
    resp = await client.post(
        "/v1/batch",
        headers={**auth_headers, "content-length": str(len(oversize))},
        content=oversize,
    )
    assert resp.status_code == 413, resp.text
    assert resp.json()["detail"]["code"] == "body_too_large"


@pytest.mark.parametrize(
    ("path", "size_bytes", "expect_413"),
    [
        ("/v1/tts", 51 * 1024 * 1024, True),  # > 50 MB limit
        ("/v1/tts", 1 * 1024 * 1024, False),  # well under
        ("/v1/audio/speech", 128 * 1024, True),  # > 64 KB limit
    ],
)
async def test_body_size_limit_matrix(
    client: AsyncClient,
    auth_headers: dict[str, str],
    path: str,
    size_bytes: int,
    expect_413: bool,
) -> None:
    body = "x" * size_bytes
    resp = await client.post(
        path,
        headers={
            **auth_headers,
            "content-length": str(len(body)),
            "content-type": "application/json",
        },
        content=body,
    )
    if expect_413:
        assert resp.status_code == 413, f"{path} {size_bytes} -> {resp.status_code}"
    else:
        assert resp.status_code != 413, f"{path} {size_bytes} -> {resp.status_code}"
