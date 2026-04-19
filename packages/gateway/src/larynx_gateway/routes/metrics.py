"""GET /metrics — Prometheus exposition endpoint.

Serves whatever the ``prometheus_client`` default registry has
collected across the process. No auth on this route — /metrics is
scraped by the Prometheus server on the same host network and would
be firewalled at the Tailscale boundary in production.

Worker metrics are surfaced here too: each in-process worker
(VoxCPM, Fun-ASR, VAD+punc, training) instruments the default
registry directly, so this endpoint's output is the union.

``/metrics/workers`` proxies the per-worker sidecars (funasr on
:9101, vad_punc on :9102 by default) and returns the concatenation
as a single text/plain body. The proxy is operability-first — if a
sidecar is unreachable, the endpoint still returns 200 with a
comment marking that worker as down rather than failing the whole
scrape.

See ORCHESTRATION-M8.md §3.1 + §7.2.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import structlog
from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from larynx_gateway.config import Settings, get_settings

router = APIRouter(tags=["metrics"])
log = structlog.get_logger(__name__)

# Per-worker scrape budget. 2s is plenty for a Prometheus exposition
# response — if a sidecar is slower than this it's effectively down
# for the purposes of this aggregate.
_WORKER_SCRAPE_TIMEOUT_S = 2.0


async def get_scrape_client() -> AsyncIterator[httpx.AsyncClient]:
    """Yield a short-lived httpx client for upstream worker scrapes.

    Dependency-injected so tests can override with an
    :class:`httpx.MockTransport` without monkeypatching the stdlib.
    """
    async with httpx.AsyncClient(timeout=_WORKER_SCRAPE_TIMEOUT_S) as client:
        yield client


def _worker_targets(settings: Settings) -> list[tuple[str, str]]:
    return [
        ("funasr", settings.larynx_funasr_metrics_url),
        ("vad_punc", settings.larynx_vad_punc_metrics_url),
    ]


async def _scrape(client: httpx.AsyncClient, name: str, url: str) -> bytes:
    """Fetch one worker's /metrics and wrap it with a source header.

    Never raises: a failed scrape becomes a comment line in the output
    so the aggregated response still parses as valid Prometheus text.
    """
    header = f"# source: {name} ({url})\n".encode()
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        log.warning(
            "metrics.worker_scrape_failed",
            target=name,
            url=url,
            err=type(e).__name__,
            detail=str(e),
        )
        return f"# {name} unreachable: {type(e).__name__}\n".encode()
    return header + resp.content


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@router.get("/metrics/workers", include_in_schema=False)
async def metrics_workers(
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_scrape_client),
) -> Response:
    targets = _worker_targets(settings)
    pieces = await asyncio.gather(*(_scrape(client, name, url) for name, url in targets))
    body = b"\n".join(pieces)
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)
