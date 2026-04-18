"""GET /metrics — Prometheus exposition endpoint.

Serves whatever the ``prometheus_client`` default registry has
collected across the process. No auth on this route — /metrics is
scraped by the Prometheus server on the same host network and would
be firewalled at the Tailscale boundary in production.

Worker metrics are surfaced here too: each in-process worker
(VoxCPM, Fun-ASR, VAD+punc, training) instruments the default
registry directly, so this endpoint's output is the union.

See ORCHESTRATION-M8.md §3.1.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["metrics"])


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
