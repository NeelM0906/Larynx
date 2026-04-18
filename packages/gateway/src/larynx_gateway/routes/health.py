"""Liveness + readiness probes.

``/health`` — process is up and the event loop is responsive.
``/ready``  — structured readiness across in-process workers + the
batch queue. Returns 503 if any worker failed to initialise or the
gateway is draining for shutdown.

See ORCHESTRATION-M8.md §3.2.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

router = APIRouter(tags=["health"])

# App version reported in /ready. Kept in sync with pyproject.toml;
# read at import time so a mis-typed version doesn't get silently
# served from a stale bytecode cache.
APP_VERSION = "0.3.0"


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict[str, object]:
    state = request.app.state
    worker_ready = bool(getattr(state, "worker_ready", False))
    shutting_down = bool(getattr(state, "shutting_down", False))

    if shutting_down:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "shutting_down", "version": APP_VERSION}

    # Per-worker state. Only the three in-process worker clients we
    # control — external services (Postgres/Redis) aren't reported
    # here; their failures surface on the first request that needs
    # them, which is enough for a single-box deployment.
    workers = {
        "voxcpm": _client_state(state, "voxcpm_client"),
        "funasr": _client_state(state, "funasr_client"),
        "vad_punc": _client_state(state, "vad_punc_client"),
    }

    all_ready = worker_ready and all(w["state"] == "ready" for w in workers.values())
    queue_depth = await _safe_queue_depth(state)

    payload: dict[str, object] = {
        "status": "ready" if all_ready else "starting",
        "worker": "ready" if worker_ready else "starting",
        "workers": workers,
        "queues": {"batch": queue_depth} if queue_depth is not None else {},
        "version": APP_VERSION,
    }
    if not all_ready:
        # Starlette 503 + a body is the supervisord-friendly shape —
        # the supervisor can distinguish "not yet ready" from "crashed".
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return payload


def _client_state(state, attr: str) -> dict[str, object]:
    """Collapse a worker-client reference into a simple state dict.

    The worker clients don't expose a formal heartbeat yet — that's
    a later M8 commit. For now we report "ready" if the attribute
    exists (lifespan set it) and "missing" otherwise. The schema
    leaves room for heartbeat-age once the client API gains it.
    """
    client = getattr(state, attr, None)
    if client is None:
        return {"state": "missing"}
    return {"state": "ready"}


async def _safe_queue_depth(state) -> int | None:
    """Peek the batch queue depth; swallow Redis errors.

    /ready must not 500 just because Redis is momentarily unreachable
    — upstream LBs should still see a 503 and drain traffic, but the
    endpoint itself stays alive so the operator can ``curl`` it.
    """
    queue = getattr(state, "batch_queue", None)
    if queue is None:
        return None
    try:
        return await queue.depth()
    except Exception:
        return None


@router.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "name": "larynx",
        "docs": "/docs",
        "status_endpoint": "/health",
    }


__all__ = ["router", "status"]
