"""Unauthenticated liveness and readiness probes.

``/health`` = process is up and the event loop is responsive.
``/ready``  = worker client is started and the worker has replied at least
once to a trivial request. M1 treats 'worker loop is running' as ready; the
worker -> gateway response queue is not probed until M3 when backpressure
matters.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, status

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, object]:
    worker_ready = bool(getattr(request.app.state, "worker_ready", False))
    payload = {
        "status": "ok" if worker_ready else "starting",
        "worker": "ready" if worker_ready else "starting",
    }
    if not worker_ready:
        return payload  # 200 but status=starting is intentional; supervisors can poll
    return payload


@router.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "name": "larynx",
        "docs": "/docs",
        "status_endpoint": "/health",
    }


__all__ = ["router", "status"]
