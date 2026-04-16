"""FastAPI dependency-injection helpers.

The app holds long-lived objects (worker client, DB engine) on
``app.state``; these helpers surface them as FastAPI dependencies so routes
stay decoupled from how they were constructed.
"""

from __future__ import annotations

from fastapi import Request

from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient


def get_voxcpm_client(request: Request) -> VoxCPMClient:
    client: VoxCPMClient | None = getattr(request.app.state, "voxcpm_client", None)
    if client is None:
        raise RuntimeError("voxcpm_client not initialised (lifespan did not run?)")
    return client
