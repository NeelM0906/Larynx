"""FastAPI app factory + lifespan.

For M1 the gateway and VoxCPM2 worker share a Python process. The lifespan
(a) initialises the DB engine, (b) loads the VoxCPM2 model (mock or real),
(c) starts the in-process worker loop and the gateway-side client over a
shared ``WorkerChannel``. Shutdown is symmetric.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from larynx_voxcpm_worker.model_manager import VoxCPMModelManager
from larynx_voxcpm_worker.server import WorkerServer

from larynx_gateway.config import Settings, get_settings
from larynx_gateway.db.session import dispose_engine, init_engine
from larynx_gateway.logging import configure_logging
from larynx_gateway.routes import health, tts
from larynx_gateway.workers_client.base import WorkerChannel
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    # DB. Creating the engine doesn't require the DB to be reachable; it only
    # matters when the first session is opened, which happens in M2. For M1
    # this means the gateway starts cleanly even if Postgres is down.
    init_engine(settings.database_url)

    # VoxCPM worker + client (in-process).
    import os

    os.environ.setdefault("LARYNX_TTS_MODE", settings.larynx_tts_mode)
    os.environ.setdefault("LARYNX_VOXCPM_GPU", str(settings.larynx_voxcpm_gpu))

    manager = await VoxCPMModelManager.from_env()
    channel = WorkerChannel()
    worker = WorkerServer(channel, manager)
    client = VoxCPMClient(channel)

    await client.start()
    await worker.start()

    app.state.voxcpm_manager = manager
    app.state.voxcpm_worker = worker
    app.state.voxcpm_client = client
    app.state.worker_ready = True

    log.info(
        "gateway.ready",
        tts_mode=manager.mode.value,
        port=settings.larynx_port,
        db=_redact(settings.database_url),
    )

    try:
        yield
    finally:
        app.state.worker_ready = False
        await worker.stop()
        await client.stop()
        await dispose_engine()
        log.info("gateway.shutdown_complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.larynx_log_level, json_output=settings.larynx_log_json)

    app = FastAPI(
        title="Larynx Gateway",
        version="0.1.0",
        description="Self-hosted voice AI platform — M1",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.worker_ready = False

    app.include_router(health.router)
    app.include_router(tts.router)

    return app


def _redact(url: str) -> str:
    """Strip credentials before logging a DB URL."""
    if "@" not in url:
        return url
    scheme_rest = url.split("://", 1)
    if len(scheme_rest) != 2:
        return url
    scheme, rest = scheme_rest
    _, host = rest.split("@", 1)
    return f"{scheme}://***@{host}"


app = create_app()


def run() -> None:
    """Entrypoint used by `uv run larynx-gateway`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "larynx_gateway.main:app",
        host=settings.larynx_host,
        port=settings.larynx_port,
        log_level=settings.larynx_log_level.lower(),
    )


if __name__ == "__main__":
    run()
