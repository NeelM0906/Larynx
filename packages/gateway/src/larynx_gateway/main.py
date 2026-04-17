"""FastAPI app factory + lifespan.

Gateway + VoxCPM2 worker share a Python process. Lifespan responsibilities:
- initialise DB engine + Redis client
- load the VoxCPM2 model (mock or real) via VoxCPMModelManager
- start the in-process worker loop and the gateway-side client over a
  shared WorkerChannel
- construct the LatentCache (Redis + disk) and hand it to routes
Shutdown is symmetric.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from larynx_voxcpm_worker.model_manager import VoxCPMModelManager
from larynx_voxcpm_worker.server import WorkerServer

from larynx_gateway.config import Settings, get_settings
from larynx_gateway.db.session import dispose_engine, init_engine
from larynx_gateway.logging import configure_logging
from larynx_gateway.routes import health, tts, voices
from larynx_gateway.services.latent_cache import LatentCache, build_redis_client
from larynx_gateway.services.voice_files import resolve_data_dir
from larynx_gateway.workers_client.base import WorkerChannel
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    # DB. Engine creation is lazy about actually connecting; requests that
    # need a session will fail if Postgres is down, but the gateway boots.
    init_engine(settings.database_url)

    # Ensure data_dir exists before the worker/cache try to write into it.
    data_dir = resolve_data_dir(settings.larynx_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    app.state.data_dir = data_dir

    # Redis + latent cache. Connection is lazy — if Redis isn't reachable
    # the first request touching the cache will raise; the gateway itself
    # still boots so /health + migrate workflows work.
    redis_client = build_redis_client(settings.redis_url)
    latent_cache = LatentCache(redis_client, data_dir, ttl_s=settings.larynx_latent_cache_ttl_s)
    app.state.redis_client = redis_client
    app.state.latent_cache = latent_cache
    app.state.design_ttl_s = settings.larynx_voice_design_ttl_s

    # VoxCPM worker + client (in-process).
    os.environ.setdefault("LARYNX_TTS_MODE", settings.larynx_tts_mode)
    os.environ.setdefault("LARYNX_VOXCPM_GPU", str(settings.larynx_voxcpm_gpu))
    os.environ.setdefault("LARYNX_VOXCPM_MODEL", settings.larynx_voxcpm_model)
    os.environ.setdefault(
        "LARYNX_VOXCPM_INFERENCE_TIMESTEPS", str(settings.larynx_voxcpm_inference_timesteps)
    )

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
        redis=_redact(settings.redis_url),
        data_dir=str(data_dir),
    )

    try:
        yield
    finally:
        app.state.worker_ready = False
        await worker.stop()
        await client.stop()
        try:
            await redis_client.aclose()
        except Exception:
            pass
        await dispose_engine()
        log.info("gateway.shutdown_complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.larynx_log_level, json_output=settings.larynx_log_json)

    app = FastAPI(
        title="Larynx Gateway",
        version="0.2.0",
        description="Self-hosted voice AI platform — M2",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.worker_ready = False

    app.include_router(health.router)
    app.include_router(tts.router)
    app.include_router(voices.router)

    return app


def _redact(url: str) -> str:
    """Strip credentials before logging a DB/Redis URL."""
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
