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

import asyncio
import os
import pathlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from larynx_funasr_worker.model_manager import FunASRModelManager
from larynx_funasr_worker.server import WorkerServer as FunASRWorkerServer
from larynx_vad_punc_worker.model_manager import VadPuncModelManager
from larynx_vad_punc_worker.server import WorkerServer as VadPuncWorkerServer
from larynx_voxcpm_worker.model_manager import VoxCPMModelManager
from larynx_voxcpm_worker.server import WorkerServer

from larynx_gateway.config import Settings, get_settings
from larynx_gateway.db.session import dispose_engine, get_session, init_engine
from larynx_gateway.logging import configure_logging
from larynx_gateway.routes import (
    conversation,
    finetune,
    health,
    stt,
    stt_stream,
    tts,
    tts_stream,
    voices,
)
from larynx_gateway.services.boot_reconcile import load_lora_voices, reap_orphan_jobs
from larynx_gateway.services.latent_cache import LatentCache, build_redis_client
from larynx_gateway.services.llm_client import LLMClient
from larynx_gateway.services.training_logs import TrainingLogStore
from larynx_gateway.services.voice_files import resolve_data_dir
from larynx_gateway.workers_client.base import WorkerChannel
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient
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

    # M7 fine-tune scaffolding — shared across every training job.
    # Tests may pre-populate ``training_subprocess_hook`` on app.state
    # before the lifespan runs; production uses the default
    # run_training_subprocess.
    app.state.training_log_store = TrainingLogStore(redis_client)
    app.state.gpu_train_lock = asyncio.Lock()
    app.state.ft_jobs = {}

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

    # Fun-ASR STT worker + client (in-process for M3; splits to its own
    # supervisord program when we move to separate GPU/CPU containers).
    os.environ.setdefault("LARYNX_STT_MODE", settings.larynx_stt_mode)
    os.environ.setdefault("LARYNX_FUNASR_GPU", str(settings.larynx_funasr_gpu))
    os.environ.setdefault("LARYNX_FUNASR_GPU_MEM_UTIL", str(settings.larynx_funasr_gpu_mem_util))

    funasr_manager = await FunASRModelManager.from_env()
    funasr_channel = WorkerChannel()
    funasr_worker = FunASRWorkerServer(funasr_channel, funasr_manager)
    funasr_client = FunASRClient(funasr_channel)
    await funasr_client.start()
    await funasr_worker.start()

    app.state.funasr_manager = funasr_manager
    app.state.funasr_worker = funasr_worker
    app.state.funasr_client = funasr_client

    # VAD + Punctuation worker (CPU-only; always in-process since the
    # overhead of IPC dominates its CPU cost).
    os.environ.setdefault("LARYNX_VAD_PUNC_MODE", settings.larynx_vad_punc_mode)
    os.environ.setdefault("LARYNX_VAD_MODEL", settings.larynx_vad_model)
    os.environ.setdefault("LARYNX_PUNC_MODEL", settings.larynx_punc_model)

    vad_punc_manager = await VadPuncModelManager.from_env()
    vad_punc_channel = WorkerChannel()
    vad_punc_worker = VadPuncWorkerServer(vad_punc_channel, vad_punc_manager)
    vad_punc_client = VadPuncClient(vad_punc_channel)
    await vad_punc_client.start()
    await vad_punc_worker.start()

    app.state.vad_punc_manager = vad_punc_manager
    app.state.vad_punc_worker = vad_punc_worker
    app.state.vad_punc_client = vad_punc_client

    # M7 boot reconciliation — must run AFTER voxcpm_worker is ready and
    # BEFORE we serve HTTP traffic, so a synthesis request using a LoRA
    # voice never races ahead of its register_lora. See
    # ORCHESTRATION-M7.md §8.5.
    async for db_session in get_session():
        unloaded = await load_lora_voices(db_session, client)
        await reap_orphan_jobs(db_session, grace_seconds=settings.larynx_ft_orphan_grace_seconds)
        # Stash the loaded/failed status for the /v1/voices route so it
        # can surface an ``unloaded`` flag on failed rows. Transient —
        # not persisted, recomputed on next boot.
        app.state.lora_load_status = {vid: ok for vid, ok in unloaded.items()}
        break

    # LLM client (OpenRouter). Shared across conversation sessions; one
    # httpx connection pool per gateway process. API key can be empty
    # during non-M5 test runs — the client only blows up on first use.
    llm_client = LLMClient(
        api_key=settings.larynx_openrouter_api_key,
        base_url=settings.larynx_openrouter_base_url,
        http_referer=settings.larynx_openrouter_http_referer or None,
        x_title=settings.larynx_openrouter_x_title or None,
    )
    app.state.llm_client = llm_client
    app.state.llm_default_model = settings.larynx_llm_default_model

    app.state.worker_ready = True

    log.info(
        "gateway.ready",
        tts_mode=manager.mode.value,
        stt_mode=funasr_manager.mode.value,
        vad_punc_mode=vad_punc_manager.mode.value,
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
        await funasr_worker.stop()
        await funasr_client.stop()
        await vad_punc_worker.stop()
        await vad_punc_client.stop()
        await llm_client.aclose()
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
    app.include_router(tts_stream.router)
    app.include_router(voices.router)
    app.include_router(stt.router)
    app.include_router(stt_stream.router)
    app.include_router(conversation.router)
    app.include_router(finetune.router)
    from larynx_gateway.routes import openai_compat

    app.include_router(openai_compat.router)

    # Serve the playground HTML alongside the API so a single ngrok/
    # Tailscale tunnel can cover both origins. Paths under /playground
    # are static HTML + JS; all API paths (/v1/*, /health, /ready) are
    # registered above and take precedence.
    _playground_dir = pathlib.Path(__file__).resolve().parents[4] / "apps" / "playground-test"
    if _playground_dir.is_dir():
        app.mount(
            "/playground", StaticFiles(directory=_playground_dir, html=True), name="playground"
        )

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
