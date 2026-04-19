"""Standalone entrypoint for the Fun-ASR worker.

Unused in the in-process gateway path — the gateway's lifespan starts
``WorkerServer`` directly on a shared ``WorkerChannel``. This
entrypoint is what supervisord runs when the worker is split into its
own process (on the GPU box where the Fun-ASR-vllm process holds the
CUDA context). Standalone-process only: this is where the Prometheus
metrics sidecar binds :9101.
"""

from __future__ import annotations

import asyncio
import signal

import structlog
from larynx_shared.ipc.client_base import WorkerChannel

from larynx_funasr_worker.metrics_server import MetricsSidecar
from larynx_funasr_worker.model_manager import FunASRModelManager
from larynx_funasr_worker.server import WorkerServer

log = structlog.get_logger(__name__)


async def _run() -> None:
    manager = await FunASRModelManager.from_env()
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    sidecar = MetricsSidecar()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    await sidecar.start()
    await server.start()
    try:
        await stop_event.wait()
    finally:
        await server.stop()
        await sidecar.stop()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
