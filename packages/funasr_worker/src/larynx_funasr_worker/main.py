"""Standalone entrypoint for the Fun-ASR worker.

Unused in M3 — the gateway's lifespan starts this worker in-process via
``WorkerServer`` directly on a shared ``WorkerChannel``. This entrypoint
exists so the worker can split into its own supervisord program (on the
GPU box where the Fun-ASR-vllm process holds the CUDA context) without
code changes.
"""

from __future__ import annotations

import asyncio

import structlog
from larynx_shared.ipc.client_base import WorkerChannel

from larynx_funasr_worker.model_manager import FunASRModelManager
from larynx_funasr_worker.server import WorkerServer, install_signal_handlers

log = structlog.get_logger(__name__)


async def _run() -> None:
    manager = await FunASRModelManager.from_env()
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    install_signal_handlers(server)
    await server.start()
    await asyncio.Event().wait()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
