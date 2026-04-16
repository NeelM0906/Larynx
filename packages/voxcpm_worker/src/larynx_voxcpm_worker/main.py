"""Standalone entrypoint for the VoxCPM2 worker.

Not used in M1 — the gateway's lifespan starts the worker in-process via
``WorkerServer`` directly. This entrypoint exists so the worker can be split
into its own supervisord program without code changes when we need to
(likely when a second GPU's worth of work moves here).
"""

from __future__ import annotations

import asyncio

import structlog
from larynx_shared.ipc.client_base import WorkerChannel

from larynx_voxcpm_worker.model_manager import VoxCPMModelManager
from larynx_voxcpm_worker.server import WorkerServer, install_signal_handlers

log = structlog.get_logger(__name__)


async def _run() -> None:
    manager = await asyncio.to_thread(VoxCPMModelManager.from_env)
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    install_signal_handlers(server)
    await server.start()
    # Block forever until SIGTERM triggers server.stop() via the signal handler.
    await asyncio.Event().wait()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
