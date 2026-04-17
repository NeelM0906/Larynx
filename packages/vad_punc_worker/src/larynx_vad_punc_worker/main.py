"""Standalone entrypoint for the VAD+Punctuation worker.

Not used in M3 (the gateway starts this in-process). Kept here for the
same reason as the VoxCPM / Fun-ASR main.py — when we split workers into
their own supervisord programs the transport stays the same.
"""

from __future__ import annotations

import asyncio

import structlog
from larynx_shared.ipc.client_base import WorkerChannel

from larynx_vad_punc_worker.model_manager import VadPuncModelManager
from larynx_vad_punc_worker.server import WorkerServer, install_signal_handlers

log = structlog.get_logger(__name__)


async def _run() -> None:
    manager = await VadPuncModelManager.from_env()
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    install_signal_handlers(server)
    await server.start()
    await asyncio.Event().wait()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
