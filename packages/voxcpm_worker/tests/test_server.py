"""End-to-end IPC test: gateway-side client <-> in-process worker server."""

from __future__ import annotations

import pytest
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient
from larynx_shared.ipc import WorkerChannel
from larynx_voxcpm_worker.model_manager import (
    MockVoxCPMBackend,
    ModelMode,
    VoxCPMModelManager,
)
from larynx_voxcpm_worker.server import WorkerServer


@pytest.mark.asyncio
async def test_client_roundtrip_over_in_process_channel() -> None:
    manager = VoxCPMModelManager(MockVoxCPMBackend(), ModelMode.MOCK)
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    client = VoxCPMClient(channel)

    await client.start()
    await server.start()
    try:
        resp = await client.synthesize_text("ping", sample_rate=16000)
        assert resp.sample_rate == 16000
        assert resp.duration_ms > 0
        assert len(resp.pcm_s16le) == 2 * int(16000 * resp.duration_ms / 1000)
    finally:
        await server.stop()
        await client.stop()
