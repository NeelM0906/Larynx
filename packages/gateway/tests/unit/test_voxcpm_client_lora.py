"""Gateway-side VoxCPMClient LoRA methods.

Covers ``load_lora`` / ``unload_lora`` / ``list_loras`` and the
per-request ``lora_name`` kwarg on ``synthesize_text`` and
``synthesize_text_stream``. Runs against the real WorkerServer +
MockVoxCPMBackend — no fakes per the no-fakes-in-tests memory.
"""

from __future__ import annotations

import pytest
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient
from larynx_shared.ipc import WorkerChannel, WorkerError
from larynx_voxcpm_worker.model_manager import MockVoxCPMBackend, VoxCPMModelManager
from larynx_voxcpm_worker.server import WorkerServer


@pytest.fixture
async def client_server():
    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    client = VoxCPMClient(channel)
    await client.start()
    await server.start()
    try:
        yield client
    finally:
        await server.stop()
        await client.stop()


@pytest.mark.asyncio
async def test_list_loras_empty_initially(client_server: VoxCPMClient) -> None:
    assert await client_server.list_loras() == []


@pytest.mark.asyncio
async def test_load_list_unload_flow(client_server: VoxCPMClient) -> None:
    await client_server.load_lora("voice-a", "/tmp/a")
    await client_server.load_lora("voice-b", "/tmp/b")
    assert await client_server.list_loras() == ["voice-a", "voice-b"]

    await client_server.unload_lora("voice-a")
    assert await client_server.list_loras() == ["voice-b"]


@pytest.mark.asyncio
async def test_load_duplicate_raises_worker_error(client_server: VoxCPMClient) -> None:
    await client_server.load_lora("voice-a", "/tmp/a")
    with pytest.raises(WorkerError) as excinfo:
        await client_server.load_lora("voice-a", "/tmp/a")
    assert excinfo.value.code == "lora_invalid"


@pytest.mark.asyncio
async def test_unload_unknown_raises_worker_error(client_server: VoxCPMClient) -> None:
    with pytest.raises(WorkerError) as excinfo:
        await client_server.unload_lora("never-loaded")
    assert excinfo.value.code == "lora_invalid"


@pytest.mark.asyncio
async def test_synthesize_text_threads_lora_name(client_server: VoxCPMClient) -> None:
    await client_server.load_lora("voice-a", "/tmp/a")

    base = await client_server.synthesize_text("hello world", sample_rate=24000)
    with_lora = await client_server.synthesize_text(
        "hello world", sample_rate=24000, lora_name="voice-a"
    )
    # Same text, distinct lora_name path -> distinct PCM (pitch-shifted
    # by the mock backend).
    assert base.pcm_s16le != with_lora.pcm_s16le


@pytest.mark.asyncio
async def test_synthesize_text_stream_threads_lora_name(
    client_server: VoxCPMClient,
) -> None:
    # Confirms the kwarg travels through the streaming path too. We don't
    # assert on content — the one-shot test above covers the pitch shift.
    await client_server.load_lora("voice-a", "/tmp/a")
    chunks_count = 0
    async with client_server.synthesize_text_stream(
        "hello", sample_rate=24000, lora_name="voice-a"
    ) as frames:
        async for _ in frames:
            chunks_count += 1
    assert chunks_count >= 2  # at least one chunk + one done
