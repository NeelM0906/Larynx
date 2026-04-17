"""End-to-end IPC test: LoRA hot-swap messages round-trip.

Exercises the full request → server.dispatch → backend → response path for
LoadLoraRequest / UnloadLoraRequest / ListLorasRequest, plus the
``lora_name`` threading on SynthesizeRequest. Uses the mock backend so no
GPU / no real LoRA artifact is required.

Gateway-side convenience methods (``VoxCPMClient.load_lora`` etc.) are
intentionally NOT tested here — those are added in their own commit per
the design doc's task split (§8.1 is two commits: worker API, then
client methods).
"""

from __future__ import annotations

import pytest
from larynx_shared.ipc import WorkerChannel
from larynx_shared.ipc.messages import (
    ErrorMessage,
    ListLorasRequest,
    ListLorasResponse,
    LoadLoraRequest,
    LoadLoraResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    UnloadLoraRequest,
    UnloadLoraResponse,
)
from larynx_voxcpm_worker.model_manager import MockVoxCPMBackend, VoxCPMModelManager
from larynx_voxcpm_worker.server import WorkerServer


async def _request(channel: WorkerChannel, message, *, timeout: float = 5.0):
    """Send one request and wait for exactly one response (for unary RPCs)."""
    import asyncio

    await channel.requests.put(message)
    return await asyncio.wait_for(channel.responses.get(), timeout=timeout)


@pytest.mark.asyncio
async def test_load_list_unload_roundtrip() -> None:
    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    await server.start()
    try:
        # Initial list is empty.
        list_resp = await _request(channel, ListLorasRequest())
        assert isinstance(list_resp, ListLorasResponse)
        assert list_resp.names == []

        # Load two LoRAs.
        load_a = await _request(channel, LoadLoraRequest(name="voice-a", path="/tmp/a"))
        load_b = await _request(channel, LoadLoraRequest(name="voice-b", path="/tmp/b"))
        assert isinstance(load_a, LoadLoraResponse)
        assert isinstance(load_b, LoadLoraResponse)
        assert load_a.name == "voice-a"
        assert load_b.name == "voice-b"

        # Both show up in list (sorted).
        list_resp = await _request(channel, ListLorasRequest())
        assert isinstance(list_resp, ListLorasResponse)
        assert list_resp.names == ["voice-a", "voice-b"]

        # Unload one, list reflects it.
        unload_a = await _request(channel, UnloadLoraRequest(name="voice-a"))
        assert isinstance(unload_a, UnloadLoraResponse)
        list_resp = await _request(channel, ListLorasRequest())
        assert isinstance(list_resp, ListLorasResponse)
        assert list_resp.names == ["voice-b"]
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_load_duplicate_returns_lora_invalid_error() -> None:
    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    await server.start()
    try:
        await _request(channel, LoadLoraRequest(name="voice-a", path="/tmp/a"))
        second = await _request(channel, LoadLoraRequest(name="voice-a", path="/tmp/a"))
        assert isinstance(second, ErrorMessage)
        assert second.code == "lora_invalid"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_unload_unknown_returns_lora_invalid_error() -> None:
    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    await server.start()
    try:
        resp = await _request(channel, UnloadLoraRequest(name="never-loaded"))
        assert isinstance(resp, ErrorMessage)
        assert resp.code == "lora_invalid"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_synthesize_with_registered_lora_differs_from_base() -> None:
    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    await server.start()
    try:
        await _request(channel, LoadLoraRequest(name="voice-a", path="/tmp/a"))

        base = await _request(
            channel, SynthesizeRequest(text="hello world", sample_rate=24000, lora_name=None)
        )
        with_lora = await _request(
            channel,
            SynthesizeRequest(text="hello world", sample_rate=24000, lora_name="voice-a"),
        )
        assert isinstance(base, SynthesizeResponse)
        assert isinstance(with_lora, SynthesizeResponse)
        assert base.pcm_s16le != with_lora.pcm_s16le
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_synthesize_with_unknown_lora_returns_invalid_input() -> None:
    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    await server.start()
    try:
        resp = await _request(
            channel,
            SynthesizeRequest(text="hello world", sample_rate=24000, lora_name="missing"),
        )
        assert isinstance(resp, ErrorMessage)
        # The backend raises ValueError for unknown lora_name, which the
        # synthesize handler maps to invalid_input (matches the existing
        # empty-text path).
        assert resp.code == "invalid_input"
    finally:
        await server.stop()
