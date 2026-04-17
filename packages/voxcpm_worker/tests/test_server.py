"""End-to-end IPC test: gateway-side client <-> in-process worker server."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient
from larynx_shared.ipc import WorkerChannel
from larynx_voxcpm_worker.model_manager import MockVoxCPMBackend, VoxCPMModelManager
from larynx_voxcpm_worker.server import WorkerServer


@pytest.mark.asyncio
async def test_synthesize_roundtrip() -> None:
    manager = VoxCPMModelManager(MockVoxCPMBackend())
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


@pytest.mark.asyncio
async def test_encode_reference_roundtrip() -> None:
    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    client = VoxCPMClient(channel)

    await client.start()
    await server.start()
    try:
        rng = np.random.default_rng(7)
        samples = rng.standard_normal(24000).astype(np.float32) * 0.1
        buf = io.BytesIO()
        sf.write(buf, samples, 24000, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        resp = await client.encode_reference(wav_bytes)
        assert len(resp.latents) == 4 * resp.feat_dim * resp.num_frames
        assert resp.encoder_sample_rate == 24000
    finally:
        await server.stop()
        await client.stop()


@pytest.mark.asyncio
async def test_synthesize_with_reference_latents() -> None:
    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    client = VoxCPMClient(channel)

    await client.start()
    await server.start()
    try:
        rng = np.random.default_rng(11)
        samples = rng.standard_normal(24000).astype(np.float32) * 0.1
        buf = io.BytesIO()
        sf.write(buf, samples, 24000, format="WAV", subtype="PCM_16")
        enc = await client.encode_reference(buf.getvalue())

        base = await client.synthesize_text("same text", sample_rate=24000)
        cloned = await client.synthesize_text(
            "same text", sample_rate=24000, ref_audio_latents=enc.latents
        )
        assert base.pcm_s16le != cloned.pcm_s16le
    finally:
        await server.stop()
        await client.stop()
