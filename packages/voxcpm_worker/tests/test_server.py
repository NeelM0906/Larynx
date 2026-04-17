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
        assert resp.encoder_sample_rate == 16000
    finally:
        await server.stop()
        await client.stop()


@pytest.mark.asyncio
async def test_synthesize_stream_yields_chunks_then_done() -> None:
    from larynx_shared.ipc import SynthesizeChunkFrame, SynthesizeDoneFrame

    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    client = VoxCPMClient(channel)
    await client.start()
    await server.start()
    try:
        pcm_chunks: list[bytes] = []
        chunk_indices: list[int] = []
        done: SynthesizeDoneFrame | None = None
        async with client.synthesize_text_stream(
            "streaming mock synthesis test", sample_rate=24000
        ) as frames:
            async for f in frames:
                if isinstance(f, SynthesizeChunkFrame):
                    pcm_chunks.append(f.pcm_s16le)
                    chunk_indices.append(f.chunk_index)
                    assert f.sample_rate == 24000
                elif isinstance(f, SynthesizeDoneFrame):
                    done = f
        # chunk_index is zero-based and monotonically increasing.
        assert chunk_indices == list(range(len(pcm_chunks)))
        # concatenated chunks are non-empty and match done.total_duration_ms.
        total_pcm_bytes = sum(len(c) for c in pcm_chunks)
        assert total_pcm_bytes > 0
        assert done is not None
        assert done.chunk_count == len(pcm_chunks)
        # Bytes per sample = 2 (int16). duration_ms ≈ samples/sr * 1000.
        expected_ms = int(1000 * (total_pcm_bytes / 2) / 24000)
        assert abs(expected_ms - done.total_duration_ms) <= 1
    finally:
        await server.stop()
        await client.stop()


@pytest.mark.asyncio
async def test_synthesize_stream_cancel_on_early_exit() -> None:
    import asyncio

    from larynx_shared.ipc import SynthesizeChunkFrame

    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    client = VoxCPMClient(channel)
    await client.start()
    await server.start()
    try:
        # A long text produces many chunks in the mock; breaking after the
        # first one should cause the rest to be cancelled.
        long_text = "word " * 200
        saw = 0
        async with client.synthesize_text_stream(long_text, sample_rate=24000) as frames:
            async for f in frames:
                if isinstance(f, SynthesizeChunkFrame):
                    saw += 1
                    break
        assert saw == 1
        # After exit, the server should finish up (cancel is a no-op for
        # the mock since chunks are fast, but the inflight set must clear).
        await asyncio.sleep(0.1)
        assert not server._inflight  # type: ignore[attr-defined]
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
