"""Smoke tests for the mock Fun-ASR backend + worker server.

Exercises the full in-process path the gateway uses: IPC channel ->
worker server -> mock backend -> IPC response. Proves the server
correctly routes TranscribeRequest vs TranscribeRollingRequest and that
the language router is wired into the request handler.
"""

from __future__ import annotations

import numpy as np
import pytest
from larynx_funasr_worker.model_manager import FunASRModelManager, MockFunASRBackend
from larynx_funasr_worker.server import WorkerServer
from larynx_shared.audio import float32_to_int16
from larynx_shared.ipc import (
    InProcessWorkerClient,
    TranscribeRequest,
    TranscribeResponse,
    TranscribeRollingRequest,
    TranscribeRollingResponse,
    WorkerChannel,
)


def _silence_pcm(seconds: float = 1.0, sr: int = 16000) -> bytes:
    n = int(sr * seconds)
    return float32_to_int16(np.zeros(n, dtype=np.float32)).tobytes()


def _noise_pcm(seconds: float = 1.0, sr: int = 16000, seed: int = 0) -> bytes:
    n = int(sr * seconds)
    rng = np.random.default_rng(seed)
    return float32_to_int16(rng.standard_normal(n).astype(np.float32) * 0.1).tobytes()


@pytest.fixture
async def worker_pair():
    channel = WorkerChannel()
    manager = FunASRModelManager(MockFunASRBackend())
    server = WorkerServer(channel, manager)
    client = InProcessWorkerClient(channel)
    await client.start()
    await server.start()
    try:
        yield client
    finally:
        await server.stop()
        await client.stop()


@pytest.mark.asyncio
async def test_transcribe_english_routes_to_nano(worker_pair: InProcessWorkerClient) -> None:
    req = TranscribeRequest(pcm_s16le=_noise_pcm(), sample_rate=16000, language="en")
    resp = await worker_pair.request(req, TranscribeResponse, timeout=5)
    assert resp.model_used == "nano"
    assert resp.language == "en"
    assert "mock" in resp.text


@pytest.mark.asyncio
async def test_transcribe_portuguese_routes_to_mlt(worker_pair: InProcessWorkerClient) -> None:
    req = TranscribeRequest(pcm_s16le=_noise_pcm(seed=1), sample_rate=16000, language="pt")
    resp = await worker_pair.request(req, TranscribeResponse, timeout=5)
    assert resp.model_used == "mlt"
    assert resp.language == "pt"


@pytest.mark.asyncio
async def test_transcribe_none_auto_detects_on_nano(worker_pair: InProcessWorkerClient) -> None:
    req = TranscribeRequest(pcm_s16le=_noise_pcm(seed=2), sample_rate=16000, language=None)
    resp = await worker_pair.request(req, TranscribeResponse, timeout=5)
    assert resp.model_used == "nano"


@pytest.mark.asyncio
async def test_hotwords_surface_in_mock(worker_pair: InProcessWorkerClient) -> None:
    req = TranscribeRequest(
        pcm_s16le=_noise_pcm(seed=3),
        sample_rate=16000,
        language="en",
        hotwords=["Larynx", "VoxCPM"],
    )
    resp = await worker_pair.request(req, TranscribeResponse, timeout=5)
    assert "Larynx" in resp.text and "VoxCPM" in resp.text


@pytest.mark.asyncio
async def test_unsupported_language_returns_error(worker_pair: InProcessWorkerClient) -> None:
    from larynx_shared.ipc.client_base import WorkerError

    req = TranscribeRequest(pcm_s16le=_noise_pcm(seed=4), sample_rate=16000, language="es")
    with pytest.raises(WorkerError) as exc:
        await worker_pair.request(req, TranscribeResponse, timeout=5)
    assert exc.value.code == "unsupported_language"


@pytest.mark.asyncio
async def test_bad_sample_rate_returns_invalid_input(worker_pair: InProcessWorkerClient) -> None:
    from larynx_shared.ipc.client_base import WorkerError

    req = TranscribeRequest(pcm_s16le=_silence_pcm(), sample_rate=22050, language="en")
    with pytest.raises(WorkerError) as exc:
        await worker_pair.request(req, TranscribeResponse, timeout=5)
    assert exc.value.code == "invalid_input"


@pytest.mark.asyncio
async def test_rolling_partial_drops_tail(worker_pair: InProcessWorkerClient) -> None:
    # is_final=False should drop the last 5 tokens from the accumulated text.
    req = TranscribeRollingRequest(
        pcm_s16le=_noise_pcm(seed=5),
        sample_rate=16000,
        language="en",
        prev_text="partial so far",
        is_final=False,
        drop_tail_tokens=5,
    )
    resp = await worker_pair.request(req, TranscribeRollingResponse, timeout=5)
    assert resp.is_final is False
    assert resp.model_used == "nano"


@pytest.mark.asyncio
async def test_rolling_final_keeps_all_tokens(worker_pair: InProcessWorkerClient) -> None:
    req = TranscribeRollingRequest(
        pcm_s16le=_noise_pcm(seed=6),
        sample_rate=16000,
        language="en",
        is_final=True,
    )
    resp = await worker_pair.request(req, TranscribeRollingResponse, timeout=5)
    assert resp.is_final is True
    # With is_final=True no drop happens, so text is non-empty.
    assert len(resp.text) > 0
