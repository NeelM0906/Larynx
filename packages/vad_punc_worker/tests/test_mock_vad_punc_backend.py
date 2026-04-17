"""Smoke tests for the mock VAD + Punctuation worker."""

from __future__ import annotations

import numpy as np
import pytest
from larynx_shared.audio import float32_to_int16
from larynx_shared.ipc import (
    DetectSegmentsRequest,
    DetectSegmentsResponse,
    InProcessWorkerClient,
    PunctuateRequest,
    PunctuateResponse,
    WorkerChannel,
)
from larynx_vad_punc_worker.model_manager import MockVadPuncBackend, VadPuncModelManager
from larynx_vad_punc_worker.server import WorkerServer


def _sine_pcm(seconds: float = 1.0, sr: int = 16000, freq: float = 200.0) -> bytes:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / sr
    wave = 0.3 * np.sin(2 * np.pi * freq * t, dtype=np.float32)
    return float32_to_int16(wave).tobytes()


def _silence_pcm(seconds: float = 1.0, sr: int = 16000) -> bytes:
    n = int(sr * seconds)
    return float32_to_int16(np.zeros(n, dtype=np.float32)).tobytes()


@pytest.fixture
async def worker_pair():
    channel = WorkerChannel()
    manager = VadPuncModelManager(MockVadPuncBackend())
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
async def test_segment_detects_voiced_region(worker_pair: InProcessWorkerClient) -> None:
    req = DetectSegmentsRequest(pcm_s16le=_sine_pcm(seconds=0.5), sample_rate=16000)
    resp = await worker_pair.request(req, DetectSegmentsResponse, timeout=5)
    assert len(resp.segments) == 1
    seg = resp.segments[0]
    assert seg.is_speech is True
    assert seg.end_ms > seg.start_ms


@pytest.mark.asyncio
async def test_segment_silence_returns_empty(worker_pair: InProcessWorkerClient) -> None:
    req = DetectSegmentsRequest(pcm_s16le=_silence_pcm(), sample_rate=16000)
    resp = await worker_pair.request(req, DetectSegmentsResponse, timeout=5)
    assert resp.segments == []


@pytest.mark.asyncio
async def test_punctuate_english_capitalises_and_adds_period(
    worker_pair: InProcessWorkerClient,
) -> None:
    req = PunctuateRequest(text="hello world", language="en")
    resp = await worker_pair.request(req, PunctuateResponse, timeout=5)
    assert resp.applied is True
    assert resp.text == "Hello world."


@pytest.mark.asyncio
async def test_punctuate_skips_mlt_languages(worker_pair: InProcessWorkerClient) -> None:
    """Fun-ASR MLT already emits inline punctuation via itn=True, so the
    mock (and the real ct-punc) short-circuit for non-zh/en languages."""
    req = PunctuateRequest(text="olá mundo como vai", language="pt")
    resp = await worker_pair.request(req, PunctuateResponse, timeout=5)
    assert resp.applied is False
    assert resp.text == "olá mundo como vai"


@pytest.mark.asyncio
async def test_punctuate_empty_text_no_op(worker_pair: InProcessWorkerClient) -> None:
    req = PunctuateRequest(text="", language="en")
    resp = await worker_pair.request(req, PunctuateResponse, timeout=5)
    assert resp.applied is False
    assert resp.text == ""
