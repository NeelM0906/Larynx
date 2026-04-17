"""Unit + IPC tests for the streaming VAD session."""

from __future__ import annotations

import numpy as np
import pytest
from larynx_shared.ipc import WorkerChannel

from larynx_gateway.workers_client.vad_punc_client import VadPuncClient
from larynx_vad_punc_worker.model_manager import MockVadPuncBackend, VadPuncModelManager
from larynx_vad_punc_worker.server import WorkerServer
from larynx_vad_punc_worker.streaming_vad import MockStreamingVad


def _pcm_silence(ms: int, sr: int = 16000) -> bytes:
    return (np.zeros(sr * ms // 1000, dtype=np.int16)).tobytes()


def _pcm_tone(ms: int, sr: int = 16000, freq: float = 220.0, amp: float = 0.5) -> bytes:
    t = np.arange(sr * ms // 1000, dtype=np.float32) / sr
    samples = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return (samples * 32767).astype(np.int16).tobytes()


@pytest.mark.asyncio
async def test_mock_vad_start_then_end_events() -> None:
    vad = MockStreamingVad()
    await vad.open("s1", sample_rate=16000, speech_end_silence_ms=200)

    # 500ms of tone → speech_start somewhere in the first ~40-60ms.
    events1, state1, _ = await vad.feed("s1", _pcm_tone(500))
    assert state1 == "speaking"
    assert any(e.event == "speech_start" for e in events1)

    # Then 500ms of silence (> 200ms window) → speech_end fires.
    events2, state2, _ = await vad.feed("s1", _pcm_silence(500))
    assert state2 == "silent"
    assert any(e.event == "speech_end" for e in events2)

    await vad.close("s1")


@pytest.mark.asyncio
async def test_mock_vad_sessions_are_isolated() -> None:
    vad = MockStreamingVad()
    await vad.open("a", sample_rate=16000, speech_end_silence_ms=200)
    await vad.open("b", sample_rate=16000, speech_end_silence_ms=200)

    # Only session "a" sees voice; "b" stays silent even though they share
    # the same backend.
    _, state_a, _ = await vad.feed("a", _pcm_tone(500))
    _, state_b, _ = await vad.feed("b", _pcm_silence(500))
    assert state_a == "speaking"
    assert state_b == "silent"


@pytest.mark.asyncio
async def test_vad_stream_roundtrip_via_client() -> None:
    manager = VadPuncModelManager(MockVadPuncBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    client = VadPuncClient(channel)

    await client.start()
    await server.start()
    try:
        resp = await client.vad_stream_open(session_id="xyz", sample_rate=16000)
        assert resp.session_id == "xyz"

        # Feed 500ms of speech, expect speech_start.
        feed_resp = await client.vad_stream_feed(
            session_id="xyz", pcm_s16le=_pcm_tone(500)
        )
        assert feed_resp.vad_state == "speaking"
        assert any(e.event == "speech_start" for e in feed_resp.events)

        # Feed 500ms of silence, expect speech_end.
        feed_resp = await client.vad_stream_feed(
            session_id="xyz", pcm_s16le=_pcm_silence(500)
        )
        assert feed_resp.vad_state == "silent"
        assert any(e.event == "speech_end" for e in feed_resp.events)

        close = await client.vad_stream_close(session_id="xyz")
        assert close.session_id == "xyz"
    finally:
        await server.stop()
        await client.stop()


@pytest.mark.asyncio
async def test_vad_stream_feed_on_unknown_session_errors() -> None:
    manager = VadPuncModelManager(MockVadPuncBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    client = VadPuncClient(channel)
    await client.start()
    await server.start()
    try:
        from larynx_shared.ipc import WorkerError

        with pytest.raises(WorkerError) as exc:
            await client.vad_stream_feed(session_id="nope", pcm_s16le=_pcm_silence(20))
        assert exc.value.code == "unknown_session"
    finally:
        await server.stop()
        await client.stop()
