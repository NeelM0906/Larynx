"""Unit test for the streaming STT session service.

Drives the service with in-process mock workers + a generator that yields
synthetic PCM frames, then asserts the event sequence matches the VAD +
rolling-decode spec.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from larynx_shared.ipc import WorkerChannel

from larynx_gateway.services.stt_stream_service import STTStreamConfig, STTStreamSession
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient


def _tone(ms: int, sr: int = 16000, freq: float = 220.0, amp: float = 0.5) -> bytes:
    n = sr * ms // 1000
    t = np.arange(n, dtype=np.float32) / sr
    s = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return (s * 32767).astype(np.int16).tobytes()


def _silence(ms: int, sr: int = 16000) -> bytes:
    return (np.zeros(sr * ms // 1000, dtype=np.int16)).tobytes()


@pytest.mark.asyncio
async def test_service_emits_full_event_sequence() -> None:
    from larynx_funasr_worker.model_manager import FunASRModelManager, MockFunASRBackend
    from larynx_funasr_worker.server import WorkerServer as FunASRWorkerServer
    from larynx_vad_punc_worker.model_manager import (
        MockVadPuncBackend,
        VadPuncModelManager,
    )
    from larynx_vad_punc_worker.server import WorkerServer as VadPuncWorkerServer

    funasr_ch = WorkerChannel()
    funasr_mgr = FunASRModelManager(MockFunASRBackend())
    funasr_srv = FunASRWorkerServer(funasr_ch, funasr_mgr)
    funasr_cli = FunASRClient(funasr_ch)

    vad_ch = WorkerChannel()
    vad_mgr = VadPuncModelManager(MockVadPuncBackend())
    vad_srv = VadPuncWorkerServer(vad_ch, vad_mgr)
    vad_cli = VadPuncClient(vad_ch)

    await funasr_cli.start()
    await vad_cli.start()
    await funasr_srv.start()
    await vad_srv.start()

    try:
        cfg = STTStreamConfig(
            sample_rate=16000,
            chunk_interval_ms=200,  # tighter so a partial fires inside the tone
            speech_end_silence_ms=200,
        )
        session = STTStreamSession(funasr=funasr_cli, vad=vad_cli, cfg=cfg)

        async def source():
            # 1.5s tone (enough for one partial at chunk_interval=200ms)…
            for _ in range(15):
                yield _tone(100)
                # Give partials_loop a chance to wake up by yielding.
                await asyncio.sleep(0.02)
            # …followed by 600ms silence to trigger speech_end (200ms window).
            for _ in range(6):
                yield _silence(100)
                await asyncio.sleep(0.02)

        # Collect events concurrently with run()
        events: list[dict] = []

        async def drain() -> None:
            async for ev in session.events():
                events.append(ev)

        drain_task = asyncio.create_task(drain())
        await session.run(source())
        await drain_task

        kinds = [e["type"] for e in events]
        assert "speech_start" in kinds, f"kinds={kinds}"
        assert "speech_end" in kinds, f"kinds={kinds}"
        assert "final" in kinds, f"kinds={kinds}"
        # A real-model run would always see partials; with the mock VAD and
        # tight 200ms chunk interval there should be at least one.
        assert any(e["type"] == "partial" for e in events), f"kinds={kinds}"
    finally:
        await funasr_srv.stop()
        await vad_srv.stop()
        await funasr_cli.stop()
        await vad_cli.stop()


async def _spin_up_workers() -> tuple[FunASRClient, VadPuncClient, list]:
    """Boot fresh mock workers; return (funasr_cli, vad_cli, [srv, srv, cli, cli]).

    The caller is responsible for stopping the returned servers + clients.
    Factored out so each ordinal test gets an isolated VAD session state.
    """
    from larynx_funasr_worker.model_manager import FunASRModelManager, MockFunASRBackend
    from larynx_funasr_worker.server import WorkerServer as FunASRWorkerServer
    from larynx_vad_punc_worker.model_manager import (
        MockVadPuncBackend,
        VadPuncModelManager,
    )
    from larynx_vad_punc_worker.server import WorkerServer as VadPuncWorkerServer

    funasr_ch = WorkerChannel()
    funasr_srv = FunASRWorkerServer(funasr_ch, FunASRModelManager(MockFunASRBackend()))
    funasr_cli = FunASRClient(funasr_ch)

    vad_ch = WorkerChannel()
    vad_srv = VadPuncWorkerServer(vad_ch, VadPuncModelManager(MockVadPuncBackend()))
    vad_cli = VadPuncClient(vad_ch)

    await funasr_cli.start()
    await vad_cli.start()
    await funasr_srv.start()
    await vad_srv.start()
    return funasr_cli, vad_cli, [funasr_srv, vad_srv, funasr_cli, vad_cli]


@pytest.mark.asyncio
async def test_utterance_ordinal_starts_at_one_and_is_stamped_on_all_events() -> None:
    """First utterance carries ordinal=1 on speech_start, speech_end, partial, final."""
    funasr, vad, to_close = await _spin_up_workers()
    try:
        cfg = STTStreamConfig(
            sample_rate=16000,
            chunk_interval_ms=200,
            speech_end_silence_ms=200,
        )
        session = STTStreamSession(funasr=funasr, vad=vad, cfg=cfg)

        async def source():
            for _ in range(15):  # 1.5s tone
                yield _tone(100)
                await asyncio.sleep(0.02)
            for _ in range(6):  # 600ms silence > 200ms speech_end window
                yield _silence(100)
                await asyncio.sleep(0.02)

        events: list[dict] = []

        async def drain() -> None:
            async for ev in session.events():
                events.append(ev)

        drain_task = asyncio.create_task(drain())
        await session.run(source())
        await drain_task

        stamped = [e for e in events if e["type"] in {"speech_start", "speech_end", "partial", "final"}]
        assert stamped, "expected at least one stamped event"
        for ev in stamped:
            assert "utterance_ordinal" in ev, f"missing ordinal on {ev}"
            assert ev["utterance_ordinal"] == 1, f"expected ordinal=1, got {ev}"
    finally:
        for srv in to_close[:2]:
            await srv.stop()
        for cli in to_close[2:]:
            await cli.stop()


@pytest.mark.asyncio
async def test_utterance_ordinal_increments_across_utterances() -> None:
    """Two tone→silence cycles → events stamped with ordinals 1 then 2."""
    funasr, vad, to_close = await _spin_up_workers()
    try:
        cfg = STTStreamConfig(
            sample_rate=16000,
            chunk_interval_ms=200,
            speech_end_silence_ms=200,
        )
        session = STTStreamSession(funasr=funasr, vad=vad, cfg=cfg)

        async def source():
            # Utterance 1: 1s tone → 500ms silence (triggers end).
            for _ in range(10):
                yield _tone(100)
                await asyncio.sleep(0.02)
            for _ in range(5):
                yield _silence(100)
                await asyncio.sleep(0.02)
            # Utterance 2: 1s tone → 500ms silence.
            for _ in range(10):
                yield _tone(100)
                await asyncio.sleep(0.02)
            for _ in range(5):
                yield _silence(100)
                await asyncio.sleep(0.02)

        events: list[dict] = []

        async def drain() -> None:
            async for ev in session.events():
                events.append(ev)

        drain_task = asyncio.create_task(drain())
        await session.run(source())
        await drain_task

        starts = [e for e in events if e["type"] == "speech_start"]
        ends = [e for e in events if e["type"] == "speech_end"]
        finals = [e for e in events if e["type"] == "final"]
        assert len(starts) == 2, f"expected 2 speech_start, got {len(starts)}: {events}"
        assert [s["utterance_ordinal"] for s in starts] == [1, 2]
        assert len(ends) == 2
        assert [e["utterance_ordinal"] for e in ends] == [1, 2]
        # Every final belongs to a finished utterance — match ordinals in order.
        assert [f["utterance_ordinal"] for f in finals] == [1, 2], (
            f"final ordinals: {[f['utterance_ordinal'] for f in finals]}"
        )
    finally:
        for srv in to_close[:2]:
            await srv.stop()
        for cli in to_close[2:]:
            await cli.stop()


@pytest.mark.asyncio
async def test_partial_ordinal_matches_enclosing_utterance() -> None:
    """Partial events during an utterance carry the same ordinal as its speech_start."""
    funasr, vad, to_close = await _spin_up_workers()
    try:
        cfg = STTStreamConfig(
            sample_rate=16000,
            chunk_interval_ms=150,  # tight enough that partials fire during each tone
            speech_end_silence_ms=200,
        )
        session = STTStreamSession(funasr=funasr, vad=vad, cfg=cfg)

        async def source():
            for _ in range(12):  # 1.2s tone → partials should fire
                yield _tone(100)
                await asyncio.sleep(0.03)
            for _ in range(6):  # silence
                yield _silence(100)
                await asyncio.sleep(0.02)
            for _ in range(12):  # second utterance
                yield _tone(100)
                await asyncio.sleep(0.03)
            for _ in range(6):
                yield _silence(100)
                await asyncio.sleep(0.02)

        events: list[dict] = []

        async def drain() -> None:
            async for ev in session.events():
                events.append(ev)

        drain_task = asyncio.create_task(drain())
        await session.run(source())
        await drain_task

        partials = [e for e in events if e["type"] == "partial"]
        assert partials, "expected at least one partial in this run"
        # Reconstruct which utterance each partial belongs to by walking
        # the event stream and tracking the current speech_start ordinal.
        current_ordinal: int | None = None
        for ev in events:
            if ev["type"] == "speech_start":
                current_ordinal = ev["utterance_ordinal"]
            elif ev["type"] == "partial":
                assert current_ordinal is not None
                assert ev["utterance_ordinal"] == current_ordinal, (
                    f"partial ordinal {ev['utterance_ordinal']} does not match "
                    f"current utterance {current_ordinal}: {ev}"
                )
    finally:
        for srv in to_close[:2]:
            await srv.stop()
        for cli in to_close[2:]:
            await cli.stop()
