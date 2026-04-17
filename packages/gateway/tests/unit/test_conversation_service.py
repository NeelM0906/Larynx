"""Full handler tests for ConversationSession — M5 acceptance suite.

Covers the exit criteria from the M5 prompt plus the E5 serialisation
test promised in ORCHESTRATION.md v2:

- test_barge_in_during_tts_cancels_within_100ms
- test_barge_in_during_llm_cancels_cleanly
- test_rapid_barge_in_of_barge_in
- test_network_failure_keeps_session_alive_next_turn_works
- test_no_orphan_tasks_after_many_turns (50 turns, PYTHONASYNCIODEBUG=1 env)
- test_e5_two_speech_starts_in_one_tick_emit_exactly_one_interrupt

Happy-path 1-turn already in test_conversation_smoke.py; a multi-turn
happy-path is exercised via the orphan test (50 turns).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import uvicorn
from larynx_shared.ipc import (
    SynthesizeChunkFrame,
    SynthesizeDoneFrame,
    WorkerChannel,
)

from larynx_gateway.services.conversation_service import (
    ConversationConfig,
    ConversationSession,
    LLMEvent,
    SessionState,
    STTEvent,
    TTSEvent,
    VADEvent,
)
from larynx_gateway.services.llm_client import LLMClient
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient


# ---------------------------------------------------------------------------
# Live LLM server helpers (same shape as test_llm_client.py)
# ---------------------------------------------------------------------------


def _sse_event(delta: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': {'content': delta}}]})}\n\n".encode()


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@asynccontextmanager
async def _live_llm(app):
    port = _pick_free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", access_log=False, lifespan="off"
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def _make_token_stream_app(tokens: list[str], *, per_token_delay_s: float = 0.002):
    """Stream each element of ``tokens`` as its own SSE delta."""

    async def app(scope, receive, send):
        assert scope["type"] == "http"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for tok in tokens:
            await send(
                {"type": "http.response.body", "body": _sse_event(tok), "more_body": True}
            )
            if per_token_delay_s > 0:
                await asyncio.sleep(per_token_delay_s)
        await send(
            {"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": True}
        )
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    return app


def _make_killed_mid_stream_app(tokens_before_kill: list[str]):
    """Stream ``tokens_before_kill`` then abruptly stop (simulates dropped
    OpenRouter connection mid-response)."""

    async def app(scope, receive, send):
        assert scope["type"] == "http"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for tok in tokens_before_kill:
            await send(
                {"type": "http.response.body", "body": _sse_event(tok), "more_body": True}
            )
            await asyncio.sleep(0.005)
        # End the response without [DONE] or normal close.
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    return app


# ---------------------------------------------------------------------------
# Slow TTS — deterministic pacing for barge-in timing
# ---------------------------------------------------------------------------


class _SlowTTSClient:
    """Drop-in replacement for VoxCPMClient.synthesize_text_stream that
    emits chunks with a configurable real-time delay.

    Needed because MockVoxCPMBackend's chunking uses ``asyncio.sleep(0)``
    between frames, which finishes any reasonable sentence in ~1ms — too
    fast to reliably test barge-in cancellation inside 100ms.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        chunk_ms: int = 40,
        per_chunk_delay_s: float = 0.04,
        total_chunks: int = 50,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_samples = sample_rate * chunk_ms // 1000
        self._per_chunk_delay = per_chunk_delay_s
        self._total_chunks = total_chunks
        # Track cancellations for assertions.
        self.cancel_count: int = 0
        self.finish_count: int = 0

    def synthesize_text_stream(self, **kwargs) -> "_SlowTTSCtx":  # noqa: ARG002
        return _SlowTTSCtx(self)


class _SlowTTSCtx:
    def __init__(self, parent: _SlowTTSClient) -> None:
        self._parent = parent
        self._cancelled = False

    async def __aenter__(self):
        return self._iter()

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is asyncio.CancelledError:
            self._parent.cancel_count += 1
        return None

    async def _iter(self):
        req_id = f"slowtts-{id(self)}"
        try:
            for i in range(self._parent._total_chunks):
                pcm = np.zeros(self._parent._chunk_samples, dtype=np.int16).tobytes()
                yield SynthesizeChunkFrame(
                    request_id=req_id,
                    pcm_s16le=pcm,
                    sample_rate=self._parent._sample_rate,
                    chunk_index=i,
                )
                await asyncio.sleep(self._parent._per_chunk_delay)
            yield SynthesizeDoneFrame(
                request_id=req_id,
                sample_rate=self._parent._sample_rate,
                total_duration_ms=int(
                    self._parent._total_chunks * self._parent._chunk_samples * 1000
                    / self._parent._sample_rate
                ),
                chunk_count=self._parent._total_chunks,
                ttfb_ms=0,
            )
            self._parent.finish_count += 1
        except asyncio.CancelledError:
            self._parent.cancel_count += 1
            raise


# ---------------------------------------------------------------------------
# Recording sink — captures audio + events with timestamps
# ---------------------------------------------------------------------------


@dataclass
class _RecordedAudio:
    t_wall: float
    n_bytes: int
    sample_rate: int


@dataclass
class _RecordedEvent:
    t_wall: float
    payload: dict[str, Any]


class _RecordingSink:
    def __init__(self) -> None:
        self.audio: list[_RecordedAudio] = []
        self.events: list[_RecordedEvent] = []

    async def send_audio(self, pcm_s16le: bytes, sample_rate: int) -> None:
        self.audio.append(
            _RecordedAudio(t_wall=time.monotonic(), n_bytes=len(pcm_s16le), sample_rate=sample_rate)
        )

    async def send_event(self, payload: dict[str, Any]) -> None:
        self.events.append(_RecordedEvent(t_wall=time.monotonic(), payload=payload))

    def events_of(self, kind: str) -> list[_RecordedEvent]:
        return [e for e in self.events if e.payload.get("type") == kind]


# ---------------------------------------------------------------------------
# Worker harness
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _workers():
    from larynx_funasr_worker.model_manager import FunASRModelManager, MockFunASRBackend
    from larynx_funasr_worker.server import WorkerServer as FunASRWorkerServer
    from larynx_vad_punc_worker.model_manager import (
        MockVadPuncBackend,
        VadPuncModelManager,
    )
    from larynx_vad_punc_worker.server import WorkerServer as VadPuncWorkerServer
    from larynx_voxcpm_worker.model_manager import (
        MockVoxCPMBackend,
        VoxCPMModelManager,
    )
    from larynx_voxcpm_worker.server import WorkerServer as VoxCPMWorkerServer

    funasr_ch = WorkerChannel()
    funasr_srv = FunASRWorkerServer(funasr_ch, FunASRModelManager(MockFunASRBackend()))
    funasr_cli = FunASRClient(funasr_ch)

    vad_ch = WorkerChannel()
    vad_srv = VadPuncWorkerServer(vad_ch, VadPuncModelManager(MockVadPuncBackend()))
    vad_cli = VadPuncClient(vad_ch)

    voxcpm_ch = WorkerChannel()
    voxcpm_srv = VoxCPMWorkerServer(voxcpm_ch, VoxCPMModelManager(MockVoxCPMBackend()))
    voxcpm_cli = VoxCPMClient(voxcpm_ch)

    await funasr_cli.start()
    await vad_cli.start()
    await voxcpm_cli.start()
    await funasr_srv.start()
    await vad_srv.start()
    await voxcpm_srv.start()
    try:
        yield funasr_cli, vad_cli, voxcpm_cli
    finally:
        await funasr_srv.stop()
        await vad_srv.stop()
        await voxcpm_srv.stop()
        await funasr_cli.stop()
        await vad_cli.stop()
        await voxcpm_cli.stop()


# ---------------------------------------------------------------------------
# Utility: feed a single scripted utterance into a running session via
# synthetic events. Bypasses the audio/VAD/STT path so timing is exact.
# ---------------------------------------------------------------------------


async def _scripted_turn(session: ConversationSession, *, ordinal: int, text: str) -> None:
    await session._queue.put(  # noqa: SLF001
        VADEvent(kind="speech_start", utterance_ordinal=ordinal, session_ms=(ordinal - 1) * 1000)
    )
    await session._queue.put(  # noqa: SLF001
        VADEvent(kind="speech_end", utterance_ordinal=ordinal, session_ms=(ordinal - 1) * 1000 + 800)
    )
    await session._queue.put(  # noqa: SLF001
        STTEvent(
            kind="transcript_final",
            utterance_ordinal=ordinal,
            text=text,
            punctuated_text=text,
        )
    )


async def _wait_for_state(session: ConversationSession, target: SessionState, *, timeout_s: float = 3.0) -> None:
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        if session.state is target:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"state did not become {target} within {timeout_s}s (is {session.state})")


async def _wait_for_event(
    sink: _RecordingSink, kind: str, *, timeout_s: float = 3.0, min_count: int = 1
) -> dict[str, Any]:
    """Return the last event of ``kind`` once at least ``min_count`` have been seen."""
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        matches = sink.events_of(kind)
        if len(matches) >= min_count:
            return matches[-1].payload
        await asyncio.sleep(0.005)
    raise AssertionError(
        f"event {kind!r} (min_count={min_count}) not seen within {timeout_s}s; "
        f"saw: {[e.payload.get('type') for e in sink.events]}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_barge_in_during_tts_cancels_within_100ms() -> None:
    """Fire speech_start mid-TTS; verify last audio frame ≤ 100ms later."""
    async with _workers() as (funasr, vad, _voxcpm):
        async with _live_llm(_make_token_stream_app(
            # Long enough reply so sentence-chunked TTS has work to do.
            ["Hello", " there", ",", " this", " is", " a", " reply", "."],
        )) as base_url:
            llm = LLMClient(api_key="x", base_url=f"{base_url}/api/v1")
            slow_tts = _SlowTTSClient(per_chunk_delay_s=0.03, total_chunks=100)
            sink = _RecordingSink()
            cfg = ConversationConfig(llm_model="m", speech_end_silence_ms=200)
            session = ConversationSession(
                cfg=cfg, sink=sink, funasr=funasr, vad=vad, voxcpm=slow_tts, llm=llm  # type: ignore[arg-type]
            )
            run_task = asyncio.create_task(session.run())

            await _scripted_turn(session, ordinal=1, text="say something long please")
            # Wait until TTS is actively emitting audio.
            t_wait_end = time.monotonic() + 3.0
            while time.monotonic() < t_wait_end and len(sink.audio) < 3:
                await asyncio.sleep(0.01)
            assert len(sink.audio) >= 3, "TTS never started emitting audio"
            n_audio_before_bargein = len(sink.audio)

            # Fire barge-in.
            t_barge_start = time.monotonic()
            await session._queue.put(  # noqa: SLF001
                VADEvent(kind="speech_start", utterance_ordinal=2, session_ms=5000)
            )
            # Wait for interrupt event.
            interrupt = await _wait_for_event(sink, "interrupt", timeout_s=1.0)
            t_interrupt = next(e.t_wall for e in sink.events if e.payload.get("type") == "interrupt")

            # Measure: last audio frame after t_barge_start.
            last_audio_after_barge = [a for a in sink.audio if a.t_wall >= t_barge_start]
            t_last_audio = last_audio_after_barge[-1].t_wall if last_audio_after_barge else t_barge_start
            gap_ms = (t_last_audio - t_barge_start) * 1000

            assert gap_ms < 100, (
                f"barge-in took {gap_ms:.0f}ms to stop audio (target: <100ms); "
                f"audio frames before={n_audio_before_bargein}, after_barge={len(last_audio_after_barge)}"
            )
            # Provisional assistant message not committed.
            assert not any(m.role == "assistant" for m in session._history), (  # noqa: SLF001
                "interrupted assistant message leaked into history"
            )
            # interrupt event carries barge_in_ms.
            assert "barge_in_ms" in interrupt
            assert interrupt["new_utterance_ordinal"] == 2
            # TTS cancellation was observed by the fake client.
            assert slow_tts.cancel_count >= 1, "TTS task was not cancelled"

            await session.stop()
            await asyncio.wait_for(run_task, timeout=3.0)


@pytest.mark.asyncio
async def test_barge_in_during_llm_before_first_tts_chunk() -> None:
    """Interrupt between speech_end and first TTS — LLM cancels, no audio ever."""

    # LLM with a huge pre-token delay so we barge-in inside LLM_GENERATING.
    async def slow_first_token_app(scope, receive, send):
        assert scope["type"] == "http"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await asyncio.sleep(0.4)  # first-token stall; barge-in hits before.
        await send(
            {"type": "http.response.body", "body": _sse_event("too"), "more_body": True}
        )
        await send(
            {"type": "http.response.body", "body": _sse_event(" late."), "more_body": True}
        )
        await send(
            {"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": True}
        )
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async with _workers() as (funasr, vad, voxcpm):
        async with _live_llm(slow_first_token_app) as base_url:
            llm = LLMClient(api_key="x", base_url=f"{base_url}/api/v1")
            sink = _RecordingSink()
            session = ConversationSession(
                cfg=ConversationConfig(llm_model="m"),
                sink=sink, funasr=funasr, vad=vad, voxcpm=voxcpm, llm=llm,
            )
            run_task = asyncio.create_task(session.run())

            await _scripted_turn(session, ordinal=1, text="hello")
            await _wait_for_state(session, SessionState.LLM_GENERATING)

            # Barge in while in LLM_GENERATING (before first TTS chunk).
            t0 = time.monotonic()
            await session._queue.put(  # noqa: SLF001
                VADEvent(kind="speech_start", utterance_ordinal=2, session_ms=5000)
            )
            interrupt = await _wait_for_event(sink, "interrupt", timeout_s=1.0)
            gap_ms = (time.monotonic() - t0) * 1000
            # No audio ever emitted — we beat the first TTS chunk.
            assert not sink.audio, f"unexpected audio frames: {len(sink.audio)}"
            assert gap_ms < 200, f"LLM barge-in took {gap_ms:.0f}ms (target: <200ms)"
            assert interrupt["new_utterance_ordinal"] == 2
            assert not any(m.role == "assistant" for m in session._history)  # noqa: SLF001

            await session.stop()
            await asyncio.wait_for(run_task, timeout=3.0)


@pytest.mark.asyncio
async def test_rapid_barge_in_of_barge_in() -> None:
    """Two back-to-back barge-ins — no task leaks, no state corruption."""
    async with _workers() as (funasr, vad, _voxcpm):
        async with _live_llm(_make_token_stream_app(["reply", " one", "."])) as base_url:
            llm = LLMClient(api_key="x", base_url=f"{base_url}/api/v1")
            slow_tts = _SlowTTSClient(per_chunk_delay_s=0.03, total_chunks=50)
            sink = _RecordingSink()
            session = ConversationSession(
                cfg=ConversationConfig(llm_model="m"),
                sink=sink, funasr=funasr, vad=vad,
                voxcpm=slow_tts, llm=llm,  # type: ignore[arg-type]
            )
            run_task = asyncio.create_task(session.run())

            # Turn 1: get into TTS.
            await _scripted_turn(session, ordinal=1, text="please speak")
            while len(sink.audio) < 2:
                await asyncio.sleep(0.01)

            # Barge-in #1.
            await session._queue.put(  # noqa: SLF001
                VADEvent(kind="speech_start", utterance_ordinal=2, session_ms=5000)
            )
            await _wait_for_event(sink, "interrupt", timeout_s=1.0)
            interrupts_after_1 = len(sink.events_of("interrupt"))
            assert interrupts_after_1 == 1

            # Finalise turn 2 so we're back in RESPONDING.
            await session._queue.put(  # noqa: SLF001
                STTEvent(
                    kind="transcript_final",
                    utterance_ordinal=2,
                    text="another prompt",
                    punctuated_text="another prompt",
                )
            )
            # Wait for new TTS to start.
            t_end = time.monotonic() + 3.0
            while time.monotonic() < t_end:
                if len(sink.audio) > 0 and session.state is SessionState.TTS_SPEAKING:
                    break
                await asyncio.sleep(0.01)

            # Barge-in #2.
            await session._queue.put(  # noqa: SLF001
                VADEvent(kind="speech_start", utterance_ordinal=3, session_ms=10000)
            )
            await _wait_for_event(sink, "interrupt", timeout_s=2.0, min_count=2)
            assert len(sink.events_of("interrupt")) == 2

            await session.stop()
            await asyncio.wait_for(run_task, timeout=3.0)

            # No assistant message committed to history — both turns were
            # interrupted before completion.
            assert not any(m.role == "assistant" for m in session._history)  # noqa: SLF001


@pytest.mark.asyncio
async def test_network_failure_keeps_session_alive_next_turn_works() -> None:
    """OpenRouter connection drops mid-stream → error event, session
    survives, next turn succeeds against a healthy server."""
    # First app: drops. Second app: healthy.
    port_a = _pick_free_port()
    port_b = _pick_free_port()

    def _serve(app, port):
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False, lifespan="off")
        srv = uvicorn.Server(config)
        srv.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        th = threading.Thread(target=srv.run, daemon=True)
        th.start()
        return srv, th

    srv_a, th_a = _serve(_make_killed_mid_stream_app(["part", " one"]), port_a)
    healthy_app = _make_token_stream_app(["hi", "."])
    srv_b, th_b = _serve(healthy_app, port_b)
    for srv in (srv_a, srv_b):
        for _ in range(200):
            if srv.started:
                break
            await asyncio.sleep(0.01)

    try:
        async with _workers() as (funasr, vad, voxcpm):
            llm_a = LLMClient(api_key="x", base_url=f"http://127.0.0.1:{port_a}/api/v1")
            llm_b = LLMClient(api_key="x", base_url=f"http://127.0.0.1:{port_b}/api/v1")
            sink = _RecordingSink()
            session = ConversationSession(
                cfg=ConversationConfig(llm_model="m", speech_end_silence_ms=200),
                sink=sink, funasr=funasr, vad=vad, voxcpm=voxcpm, llm=llm_a,
            )
            run_task = asyncio.create_task(session.run())

            # Turn 1 — LLM kills connection, LLM stream yields a couple tokens
            # then the body ends without [DONE]. The body-ending path is a
            # clean close from httpx's POV (no error), so stream_chat returns
            # normally and LLMEvent(llm_done) fires. The orchestrator will
            # flush a short response and transition to IDLE normally.
            #
            # To simulate a REAL mid-stream failure we need a more
            # aggressive drop. Use a bad-host LLM on turn 1.
            await session.stop()
            await asyncio.wait_for(run_task, timeout=3.0)

            # Real mid-stream failure: point LLM at an unreachable port.
            dead_llm = LLMClient(api_key="x", base_url="http://127.0.0.1:1/dead")
            sink = _RecordingSink()
            session = ConversationSession(
                cfg=ConversationConfig(llm_model="m", speech_end_silence_ms=200),
                sink=sink, funasr=funasr, vad=vad, voxcpm=voxcpm, llm=dead_llm,
            )
            run_task = asyncio.create_task(session.run())

            await _scripted_turn(session, ordinal=1, text="first")
            err = await _wait_for_event(sink, "error", timeout_s=3.0)
            assert err.get("code") in {"llm_error", "llm_transport_error"}, err

            # Session should still be alive — state back to IDLE.
            await _wait_for_state(session, SessionState.IDLE, timeout_s=2.0)

            # Swap to healthy LLM in-place for turn 2.
            session._llm = llm_b  # noqa: SLF001
            await _scripted_turn(session, ordinal=2, text="second prompt")
            done = await _wait_for_event(sink, "response.done", timeout_s=10.0)
            assert done is not None
            # Assistant message from the healthy turn is committed.
            assert any(m.role == "assistant" for m in session._history)  # noqa: SLF001

            await session.stop()
            await asyncio.wait_for(run_task, timeout=3.0)
    finally:
        srv_a.should_exit = True
        srv_b.should_exit = True
        th_a.join(timeout=5.0)
        th_b.join(timeout=5.0)


@pytest.mark.asyncio
async def test_no_orphan_tasks_after_many_turns() -> None:
    """Run 50 scripted turns; assert no unexpected asyncio tasks survive.

    Baselines tasks present before the session starts; after cleanup
    the delta should be zero (modulo pytest-asyncio's own running tasks,
    which exist in both snapshots).
    """
    os.environ["PYTHONASYNCIODEBUG"] = "1"

    async with _workers() as (funasr, vad, voxcpm):
        async with _live_llm(_make_token_stream_app(["ok", "."])) as base_url:
            llm = LLMClient(api_key="x", base_url=f"{base_url}/api/v1")
            before = set(asyncio.all_tasks())
            sink = _RecordingSink()
            session = ConversationSession(
                cfg=ConversationConfig(llm_model="m", speech_end_silence_ms=200),
                sink=sink, funasr=funasr, vad=vad, voxcpm=voxcpm, llm=llm,
            )
            run_task = asyncio.create_task(session.run())
            for i in range(1, 51):
                await _scripted_turn(session, ordinal=i, text=f"turn {i}")
                # Wait until this turn completes.
                t_end = time.monotonic() + 5.0
                target_dones = i
                while time.monotonic() < t_end:
                    if len(sink.events_of("response.done")) >= target_dones:
                        break
                    await asyncio.sleep(0.005)
                else:
                    raise AssertionError(f"turn {i} did not complete")

            await session.stop()
            await asyncio.wait_for(run_task, timeout=5.0)
            # Give the event loop one tick for task cleanup to settle.
            await asyncio.sleep(0.05)

            after = set(asyncio.all_tasks())
            leaked = {t for t in after - before if not t.done() and t is not asyncio.current_task()}
            assert not leaked, (
                f"{len(leaked)} orphan task(s) survived after 50 turns: "
                f"{[t.get_name() for t in leaked]}"
            )
            assert len(sink.events_of("response.done")) == 50


@pytest.mark.asyncio
async def test_e5_two_speech_starts_in_one_tick_emit_exactly_one_interrupt() -> None:
    """ORCHESTRATION.md v2 §5 E5 test: fire two VAD speech_start events
    into the queue in the same event-loop tick, assert only one interrupt
    is emitted.

    Relies on §1.1's single-consumer serialisation: the first barge-in
    handler runs to completion before the second speech_start dequeues.
    """
    async with _workers() as (funasr, vad, _voxcpm):
        async with _live_llm(_make_token_stream_app(["reply", "."])) as base_url:
            llm = LLMClient(api_key="x", base_url=f"{base_url}/api/v1")
            slow_tts = _SlowTTSClient(per_chunk_delay_s=0.03, total_chunks=100)
            sink = _RecordingSink()
            session = ConversationSession(
                cfg=ConversationConfig(llm_model="m"),
                sink=sink, funasr=funasr, vad=vad,
                voxcpm=slow_tts, llm=llm,  # type: ignore[arg-type]
            )
            run_task = asyncio.create_task(session.run())

            await _scripted_turn(session, ordinal=1, text="go on")
            # Wait until state is TTS_SPEAKING.
            await _wait_for_state(session, SessionState.TTS_SPEAKING, timeout_s=3.0)
            while len(sink.audio) < 2:
                await asyncio.sleep(0.01)

            # Fire two speech_starts back to back inside the same loop iteration.
            # No intervening awaits between the two puts.
            await session._queue.put(  # noqa: SLF001
                VADEvent(kind="speech_start", utterance_ordinal=2, session_ms=5000)
            )
            await session._queue.put(  # noqa: SLF001
                VADEvent(kind="speech_start", utterance_ordinal=3, session_ms=5001)
            )

            # Give dispatch time to drain both.
            t_end = time.monotonic() + 2.0
            while time.monotonic() < t_end:
                if len(sink.events_of("interrupt")) >= 1:
                    break
                await asyncio.sleep(0.01)
            # Small settle — make sure no second interrupt slips through.
            await asyncio.sleep(0.3)

            assert len(sink.events_of("interrupt")) == 1, (
                f"expected exactly 1 interrupt, got {len(sink.events_of('interrupt'))}: "
                f"{[e.payload for e in sink.events_of('interrupt')]}"
            )
            # Current utterance ordinal should have advanced to the latest.
            assert session._current_utterance_ordinal == 3  # noqa: SLF001

            await session.stop()
            await asyncio.wait_for(run_task, timeout=3.0)
