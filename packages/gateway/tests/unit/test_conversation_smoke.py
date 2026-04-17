"""Smoke test: 1-turn conversation drives every stage end-to-end.

Exercises the handler-wired ConversationSession against:
- real STT (mock backend) + VAD (mock backend)
- real VoxCPM worker (mock backend)
- an OpenRouter-shaped uvicorn server for the LLM

Full barge-in / orphan-task / E5-serialisation tests live in
test_conversation_service.py (task #6).
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest
import uvicorn
from larynx_gateway.services.conversation_service import (
    ConversationConfig,
    ConversationSession,
    SessionState,
)
from larynx_gateway.services.llm_client import LLMClient
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient
from larynx_shared.ipc import WorkerChannel

# --- PCM helpers --------------------------------------------------------


def _tone(ms: int, sr: int = 16000, freq: float = 220.0, amp: float = 0.5) -> bytes:
    n = sr * ms // 1000
    t = np.arange(n, dtype=np.float32) / sr
    s = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return (s * 32767).astype(np.int16).tobytes()


def _silence(ms: int, sr: int = 16000) -> bytes:
    return (np.zeros(sr * ms // 1000, dtype=np.int16)).tobytes()


# --- Live OpenRouter-shaped LLM server ----------------------------------


def _sse_event(delta: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': {'content': delta}}]})}\n\n".encode()


def _make_llm_app(response: str):
    """Build an ASGI app that streams ``response`` one token at a time.

    Splits on whitespace to mimic real-world token boundaries. Ends with
    an explicit ``[DONE]`` sentinel even though stream close is
    authoritative — exercises that code path too.
    """

    async def app(scope, receive, send):
        assert scope["type"] == "http"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        # First token immediately; subsequent with a tiny stagger so
        # httpx's line buffer flushes between them.
        tokens = response.split(" ")
        for i, tok in enumerate(tokens):
            piece = tok if i == 0 else " " + tok
            await send({"type": "http.response.body", "body": _sse_event(piece), "more_body": True})
            await asyncio.sleep(0.005)
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    return app


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@asynccontextmanager
async def _live_llm_server(app):
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


# --- Recording sink -----------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.audio: list[tuple[int, int]] = []  # (bytes_len, sample_rate)
        self.events: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def send_audio(self, pcm_s16le: bytes, sample_rate: int) -> None:
        async with self._lock:
            self.audio.append((len(pcm_s16le), sample_rate))

    async def send_event(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            self.events.append(payload)

    def events_of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == kind]


# --- Worker spin-up -----------------------------------------------------


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
    voxcpm_mgr = VoxCPMModelManager(MockVoxCPMBackend())
    voxcpm_srv = VoxCPMWorkerServer(voxcpm_ch, voxcpm_mgr)
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


# --- Tests --------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversation_one_turn_happy_path() -> None:
    """Drive a tone → silence utterance → verify full event sequence."""
    async with (
        _workers() as (funasr, vad, voxcpm),
        _live_llm_server(_make_llm_app("Hello there, this is a reply.")) as base_url,
    ):
        llm = LLMClient(api_key="test", base_url=f"{base_url}/api/v1")
        sink = RecordingSink()
        cfg = ConversationConfig(
            llm_model="test-model",
            output_sample_rate=24000,
            speech_end_silence_ms=200,
            partial_interval_ms=200,
        )
        session = ConversationSession(
            cfg=cfg,
            sink=sink,
            funasr=funasr,
            vad=vad,
            voxcpm=voxcpm,
            llm=llm,
        )
        run_task = asyncio.create_task(session.run())
        # Feed 1s tone → 500ms silence to trigger one full turn.
        for _ in range(10):
            await session.feed_audio(_tone(100))
            await asyncio.sleep(0.03)
        for _ in range(6):
            await session.feed_audio(_silence(100))
            await asyncio.sleep(0.03)
        # Wait for response.done to arrive.
        for _ in range(400):
            if sink.events_of("response.done"):
                break
            await asyncio.sleep(0.01)
        assert sink.events_of("response.done"), (
            f"no response.done — events={[e['type'] for e in sink.events]}"
        )
        await session.stop()
        await asyncio.wait_for(run_task, timeout=3.0)

        # Verify expected stage events landed in order.
        kinds = [e["type"] for e in sink.events]
        for expected in (
            "input.speech_start",
            "input.speech_end",
            "transcript.final",
            "response.text_delta",
            "response.done",
        ):
            assert expected in kinds, f"missing {expected}; kinds={kinds}"
        # TTS audio frames streamed.
        assert sink.audio, "no TTS audio frames emitted"
        # Final state is IDLE.
        assert session.state is SessionState.IDLE
        # Assistant message committed to history.
        assert any(m.role == "assistant" for m in session._history)  # noqa: SLF001


@pytest.mark.asyncio
async def test_filler_transcript_skips_llm() -> None:
    """A transcript that normalises to a known filler token should NOT
    trigger an LLM call — the session returns to IDLE immediately."""
    # We don't need a live LLM server at all — if the filler check works,
    # nothing should touch the LLM. Point at an invalid URL; if the LLM
    # is hit the test will fail with a connection error.
    bad_llm = LLMClient(api_key="test", base_url="http://127.0.0.1:1/never")

    async with _workers() as (funasr, vad, voxcpm):
        sink = RecordingSink()

        # Force the STT path to emit a filler-only final by patching the
        # session in-flight — easier than getting the mock STT to produce
        # exactly "uh". We do it by pushing a synthesised STTEvent after
        # spinning up.
        from larynx_gateway.services.conversation_service import STTEvent, VADEvent

        cfg = ConversationConfig(llm_model="test-model", speech_end_silence_ms=200)
        session = ConversationSession(
            cfg=cfg, sink=sink, funasr=funasr, vad=vad, voxcpm=voxcpm, llm=bad_llm
        )
        run_task = asyncio.create_task(session.run())
        # Push a synthetic speech_start + speech_end + filler final.
        await session._queue.put(VADEvent(kind="speech_start", utterance_ordinal=1, session_ms=0))  # noqa: SLF001
        await session._queue.put(VADEvent(kind="speech_end", utterance_ordinal=1, session_ms=1000))  # noqa: SLF001
        await session._queue.put(  # noqa: SLF001
            STTEvent(
                kind="transcript_final",
                utterance_ordinal=1,
                text="uh",
                punctuated_text="Uh.",
            )
        )
        # Give dispatch a few ticks.
        await asyncio.sleep(0.1)
        await session.stop()
        await asyncio.wait_for(run_task, timeout=3.0)

        kinds = [e["type"] for e in sink.events]
        assert "transcript.final" in kinds
        # No LLM work — no response.text_delta and no response.done.
        assert "response.text_delta" not in kinds
        assert "response.done" not in kinds
        assert "error" not in kinds, f"unexpected error: {sink.events}"
