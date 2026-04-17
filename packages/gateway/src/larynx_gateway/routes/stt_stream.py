"""WS /v1/stt/stream — streaming speech-to-text with rolling partials.

Protocol:
  1. Client sends JSON config frame (first message):
     {
       "type": "config",
       "sample_rate": 16000,
       "language": "en" | null,
       "hotwords": ["foo", "bar"],
       "chunk_interval_ms": 720,
       "speech_end_silence_ms": 300
     }

  2. Client streams binary PCM16 LE frames at ``sample_rate`` mono.

  3. Server emits JSON events (text WS frames):
     - {"type":"speech_start","session_ms":N}
     - {"type":"partial","text":"...","interval_ms":N,"decode_ms":N}
     - {"type":"speech_end","session_ms":N}
     - {"type":"final","text":"...","punctuated_text":"...","language":"en", ...}
     - {"type":"error","code":"...","message":"..."}
     - {"type":"heartbeat","vad_state":"...","session_ms":N}  (periodic)

  Closing: either side may close the WS; the server then drains pending
  partials/finals before replying with close.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from prometheus_client import Histogram
from pydantic import BaseModel, Field, ValidationError

from larynx_gateway.services.stt_stream_service import STTStreamConfig, STTStreamSession
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient
from larynx_gateway.ws_auth import require_ws_bearer_token

router = APIRouter(prefix="/v1", tags=["stt"])
log = structlog.get_logger(__name__)


_partial_interval = Histogram(
    "larynx_stt_stream_partial_interval_seconds",
    "Wall-clock interval between consecutive partial emissions per session",
    buckets=(0.2, 0.4, 0.6, 0.72, 0.9, 1.2, 1.8, 3.0),
)
_finalization_seconds = Histogram(
    "larynx_stt_stream_finalization_seconds",
    "Time from VAD speech_end event to final event emission",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0),
)


class _STTStreamConfigFrame(BaseModel):
    type: str = "config"
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    language: str | None = None
    hotwords: list[str] = Field(default_factory=list)
    chunk_interval_ms: int = Field(default=720, ge=200, le=2000)
    speech_end_silence_ms: int = Field(default=300, ge=100, le=2000)
    itn: bool = True
    drop_tail_tokens: int = Field(default=5, ge=0, le=20)


@router.websocket("/stt/stream")
async def ws_stt_stream(ws: WebSocket) -> None:
    await ws.accept()
    if not await require_ws_bearer_token(ws):
        return

    session_id = uuid.uuid4().hex
    ws_log = log.bind(session_id=session_id, endpoint="/v1/stt/stream")
    ws_log.info("stt_stream.connected")

    try:
        raw = await ws.receive_text()
        cfg_frame = _STTStreamConfigFrame.model_validate(json.loads(raw))
    except (ValidationError, json.JSONDecodeError, ValueError, WebSocketDisconnect) as e:
        await _send_error(ws, "invalid_config", str(e))
        return

    cfg = STTStreamConfig(
        sample_rate=cfg_frame.sample_rate,
        language=cfg_frame.language,
        hotwords=tuple(cfg_frame.hotwords),
        chunk_interval_ms=cfg_frame.chunk_interval_ms,
        speech_end_silence_ms=cfg_frame.speech_end_silence_ms,
        itn=cfg_frame.itn,
        drop_tail_tokens=cfg_frame.drop_tail_tokens,
    )

    funasr: FunASRClient = ws.app.state.funasr_client
    vad: VadPuncClient = ws.app.state.vad_punc_client

    session = STTStreamSession(funasr=funasr, vad=vad, cfg=cfg, session_id=session_id)

    # Collect incoming PCM from WS into an async generator the service consumes.
    pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def pcm_source() -> "asyncio.AsyncGenerator[bytes, None]":
        while True:
            item = await pcm_queue.get()
            if item is None:
                return
            yield item

    async def receiver() -> None:
        """Pump WS binary frames into pcm_queue; terminate on close."""
        try:
            while True:
                msg = await ws.receive()
                t = msg.get("type")
                if t == "websocket.disconnect":
                    break
                if t == "websocket.receive":
                    if msg.get("bytes") is not None:
                        await pcm_queue.put(msg["bytes"])
                    elif msg.get("text") is not None:
                        # Treat a "stop" text frame as end-of-audio.
                        try:
                            payload = json.loads(msg["text"])
                            if isinstance(payload, dict) and payload.get("type") == "stop":
                                break
                        except json.JSONDecodeError:
                            continue
                else:
                    break
        finally:
            await pcm_queue.put(None)

    # --- run the session + forward its events to the WS ---
    speech_end_ts: dict[str, float] = {}
    last_partial_ts: dict[str, float] = {"t": 0.0}

    async def forwarder() -> None:
        async for ev in session.events():
            try:
                if ev["type"] == "partial":
                    now = time.monotonic()
                    prev = last_partial_ts["t"]
                    if prev > 0:
                        _partial_interval.observe(now - prev)
                    last_partial_ts["t"] = now
                elif ev["type"] == "speech_end":
                    speech_end_ts["t"] = time.monotonic()
                elif ev["type"] == "final" and "t" in speech_end_ts:
                    _finalization_seconds.observe(
                        time.monotonic() - speech_end_ts.pop("t")
                    )
                await ws.send_text(json.dumps(ev))
            except Exception:  # pragma: no cover — client already gone
                return

    try:
        recv_task = asyncio.create_task(receiver(), name=f"stt-recv-{session_id[:8]}")
        fwd_task = asyncio.create_task(forwarder(), name=f"stt-fwd-{session_id[:8]}")
        # Session.run drains events queue on exit, which closes the forwarder
        # naturally; we await all three.
        await session.run(pcm_source())
        await fwd_task
        recv_task.cancel()
        try:
            await recv_task
        except (asyncio.CancelledError, Exception):
            pass
    except WebSocketDisconnect:
        ws_log.info("stt_stream.client_disconnect")
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        ws_log.exception("stt_stream.failed")
        await _send_error(ws, "internal_error", str(e))
        return

    try:
        await ws.close()
    except Exception:  # pragma: no cover
        pass


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    try:
        await ws.send_text(
            json.dumps({"type": "error", "code": code, "message": message})
        )
    except Exception:  # pragma: no cover
        return
    try:
        await ws.close()
    except Exception:  # pragma: no cover
        return
