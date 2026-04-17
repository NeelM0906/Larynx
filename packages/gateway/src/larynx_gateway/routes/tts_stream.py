"""WS /v1/tts/stream — streaming text-to-speech.

Protocol:
  client → server  (text message, JSON):
    {
      "type": "synthesize",
      "text": "...",
      "voice_id": "...?",           // optional
      "prompt_text": "...?",        // optional (ultimate cloning with a voice)
      "sample_rate": 24000,
      "cfg_value": 2.0,
      "temperature": 1.0,
      "crossfade_ms": 10.0          // optional, default 10ms
    }

  server → client  (binary messages): raw PCM16 little-endian frames at
    the requested sample_rate, with a 10ms linear crossfade applied across
    chunk boundaries (last 10ms of the previous frame + first 10ms of the
    next frame are blended so callers don't hear a click).

  server → client  (final text message, JSON):
    { "type": "done", "total_duration_ms": N, "ttfb_ms": N,
      "chunk_count": N, "sample_rate": N }

  server → client  (on error, text JSON then close):
    { "type": "error", "code": "...", "message": "..." }

Disconnects: if the client closes mid-stream, we exit the streaming context
manager, which sends a CancelStreamRequest to the worker so the GPU stops
generating for this session.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from larynx_shared.audio import crossfade_chunks
from larynx_shared.ipc import SynthesizeChunkFrame, SynthesizeDoneFrame
from larynx_shared.ipc.client_base import WorkerError
from prometheus_client import Histogram
from pydantic import BaseModel, Field, ValidationError

from larynx_gateway.services import tts_service
from larynx_gateway.services.voice_library import VoiceLibrary
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient
from larynx_gateway.ws_auth import require_ws_bearer_token

router = APIRouter(prefix="/v1", tags=["tts"])
log = structlog.get_logger(__name__)


_tts_ttfb = Histogram(
    "larynx_tts_stream_ttfb_seconds",
    "Time from WS config receipt to first PCM frame emitted (gateway-side)",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0),
)
_tts_total_duration = Histogram(
    "larynx_tts_stream_total_seconds",
    "End-to-end streaming synthesis time (config → done frame)",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)


class _TTSStreamConfig(BaseModel):
    """Validated config frame sent by the client on WS open."""

    type: str = "synthesize"
    text: str = Field(min_length=1, max_length=5000)
    voice_id: str | None = None
    prompt_text: str | None = Field(default=None, max_length=500)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    cfg_value: float = Field(default=2.0, ge=0.0, le=10.0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    crossfade_ms: float = Field(default=10.0, ge=0.0, le=50.0)


@router.websocket("/tts/stream")
async def ws_tts_stream(ws: WebSocket) -> None:
    await ws.accept()
    # Reject new sessions once the gateway has started draining — the
    # /ready endpoint already returns 503 in this state, but LBs may
    # still have an in-flight handshake we can short-circuit here.
    if getattr(ws.app.state, "shutting_down", False):
        await _send_error(ws, "shutting_down", "gateway is draining")
        return
    if not await require_ws_bearer_token(ws):
        return

    session_id = uuid.uuid4().hex
    ws_log = log.bind(session_id=session_id, endpoint="/v1/tts/stream")
    ws_log.info("tts_stream.connected")

    try:
        # 1. Receive and validate the config frame.
        try:
            raw = await ws.receive_text()
            cfg = _TTSStreamConfig.model_validate(json.loads(raw))
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            await _send_error(ws, "invalid_config", str(e))
            return

        # 2. Resolve voice conditioning (reuse REST service helpers).
        client: VoxCPMClient = ws.app.state.voxcpm_client
        library = _build_library(ws)
        from larynx_gateway.schemas.tts import TTSRequest

        tts_req = TTSRequest(
            text=cfg.text,
            voice_id=cfg.voice_id,
            sample_rate=cfg.sample_rate,
            cfg_value=cfg.cfg_value,
            temperature=cfg.temperature,
            prompt_text=cfg.prompt_text,
        )
        try:
            conditioning = await tts_service.resolve_conditioning(tts_req, library, voxcpm=client)
        except ValueError as e:
            await _send_error(ws, "invalid_input", str(e))
            return
        if conditioning is None:
            await _send_error(ws, "voice_not_found", f"voice_id={cfg.voice_id!r}")
            return

        # 3. Open the streaming RPC, splice 10ms crossfade across boundaries.
        t_start = time.perf_counter()
        ttfb_observed = False
        pending: bytes = b""  # accumulates the "tail" to be emitted next
        chunks_sent = 0
        total_bytes = 0

        async with client.synthesize_text_stream(
            text=cfg.text,
            sample_rate=cfg.sample_rate,
            voice_id=cfg.voice_id,
            ref_audio_latents=conditioning.ref_audio_latents,
            prompt_audio_latents=conditioning.prompt_audio_latents,
            prompt_text=conditioning.prompt_text,
            cfg_value=cfg.cfg_value,
            temperature=cfg.temperature,
        ) as frames:
            async for frame in frames:
                if isinstance(frame, SynthesizeChunkFrame):
                    tail, head = crossfade_chunks(
                        pending,
                        frame.pcm_s16le,
                        frame.sample_rate,
                        overlap_ms=cfg.crossfade_ms,
                    )
                    # Flush the finalised tail; keep head as the new pending
                    # tail so the next chunk can crossfade into it.
                    if tail:
                        await ws.send_bytes(tail)
                        total_bytes += len(tail)
                        chunks_sent += 1
                        if not ttfb_observed:
                            ttfb_ms = int((time.perf_counter() - t_start) * 1000)
                            _tts_ttfb.observe(ttfb_ms / 1000.0)
                            ttfb_observed = True
                            ws_log.info("tts_stream.ttfb", ttfb_ms=ttfb_ms)
                    pending = head
                elif isinstance(frame, SynthesizeDoneFrame):
                    # Flush whatever's pending (no chunk to crossfade into).
                    if pending:
                        await ws.send_bytes(pending)
                        total_bytes += len(pending)
                        chunks_sent += 1
                        pending = b""
                    total_s = time.perf_counter() - t_start
                    _tts_total_duration.observe(total_s)
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "done",
                                "total_duration_ms": frame.total_duration_ms,
                                "ttfb_ms": int((time.perf_counter() - t_start) * 1000)
                                if not ttfb_observed
                                else int(total_s * 1000),
                                "chunk_count": chunks_sent,
                                "sample_rate": frame.sample_rate,
                                "bytes_sent": total_bytes,
                            }
                        )
                    )
                    ws_log.info(
                        "tts_stream.done",
                        chars=len(cfg.text),
                        chunks=chunks_sent,
                        bytes=total_bytes,
                        total_ms=int(total_s * 1000),
                        audio_duration_ms=frame.total_duration_ms,
                        ttfb_ms=frame.ttfb_ms,
                    )
    except WebSocketDisconnect:
        ws_log.info("tts_stream.client_disconnect")
        return
    except WorkerError as e:
        ws_log.warning("tts_stream.worker_error", code=e.code, message=e.message)
        await _send_error(ws, e.code, e.message)
        return
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        ws_log.exception("tts_stream.failed")
        await _send_error(ws, "internal_error", str(e))
        return

    # Polite close on the happy path.
    await ws.close()


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    try:
        await ws.send_text(json.dumps({"type": "error", "code": code, "message": message}))
    except Exception:  # pragma: no cover — client already gone
        return
    try:
        await ws.close()
    except Exception:  # pragma: no cover
        return


def _build_library(ws: WebSocket) -> VoiceLibrary:
    """Construct a transient ``VoiceLibrary`` for a WS handler.

    Regular routes get this via ``Depends(get_voice_library)`` but WS routes
    don't chain FastAPI dependencies the same way. We open a short-lived DB
    session for the voice lookup at the top of the handler. The session is
    garbage-collected once the library goes out of scope.
    """
    from larynx_gateway.db.session import get_session_factory

    session_factory = get_session_factory()
    session = session_factory()
    return VoiceLibrary(
        session=session,
        voxcpm=ws.app.state.voxcpm_client,
        cache=ws.app.state.latent_cache,
        data_dir=ws.app.state.data_dir,
        design_ttl_s=ws.app.state.design_ttl_s,
    )
