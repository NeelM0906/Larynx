"""WS /v1/conversation — full-duplex interruptible conversation.

Thin transport layer. All state + cancellation semantics live in
:class:`larynx_gateway.services.conversation_service.ConversationSession`.
The route's job:

  1. Accept WS + bearer-token auth.
  2. Receive one JSON config message.
  3. Build a ``ConversationSession`` with a sink that writes to the WS.
  4. Pump binary frames (PCM16 LE) into the session and control JSON
     messages into its control channel.
  5. On disconnect, stop the session and drain its cleanup.

Protocol summary (see PRD §5.6 for event catalogue):

  client → server  (first text msg, JSON config):
    {
      "type": "config",
      "voice_id": "...?",
      "llm_model": "anthropic/claude-haiku-4.5",
      "system_prompt": "...?",
      "input_sample_rate": 16000,
      "output_sample_rate": 24000,
      "speech_end_silence_ms": 300,
      "partial_interval_ms": 720
    }

  client → server  (binary): 16-bit LE PCM mono at input_sample_rate.
  client → server  (text JSON): control events, e.g. {"type":"session.end"}.

  server → client  (binary): 16-bit LE PCM mono at output_sample_rate
    (TTS output — play directly).
  server → client  (text JSON): session.status, input.speech_start/end,
    transcript.partial/final, response.text_delta, interrupt,
    response.done, error.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

from larynx_gateway.schemas.tts import TTSRequest
from larynx_gateway.services import tts_service
from larynx_gateway.services.conversation_service import (
    ConversationConfig,
    ConversationSession,
)
from larynx_gateway.services.llm_client import LLMClient
from larynx_gateway.services.voice_library import VoiceLibrary
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient
from larynx_gateway.ws_auth import require_ws_bearer_token

router = APIRouter(prefix="/v1", tags=["conversation"])
log = structlog.get_logger(__name__)


class _ConversationConfigFrame(BaseModel):
    type: str = "config"
    voice_id: str | None = None
    llm_model: str | None = None  # default comes from app state
    system_prompt: str = ""
    # ISO-639 code for STT. Defaults to "en" so Fun-ASR-Nano's auto-detect
    # doesn't misclassify short English utterances as Chinese filler tokens.
    # Pass null to opt back into auto-detect.
    language: str | None = "en"
    input_sample_rate: int = Field(default=16000, ge=8000, le=48000)
    output_sample_rate: int = Field(default=24000, ge=8000, le=48000)
    speech_end_silence_ms: int = Field(default=300, ge=100, le=2000)
    partial_interval_ms: int = Field(default=720, ge=200, le=2000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=512, ge=1, le=4096)
    llm_read_timeout_s: float = Field(default=15.0, ge=1.0, le=120.0)


class _WSSink:
    """ClientSink implementation that writes to a FastAPI WebSocket.

    A lock serialises concurrent writes — audio frames, JSON events, and
    control messages may be produced from different tasks inside the
    session, but the WS protocol only tolerates one outgoing frame at a
    time. Small critical sections; overhead is negligible at v1 concurrency.
    """

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._lock = asyncio.Lock()
        self._closed = False

    async def send_audio(self, pcm_s16le: bytes, sample_rate: int) -> None:  # noqa: ARG002
        if self._closed:
            return
        async with self._lock:
            try:
                await self._ws.send_bytes(pcm_s16le)
            except Exception:
                self._closed = True

    async def send_event(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        async with self._lock:
            try:
                await self._ws.send_text(json.dumps(payload))
            except Exception:
                self._closed = True


@router.websocket("/conversation")
async def ws_conversation(ws: WebSocket) -> None:
    await ws.accept()
    if getattr(ws.app.state, "shutting_down", False):
        await _send_error(ws, "shutting_down", "gateway is draining")
        return
    if not await require_ws_bearer_token(ws):
        return

    session_id = uuid.uuid4().hex
    ws_log = log.bind(session_id=session_id, endpoint="/v1/conversation")
    ws_log.info("conversation.connected")

    try:
        raw = await ws.receive_text()
        cfg_frame = _ConversationConfigFrame.model_validate(json.loads(raw))
    except (ValidationError, json.JSONDecodeError, ValueError, WebSocketDisconnect) as e:
        await _send_error(ws, "invalid_config", str(e))
        return

    voxcpm: VoxCPMClient = ws.app.state.voxcpm_client

    # Resolve voice_id -> conditioning ONCE. The VoxCPM worker ignores the
    # voice_id string; synthesis is deterministic only when ref_audio_latents
    # (or lora_name) are passed. Resolving per-session and reusing the same
    # conditioning on every sentence is what keeps the voice consistent —
    # otherwise each sentence synthesises with random conditioning.
    ref_latents: bytes | None = None
    prompt_latents: bytes | None = None
    prompt_text: str = ""
    lora_name: str | None = None
    if cfg_frame.voice_id is not None:
        library = _build_library(ws)
        tts_req = TTSRequest(
            text="x",  # placeholder — resolve_conditioning only reads voice_id here
            voice_id=cfg_frame.voice_id,
            sample_rate=cfg_frame.output_sample_rate,
        )
        try:
            conditioning = await tts_service.resolve_conditioning(
                tts_req, library, voxcpm=voxcpm
            )
        except ValueError as e:
            await _send_error(ws, "invalid_input", str(e))
            return
        if conditioning is None:
            await _send_error(ws, "voice_not_found", f"voice_id={cfg_frame.voice_id!r}")
            return
        ref_latents = conditioning.ref_audio_latents
        prompt_latents = conditioning.prompt_audio_latents
        prompt_text = conditioning.prompt_text
        lora_name = conditioning.lora_name

    default_model: str = getattr(ws.app.state, "llm_default_model", "anthropic/claude-haiku-4.5")
    cfg = ConversationConfig(
        voice_id=cfg_frame.voice_id,
        ref_audio_latents=ref_latents,
        prompt_audio_latents=prompt_latents,
        prompt_text=prompt_text,
        lora_name=lora_name,
        llm_model=cfg_frame.llm_model or default_model,
        system_prompt=cfg_frame.system_prompt,
        stt_language=cfg_frame.language,
        input_sample_rate=cfg_frame.input_sample_rate,
        output_sample_rate=cfg_frame.output_sample_rate,
        speech_end_silence_ms=cfg_frame.speech_end_silence_ms,
        partial_interval_ms=cfg_frame.partial_interval_ms,
        temperature=cfg_frame.temperature,
        max_tokens=cfg_frame.max_tokens,
        llm_read_timeout_s=cfg_frame.llm_read_timeout_s,
    )

    funasr: FunASRClient = ws.app.state.funasr_client
    vad: VadPuncClient = ws.app.state.vad_punc_client
    llm: LLMClient = ws.app.state.llm_client

    sink = _WSSink(ws)
    session = ConversationSession(
        cfg=cfg,
        sink=sink,
        funasr=funasr,
        vad=vad,
        voxcpm=voxcpm,
        llm=llm,
        session_id=session_id,
    )

    run_task = asyncio.create_task(session.run(), name=f"conv-run-{session_id[:8]}")
    try:
        while True:
            msg = await ws.receive()
            t = msg.get("type")
            if t == "websocket.disconnect":
                break
            if t != "websocket.receive":
                continue
            if msg.get("bytes") is not None:
                await session.feed_audio(msg["bytes"])
            elif msg.get("text") is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    await session.client_control(payload)
                    if payload.get("type") == "session.end":
                        break
    except WebSocketDisconnect:
        ws_log.info("conversation.client_disconnect")
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        ws_log.exception("conversation.receive_loop_failed")
    finally:
        await session.stop()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(run_task, timeout=5.0)
        with contextlib.suppress(Exception):
            await ws.close()
        ws_log.info("conversation.closed")


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    with contextlib.suppress(Exception):
        await ws.send_text(json.dumps({"type": "error", "code": code, "message": message}))
    with contextlib.suppress(Exception):
        await ws.close()


def _build_library(ws: WebSocket) -> VoiceLibrary:
    """Construct a transient ``VoiceLibrary`` for this WS handler.

    Mirrors ``routes/tts_stream._build_library``: WS routes don't chain
    FastAPI dependencies, so we open a short-lived DB session for the
    voice lookup. It's garbage-collected once the library goes out of
    scope — we only need it for the one resolve call at session start.
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
