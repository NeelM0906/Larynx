"""OpenAI-compatible audio endpoints (PRD §5.9).

- ``POST /v1/audio/transcriptions`` — Whisper-shaped STT (M3)
- ``POST /v1/audio/speech``         — OpenAI TTS shim (M8 Part B)

Transcriptions contract matches OpenAI's Whisper API:

- ``file``            — required, audio upload
- ``model``           — required by OpenAI's spec but ignored here (we
                        always route through the language_router)
- ``language``        — optional ISO-639 code
- ``response_format`` — ``json`` (default) or ``text``
- ``prompt``          — OpenAI's prompt arg maps to our ``hotwords``
                        (single string; we split on commas)

The response shape follows OpenAI's basic ``json`` (``{"text": "..."}``)
plus the ``language`` key we've populated for free. ``verbose_json`` with
segments is a future addition — Fun-ASR doesn't expose timestamps yet.

Speech contract matches OpenAI's TTS API. See :mod:`.schemas.openai`.
Error bodies follow OpenAI's shape: ``{"error": {"type", "code",
"message"}}`` so the OpenAI SDK's exception types fire correctly.
"""

from __future__ import annotations

from typing import Literal

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from larynx_shared.audio import pack_wav
from larynx_shared.audio.encode import encode as encode_audio
from larynx_shared.audio.encode import pyav_available
from larynx_shared.ipc.client_base import WorkerError

from larynx_gateway.auth import require_bearer_token
from larynx_gateway.deps import get_funasr_client, get_vad_punc_client, get_voice_library
from larynx_gateway.schemas.openai import OpenAIResponseFormat, OpenAISpeechRequest
from larynx_gateway.schemas.stt import OpenAITranscriptionResponse
from larynx_gateway.schemas.tts import TTSRequest
from larynx_gateway.services import stt_service, tts_service
from larynx_gateway.services.voice_library import VoiceLibrary
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient

router = APIRouter(prefix="/v1", tags=["openai-compat"])
log = structlog.get_logger(__name__)


ResponseFormat = Literal["json", "text"]


# Codecs that require pyav (FFmpeg). wav + pcm stay in-process.
_PYAV_FORMATS: frozenset[str] = frozenset({"mp3", "opus", "aac", "flac"})
_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/L16",
}


def _openai_error(*, status_code: int, type_: str, code: str, message: str) -> JSONResponse:
    """Build an OpenAI-shaped error response.

    The OpenAI SDK parses this shape into typed exceptions; FastAPI's
    default ``{"detail": ...}`` body would break ``openai.APIError``
    subclassing on the client.
    """
    return JSONResponse(
        status_code=status_code,
        content={"error": {"type": type_, "code": code, "message": message}},
    )


@router.post(
    "/audio/transcriptions",
    dependencies=[Depends(require_bearer_token)],
    responses={
        400: {"description": "Invalid request"},
        401: {"description": "Missing or invalid bearer token"},
        503: {"description": "STT worker unavailable"},
    },
)
async def post_transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),  # accepted for API compat; ignored
    language: str | None = Form(default=None),
    response_format: ResponseFormat = Form(default="json"),
    prompt: str | None = Form(default=None),
    funasr: FunASRClient = Depends(get_funasr_client),
    vad_punc: VadPuncClient = Depends(get_vad_punc_client),
):
    _ = model  # OpenAI clients pass "whisper-1"; we always route via Fun-ASR
    if response_format not in {"json", "text"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported response_format {response_format!r}; supported: json, text",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="uploaded file is empty"
        )

    # OpenAI's ``prompt`` is a single biasing string; we treat comma-separated
    # tokens as hotwords, matching the native /v1/stt behaviour.
    hotwords = [w.strip() for w in (prompt or "").split(",") if w.strip()]

    try:
        result = await stt_service.transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename,
            language=language,
            hotwords=hotwords,
            itn=True,
            punctuate=True,
            trim_silence=True,
            funasr=funasr,
            vad_punc=vad_punc,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except WorkerError as e:
        if e.code in {"invalid_input", "unsupported_language"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e
        log.error("openai.transcribe_worker_error", code=e.code, message=e.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"stt worker error: {e.code}",
        ) from e

    log.info(
        "openai.transcribe",
        filename=file.filename,
        bytes=len(audio_bytes),
        language_out=result.language,
        model_used=result.model_used,
        response_format=response_format,
    )

    if response_format == "text":
        return PlainTextResponse(content=result.text)

    return OpenAITranscriptionResponse(text=result.text, language=result.language)


# ---------------------------------------------------------------------------
# POST /v1/audio/speech — OpenAI TTS shim (M8 Part B).
#
# Speed mapping (PRD §5.9 / ORCHESTRATION-M8.md §2.1):
#   speed > 1.0 → would reduce ``inference_timesteps`` proportionally
#                 (min 6) and raise ``chunk_stride``.
#   speed < 1.0 → would increase ``inference_timesteps`` up to 16 and
#                 lower ``chunk_stride`` by ~speed.
#   speed == 1.0 → passthrough.
#
# IMPLEMENTATION NOTE: in the current worker build, ``inference_timesteps``
# is a boot-time init knob on the nano-vllm-voxcpm engine (see
# ``voxcpm_worker.model_manager``) and ``chunk_stride`` isn't exposed on
# the synthesis RPC at all. We accept ``speed`` and validate it, but the
# only per-request knob we can steer today is ``temperature`` — at
# higher speeds we nudge temperature down slightly to tighten prosody,
# which tracks the OpenAI "speed" affordance loosely. A fuller mapping
# lands when the worker exposes per-request timesteps + chunk_stride.
# ---------------------------------------------------------------------------


def _map_speed_to_knobs(speed: float, base_temperature: float) -> dict[str, float]:
    """Return overrides for TTSRequest engine knobs given ``speed``.

    Today this only nudges ``temperature`` because the per-request
    timesteps + chunk_stride knobs aren't wired through the worker RPC
    yet. See the module docstring.
    """
    if speed == 1.0:
        return {}
    # Slightly tighter prosody at higher speeds, slightly looser at lower.
    # Bounded by the TTSRequest temperature range [0.0, 2.0].
    scale = 1.0 / max(speed, 0.25)
    tempered = max(0.1, min(2.0, base_temperature * scale))
    return {"temperature": tempered}


@router.post(
    "/audio/speech",
    dependencies=[Depends(require_bearer_token)],
    responses={
        200: {
            "content": {
                "audio/mpeg": {},
                "audio/ogg": {},
                "audio/aac": {},
                "audio/flac": {},
                "audio/wav": {},
                "audio/L16": {},
            }
        },
        400: {"description": "Invalid request"},
        401: {"description": "Missing or invalid bearer token"},
        404: {"description": "Voice not found"},
        500: {"description": "Synthesis failed"},
        501: {"description": "Requested codec unavailable (pyav missing)"},
    },
)
async def post_speech(
    request: Request,
    library: VoiceLibrary = Depends(get_voice_library),
) -> Response:
    # Parse + validate body ourselves so we can surface OpenAI-shaped
    # error bodies rather than FastAPI's ``{"detail": ...}`` default.
    try:
        body = await request.json()
    except Exception as e:
        return _openai_error(
            status_code=400,
            type_="invalid_request_error",
            code="invalid_json",
            message=f"invalid JSON body: {e}",
        )
    try:
        req = OpenAISpeechRequest.model_validate(body)
    except Exception as e:
        return _openai_error(
            status_code=400,
            type_="invalid_request_error",
            code="invalid_request",
            message=str(e),
        )

    # Voice lookup by name (distinct rows per OpenAI preset, see
    # scripts/load_demo_voices.py).
    voice = await library.get_by_name(req.voice)
    if voice is None:
        return _openai_error(
            status_code=404,
            type_="invalid_request_error",
            code="voice_not_found",
            message=f"no voice named {req.voice!r}",
        )

    # Reject codecs that require pyav when pyav isn't importable.
    if req.response_format in _PYAV_FORMATS and not pyav_available():
        return _openai_error(
            status_code=501,
            type_="server_error",
            code="codec_unavailable",
            message=f"response_format={req.response_format!r} requires pyav (not installed)",
        )

    # Build a native TTSRequest and reuse resolve_conditioning + synth.
    # We always ask the worker for raw PCM (output_format=pcm16) and do
    # the container encode at this edge — keeps the worker format-agnostic
    # and avoids double-packaging for wav.
    base_temperature = 1.0
    speed_overrides = _map_speed_to_knobs(req.speed, base_temperature)
    tts_req = TTSRequest(
        text=req.input,
        voice_id=voice.id,
        sample_rate=24000,
        output_format="pcm16",
        temperature=speed_overrides.get("temperature", base_temperature),
    )

    client: VoxCPMClient = request.app.state.voxcpm_client

    try:
        conditioning = await tts_service.resolve_conditioning(tts_req, library, voxcpm=client)
    except ValueError as e:
        return _openai_error(
            status_code=400,
            type_="invalid_request_error",
            code="invalid_request",
            message=str(e),
        )
    if conditioning is None:
        # Race: voice was deleted between get_by_name and latent fetch.
        return _openai_error(
            status_code=404,
            type_="invalid_request_error",
            code="voice_not_found",
            message=f"no voice named {req.voice!r}",
        )

    try:
        result = await tts_service.synthesize(tts_req, conditioning, client)
    except WorkerError as e:
        log.error("openai.speech_worker_error", code=e.code, message=e.message)
        return _openai_error(
            status_code=503,
            type_="server_error",
            code="tts_worker_unavailable",
            message=f"tts worker error: {e.code}",
        )

    # The service packaged as audio/L16 (pcm); grab the raw PCM bytes and
    # re-encode at this edge in the requested container.
    pcm_bytes = result.audio  # raw s16le mono at result.sample_rate
    try:
        audio_bytes, content_type = _package_openai(
            pcm_bytes, result.sample_rate, req.response_format
        )
    except RuntimeError as e:
        # Codec not available at runtime (pyav present but FFmpeg build
        # lacks this encoder).
        return _openai_error(
            status_code=501,
            type_="server_error",
            code="codec_unavailable",
            message=str(e),
        )
    except ValueError as e:
        return _openai_error(
            status_code=400,
            type_="invalid_request_error",
            code="invalid_request",
            message=str(e),
        )

    log.info(
        "openai.speech",
        chars=len(req.input),
        voice=req.voice,
        voice_id=voice.id,
        response_format=req.response_format,
        speed=req.speed,
        bytes=len(audio_bytes),
        duration_ms=result.duration_ms,
        generation_time_ms=result.generation_time_ms,
    )
    headers = {
        "X-Voice-ID": voice.id,
        "X-Generation-Time-Ms": str(result.generation_time_ms),
        "X-Audio-Duration-Ms": str(result.duration_ms),
        "X-Sample-Rate": str(result.sample_rate),
    }
    return Response(content=audio_bytes, media_type=content_type, headers=headers)


def _package_openai(pcm: bytes, sample_rate: int, fmt: OpenAIResponseFormat) -> tuple[bytes, str]:
    """Wrap raw PCM in the caller's chosen container.

    wav + pcm stay in-process; mp3/opus/aac/flac go through pyav. Split
    out of ``tts_service._package`` so the native /v1/tts response
    shape (wav/pcm16) stays frozen for M2 clients.
    """
    if fmt == "wav":
        return pack_wav(pcm, sample_rate=sample_rate), _CONTENT_TYPES["wav"]
    if fmt == "pcm":
        return pcm, _CONTENT_TYPES["pcm"]
    if fmt in _PYAV_FORMATS:
        encoded = encode_audio(pcm, sample_rate=sample_rate, fmt=fmt)  # type: ignore[arg-type]
        return encoded, _CONTENT_TYPES[fmt]
    raise ValueError(f"unsupported response_format {fmt!r}")
