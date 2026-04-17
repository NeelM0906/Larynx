"""POST /v1/stt — speech-to-text via Fun-ASR + optional CT-Transformer punctuation.

Multipart form contract (see PRD §5.3):

- ``file``         — audio blob (wav/mp3/flac/ogg/...)
- ``language``     — ISO-639 code; omit to auto-detect (Nano only)
- ``hotwords``     — comma-separated list of terms to boost
- ``itn``          — bool (default true); Fun-ASR's inverse-text-norm flag
- ``punctuate``    — bool (default true); routes transcript through ct-punc
- ``trim_silence`` — bool (default true); fsmn-vad trims leading/trailing silence

Response: ``STTResponse`` (JSON). Errors surface as 400 (bad input),
401 (auth), 404 not applicable here, 503 (worker unavailable).
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from larynx_shared.ipc.client_base import WorkerError

from larynx_gateway.auth import require_bearer_token
from larynx_gateway.deps import get_funasr_client, get_vad_punc_client
from larynx_gateway.schemas.stt import STTResponse
from larynx_gateway.services import stt_service
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient

router = APIRouter(prefix="/v1", tags=["stt"])
log = structlog.get_logger(__name__)


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"true", "1", "yes", "on"}:
        return True
    if v in {"false", "0", "no", "off"}:
        return False
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"expected boolean, got {raw!r}",
    )


def _parse_hotwords(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [w.strip() for w in raw.split(",") if w.strip()]


@router.post(
    "/stt",
    response_model=STTResponse,
    dependencies=[Depends(require_bearer_token)],
    responses={
        400: {"description": "Invalid request"},
        401: {"description": "Missing or invalid bearer token"},
        503: {"description": "STT worker unavailable"},
    },
)
async def post_stt(
    file: UploadFile = File(..., description="Audio blob (wav/mp3/flac/...)."),
    language: str | None = Form(default=None),
    hotwords: str | None = Form(default=None),
    itn: str | None = Form(default=None),
    punctuate: str | None = Form(default=None),
    trim_silence: str | None = Form(default=None),
    funasr: FunASRClient = Depends(get_funasr_client),
    vad_punc: VadPuncClient = Depends(get_vad_punc_client),
) -> STTResponse:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="uploaded file is empty"
        )

    parsed_hotwords = _parse_hotwords(hotwords)
    parsed_itn = _parse_bool(itn, True)
    parsed_punctuate = _parse_bool(punctuate, True)
    parsed_trim = _parse_bool(trim_silence, True)

    try:
        result = await stt_service.transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename,
            language=language,
            hotwords=parsed_hotwords,
            itn=parsed_itn,
            punctuate=parsed_punctuate,
            trim_silence=parsed_trim,
            funasr=funasr,
            vad_punc=vad_punc,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except WorkerError as e:
        if e.code in {"invalid_input", "unsupported_language"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e
        log.error("stt.worker_error", code=e.code, message=e.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"stt worker error: {e.code}",
        ) from e

    log.info(
        "stt.request",
        filename=file.filename,
        bytes=len(audio_bytes),
        language_in=language,
        language_out=result.language,
        model_used=result.model_used,
        hotwords=len(parsed_hotwords),
        itn=parsed_itn,
        punctuated=result.punctuated,
        duration_ms=result.duration_ms,
        processing_ms=result.processing_ms,
    )

    return STTResponse(
        text=result.text,
        language=result.language,
        model_used=result.model_used,
        duration_ms=result.duration_ms,
        processing_ms=result.processing_ms,
        punctuated=result.punctuated,
    )
