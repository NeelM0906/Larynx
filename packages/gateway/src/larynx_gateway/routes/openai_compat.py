"""OpenAI-compatible audio endpoints (PRD §5.9).

Only ``POST /v1/audio/transcriptions`` is implemented in M3. The
``/v1/audio/speech`` TTS shim lands in M8.

Contract matches OpenAI's Whisper API:

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
"""

from __future__ import annotations

from typing import Literal

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse
from larynx_shared.ipc.client_base import WorkerError

from larynx_gateway.auth import require_bearer_token
from larynx_gateway.deps import get_funasr_client, get_vad_punc_client
from larynx_gateway.schemas.stt import OpenAITranscriptionResponse
from larynx_gateway.services import stt_service
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient

router = APIRouter(prefix="/v1", tags=["openai-compat"])
log = structlog.get_logger(__name__)


ResponseFormat = Literal["json", "text"]


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
