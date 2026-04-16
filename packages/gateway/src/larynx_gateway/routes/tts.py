"""POST /v1/tts — single-shot text-to-speech."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response, status
from larynx_shared.ipc.client_base import WorkerError

from larynx_gateway.auth import require_bearer_token
from larynx_gateway.deps import get_voxcpm_client
from larynx_gateway.schemas.tts import TTSRequest
from larynx_gateway.services import tts_service
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient

router = APIRouter(prefix="/v1", tags=["tts"])
log = structlog.get_logger(__name__)


@router.post(
    "/tts",
    dependencies=[Depends(require_bearer_token)],
    responses={
        200: {"content": {"audio/wav": {}, "audio/L16": {}}},
        400: {"description": "Invalid request"},
        401: {"description": "Missing or invalid bearer token"},
        503: {"description": "TTS worker unavailable"},
    },
)
async def post_tts(
    req: TTSRequest,
    client: VoxCPMClient = Depends(get_voxcpm_client),
) -> Response:
    try:
        result = await tts_service.synthesize(req, client)
    except WorkerError as e:
        if e.code == "invalid_input":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e
        log.error("tts.worker_error", code=e.code, message=e.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"tts worker error: {e.code}",
        ) from e

    headers = {
        "X-Voice-ID": result.voice_id,
        "X-Generation-Time-Ms": str(result.generation_time_ms),
        "X-Audio-Duration-Ms": str(result.duration_ms),
        "X-Sample-Rate": str(result.sample_rate),
    }
    log.info(
        "tts.request",
        chars=len(req.text),
        voice_id=result.voice_id,
        sample_rate=result.sample_rate,
        duration_ms=result.duration_ms,
        generation_time_ms=result.generation_time_ms,
    )
    return Response(content=result.audio, media_type=result.content_type, headers=headers)
