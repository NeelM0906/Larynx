"""POST /v1/tts — text-to-speech with optional voice cloning.

Accepts JSON for text-only / voice_id synthesis and multipart/form-data
for ad-hoc `reference_audio` / `prompt_audio` uploads. The two shapes
share the same route so clients can send whichever matches their setup.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from larynx_shared.ipc.client_base import WorkerError
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from larynx_gateway.auth import require_bearer_token
from larynx_gateway.deps import get_voice_library, get_voxcpm_client
from larynx_gateway.schemas.tts import TTSRequest
from larynx_gateway.services import tts_service
from larynx_gateway.services.voice_library import VoiceLibrary
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
        404: {"description": "voice_id not found"},
        503: {"description": "TTS worker unavailable"},
    },
)
async def post_tts(
    request: Request,
    client: VoxCPMClient = Depends(get_voxcpm_client),
    library: VoiceLibrary = Depends(get_voice_library),
) -> Response:
    req, inline_ref, inline_prompt, inline_prompt_text = await _parse_request(request)

    try:
        conditioning = await tts_service.resolve_conditioning(
            req,
            library,
            inline_reference_audio=inline_ref,
            inline_prompt_audio=inline_prompt,
            inline_prompt_text=inline_prompt_text,
            voxcpm=client,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    if conditioning is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "voice_not_found", "message": f"voice_id={req.voice_id!r}"},
        )

    try:
        result = await tts_service.synthesize(req, conditioning, client)
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
        has_ref_latents=conditioning.ref_audio_latents is not None,
        has_prompt_latents=conditioning.prompt_audio_latents is not None,
    )
    return Response(content=result.audio, media_type=result.content_type, headers=headers)


async def _parse_request(
    request: Request,
) -> tuple[TTSRequest, bytes | None, bytes | None, str | None]:
    """Pick the right parse path based on Content-Type.

    FastAPI's Form/File/Body defaults can't handle "either JSON body or
    multipart" on one route, so we route manually off the Content-Type.
    """
    content_type = (request.headers.get("content-type") or "").lower()

    if content_type.startswith("multipart/form-data"):
        return await _parse_multipart(request)

    # Default: JSON.
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"invalid JSON body: {e}"
        ) from e
    try:
        req = TTSRequest.model_validate(body)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors()
        ) from e
    return req, None, None, None


async def _parse_multipart(
    request: Request,
) -> tuple[TTSRequest, bytes | None, bytes | None, str | None]:
    form = await request.form()

    # Text / engine knobs from form fields.
    try:
        req = TTSRequest.model_validate(
            {
                k: v
                for k, v in form.items()
                if k
                in {
                    "text",
                    "voice_id",
                    "prompt_text",
                    "sample_rate",
                    "output_format",
                    "cfg_value",
                    "temperature",
                    "language",
                }
                and isinstance(v, str)
            }
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors()
        ) from e

    reference_audio_bytes: bytes | None = None
    ref = form.get("reference_audio")
    if isinstance(ref, UploadFile):
        reference_audio_bytes = await ref.read() or None

    prompt_audio_bytes: bytes | None = None
    pa = form.get("prompt_audio")
    if isinstance(pa, UploadFile):
        prompt_audio_bytes = await pa.read() or None

    inline_prompt_text = form.get("prompt_text")
    inline_prompt_text = (
        inline_prompt_text if isinstance(inline_prompt_text, str) and inline_prompt_text else None
    )

    return req, reference_audio_bytes, prompt_audio_bytes, inline_prompt_text
