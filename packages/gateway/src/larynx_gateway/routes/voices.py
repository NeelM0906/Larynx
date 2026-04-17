"""POST/GET/DELETE /v1/voices + voice-design endpoints.

All routes require the bearer token. PRD §5.5 spec:
  - POST /v1/voices                         multipart upload
  - GET /v1/voices                          paginated list
  - GET /v1/voices/{id}                     metadata
  - GET /v1/voices/{id}/audio               original reference audio
  - DELETE /v1/voices/{id}
  - POST /v1/voices/design                  render preview
  - POST /v1/voices/design/{preview_id}/save  promote to permanent
"""

from __future__ import annotations

import pathlib

import structlog
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from larynx_gateway.auth import require_bearer_token
from larynx_gateway.deps import get_data_dir, get_voice_library
from larynx_gateway.schemas.voice import (
    VoiceDesignPreviewResponse,
    VoiceDesignRequest,
    VoiceDesignSaveRequest,
    VoiceListResponse,
    VoiceResponse,
)
from larynx_gateway.services.voice_files import VoiceFiles
from larynx_gateway.services.voice_library import VoiceLibrary, VoiceLibraryError

router = APIRouter(prefix="/v1/voices", tags=["voices"])
log = structlog.get_logger(__name__)


def _to_response(voice: object) -> VoiceResponse:
    return VoiceResponse.model_validate(voice, from_attributes=True)


def _raise_library(exc: VoiceLibraryError) -> None:
    raise HTTPException(
        status_code=exc.status, detail={"code": exc.code, "message": exc.message}
    ) from exc


@router.post(
    "",
    dependencies=[Depends(require_bearer_token)],
    response_model=VoiceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a reference audio clip as a new voice",
)
async def upload_voice(
    name: str = Form(min_length=1, max_length=128),
    description: str | None = Form(default=None, max_length=500),
    prompt_text: str | None = Form(
        default=None,
        max_length=500,
        description="Optional transcript of the reference — enables "
        "VoxCPM2's ultimate cloning mode at synthesis time.",
    ),
    audio: UploadFile = File(...),
    library: VoiceLibrary = Depends(get_voice_library),
) -> VoiceResponse:
    audio_bytes = await audio.read()
    wav_format = (audio.filename or "").rsplit(".", 1)[-1].lower() if audio.filename else "wav"
    if wav_format not in {"wav", "mp3", "flac", "ogg", "m4a"}:
        wav_format = "wav"
    try:
        uploaded = await library.upload(
            name=name,
            description=description,
            audio=audio_bytes,
            wav_format=wav_format,
            prompt_text=prompt_text,
            source="uploaded",
        )
    except VoiceLibraryError as e:
        _raise_library(e)
    return _to_response(uploaded.voice)


@router.get(
    "",
    dependencies=[Depends(require_bearer_token)],
    response_model=VoiceListResponse,
    summary="List voices",
)
async def list_voices(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    library: VoiceLibrary = Depends(get_voice_library),
) -> VoiceListResponse:
    voices, total = await library.list(limit=limit, offset=offset)
    return VoiceListResponse(
        voices=[_to_response(v) for v in voices],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{voice_id}",
    dependencies=[Depends(require_bearer_token)],
    response_model=VoiceResponse,
    summary="Fetch a single voice",
)
async def get_voice(
    voice_id: str,
    library: VoiceLibrary = Depends(get_voice_library),
) -> VoiceResponse:
    voice = await library.get(voice_id)
    if voice is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": f"no voice with id {voice_id!r}"},
        )
    return _to_response(voice)


@router.get(
    "/{voice_id}/audio",
    dependencies=[Depends(require_bearer_token)],
    summary="Download the reference audio for a voice",
)
async def get_voice_audio(
    voice_id: str,
    library: VoiceLibrary = Depends(get_voice_library),
    data_dir: pathlib.Path = Depends(get_data_dir),
) -> Response:
    voice = await library.get(voice_id)
    if voice is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": f"no voice with id {voice_id!r}"},
        )
    files = VoiceFiles(voice_id=voice_id, root=data_dir)
    if not files.reference_audio.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "audio_missing", "message": "reference audio not on disk"},
        )
    return Response(
        content=files.reference_audio.read_bytes(),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{voice.name}.wav"'},
    )


@router.delete(
    "/{voice_id}",
    dependencies=[Depends(require_bearer_token)],
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a voice + its cached latents + reference audio",
)
async def delete_voice(
    voice_id: str,
    library: VoiceLibrary = Depends(get_voice_library),
) -> Response:
    deleted = await library.delete(voice_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": f"no voice with id {voice_id!r}"},
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Voice design
# ---------------------------------------------------------------------------


@router.post(
    "/design",
    dependencies=[Depends(require_bearer_token)],
    response_model=VoiceDesignPreviewResponse,
    summary="Create a voice from a natural-language description",
)
async def design_preview(
    body: VoiceDesignRequest,
    library: VoiceLibrary = Depends(get_voice_library),
) -> VoiceDesignPreviewResponse:
    try:
        preview = await library.create_design_preview(
            name=body.name,
            description=body.description,
            design_prompt=body.design_prompt,
            preview_text=body.preview_text,
        )
    except VoiceLibraryError as e:
        _raise_library(e)
    return VoiceDesignPreviewResponse(
        preview_id=preview.preview_id,
        expires_in_s=int(preview.expires_at - __import__("time").time()),
        name=preview.name,
        description=preview.description,
        design_prompt=preview.design_prompt,
        preview_text=preview.preview_text,
        sample_rate=preview.sample_rate,
        duration_ms=preview.duration_ms,
    )


@router.get(
    "/design/{preview_id}/audio",
    dependencies=[Depends(require_bearer_token)],
    summary="Fetch the preview audio for a design before saving",
)
async def design_preview_audio(
    preview_id: str,
    library: VoiceLibrary = Depends(get_voice_library),
) -> Response:
    files = library.load_design_preview(preview_id)
    if files is None or not files.preview_audio.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "preview_not_found", "message": preview_id},
        )
    return Response(content=files.preview_audio.read_bytes(), media_type="audio/wav")


@router.post(
    "/design/{preview_id}/save",
    dependencies=[Depends(require_bearer_token)],
    response_model=VoiceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Commit a design preview as a permanent voice",
)
async def save_design(
    preview_id: str,
    body: VoiceDesignSaveRequest | None = None,
    library: VoiceLibrary = Depends(get_voice_library),
) -> VoiceResponse:
    try:
        uploaded = await library.save_design(
            preview_id,
            name_override=body.name if body else None,
            description_override=body.description if body else None,
        )
    except VoiceLibraryError as e:
        _raise_library(e)
    return _to_response(uploaded.voice)
