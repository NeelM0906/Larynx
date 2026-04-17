"""POST /v1/finetune/* — LoRA fine-tuning endpoints.

This module ships the dataset-upload route first; job create / get /
delete / SSE logs land in follow-on commits.

Spec: PRD §5.8 + ORCHESTRATION-M7.md.
"""

from __future__ import annotations

import pathlib
import shutil
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from larynx_shared.paths import SUPPORTED_AUDIO_SUFFIXES, DatasetPaths
from larynx_training_worker.dataset_prep import (
    validate_dataset_phase_a,
)

from larynx_gateway.auth import require_bearer_token
from larynx_gateway.deps import get_data_dir
from larynx_gateway.schemas.finetune import DatasetUploadResponse

router = APIRouter(prefix="/v1/finetune", tags=["finetune"])
log = structlog.get_logger(__name__)


_MANIFEST_FILENAME = "transcripts.jsonl"


def _safe_filename(raw: str) -> str:
    """Reject path-traversal attempts + normalise to a bare filename.

    Upstream tolerates absolute paths in transcripts.jsonl but the
    upload API must never write outside the dataset dir.
    """
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_filename", "detail": "empty filename"},
        )
    candidate = pathlib.PurePosixPath(raw.replace("\\", "/")).name
    if raw != candidate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "bad_filename", "detail": f"{raw!r} contains path separators"},
        )
    return candidate


@router.post(
    "/datasets",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_bearer_token)],
)
async def upload_dataset(
    files: list[UploadFile],
    data_dir: pathlib.Path = Depends(get_data_dir),
) -> DatasetUploadResponse:
    """Multipart upload of audio + optional ``transcripts.jsonl``.

    Files land in ``{DATA_DIR}/datasets/{dataset_id}.staging/`` first.
    Phase-A validation runs there; on success the dir is renamed to
    the final path so a partially-valid dataset is never visible to
    the training_worker. Failure cleans up the staging dir.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "no_files", "detail": "at least one file is required"},
        )

    dataset_id = uuid.uuid4().hex
    dataset = DatasetPaths(data_dir=data_dir, dataset_id=dataset_id)
    staging = dataset.staging_dir
    staging_audio = staging / "audio"
    staging_manifest = staging / _MANIFEST_FILENAME
    staging_audio.mkdir(parents=True, exist_ok=True)

    try:
        for upload in files:
            filename = _safe_filename(upload.filename or "")
            if filename == _MANIFEST_FILENAME:
                await _write_upload(upload, staging_manifest)
                continue
            if pathlib.Path(filename).suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "unsupported_file",
                        "detail": (
                            f"{filename!r}: only {sorted(SUPPORTED_AUDIO_SUFFIXES)} "
                            f"+ {_MANIFEST_FILENAME!r} are accepted"
                        ),
                    },
                )
            await _write_upload(upload, staging_audio / filename)
    except HTTPException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(staging, ignore_errors=True)
        log.exception("finetune.upload_failed", dataset_id=dataset_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "upload_failed", "detail": str(e)},
        ) from e

    # Phase A runs against the staging dir — it takes a DatasetPaths so
    # we build one pointing at the staging layout.
    staging_dataset = _DatasetPathsView(data_dir=data_dir, dataset_id=dataset_id, staging=True)
    report = validate_dataset_phase_a(staging_dataset)
    if not report.ok:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "dataset_invalid",
                "detail": "dataset failed Phase A validation",
                "issues": [i.model_dump() for i in report.issues],
                "num_clips": report.num_clips,
                "total_duration_s": report.total_duration_s,
            },
        )

    # Atomic promote: rename staging/ -> {dataset_id}/. Using shutil.move
    # here because the parent might exist from a prior upload (not our
    # case — dataset_id is freshly uuid4'd — but the directory layout
    # doesn't let us rely on target-absent guarantees in general).
    dataset.base_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(dataset.base_dir))

    log.info(
        "finetune.dataset_uploaded",
        dataset_id=dataset_id,
        num_clips=report.num_clips,
        total_duration_s=report.total_duration_s,
    )
    return DatasetUploadResponse(dataset_id=dataset_id, report=report)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _write_upload(upload: UploadFile, dest: pathlib.Path) -> None:
    """Stream an UploadFile to disk with a bounded chunk size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


class _DatasetPathsView(DatasetPaths):
    """Variant that points at ``staging_dir`` as if it were ``base_dir``.

    Lets :func:`validate_dataset_phase_a` run against the upload while
    it's still in the staging directory. ``DatasetPaths`` is a frozen
    dataclass — we override the property instead of mutating state.
    """

    def __init__(self, *, data_dir: pathlib.Path | str, dataset_id: str, staging: bool) -> None:
        super().__init__(data_dir=data_dir, dataset_id=dataset_id)
        # Uses ``__setattr__`` escape hatch from the frozen dataclass to
        # stash a discriminator; actual override is in ``base_dir``.
        object.__setattr__(self, "_staging_view", staging)

    @property
    def base_dir(self) -> pathlib.Path:  # type: ignore[override]
        if getattr(self, "_staging_view", False):
            return self.staging_dir
        return super().base_dir
