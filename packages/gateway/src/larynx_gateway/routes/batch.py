"""REST routes for /v1/batch.

Thin wrapper over :mod:`services.batch_service`. The synthesis happens
asynchronously in the batch worker (see ``workers/batch_worker.py``);
these routes only own create/list/cancel/status + artifact serving.

See ORCHESTRATION-M8.md §1.2.
"""

from __future__ import annotations

import pathlib

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from larynx_gateway.auth import require_bearer_token
from larynx_gateway.db.models import BatchItem, BatchJob
from larynx_gateway.deps import get_data_dir, get_db_session
from larynx_gateway.schemas.batch import (
    BatchCancelResponse,
    BatchCreateRequest,
    BatchCreateResponse,
    BatchJobStatus,
)
from larynx_gateway.services import batch_service

router = APIRouter(prefix="/v1/batch", tags=["batch"])
log = structlog.get_logger(__name__)


def _batch_queue(request: Request):
    """Look up the queue stashed on app.state during lifespan.

    Imported lazily so tests that don't exercise the queue don't
    have to stand one up.
    """
    queue = getattr(request.app.state, "batch_queue", None)
    return queue


@router.post(
    "",
    response_model=BatchCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_bearer_token)],
)
async def create_batch(
    req: BatchCreateRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> BatchCreateResponse:
    job_id = await batch_service.create_job(session, req)
    queue = _batch_queue(request)
    if queue is not None:
        for idx in range(len(req.items)):
            await queue.enqueue_item(batch_job_id=job_id, item_idx=idx)
    log.info("batch.created", job_id=job_id, num_items=len(req.items), retain=req.retain)
    return BatchCreateResponse(job_id=job_id)


@router.get(
    "/{job_id}",
    response_model=BatchJobStatus,
    dependencies=[Depends(require_bearer_token)],
)
async def get_batch(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> BatchJobStatus:
    status_obj = await batch_service.get_job_status(session, job_id)
    if status_obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "job_not_found", "detail": job_id},
        )
    return status_obj


@router.delete(
    "/{job_id}",
    response_model=BatchCancelResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_bearer_token)],
)
async def cancel_batch(
    job_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> BatchCancelResponse:
    job = await batch_service.cancel_job(session, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "job_not_found", "detail": job_id},
        )
    queue = _batch_queue(request)
    if queue is not None:
        # Purge not-yet-picked-up items from the queue. RUNNING items
        # are already out of the queue and run to completion.
        await queue.cancel_job(batch_job_id=job_id)
    log.info("batch.cancelled", job_id=job_id, state=job.state)
    return BatchCancelResponse(job_id=job_id, state=job.state)


@router.get(
    "/{job_id}/items/{item_idx}",
    dependencies=[Depends(require_bearer_token)],
)
async def get_batch_item_artifact(
    job_id: str,
    item_idx: int,
    data_dir: pathlib.Path = Depends(get_data_dir),
    session: AsyncSession = Depends(get_db_session),
) -> FileResponse:
    """Serve the generated audio for one DONE batch item."""
    # Join against the job to check expiry + retain in one round-trip.
    row = (
        await session.execute(
            select(BatchItem, BatchJob)
            .join(BatchJob, BatchItem.job_id == BatchJob.id)
            .where(BatchItem.job_id == job_id, BatchItem.item_idx == item_idx)
        )
    ).one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "item_not_found", "detail": f"{job_id}/{item_idx}"},
        )
    item, job = row
    if item.state != "DONE" or item.output_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "item_not_ready", "state": item.state},
        )
    # Expired jobs are 410 so clients can tell the difference from a
    # cold-miss 404 (cleanup cron deleted it).
    from datetime import UTC, datetime

    if not job.retain and job.expires_at is not None and job.expires_at < datetime.now(UTC):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail={"code": "item_expired", "expires_at": job.expires_at.isoformat()},
        )

    path = pathlib.Path(item.output_path)
    # output_path may be stored relative to DATA_DIR or absolute — normalise.
    if not path.is_absolute():
        path = (data_dir / path).resolve()
    if not path.is_file():
        # On-disk artifact missing but row says DONE — likely an
        # out-of-band deletion. Surface 410 so clients don't retry
        # indefinitely.
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail={"code": "artifact_missing", "detail": str(path)},
        )

    media_type = _media_type_for(item.output_format or "wav")
    return FileResponse(
        path=path,
        media_type=media_type,
        filename=path.name,
        headers={"ETag": item.etag or ""},
    )


def _media_type_for(fmt: str) -> str:
    return {
        "wav": "audio/wav",
        "pcm16": "audio/L16",
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
    }.get(fmt, "application/octet-stream")
