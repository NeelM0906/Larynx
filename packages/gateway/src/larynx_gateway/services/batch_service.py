"""Batch TTS service — persistence + state-machine owner.

Hands out ``job_id``, seeds ``BatchItem`` rows, and owns the
transitions that keep ``BatchJob.num_completed`` / ``num_failed`` in
lockstep with item state. The batch worker calls back into this
module after each item to mark it DONE/FAILED/CANCELLED — doing the
bookkeeping in one place keeps the invariant (counters == row counts
of the matching item state) auditable from a single file.

Queueing (``enqueue_items``) and actual synthesis live elsewhere:
- ``services/batch_queue.py`` owns the Arq pool + job enqueue.
- ``routes/internal_batch.py`` is the loopback HTTP endpoint the Arq
  worker calls when it has GPU budget.

Design: ORCHESTRATION-M8.md §1.1, §1.4.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from larynx_gateway.db.models import BatchItem, BatchJob
from larynx_gateway.schemas.batch import (
    BatchCreateRequest,
    BatchItemRequest,
    BatchItemStatus,
    BatchJobStatus,
)

# Default retention window. Jobs with retain=True skip the cleanup
# sweep entirely and are kept until an admin deletes them.
DEFAULT_RETENTION_DAYS = 7


@dataclass(frozen=True)
class ItemArtifact:
    """What the batch worker reports after finishing a single item."""

    audio_bytes: bytes
    output_format: str
    sample_rate: int
    duration_ms: int
    generation_time_ms: int


def _item_output_path(data_dir: pathlib.Path, job_id: str, item_idx: int, ext: str) -> pathlib.Path:
    """5-digit padded filename so lexical sort == numeric sort."""
    return data_dir / "batch" / job_id / f"{item_idx:05d}.{ext}"


def _now() -> datetime:
    return datetime.now(UTC)


async def create_job(
    session: AsyncSession,
    req: BatchCreateRequest,
) -> str:
    """Persist a BatchJob + N BatchItem rows in a single transaction.

    Returns the new ``job_id``. Does NOT enqueue — the caller hands
    the job_id to ``BatchQueue.enqueue_item()`` once per item after
    commit completes.
    """
    job_id = uuid.uuid4().hex
    created = _now()
    expires = None if req.retain else created + timedelta(days=DEFAULT_RETENTION_DAYS)

    job = BatchJob(
        id=job_id,
        state="QUEUED",
        num_items=len(req.items),
        num_completed=0,
        num_failed=0,
        retain=req.retain,
        created_at=created,
        expires_at=expires,
    )
    session.add(job)

    for idx, item in enumerate(req.items):
        session.add(_build_item_row(job_id, idx, item))

    await session.commit()
    return job_id


def _build_item_row(job_id: str, idx: int, item: BatchItemRequest) -> BatchItem:
    return BatchItem(
        id=uuid.uuid4().hex,
        job_id=job_id,
        item_idx=idx,
        state="QUEUED",
        voice_id=item.voice_id,
        text=item.text,
        params_json=json.dumps(item.params.model_dump()),
    )


async def get_job_status(
    session: AsyncSession,
    job_id: str,
) -> BatchJobStatus | None:
    """Fetch a batch job + its items for GET /v1/batch/{id}."""
    job = (
        await session.execute(select(BatchJob).where(BatchJob.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        return None

    item_rows = (
        (
            await session.execute(
                select(BatchItem)
                .where(BatchItem.job_id == job_id)
                .order_by(BatchItem.item_idx.asc())
            )
        )
        .scalars()
        .all()
    )

    progress = (job.num_completed + job.num_failed) / job.num_items if job.num_items else 0.0
    items = [
        BatchItemStatus(
            idx=row.item_idx,
            state=row.state,
            voice_id=row.voice_id,
            url=f"/v1/batch/{job_id}/items/{row.item_idx}" if row.state == "DONE" else None,
            duration_ms=row.duration_ms,
            generation_time_ms=row.generation_time_ms,
            error_code=row.error_code,
            error_detail=row.error_detail,
        )
        for row in item_rows
    ]

    return BatchJobStatus(
        job_id=job.id,
        state=job.state,
        progress=progress,
        num_items=job.num_items,
        num_completed=job.num_completed,
        num_failed=job.num_failed,
        retain=job.retain,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        expires_at=job.expires_at,
        error_code=job.error_code,
        items=items,
    )


async def cancel_job(session: AsyncSession, job_id: str) -> BatchJob | None:
    """Mark the job + its still-queued items as CANCELLED.

    Already-RUNNING items are left alone and run to completion (per
    the state-machine in ORCHESTRATION-M8.md §1.4). DONE items are
    preserved untouched so their artifacts remain available.
    """
    job = (
        await session.execute(select(BatchJob).where(BatchJob.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        return None

    if job.state in ("COMPLETED", "CANCELLED", "FAILED"):
        return job  # idempotent — no state change needed

    # Flip QUEUED items to CANCELLED; RUNNING items are left alone.
    await session.execute(
        update(BatchItem)
        .where(BatchItem.job_id == job_id, BatchItem.state == "QUEUED")
        .values(state="CANCELLED", finished_at=_now())
    )

    job.state = "CANCELLED"
    job.finished_at = _now()
    await session.commit()
    return job


async def record_item_started(session: AsyncSession, job_id: str, item_idx: int) -> None:
    """Transition item QUEUED → RUNNING + flip the job to RUNNING on first pickup."""
    item = await _lookup_item(session, job_id, item_idx)
    if item is None or item.state != "QUEUED":
        return
    item.state = "RUNNING"
    item.started_at = _now()

    job = (await session.execute(select(BatchJob).where(BatchJob.id == job_id))).scalar_one()
    if job.state == "QUEUED":
        job.state = "RUNNING"
        job.started_at = _now()
    await session.commit()


async def record_item_done(
    session: AsyncSession,
    job_id: str,
    item_idx: int,
    artifact_path: pathlib.Path,
    artifact: ItemArtifact,
) -> None:
    """Write the item as DONE + bump the job counters atomically.

    Artifact bytes are already persisted to disk by the caller — this
    only updates DB state. We hash the bytes for the ETag here rather
    than rely on the caller.
    """
    item = await _lookup_item(session, job_id, item_idx)
    if item is None:
        return
    etag = hashlib.sha256(artifact.audio_bytes).hexdigest()
    item.state = "DONE"
    item.output_path = str(artifact_path)
    item.output_format = artifact.output_format
    item.sample_rate = artifact.sample_rate
    item.duration_ms = artifact.duration_ms
    item.generation_time_ms = artifact.generation_time_ms
    item.etag = etag
    item.finished_at = _now()

    await _bump_counters_and_maybe_finish(session, job_id, delta_done=1, delta_failed=0)


async def record_item_failed(
    session: AsyncSession,
    job_id: str,
    item_idx: int,
    error_code: str,
    error_detail: str,
) -> None:
    item = await _lookup_item(session, job_id, item_idx)
    if item is None:
        return
    item.state = "FAILED"
    item.error_code = error_code
    item.error_detail = error_detail
    item.finished_at = _now()

    await _bump_counters_and_maybe_finish(session, job_id, delta_done=0, delta_failed=1)


async def _bump_counters_and_maybe_finish(
    session: AsyncSession,
    job_id: str,
    *,
    delta_done: int,
    delta_failed: int,
) -> None:
    job = (await session.execute(select(BatchJob).where(BatchJob.id == job_id))).scalar_one()
    job.num_completed += delta_done
    job.num_failed += delta_failed

    if job.num_completed + job.num_failed >= job.num_items:
        # Terminal. FAILED only if literally every item failed;
        # partial failures are surfaced as COMPLETED with
        # num_failed > 0 so the client can retry specific indices
        # without re-submitting a whole job.
        job.state = "FAILED" if job.num_completed == 0 else "COMPLETED"
        job.finished_at = _now()
    await session.commit()


async def _lookup_item(session: AsyncSession, job_id: str, item_idx: int) -> BatchItem | None:
    return (
        await session.execute(
            select(BatchItem).where(BatchItem.job_id == job_id, BatchItem.item_idx == item_idx)
        )
    ).scalar_one_or_none()
