"""Daily cleanup of expired batch artifacts.

Scheduled by ``workers/cleanup_cron.py``. Runs inside the gateway
process on a simple ``asyncio.sleep`` timer — no Arq cron needed.

Scope:
1. Delete ``BatchJob`` rows where ``retain=False`` and
   ``expires_at < now()``. The FK cascade handles the items.
2. Remove the on-disk artifact tree for each deleted job.
3. Orphan-sweep ``${DATA_DIR}/batch/*`` for subdirectories whose
   job_id isn't in the DB (paranoia; covers any leftover state from
   a crash mid-write).

See ORCHESTRATION-M8.md §3.5.
"""

from __future__ import annotations

import pathlib
import shutil
from datetime import UTC, datetime

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from larynx_gateway.db.models import BatchJob

log = structlog.get_logger(__name__)


async def run_cleanup(
    session: AsyncSession,
    data_dir: pathlib.Path,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    """Execute one cleanup pass. Returns a counts summary for logging."""
    now = now or datetime.now(UTC)

    expired_ids = (
        (
            await session.execute(
                select(BatchJob.id).where(
                    BatchJob.retain.is_(False),
                    BatchJob.expires_at.is_not(None),
                    BatchJob.expires_at < now,
                )
            )
        )
        .scalars()
        .all()
    )

    # Delete DB rows first so the FK cascade drops items. Files come
    # next — an interrupted sweep leaves some orphan directories
    # that the next pass catches.
    deleted_rows = 0
    for job_id in expired_ids:
        await session.execute(delete(BatchJob).where(BatchJob.id == job_id))
        deleted_rows += 1
    await session.commit()

    deleted_dirs = 0
    for job_id in expired_ids:
        job_dir = data_dir / "batch" / job_id
        if job_dir.is_dir():
            shutil.rmtree(job_dir, ignore_errors=True)
            deleted_dirs += 1

    # Orphan pass: any on-disk job dir whose row is gone.
    orphan_dirs = 0
    batch_root = data_dir / "batch"
    if batch_root.is_dir():
        live_ids = {row for row in ((await session.execute(select(BatchJob.id))).scalars().all())}
        for child in batch_root.iterdir():
            if child.is_dir() and child.name not in live_ids:
                shutil.rmtree(child, ignore_errors=True)
                orphan_dirs += 1

    log.info(
        "batch.cleanup",
        deleted_rows=deleted_rows,
        deleted_dirs=deleted_dirs,
        orphan_dirs=orphan_dirs,
    )
    return {
        "deleted_rows": deleted_rows,
        "deleted_dirs": deleted_dirs,
        "orphan_dirs": orphan_dirs,
    }
