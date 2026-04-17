"""Boot-time reconciliation between DB state and runtime state.

Two tasks run at gateway startup, after the voxcpm_worker is ``ready``
and before the HTTP server starts accepting traffic:

1. **Hot-load every LoRA voice** — DB rows with ``source='lora'`` must
   be registered with the voxcpm_worker so ``/v1/tts`` can synthesise
   them immediately; the nanovllm engine only knows what we've told
   it since process start.

2. **Reap orphan jobs** — any ``FineTuneJob`` in a non-terminal state
   with a timestamp older than ``ORPHAN_JOB_GRACE`` was owned by a
   previous gateway process that crashed; the CUDA subprocess is
   gone. Mark these ``FAILED(gateway_restarted)`` so stale rows don't
   lock up the training UI.

See ORCHESTRATION-M7.md §1.1 (E-LoRA-2) and §8.5.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from larynx_gateway.db.models import FineTuneJob, Voice
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient

log = structlog.get_logger(__name__)

# States from which the orphan reaper can transition a job directly to
# FAILED. Terminal states (SUCCEEDED / FAILED / CANCELLED) are excluded
# — we never rewrite those.
_NON_TERMINAL_STATES: tuple[str, ...] = ("QUEUED", "PREPARING", "TRAINING", "REGISTERING")


async def load_lora_voices(
    session: AsyncSession,
    voxcpm_client: VoxCPMClient,
) -> dict[str, bool]:
    """Fire ``LoadLoraRequest`` for every ``Voice`` with ``source='lora'``.

    Returns a mapping ``{voice_id: True_if_loaded}`` so the caller can
    stash failed ones on ``app.state`` for the /v1/voices response. A
    failure for one voice never prevents the others from loading — we
    fan out in parallel via ``asyncio.gather(..., return_exceptions=True)``.
    """
    rows = (await session.execute(select(Voice).where(Voice.source == "lora"))).scalars().all()
    if not rows:
        log.info("boot.lora_reconcile.empty")
        return {}

    async def _load_one(voice: Voice) -> tuple[str, bool]:
        if not voice.lora_path:
            log.warning(
                "boot.lora_reconcile.missing_path",
                voice_id=voice.id,
                name=voice.name,
            )
            return voice.id, False
        try:
            await voxcpm_client.load_lora(voice.id, voice.lora_path)
            return voice.id, True
        except Exception as e:  # noqa: BLE001
            log.warning(
                "boot.lora_reconcile.load_failed",
                voice_id=voice.id,
                name=voice.name,
                path=voice.lora_path,
                error=str(e),
            )
            return voice.id, False

    results = await asyncio.gather(*(_load_one(v) for v in rows))
    status = dict(results)
    log.info(
        "boot.lora_reconcile.done",
        total=len(rows),
        loaded=sum(1 for ok in status.values() if ok),
        failed=sum(1 for ok in status.values() if not ok),
    )
    return status


async def reap_orphan_jobs(
    session: AsyncSession,
    grace_seconds: int,
) -> int:
    """Mark non-terminal jobs older than ``grace_seconds`` as FAILED.

    "Older than" uses ``started_at`` when present and falls back to
    ``created_at`` for QUEUED rows that never transitioned. Jobs still
    within the grace window are left alone — a human operator might be
    mid-restart and legitimately expect their job to reappear.

    Returns the number of rows updated.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=grace_seconds)
    # started_at < cutoff OR (started_at IS NULL AND created_at < cutoff).
    stmt = (
        update(FineTuneJob)
        .where(FineTuneJob.state.in_(_NON_TERMINAL_STATES))
        .where(
            or_(
                FineTuneJob.started_at < cutoff,
                (FineTuneJob.started_at.is_(None)) & (FineTuneJob.created_at < cutoff),
            )
        )
        .values(
            state="FAILED",
            error_code="gateway_restarted",
            error_detail=(
                "Gateway restarted while the job was in flight; the training "
                "subprocess was orphaned and has been reaped."
            ),
            finished_at=func.now(),
        )
    )
    result = await session.execute(stmt)
    count = result.rowcount or 0
    await session.commit()
    if count:
        log.warning("boot.orphan_reap", count=count, grace_seconds=grace_seconds)
    else:
        log.info("boot.orphan_reap.empty", grace_seconds=grace_seconds)
    return count
