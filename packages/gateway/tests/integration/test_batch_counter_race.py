"""Regression guard for bugs/005 — batch counter read-modify-write race.

The pre-fix ``_bump_counters_and_maybe_finish`` is a SELECT → mutate in
Python → COMMIT cycle without row locking. When two batch consumers
finish items within the same event-loop yield window, both SELECT the
same ``num_completed``, both compute ``+1``, and one UPDATE silently
overwrites the other — classic lost-update race. The end result is a
batch stuck at ``state=RUNNING`` with ``num_completed = num_items - 1``
because the terminal transition is gated on the counter.

This test reproduces the race directly by firing N concurrent bumps
against a seeded job row using the real async SQLAlchemy engine + real
Postgres. Unlike the higher-level ``test_batch_create_and_run`` (which
used to flake ~1-in-3 full-suite runs), this test fails with near-100%
probability pre-fix — the concurrent gather guarantees the event loop
interleaves every SELECT ahead of any COMMIT.

Post-fix: ``_bump_counters_and_maybe_finish`` uses
``UPDATE ... RETURNING`` so Postgres' row lock serializes the
concurrent bumps. Every increment lands; the terminal transition
fires when the post-increment total reaches ``num_items``.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from httpx import AsyncClient
from larynx_gateway.db.models import BatchJob
from larynx_gateway.db.session import get_session_factory
from larynx_gateway.services.batch_service import _bump_counters_and_maybe_finish
from sqlalchemy import select


async def _seed_job(job_id: str, num_items: int) -> None:
    factory = get_session_factory()
    async with factory() as session:
        session.add(
            BatchJob(
                id=job_id,
                state="RUNNING",
                num_items=num_items,
                num_completed=0,
                num_failed=0,
                retain=False,
                created_at=datetime.now(UTC),
                started_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def _read_job(job_id: str) -> BatchJob:
    factory = get_session_factory()
    async with factory() as session:
        return (await session.execute(select(BatchJob).where(BatchJob.id == job_id))).scalar_one()


async def test_concurrent_done_bumps_preserve_total(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Fire N concurrent done-bumps; every increment must land on the row.

    The ``client`` fixture is pulled only for its lifespan side-effect
    — it initialises the engine + session factory.
    """
    n = 20
    job_id = uuid.uuid4().hex
    await _seed_job(job_id, num_items=n)

    async def _bump_done() -> None:
        factory = get_session_factory()
        async with factory() as session:
            await _bump_counters_and_maybe_finish(session, job_id, delta_done=1, delta_failed=0)

    await asyncio.gather(*[_bump_done() for _ in range(n)])

    job = await _read_job(job_id)
    assert job.num_completed == n, (
        f"lost increments: expected num_completed={n}, "
        f"got {job.num_completed} (delta={n - job.num_completed})"
    )


async def test_concurrent_mixed_bumps_transition_to_terminal(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Mix of done + failed bumps must sum to num_items and flip to terminal.

    With 18 done + 2 failed concurrent against num_items=20, the race
    loses at least one increment pre-fix, so the job stays RUNNING.
    Post-fix, ``num_completed + num_failed == num_items`` and the
    state has transitioned to COMPLETED (not FAILED, since
    num_completed > 0).
    """
    n_done = 18
    n_failed = 2
    total = n_done + n_failed
    job_id = uuid.uuid4().hex
    await _seed_job(job_id, num_items=total)

    factory = get_session_factory()

    async def _bump(*, done: bool) -> None:
        async with factory() as session:
            await _bump_counters_and_maybe_finish(
                session,
                job_id,
                delta_done=1 if done else 0,
                delta_failed=0 if done else 1,
            )

    tasks = [_bump(done=True) for _ in range(n_done)] + [_bump(done=False) for _ in range(n_failed)]
    await asyncio.gather(*tasks)

    job = await _read_job(job_id)
    assert job.num_completed == n_done
    assert job.num_failed == n_failed
    assert job.num_completed + job.num_failed == total
    assert job.state == "COMPLETED", f"expected COMPLETED, got {job.state}"
    assert job.finished_at is not None
