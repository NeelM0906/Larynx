"""Integration test for migration 0003 + FineTuneJob model.

Exercises the real Postgres schema (via the conftest fixture that
creates the test DB and runs ``alembic upgrade head``). Covers:

- Voice gains ``lora_path`` / ``ft_job_id`` (nullable, indexed).
- FineTuneJob round-trips through a real async session with the
  columns and defaults ORCHESTRATION-M7.md §4.2 calls for.
- Voice.source accepts the new ``'lora'`` value end-to-end.

No fakes — the test uses the same migration path production will see.
"""

from __future__ import annotations

import pathlib

import pytest
import pytest_asyncio
from larynx_gateway.db.models import FineTuneJob, Voice
from larynx_gateway.db.session import dispose_engine, get_session, init_engine
from sqlalchemy import inspect, select, text

from tests.conftest import TEST_DB_URL_SQLA, _ensure_test_db, _reset_test_db


@pytest_asyncio.fixture
async def session(tmp_path: pathlib.Path):  # noqa: ARG001 — tmp_path kept for parity
    _ensure_test_db()
    _reset_test_db()
    init_engine(TEST_DB_URL_SQLA)
    try:
        async for s in get_session():
            yield s
    finally:
        await dispose_engine()


@pytest.mark.asyncio
async def test_voices_table_gained_lora_columns(session) -> None:
    def _inspect(sync_conn):
        insp = inspect(sync_conn)
        cols = {c["name"] for c in insp.get_columns("voices")}
        idx = {i["name"] for i in insp.get_indexes("voices")}
        return cols, idx

    conn = await session.connection()
    cols, idx = await conn.run_sync(_inspect)
    assert "lora_path" in cols
    assert "ft_job_id" in cols
    assert "idx_voices_ft_job_id" in idx


@pytest.mark.asyncio
async def test_fine_tune_jobs_table_exists(session) -> None:
    def _inspect(sync_conn):
        insp = inspect(sync_conn)
        tables = set(insp.get_table_names())
        if "fine_tune_jobs" not in tables:
            return tables, set(), set()
        cols = {c["name"] for c in insp.get_columns("fine_tune_jobs")}
        idx = {i["name"] for i in insp.get_indexes("fine_tune_jobs")}
        return tables, cols, idx

    conn = await session.connection()
    tables, cols, idx = await conn.run_sync(_inspect)
    assert "fine_tune_jobs" in tables, f"missing table; have {tables}"
    # Spot-check all the design-doc-§4.2 columns. Misses here mean the
    # migration drifted from the design.
    expected_cols = {
        "id",
        "name",
        "dataset_id",
        "state",
        "voice_id",
        "config_json",
        "resolved_config_path",
        "log_key",
        "error_code",
        "error_detail",
        "current_step",
        "max_steps",
        "created_at",
        "started_at",
        "finished_at",
    }
    assert expected_cols <= cols, f"missing {expected_cols - cols}"
    assert "idx_ftjobs_state" in idx
    assert "idx_ftjobs_voice_id" in idx


@pytest.mark.asyncio
async def test_fine_tune_job_round_trip(session) -> None:
    job = FineTuneJob(
        id="00000000-0000-0000-0000-0000000000aa",
        name="test-voice",
        dataset_id="ds-42",
        state="QUEUED",
        config_json='{"lora_rank": 32}',
        resolved_config_path="/tmp/job/train_config.yaml",
        log_key="logs:training:00000000-0000-0000-0000-0000000000aa",
        max_steps=1000,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    # Server defaults land.
    assert job.created_at is not None
    assert job.current_step == 0
    assert job.error_code is None
    assert job.voice_id is None
    assert job.started_at is None
    assert job.finished_at is None

    # And it's queryable back by state.
    found = (
        await session.execute(select(FineTuneJob).where(FineTuneJob.state == "QUEUED"))
    ).scalar_one()
    assert found.id == job.id


@pytest.mark.asyncio
async def test_voice_source_lora_round_trip(session) -> None:
    voice = Voice(
        id="00000000-0000-0000-0000-0000000000bb",
        name="lora-voice-test",
        source="lora",
        lora_path="/data/lora_weights/00000000-0000-0000-0000-0000000000bb",
        ft_job_id="00000000-0000-0000-0000-0000000000aa",
    )
    session.add(voice)
    await session.commit()
    await session.refresh(voice)

    found = (
        await session.execute(select(Voice).where(Voice.name == "lora-voice-test"))
    ).scalar_one()
    assert found.source == "lora"
    assert found.lora_path == "/data/lora_weights/00000000-0000-0000-0000-0000000000bb"
    assert found.ft_job_id == "00000000-0000-0000-0000-0000000000aa"


@pytest.mark.asyncio
async def test_migration_head_is_0003(session) -> None:
    # The current migration head must be the M7 revision. This catches
    # "someone forgot to add a new revision downstream of 0003" drift.
    row = (await session.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
    assert row == "0003_finetune_artifacts"
