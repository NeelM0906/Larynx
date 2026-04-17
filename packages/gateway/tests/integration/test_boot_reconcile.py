"""Boot-time reconciliation: hot-load LoRA voices + reap orphan jobs.

Per ORCHESTRATION-M7.md §8.5. Tests exercise real Postgres + real
VoxCPMClient (over WorkerChannel + MockVoxCPMBackend) — no fakes.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from larynx_gateway.db.models import FineTuneJob, Voice
from larynx_gateway.db.session import dispose_engine, get_session, init_engine
from larynx_gateway.services.boot_reconcile import (
    load_lora_voices,
    reap_orphan_jobs,
)
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient
from larynx_shared.ipc import WorkerChannel
from larynx_voxcpm_worker.model_manager import MockVoxCPMBackend, VoxCPMModelManager
from larynx_voxcpm_worker.server import WorkerServer
from sqlalchemy import select

from tests.conftest import TEST_DB_URL_SQLA, _ensure_test_db, _reset_test_db


@pytest_asyncio.fixture
async def session(tmp_path: pathlib.Path):  # noqa: ARG001
    _ensure_test_db()
    _reset_test_db()
    init_engine(TEST_DB_URL_SQLA)
    try:
        async for s in get_session():
            yield s
    finally:
        await dispose_engine()


@pytest_asyncio.fixture
async def voxcpm_client():
    manager = VoxCPMModelManager(MockVoxCPMBackend())
    channel = WorkerChannel()
    server = WorkerServer(channel, manager)
    client = VoxCPMClient(channel)
    await client.start()
    await server.start()
    try:
        yield client
    finally:
        await server.stop()
        await client.stop()


# -- load_lora_voices ------------------------------------------------------


@pytest.mark.asyncio
async def test_load_lora_voices_registers_all(session, voxcpm_client) -> None:
    # Seed two LoRA voices + one non-LoRA (which must be ignored).
    session.add_all(
        [
            Voice(id="v1", name="lora-one", source="lora", lora_path="/tmp/one"),
            Voice(id="v2", name="lora-two", source="lora", lora_path="/tmp/two"),
            Voice(id="v3", name="uploaded-ignored", source="uploaded"),
        ]
    )
    await session.commit()

    status = await load_lora_voices(session, voxcpm_client)
    assert status == {"v1": True, "v2": True}

    # Engine confirms both are registered.
    assert sorted(await voxcpm_client.list_loras()) == ["v1", "v2"]


@pytest.mark.asyncio
async def test_load_lora_voices_no_lora_rows_is_noop(session, voxcpm_client) -> None:
    session.add(Voice(id="v1", name="uploaded", source="uploaded"))
    await session.commit()

    status = await load_lora_voices(session, voxcpm_client)
    assert status == {}
    assert await voxcpm_client.list_loras() == []


@pytest.mark.asyncio
async def test_load_lora_voices_missing_path_marks_unloaded(session, voxcpm_client) -> None:
    # A voice with source='lora' but no lora_path is a DB inconsistency
    # from manual intervention or an upgrade glitch. We log + mark
    # unloaded rather than refuse to boot.
    session.add(Voice(id="v1", name="bad-lora", source="lora", lora_path=None))
    await session.commit()

    status = await load_lora_voices(session, voxcpm_client)
    assert status == {"v1": False}
    assert await voxcpm_client.list_loras() == []


@pytest.mark.asyncio
async def test_load_lora_voices_duplicate_name_does_not_fail_others(session, voxcpm_client) -> None:
    # Pre-register v1 so its load fails (duplicate), then the reconciler
    # should still load v2.
    await voxcpm_client.load_lora("v1", "/tmp/one")
    session.add_all(
        [
            Voice(id="v1", name="lora-one", source="lora", lora_path="/tmp/one"),
            Voice(id="v2", name="lora-two", source="lora", lora_path="/tmp/two"),
        ]
    )
    await session.commit()

    status = await load_lora_voices(session, voxcpm_client)
    assert status == {"v1": False, "v2": True}
    # v1 stays registered from the pre-seed; v2 is newly loaded.
    assert sorted(await voxcpm_client.list_loras()) == ["v1", "v2"]


# -- reap_orphan_jobs ------------------------------------------------------


@pytest.mark.asyncio
async def test_reap_orphan_jobs_marks_stale_training_as_failed(session) -> None:
    now = datetime.now(UTC)
    stale_started = now - timedelta(seconds=120)
    session.add_all(
        [
            FineTuneJob(
                id="j1",
                name="stale",
                dataset_id="ds1",
                state="TRAINING",
                config_json="{}",
                resolved_config_path="/tmp/j1/config.yaml",
                log_key="logs:training:j1",
                max_steps=1000,
                started_at=stale_started,
            ),
            FineTuneJob(
                id="j2",
                name="just-started",
                dataset_id="ds1",
                state="TRAINING",
                config_json="{}",
                resolved_config_path="/tmp/j2/config.yaml",
                log_key="logs:training:j2",
                max_steps=1000,
                started_at=now,  # within grace
            ),
        ]
    )
    await session.commit()

    count = await reap_orphan_jobs(session, grace_seconds=60)
    assert count == 1

    j1 = (await session.execute(select(FineTuneJob).where(FineTuneJob.id == "j1"))).scalar_one()
    j2 = (await session.execute(select(FineTuneJob).where(FineTuneJob.id == "j2"))).scalar_one()
    assert j1.state == "FAILED"
    assert j1.error_code == "gateway_restarted"
    assert j1.finished_at is not None
    assert j2.state == "TRAINING"  # within grace — untouched


@pytest.mark.asyncio
async def test_reap_orphan_jobs_ignores_terminal_states(session) -> None:
    old = datetime.now(UTC) - timedelta(hours=1)
    session.add_all(
        [
            FineTuneJob(
                id=f"j{i}",
                name=f"done-{i}",
                dataset_id="ds1",
                state=state,
                config_json="{}",
                resolved_config_path=f"/tmp/j{i}/config.yaml",
                log_key=f"logs:training:j{i}",
                max_steps=1000,
                started_at=old,
                finished_at=old,
            )
            for i, state in enumerate(("SUCCEEDED", "FAILED", "CANCELLED"))
        ]
    )
    await session.commit()

    count = await reap_orphan_jobs(session, grace_seconds=60)
    assert count == 0


@pytest.mark.asyncio
async def test_reap_orphan_jobs_reaps_queued_and_preparing_and_registering(
    session,
) -> None:
    old = datetime.now(UTC) - timedelta(minutes=10)
    session.add_all(
        [
            FineTuneJob(
                id=f"j{i}",
                name=f"stale-{i}",
                dataset_id="ds1",
                state=state,
                config_json="{}",
                resolved_config_path=f"/tmp/j{i}/config.yaml",
                log_key=f"logs:training:j{i}",
                max_steps=1000,
                started_at=old,
            )
            for i, state in enumerate(("QUEUED", "PREPARING", "REGISTERING"))
        ]
    )
    await session.commit()

    count = await reap_orphan_jobs(session, grace_seconds=60)
    assert count == 3
    rows = (await session.execute(select(FineTuneJob))).scalars().all()
    assert {r.state for r in rows} == {"FAILED"}


@pytest.mark.asyncio
async def test_reap_orphan_jobs_respects_created_at_when_started_at_null(
    session,
) -> None:
    # Pure-QUEUED jobs may not have started_at yet. Orphan reaper must
    # fall back to created_at, otherwise QUEUED rows would never age out.
    # We can't easily forge created_at (server_default is NOW()) so this
    # test just verifies that a fresh QUEUED row is NOT reaped.
    session.add(
        FineTuneJob(
            id="j1",
            name="queued-fresh",
            dataset_id="ds1",
            state="QUEUED",
            config_json="{}",
            resolved_config_path="/tmp/j1/config.yaml",
            log_key="logs:training:j1",
            max_steps=1000,
        )
    )
    await session.commit()

    count = await reap_orphan_jobs(session, grace_seconds=60)
    assert count == 0
    j1 = (await session.execute(select(FineTuneJob).where(FineTuneJob.id == "j1"))).scalar_one()
    assert j1.state == "QUEUED"
