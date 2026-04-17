"""Five-state job orchestrator — drives FineTuneJob from QUEUED to terminal.

Injectable phase hooks let the tests exercise every transition without
spawning a real training script or calling a real VoxCPMClient. The
hook defaults in production wire up :func:`run_training_subprocess`
and :meth:`VoxCPMClient.load_lora`; here we swap them for async
callables that record calls + simulate outcomes.

Lives in the gateway (not the training_worker) because it touches
the DB models — see ORCHESTRATION-M7.md §5.3: training_worker keeps
pure primitives (dataset_prep, config_builder, subprocess_runner);
DB-aware state-machine orchestration is a gateway service.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
import pytest_asyncio
from larynx_gateway.db.models import FineTuneJob, Voice
from larynx_gateway.db.session import dispose_engine, get_session, init_engine
from larynx_gateway.services.latent_cache import build_redis_client
from larynx_gateway.services.training_logs import TrainingLogStore
from larynx_gateway.services.training_orchestrator import JobRunResult, run_job
from larynx_shared.paths import DatasetPaths, JobPaths, lora_weights_dir
from larynx_training_worker.subprocess_runner import RunnerOutcome
from sqlalchemy import select

from tests.conftest import TEST_DB_URL_SQLA, _ensure_test_db, _reset_test_db

TEST_REDIS_URL = "redis://localhost:6380/15"


# -- fixtures ---------------------------------------------------------------


async def _redis_reachable() -> bool:
    try:
        client = build_redis_client(TEST_REDIS_URL)
        await client.ping()
        await client.aclose()
        return True
    except Exception:
        return False


@pytest_asyncio.fixture
async def redis_client():
    if not await _redis_reachable():
        pytest.skip("Redis not reachable at redis://localhost:6380/15.")
    client = build_redis_client(TEST_REDIS_URL)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest_asyncio.fixture
async def session():
    _ensure_test_db()
    _reset_test_db()
    init_engine(TEST_DB_URL_SQLA)
    try:
        async for s in get_session():
            yield s
    finally:
        await dispose_engine()


@pytest_asyncio.fixture
async def log_store(redis_client) -> TrainingLogStore:
    return TrainingLogStore(redis_client, maxlen=1000, ttl_s=60)


async def _seed_job(session, job_id: str = "j1", voice_name: str = "nimbus") -> FineTuneJob:
    job = FineTuneJob(
        id=job_id,
        name=voice_name,
        dataset_id="ds-1",
        state="QUEUED",
        config_json="{}",
        resolved_config_path="",
        log_key=f"logs:training:{job_id}",
        max_steps=10,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


def _seed_dataset(tmp_path: pathlib.Path, dataset_id: str = "ds-1") -> DatasetPaths:
    import numpy as np
    import soundfile as sf

    dp = DatasetPaths(data_dir=tmp_path, dataset_id=dataset_id)
    dp.audio_dir.mkdir(parents=True)
    for i in range(30):
        samples = np.linspace(-0.3, 0.3, 16_000 * 11, dtype=np.float32)
        sf.write(dp.audio_dir / f"c{i:02d}.wav", samples, 16_000, subtype="PCM_16")
    with dp.transcripts_jsonl.open("w") as fh:
        for i in range(30):
            fh.write(
                json.dumps({"audio": str(dp.audio_dir / f"c{i:02d}.wav"), "text": f"line {i}"})
                + "\n"
            )
    return dp


def _make_subprocess_hook(
    *,
    outcome: RunnerOutcome,
    emit_steps: int = 5,
    delay: float = 0.0,
    wait_for_cancel: bool = False,
    write_artifacts: bool = True,
) -> Callable[..., Awaitable[RunnerOutcome]]:
    """Fake subprocess runner. Records which args it got + optionally
    writes the lora_weights / lora_config artifacts that the real
    runner's success path inspects.
    """

    async def hook(
        *,
        script_path: pathlib.Path,
        job_paths: JobPaths,
        on_log: Callable[[str], None],
        on_state: Callable[[dict[str, Any]], None],
        max_steps: int,
        cancel_event: asyncio.Event,
        wall_timeout_seconds: int = 86_400,
        cancel_grace_seconds: int = 30,
        extra_env: dict[str, str] | None = None,
    ) -> RunnerOutcome:
        del script_path, wall_timeout_seconds, cancel_grace_seconds, extra_env, max_steps
        for step in range(emit_steps):
            on_log(f"step={step} loss/diff={1.0 - 0.1 * step}")
            on_state({"step": step, "loss_diff": 1.0 - 0.1 * step})
            if delay:
                await asyncio.sleep(delay)
            if wait_for_cancel and cancel_event.is_set():
                return RunnerOutcome.CANCELLED
        if wait_for_cancel:
            while not cancel_event.is_set():
                await asyncio.sleep(0.01)
            return RunnerOutcome.CANCELLED
        if write_artifacts and outcome is RunnerOutcome.SUCCESS:
            job_paths.latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            job_paths.latest_lora_weights.write_bytes(b"fake-weights")
            job_paths.latest_lora_config.write_text(
                json.dumps({"base_model": "fake", "lora_config": {"r": 32, "alpha": 32}})
            )
        return outcome

    return hook


# -- happy path -------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_drives_to_succeeded(
    session, log_store: TrainingLogStore, tmp_path: pathlib.Path
) -> None:
    job = await _seed_job(session)
    _seed_dataset(tmp_path)

    voxcpm_calls: list[tuple[str, str]] = []

    async def fake_load_lora(name: str, path: str) -> None:
        voxcpm_calls.append((name, path))

    result = await run_job(
        job_id=job.id,
        session=session,
        log_store=log_store,
        data_dir=tmp_path,
        pretrained_path=str(tmp_path / "pretrained"),
        upstream_script_path=pathlib.Path("/unused/for/test"),
        gpu_lock=asyncio.Lock(),
        cancel_event=asyncio.Event(),
        subprocess_hook=_make_subprocess_hook(outcome=RunnerOutcome.SUCCESS, emit_steps=10),
        load_lora_hook=fake_load_lora,
    )
    assert result is JobRunResult.SUCCEEDED

    # Job row transitioned through all states to SUCCEEDED.
    await session.refresh(job)
    assert job.state == "SUCCEEDED"
    assert job.voice_id is not None
    assert job.started_at is not None
    assert job.finished_at is not None
    assert job.current_step == 9

    # Voice row got written with source='lora' and a lora_path.
    voice = (await session.execute(select(Voice).where(Voice.id == job.voice_id))).scalar_one()
    assert voice.source == "lora"
    assert voice.name == "nimbus"
    assert voice.ft_job_id == job.id
    assert voice.lora_path == str(lora_weights_dir(tmp_path, job.voice_id))

    # LoRA was hot-loaded on the voxcpm worker.
    assert voxcpm_calls == [(job.voice_id, voice.lora_path)]

    # Weights actually landed on disk at the voice-keyed location.
    dest = lora_weights_dir(tmp_path, job.voice_id)
    assert (dest / "lora_weights.safetensors").is_file()
    assert (dest / "lora_config.json").is_file()

    # Redis stream captured subprocess output.
    logs = await log_store.tail(job.id)
    assert any("step=0" in e.line for e in logs)


# -- failure paths ----------------------------------------------------------


@pytest.mark.asyncio
async def test_dataset_missing_fails_preparing(
    session, log_store: TrainingLogStore, tmp_path: pathlib.Path
) -> None:
    # No dataset seeded — PREPARING's Phase A finds nothing and fails.
    job = await _seed_job(session)

    async def never_called_load_lora(name: str, path: str) -> None:
        raise AssertionError("must not hot-load on a PREPARING failure")

    result = await run_job(
        job_id=job.id,
        session=session,
        log_store=log_store,
        data_dir=tmp_path,
        pretrained_path=str(tmp_path / "pretrained"),
        upstream_script_path=pathlib.Path("/unused"),
        gpu_lock=asyncio.Lock(),
        cancel_event=asyncio.Event(),
        subprocess_hook=_make_subprocess_hook(outcome=RunnerOutcome.SUCCESS),
        load_lora_hook=never_called_load_lora,
    )
    assert result is JobRunResult.FAILED

    await session.refresh(job)
    assert job.state == "FAILED"
    assert job.error_code == "dataset_invalid"
    # No Voice row was written.
    assert job.voice_id is None
    voices = (await session.execute(select(Voice))).scalars().all()
    assert voices == []


@pytest.mark.asyncio
async def test_subprocess_nonzero_exit_fails_training(
    session, log_store: TrainingLogStore, tmp_path: pathlib.Path
) -> None:
    job = await _seed_job(session)
    _seed_dataset(tmp_path)

    result = await run_job(
        job_id=job.id,
        session=session,
        log_store=log_store,
        data_dir=tmp_path,
        pretrained_path=str(tmp_path / "pretrained"),
        upstream_script_path=pathlib.Path("/unused"),
        gpu_lock=asyncio.Lock(),
        cancel_event=asyncio.Event(),
        subprocess_hook=_make_subprocess_hook(outcome=RunnerOutcome.NONZERO_EXIT),
        load_lora_hook=lambda name, path: asyncio.sleep(0),
    )
    assert result is JobRunResult.FAILED

    await session.refresh(job)
    assert job.state == "FAILED"
    assert job.error_code == "nonzero_exit"
    assert job.finished_at is not None


@pytest.mark.asyncio
async def test_load_lora_failure_fails_registering(
    session, log_store: TrainingLogStore, tmp_path: pathlib.Path
) -> None:
    job = await _seed_job(session)
    _seed_dataset(tmp_path)

    async def failing_load_lora(name: str, path: str) -> None:
        raise RuntimeError("worker says no")

    result = await run_job(
        job_id=job.id,
        session=session,
        log_store=log_store,
        data_dir=tmp_path,
        pretrained_path=str(tmp_path / "pretrained"),
        upstream_script_path=pathlib.Path("/unused"),
        gpu_lock=asyncio.Lock(),
        cancel_event=asyncio.Event(),
        subprocess_hook=_make_subprocess_hook(outcome=RunnerOutcome.SUCCESS),
        load_lora_hook=failing_load_lora,
    )
    assert result is JobRunResult.FAILED

    await session.refresh(job)
    assert job.state == "FAILED"
    assert job.error_code == "hot_load_rejected"
    # No Voice row was written (§1.1 invariant: CANCELLED/FAILED never
    # leaves a Voice row behind).
    voices = (await session.execute(select(Voice))).scalars().all()
    assert voices == []


# -- cancellation -----------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_before_preparing_transitions_to_cancelled(
    session, log_store: TrainingLogStore, tmp_path: pathlib.Path
) -> None:
    # Set cancel before run_job even starts — must transition straight
    # from QUEUED to CANCELLED without touching the GPU lock, the
    # dataset, or the subprocess.
    job = await _seed_job(session)
    _seed_dataset(tmp_path)

    cancel_event = asyncio.Event()
    cancel_event.set()

    calls_seen: list[str] = []

    async def guarded_load_lora(name: str, path: str) -> None:
        calls_seen.append("load_lora")

    async def guarded_subprocess(**kwargs: object) -> RunnerOutcome:
        calls_seen.append("subprocess")
        return RunnerOutcome.SUCCESS

    result = await run_job(
        job_id=job.id,
        session=session,
        log_store=log_store,
        data_dir=tmp_path,
        pretrained_path=str(tmp_path / "pretrained"),
        upstream_script_path=pathlib.Path("/unused"),
        gpu_lock=asyncio.Lock(),
        cancel_event=cancel_event,
        subprocess_hook=guarded_subprocess,
        load_lora_hook=guarded_load_lora,
    )
    assert result is JobRunResult.CANCELLED
    assert calls_seen == []

    await session.refresh(job)
    assert job.state == "CANCELLED"


@pytest.mark.asyncio
async def test_cancel_during_training_fires_cancel_event(
    session, log_store: TrainingLogStore, tmp_path: pathlib.Path
) -> None:
    job = await _seed_job(session)
    _seed_dataset(tmp_path)

    cancel_event = asyncio.Event()

    async def delayed_cancel() -> None:
        await asyncio.sleep(0.2)
        cancel_event.set()

    hook = _make_subprocess_hook(
        outcome=RunnerOutcome.CANCELLED,
        emit_steps=3,
        wait_for_cancel=True,
        delay=0.05,
    )

    canceller = asyncio.create_task(delayed_cancel())
    result = await run_job(
        job_id=job.id,
        session=session,
        log_store=log_store,
        data_dir=tmp_path,
        pretrained_path=str(tmp_path / "pretrained"),
        upstream_script_path=pathlib.Path("/unused"),
        gpu_lock=asyncio.Lock(),
        cancel_event=cancel_event,
        subprocess_hook=hook,
        load_lora_hook=lambda name, path: asyncio.sleep(0),
    )
    await canceller
    assert result is JobRunResult.CANCELLED

    await session.refresh(job)
    assert job.state == "CANCELLED"
    # Voice row never written.
    assert job.voice_id is None


# -- GPU lock ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_gpu_lock_is_held_during_training(
    session, log_store: TrainingLogStore, tmp_path: pathlib.Path
) -> None:
    # The runner must hold ``gpu_lock`` throughout PREPARING + TRAINING +
    # REGISTERING so two jobs can't overlap on the single GPU. We test
    # that property here on one job by sampling ``gpu_lock.locked()``
    # from inside the subprocess hook; a concurrent-jobs scenario is
    # trivially implied once "lock is held" holds.
    job = await _seed_job(session)
    _seed_dataset(tmp_path)

    gpu_lock = asyncio.Lock()
    lock_held_during_training: list[bool] = []

    async def hook(**kwargs: object) -> RunnerOutcome:
        lock_held_during_training.append(gpu_lock.locked())
        job_paths = kwargs["job_paths"]
        assert isinstance(job_paths, JobPaths)
        job_paths.latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        job_paths.latest_lora_weights.write_bytes(b"x")
        job_paths.latest_lora_config.write_text(
            json.dumps({"base_model": "x", "lora_config": {"r": 32, "alpha": 32}})
        )
        return RunnerOutcome.SUCCESS

    async def fake_load_lora(name: str, path: str) -> None:
        pass

    result = await run_job(
        job_id=job.id,
        session=session,
        log_store=log_store,
        data_dir=tmp_path,
        pretrained_path=str(tmp_path / "pretrained"),
        upstream_script_path=pathlib.Path("/unused"),
        gpu_lock=gpu_lock,
        cancel_event=asyncio.Event(),
        subprocess_hook=hook,
        load_lora_hook=fake_load_lora,
    )
    assert result is JobRunResult.SUCCEEDED
    assert lock_held_during_training == [True]
    # And it gets released so the next job can acquire it.
    assert gpu_lock.locked() is False
