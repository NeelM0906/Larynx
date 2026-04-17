"""Five-state fine-tune job runner.

Drives a ``FineTuneJob`` row from ``QUEUED`` to a terminal state
(``SUCCEEDED`` / ``FAILED`` / ``CANCELLED``) via the state machine in
ORCHESTRATION-M7.md §1. Owns none of the individual phase work —
Phase A validation, config building, subprocess driving, and LoRA
hot-swap all live in their own modules; the runner ties them together
and owns the DB-row transitions + cancellation + GPU-lock ordering.

Two hooks are parameter-injectable so tests can drive every transition
without needing a real training script or a real VoxCPMClient:

- ``subprocess_hook`` — defaults to :func:`run_training_subprocess`.
- ``load_lora_hook`` — defaults to the provided VoxCPMClient's
  ``load_lora``.

Hooks must preserve the real contract: ``subprocess_hook`` calls
``on_log`` / ``on_state`` during its run and returns a
:class:`RunnerOutcome`; ``load_lora_hook`` raises on failure.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import structlog
from larynx_shared.paths import DatasetPaths, JobPaths, lora_weights_dir
from larynx_training_worker.config_builder import (
    build_training_config,
    write_training_config,
)
from larynx_training_worker.dataset_prep import (
    auto_transcribe_if_missing,
    normalise_manifest_paths,
    validate_dataset_phase_a,
    validate_transcripts_phase_b,
)
from larynx_training_worker.subprocess_runner import (
    RunnerOutcome,
    run_training_subprocess,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from larynx_gateway.db.models import FineTuneJob, Voice
from larynx_gateway.services.training_logs import TrainingLogStore

log = structlog.get_logger(__name__)


class JobRunResult(StrEnum):
    """Terminal outcome the caller maps to a user-facing response."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Map RunnerOutcome (subprocess-level) -> FineTuneJob.error_code
# (job-level). Separate enums because the subprocess outcome doesn't
# know what happened outside of its own span (e.g. a DB write can
# still fail after a clean exit).
_SUBPROCESS_ERROR_CODES: dict[RunnerOutcome, str] = {
    RunnerOutcome.NONZERO_EXIT: "nonzero_exit",
    RunnerOutcome.MISSING_ARTIFACT: "missing_artifact",
    RunnerOutcome.BAD_LORA_CONFIG: "bad_lora_config",
    RunnerOutcome.EARLY_EXIT: "early_exit",
    RunnerOutcome.WALL_TIMEOUT: "wall_timeout",
}


SubprocessHook = Callable[..., Awaitable[RunnerOutcome]]
LoadLoraHook = Callable[[str, str], Awaitable[None]]
TranscribeHook = Callable[[bytes, int], Awaitable[str]]


async def run_job(
    *,
    job_id: str,
    session: AsyncSession,
    log_store: TrainingLogStore,
    data_dir: pathlib.Path,
    pretrained_path: str,
    upstream_script_path: pathlib.Path,
    gpu_lock: asyncio.Lock,
    cancel_event: asyncio.Event,
    subprocess_hook: SubprocessHook | None = None,
    load_lora_hook: LoadLoraHook | None = None,
    transcribe_hook: TranscribeHook | None = None,
    min_seconds: int = 300,
    wall_timeout_seconds: int = 86_400,
    cancel_grace_seconds: int = 30,
) -> JobRunResult:
    """Run the five-state machine for ``job_id`` to a terminal state.

    The job row must already exist in ``state='QUEUED'``. Every state
    transition commits the row with the new state + relevant fields
    (``current_step``, ``error_code``, ``started_at``, ``finished_at``,
    ``voice_id``) so the gateway's polling route sees progress even
    while the subprocess is running.
    """
    subprocess_hook = subprocess_hook or run_training_subprocess
    if load_lora_hook is None:
        raise ValueError("load_lora_hook is required in v1 (no default client)")

    job = await _fetch_job(session, job_id)

    # Pre-flight cancel. QUEUED → CANCELLED without acquiring the GPU
    # lock or touching anything expensive (ORCHESTRATION-M7.md §1.1).
    if cancel_event.is_set():
        await _transition(session, job, state="CANCELLED", finished_at=_utcnow())
        return JobRunResult.CANCELLED

    async with gpu_lock:
        # A cancel that arrived while we were waiting for the lock
        # still short-circuits here — no dataset work, no subprocess.
        if cancel_event.is_set():
            await _transition(session, job, state="CANCELLED", finished_at=_utcnow())
            return JobRunResult.CANCELLED

        dataset_paths = DatasetPaths(data_dir=data_dir, dataset_id=job.dataset_id)
        job_paths = JobPaths(data_dir=data_dir, job_id=job_id)
        job_paths.ensure_dirs()

        # ------------------------------------------------------------
        # PREPARING
        # ------------------------------------------------------------
        await _transition(session, job, state="PREPARING", started_at=_utcnow())

        phase_a = validate_dataset_phase_a(dataset_paths, min_seconds=min_seconds)
        if not phase_a.ok:
            detail = "; ".join(i.detail for i in phase_a.issues[:5])
            await _fail(
                session,
                job,
                error_code="dataset_invalid",
                error_detail=f"dataset failed Phase A: {detail}",
            )
            return JobRunResult.FAILED
        if cancel_event.is_set():
            await _transition(session, job, state="CANCELLED", finished_at=_utcnow())
            return JobRunResult.CANCELLED

        # Auto-transcribe if the upload didn't include a manifest.
        # Opt-in via transcribe_hook — production passes a closure
        # that wraps FunASRClient.transcribe; tests pass a fake or
        # ``None`` (in which case Phase A has already decided whether
        # the dataset is training-ready).
        if transcribe_hook is not None and not dataset_paths.has_transcripts():
            try:
                n_rows = await auto_transcribe_if_missing(dataset_paths, transcribe=transcribe_hook)
                log.info("training.auto_transcribed", job_id=job_id, rows=n_rows)
            except Exception as e:  # noqa: BLE001
                await _fail(
                    session,
                    job,
                    error_code="auto_transcribe_failed",
                    error_detail=str(e),
                )
                return JobRunResult.FAILED

        # Normalise every manifest ``audio`` path to absolute before
        # training spawns — HF datasets resolves relative paths against
        # cwd, not the manifest's directory, so a bare filename in the
        # jsonl fails at training start with FileNotFoundError. The
        # helper is idempotent; it's a no-op when everything is
        # already absolute.
        if dataset_paths.has_transcripts():
            try:
                rewritten = normalise_manifest_paths(dataset_paths)
                if rewritten:
                    log.info(
                        "training.manifest_normalised",
                        job_id=job_id,
                        rows_rewritten=rewritten,
                    )
            except Exception as e:  # noqa: BLE001
                await _fail(
                    session,
                    job,
                    error_code="manifest_normalise_failed",
                    error_detail=str(e),
                )
                return JobRunResult.FAILED

        # Phase B — advisory transcript-quality check. Runs only if we
        # have both a manifest and an ASR hook; its report is written
        # to ``validation_report.json`` in the dataset dir but never
        # blocks the job (ORCHESTRATION-M7.md §2.2).
        try:
            config_overrides_dict = json.loads(job.config_json or "{}")
        except json.JSONDecodeError:
            config_overrides_dict = {}
        validate_transcripts = bool(config_overrides_dict.get("validate_transcripts", True))
        if transcribe_hook is not None and validate_transcripts and dataset_paths.has_transcripts():
            try:
                phase_b = await validate_transcripts_phase_b(
                    dataset_paths, transcribe=transcribe_hook
                )
                log.info(
                    "training.phase_b_done",
                    job_id=job_id,
                    num_samples=phase_b.num_samples,
                    suspects=len(phase_b.suspects),
                )
            except Exception as e:  # noqa: BLE001
                # Phase B is advisory — a failure here logs but doesn't
                # fail the job. The UI just won't have a report to
                # display.
                log.warning("training.phase_b_failed", job_id=job_id, error=str(e))

        try:
            overrides = json.loads(job.config_json or "{}")
        except json.JSONDecodeError:
            overrides = {}
        try:
            cfg = build_training_config(
                pretrained_path=pretrained_path,
                job_paths=job_paths,
                dataset_paths=dataset_paths,
                overrides=overrides,
            )
        except ValueError as e:
            await _fail(session, job, error_code="bad_config", error_detail=str(e))
            return JobRunResult.FAILED
        write_training_config(cfg, job_paths.train_config_yaml)
        job.resolved_config_path = str(job_paths.train_config_yaml)
        job.max_steps = int(cfg.get("max_steps", job.max_steps))
        await session.commit()

        # ------------------------------------------------------------
        # TRAINING
        # ------------------------------------------------------------
        await _transition(session, job, state="TRAINING")

        # Track the most recent step seen during TRAINING. We don't
        # write each step to the DB — the session isn't safe for
        # concurrent use, and the route layer reads in-flight progress
        # off the JobHandle's step_samples ring. The final
        # ``current_step`` value is committed once TRAINING returns.
        current_step_holder: list[int] = [0]

        # Synchronous callbacks — the subprocess runner invokes them
        # one per stdout line / one per tracker event. We can't await
        # inside them; log appends to Redis are fire-and-forget via
        # ``loop.create_task``; errors are caught + logged.
        loop = asyncio.get_running_loop()

        def on_log(line: str) -> None:
            loop.create_task(_safe_log(log_store, job_id, line))

        def on_state(event: dict[str, Any]) -> None:
            current_step_holder[0] = int(event.get("step", 0))

        outcome = await subprocess_hook(
            script_path=upstream_script_path,
            job_paths=job_paths,
            on_log=on_log,
            on_state=on_state,
            max_steps=job.max_steps,
            cancel_event=cancel_event,
            wall_timeout_seconds=wall_timeout_seconds,
            cancel_grace_seconds=cancel_grace_seconds,
        )
        # Drain any straggler progress commits before we move on.
        await asyncio.sleep(0)
        job.current_step = max(current_step_holder[0], job.current_step)
        await session.commit()

        if outcome is RunnerOutcome.CANCELLED:
            await _transition(session, job, state="CANCELLED", finished_at=_utcnow())
            return JobRunResult.CANCELLED
        if outcome is not RunnerOutcome.SUCCESS:
            await _fail(
                session,
                job,
                error_code=_SUBPROCESS_ERROR_CODES.get(outcome, "subprocess_failed"),
                error_detail=f"subprocess outcome: {outcome.value}",
            )
            return JobRunResult.FAILED

        # ------------------------------------------------------------
        # REGISTERING
        # ------------------------------------------------------------
        await _transition(session, job, state="REGISTERING")

        voice_id = uuid.uuid4().hex
        try:
            weights_dest = _copy_lora_artifacts(job_paths, data_dir, voice_id)
        except OSError as e:
            await _fail(
                session,
                job,
                error_code="artifact_copy_failed",
                error_detail=str(e),
            )
            return JobRunResult.FAILED

        # §1.1 cancellation boundary: before load_lora, cancel wins;
        # after load_lora succeeds, we commit the Voice row.
        if cancel_event.is_set():
            shutil.rmtree(weights_dest, ignore_errors=True)
            await _transition(session, job, state="CANCELLED", finished_at=_utcnow())
            return JobRunResult.CANCELLED

        try:
            await load_lora_hook(voice_id, str(weights_dest))
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(weights_dest, ignore_errors=True)
            await _fail(
                session,
                job,
                error_code="hot_load_rejected",
                error_detail=str(e),
            )
            return JobRunResult.FAILED

        # load_lora succeeded — per §1.1 revised, we commit.
        voice = Voice(
            id=voice_id,
            name=job.name,
            source="lora",
            lora_path=str(weights_dest),
            ft_job_id=job.id,
        )
        session.add(voice)
        try:
            await session.commit()
        except IntegrityError as e:
            await session.rollback()
            # Unique-name conflict (§3.4 E-LoRA-4). Best-effort unload +
            # move weights to _orphaned/ so a retry doesn't find garbage.
            await _safe_unload(load_lora_hook, voice_id)
            await _orphan_weights(weights_dest, data_dir)
            await _fail(
                session,
                job,
                error_code="voice_name_conflict",
                error_detail=str(e),
            )
            return JobRunResult.FAILED

        job.voice_id = voice_id
        await _transition(session, job, state="SUCCEEDED", finished_at=_utcnow())
        return JobRunResult.SUCCEEDED


# ---------------------------------------------------------------------------
# Helpers — small on purpose; the state transitions themselves are the
# interesting part above.
# ---------------------------------------------------------------------------


async def _fetch_job(session: AsyncSession, job_id: str) -> FineTuneJob:
    result = await session.execute(select(FineTuneJob).where(FineTuneJob.id == job_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError(f"FineTuneJob {job_id!r} does not exist")
    return row


async def _transition(
    session: AsyncSession,
    job: FineTuneJob,
    *,
    state: str,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> None:
    job.state = state  # type: ignore[assignment]  # Literal accepts these
    if started_at is not None:
        job.started_at = started_at
    if finished_at is not None:
        job.finished_at = finished_at
    await session.commit()
    log.info("training.job.state_transition", job_id=job.id, state=state)


async def _fail(
    session: AsyncSession,
    job: FineTuneJob,
    *,
    error_code: str,
    error_detail: str,
) -> None:
    job.state = "FAILED"  # type: ignore[assignment]
    job.error_code = error_code
    job.error_detail = error_detail
    job.finished_at = _utcnow()
    await session.commit()
    log.warning(
        "training.job.failed",
        job_id=job.id,
        error_code=error_code,
        error_detail=error_detail,
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


async def _safe_log(store: TrainingLogStore, job_id: str, line: str) -> None:
    try:
        await store.append(job_id, line)
    except Exception:  # noqa: BLE001
        log.warning("training.log_append_failed", job_id=job_id)


def _copy_lora_artifacts(
    job_paths: JobPaths, data_dir: pathlib.Path, voice_id: str
) -> pathlib.Path:
    """Copy ``latest/lora_weights.safetensors`` + ``lora_config.json``
    from the job-scoped checkpoint dir to the voice-keyed storage.
    """
    dest = lora_weights_dir(data_dir, voice_id)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job_paths.latest_lora_weights, dest / "lora_weights.safetensors")
    shutil.copy2(job_paths.latest_lora_config, dest / "lora_config.json")
    return dest


async def _safe_unload(load_lora_hook: LoadLoraHook, voice_id: str) -> None:
    # The load_lora_hook is our ``load`` primitive; the ``unload``
    # primitive would be symmetrical but we don't take it as a param
    # here to keep the hook surface minimal. The voxcpm_worker's
    # registry entries that never received a companion DB Voice row
    # age out naturally at the next gateway boot; this helper is a
    # placeholder for a future wire-up.
    del load_lora_hook, voice_id


async def _orphan_weights(weights_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
    """Move conflict-victim weights to ``lora_weights/_orphaned/{ts}/``.

    See ORCHESTRATION-M7.md §3.4 E-LoRA-4.
    """
    stamp = _utcnow().strftime("%Y%m%dT%H%M%SZ")
    orphan_root = data_dir / "lora_weights" / "_orphaned" / stamp
    try:
        orphan_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(weights_dir), str(orphan_root))
    except OSError as e:
        log.warning("training.orphan_weights_move_failed", src=str(weights_dir), error=str(e))
