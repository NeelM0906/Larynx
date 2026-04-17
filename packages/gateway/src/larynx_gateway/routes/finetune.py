"""POST /v1/finetune/* — LoRA fine-tuning endpoints.

Covers dataset upload, job CRUD (create / get / delete); SSE log
tailing lands in its own commit.

Spec: PRD §5.8 + ORCHESTRATION-M7.md.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from larynx_shared.paths import SUPPORTED_AUDIO_SUFFIXES, DatasetPaths
from larynx_training_worker.dataset_prep import (
    validate_dataset_phase_a,
)
from larynx_training_worker.subprocess_runner import parse_training_event
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from larynx_gateway.auth import require_bearer_token
from larynx_gateway.db.models import FineTuneJob
from larynx_gateway.db.session import get_session
from larynx_gateway.deps import (
    get_data_dir,
    get_db_session,
    get_funasr_client,
    get_voxcpm_client,
)
from larynx_gateway.schemas.finetune import (
    DatasetUploadResponse,
    FineTuneJobCreateRequest,
    FineTuneJobCreateResponse,
    FineTuneJobStatusResponse,
)
from larynx_gateway.services.training_logs import TrainingLogStore
from larynx_gateway.services.training_orchestrator import JobRunResult, run_job
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient

router = APIRouter(prefix="/v1/finetune", tags=["finetune"])
log = structlog.get_logger(__name__)


@dataclass
class _JobHandle:
    """In-memory handle per running fine-tune job — lets DELETE find the
    cancel_event and the task that ``run_job`` is waiting on.

    Also holds a short ring of recent (timestamp, step) tuples used to
    compute the EWMA-based ETA returned by the status endpoint.
    """

    cancel_event: asyncio.Event
    task: asyncio.Task[JobRunResult]
    step_samples: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=20))


def _jobs_registry(request: Request) -> dict[str, _JobHandle]:
    registry = getattr(request.app.state, "ft_jobs", None)
    if registry is None:
        registry = {}
        request.app.state.ft_jobs = registry
    return registry


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
    request: Request,
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
    min_seconds = getattr(request.app.state, "training_min_seconds", 300)
    report = validate_dataset_phase_a(staging_dataset, min_seconds=min_seconds)
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


# ---------------------------------------------------------------------------
# Jobs — create / get / delete
# ---------------------------------------------------------------------------


@router.post(
    "/jobs",
    response_model=FineTuneJobCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_bearer_token)],
)
async def create_job(
    req: FineTuneJobCreateRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    data_dir: pathlib.Path = Depends(get_data_dir),
    voxcpm_client: VoxCPMClient = Depends(get_voxcpm_client),
    funasr_client: FunASRClient = Depends(get_funasr_client),
) -> FineTuneJobCreateResponse:
    """Persist a ``FineTuneJob(state=QUEUED)`` row and spawn the
    orchestrator task in the background.

    The orchestrator uses its own session (opened inside the task) so
    this request's session isn't held for the duration of training.
    """
    # Dataset must exist on disk before we queue anything — early 404
    # beats a runtime failure 30s into PREPARING.
    dataset = DatasetPaths(data_dir=data_dir, dataset_id=req.dataset_id)
    if not dataset.base_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "dataset_not_found", "detail": req.dataset_id},
        )

    job_id = uuid.uuid4().hex
    job = FineTuneJob(
        id=job_id,
        name=req.name,
        dataset_id=req.dataset_id,
        state="QUEUED",
        config_json=json.dumps(req.config_overrides),
        resolved_config_path="",
        log_key=TrainingLogStore.key(job_id),
        max_steps=int(req.config_overrides.get("max_steps", 1000)),  # type: ignore[arg-type]
    )
    session.add(job)
    await session.commit()

    registry = _jobs_registry(request)
    cancel_event = asyncio.Event()
    handle_ref: list[_JobHandle | None] = [None]

    async def _run_with_own_session() -> JobRunResult:
        # Must open a fresh session; the route's session dies when the
        # response flushes. The orchestrator takes ownership of this
        # session for the life of the job.
        log_store: TrainingLogStore = request.app.state.training_log_store
        hook = getattr(request.app.state, "training_subprocess_hook", None)
        pretrained = getattr(request.app.state, "training_pretrained_path", None) or str(
            data_dir / "pretrained"
        )
        script_path = pathlib.Path(
            getattr(
                request.app.state,
                "training_script_path",
                "third_party/VoxCPM/scripts/train_voxcpm_finetune.py",
            )
        )
        gpu_lock: asyncio.Lock = request.app.state.gpu_train_lock

        async def load_lora_proxy(name: str, path: str) -> None:
            await voxcpm_client.load_lora(name, path)

        async def transcribe_proxy(pcm: bytes, sample_rate: int) -> str:
            resp = await funasr_client.transcribe(pcm_s16le=pcm, sample_rate=sample_rate)
            return resp.text

        def _progress_observer(step: int) -> None:
            h = handle_ref[0]
            if h is not None:
                h.step_samples.append((time.perf_counter(), step))

        async for db_session in get_session():
            try:
                return await run_job(
                    job_id=job_id,
                    session=db_session,
                    log_store=log_store,
                    data_dir=data_dir,
                    pretrained_path=pretrained,
                    upstream_script_path=script_path,
                    gpu_lock=gpu_lock,
                    cancel_event=cancel_event,
                    subprocess_hook=_wrap_hook(hook, _progress_observer)
                    if hook is not None
                    else _default_hook(_progress_observer),
                    load_lora_hook=load_lora_proxy,
                    transcribe_hook=transcribe_proxy,
                    min_seconds=getattr(
                        request.app.state, "training_min_seconds", 300
                    ),
                )
            finally:
                registry.pop(job_id, None)
        raise RuntimeError("get_session did not yield")

    task = asyncio.create_task(_run_with_own_session(), name=f"ft-job-{job_id}")
    handle = _JobHandle(cancel_event=cancel_event, task=task)
    handle_ref[0] = handle
    registry[job_id] = handle

    return FineTuneJobCreateResponse(job_id=job_id)


@router.get(
    "/jobs/{job_id}",
    response_model=FineTuneJobStatusResponse,
    dependencies=[Depends(require_bearer_token)],
)
async def get_job(
    job_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> FineTuneJobStatusResponse:
    job = (
        await session.execute(select(FineTuneJob).where(FineTuneJob.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "job_not_found", "detail": job_id},
        )
    # In-flight jobs publish their current step through the JobHandle's
    # step_samples ring (no concurrent session writes). Prefer that over
    # the DB value for a live view; fall back to the DB for terminal
    # jobs or jobs from a previous process.
    handle = _jobs_registry(request).get(job_id)
    effective_step = job.current_step
    if handle is not None and handle.step_samples:
        _, latest_step = handle.step_samples[-1]
        effective_step = max(effective_step, latest_step)
    progress = _clamp01(effective_step / job.max_steps) if job.max_steps else 0.0
    eta = _estimate_eta_seconds(request, job_id, job)
    return FineTuneJobStatusResponse(
        id=job.id,
        state=job.state,
        name=job.name,
        dataset_id=job.dataset_id,
        voice_id=job.voice_id,
        current_step=effective_step,
        max_steps=job.max_steps,
        progress=progress,
        eta_seconds=eta,
        error_code=job.error_code,
        error_detail=job.error_detail,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get(
    "/jobs/{job_id}/logs",
    dependencies=[Depends(require_bearer_token)],
)
async def stream_job_logs(
    job_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    """SSE stream of ``train_log`` + ``train_state`` events + one terminal.

    Clients may supply a ``Last-Event-ID`` header to resume; the server
    replays everything strictly after that stream id from the Redis
    stream backing the job's logs. The connection closes after the
    terminal event so clients that want fresh runs open a new request.
    """
    job = (
        await session.execute(select(FineTuneJob).where(FineTuneJob.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "job_not_found", "detail": job_id},
        )

    store: TrainingLogStore = request.app.state.training_log_store
    registry = _jobs_registry(request)
    last_event_id = request.headers.get("last-event-id")

    async def _stream():
        cursor: str | None = last_event_id
        # Replay loop — keep pulling from Redis until the DB says the
        # job is terminal AND we've drained every remaining log line.
        # ``empty_ticks`` bounds the wait so an abandoned stream doesn't
        # hold a connection forever.
        empty_ticks = 0
        while True:
            batch = await store.tail(job_id, after_id=cursor, count=200)
            if batch:
                empty_ticks = 0
                for entry in batch:
                    cursor = entry.event_id
                    yield _sse_frame("log", entry.event_id, entry.line)
                    ev = parse_training_event(entry.line)
                    if ev is not None:
                        yield _sse_frame("state", entry.event_id, json.dumps(ev))
            # Refresh the job row so we can detect terminal transitions
            # while the stream is open. SQLAlchemy's session caches rows
            # it's already seen inside a transaction, so without an
            # explicit expire + commit we'd keep reading the same
            # snapshot forever.
            session.expire_all()
            await session.commit()
            current = (
                await session.execute(select(FineTuneJob).where(FineTuneJob.id == job_id))
            ).scalar_one_or_none()
            if current is None:
                break
            if current.state in ("SUCCEEDED", "FAILED", "CANCELLED") and not batch:
                terminal_payload = {
                    "state": current.state,
                    "voice_id": current.voice_id,
                    "error_code": current.error_code,
                    "error_detail": current.error_detail,
                }
                yield _sse_frame("terminal", "terminal", json.dumps(terminal_payload))
                break
            # Job still running — poll-sleep briefly. The Redis tail
            # call returns immediately if there's nothing new, so this
            # sleep is what caps CPU on an idle stream.
            if not batch:
                empty_ticks += 1
                if empty_ticks > 600:  # ~30s of zero activity on an in-flight job
                    yield _sse_frame("error", "stream_stall", "no activity for 30s")
                    break
                await asyncio.sleep(0.05)
            # Leave the registry reference unused — it's available for
            # future cancellation visibility but not needed for the
            # replay loop itself.
            _ = registry

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_bearer_token)],
)
async def cancel_job(
    job_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Signal the running job to cancel.

    Returns 202 immediately. The orchestrator may take up to
    ``TRAIN_CANCEL_GRACE_SECONDS`` to transition to ``CANCELLED``;
    clients poll GET to confirm.
    """
    job = (
        await session.execute(select(FineTuneJob).where(FineTuneJob.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "job_not_found", "detail": job_id},
        )
    registry = _jobs_registry(request)
    handle = registry.get(job_id)
    if handle is not None:
        handle.cancel_event.set()
    # If the job is already terminal we still return 202; the idempotent
    # response keeps DELETE semantics simple for clients that don't
    # track state themselves.
    return {"job_id": job_id, "state": job.state}


# ---------------------------------------------------------------------------
# Helpers for the status endpoint
# ---------------------------------------------------------------------------


def _sse_frame(event: str, event_id: str, data: str) -> str:
    """Serialise one SSE record. ``data`` is split on newlines per the
    SSE spec so multi-line payloads encode correctly.
    """
    out = [f"id: {event_id}", f"event: {event}"]
    for line in data.split("\n"):
        out.append(f"data: {line}")
    out.append("")
    out.append("")  # blank line terminates the record
    return "\n".join(out)


def _clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def _estimate_eta_seconds(request: Request, job_id: str, job: FineTuneJob) -> float | None:
    """20-step EWMA on observed step cadence (ORCHESTRATION-M7.md §1.2)."""
    if job.state in ("SUCCEEDED", "FAILED", "CANCELLED", "QUEUED", "PREPARING"):
        return None
    handle = _jobs_registry(request).get(job_id)
    if handle is None or len(handle.step_samples) < 2:
        return None
    first_t, first_step = handle.step_samples[0]
    last_t, last_step = handle.step_samples[-1]
    if last_step <= first_step or last_t <= first_t:
        return None
    ms_per_step = (last_t - first_t) / (last_step - first_step)
    remaining = max(0, job.max_steps - job.current_step)
    return round(remaining * ms_per_step, 2)


def _wrap_hook(hook, observer):
    """Wrap a caller-supplied subprocess hook so on_state events also
    feed our step_samples ring without the caller knowing about it.
    """

    async def wrapped(**kwargs):
        original_on_state = kwargs["on_state"]

        def tap(event):
            step = int(event.get("step", 0))
            observer(step)
            original_on_state(event)

        kwargs["on_state"] = tap
        return await hook(**kwargs)

    return wrapped


def _default_hook(observer):
    """Wire the real subprocess runner into the observer."""
    from larynx_training_worker.subprocess_runner import run_training_subprocess

    async def wrapped(**kwargs):
        original_on_state = kwargs["on_state"]

        def tap(event):
            step = int(event.get("step", 0))
            observer(step)
            original_on_state(event)

        kwargs["on_state"] = tap
        return await run_training_subprocess(**kwargs)

    return wrapped
