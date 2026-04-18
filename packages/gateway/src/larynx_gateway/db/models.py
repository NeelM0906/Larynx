"""SQLAlchemy models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Application-level enum. Kept as ``typing.Literal`` (not SQL ENUM) so the
# migration story stays simple — the DB column is plain VARCHAR(32) and a
# new variant (e.g. M7's ``'lora'``) lands without an ALTER TYPE.
VoiceSource = Literal["uploaded", "designed", "seed", "lora"]

# M7 fine-tune-job lifecycle. Stored as a VARCHAR(32) for the same
# forward-migration reasons as VoiceSource.
JobState = Literal[
    "QUEUED",
    "PREPARING",
    "TRAINING",
    "REGISTERING",
    "SUCCEEDED",
    "FAILED",
    "CANCELLED",
]

# M8 batch-job lifecycle. FAILED is reserved for "no items ever started
# successfully"; partial failures report as COMPLETED with
# ``num_failed > 0`` so clients can retry specific indices without
# re-submitting the whole job. See ORCHESTRATION-M8.md §1.4.
BatchJobState = Literal["QUEUED", "RUNNING", "COMPLETED", "CANCELLED", "FAILED"]
BatchItemState = Literal["QUEUED", "RUNNING", "DONE", "FAILED", "CANCELLED"]


class Base(DeclarativeBase):
    pass


class Voice(Base):
    """Persistent voice record.

    Columns populated at different points in the voice lifecycle:
    - ``ref_audio_path`` / ``latent_path`` / ``sample_rate_hz`` / ``duration_ms``:
      set when M2 upload encodes a reference.
    - ``prompt_text``: set when the voice was created with the "ultimate
      cloning" mode (transcript of the reference audio).
    - ``design_prompt``: set for voices created via POST /v1/voices/design.
    - ``source``: see :data:`VoiceSource` — distinguishes provenance.
    - ``lora_path`` / ``ft_job_id``: populated for M7 LoRA voices.
      ``ft_job_id`` is an application-level link (no DB foreign key) so
      job pruning doesn't cascade into voices.
    """

    __tablename__ = "voices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)
    source: Mapped[VoiceSource] = mapped_column(String(32), nullable=False, default="uploaded")

    ref_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    latent_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    prompt_text: Mapped[str | None] = mapped_column(Text(), nullable=True)
    design_prompt: Mapped[str | None] = mapped_column(Text(), nullable=True)

    sample_rate_hz: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer(), nullable=True)

    lora_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    ft_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class FineTuneJob(Base):
    """Fine-tune job row — source of truth for job state + progress.

    See ORCHESTRATION-M7.md §4.2 for field-by-field rationale. Column
    types intentionally mirror Voice conventions (String(36) ids,
    timezone-aware timestamps) so cross-table joins don't need casts.
    """

    __tablename__ = "fine_tune_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[JobState] = mapped_column(String(32), nullable=False, index=True)
    # Populated when state transitions to SUCCEEDED; indexed for the
    # voice → job back-link used by the UI's "show training report"
    # affordance.
    voice_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    config_json: Mapped[str] = mapped_column(Text(), nullable=False)
    resolved_config_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    log_key: Mapped[str] = mapped_column(String(128), nullable=False)

    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text(), nullable=True)

    current_step: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    max_steps: Mapped[int] = mapped_column(Integer(), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BatchJob(Base):
    """One row per submitted batch TTS job.

    Progress counters live on the job row (not derived from scanning
    items) so GET /v1/batch/{id} stays fast even on large jobs. The
    counters and item rows must be updated in the same transaction;
    see services/batch_service.py for the invariant.
    """

    __tablename__ = "batch_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    state: Mapped[BatchJobState] = mapped_column(String(32), nullable=False, index=True)
    num_items: Mapped[int] = mapped_column(Integer(), nullable=False)
    num_completed: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    num_failed: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    retain: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )


class BatchItem(Base):
    """One row per (batch_job, item_idx) tuple.

    ``item_idx`` is preserved from the request so clients addressing a
    specific index always find the same item even if partial failures
    reshuffle completion order.
    """

    __tablename__ = "batch_items"
    __table_args__ = (UniqueConstraint("job_id", "item_idx", name="uq_batch_items_job_idx"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("batch_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    item_idx: Mapped[int] = mapped_column(Integer(), nullable=False)
    state: Mapped[BatchItemState] = mapped_column(String(32), nullable=False, index=True)

    voice_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    text: Mapped[str] = mapped_column(Text(), nullable=False)
    params_json: Mapped[str] = mapped_column(Text(), nullable=False, default="{}")

    output_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    output_format: Mapped[str | None] = mapped_column(String(16), nullable=True)
    sample_rate: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    generation_time_ms: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    etag: Mapped[str | None] = mapped_column(String(64), nullable=True)

    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text(), nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
