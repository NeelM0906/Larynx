"""SQLAlchemy models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import DateTime, Integer, String, Text, func
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
