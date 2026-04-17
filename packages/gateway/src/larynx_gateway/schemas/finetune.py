"""Pydantic request / response schemas for ``/v1/finetune/*``.

See ORCHESTRATION-M7.md §1.2 (progress shape), §2 (dataset validation
report), and the PRD §5.8 endpoint list.
"""

from __future__ import annotations

from datetime import datetime

from larynx_training_worker.dataset_prep import PhaseAReport
from pydantic import BaseModel, Field


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    report: PhaseAReport


class FineTuneJobCreateRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=128)
    # Free-form overrides merged onto voxcpm_finetune_lora.yaml. Shape is
    # intentionally loose — the config builder validates + sanitises.
    config_overrides: dict[str, object] = Field(default_factory=dict)
    validate_transcripts: bool = True


class FineTuneJobCreateResponse(BaseModel):
    job_id: str


class FineTuneJobStatusResponse(BaseModel):
    """Polled every few seconds while a job is in flight."""

    id: str
    state: str
    name: str
    dataset_id: str
    voice_id: str | None = None
    current_step: int
    max_steps: int
    progress: float  # current_step / max_steps, clamped [0, 1]
    eta_seconds: float | None = None
    error_code: str | None = None
    error_detail: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
