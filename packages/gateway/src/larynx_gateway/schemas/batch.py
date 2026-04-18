"""Request / response schemas for /v1/batch.

See ORCHESTRATION-M8.md §1.2. The per-item params are a strict *subset*
of TTSRequest fields — no inline reference audio in batch submissions,
use an existing voice_id.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Caps the payload size + keeps Arq job payloads under 1MB. Requests
# over this return 400 items_too_many.
MAX_BATCH_ITEMS = 500

# Must stay in lockstep with schemas/tts.OutputFormat. The OpenAI shim
# accepts a wider set (mp3/opus/aac/flac) but batch sticks to the same
# surface as /v1/tts for now — we can widen once pyav lands in M8 Part
# B and is wired into the batch synth path.
BatchOutputFormat = Literal["wav", "pcm16"]


class BatchItemParams(BaseModel):
    """TTS engine knobs per item.

    Kept separate from BatchItemRequest so clients can fall back to
    defaults without repeating every field. Field names match
    TTSRequest exactly so the batch service can feed this dict
    straight into TTSRequest.model_validate() without remapping.
    """

    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    output_format: BatchOutputFormat = "wav"
    cfg_value: float = Field(default=2.0, ge=0.0, le=10.0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    prompt_text: str | None = Field(default=None, max_length=500)


class BatchItemRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice_id: str | None = None
    params: BatchItemParams = Field(default_factory=BatchItemParams)


class BatchCreateRequest(BaseModel):
    items: list[BatchItemRequest] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)
    # Retained jobs skip the daily cleanup sweep. Clients set this for
    # outputs they plan to archive longer than 7 days.
    retain: bool = False


class BatchCreateResponse(BaseModel):
    job_id: str


class BatchItemStatus(BaseModel):
    idx: int
    state: str
    voice_id: str | None = None
    # Populated only when state == "DONE" — relative URL under the
    # gateway (e.g. /v1/batch/{id}/items/3). Clients append their
    # bearer token and fetch.
    url: str | None = None
    duration_ms: int | None = None
    generation_time_ms: int | None = None
    error_code: str | None = None
    error_detail: str | None = None


class BatchJobStatus(BaseModel):
    job_id: str
    state: str
    progress: float = Field(ge=0.0, le=1.0)
    num_items: int
    num_completed: int
    num_failed: int
    retain: bool
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    expires_at: datetime | None = None
    error_code: str | None = None
    items: list[BatchItemStatus]


class BatchCancelResponse(BaseModel):
    job_id: str
    state: str
