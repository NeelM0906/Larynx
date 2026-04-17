"""Request / response schemas for /v1/voices."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from larynx_gateway.db.models import VoiceSource


class VoiceResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    source: VoiceSource
    sample_rate_hz: int | None = None
    duration_ms: int | None = None
    prompt_text: str | None = None
    design_prompt: str | None = None
    created_at: datetime
    updated_at: datetime


class VoiceListResponse(BaseModel):
    voices: list[VoiceResponse]
    total: int
    limit: int
    offset: int


class VoiceDesignRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    # The prompt used by VoxCPM2's parenthetical voice-design syntax
    # (e.g. "warm, middle-aged female, slight southern lilt").
    design_prompt: str = Field(min_length=1, max_length=500)
    # Text synthesised for the preview audio so the user can listen before
    # committing; has a sensible default so the client doesn't need to
    # think about it.
    preview_text: str = Field(
        default="This is how the designed voice will sound.",
        min_length=1,
        max_length=400,
    )


class VoiceDesignPreviewResponse(BaseModel):
    preview_id: str
    expires_in_s: int
    name: str
    description: str | None = None
    design_prompt: str
    preview_text: str
    sample_rate: int
    duration_ms: int


class VoiceDesignSaveRequest(BaseModel):
    # The name originally provided at design time can be overridden here
    # (e.g. user listened to the preview and wants a different name).
    name: str | None = Field(default=None, max_length=128)
    description: str | None = None
