"""Request / response schemas for /v1/tts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

OutputFormat = Literal["wav", "pcm16"]  # mp3 lands in M4 behind ffmpeg


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice_id: str | None = Field(default=None, description="M2+ — ignored in M1")
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    output_format: OutputFormat = "wav"
    cfg_value: float = Field(default=2.0, ge=0.0, le=10.0)
