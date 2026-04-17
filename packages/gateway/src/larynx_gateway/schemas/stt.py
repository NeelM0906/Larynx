"""Request / response schemas for /v1/stt + /v1/audio/transcriptions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ModelUsed = Literal["nano", "mlt"]


class STTResponse(BaseModel):
    """Response body for ``POST /v1/stt`` (see PRD §5.3)."""

    text: str
    language: str
    model_used: ModelUsed
    duration_ms: int
    processing_ms: int
    punctuated: bool


# OpenAI's /v1/audio/transcriptions returns {"text": "..."} when
# ``response_format=json`` and a plain string for ``response_format=text``.
# Clients that expect the ``language`` key from the verbose-json format
# still work — we always populate it, matching Whisper's shape.
class OpenAITranscriptionResponse(BaseModel):
    text: str
    language: str
