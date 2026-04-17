"""Request / response schemas for the OpenAI-compatible shim routes.

Kept separate from ``schemas/tts.py`` because the OpenAI shape is
stable and we don't want drift in our native ``TTSRequest`` to leak into
the shim, or vice-versa. The native schema is a superset; the shim maps
down to it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

OpenAIResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class OpenAISpeechRequest(BaseModel):
    """Body for ``POST /v1/audio/speech`` — matches OpenAI TTS exactly.

    The ``model`` field is accepted for wire compatibility with the
    OpenAI SDK (clients pass ``tts-1`` / ``tts-1-hd``) and otherwise
    ignored — we always route through VoxCPM2.
    """

    model: str = Field(
        default="tts-1",
        description="Accepted for OpenAI SDK compatibility; ignored — VoxCPM2 is always used.",
    )
    input: str = Field(min_length=1, max_length=5000)
    voice: str = Field(min_length=1, max_length=128)
    response_format: OpenAIResponseFormat = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class OpenAIErrorBody(BaseModel):
    type: str
    code: str
    message: str


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorBody


__all__ = [
    "OpenAIErrorBody",
    "OpenAIErrorResponse",
    "OpenAIResponseFormat",
    "OpenAISpeechRequest",
]
