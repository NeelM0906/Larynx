"""STT request/response types: one-shot and rolling-buffer transcription.

Audio travels as int16 LE PCM at ``sample_rate`` (typically 16 kHz mono).
Callers resample upstream so the worker never has to branch on format.
``language`` is an ISO-639 code (e.g. "en", "zh", "pt"); the worker maps
it to Fun-ASR's Chinese-name convention and chooses Nano vs MLT via
``language_router``.
"""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import Field, field_serializer, field_validator

from larynx_shared.ipc.messages.base import (
    RequestMessage,
    ResponseMessage,
    _coerce_bytes,
)


class TranscribeRequest(RequestMessage):
    kind: Literal["transcribe"] = "transcribe"
    pcm_s16le: bytes
    sample_rate: int = 16000
    language: str | None = None  # ISO-639 code, None = auto-detect (Nano)
    hotwords: list[str] = Field(default_factory=list)
    itn: bool = True

    @field_validator("pcm_s16le", mode="before")
    @classmethod
    def _decode_pcm(cls, v: object) -> bytes:
        out = _coerce_bytes(v)
        if out is None:
            raise ValueError("pcm_s16le must not be null")
        return out

    @field_serializer("pcm_s16le", when_used="json")
    def _ser_pcm(self, v: bytes) -> str:
        return base64.b64encode(v).decode("ascii")


class TranscribeResponse(ResponseMessage):
    kind: Literal["transcribe"] = "transcribe"
    text: str
    # ISO-639 code that was actually used. Either echoes the caller's
    # ``language`` or — when ``language`` was None — reports what Fun-ASR
    # auto-detected (best-effort; Fun-ASR-Nano does not always tag output).
    language: str
    model_used: Literal["nano", "mlt"]


class TranscribeRollingRequest(RequestMessage):
    """Streaming rolling-buffer decode (see PRD §5.4).

    Each intermediate call passes the growing audio buffer + the previous
    partial as ``prev_text`` for context continuity. When ``is_final`` is
    False the worker drops the last ``drop_tail_tokens`` tokens from the
    result (they're the ones most likely to be revised). When True, the
    full decode is returned.
    """

    kind: Literal["transcribe_rolling"] = "transcribe_rolling"
    pcm_s16le: bytes
    sample_rate: int = 16000
    language: str | None = None
    hotwords: list[str] = Field(default_factory=list)
    itn: bool = True
    prev_text: str = ""
    is_final: bool = False
    drop_tail_tokens: int = 5

    @field_validator("pcm_s16le", mode="before")
    @classmethod
    def _decode_pcm(cls, v: object) -> bytes:
        out = _coerce_bytes(v)
        if out is None:
            raise ValueError("pcm_s16le must not be null")
        return out

    @field_serializer("pcm_s16le", when_used="json")
    def _ser_pcm(self, v: bytes) -> str:
        return base64.b64encode(v).decode("ascii")


class TranscribeRollingResponse(ResponseMessage):
    kind: Literal["transcribe_rolling"] = "transcribe_rolling"
    text: str
    language: str
    model_used: Literal["nano", "mlt"]
    is_final: bool


__all__ = [
    "TranscribeRequest",
    "TranscribeResponse",
    "TranscribeRollingRequest",
    "TranscribeRollingResponse",
]
