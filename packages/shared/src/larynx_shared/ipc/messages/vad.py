"""VAD + Punctuation types, handled by the CPU worker (vad_punc_worker).

Covers one-shot segment detection, one-shot punctuation, and the streaming
VAD session protocol used by WS ``/v1/stt/stream`` (open → feed*+ → close).
"""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import BaseModel, Field, field_serializer, field_validator

from larynx_shared.ipc.messages.base import (
    RequestMessage,
    ResponseMessage,
    _coerce_bytes,
)

# ---------------------------------------------------------------------------
# VAD + Punctuation — handled by the CPU worker (vad_punc_worker)
# ---------------------------------------------------------------------------


class DetectSegmentsRequest(RequestMessage):
    """Run fsmn-vad over the audio and return voiced regions."""

    kind: Literal["detect_segments"] = "detect_segments"
    pcm_s16le: bytes
    sample_rate: int = 16000

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


class Segment(BaseModel):
    start_ms: int
    end_ms: int
    is_speech: bool = True


class DetectSegmentsResponse(ResponseMessage):
    kind: Literal["detect_segments"] = "detect_segments"
    segments: list[Segment]


class PunctuateRequest(RequestMessage):
    kind: Literal["punctuate"] = "punctuate"
    text: str
    # ISO-639; used to short-circuit languages that ct-punc can't handle
    # (anything outside zh/en). Caller may pass None — worker defaults to
    # "en" behaviour (pass-through when ct-punc adds nothing).
    language: str | None = None


class PunctuateResponse(ResponseMessage):
    kind: Literal["punctuate"] = "punctuate"
    text: str
    # Echoes whether punctuation was actually applied (False when the
    # language was outside ct-punc's supported set — Fun-ASR's itn=True
    # output is already punctuated in-line for MLT languages).
    applied: bool


# ---------------------------------------------------------------------------
# Streaming VAD segmentation (WS /v1/stt/stream feeds this)
#
# Each streaming STT session opens a VAD session, feeds 20ms PCM frames, and
# closes it when the WebSocket closes. Every Feed returns the list of
# VAD events produced by incorporating those new samples; the gateway emits
# speech_start / speech_end to the WS client as those fire. Heartbeats are
# synthesised on the gateway from a timer — the VAD worker itself only needs
# to report state changes.
# ---------------------------------------------------------------------------


VadStreamEventType = Literal["speech_start", "speech_end"]


class VadStreamEvent(BaseModel):
    event: VadStreamEventType
    vad_state: Literal["speaking", "silent"]
    # Offset in milliseconds from session open to the sample that triggered
    # this event. Gateway uses it to slice audio buffer.
    session_ms: int


class VadStreamOpenRequest(RequestMessage):
    kind: Literal["vad_stream_open"] = "vad_stream_open"
    session_id: str
    sample_rate: int = 16000
    # Silence window (ms) after which a speech segment is considered closed.
    # FSMN-VAD's own endpoint detector uses something similar; this value
    # lets the gateway tune for interactivity vs. false triggers.
    speech_end_silence_ms: int = 300


class VadStreamOpenResponse(ResponseMessage):
    kind: Literal["vad_stream_open"] = "vad_stream_open"
    session_id: str
    sample_rate: int


class VadStreamFeedRequest(RequestMessage):
    kind: Literal["vad_stream_feed"] = "vad_stream_feed"
    session_id: str
    pcm_s16le: bytes
    is_final: bool = False

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


class VadStreamFeedResponse(ResponseMessage):
    kind: Literal["vad_stream_feed"] = "vad_stream_feed"
    session_id: str
    events: list[VadStreamEvent] = Field(default_factory=list)
    vad_state: Literal["speaking", "silent"]
    session_ms: int


class VadStreamCloseRequest(RequestMessage):
    kind: Literal["vad_stream_close"] = "vad_stream_close"
    session_id: str


class VadStreamCloseResponse(ResponseMessage):
    kind: Literal["vad_stream_close"] = "vad_stream_close"
    session_id: str


__all__ = [
    "DetectSegmentsRequest",
    "DetectSegmentsResponse",
    "PunctuateRequest",
    "PunctuateResponse",
    "Segment",
    "VadStreamCloseRequest",
    "VadStreamCloseResponse",
    "VadStreamEvent",
    "VadStreamEventType",
    "VadStreamFeedRequest",
    "VadStreamFeedResponse",
    "VadStreamOpenRequest",
    "VadStreamOpenResponse",
]
