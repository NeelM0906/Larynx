"""Typed IPC messages exchanged between the gateway and worker processes.

The goal: the gateway talks to workers through a single message protocol,
regardless of whether the transport is an in-process asyncio.Queue (v1) or a
socket/gRPC stream (future). Keeping these as plain pydantic models means the
same types serialise over the wire later without a rewrite.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex


class RequestMessage(BaseModel):
    """Base class for every request the gateway sends to a worker."""

    request_id: str = Field(default_factory=_new_id)
    kind: str


class ResponseMessage(BaseModel):
    """Base class for every successful reply a worker sends back."""

    request_id: str
    kind: str


class ErrorMessage(BaseModel):
    """Returned when a worker could not fulfil a request."""

    request_id: str
    kind: Literal["error"] = "error"
    code: str
    message: str


class Heartbeat(BaseModel):
    """Periodic liveness signal. Not used in M1 but defined so the protocol
    is complete; the supervisor will consume these in later milestones."""

    kind: Literal["heartbeat"] = "heartbeat"
    worker: str
    timestamp: float


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class SynthesizeRequest(RequestMessage):
    kind: Literal["synthesize"] = "synthesize"
    text: str
    sample_rate: int = 24000
    # Voice-library fields — defined now so M2 doesn't touch the protocol.
    voice_id: str | None = None
    cfg_value: float = 2.0
    inference_timesteps: int = 10


class SynthesizeResponse(ResponseMessage):
    kind: Literal["synthesize"] = "synthesize"
    # Raw int16 PCM little-endian audio. Callers wrap it into a container
    # (WAV / MP3) at the edge — keeps the worker output format-agnostic.
    pcm_s16le: bytes
    sample_rate: int
    duration_ms: int
