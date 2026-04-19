"""Transport-agnostic envelope and streaming primitives.

Every RPC in the IPC layer is either a single request/response pair or a
streaming RPC (one request → many responses sharing ``request_id``). The
base classes here carry the discriminator (``kind``) and id plumbing;
per-domain modules (tts/stt/vad/lora/training) define concrete payload
types on top of them.
"""

from __future__ import annotations

import base64
import uuid
from typing import Literal

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Byte-field handling
#
# Pydantic v2 serialises `bytes` to base64 by default for JSON mode, but
# we also want the in-process asyncio.Queue path (which keeps raw Python
# objects) to pass bytes through unmodified. The validator below accepts
# either raw bytes (in-process) or a base64 string (deserialised from JSON)
# so the same models work across both transports.
# ---------------------------------------------------------------------------


def _coerce_bytes(v: object) -> bytes | None:
    if v is None or isinstance(v, bytes):
        return v  # type: ignore[return-value]
    if isinstance(v, bytearray):
        return bytes(v)
    if isinstance(v, str):
        return base64.b64decode(v)
    raise TypeError(f"expected bytes or base64 str, got {type(v).__name__}")


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
# Streaming protocol primitives
#
# A streaming RPC is: one ``RequestMessage`` → many responses that share the
# request's ``request_id``. Each intermediate response is a ``StreamChunk``
# subclass; the terminal frame is a ``StreamEnd`` subclass. An ``ErrorMessage``
# also terminates the stream. ``CancelStreamRequest`` is sent gateway → worker
# when the consumer abandons the stream.
# ---------------------------------------------------------------------------


class StreamChunk(ResponseMessage):
    """Marker base for intermediate frames of a streaming RPC."""


class StreamEnd(ResponseMessage):
    """Marker base for the terminal frame of a streaming RPC."""


class CancelStreamRequest(RequestMessage):
    """Ask the worker to stop producing chunks for a specific in-flight request.

    The worker looks up the task by ``target_request_id`` and cancels it.
    It's a best-effort hint — if the worker has already emitted ``StreamEnd``
    the cancel is a no-op.
    """

    kind: Literal["cancel_stream"] = "cancel_stream"
    target_request_id: str


__all__ = [
    "_coerce_bytes",
    "_new_id",
    "CancelStreamRequest",
    "ErrorMessage",
    "Heartbeat",
    "RequestMessage",
    "ResponseMessage",
    "StreamChunk",
    "StreamEnd",
]
