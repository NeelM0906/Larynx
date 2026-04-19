"""Typed IPC messages exchanged between the gateway and worker processes.

The goal: the gateway talks to workers through a single message protocol,
regardless of whether the transport is an in-process asyncio.Queue (v1) or a
socket/gRPC stream (future). Keeping these as plain pydantic models means the
same types serialise over the wire later without a rewrite.

The module is split into per-domain submodules (base / tts / stt / vad /
lora / training) and this package re-exports everything so historical
``from larynx_shared.ipc.messages import X`` imports keep working.
"""

from __future__ import annotations

from larynx_shared.ipc.messages.base import (
    CancelStreamRequest,
    ErrorMessage,
    Heartbeat,
    RequestMessage,
    ResponseMessage,
    StreamChunk,
    StreamEnd,
    _coerce_bytes,
    _new_id,
)
from larynx_shared.ipc.messages.lora import (
    ListLorasRequest,
    ListLorasResponse,
    LoadLoraRequest,
    LoadLoraResponse,
    UnloadLoraRequest,
    UnloadLoraResponse,
)
from larynx_shared.ipc.messages.stt import (
    TranscribeRequest,
    TranscribeResponse,
    TranscribeRollingRequest,
    TranscribeRollingResponse,
)
from larynx_shared.ipc.messages.training import (
    TrainDoneFrame,
    TrainLogChunk,
    TrainLoraRequest,
    TrainStateChunk,
)
from larynx_shared.ipc.messages.tts import (
    EncodeReferenceRequest,
    EncodeReferenceResponse,
    SynthesizeChunkFrame,
    SynthesizeDoneFrame,
    SynthesizeRequest,
    SynthesizeResponse,
    SynthesizeStreamRequest,
)
from larynx_shared.ipc.messages.vad import (
    DetectSegmentsRequest,
    DetectSegmentsResponse,
    PunctuateRequest,
    PunctuateResponse,
    Segment,
    VadStreamCloseRequest,
    VadStreamCloseResponse,
    VadStreamEvent,
    VadStreamEventType,
    VadStreamFeedRequest,
    VadStreamFeedResponse,
    VadStreamOpenRequest,
    VadStreamOpenResponse,
)

__all__ = [
    "_coerce_bytes",
    "_new_id",
    "CancelStreamRequest",
    "DetectSegmentsRequest",
    "DetectSegmentsResponse",
    "EncodeReferenceRequest",
    "EncodeReferenceResponse",
    "ErrorMessage",
    "Heartbeat",
    "ListLorasRequest",
    "ListLorasResponse",
    "LoadLoraRequest",
    "LoadLoraResponse",
    "PunctuateRequest",
    "PunctuateResponse",
    "RequestMessage",
    "ResponseMessage",
    "Segment",
    "StreamChunk",
    "StreamEnd",
    "SynthesizeChunkFrame",
    "SynthesizeDoneFrame",
    "SynthesizeRequest",
    "SynthesizeResponse",
    "SynthesizeStreamRequest",
    "TrainDoneFrame",
    "TrainLogChunk",
    "TrainLoraRequest",
    "TrainStateChunk",
    "TranscribeRequest",
    "TranscribeResponse",
    "TranscribeRollingRequest",
    "TranscribeRollingResponse",
    "UnloadLoraRequest",
    "UnloadLoraResponse",
    "VadStreamCloseRequest",
    "VadStreamCloseResponse",
    "VadStreamEvent",
    "VadStreamEventType",
    "VadStreamFeedRequest",
    "VadStreamFeedResponse",
    "VadStreamOpenRequest",
    "VadStreamOpenResponse",
]
