"""Typed IPC messages exchanged between the gateway and worker processes.

The goal: the gateway talks to workers through a single message protocol,
regardless of whether the transport is an in-process asyncio.Queue (v1) or a
socket/gRPC stream (future). Keeping these as plain pydantic models means the
same types serialise over the wire later without a rewrite.
"""

from __future__ import annotations

import base64
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_serializer, field_validator


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
# TTS — synthesize
# ---------------------------------------------------------------------------


class SynthesizeRequest(RequestMessage):
    kind: Literal["synthesize"] = "synthesize"
    text: str
    sample_rate: int = 24000

    # Voice conditioning. `ref_audio_latents` drives basic cloning; adding
    # `prompt_audio_latents` + `prompt_text` engages VoxCPM2's "ultimate
    # cloning" mode which also anchors the model on the reference speaker's
    # prosody given the transcript (see PRD §5.1).
    voice_id: str | None = None
    ref_audio_latents: bytes | None = None
    prompt_audio_latents: bytes | None = None
    prompt_text: str = ""

    # Model generation controls. `inference_timesteps` is deliberately NOT
    # here — it's an init-time knob on the nano-vllm-voxcpm engine, not a
    # per-request parameter (see nanovllm_voxcpm.models.voxcpm2.server).
    cfg_value: float = 2.0
    temperature: float = 1.0
    max_generate_length: int = 2000

    @field_validator("ref_audio_latents", "prompt_audio_latents", mode="before")
    @classmethod
    def _decode_latents(cls, v: object) -> bytes | None:
        return _coerce_bytes(v)

    @field_serializer("ref_audio_latents", "prompt_audio_latents", when_used="json")
    def _ser_latents(self, v: bytes | None) -> str | None:
        return base64.b64encode(v).decode("ascii") if v is not None else None


class SynthesizeResponse(ResponseMessage):
    kind: Literal["synthesize"] = "synthesize"
    # Raw int16 PCM little-endian audio. Callers wrap it into a container
    # (WAV / MP3) at the edge — keeps the worker output format-agnostic.
    pcm_s16le: bytes
    sample_rate: int
    duration_ms: int

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


# ---------------------------------------------------------------------------
# TTS — reference encoding (voice cloning)
# ---------------------------------------------------------------------------


class EncodeReferenceRequest(RequestMessage):
    """Ask the worker to encode a reference audio clip to VoxCPM2 latents.

    The returned bytes are the raw float32 tensor (N × feat_dim) that the
    nano-vllm-voxcpm engine expects as `ref_audio_latents` on subsequent
    `generate()` calls. Latent caching persists exactly these bytes.
    """

    kind: Literal["encode_reference"] = "encode_reference"
    audio: bytes
    wav_format: str = "wav"  # librosa-supported format name (wav, mp3, flac, …)

    @field_validator("audio", mode="before")
    @classmethod
    def _decode_audio(cls, v: object) -> bytes:
        out = _coerce_bytes(v)
        if out is None:
            raise ValueError("audio must not be null")
        return out

    @field_serializer("audio", when_used="json")
    def _ser_audio(self, v: bytes) -> str:
        return base64.b64encode(v).decode("ascii")


class EncodeReferenceResponse(ResponseMessage):
    kind: Literal["encode_reference"] = "encode_reference"
    latents: bytes
    feat_dim: int
    num_frames: int
    # Encoder sample rate the worker used (24000 on VoxCPM2). Callers may
    # stash this alongside the latents so they can validate cross-version.
    encoder_sample_rate: int

    @field_validator("latents", mode="before")
    @classmethod
    def _decode_latents(cls, v: object) -> bytes:
        out = _coerce_bytes(v)
        if out is None:
            raise ValueError("latents must not be null")
        return out

    @field_serializer("latents", when_used="json")
    def _ser_latents(self, v: bytes) -> str:
        return base64.b64encode(v).decode("ascii")
