"""TTS request/response types: synthesize, streaming synthesize, reference encode."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import field_serializer, field_validator

from larynx_shared.ipc.messages.base import (
    RequestMessage,
    ResponseMessage,
    StreamChunk,
    StreamEnd,
    _coerce_bytes,
)

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

    # LoRA adapter selection (per-request, see ORCHESTRATION-M7.md §3). None
    # means use base weights. The name must be one that a previous
    # ``LoadLoraRequest`` registered, otherwise the worker errors out.
    lora_name: str | None = None

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
# TTS — streaming synthesis (WS /v1/tts/stream)
#
# SynthesizeStreamRequest has the same conditioning fields as SynthesizeRequest
# but the worker emits multiple ``SynthesizeChunkFrame`` messages followed by
# a ``SynthesizeDoneFrame``. The frames share ``request_id`` with the request
# so the client dispatcher routes them to the stream queue.
# ---------------------------------------------------------------------------


class SynthesizeStreamRequest(RequestMessage):
    kind: Literal["synthesize_stream"] = "synthesize_stream"
    text: str
    sample_rate: int = 24000
    voice_id: str | None = None
    ref_audio_latents: bytes | None = None
    prompt_audio_latents: bytes | None = None
    prompt_text: str = ""
    cfg_value: float = 2.0
    temperature: float = 1.0
    max_generate_length: int = 2000
    # Crossfade window applied between emitted chunks (gateway-side). The
    # worker doesn't crossfade itself — it forwards the raw chunks it receives
    # from VoxCPM so callers that want joint-chunk audio (e.g. batch jobs
    # reassembling to WAV) can do their own smoothing.
    crossfade_ms: float = 10.0
    # See SynthesizeRequest.lora_name.
    lora_name: str | None = None

    @field_validator("ref_audio_latents", "prompt_audio_latents", mode="before")
    @classmethod
    def _decode_latents(cls, v: object) -> bytes | None:
        return _coerce_bytes(v)

    @field_serializer("ref_audio_latents", "prompt_audio_latents", when_used="json")
    def _ser_latents(self, v: bytes | None) -> str | None:
        return base64.b64encode(v).decode("ascii") if v is not None else None


class SynthesizeChunkFrame(StreamChunk):
    kind: Literal["synthesize_chunk"] = "synthesize_chunk"
    pcm_s16le: bytes
    sample_rate: int
    # Zero-based index within this stream; handy for logging + ordering
    # invariants in tests (chunks should arrive monotonically).
    chunk_index: int

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


class SynthesizeDoneFrame(StreamEnd):
    kind: Literal["synthesize_done"] = "synthesize_done"
    sample_rate: int
    total_duration_ms: int
    chunk_count: int
    # Worker-measured time-to-first-chunk (GPU-side). The gateway measures
    # its own TTFB (from WS connect → first binary frame); both numbers are
    # useful — they let us attribute latency to GPU vs IPC/WS path.
    ttfb_ms: int


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


__all__ = [
    "EncodeReferenceRequest",
    "EncodeReferenceResponse",
    "SynthesizeChunkFrame",
    "SynthesizeDoneFrame",
    "SynthesizeRequest",
    "SynthesizeResponse",
    "SynthesizeStreamRequest",
]
