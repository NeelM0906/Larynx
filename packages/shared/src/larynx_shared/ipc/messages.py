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


# ---------------------------------------------------------------------------
# TTS — LoRA hot-swap (see ORCHESTRATION-M7.md §3)
#
# nano-vllm-voxcpm's ``AsyncVoxCPM2ServerPool`` exposes
# register_lora / unregister_lora / list_loras; we mirror that into the
# IPC protocol so the gateway can hot-load a fine-tuned LoRA after the
# training_worker writes its weights to disk. Per-request selection lives
# on ``SynthesizeRequest.lora_name`` above.
# ---------------------------------------------------------------------------


class LoadLoraRequest(RequestMessage):
    kind: Literal["load_lora"] = "load_lora"
    name: str
    path: str  # directory holding lora_weights.safetensors + lora_config.json


class LoadLoraResponse(ResponseMessage):
    kind: Literal["load_lora"] = "load_lora"
    name: str


class UnloadLoraRequest(RequestMessage):
    kind: Literal["unload_lora"] = "unload_lora"
    name: str


class UnloadLoraResponse(ResponseMessage):
    kind: Literal["unload_lora"] = "unload_lora"
    name: str


class ListLorasRequest(RequestMessage):
    kind: Literal["list_loras"] = "list_loras"


class ListLorasResponse(ResponseMessage):
    kind: Literal["list_loras"] = "list_loras"
    names: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Training — LoRA fine-tune (see ORCHESTRATION-M7.md §5.3)
#
# One TrainLoraRequest kicks off a fine-tune. The worker then streams
# TrainLogChunk (one per stdout line), TrainStateChunk (when a tracker
# step line is parseable), and ends with exactly one TrainDoneFrame. A
# CancelStreamRequest mid-flight triggers §1.1 cancellation and the
# TrainDoneFrame reports state=CANCELLED.
# ---------------------------------------------------------------------------


class TrainLoraRequest(RequestMessage):
    kind: Literal["train_lora"] = "train_lora"
    job_id: str
    dataset_id: str
    voice_name: str
    # Arbitrary overrides merged onto the upstream voxcpm_finetune_lora.yaml
    # template — rank, alpha, num_iters, learning_rate, etc. Unknown keys
    # pass through unmodified so the gateway doesn't have to mirror every
    # upstream field. Serialised as JSON over the wire.
    config_overrides: dict[str, object] = Field(default_factory=dict)
    # Explicit opt-out of Phase-B transcript quality check. Default True;
    # see ORCHESTRATION-M7.md §2.2.
    validate_transcripts: bool = True


class TrainLogChunk(StreamChunk):
    kind: Literal["train_log"] = "train_log"
    line: str


class TrainStateChunk(StreamChunk):
    """Structured progress extracted from an upstream tracker line.

    Emitted opportunistically — not every training line parses to a
    state event. Callers should treat a gap between events as "no new
    progress", not "training stuck".
    """

    kind: Literal["train_state"] = "train_state"
    step: int
    loss_diff: float | None = None
    loss_stop: float | None = None
    lr: float | None = None
    epoch: float | None = None


class TrainDoneFrame(StreamEnd):
    kind: Literal["train_done"] = "train_done"
    # "SUCCEEDED" | "FAILED" | "CANCELLED". String rather than Literal so
    # adding a state here doesn't cascade into a mypy update everywhere.
    state: str
    voice_id: str | None = None
    error_code: str | None = None
    error_detail: str | None = None


# ---------------------------------------------------------------------------
# STT — transcription
#
# Audio travels as int16 LE PCM at ``sample_rate`` (typically 16 kHz mono).
# Callers resample upstream so the worker never has to branch on format.
# ``language`` is an ISO-639 code (e.g. "en", "zh", "pt"); the worker maps
# it to Fun-ASR's Chinese-name convention and chooses Nano vs MLT via
# ``language_router``.
# ---------------------------------------------------------------------------


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
