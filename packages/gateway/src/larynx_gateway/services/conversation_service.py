"""Conversation orchestrator — M5.

Implements the state machine + event queue in ORCHESTRATION.md v2:

- §1     Per-session state machine (IDLE → USER_SPEAKING → LLM_GENERATING
         → TTS_SPEAKING → IDLE, with barge-in back to USER_SPEAKING)
- §1.1   Single-consumer event queue fan-in from STT / LLM / TTS / WS
- §2     Per-turn ``turn_cancel`` asyncio.Event; ``SHUTDOWN_GRACE_SECONDS``
         for orderly close
- §3     Barge-in cancel order: fire both .cancel() calls first, then
         await TTS (blocks until audio stops), then await LLM
- §4     Display-only partials for v1
- §5 E2  LLM timeout: let the current in-flight TTS sentence finish
         before erroring
- §5 E7  FILLER_TOKENS + normalisation check skips LLM calls on noise-
         driven false finals
- §7     Reads STTStreamSession's utterance_ordinal to drop stale events
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import string
import time
import tomllib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import structlog
from larynx_shared.ipc import SynthesizeChunkFrame, SynthesizeDoneFrame

from larynx_gateway.services.llm_client import (
    ChatMessage,
    LLMClient,
    LLMHTTPError,
    LLMTimeoutError,
)
from larynx_gateway.services.stt_stream_service import (
    STTStreamConfig,
    STTStreamSession,
)
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


SHUTDOWN_GRACE_SECONDS = 2.0  # §2 — worst-case in-flight SSE close + slack


# Sentence boundary: ASCII .?! and CJK 。！？, followed by whitespace or
# end-of-string. Runs over the accumulated LLM buffer each token; when
# matched, everything up to the boundary is handed to TTS.
_SENTENCE_BOUNDARY_CHARS = frozenset(".!?。！？")

# Hard-coded fallback so the module is usable when no TOML is present.
# Loader in ``_load_filler_tokens`` replaces this with a file-driven set
# if ``config/conversation/filler_tokens.toml`` exists.
_DEFAULT_FILLER_TOKENS: frozenset[str] = frozenset(
    {
        # English
        "uh", "um", "hmm", "mm", "ah", "er", "eh", "huh",
        # Chinese
        "嗯", "呃", "啊", "哦", "哎",
        # Japanese
        "えー", "あのー", "ええと", "えっと", "うーん",
    }
)


def _repo_filler_tokens_path() -> Path:
    # gateway/src/larynx_gateway/services/conversation_service.py
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # resolved().parents[5] → repo root
    return Path(__file__).resolve().parents[5] / "config" / "conversation" / "filler_tokens.toml"


def _load_filler_tokens() -> frozenset[str]:
    path = _repo_filler_tokens_path()
    if not path.is_file():
        return _DEFAULT_FILLER_TOKENS
    try:
        with path.open("rb") as f:
            doc = tomllib.load(f)
        section = doc.get("filler_tokens", {})
        tokens: set[str] = set()
        for _lang, items in section.items():
            if isinstance(items, list):
                tokens.update(str(x) for x in items)
        return frozenset(tokens) or _DEFAULT_FILLER_TOKENS
    except Exception:  # noqa: BLE001
        log.warning("conversation.filler_tokens_load_failed", path=str(path))
        return _DEFAULT_FILLER_TOKENS


FILLER_TOKENS: frozenset[str] = _load_filler_tokens()


def _is_filler_or_empty(text: str) -> bool:
    normalized = text.strip().lower().rstrip(string.punctuation + "。！？，、")
    normalized = normalized.strip()
    if not normalized:
        return True
    return normalized in FILLER_TOKENS


_SENT_SPLIT_RE = re.compile(r"([.!?。！？])(\s|$)")


def _pop_complete_sentence(buf: str) -> tuple[str | None, str]:
    """Return ``(sentence, remainder)`` if ``buf`` contains a boundary.

    A boundary is one of ``.?!。！？`` followed by whitespace or end-of-
    string. If no boundary is present, returns ``(None, buf)``.
    """
    m = _SENT_SPLIT_RE.search(buf)
    if not m:
        return None, buf
    end = m.end(1)  # position just after the punctuation char
    sentence = buf[:end].strip()
    remainder = buf[end:].lstrip()
    if not sentence:
        return None, remainder
    return sentence, remainder


# ---------------------------------------------------------------------------
# SessionEvent — tagged union
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VADEvent:
    kind: str  # "speech_start" | "speech_end"
    utterance_ordinal: int
    session_ms: int


@dataclass(frozen=True)
class STTEvent:
    kind: str  # "transcript_partial" | "transcript_final" | "stt_error"
    utterance_ordinal: int
    text: str = ""
    punctuated_text: str = ""
    language: str | None = None
    code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class LLMEvent:
    kind: str  # "llm_token" | "llm_done" | "llm_error" | "llm_timeout"
    utterance_ordinal: int
    delta: str = ""
    code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class TTSEvent:
    kind: str  # "tts_chunk" | "tts_done" | "tts_error"
    utterance_ordinal: int
    pcm_s16le: bytes = b""
    sample_rate: int = 0
    code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class WSControlEvent:
    payload: dict[str, Any] = field(default_factory=dict)


SessionEvent = VADEvent | STTEvent | LLMEvent | TTSEvent | WSControlEvent


# ---------------------------------------------------------------------------
# State machine + config
# ---------------------------------------------------------------------------


class SessionState(StrEnum):
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    LLM_GENERATING = "llm_generating"
    TTS_SPEAKING = "tts_speaking"


@dataclass
class ConversationConfig:
    voice_id: str | None = None
    llm_model: str = "anthropic/claude-haiku-4.5"
    system_prompt: str = ""
    # User-side PCM is 16kHz mono int16 — FunASR requirement.
    input_sample_rate: int = 16000
    # TTS-side PCM rate sent back to the client.
    output_sample_rate: int = 24000
    # VAD silence window before speech_end fires — tuneable.
    speech_end_silence_ms: int = 300
    # Cadence of rolling-buffer partials from STT.
    partial_interval_ms: int = 720
    # LLM knobs.
    temperature: float = 0.7
    max_tokens: int | None = 512
    llm_read_timeout_s: float = 15.0
    # TTS knobs passed straight through to VoxCPM.
    cfg_value: float = 2.0
    tts_temperature: float = 1.0
    # Speculative LLM on partials (v1.5) — off by default. Architectural
    # hook only; not wired in v1.
    speculative_llm_enabled: bool = False


# ---------------------------------------------------------------------------
# Client sink — route and tests share this interface
# ---------------------------------------------------------------------------


class ClientSink(Protocol):
    """Pluggable output — route wraps a WebSocket; tests record calls."""

    async def send_audio(self, pcm_s16le: bytes, sample_rate: int) -> None: ...
    async def send_event(self, payload: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# ConversationSession — the orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _TurnTimings:
    """Per-turn stage timestamps. Emitted with the final turn event for
    the observability report (exit criteria call for p50/p95 per stage)."""

    speech_end_wall: float | None = None
    stt_final_wall: float | None = None
    llm_first_token_wall: float | None = None
    tts_ttfb_wall: float | None = None
    turn_complete_wall: float | None = None


class ConversationSession:
    """Per-connection orchestrator. See ORCHESTRATION.md v2."""

    def __init__(
        self,
        *,
        cfg: ConversationConfig,
        sink: ClientSink,
        funasr: FunASRClient,
        vad: VadPuncClient,
        voxcpm: VoxCPMClient,
        llm: LLMClient,
        session_id: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._sink = sink
        self._funasr = funasr
        self._vad = vad
        self._voxcpm = voxcpm
        self._llm = llm
        self._session_id = session_id or uuid.uuid4().hex

        self._state: SessionState = SessionState.IDLE
        self._queue: asyncio.Queue[SessionEvent | None] = asyncio.Queue()

        # Per-turn state.
        self._pending_llm_task: asyncio.Task[None] | None = None
        self._pending_tts_task: asyncio.Task[None] | None = None
        self._turn_cancel: asyncio.Event = asyncio.Event()
        self._turn_ordinal: int = 0  # utterance_ordinal of the active turn
        self._current_utterance_ordinal: int = 0

        # LLM → TTS sentence pipeline.
        self._llm_token_buffer: str = ""  # accumulates tokens until a boundary
        self._sentence_queue: list[str] = []  # queued sentences waiting for TTS
        self._provisional_assistant_text: str = ""  # committed to history at clean finish
        self._llm_completed_for_turn: bool = False  # set when LLMEvent(llm_done) fires
        self._llm_error_pending: tuple[str, str] | None = None  # (code, msg) for E2

        # History (user + assistant messages).
        self._history: list[ChatMessage] = []
        if self._cfg.system_prompt:
            self._history.append(ChatMessage("system", self._cfg.system_prompt))

        # STT + audio input plumbing.
        self._pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        stt_cfg = STTStreamConfig(
            sample_rate=self._cfg.input_sample_rate,
            chunk_interval_ms=self._cfg.partial_interval_ms,
            speech_end_silence_ms=self._cfg.speech_end_silence_ms,
        )
        self._stt_session = STTStreamSession(
            funasr=self._funasr, vad=self._vad, cfg=stt_cfg, session_id=self._session_id
        )
        self._stt_session_task: asyncio.Task[None] | None = None
        self._stt_adapter_task: asyncio.Task[None] | None = None

        self._stopping: asyncio.Event = asyncio.Event()
        self._timings: _TurnTimings = _TurnTimings()

    # -- public surface ---------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> SessionState:
        return self._state

    async def feed_audio(self, pcm_s16le: bytes) -> None:
        """Client → server: one frame of PCM16LE at ``input_sample_rate``."""
        await self._pcm_queue.put(pcm_s16le)

    async def client_control(self, payload: dict[str, Any]) -> None:
        """Client → server JSON control frame (e.g. ``session.end``)."""
        await self._queue.put(WSControlEvent(payload=payload))

    async def stop(self) -> None:
        """Request orderly shutdown. Safe to call multiple times."""
        self._stopping.set()
        # Wake the pcm source and the event loop.
        await self._pcm_queue.put(None)
        await self._queue.put(None)

    async def run(self) -> None:
        """Top-level driver. Spawns producers, consumes events, cleans up."""
        self._stt_session_task = asyncio.create_task(
            self._stt_session.run(self._pcm_source()),
            name=f"conv-stt-{self._session_id[:8]}",
        )
        self._stt_adapter_task = asyncio.create_task(
            self._stt_adapter(), name=f"conv-stt-adapter-{self._session_id[:8]}"
        )
        try:
            while not self._stopping.is_set():
                ev = await self._queue.get()
                if ev is None:
                    break
                await self._dispatch(ev)
        finally:
            await self._shutdown()

    # -- producer: PCM source --------------------------------------------

    async def _pcm_source(self) -> AsyncIterator[bytes]:
        while True:
            pcm = await self._pcm_queue.get()
            if pcm is None:
                return
            yield pcm

    # -- producer: STT adapter -------------------------------------------

    async def _stt_adapter(self) -> None:
        """Fan ``STTStreamSession.events()`` into the main queue.

        The STT session emits plain dicts per event_stream_service; we
        translate to typed SessionEvent instances so _dispatch can use
        exhaustive isinstance.
        """
        try:
            async for ev in self._stt_session.events():
                et = ev.get("type")
                ord_ = int(ev.get("utterance_ordinal", 0))
                if et == "speech_start":
                    await self._queue.put(
                        VADEvent(kind="speech_start", utterance_ordinal=ord_, session_ms=int(ev.get("session_ms", 0)))
                    )
                elif et == "speech_end":
                    await self._queue.put(
                        VADEvent(kind="speech_end", utterance_ordinal=ord_, session_ms=int(ev.get("session_ms", 0)))
                    )
                elif et == "partial":
                    await self._queue.put(
                        STTEvent(
                            kind="transcript_partial",
                            utterance_ordinal=ord_,
                            text=ev.get("text", ""),
                            language=ev.get("language"),
                        )
                    )
                elif et == "final":
                    await self._queue.put(
                        STTEvent(
                            kind="transcript_final",
                            utterance_ordinal=ord_,
                            text=ev.get("text", ""),
                            punctuated_text=ev.get("punctuated_text", ""),
                            language=ev.get("language"),
                        )
                    )
                elif et == "error":
                    await self._queue.put(
                        STTEvent(
                            kind="stt_error",
                            utterance_ordinal=ord_,
                            code=ev.get("code"),
                            message=ev.get("message"),
                        )
                    )
        except asyncio.CancelledError:
            raise

    # -- dispatch ---------------------------------------------------------

    async def _dispatch(self, ev: SessionEvent) -> None:
        if isinstance(ev, VADEvent):
            await self._on_vad(ev)
        elif isinstance(ev, STTEvent):
            await self._on_stt(ev)
        elif isinstance(ev, LLMEvent):
            await self._on_llm(ev)
        elif isinstance(ev, TTSEvent):
            await self._on_tts(ev)
        elif isinstance(ev, WSControlEvent):
            await self._on_ws_control(ev)
        else:  # pragma: no cover — exhaustive by construction
            raise AssertionError(f"unhandled event type {type(ev).__name__}")

    # -- VAD --------------------------------------------------------------

    async def _on_vad(self, ev: VADEvent) -> None:
        if ev.kind == "speech_start":
            # §5 E5 serialisation — if a barge-in is applicable, run it
            # to completion before handling any other event. Because this
            # is the only consumer and we await the handler here, two
            # concurrent speech_start pushes from the STT adapter
            # serialise cleanly.
            if self._state in {SessionState.LLM_GENERATING, SessionState.TTS_SPEAKING}:
                await self._barge_in(new_ordinal=ev.utterance_ordinal)
            # In any case, reflect the new utterance ordinal.
            self._current_utterance_ordinal = ev.utterance_ordinal
            self._state = SessionState.USER_SPEAKING
            self._reset_status()
            await self._sink.send_event(
                {
                    "type": "input.speech_start",
                    "session_ms": ev.session_ms,
                    "utterance_ordinal": ev.utterance_ordinal,
                }
            )
            await self._send_status()
        elif ev.kind == "speech_end":
            # Drop out-of-order speech_end events for stale utterances.
            if ev.utterance_ordinal != self._current_utterance_ordinal:
                return
            self._timings.speech_end_wall = time.monotonic()
            await self._sink.send_event(
                {
                    "type": "input.speech_end",
                    "session_ms": ev.session_ms,
                    "utterance_ordinal": ev.utterance_ordinal,
                }
            )

    # -- STT --------------------------------------------------------------

    async def _on_stt(self, ev: STTEvent) -> None:
        # §7 — drop stale partials/finals.
        if ev.utterance_ordinal < self._current_utterance_ordinal:
            return
        if ev.kind == "transcript_partial":
            # Display only; no state change.
            await self._sink.send_event(
                {
                    "type": "transcript.partial",
                    "text": ev.text,
                    "language": ev.language,
                    "utterance_ordinal": ev.utterance_ordinal,
                }
            )
        elif ev.kind == "transcript_final":
            self._timings.stt_final_wall = time.monotonic()
            text = ev.punctuated_text or ev.text
            await self._sink.send_event(
                {
                    "type": "transcript.final",
                    "text": text,
                    "raw_text": ev.text,
                    "language": ev.language,
                    "utterance_ordinal": ev.utterance_ordinal,
                }
            )
            if _is_filler_or_empty(text):
                # §5 E7 — skip LLM on empty / filler-only transcripts.
                log.info(
                    "conversation.filler_skip",
                    session_id=self._session_id,
                    text=text[:80],
                )
                self._state = SessionState.IDLE
                await self._send_status()
                return
            # Kick off LLM for this turn.
            self._state = SessionState.LLM_GENERATING
            self._turn_ordinal = ev.utterance_ordinal
            self._turn_cancel = asyncio.Event()
            self._llm_token_buffer = ""
            self._sentence_queue = []
            self._provisional_assistant_text = ""
            self._llm_completed_for_turn = False
            self._llm_error_pending = None
            self._history.append(ChatMessage("user", text))
            messages = list(self._history)
            self._pending_llm_task = asyncio.create_task(
                self._run_llm(ordinal=self._turn_ordinal, messages=messages),
                name=f"conv-llm-{self._session_id[:8]}-{self._turn_ordinal}",
            )
            await self._send_status()
        elif ev.kind == "stt_error":
            await self._sink.send_event(
                {
                    "type": "error",
                    "code": ev.code or "stt_error",
                    "message": ev.message or "",
                }
            )

    # -- LLM --------------------------------------------------------------

    async def _run_llm(self, *, ordinal: int, messages: list[ChatMessage]) -> None:
        """LLM producer task. Pushes LLMEvents onto the main queue."""
        try:
            async for delta in self._llm.stream_chat(
                messages,
                model=self._cfg.llm_model,
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_tokens,
                read_timeout=self._cfg.llm_read_timeout_s,
            ):
                await self._queue.put(
                    LLMEvent(kind="llm_token", utterance_ordinal=ordinal, delta=delta)
                )
        except asyncio.CancelledError:
            raise
        except LLMTimeoutError as e:
            await self._queue.put(
                LLMEvent(
                    kind="llm_timeout",
                    utterance_ordinal=ordinal,
                    code=e.code,
                    message=e.message,
                )
            )
            return
        except LLMHTTPError as e:
            await self._queue.put(
                LLMEvent(
                    kind="llm_error",
                    utterance_ordinal=ordinal,
                    code=e.code,
                    message=f"{e.status}: {e.message}",
                )
            )
            return
        except Exception as e:  # noqa: BLE001
            await self._queue.put(
                LLMEvent(
                    kind="llm_error",
                    utterance_ordinal=ordinal,
                    code="llm_error",
                    message=str(e),
                )
            )
            return
        await self._queue.put(
            LLMEvent(kind="llm_done", utterance_ordinal=ordinal)
        )

    async def _on_llm(self, ev: LLMEvent) -> None:
        # Drop events from a turn that was cancelled out from under us.
        if ev.utterance_ordinal != self._turn_ordinal:
            return
        if ev.kind == "llm_token":
            if self._timings.llm_first_token_wall is None:
                self._timings.llm_first_token_wall = time.monotonic()
            self._provisional_assistant_text += ev.delta
            self._llm_token_buffer += ev.delta
            await self._sink.send_event(
                {"type": "response.text_delta", "delta": ev.delta}
            )
            # Drain complete sentences into the TTS queue.
            while True:
                sentence, remainder = _pop_complete_sentence(self._llm_token_buffer)
                if sentence is None:
                    break
                self._llm_token_buffer = remainder
                await self._queue_sentence_for_tts(sentence)
        elif ev.kind == "llm_done":
            self._llm_completed_for_turn = True
            # Flush any residual tail (no trailing whitespace → no
            # boundary hit) as the final sentence.
            tail = self._llm_token_buffer.strip()
            self._llm_token_buffer = ""
            if tail:
                await self._queue_sentence_for_tts(tail)
            await self._maybe_finish_turn()
        elif ev.kind == "llm_error":
            self._llm_error_pending = (ev.code or "llm_error", ev.message or "")
            await self._handle_llm_error_or_timeout()
        elif ev.kind == "llm_timeout":
            self._llm_error_pending = (ev.code or "llm_timeout", ev.message or "")
            await self._handle_llm_error_or_timeout()

    async def _handle_llm_error_or_timeout(self) -> None:
        """§5 E2: if TTS is in flight with a sentence already handed off,
        let it finish; only then emit the error. If no TTS task ever
        started, emit immediately and transition to IDLE."""
        if self._pending_tts_task is not None and not self._pending_tts_task.done():
            # TTS will finish naturally; its tts_done handler calls
            # _maybe_finish_turn, which sees the pending error and emits.
            return
        await self._emit_pending_llm_error()

    async def _emit_pending_llm_error(self) -> None:
        assert self._llm_error_pending is not None
        code, msg = self._llm_error_pending
        self._llm_error_pending = None
        # Drop any enqueued sentences that were never handed off.
        self._sentence_queue.clear()
        self._provisional_assistant_text = ""
        await self._sink.send_event(
            {"type": "error", "code": code, "message": msg}
        )
        self._state = SessionState.IDLE
        self._pending_llm_task = None
        await self._send_status()

    # -- TTS --------------------------------------------------------------

    async def _queue_sentence_for_tts(self, sentence: str) -> None:
        self._sentence_queue.append(sentence)
        await self._maybe_start_tts()

    async def _maybe_start_tts(self) -> None:
        if self._pending_tts_task is not None and not self._pending_tts_task.done():
            return
        if not self._sentence_queue:
            return
        sentence = self._sentence_queue.pop(0)
        self._state = SessionState.TTS_SPEAKING
        self._pending_tts_task = asyncio.create_task(
            self._run_tts(ordinal=self._turn_ordinal, text=sentence),
            name=f"conv-tts-{self._session_id[:8]}-{self._turn_ordinal}",
        )
        await self._send_status()

    async def _run_tts(self, *, ordinal: int, text: str) -> None:
        try:
            first_chunk = True
            async with self._voxcpm.synthesize_text_stream(
                text=text,
                sample_rate=self._cfg.output_sample_rate,
                voice_id=self._cfg.voice_id,
                cfg_value=self._cfg.cfg_value,
                temperature=self._cfg.tts_temperature,
            ) as frames:
                async for frame in frames:
                    if isinstance(frame, SynthesizeChunkFrame):
                        if first_chunk:
                            first_chunk = False
                        await self._queue.put(
                            TTSEvent(
                                kind="tts_chunk",
                                utterance_ordinal=ordinal,
                                pcm_s16le=frame.pcm_s16le,
                                sample_rate=frame.sample_rate,
                            )
                        )
                    elif isinstance(frame, SynthesizeDoneFrame):
                        await self._queue.put(
                            TTSEvent(kind="tts_done", utterance_ordinal=ordinal)
                        )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            await self._queue.put(
                TTSEvent(
                    kind="tts_error",
                    utterance_ordinal=ordinal,
                    code="tts_error",
                    message=str(e),
                )
            )

    async def _on_tts(self, ev: TTSEvent) -> None:
        if ev.utterance_ordinal != self._turn_ordinal:
            return
        if ev.kind == "tts_chunk":
            if self._timings.tts_ttfb_wall is None:
                self._timings.tts_ttfb_wall = time.monotonic()
            await self._sink.send_audio(ev.pcm_s16le, ev.sample_rate)
        elif ev.kind == "tts_done":
            self._pending_tts_task = None
            if self._sentence_queue:
                await self._maybe_start_tts()
            else:
                await self._maybe_finish_turn()
        elif ev.kind == "tts_error":
            await self._sink.send_event(
                {"type": "error", "code": ev.code or "tts_error", "message": ev.message or ""}
            )
            # Cancel any in-flight LLM since we can no longer speak its
            # output, then finish.
            if self._pending_llm_task and not self._pending_llm_task.done():
                self._pending_llm_task.cancel()
                with contextlib.suppress(BaseException):
                    await self._pending_llm_task
            self._pending_tts_task = None
            self._pending_llm_task = None
            self._provisional_assistant_text = ""
            self._sentence_queue.clear()
            self._state = SessionState.IDLE
            await self._send_status()

    # -- turn completion --------------------------------------------------

    async def _maybe_finish_turn(self) -> None:
        """Check whether the current turn has completely resolved.

        A turn is done when:
          - LLM has emitted llm_done
          - sentence_queue is empty (all sentences handed off)
          - pending_tts_task is None (last handoff finished)

        Only then do we append the assistant response to history and
        transition to IDLE.
        """
        if not self._llm_completed_for_turn:
            return
        if self._pending_tts_task is not None and not self._pending_tts_task.done():
            return
        if self._sentence_queue:
            return
        # §5 E2: if an LLM error/timeout was deferred because TTS had to
        # drain, emit it now instead of committing history.
        if self._llm_error_pending is not None:
            await self._emit_pending_llm_error()
            return
        # Clean finish.
        if self._provisional_assistant_text.strip():
            self._history.append(
                ChatMessage("assistant", self._provisional_assistant_text.strip())
            )
        self._provisional_assistant_text = ""
        self._pending_llm_task = None
        self._timings.turn_complete_wall = time.monotonic()
        await self._sink.send_event(
            {
                "type": "response.done",
                "turn_latency_ms": self._compute_turn_latency_ms(),
                "stage_timings_ms": self._stage_timings_ms(),
            }
        )
        self._state = SessionState.IDLE
        self._timings = _TurnTimings()
        await self._send_status()

    def _compute_turn_latency_ms(self) -> int | None:
        t = self._timings
        if t.speech_end_wall is None or t.tts_ttfb_wall is None:
            return None
        return int((t.tts_ttfb_wall - t.speech_end_wall) * 1000)

    def _stage_timings_ms(self) -> dict[str, int | None]:
        t = self._timings

        def _delta(a: float | None, b: float | None) -> int | None:
            return int((b - a) * 1000) if (a is not None and b is not None) else None

        return {
            "stt_final_after_speech_end_ms": _delta(t.speech_end_wall, t.stt_final_wall),
            "llm_first_token_after_stt_final_ms": _delta(
                t.stt_final_wall, t.llm_first_token_wall
            ),
            "tts_ttfb_after_llm_first_token_ms": _delta(
                t.llm_first_token_wall, t.tts_ttfb_wall
            ),
            "end_to_end_ms": self._compute_turn_latency_ms(),
        }

    # -- barge-in ---------------------------------------------------------

    async def _barge_in(self, *, new_ordinal: int) -> None:
        """§3: cancels fire first, then serial awaits (TTS → LLM)."""
        t0 = time.monotonic()
        log.info(
            "conversation.barge_in",
            session_id=self._session_id,
            old_ordinal=self._turn_ordinal,
            new_ordinal=new_ordinal,
            state=self._state.value,
        )
        # 1. Fire both cancels non-blocking so LLM cancel runs concurrently
        #    with TTS drain — load-bearing invariant per §3.
        if self._pending_tts_task is not None and not self._pending_tts_task.done():
            self._pending_tts_task.cancel()
        if self._pending_llm_task is not None and not self._pending_llm_task.done():
            self._pending_llm_task.cancel()
        self._turn_cancel.set()

        # 2. Await TTS — this is the gate on audio actually stopping.
        if self._pending_tts_task is not None:
            with contextlib.suppress(BaseException):
                await self._pending_tts_task
        # 3. Await LLM (usually already done by now).
        if self._pending_llm_task is not None:
            with contextlib.suppress(BaseException):
                await self._pending_llm_task

        # 4. Drop any buffered crossfade tail / sentence queue.
        self._sentence_queue.clear()
        self._llm_token_buffer = ""

        # 5. Emit interrupt event.
        barge_in_ms = int((time.monotonic() - t0) * 1000)
        await self._sink.send_event(
            {
                "type": "interrupt",
                "reason": "barge_in",
                "barge_in_ms": barge_in_ms,
                "new_utterance_ordinal": new_ordinal,
            }
        )
        log.info(
            "conversation.barge_in_complete",
            session_id=self._session_id,
            barge_in_ms=barge_in_ms,
        )

        # 6. Drop provisional assistant message (do not commit to history).
        self._provisional_assistant_text = ""
        self._llm_completed_for_turn = False
        self._llm_error_pending = None
        self._pending_tts_task = None
        self._pending_llm_task = None

        # 7. State transition happens in _on_vad after this returns —
        #    it sets state = USER_SPEAKING and updates the ordinal.

    # -- WS control -------------------------------------------------------

    async def _on_ws_control(self, ev: WSControlEvent) -> None:
        kind = ev.payload.get("type")
        if kind == "session.end":
            await self.stop()
        # Other control types (config updates etc.) land here in v1.1.

    # -- status / shutdown ------------------------------------------------

    def _reset_status(self) -> None:
        self._timings = _TurnTimings()

    async def _send_status(self) -> None:
        await self._sink.send_event({"type": "session.status", "state": self._state.value})

    async def _shutdown(self) -> None:
        """Clean teardown — §2. No orphan tasks survive this."""
        self._stopping.set()
        # Fire cancels on per-turn tasks.
        if self._pending_llm_task and not self._pending_llm_task.done():
            self._pending_llm_task.cancel()
        if self._pending_tts_task and not self._pending_tts_task.done():
            self._pending_tts_task.cancel()
        for task in (self._pending_llm_task, self._pending_tts_task):
            if task is None:
                continue
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(task, timeout=SHUTDOWN_GRACE_SECONDS)
        # End the PCM source so STT session exits cleanly.
        await self._pcm_queue.put(None)
        if self._stt_session_task is not None:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(
                    self._stt_session_task, timeout=SHUTDOWN_GRACE_SECONDS
                )
        if self._stt_adapter_task is not None:
            self._stt_adapter_task.cancel()
            with contextlib.suppress(BaseException):
                await self._stt_adapter_task
        self._pending_llm_task = None
        self._pending_tts_task = None
        self._stt_session_task = None
        self._stt_adapter_task = None


__all__ = [
    "ClientSink",
    "ConversationConfig",
    "ConversationSession",
    "FILLER_TOKENS",
    "LLMEvent",
    "SHUTDOWN_GRACE_SECONDS",
    "STTEvent",
    "SessionEvent",
    "SessionState",
    "TTSEvent",
    "VADEvent",
    "WSControlEvent",
]
