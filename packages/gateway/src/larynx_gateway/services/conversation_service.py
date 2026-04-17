"""Conversation orchestrator — M5 scaffold.

Implements the event-queue skeleton described in ORCHESTRATION.md v2 §1.1:

- ``SessionEvent`` is a tagged union of five concrete event types (VAD, STT,
  LLM, TTS, WS control). They are plain dataclasses; the tag lives in the
  class, not a discriminator field, so ``isinstance`` + exhaustive handler
  dispatch is type-checker friendly.
- ``ConversationSession`` owns a single ``asyncio.Queue[SessionEvent]``,
  a ``SessionState`` machine, and task handles for the per-turn LLM / TTS
  work. ``_run`` is the single consumer that dispatches events to handlers.
- The rule from §1.1 is: producers push events, the consumer mutates
  state. No producer touches ``_state``, history, or task handles.

This module is intentionally scaffold-only on initial landing — ``_dispatch``
asserts on every event type so we can land the shape and its tests, then
fill handlers in a subsequent commit. The assertion is load-bearing during
development: any pipeline that hits ``_dispatch`` with an unhandled event
will fail loudly rather than silently drop work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# ---------------------------------------------------------------------------
# SessionEvent — tagged union fan-in from all producers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VADEvent:
    """VAD lifecycle signal fan-in from ``STTStreamSession``.

    ``kind`` is either ``"speech_start"`` or ``"speech_end"``. The ordinal
    identifies which utterance the event belongs to; the orchestrator uses
    it to drop partials / finals that resolve on the wrong side of a
    ``speech_end`` (see §5 E4 + §7 of ORCHESTRATION.md).
    """

    kind: str  # "speech_start" | "speech_end"
    utterance_ordinal: int
    session_ms: int


@dataclass(frozen=True)
class STTEvent:
    """Transcript fan-in from ``STTStreamSession``.

    ``kind`` is ``"transcript_partial"`` | ``"transcript_final"`` | ``"stt_error"``.
    For errors, ``text`` may be empty and ``code`` / ``message`` describe
    the failure.
    """

    kind: str
    utterance_ordinal: int
    text: str = ""
    punctuated_text: str = ""
    language: str | None = None
    code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class LLMEvent:
    """Token stream fan-in from ``pending_llm_task``.

    ``kind`` is ``"llm_token"`` | ``"llm_done"`` | ``"llm_error"`` |
    ``"llm_timeout"``. The ordinal is the utterance that kicked off this
    LLM call — a handler can drop events whose ordinal no longer matches
    the active turn if a barge-in rotated the turn underneath.
    """

    kind: str
    utterance_ordinal: int
    delta: str = ""
    code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class TTSEvent:
    """PCM + lifecycle fan-in from ``pending_tts_task``.

    Carries PCM bytes for ``"tts_chunk"``. The ``sample_rate`` is redundant
    with session config but carrying it makes single-chunk tests simpler.
    """

    kind: str  # "tts_chunk" | "tts_done" | "tts_error"
    utterance_ordinal: int
    pcm_s16le: bytes = b""
    sample_rate: int = 0
    code: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class WSControlEvent:
    """Client-sent JSON control message fan-in.

    ``payload`` is the raw parsed JSON object. The handler decides what to
    do with it; common ``type`` values in v1 are ``"session.end"``,
    ``"config.update"``.
    """

    payload: dict[str, Any] = field(default_factory=dict)


# Sum type alias. All five classes appear here so static analysers can
# exhaustively check handler dispatch.
SessionEvent = VADEvent | STTEvent | LLMEvent | TTSEvent | WSControlEvent


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class SessionState(StrEnum):
    """Per-session state from ORCHESTRATION.md v2 §1.

    ``USER_SPEAKING`` collapses the PRD's (USER_SPEAKING, TRANSCRIBING) pair
    because rolling-buffer re-decode runs unconditionally during user
    speech; there is no observable transition between them.
    """

    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    LLM_GENERATING = "llm_generating"
    TTS_SPEAKING = "tts_speaking"


# ---------------------------------------------------------------------------
# ConversationSession — the orchestrator
# ---------------------------------------------------------------------------


@dataclass
class ConversationConfig:
    """Per-connection knobs. Additional fields land with later commits.

    v1 scope: voice, LLM model, system prompt, VAD sensitivity.
    Speculative-LLM (v1.5) gets its ``speculative_llm_enabled`` flag in the
    ConversationSession implementation commit — not here — so the scaffold
    stays minimal.
    """

    voice_id: str | None = None
    llm_model: str = "anthropic/claude-haiku-4.5"
    system_prompt: str = ""
    sample_rate: int = 16000
    # Minimum utterance ms required before the final transcript is routed
    # to the LLM — filters out short filler + noise. Full filter list
    # (FILLER_TOKENS) lands with the implementation.
    min_utterance_ms: int = 120


class ConversationSession:
    """Scaffold: owns the event queue, state machine, task handles.

    ``run`` drives the single consumer loop. ``enqueue`` is the producer
    entry point. Handlers are not wired in this commit — ``_dispatch``
    asserts on every event so the scaffold's shape is exercised by tests
    but any attempt to run a real conversation fails loudly with a clear
    message pointing at the missing handler.
    """

    def __init__(self, *, cfg: ConversationConfig, session_id: str) -> None:
        self._cfg = cfg
        self._session_id = session_id
        self._state: SessionState = SessionState.IDLE
        self._queue: asyncio.Queue[SessionEvent | None] = asyncio.Queue()
        # Per-turn task handles. Populated by the implementation commit.
        # The types are intentionally loose to avoid premature coupling.
        self._pending_llm_task: asyncio.Task[None] | None = None
        self._pending_tts_task: asyncio.Task[None] | None = None
        # Per-turn cancellation token. Reset at USER_SPEAKING →
        # LLM_GENERATING entry so a prior-turn signal can't leak forward.
        self._turn_cancel: asyncio.Event = asyncio.Event()
        # The current utterance the orchestrator is tracking — used by
        # handlers to dedupe stale STT events per §7.
        self._current_utterance_ordinal: int = 0
        self._conversation_history: list[dict[str, str]] = []
        self._stopping: asyncio.Event = asyncio.Event()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> SessionState:
        return self._state

    async def enqueue(self, event: SessionEvent) -> None:
        """Producer entry point. Push an event for the consumer to handle.

        Producers must never mutate session state directly — they only push
        here. This is the invariant that makes concurrent producers safe.
        """
        await self._queue.put(event)

    async def stop(self) -> None:
        """Request orderly shutdown of ``run``.

        Pushes a sentinel (``None``) so the consumer loop wakes up. Task
        cleanup + worker stream close lives in the implementation commit.
        """
        self._stopping.set()
        await self._queue.put(None)

    async def run(self) -> None:
        """Single-consumer event loop.

        Per §1.1: exactly one consumer, handlers are awaited to completion
        before the next ``queue.get()``, no event is handled concurrently
        with another. That property is what §5 E5 (re-entrant barge-in)
        relies on — a second ``VADEvent(speech_start)`` arriving during
        the first barge-in handler sits in the queue until the handler
        returns.
        """
        while not self._stopping.is_set():
            ev = await self._queue.get()
            if ev is None:
                return
            await self._dispatch(ev)

    async def _dispatch(self, ev: SessionEvent) -> None:
        """Scaffold stub — asserts on every event.

        Wired handlers land in the implementation commit. The assertion
        guarantees no event type silently no-ops during development.
        """
        raise AssertionError(
            f"ConversationSession._dispatch: no handler wired for {type(ev).__name__} "
            f"(kind={getattr(ev, 'kind', None)!r}). This scaffold rejects all events "
            "by design — the M5 implementation commit fills in handlers."
        )


__all__ = [
    "ConversationConfig",
    "ConversationSession",
    "LLMEvent",
    "STTEvent",
    "SessionEvent",
    "SessionState",
    "TTSEvent",
    "VADEvent",
    "WSControlEvent",
]
