"""Streaming VAD session state for WS /v1/stt/stream.

The offline ``backend.segment()`` path used by the REST STT route can't drive
real-time partials — it wants the whole clip up front. This module adds a
session-keyed online VAD: 20ms PCM frames go in, ``speech_start`` /
``speech_end`` events come out. Session state lives entirely in Python so
multiple concurrent WS sessions stay isolated.

Two backends mirror the rest of the worker:

- ``MockStreamingVad``: RMS-threshold detector with a configurable silence
  debounce window. CPU-only, deterministic — exercises every gateway code
  path without pulling FunASR into the default install.

- ``FunasrStreamingVad``: wraps ``fsmn-vad`` in its online mode (the
  ``chunk_size`` / ``is_final`` / per-session ``cache`` dict API shown in
  FunASR's own ``demo_vad_online.py``). Each session owns its own cache;
  no cross-session contamination.

Both implement ``feed(pcm_s16le) -> (events, vad_state)``. The gateway
synthesizes heartbeat events from a timer; the worker itself only reports
state *changes*.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import structlog
from larynx_shared.ipc.messages import VadStreamEvent
from numpy.typing import NDArray

log = structlog.get_logger(__name__)


VadState = Literal["speaking", "silent"]


# ---------------------------------------------------------------------------
# Common session bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _SessionBase:
    session_id: str
    sample_rate: int
    speech_end_silence_ms: int
    # Total samples ever fed, used to compute session_ms for events and to
    # timestamp the current point-in-time when the gateway asks.
    samples_seen: int = 0
    state: VadState = "silent"

    def session_ms(self) -> int:
        return int(1000 * self.samples_seen / self.sample_rate) if self.sample_rate else 0


# ---------------------------------------------------------------------------
# Mock streaming VAD
# ---------------------------------------------------------------------------


@dataclass
class _MockSession(_SessionBase):
    silence_accumulated_ms: float = 0.0
    # How long we've been continuously "active" since last speech_start.
    # Emit speech_start only after ~60ms of sustained voice energy so a
    # single noisy frame doesn't trigger.
    active_accumulated_ms: float = 0.0


class StreamingVad(ABC):
    """Async-friendly streaming VAD session manager."""

    @abstractmethod
    async def open(self, session_id: str, sample_rate: int, speech_end_silence_ms: int) -> None: ...

    @abstractmethod
    async def feed(
        self, session_id: str, pcm_s16le: bytes, is_final: bool = False
    ) -> tuple[list[VadStreamEvent], VadState, int]:
        """Return (events, current_state, session_ms_after_feed)."""

    @abstractmethod
    async def close(self, session_id: str) -> None: ...


class MockStreamingVad(StreamingVad):
    """RMS-threshold streaming VAD for testing without FunASR."""

    # Window size used for per-frame RMS decisions.
    WINDOW_MS = 20
    # RMS threshold above which a window is considered voiced.
    RMS_THRESHOLD = 0.005
    # How long we require sustained voice before emitting speech_start
    # (debounces single-frame noise spikes).
    SPEECH_START_HOLD_MS = 40

    def __init__(self) -> None:
        self._sessions: dict[str, _MockSession] = {}

    async def open(self, session_id: str, sample_rate: int, speech_end_silence_ms: int) -> None:
        self._sessions[session_id] = _MockSession(
            session_id=session_id,
            sample_rate=sample_rate,
            speech_end_silence_ms=speech_end_silence_ms,
        )

    async def feed(
        self, session_id: str, pcm_s16le: bytes, is_final: bool = False
    ) -> tuple[list[VadStreamEvent], VadState, int]:
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(f"unknown vad session: {session_id}")

        if not pcm_s16le:
            return ([], s.state, s.session_ms())

        samples = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        events: list[VadStreamEvent] = []
        win = max(1, s.sample_rate * self.WINDOW_MS // 1000)
        for start in range(0, len(samples), win):
            frame = samples[start : start + win]
            if frame.size == 0:
                continue
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            ms = 1000 * frame.size / s.sample_rate
            s.samples_seen += frame.size
            voiced = rms > self.RMS_THRESHOLD
            if voiced:
                s.silence_accumulated_ms = 0.0
                s.active_accumulated_ms += ms
                if s.state == "silent" and s.active_accumulated_ms >= self.SPEECH_START_HOLD_MS:
                    s.state = "speaking"
                    events.append(
                        VadStreamEvent(
                            event="speech_start",
                            vad_state="speaking",
                            session_ms=s.session_ms(),
                        )
                    )
            else:
                s.active_accumulated_ms = 0.0
                if s.state == "speaking":
                    s.silence_accumulated_ms += ms
                    if s.silence_accumulated_ms >= s.speech_end_silence_ms:
                        s.state = "silent"
                        s.silence_accumulated_ms = 0.0
                        events.append(
                            VadStreamEvent(
                                event="speech_end",
                                vad_state="silent",
                                session_ms=s.session_ms(),
                            )
                        )
        if is_final and s.state == "speaking":
            # Close any open utterance at the trailing edge.
            s.state = "silent"
            events.append(
                VadStreamEvent(
                    event="speech_end",
                    vad_state="silent",
                    session_ms=s.session_ms(),
                )
            )
        return (events, s.state, s.session_ms())

    async def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Real streaming VAD — fsmn-vad online mode
# ---------------------------------------------------------------------------


@dataclass
class _FsmnSession(_SessionBase):
    # FunASR's online VAD keeps per-session rolling state in a plain dict;
    # ``cache={}`` on first call initialises it, and subsequent calls
    # mutate it in place.
    cache: dict[str, Any] = field(default_factory=dict)
    # Small byte buffer for leftover samples that don't fill a chunk.
    pending: bytes = b""


class FunasrStreamingVad(StreamingVad):
    """Wraps fsmn-vad's online mode.

    The FunASR online VAD API returns a list like
    ``[[start_ms, -1]]`` for a detected speech start, ``[[-1, end_ms]]`` for
    a detected end, or ``[]`` when nothing changed on that chunk. We flatten
    those into structured :class:`VadStreamEvent` objects.

    FunASR chunk_size is configured in ms and must be fed in exact multiples;
    we buffer any remainder on the session so WS clients can ship arbitrary
    frame sizes.

    Thread safety: ``AutoModel.generate`` mutates model-internal buffers
    (beyond the per-session ``cache`` dict) — concurrent calls from
    different WS sessions under ``asyncio.to_thread`` corrupt state and
    trigger ``IndexError`` inside ``GetFrameState``. We serialise every
    ``generate`` call with an async lock so concurrent sessions time-slice
    the model cleanly. Real VAD cost is ~5-20ms per 200ms chunk, so 4
    sessions still keep up with their audio clocks.
    """

    # FunASR expects 200ms chunks for its default fsmn-vad weights. Changing
    # this requires retraining, so we hard-code and buffer around it.
    CHUNK_SIZE_MS = 200

    def __init__(self, vad_model: Any) -> None:
        self._vad = vad_model
        self._sessions: dict[str, _FsmnSession] = {}
        self._model_lock = asyncio.Lock()

    async def open(self, session_id: str, sample_rate: int, speech_end_silence_ms: int) -> None:
        self._sessions[session_id] = _FsmnSession(
            session_id=session_id,
            sample_rate=sample_rate,
            speech_end_silence_ms=speech_end_silence_ms,
        )

    async def feed(
        self, session_id: str, pcm_s16le: bytes, is_final: bool = False
    ) -> tuple[list[VadStreamEvent], VadState, int]:
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(f"unknown vad session: {session_id}")

        buf = s.pending + pcm_s16le
        s.pending = b""
        chunk_samples = s.sample_rate * self.CHUNK_SIZE_MS // 1000
        chunk_bytes = chunk_samples * 2  # int16
        events: list[VadStreamEvent] = []

        def _process(pcm_chunk: bytes, final: bool) -> list[list[int]]:
            audio = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            res = self._vad.generate(
                input=audio,
                cache=s.cache,
                is_final=final,
                chunk_size=self.CHUNK_SIZE_MS,
            )
            if not res:
                return []
            return list(res[0].get("value") or [])

        async def _consume(pcm_chunk: bytes, final: bool) -> None:
            # Serialise concurrent sessions through the single fsmn-vad
            # model — see class docstring. Lock held only during the
            # generate() call so overhead stays minimal.
            async with self._model_lock:
                pairs = await asyncio.to_thread(_process, pcm_chunk, final)
            s.samples_seen += len(pcm_chunk) // 2
            for pair in pairs:
                events.extend(self._interpret(pair, s))

        # Feed full chunks synchronously (to_thread inside) so we respect
        # the model's window size. Remainder goes on the pending buffer.
        cursor = 0
        while len(buf) - cursor >= chunk_bytes:
            await _consume(buf[cursor : cursor + chunk_bytes], final=False)
            cursor += chunk_bytes
        s.pending = buf[cursor:]

        if is_final and s.pending:
            # Flush the tail with is_final so fsmn-vad closes any open
            # utterance.
            await _consume(s.pending, final=True)
            s.pending = b""
        elif is_final:
            # No tail but still need to notify fsmn-vad it's over.
            await _consume(b"\x00\x00" * (chunk_samples // 4), final=True)

        return (events, s.state, s.session_ms())

    def _interpret(self, pair: list[int], s: _FsmnSession) -> Iterator[VadStreamEvent]:
        """Translate a fsmn-vad ``[start, end]`` pair into our event shape.

        fsmn-vad online mode returns:
          - ``[start_ms, -1]``  → speech_start at start_ms
          - ``[-1, end_ms]``    → speech_end at end_ms
          - ``[start_ms, end_ms]`` → both (rare; emit both events)
        """
        if not pair or len(pair) != 2:
            return
        start_ms, end_ms = int(pair[0]), int(pair[1])
        if start_ms >= 0 and s.state != "speaking":
            s.state = "speaking"
            yield VadStreamEvent(event="speech_start", vad_state="speaking", session_ms=start_ms)
        if end_ms >= 0 and s.state != "silent":
            s.state = "silent"
            yield VadStreamEvent(event="speech_end", vad_state="silent", session_ms=end_ms)

    async def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


__all__ = [
    "FunasrStreamingVad",
    "MockStreamingVad",
    "StreamingVad",
    "VadState",
]


# Helper: used by the server dispatcher to construct the right streaming
# VAD for the currently-loaded backend. Kept in this module to avoid the
# server file growing a backend-type switch.
def build_streaming_vad(manager_mode: str, backend: Any) -> StreamingVad:
    if manager_mode == "real" and getattr(backend, "_vad", None) is not None:
        return FunasrStreamingVad(backend._vad)  # noqa: SLF001 — intentional peek
    return MockStreamingVad()


NDArrayF32 = NDArray[np.float32]  # re-export for tests / callers
