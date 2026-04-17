"""Streaming STT session orchestrator.

This module owns the rolling-buffer re-decode state machine that backs
WS /v1/stt/stream. Keeping it out of the route file makes it unit-testable
against fake workers (the route file focuses on WS framing + auth + logging).

State machine — in words:

    silent ── VAD speech_start ── speaking ──────► partials every 720ms
                                     │
                                     └── VAD speech_end ──► final (is_final=True)
                                                                │
                                                                ▼
                                                     back to silent, prev_text=""

Concurrency: each session runs two cooperating tasks inside ``run``:

- **ingest** consumes PCM frames from the WS, appends to ``audio_buffer``,
  and drives VAD events. VAD events emit ``speech_start`` / ``speech_end``
  on the out-bound channel and flip ``_state``.
- **partials** is a timer that fires every ``chunk_interval_ms``. When the
  session is ``speaking`` it slices the current utterance from
  ``audio_buffer`` and calls ``funasr.transcribe_rolling(is_final=False)``,
  then emits a ``partial`` event with the dropped-tail text.

Everything sent to the client flows through a single ``asyncio.Queue`` of
event dicts so the route's WS-write loop doesn't have to coordinate with
the ingest / partial tasks directly.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog
from larynx_shared.ipc.client_base import WorkerError

from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient

log = structlog.get_logger(__name__)


VadState = Literal["silent", "speaking"]
EventType = Literal["speech_start", "speech_end", "partial", "final", "heartbeat", "error"]


@dataclass
class STTStreamConfig:
    sample_rate: int = 16000
    language: str | None = None
    hotwords: tuple[str, ...] = ()
    chunk_interval_ms: int = 720
    speech_end_silence_ms: int = 300
    heartbeat_interval_ms: int = 1000  # gateway-side heartbeat (optional)
    itn: bool = True
    drop_tail_tokens: int = 5


@dataclass
class _Session:
    session_id: str
    cfg: STTStreamConfig
    audio_buffer: bytearray = field(default_factory=bytearray)
    prev_text: str = ""
    state: VadState = "silent"
    utterance_start_byte: int = 0
    utterance_start_ms: int = 0
    last_partial_wall: float = 0.0
    # Monotonically-increasing id for the current utterance. Incremented
    # on every speech_start; stamped onto speech_start / speech_end /
    # partial / final events so downstream consumers (notably the
    # ConversationSession orchestrator in M5) can drop events that
    # belong to an already-closed utterance. Starts at 0; the first
    # utterance is ordinal 1.
    utterance_ordinal: int = 0
    # Snapshots used by tests / logs.
    partials_emitted: int = 0
    finals_emitted: int = 0

    @property
    def bytes_per_sample(self) -> int:
        return 2  # int16 LE

    def bytes_at_ms(self, ms: int) -> int:
        return ms * self.cfg.sample_rate * self.bytes_per_sample // 1000


class STTStreamSession:
    """One session. Call ``run(ws_source)`` and consume ``events()``."""

    def __init__(
        self,
        *,
        funasr: FunASRClient,
        vad: VadPuncClient,
        cfg: STTStreamConfig,
        session_id: str | None = None,
    ) -> None:
        self._funasr = funasr
        self._vad = vad
        self._session = _Session(session_id=session_id or uuid.uuid4().hex, cfg=cfg)
        self._events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._closed = asyncio.Event()
        # Protects `prev_text` + state transitions between ingest and
        # partials tasks — both read/write these fields.
        self._lock = asyncio.Lock()
        # The partial-producer task may want to cancel an in-flight decode
        # when the session closes; we hold its task here.
        self._partial_task: asyncio.Task[None] | None = None

    @property
    def session_id(self) -> str:
        return self._session.session_id

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            yield item

    async def run(self, pcm_source: AsyncIterator[bytes]) -> None:
        """Drive the session until ``pcm_source`` closes or errors.

        On return, all partial/final events have been drained into the
        events queue and the VAD session is closed on the worker side.
        """
        await self._vad.vad_stream_open(
            session_id=self._session.session_id,
            sample_rate=self._session.cfg.sample_rate,
            speech_end_silence_ms=self._session.cfg.speech_end_silence_ms,
        )
        self._partial_task = asyncio.create_task(
            self._partials_loop(), name=f"stt-partials-{self._session.session_id[:8]}"
        )
        try:
            async for pcm in pcm_source:
                if not pcm:
                    continue
                await self._feed(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("stt_stream.session_error", session_id=self._session.session_id)
            await self._emit(
                {"type": "error", "code": "internal_error", "message": str(e)}
            )
        finally:
            # Final flush: tell VAD this is the end; close any open
            # utterance as final before we tear down.
            try:
                feed = await self._vad.vad_stream_feed(
                    session_id=self._session.session_id,
                    pcm_s16le=b"",
                    is_final=True,
                )
                await self._handle_vad_events(feed.events, feed.vad_state, feed.session_ms)
            except WorkerError:
                pass
            if self._session.state == "speaking":
                # VAD didn't close the utterance for us — emit a best-effort
                # final using all audio received so far as the end mark.
                end_ms = int(
                    1000
                    * (len(self._session.audio_buffer) // self._session.bytes_per_sample)
                    / self._session.cfg.sample_rate
                )
                await self._finalise_utterance(
                    end_ms=end_ms, ordinal=self._session.utterance_ordinal
                )
            try:
                await self._vad.vad_stream_close(session_id=self._session.session_id)
            except WorkerError:
                pass
            if self._partial_task is not None:
                self._partial_task.cancel()
                try:
                    await self._partial_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._closed.set()
            await self._events.put(None)
            log.info(
                "stt_stream.session_closed",
                session_id=self._session.session_id,
                audio_bytes=len(self._session.audio_buffer),
                partials=self._session.partials_emitted,
                finals=self._session.finals_emitted,
            )

    # -- internals -----------------------------------------------------------

    async def _emit(self, event: dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        await self._events.put(event)

    async def _feed(self, pcm: bytes) -> None:
        # Append to the session audio buffer and feed VAD.
        async with self._lock:
            self._session.audio_buffer.extend(pcm)
        try:
            feed = await self._vad.vad_stream_feed(
                session_id=self._session.session_id, pcm_s16le=pcm
            )
        except WorkerError as e:
            await self._emit(
                {"type": "error", "code": e.code, "message": e.message}
            )
            return
        await self._handle_vad_events(feed.events, feed.vad_state, feed.session_ms)

    async def _handle_vad_events(
        self,
        events: list[Any],
        vad_state: str,
        session_ms: int,
    ) -> None:
        for ev in events:
            if ev.event == "speech_start":
                async with self._lock:
                    self._session.state = "speaking"
                    self._session.utterance_start_ms = ev.session_ms
                    self._session.utterance_start_byte = self._session.bytes_at_ms(
                        ev.session_ms
                    )
                    self._session.prev_text = ""
                    self._session.last_partial_wall = time.monotonic()
                    self._session.utterance_ordinal += 1
                    ordinal = self._session.utterance_ordinal
                await self._emit(
                    {
                        "type": "speech_start",
                        "session_ms": ev.session_ms,
                        "utterance_ordinal": ordinal,
                    }
                )
            elif ev.event == "speech_end":
                async with self._lock:
                    ordinal = self._session.utterance_ordinal
                await self._emit(
                    {
                        "type": "speech_end",
                        "session_ms": ev.session_ms,
                        "utterance_ordinal": ordinal,
                    }
                )
                await self._finalise_utterance(end_ms=ev.session_ms, ordinal=ordinal)

    async def _finalise_utterance(self, end_ms: int, ordinal: int) -> None:
        async with self._lock:
            if self._session.state != "speaking":
                return
            start_byte = self._session.utterance_start_byte
            end_byte = self._session.bytes_at_ms(end_ms)
            if end_byte <= start_byte:
                self._session.state = "silent"
                return
            pcm = bytes(self._session.audio_buffer[start_byte:end_byte])
            utterance_ms = end_ms - self._session.utterance_start_ms
            self._session.state = "silent"
            self._session.prev_text = ""
        t_decode_start = time.monotonic()
        try:
            roll = await self._funasr.transcribe_rolling(
                pcm_s16le=pcm,
                sample_rate=self._session.cfg.sample_rate,
                language=self._session.cfg.language,
                hotwords=list(self._session.cfg.hotwords),
                itn=self._session.cfg.itn,
                prev_text="",
                is_final=True,
                drop_tail_tokens=self._session.cfg.drop_tail_tokens,
            )
        except WorkerError as e:
            await self._emit({"type": "error", "code": e.code, "message": e.message})
            return
        punct_text = roll.text
        punct_applied = False
        if roll.text.strip():
            try:
                punc = await self._vad.punctuate(text=roll.text, language=roll.language)
                punct_text = punc.text
                punct_applied = punc.applied
            except WorkerError:
                # Keep the un-punctuated text on punc failure; it's better
                # than dropping the final entirely.
                pass
        finalise_ms = int((time.monotonic() - t_decode_start) * 1000)
        self._session.finals_emitted += 1
        await self._emit(
            {
                "type": "final",
                "text": roll.text,
                "punctuated_text": punct_text,
                "punctuation_applied": punct_applied,
                "language": roll.language,
                "model_used": roll.model_used,
                "utterance_duration_ms": max(0, utterance_ms),
                "finalize_ms": finalise_ms,
                "utterance_ordinal": ordinal,
            }
        )

    async def _partials_loop(self) -> None:
        """Emit rolling partial transcripts every chunk_interval_ms.

        Sleeps between ticks to stay at the requested cadence even if a
        decode ran long; the rolling-buffer pattern tolerates slight
        overrun because each decode starts fresh from the utterance head.
        """
        interval = self._session.cfg.chunk_interval_ms / 1000.0
        while not self._closed.is_set():
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
            async with self._lock:
                if self._session.state != "speaking":
                    continue
                pcm = bytes(
                    self._session.audio_buffer[self._session.utterance_start_byte :]
                )
                prev_text = self._session.prev_text
                ordinal_at_decode_start = self._session.utterance_ordinal
            if len(pcm) < self._session.bytes_at_ms(80):
                # <80ms of audio in the utterance — too little for a decode
                # to produce a stable token; skip this tick.
                continue
            t0 = time.monotonic()
            try:
                roll = await self._funasr.transcribe_rolling(
                    pcm_s16le=pcm,
                    sample_rate=self._session.cfg.sample_rate,
                    language=self._session.cfg.language,
                    hotwords=list(self._session.cfg.hotwords),
                    itn=self._session.cfg.itn,
                    prev_text=prev_text,
                    is_final=False,
                    drop_tail_tokens=self._session.cfg.drop_tail_tokens,
                )
            except WorkerError as e:
                await self._emit(
                    {"type": "error", "code": e.code, "message": e.message}
                )
                continue
            decode_ms = int((time.monotonic() - t0) * 1000)
            now = time.monotonic()
            interval_ms = int((now - self._session.last_partial_wall) * 1000)
            self._session.last_partial_wall = now
            async with self._lock:
                if self._session.state != "speaking":
                    # Raced with a speech_end; drop the partial to keep
                    # the event ordering clean.
                    continue
                if self._session.utterance_ordinal != ordinal_at_decode_start:
                    # A new speech_start fired while the decode was in
                    # flight. The text we have describes audio that now
                    # belongs to a previous utterance — drop it rather
                    # than stamp a stale ordinal.
                    continue
                self._session.prev_text = roll.text
                self._session.partials_emitted += 1
                ordinal = self._session.utterance_ordinal
            await self._emit(
                {
                    "type": "partial",
                    "text": roll.text,
                    "language": roll.language,
                    "decode_ms": decode_ms,
                    "interval_ms": interval_ms,
                    "utterance_ordinal": ordinal,
                }
            )
