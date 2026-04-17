# Conversation Orchestration — M5 Design (v2)

Scope: `WS /v1/conversation`. Drives VAD + streaming STT + OpenRouter LLM +
streaming TTS as one full-duplex, interruptible pipeline. Target p50 turn
latency ≤ 700ms (PRD §6). LLM is the dominant cost; everything else must
minimise added serial latency and cancel cleanly on barge-in.

The orchestrator lives in `services/conversation_service.py`. It composes the
existing `STTStreamSession` (which already owns VAD + rolling Fun-ASR) rather
than re-implementing either, and wires LLM + TTS onto its event stream.

---

## Changes from v1

- **§3 barge-in cancels now parallelised (Issue 1).** Both `.cancel()` calls
  fire before any `await`, so LLM cancellation runs concurrently with the TTS
  drain instead of serialising behind it.
- **§1.1 Event queue made explicit (Issue 7).** Single-consumer
  `asyncio.Queue[SessionEvent]` is now a named architectural element; all
  state transitions happen inside its single consumer loop.
- **§7 Implementation prerequisites added (Issue 3).** STT ordinal stamping
  is a cross-component change on `STTStreamSession` and is called out as a
  pre-requisite before orchestrator code starts.
- **§5 E5 re-entrant barge-in resolved via Option B (Issue 6).** The event
  queue's single-consumer loop awaits each handler to completion before the
  next `get()`, so concurrent VAD triggers serialise for free. Explicit test
  requirement documented.
- **Inline fixes.** §3 preamble wording (Issue 2); §5 E2 LLM-timeout decision
  — let the in-flight TTS sentence finish before erroring (Issue 4); §5 E7
  empty-transcript normalisation + filler-token config (Issue 5); §2 grace
  constant, §3 no-op note, §5 E6 timing claim, §4 v1.5 cadence phrasing
  (nits).

---

## 1. Per-session state machine

Exactly one session per WS connection. One `ConversationSession` owns all
pipeline state and holds a single `_state: SessionState` enum. State
transitions happen only in one place — the event consumer in
`ConversationSession._run` — so the machine is race-free without a lock
around the state variable itself. See §1.1 for why that invariant holds.

```
                ┌──────────────────────────────────────────────┐
                │  IDLE                                        │
                │  (no user speech, no model work in flight)   │
                └──────────────┬───────────────────────────────┘
                               │ VAD speech_start
                               ▼
                ┌──────────────────────────────────────────────┐
                │  USER_SPEAKING                               │
                │  (VAD says speech active; STT rolling loop   │
                │   emits transcript.partial every ~720ms —    │
                │   display-only in v1)                        │
                └──────────────┬───────────────────────────────┘
                               │ VAD speech_end
                               ▼  (final transcript resolves on same event)
                ┌──────────────────────────────────────────────┐
                │  RESPONDING  (superstate)                    │
                │                                              │
                │   ┌────────────────────────────────────┐     │
                │   │ LLM_GENERATING                     │     │
                │   │ pending_llm_task streaming; tokens │     │
                │   │ accumulate in sentence buffer      │     │
                │   └────────────┬───────────────────────┘     │
                │                │ first sentence boundary     │
                │                ▼                             │
                │   ┌────────────────────────────────────┐     │
                │   │ TTS_SPEAKING                       │     │
                │   │ pending_tts_task streaming PCM     │     │
                │   │ (overlaps with LLM tail)           │     │
                │   └────────────┬───────────────────────┘     │
                └────────────────┼─────────────────────────────┘
                                 │ LLM done AND TTS done
                                 ▼
                              IDLE
```

Barge-in is an edge OUT OF `USER_SPEAKING → RESPONDING.*`: if a new
`speech_start` arrives while `_state ∈ {LLM_GENERATING, TTS_SPEAKING}`, we
transition back to `USER_SPEAKING` via the cancellation sequence in §3.

The assistant message is **provisional** while in `RESPONDING` — it's only
appended to `conversation_history` at the `→ IDLE` edge (clean finish).
Interrupted turns never enter history.

USER_SPEAKING and TRANSCRIBING in the PRD description collapse into one
state here because rolling-buffer re-decode is an unconditional background
behaviour during user speech — there is no transition between "speaking" and
"transcribing"; partials just flow.

### 1.1 Event queue

Everything that drives state passes through a **single
`asyncio.Queue[SessionEvent]`** owned by `ConversationSession`. There is
exactly one consumer (`_run`) and multiple producers. This is the load-
bearing architectural invariant of the whole module.

**`SessionEvent` is a tagged union.** Concrete variants:

- `VADEvent` — `speech_start`, `speech_end`. Carries `utterance_ordinal`
  (see §7).
- `STTEvent` — `transcript_partial`, `transcript_final`, `stt_error`.
  Carries `utterance_ordinal`.
- `LLMEvent` — `llm_token`, `llm_done`, `llm_error`, `llm_timeout`.
- `TTSEvent` — `tts_chunk` (carries `pcm_s16le: bytes`), `tts_done`,
  `tts_error`.
- `WSControlEvent` — client-sent JSON control messages (`session.end`,
  config updates, etc.).

**Producers** (all run as tasks owned by the session; all push to the
queue, none mutate state):

- **STT adapter task**: one long-lived task that iterates
  `STTStreamSession.events()` and fans each event into `VADEvent` or
  `STTEvent` on the queue. Lives for the whole connection.
- **`pending_llm_task`**: drives `llm_client.stream_chat`; pushes
  `llm_token` / `llm_done` / `llm_error` / `llm_timeout`. Per-turn.
- **`pending_tts_task`**: drives `voxcpm.synthesize_text_stream`; pushes
  `tts_chunk` / `tts_done` / `tts_error`. Per-turn.
- **WS receive task**: reads client frames; pushes `WSControlEvent` for
  JSON messages, discards binary (clients don't send binary after config
  in v1).

**Consumer**: `_run` is a single loop, roughly:

```python
while not stopping:
    ev = await queue.get()
    await self._dispatch(ev)   # handler may transition state
```

`_dispatch` awaits the per-event handler to **completion** before
returning to `queue.get()`. Because there is exactly one consumer and
handlers are awaited inline (never spawned as tasks), two events never
execute their handlers concurrently. This is what §5 E5's Option B relies
on.

**Consequence**: TTS PCM frames flow through the queue as `TTSChunkEvent`
before being written to the WS by the handler. Not the throughput
architecture you'd draw for a high-fan-out system, but at v1 concurrency
(8 sessions, ~20ms frames) queue latency is noise. Resist the urge to
optimise before measuring.

**Rule**: no producer ever calls a method that mutates `_state`, history,
or task handles. Producers only push. Violations must be caught in review.

---

## 2. Cancellation token propagation

Each session owns a **per-turn** `asyncio.Event` called `turn_cancel`. It is
the single cancellation signal for everything done on behalf of the current
assistant response:

- `pending_llm_task`: a task wrapping `llm_client.stream_chat(...)`. It reads
  from an `httpx` SSE stream; on `turn_cancel.set()` the outer task gets
  `.cancel()` which raises `asyncio.CancelledError` inside the SSE loop,
  closes the httpx connection, and returns.
- `pending_tts_task`: a task that owns `voxcpm.synthesize_text_stream(...)`
  as an `async with` context manager. Cancelling the task exits the context
  manager, which sends `CancelStreamRequest` to the worker (already
  implemented in `InProcessWorkerClient.stream`). No PCM frames leak after
  that point because the TTS task is the only producer of `TTSChunkEvent`
  and handler dispatch stops enqueueing them as soon as the task is dead.
- Sentence buffer and token → sentence fan-out: plain in-memory state owned
  by the LLM task; dies with it.

`turn_cancel` is reset at every `USER_SPEAKING → LLM_GENERATING` entry (new
`asyncio.Event`) so a cancelled turn's signal can't leak into the next turn.

The STT stream session (`STTStreamSession`) is **not** per-turn — it lives
for the entire connection and survives barge-in. Its own cancellation is
tied to WS disconnect, not `turn_cancel`.

Clean shutdown order (WS disconnect): flip `turn_cancel` → await both task
handles with `SHUTDOWN_GRACE_SECONDS` (module constant, default **2s**) →
cancel STT session → close VAD stream on worker → drain event queue →
return. The 2s value is sized to cover a worst-case in-flight SSE read
waiting on TCP (~1s typical keep-alive round-trip) plus one event-loop
tick of slack; it is not a latency budget. Override per-deployment via
config if needed.

---

## 3. Barge-in sequence (authoritative order)

Triggered the moment the STT adapter pushes a `VADEvent(speech_start)`
while `_state ∈ {LLM_GENERATING, TTS_SPEAKING}`. The barge-in handler runs
as a single awaited coroutine inside `_run`'s event-processing loop; no
other events are processed until this coroutine returns (see §1.1). That
is what makes the sequence below atomic from the session's point of view.

Steps:

1. **Fire both cancels (non-blocking).**
   - 1a. `pending_tts_task.cancel()`
   - 1b. `pending_llm_task.cancel()`
2. **Await TTS** with `suppress(CancelledError)`. This blocks until PCM
   emission has stopped. Budget: from the VAD speech_start event to the
   last WS `send_bytes` call ≤ 100ms.
3. **Await LLM** with `suppress(CancelledError)`. By this point it is
   usually already done because its cancel ran in parallel with step 2.
4. **Discard any pending crossfade tail** held by the TTS adapter. If
   there is none (common case — cancellation typically interrupts
   mid-chunk), this step is a no-op.
5. **Emit `interrupt` event** to the client as a JSON text frame
   (`{"type": "interrupt", "reason": "barge_in"}`). The client uses it to
   stop the audio element and mark the provisional assistant message with
   a `…` truncation marker.
6. **Drop provisional assistant message.** Do NOT append the partial LLM
   response to `conversation_history`. The next LLM call sees history as
   if the interrupted turn never happened.
7. **Transition `_state ← USER_SPEAKING`.** The STTStreamSession is
   already re-entering `speaking`; the rolling-partials loop resumes
   naturally.

**Load-bearing invariant**: issuing both `.cancel()` calls before awaiting
either is the point. It lets LLM cancellation race against TTS drain
instead of serialising behind it, which saves both OpenRouter spend and
wall-clock before the system is ready for the next turn. User-perceived
latency is unchanged — it is gated on step 2, the TTS drain.

---

## 4. Partials handling

v1: **display-only.** Every `STTEvent(transcript_partial)` is forwarded to
the client as `{"type": "transcript.partial", "text": ...}`. The LLM is
NOT triggered until `speech_end → transcript_final`.

Why display-only for v1: PRD §6 calls out speculative-LLM-on-partials as a
v1.5 unlock because partial-stability detection, speculation cancel-and-
re-issue, and OpenRouter request accounting are each their own small
project. Shipping them together with barge-in would triple the surface
area of things that can race.

v1.5 (architecture support, not wired): `ConversationSession` already owns
a `pending_llm_task` handle and a `turn_cancel` token. To add speculative
LLM, a `StabilityTracker` watches partials, fires `pending_llm_task` early
when "the partial is token-identical across ≥ 2 consecutive emissions
(≈1440ms at the current 720ms cadence)", and on actual final either (a)
keeps the task if final == speculated text, or (b) `turn_cancel.set()` +
restart if not. No state-machine changes required — speculation just
moves the entry edge to `LLM_GENERATING` earlier. We will add a
`speculative_llm_enabled: bool = False` config field now so the v1.5 flip
is additive.

---

## 5. Edge cases

**E1. User speaks before previous TTS finishes.** Already the barge-in
case above — this is the normal path, not an error. Order is §3.

**E2. Network stall during LLM streaming.** `stream_chat` uses an httpx
read timeout (default 15s, configurable per-session). On timeout, the
LLM task pushes `LLMEvent(llm_timeout)`. **v1 behaviour**: if a TTS task
is in flight (a sentence had already been handed off), let it run to its
natural `TTSEvent(tts_done)`; only then emit
`{"type": "error", "code": "llm_timeout"}` and drop the provisional
assistant message. If no TTS was ever started (timeout fired before the
first sentence boundary), emit the error immediately and cancel nothing.
Either way, transition `→ IDLE`. WS session stays alive — the next user
utterance starts a fresh turn. This preserves user-perceivable phrase
integrity on the most common timeout shape (slow first token, then a
stall mid-response). It is distinct from WS close.

**E3. Network stall during TTS streaming.** `voxcpm.synthesize_text_stream`
already has an `idle_timeout` (60s default). On idle timeout the context
manager raises; the TTS task pushes `TTSEvent(tts_error)`. Handler emits
`{"type": "error", "code": "tts_idle_timeout"}` and drops the
provisional message. LLM task is cancelled via `turn_cancel` in the same
handler. `→ IDLE`.

**E4. STT partial arrives after speech_end (race).** Handled by the
utterance-ordinal check (§7).

**E5. Barge-in during cancellation (barge-in-of-barge-in).** Resolved by
the event-queue serialisation in §1.1: the first barge-in handler runs
to completion (steps 1–7 of §3) before `_run` calls `queue.get()` again.
A second `VADEvent(speech_start)` arriving during that window sits in
the queue until the handler returns, then gets dispatched against the
newly-established `USER_SPEAKING` state — at which point it is not a
barge-in at all, just a redundant speech_start (ignored if already in
USER_SPEAKING with the same ordinal; see §7).

Test requirement: fire two `VADEvent(speech_start)` events into the
queue within one event-loop tick of each other and assert that exactly
one `interrupt` event is emitted on the WS. This validates the
serialisation invariant, not just the ordinal dedupe.

**E6. LLM completes between VAD speech_start (barge-in detected) and our
cancel call.** Race window is narrow — at most one event-loop tick. If
LLM finished, `pending_llm_task` is already done; `.cancel()` returns
`False` and the `await` returns immediately. TTS may have queued the
final sentence; the same flow cancels it. No special handling needed —
the normal path works.

**E7. Speech_end with empty / filler-only transcript.** "Empty" is
underspecified by default because Fun-ASR emits noise-driven false
finals (single fillers, throat clears). Definition:

```python
normalized = text.strip().lower().rstrip(string.punctuation)
is_empty = len(normalized) == 0 or normalized in FILLER_TOKENS
```

`FILLER_TOKENS` is a union of per-language sets loaded from a TOML/YAML
config file (`config/conversation/filler_tokens.toml`) so the set can be
tuned without code changes. Initial seed:

```toml
[filler_tokens]
en = ["uh", "um", "hmm", "mm", "ah"]
zh = ["嗯", "呃", "啊"]
ja = ["えー", "あのー", "ええと"]
```

When `is_empty`, skip the LLM call, emit nothing to the client (no error —
this is a normal no-op), transition `USER_SPEAKING → IDLE`. Don't
generate an assistant response for silence or filler-only input.

**E8. Client disconnects mid-TTS.** Normal `WebSocketDisconnect` path:
same as clean shutdown, §2 cleanup order. No `interrupt` event emitted
(socket is dead).

---

## 6. Things this doc deliberately does NOT specify

- Audio format / sample rate on the wire (decided in route, not orchestrator).
- OpenRouter auth / retry (lives in `llm_client.py`).
- Sentence boundary detection regex (tactical; in LLM task body).
- WS protocol framing details (route concern).

These are listed so reviewers know the gap is intentional.

---

## 7. Implementation prerequisites

Must land **before** any `conversation_service.py` code is written.

### 7.1 STTStreamSession utterance ordinal

`STTStreamSession` currently has a defensive `state != "speaking"` check
when emitting partials (`stt_stream_service.py:335`) but does not tag
events with any identifier of which utterance they belong to. The
orchestrator needs this to drop partials that resolve on the wrong side
of a `speech_end`.

Required changes (in the `funasr_worker`/gateway STT stream module, not
in the conversation package):

- Add `utterance_ordinal: int` to `_Session`. Starts at 0. Increments by
  1 on every `speech_start` handled in `_handle_vad_events`.
- Stamp `utterance_ordinal` on every emitted event of these types:
  `speech_start`, `speech_end`, `partial`, `final`. (Error and heartbeat
  events do not need it.)
- The ordinal is captured at the point of utterance start; partials
  emitted by `_partials_loop` read it under the same lock that guards
  the `speaking` state check, so a partial cannot be stamped with an
  ordinal that has already been superseded by a new `speech_start`.

### 7.2 ConversationSession consumption

- Track `current_utterance_ordinal: int` on `ConversationSession`,
  updated on every `VADEvent(speech_start)` received from the adapter.
- On any `STTEvent` whose `utterance_ordinal <
  current_utterance_ordinal`, drop silently (no client emit, no state
  change).
- On duplicate `VADEvent(speech_start)` with the same ordinal as
  `current_utterance_ordinal` while already in `USER_SPEAKING`, ignore
  (idempotent). Used to keep E5's redundant-speech_start path clean.

### 7.3 Test placement

- Ordinal stamping tests live in the STT package (unit tests against
  `STTStreamSession`, no conversation session involved).
- Drop-stale-partial tests live in the conversation package.

Keeping the responsibilities split this way means other consumers of
`STTStreamSession` (e.g. the existing `WS /v1/stt/stream` route) get the
ordinal stamping for free with no behavioural change — ordinals are an
additive field the STT route can include or ignore.
