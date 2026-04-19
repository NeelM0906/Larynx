# Conversation tab — design doc

Lives at `/conversation`. Replaces the "Coming soon" placeholder with
a real full-duplex client against `WS /v1/conversation`.

This doc is the readback of the backend protocol I extracted before
writing any UI code, plus the state machine and barge-in semantics
I'll implement on top of it.

## Backend contract (from `packages/gateway/src/larynx_gateway/routes/conversation.py`)

### Connection

`WS /v1/conversation` — bearer token via `?token=` query param (matches
the gateway's `require_ws_bearer_token`).

### Client → server

1. **First message (text, JSON)** — `_ConversationConfigFrame`:

   ```json
   {
     "type": "config",
     "voice_id": "<optional uuid>",
     "llm_model": null,
     "system_prompt": "",
     "input_sample_rate": 16000,
     "output_sample_rate": 24000,
     "speech_end_silence_ms": 300,
     "partial_interval_ms": 720,
     "temperature": 0.7,
     "max_tokens": 512,
     "llm_read_timeout_s": 15.0
   }
   ```

2. **Binary frames** — 16-bit little-endian PCM mono at
   `input_sample_rate` (16000 Hz). Chunk size is not constrained by
   the protocol; we'll send ~20–40 ms chunks (640–1280 samples).

3. **Text control JSON** — only one type is honoured today:
   `{"type": "session.end"}`. No client-side explicit interrupt
   message — barge-in is server-initiated on VAD.

### Server → client

1. **Binary frames** — PCM16 LE mono at `output_sample_rate` (24000 Hz).
   These are raw TTS chunks; play them back verbatim.

2. **Text JSON events:**

   | type | payload | when |
   |---|---|---|
   | `session.status` | `{state: "idle"\|"user_speaking"\|"llm_generating"\|"tts_speaking"}` | every state transition |
   | `input.speech_start` | `{}` | VAD detects user speech |
   | `input.speech_end` | `{}` | VAD silence window elapsed |
   | `transcript.partial` | `{text, language, utterance_ordinal}` | rolling Fun-ASR partial |
   | `transcript.final` | `{text, raw_text, language, utterance_ordinal}` | utterance finalised (punctuated) |
   | `response.text_delta` | `{delta}` | LLM token stream |
   | `interrupt` | `{reason: "barge_in", barge_in_ms, new_utterance_ordinal}` | server cancelled provisional response |
   | `response.done` | `{turn_latency_ms, stage_timings_ms}` | assistant turn finished |
   | `error` | `{code, message}` | any stage error |

## Client state machine

Mirrors `SessionState` on the server so the UI and backend stay in
step. The server broadcasts `session.status` on every transition, so
the client is effectively a display of server state.

```
                ┌────────┐
                │  idle  │◄──────────────┐
                └────┬───┘               │
 input.speech_start  │        response.done / interrupt ack
                     ▼                   │
           ┌──────────────────┐          │
           │  user_speaking   │          │
           └────────┬─────────┘          │
 transcript.final   │                    │
                    ▼                    │
          ┌───────────────────┐          │
          │  llm_generating   │          │
          └────────┬──────────┘          │
 first tts chunk   │                     │
                   ▼                     │
           ┌─────────────────┐           │
           │  tts_speaking   │───────────┘
           └─────────────────┘
                   │
  speech_start     │  (server-initiated barge-in)
                   ▼
           ┌───────────────┐
           │   interrupt   │
           └───────┬───────┘
                   │  → user_speaking (new ordinal)
                   ▼
```

The UI shows a pill with the current state (listening / thinking /
speaking) driven by `session.status`. We don't try to predict state
locally — always follow the server.

## Audio capture (mic → server PCM16 @16k)

1. `getUserMedia({ audio: { channelCount: 1 } })`.
2. Pipe the stream into an `AudioContext` via `createMediaStreamSource`.
3. An `AudioWorklet` (fallback: `ScriptProcessorNode`) emits Float32
   frames at the `AudioContext` sample rate (typically 48000 Hz).
4. Downsample linearly to 16000 Hz, then Int16-encode.
5. Send as WS binary frame every ~40 ms.

The first chunk warm-starts the VAD state machine on the server;
after that it's a continuous stream until the user hits "End
conversation" (which sends `session.end`).

## Audio playback (server PCM16 @24k → user's ears)

`<audio>` doesn't give fast-enough stop for barge-in. Use the Web
Audio API.

- One `AudioContext` for output (separate from input so sample rates
  can differ cleanly).
- For each inbound binary chunk:
  - Decode PCM16 LE → Float32.
  - Resample to the output `AudioContext` rate if it differs.
  - Schedule via an `AudioBufferSourceNode` chained to a shared
    `GainNode`.
- Track the "next start time" so chunks play gap-lessly at their
  natural 24 kHz cadence.
- On `interrupt` event (or "Stop AI" button):
  1. Fade `GainNode` to 0 over 5 ms (`linearRampToValueAtTime`).
  2. Stop all queued source nodes; reset scheduling clock.
  3. Restore gain after a 20 ms guard window.

## Barge-in

Server-initiated. The client doesn't send an explicit "interrupt
please." Flow:

1. User starts speaking during `tts_speaking`.
2. Server's VAD fires `input.speech_start`. Server transitions state
   internally, stops provisional LLM/TTS, emits `interrupt` event
   (and fresh `session.status`).
3. Client, on receipt of `interrupt`, fades out local playback and
   zeros the playback queue so the user isn't hearing the last few
   buffered chunks.
4. UI flips from "speaking" pill to "listening" pill via the
   `session.status` event that follows.

The "Stop AI" button on the UI is a **local-playback** mute only —
the protocol has no explicit cancel control. It fades local audio
and clears the queue so the user hears silence; the server's LLM
stage is not cancelled unless the user also starts talking.
Documented as such in the UI copy.

## Reconnect

On WS close while state ≠ idle: show a "Connection lost —
reconnecting…" banner and auto-retry up to ~5 times with
exponential backoff. Reconnect opens a new session (server state
doesn't persist across disconnects), so the transcript resets
cleanly. Older turns already in the transcript remain on-screen so
the user can see history — just mark them as "from the previous
connection."

## Error handling

| error.code from server | UI treatment |
|---|---|
| `invalid_config` | Show modal, offer reconnect |
| `shutting_down` | "Gateway is draining — try again in a moment." |
| `stt_error` | Show inline in transcript rail, drop turn |
| `llm_error` / `llm_timeout` | Show inline, state returns to idle |
| `tts_error` | Show inline, state returns to idle |

Mic permission denied → show the same "Please allow microphone
access" copy the Transcribe tab uses.

## Known backend gaps (to flag if hit)

- No client-initiated `interrupt` control — UI's "Stop AI" is
  local-only. If we ever need real cancel, backend needs a new
  `session.interrupt` control type.
- `response.text_delta` is chunked by LLM token, not sentence —
  display has to render incrementally rather than waiting for
  sentence boundaries.
- Mock mode's conversation implementation is untested from this
  branch; if the mock path doesn't emit the right event sequence,
  that's a legitimate finding to file as `bugs/NNN`.

## UI layout

```
┌─────────────────────────────────────────────────┐
│  05 · Conversation                              │
│  Talk to the gateway.                           │
│                                                 │
│  ─────────────────────────────────────────────  │
│                                                 │
│  [ Voice picker ▾ ]     [ ● state pill ]        │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  you      hello, what's your name?         │  │
│  │  gateway  My name is Aurora. …            │  │
│  │  you      tell me a joke                  │  │
│  │  gateway  ▏█ (live transcript partial)   │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌─ [mic toggle] ─ [ Stop AI ] ─────────────┐  │
│  └──────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

Mic is toggle-to-talk, not push-to-talk — matches the server's
VAD-driven model, and testers don't have to hold a button. When
the mic is on, PCM streams continuously; the server segments turns
via VAD and barge-in is automatic on interrupt.

"Stop AI" is only visible while state == `tts_speaking`.
