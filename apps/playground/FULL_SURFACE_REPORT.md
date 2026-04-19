# Playground — full surface report

Work on branch `feat/frontend-full-surface`. The three previously
"Coming soon" placeholder tabs — Design, Transcribe, Conversation —
are now live, each wired to its gateway endpoint. Every backend
feature the gateway exposes is reachable through the browser.

## Tabs shipped

### 03 · Design
- Prompt textarea with three VoxCPM2-style example chips.
- Optional sample-text field (default: `Hello. This is a preview of
  the designed voice.`).
- `POST /v1/voices/design` → `GET /v1/voices/design/{id}/audio` →
  `<audio>` preview.
- Save form with required voice name → `POST /v1/voices/design/{id}/save`.
- Editing the prompt or sample text clears the pending preview so
  stale renders can't be saved.
- Success panel: **Test in TTS →** + **Go to library** + **Design
  another**.
- Commit: `5533c28`.

### 06 · Transcribe
- Segmented control: **Upload audio** / **Record**.
- Upload: drag-and-drop or file picker for `.wav .mp3 .flac .m4a
  .webm .ogg`, 100 MB cap (matches gateway body limit).
- Record: `MediaRecorder` in-browser capture with a pulsing
  circular REC button, mm:ss elapsed display, re-record.
- Language selector (auto + Nano + MLT from
  `funasr_worker/language_router.py`), hotwords field, ITN and
  punctuation toggles.
- `POST /v1/stt` — small uploads go through `apiFetch`; uploads
  over 5 MB use XHR so testers see real upload-progress bytes.
- Result panel: transcript (readonly, copyable), detected language,
  model used, word count, audio duration, processing RTF, **Copy**
  + **Download .txt**.
- No SRT — `/v1/stt` doesn't return word timestamps, faking them
  would be worse than omitting them.
- Commit: `6cdf42d`.

### 05 · Conversation
- Backend-contract readback shipped first as a design doc
  (`apps/playground/CONVERSATION_DESIGN.md`, commit `7a296b5`).
- New modules in `src/lib/conversation/`:
  - `audio-capture.ts` — `getUserMedia` → ScriptProcessor → PCM16 @ 16 kHz.
  - `audio-playback.ts` — scheduled `AudioBufferSourceNode` chain
    off a master `GainNode` for fast barge-in fadeouts.
- WS `/v1/conversation` session follows server-driven state —
  `session.status` events update the UI pill rather than local
  prediction.
- Voice picker (reuses `/v1/voices`), transcript rail with partial
  `▏` cursor, streaming assistant text from `response.text_delta`.
- Barge-in: on server `interrupt` event, local playback fades to 0
  over 5 ms, the in-flight assistant turn is marked as
  `interrupted`, then `session.status` flips the pill back to
  `listening`. "Stop AI" button is local-playback-only (the
  protocol has no client-initiated interrupt control; UI copy says
  so).
- Reconnect with exponential backoff, max 5 attempts; session
  epoch bump tags prior-session turns as `·prev` so the transcript
  isn't mashed together.
- Commit: `8998fb3`.

## Flows verified end-to-end (manual)

Ran against a local mock-mode gateway. Real hardware verification
will happen against staging.

| Flow | Result |
|---|---|
| Design: prompt → preview → save → appears in Library → Test in TTS handoff | ✅ |
| Design: edit prompt after preview → save form hides until regenerate | ✅ |
| Design: name clash on save → clean humanised 409 | ✅ |
| Transcribe: upload a .wav → transcript + metadata | ✅ |
| Transcribe: record 5 s in-browser → transcript | ✅ |
| Transcribe: copy to clipboard, download .txt | ✅ |
| Transcribe: language selector changes model_used between nano/mlt | ✅ |
| Transcribe: large upload shows percentage | ✅ (forced with a 10 MB dummy) |
| Conversation: start → speak → AI reply transcript + audio | ✅ |
| Conversation: barge-in (speak during AI playback) → fade + interrupted marker | ✅ |
| Conversation: refresh page → history clears (no persistence, expected) | ✅ |
| Conversation: kill WS server-side → auto-reconnect banner | ✅ |

## Backend bugs filed

None new. The Conversation design doc flags three soft observations
that aren't bugs but may become follow-ups:

1. No client-initiated `interrupt` control — client's "Stop AI" is
   local-only. If we want real user-initiated cancel without
   speaking, the gateway's `_on_ws_control` needs to grow a
   `session.interrupt` branch.
2. `response.text_delta` cadence is per-token; the UI has to render
   incrementally. Not a bug, but a future sentence-boundary option
   would make the transcript rail feel less jittery.
3. Mock-mode conversation is not exercised by any gateway test —
   I did not hit a bug there in manual testing, but it's an
   untested path from this branch.

## Bundle size

| Route | Before (placeholder) | After |
|---|---|---|
| `/design` | 0 B page / 162 kB First Load | 3.09 kB / 165 kB |
| `/transcribe` | 0 B / 162 kB | 5.13 kB / 167 kB |
| `/conversation` | 0 B / 162 kB | 5.2 kB / 167 kB |
| Shared First-Load JS | 174 kB | 174 kB (unchanged) |

Total page-specific JS added: **~13.4 kB**. No new runtime
dependencies — Web Audio, MediaRecorder, and WebSocket are all
browser-native; everything else reuses `apiFetch`, `humanizeApiError`,
`getToken`, and the existing UI kit.

## Three flows to try first (tester priority)

1. **Conversation end-to-end.** Open `/conversation`, start it, say
   something, interrupt the AI, watch the transcript. This is the
   highest-risk surface — real-time, multi-seam, Web Audio timing.
2. **Transcribe in-browser recording.** Click "Record", talk for
   10 seconds, stop, transcribe. Exercises the `MediaRecorder` →
   `MediaRecorder.mimeType` → gateway chain; browser codec choices
   will vary and we want to catch any Fun-ASR path that chokes on a
   specific container.
3. **Design → save → TTS handoff.** Describe a voice, preview,
   save, click **Test in TTS →**. Exercises voice-library mutation
   + the shared voice-id handoff in one pass.

## Rough edges (polish follow-ups, not bugs)

- Transcribe `MediaRecorder` picks the first supported mime type in
  preference order — we don't expose codec choice to testers. If a
  tester hits a container Fun-ASR rejects, the error shows up at
  transcribe time, not at record time. Worth a pre-upload mime
  check later.
- Conversation transcript uses `${role} ${text}` layout; long AI
  replies scroll a decent chunk. A future enhancement could
  collapse older turns.
- Reconnect resets the session-level conversation history (server
  state is ephemeral); we visually tag prior-session turns with
  `·prev` but there's no re-hydration. Acceptable for v1.
- Design save auto-seeds the voice name from the auto-generated
  preview stub (`preview-<timestamp>`) — testers always want to
  rename it. Could nudge toward something nicer-looking once we
  have telemetry on what names get picked.

## One observation

The hardest part of this build wasn't the WebSocket plumbing — it
was fitting the server-driven state machine into the editorial
design language. The aesthetic is static-feeling (print-like,
hairline rules, serif title). Real-time state — a pulsing pill
flipping between four values over ~200 ms windows during a
turn — cuts against that. I kept the pill small and let the
transcript rail be the main "live" surface, because the page can't
look like a dashboard without breaking the rest of the tabs. The
trade-off: when you're actively conversing, the UI is more reactive
than the rest of the app, and the transition from Library (mostly
static) to Conversation (always moving) feels slightly jarring. A
future iteration could warm up the pill's motion vocabulary across
other tabs so it doesn't stand out as the only "alive" surface.
