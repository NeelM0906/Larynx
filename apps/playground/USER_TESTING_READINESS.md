# Playground — User-Testing Readiness

*Phase 1 discovery for `feat/frontend-polish`. Written 2026-04-18 against
commit `0f870a8` (merge of `feat/m7-finetune` into `main`).*

---

## Phase 2 completion update — 2026-04-19

Scope actually shipped under Option 2 (TTS + Clone + Library built;
Design/Transcribe/Conversation intentionally deferred). Polish folded
in as each feature landed.

**Fixed** (from the §2/§3 backlog below):

| Ref   | Item                                                    | How              |
| ----- | ------------------------------------------------------- | ---------------- |
| B1    | SignOutButton defined but not rendered                  | Sign-out in nav + confirm modal |
| B2    | AuthGate accepts any non-empty token                    | GET /v1/voices validates on save; rejects with explicit copy |
| B3    | Fine-tune upload errors show raw JSON                   | Shared humanizeApiError + ErrorPanel ("Show details" expander) |
| B4    | Fine-tune submit errors show `String(e)`                | Same (humanizeApiError) |
| B5    | Cancel has no confirmation + no feedback                | Dialog confirm; "Cancelling…" state until SSE terminal event |
| B6    | Finetune → `/library?voice=<id>` lands on a stub        | Library is now built and scrollIntoViews + rings the matching card |
| B7    | Log pane scroll-jacks mid-training                      | Stick-to-bottom only when user is within 12 px of the end |
| R1    | Footer still says M6                                    | Home STATUS labels now read "Ready" / "Coming soon" honestly |
| R4    | No file-size hint on finetune upload                    | "Max 500 MB total" + running total with client-side gate |
| R5    | Log pane has no copy/download                           | "Download log" → .txt with job-id-prefixed filename |
| R7    | Progress bar snaps in big increments                    | Explicit transition on width |

**Deferred** per the Option-2 scope note:

- Design (`M6.3`) → ComingSoon card with SparklesIcon; endpoints
  (`POST /v1/voices/design`, `POST /v1/voices/design/{id}/save`) are
  live but no UI this cycle.
- Transcribe (`M6.6`) → ComingSoon with CaptionsIcon; `POST
  /v1/audio/transcriptions`, `POST /v1/stt`, `WS /v1/stt/stream`
  are live but no UI.
- Conversation (`M6.5`) → ComingSoon with MessagesSquareIcon;
  explicitly called out as *deferred until the backend's WS
  /v1/conversation gets a clean real-model run*.
- R8 (`useWebSocket` stale deps) → not hot path now (hook is unused);
  leaving the `eslint-disable` in place until Conversation ships.
- R3 (iterations cap at 2000), R2 (token visibility), R6 (hydration
  skeleton on /finetune) → intentionally skipped as minor polish;
  testable without them.
- M2 (token-acquisition guidance copy), M3 (global error boundary),
  M5 (analytics), M6 (favicon) → out of scope for this cycle.

**What's now testable end-to-end** (mock-mode gateway, change-me-please
token):

1. Authenticate — AuthGate, token validation, sign-out.
2. Library — list, delete, test-in-TTS, empty state, refresh.
3. TTS — text input, voice picker, advanced params, synthesize,
   play, download, `?voice=<id>` pre-select from Library or Clone.
4. Clone — file upload, name + description + transcript, submit,
   success card with Test-in-TTS and Go-to-library.
5. Fine-tune — upload, validate, configure, watch, cancel with
   confirm, terminal state, redirect to library (now real).

See `USER_TESTING_GUIDE.md` for the tester-facing flow and
`PHASE_3_REPORT.md` for the overall report.

Bundle delta: starting 167 kB shared → 173 kB shared (+6 kB for
ErrorPanel, humanizeApiError, sign-out Dialog wiring, ComingSoon
component). Three new client routes added at 2.85 / 3.08 / 3.51 kB
each. Fine-tune grew 4.56 → 5.45 kB for the cancel Dialog + log
download handler.

---

## TL;DR — scope mismatch

The root `README.md` status table says `M6 playground UI | done | Next.js,
all tabs`. The code disagrees: **only 1 of 7 tabs is built.** The other
six are one-file placeholders that render `<PageShell>` wrapping a dashed
border box with text like *"TTS form + player — lands in M6.1"*.

| Tab            | Route           | State              |
| -------------- | --------------- | ------------------ |
| Home           | `/`             | Built (nav grid)   |
| TTS            | `/tts`          | **Stub** (M6.1)    |
| Clone          | `/clone`        | **Stub** (M6.2)    |
| Design         | `/design`       | **Stub** (M6.3)    |
| Library        | `/library`      | **Stub** (M6.4)    |
| Conversation   | `/conversation` | **Stub** (M6.5)    |
| Transcribe     | `/transcribe`   | **Stub** (M6.6)    |
| Fine-tune      | `/finetune`     | Built (M7)         |

Shared chrome (AuthGate token dialog, sticky nav, home tab grid, shared
`PageShell`, `apiFetch` helper, `useWebSocket` hook) *is* present. The
fine-tune wizard is genuinely complete: 4-step flow, SSE log stream,
cancel, terminal states, auto-redirect to library on success.

So "the interactive elements aren't polished enough" isn't the gap.
**The gap is that six of the advertised features aren't implemented.**
Running a tester through the current build would give them a tour of
placeholder boxes with the word "lands" on them.

**Recommendation before Phase 2:** decide whether Phase 2 is (a) polish
the one built page + ship under-promised ("fine-tune demo"), (b) build
out the six placeholder pages against existing backend endpoints, or
(c) something in between.

---

## § 1. What works end-to-end

Evidence: `npm ci` clean, `npm run build` clean (12 static routes,
167 kB shared JS, fine-tune is the only client-hydrated route at
4.78 kB extra), `npm run dev` served every route as 200 on localhost.

- **Home `/`** — grid of 7 tab cards with editorial kicker, serif
  headline, blurb, and hover state. Navigation via Next.js `<Link>`.
  Footer shows "Larynx · internal bench · M6 · playground".
- **AuthGate (`src/components/auth-gate.tsx`)** — modal dialog at app
  mount if no token; on paste + Enter or Save click, writes to
  `localStorage['larynx.token']` and dispatches
  `larynx:token-changed`. Subsequent tabs/pages pick it up. Gate
  closes; children render.
- **Sticky nav (`src/components/nav.tsx`)** — 7 tab links with
  active-route underline, mono numeral prefix, Brand link to home.
- **`apiFetch` (`src/lib/api-client.ts`)** — attaches
  `Authorization: Bearer <token>` from localStorage, sets JSON
  Content-Type unless body is `FormData`, throws `ApiError` with
  status + parsed body on non-2xx, returns blob for non-JSON 2xx.
- **`useWebSocket` hook (`src/lib/ws-client.ts`)** — opens a single
  socket, appends `?token=…` query string, exposes `{status, send,
  socket}`. *Not imported anywhere.* Ready for the conversation /
  streaming tabs.
- **Fine-tune `/finetune`** — 4-step wizard:
  1. **Upload** — multi-file picker (`.wav`/`.flac`/`.mp3`/`.jsonl`),
     per-file size list, calls `POST /v1/finetune/datasets`
     (multipart). On `dataset_invalid` 4xx surfaces the report's
     issues inline rather than as a raw error.
  2. **Validate** — shows num_clips, total duration, sample-rate
     histogram, issue list. Gate "Configure training" unless
     `report.ok`.
  3. **Configure** — voice-name text field, three sliders (LoRA rank,
     LoRA alpha, iterations), submit calls `POST /v1/finetune/jobs`
     with `config_overrides`.
  4. **Watch** — `GET /v1/finetune/jobs/{id}/logs` SSE stream.
     Progress bar, last-state readout (loss/diff, loss/stop, lr),
     scrolling log pane (emerald-on-black), cancel button (calls
     `DELETE /v1/finetune/jobs/{id}`), terminal card with state
     (`SUCCEEDED`/`FAILED`/`CANCELLED`). On success, waits 1.5 s
     then `router.push('/library?voice=<id>')`.

---

## § 2. What's broken

> Note: "broken" here means "visibly wrong in the built surface."
> Anything that's just-a-stub is logged in §4, not here.

### B1. `SignOutButton` is defined but never rendered
- **File:** `src/components/auth-gate.tsx:92-103`
- **Repro:** Set any token. Look around the app. There's no way to
  clear it, see it, or change it — short of opening devtools and
  running `localStorage.removeItem('larynx.token')`.
- **Impact:** Any tester who pastes the wrong token is stuck. Any
  tester who wants to try a second token is stuck. Any tester who
  wants to confirm *which* token they pasted is stuck.
- **Fix size:** 30 min. Import `SignOutButton` into `<Nav>`, render
  in the top-right. Optional polish: show first/last 4 chars of the
  saved token so you know which env you're pointed at.

### B2. AuthGate accepts any non-empty string
- **File:** `src/components/auth-gate.tsx:37-39` (the save path
  just trims and stores without any validation).
- **Repro:** Paste `hunter2` into the token field. Dialog closes.
  Click any tab. Fine-tune upload page lets you pick files and hit
  "Upload" — the 401 only surfaces when the `POST /v1/finetune/
  datasets` call returns, and the error is rendered as
  `JSON.stringify(e.body.detail)` which is non-obvious to a tester.
- **Impact:** Tester assumes token worked; wastes time picking files
  before hitting the real error.
- **Fix size:** 30–60 min. Either validate with a lightweight
  `GET /v1/voices?limit=1` (or similar read-only endpoint) before
  dismissing the dialog, OR render a persistent "not authenticated"
  banner that only goes away after the first successful API call.

### B3. Fine-tune upload errors show raw JSON
- **File:** `src/app/finetune/page.tsx:96` —
  `setUploadError(JSON.stringify(body.detail ?? body))`.
- **Repro:** Submit without a token, or with a bad token, or when
  the gateway is down. The red error pane shows `"detail": "Not
  authenticated"` or a full JSON blob.
- **Impact:** Low for devs, high for testers. Doesn't prevent
  completion if they know to re-set their token, but it's ugly and
  unhelpful.
- **Fix size:** 30 min. Map known error codes to friendly copy;
  fall through to the raw body only on unknown shapes.

### B4. Submit error path too
- **File:** `src/app/finetune/page.tsx:127` — `setSubmitError(String(e))`
  produces `ApiError: 401 Unauthorized` which is only slightly
  friendlier than B3.
- **Fix size:** 30 min, same pattern as B3.

### B5. Cancel button fires with no confirmation + no immediate feedback
- **File:** `src/app/finetune/page.tsx:215-222,647-653`.
- **Repro:** Click "Cancel job" during training. Nothing visibly
  changes until the SSE stream emits a terminal `CANCELLED` event
  (could be 1–10 s depending on the worker). Meanwhile the button
  stays clickable — you can double-click it with no effect (the
  second call 409s or 404s and the error is swallowed).
- **Impact:** Tester thinks click didn't register, clicks again.
- **Fix size:** 45 min. Add a "Cancel job?" confirmation dialog
  (we have `Dialog` in `ui/dialog.tsx`), then set a
  `cancelling` state that disables the button and swaps the label
  to "Cancelling…" while we wait for terminal event.

### B6. Success path pushes to `/library?voice=<id>` — library is a stub
- **File:** `src/app/finetune/page.tsx:202-211` push; stub at
  `src/app/library/page.tsx`.
- **Repro:** Complete a training run. Terminal card says "Jumping
  to the library…". 1.5 s later you land on `/library` which is
  just "Voice card grid — lands in M6.4". The `?voice=<id>` query
  string is ignored.
- **Impact:** After the best moment in the product (a real voice
  finished training), the user is dumped on a placeholder.
- **Fix size:** Depends on what we ship. If library stays a stub,
  change the redirect to a celebratory in-page state that shows the
  voice id + a "try it" CTA (30 min). If library becomes real, it
  should at least scroll/focus the `?voice=…` row (on top of
  building the library).

### B7. Log pane scroll-jacks the user
- **File:** `src/app/finetune/page.tsx:611-614`.
- **Repro:** Start a training run that emits many log lines. Scroll
  up to read an older line. New line arrives; you're yanked back to
  the bottom.
- **Impact:** Frustrating when inspecting a specific warning.
- **Fix size:** 30 min. Stick-to-bottom when user is already at
  bottom; don't scroll if they've scrolled up.

### B8. WebSocket hook `onMessage` isn't in the effect deps array
- **File:** `src/lib/ws-client.ts:67` — `// eslint-disable-next-line
  react-hooks/exhaustive-deps`. `onMessage`/`onOpen`/`onClose` are
  captured from the first render and don't update.
- **Impact:** Not observable today because nobody uses the hook. But
  it's a latent bug the moment the conversation tab lands —
  callbacks that reference stateful React values will go stale.
- **Fix size:** 15 min. Stash callbacks in refs; read `.current`
  inside the WebSocket event handlers.

---

## § 3. What's rough but not broken

### R1. No footer M6 correction
- **File:** `src/app/layout.tsx:43` — `<span>M6 · playground</span>`.
- Technically accurate per the README but misleading given only M7
  is built. Harmless but jarring if a tester pokes around.
- **Fix size:** 5 min or leave until we actually ship M6 tabs.

### R2. No way to see what the token currently is
- Already noted under B1 — same fix covers it.

### R3. Fine-tune "Training iterations" slider caps at 2000
- **File:** `src/app/finetune/page.tsx:536`. Upstream default is
  1000 per the hint. No comment on what ranges produce what
  quality. A tester with a big dataset can't go higher; a tester
  with a tiny dataset doesn't know the floor.
- **Fix size:** Probably fine as-is; just flagging.

### R4. No indication of file-size limit on upload
- Gateway caps `/v1/finetune/datasets` at 500 MB (see
  `packages/gateway/src/larynx_gateway/middleware/body_limits.py:39`).
  Playground doesn't surface this. A tester dropping a 2 GB corpus
  gets a body-limit error.
- **Fix size:** 20 min. Add "Max 500 MB total" line to the drop
  zone; do a client-side sum check before hitting the API.

### R5. Log pane has no "copy all" / "download" affordance
- Training logs can be long and are only visible while the stream
  is open. After page reload they're gone. A tester reporting "it
  crashed on step 847" has to screenshot.
- **Fix size:** 30 min. Button next to the log pane that dumps
  current logs as a .txt download.

### R6. No loading skeleton on fine-tune page during initial client mount
- Next.js hydrates `FinetunePage` after the JS bundle loads.
  Between static SSR and hydration, buttons are non-interactive
  with no visual cue.
- **Fix size:** 30 min. Disable controls + show a spinner pre-
  hydration (or use `<Suspense>`).

### R7. Progress bar jumps in big increments
- Because we only update on `state` events (~one per training step
  batch), the bar snaps rather than animates. Minor polish.
- **Fix size:** 15 min via `transition-all` on width or skip.

### R8. `useWebSocket` reconnection is not built
- Hook closes on unmount; no exponential-backoff reconnect. Not a
  bug because no page uses it, but the conversation tab will want
  this before user testing.
- **Fix size:** 1–2 hr when we actually need it.

---

## § 4. What's missing that would block user testing

### M1. Six feature tabs are unimplemented (THE big blocker)

All six placeholder pages need at minimum a token-auth'd happy-path
flow against the existing gateway endpoints. Rough scoping below —
estimates assume the visual style already established by the
fine-tune page + PageShell is maintained (no new component libs).

| Tab          | Endpoints                                                                                                     | Sketch                                                                                                                                                                                                                                         | Estimate      |
| ------------ | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| TTS          | `POST /v1/tts`                                                                                                | Multiline text input, voice picker (fed from `GET /v1/voices`), cfg_value + inference_timesteps sliders, "Generate" button, inline `<audio>` player, download link. Optional: streaming via `WS /v1/tts/stream`.                         | **~4 hr**     |
| Clone        | `POST /v1/voices` (multipart), `POST /v1/tts`                                                                 | Single-file drop zone, name field, "Create voice" → preview text area → "Synthesize" using the new voice id.                                                                                                                                 | **~2 hr**     |
| Design       | `POST /v1/voices/design`, `POST /v1/voices/design/{preview_id}/save`                                          | Prose textarea (voice description), preview text, "Render preview" → listen → "Save to library" with a name. State machine: idle → previewing → saved.                                                                                       | **~3 hr**     |
| Library      | `GET /v1/voices`, `GET /v1/voices/{id}`, `GET /v1/voices/{id}/audio`, `DELETE /v1/voices/{id}`                | Paginated grid of voice cards (name, use count, type badge, created at), reference-clip `<audio>` play, delete with confirm. Focus/scroll-to `?voice=<id>` query param for the fine-tune success path (fixes B6).                           | **~4 hr**     |
| Conversation | `WS /v1/conversation`                                                                                         | Mic capture (`getUserMedia`), encode → PCM frames → send. Receive events: `state`, `user_partial`, `user_final`, `assistant_partial`, `assistant_delta` (audio chunks → MSE-based player). Config drawer, live state pill, transcript list. | **~1–2 days** |
| Transcribe   | `POST /v1/audio/transcriptions` (OpenAI shim) or `POST /v1/stt` (native); optional `WS /v1/stt/stream`        | File-drop or record-from-mic toggle, language picker, hotwords field (comma-sep), submit → transcript view. Streaming variant shows rolling partials.                                                                                        | **~4 hr**     |

Collectively: **~2.5–4 days** of focused work to get all six tabs to
a "happy path works" state. Budget more if we want barge-in,
streaming TTS playback, or polished error UX per tab.

### M2. No token-acquisition guidance

A new tester sees `"Paste your gateway bearer token"` with no
context about what a token is, where to get it, or what a valid
one looks like. Playground assumes you already have credentials.

**Fix:** Expand the AuthGate copy with a link to wherever we
document test tokens (if we have one; if not, add one) and a note
like "Ask Neel in #larynx-bench for a test token." Optional: a
"Use demo token" button for an environment-scoped read-only token
so testers can poke around without asking.

**Fix size:** 15 min for copy; 1 hr if we wire up a demo-token
fetch endpoint.

### M3. No global error boundary

If any page throws during render, the whole Next.js app shows its
default error screen. No friendly "something went wrong, here's
a sentry link / reload button" fallback.

**Fix size:** 30 min. Add `src/app/error.tsx` (Next.js 15 error
boundary convention).

### M4. No "about this page" / "what to do here" affordance

For first-time users, each tab lands with a terse serif headline
and (when we build them) some inputs. A one-sentence "try typing
a sentence, then hit Generate" placeholder inside each tab would
lower the activation energy. The Fine-tune page sort of does this
with its `intro` text, which is good.

**Fix size:** 15 min per tab, done as we build them.

### M5. No analytics / feedback capture

If we want to learn anything from user testing, we need to know
what they clicked, what they gave up on, what they retried. Right
now, nothing is instrumented. A postmortem becomes "ask them what
happened" — rarely specific.

**Fix size:** 1–2 hr for a minimal event beacon (PostHog, Plausible,
or roll-your-own POST /v1/telemetry). Out of scope for this branch
unless we have a destination; flagging as preemptive.

### M6. No favicon / brand polish

`public/favicon.ico` is the Next.js default. Page title metadata is
set (`"Larynx Playground"`) — good. But the tab icon is generic.

**Fix size:** 15 min if we have a logo SVG; skip if not.

---

## § 5. Estimate

Bucketing the above into effort tiers. Counts assume one competent
frontend engineer, working on one thing at a time, against mock TTS
mode with occasional live-gateway sanity checks.

| Bucket              | §2 (broken)              | §4 (missing)              | Total   |
| ------------------- | ------------------------ | ------------------------- | ------- |
| 15–30 min           | B1, B3, B4, B7, B8, R1   | M2 (copy only), M4, M6    | ~3 hr   |
| 30–60 min           | B2, B5, B6 (partial), R4, R5, R6 | M3                | ~4 hr   |
| 2–4 hr each         | (none)                   | TTS, Clone, Design, Transcribe, Library | ~half-day each |
| Full day+           | (none)                   | Conversation (WS duplex)  | ~1–2 days |

**Rough horizons:**

- **Polish-only path** (fix §2 + §3, leave §4 M1 alone): **~1 day**.
  Ship "fine-tune demo only", under-promise the rest, tester clicks
  exactly `/finetune` and ignores the rest.
- **Minimum viable user-test path** (polish + TTS + Library +
  Transcribe): **~2.5 days**. Gives a tester three synchronous
  flows to try (generate speech, see voices, transcribe audio).
  Leaves Clone, Design, Conversation as "coming soon."
- **Full M6-as-advertised path**: **~4–5 days** including the
  conversation tab. Delivers what the README claims.

---

## Testing discipline notes

- **I did not start the full backend stack** (`make up && make
  migrate && LARYNX_TTS_MODE=mock uv run gateway`). Reason: only
  one of seven pages makes any API call, and that page
  (`/finetune`) dispatches to a GPU worker that isn't covered by
  `LARYNX_TTS_MODE=mock` anyway. Spinning up postgres + redis +
  gateway to validate "the AuthGate dialog renders" would be
  over-indexing.
- **What I would need real infra for:** exercising the fine-tune
  SSE stream end-to-end; observing how the wizard behaves on a
  real 401 vs. mock; confirming the cancel path actually reaches
  the worker. These are best tested when we have a real dataset +
  GPU — flagging for Neel to run on the target box before inviting
  testers.
- **WebSocket flows in mock mode:** irrelevant right now —
  `useWebSocket` is unused. When the conversation / streaming tabs
  land, we'll want to verify mock-mode WS frames match real-mode
  frames before relying on local testing.

---

## Nothing has been changed in this worktree yet

This report is a pure read/build/run pass. No files under
`apps/playground/**` have been edited. The only side effect is
`apps/playground/node_modules` (not committed — gitignored at repo
root via Next.js `.next/` + generic `node_modules` rules).

Handing back for prioritization. Will not proceed to Phase 2 until
directed.
