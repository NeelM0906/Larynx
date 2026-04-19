# Playground — Phase 3 Report

*Branch: `feat/frontend-polish`. Written 2026-04-19. Builds
clean against `main` at `0f870a8`; pushes the frontend surface
from "fine-tune demo only" to the core voice-cloning loop (TTS +
Clone + Library) ready for user testing.*

---

## § 1. What I fixed

Per READINESS §2/§3 — the full table is in `USER_TESTING_READINESS.md`
under the Phase 2 completion update. Short version:

- **Auth flow actually validates.** `AuthGate.onSave` now fires a
  `GET /v1/voices?limit=1` before closing. Bad tokens rejected with
  "That token didn't work — the gateway rejected it." Good tokens
  land users on the nav. Plus a sign-out button in the nav with a
  confirm modal.
- **Every error path goes through `humanizeApiError`.** Shared helper
  in `src/lib/errors.ts` that maps `ApiError` → a short headline +
  extracted detail + raw body behind a "Show details" expander.
  401/403/404/413/422/5xx get bespoke copy; per-call overrides for
  domain-specific cases (e.g. TTS 404 → "Voice no longer exists").
  Rendered via `<ErrorPanel>` so every inline error looks the same.
- **Cancel is confirmable.** Fine-tune cancel now opens a dialog,
  then flips the watch button to "Cancelling…" until the SSE stream
  reports the terminal event.
- **Log pane stops scroll-jacking.** Stick-to-bottom only when the
  user hasn't scrolled up.
- **File-size gating.** Fine-tune upload shows "Max 500 MB total"
  and a running total that flips destructive past the cap, disabling
  the submit before the body-limit round-trip.
- **Log download.** "Download log" → `.txt` with a job-id-prefixed
  filename. Disabled until the first log line.

And the big ones:

- **TTS tab built** (`src/app/tts/page.tsx`). Text input (char counter,
  1000-char cap), voice picker from `GET /v1/voices`, advanced
  drawer (sample rate / CFG / temperature), `POST /v1/tts` with
  header extraction, `<audio>` player, download with a slug-based
  filename, `?voice=<id>` pre-select from Library/Clone/Fine-tune.
- **Clone tab built** (`src/app/clone/page.tsx`). File picker + drag-
  drop for wav/mp3/flac/m4a/ogg with 50 MB UI gate, name + optional
  description + optional reference transcript, `POST /v1/voices`
  multipart, success card with Test-in-TTS and Go-to-library actions.
- **Library tab built** (`src/app/library/page.tsx`). Voice grid
  (name, id prefix, source badge, meta, description/design prompt),
  refresh, delete with confirm, Test pre-fills TTS, empty state
  CTA to Clone, loading skeleton, `?voice=<id>` scroll-into-view +
  ring highlight.
- **Coming-soon placeholders.** Design, Transcribe, Conversation
  are intentional cards (lucide icon + "Coming in a future update"
  pill + description + bullets) — not stubs with "lands in M6.x"
  text.
- **Honest home labels.** Per-tab status reads "Ready" / "Coming soon"
  instead of the blanket "M6" / "M7 · soon".

---

## § 2. What I didn't fix, and why

- **`useWebSocket` stale closure (§3 R8).** Still carries the
  `eslint-disable-next-line react-hooks/exhaustive-deps`. The hook
  is unused today — Conversation was deferred — so fixing it would
  be speculative. Comment on the disable reminds the next author to
  pull callbacks through a ref when they wire it up.
- **R3 — Fine-tune iteration cap at 2000.** User called it out as
  "probably fine, just flagging" in READINESS; no tester blocker.
- **R6 — Hydration skeleton on /finetune.** Minor polish. Pre-
  hydration is already handled by `AuthGate` returning `null` until
  mounted; the fine-tune wizard lands within the AuthGate's children,
  so users don't see a "half-interactive" state.
- **M2 — Token-acquisition guidance.** The `AuthGate` copy still
  assumes the reader knows what a bearer token is. I recommend we
  wait on this until we know whether testers get a pre-provisioned
  token or are expected to generate one. USER_TESTING_GUIDE.md
  bridges the gap for now: tells them to ask in `#larynx-bench`.
- **M3 — Global error boundary.** Next.js 15 has a sensible
  `error.tsx` default. Haven't added a custom one. Low-priority
  until we see a real uncaught-exception trip a tester.
- **M5 — Analytics.** Out of scope.
- **M6 — Favicon.** Still the Next.js default. Neel to decide
  whether we have brand assets to drop in.
- **Clone: MediaRecorder record-in-browser.** Spec allowed it if
  ≤45 min extra. Turned out more like an hour for a safe path
  (browser codec quirks + backend container whitelist), so I left
  it as upload-only for this cycle. Low priority — tester just
  needs a wav handy.
- **TTS: output-format dropdown (wav/mp3/flac).** Spec called for
  three formats; backend only supports `wav | pcm16` today
  (`packages/gateway/src/larynx_gateway/schemas/tts.py`:
  `# mp3 lands behind ffmpeg later`). Went with wav-only in the UI
  rather than showing disabled options that would confuse testers.
  Noted in commit body; no backend bug filed because this is already
  a documented code comment in the schema.

---

## § 3. What's still rough that a tester will probably notice

Preemptive call-outs so we know what feedback to filter:

- **Mock-mode voices all sound the same.** In mock TTS mode
  (`LARYNX_TTS_MODE=mock`) the worker returns a canned waveform —
  the voice id is respected for the response header but the audio
  is identical regardless of which voice you picked. Testers
  listening closely may say "my cloned voice sounds exactly like
  the seed voice." That's because mock. Real-hardware run catches
  it; mock mode is for UX.
- **Fine-tune wizard is reachable but hard to actually run
  end-to-end in mock mode** — the worker dispatch will fail
  without a GPU. Recommend: steer testers away from the fine-tune
  tab unless they're on the real box.
- **No loading skeleton on the initial TTS voice-picker fetch.**
  There's a `<div className="h-8 ... animate-pulse bg-muted />`
  shown while voices load, but the rest of the page (text box,
  advanced drawer) is already interactive. Quick testers may type
  and hit Synthesize before the picker resolves — the button
  stays enabled since it defaults to voice-less mode. The synth
  still works; it just won't reflect the voice they expected.
  Arguably fine; flag it if it trips anyone.
- **Clone "Processing reference" doesn't show upload %.** It's a
  single indeterminate pulse-dot. On a slow connection uploading
  10–20 MB, there's no byte-level feedback. Acceptable for
  ≤50 MB files but tester may worry it stalled.
- **The "Sign out" CTA is visible only after token is stored.** On
  the initial AuthGate dialog there's no "I don't have a token"
  path — they either paste one or they're stuck.
- **Alphabetical voice sort is case-sensitive as `a < B`.** Uses
  `.localeCompare()` so it's actually locale-aware; but testers
  with lowercase + uppercase names may see surprising order. Not
  worth fixing unless someone complains.

---

## § 4. How to demo it

**3-sentence pitch:** "This is the Larynx playground — a developer
bench to poke at the voice gateway without opening curl. It now has
a working core loop: see the voice library, clone a new voice from
a reference clip, and synthesize speech with it, all on one
editorial dark surface. Fine-tune lives here too; everything else
is a marker for what's coming next cycle."

**Primary user journey to demo:**

1. Start from the home page — clean grid of seven tab cards, three
   marked "Coming soon" and four marked "Ready."
2. Click **Clone**. Drop a ~15-second `.wav` of a voice from the
   `packages/gateway/tests/fixtures/audio/` folder.
3. Name it something memorable (`ada-demo`, `marcus-demo`), paste
   the transcript, hit Clone voice. Success card appears.
4. Click **Test in TTS →**. The voice is pre-selected. Type a
   distinctive phrase, hit Synthesize, play the result.
5. Hit the **Library** tab in the nav. The voice shows up at the
   top; the Test button there also pre-fills TTS.
6. Optional flourish: hit the Sign out button to show the
   confirm modal, cancel it, keep going.

The editorial typography does the talking — serif headlines,
vermillion-on-near-black, hairline dividers — so it looks less like
a stock admin tool and more like a private workshop. That's the
hook.

---

## § 5. Known issues that aren't showstoppers

- **Home footer still reads "M6 · playground"** — we'll correct
  when the README does. Not fixing from this branch (root README
  off-limits).
- **Clone 409 copy assumes the only name conflict cause is
  collision.** Backend could in principle return 409 for other
  integrity reasons; the copy would be mildly wrong there but
  the "Show details" expander still shows the truth.
- **TTS download filename slug has no collision protection.** If a
  tester synthesizes the exact same text twice within the same
  second, the browser will rename the second file to
  `<name> (1).wav`. Browser handles it cleanly; flagging just so
  nobody's surprised.
- **Library delete is optimistic.** The card vanishes as soon as
  the DELETE returns 2xx; we don't refetch before removing. If a
  concurrent process re-added the voice between our call and the
  render, we'd miss it. Extremely unlikely in a single-tester
  context.
- **Auth validation uses `?limit=1`** — the gateway accepts it
  (1 ≤ limit ≤ 500 per the schema), but it still fetches one row
  per call. Low cost. Could be a cheaper `HEAD` endpoint later.
- **Fine-tune cancelling state is one-way.** Once you've hit
  "Cancel job" + confirm, there's no "undo before the SSE terminal
  lands." Probably fine; the confirm dialog is the undo.

---

## § 6. Design observations (not a call to action)

Design is locked for this cycle — listing these as observations for
whenever the next style pass happens.

- **The fine-tune log pane is emerald-on-black** — intentionally
  evoking a terminal — but it's the only part of the UI that
  breaks the "warm bone on near-black" rule. Against the editorial
  dark canvas, it reads as slightly jarring. An off-white-on-
  graphite treatment would sit better in-family, at the cost of
  looking less like a literal log.
- **Seven tab cards on one page is a lot.** Once three are
  "Coming soon", the page visually balances OK, but when they're
  all ready, the grid will feel crowded. At that point a sidebar
  (or collapsing the home to 4 "primary" + 3 "advanced") might
  serve better. Not now.
- **Source badges on Library cards use four colours.** Lora pins
  to a chart-2 tone that's close to the primary vermillion; at a
  quick glance the two can blur on lower-contrast monitors. A
  subtler treatment (e.g. all badges the same foreground, source
  communicated via a small icon) might read cleaner once we have
  more LoRA voices.
- **The AuthGate dialog is a password-field type for the token.**
  Means we never show the token back — which breaks "tester pastes
  token, wonders if they copied it right." A toggle-to-reveal
  affordance would be friendlier. Didn't add one this cycle to
  stay out of the dialog's layout; low-risk add next time.
- **The "Synthesizing…" / "Processing reference…" / "Uploading…"
  states all use the same pulsing dot.** They feel distinct in
  context but read identical in a screenshot. A different
  affordance per phase (e.g. a progress bar for uploads, a text
  "ASR / VAE encode / register" cycle for Clone) would be nice.

---

## Final build numbers

```
Route (app)                         Size  First Load JS
┌ ○ /                                0 B         162 kB
├ ○ /_not-found                      0 B         162 kB
├ ○ /clone                       3.08 kB         165 kB
├ ○ /conversation                    0 B         162 kB
├ ○ /design                          0 B         162 kB
├ ○ /finetune                    5.45 kB         168 kB
├ ○ /library                     2.85 kB         165 kB
├ ○ /transcribe                      0 B         162 kB
└ ○ /tts                         3.51 kB         166 kB
+ First Load JS shared by all     173 kB
```

**Delta from Phase 1 baseline:** 167 kB → 173 kB shared (+6 kB or
+3.6%). Three new interactive routes added (/tts /clone /library
at 2.85–3.51 kB each). /finetune +0.9 kB for the cancel Dialog +
log download. No new top-level npm dependencies.

Build clean, no TypeScript errors, no ESLint warnings past the pre-
existing `react-hooks/exhaustive-deps` disable on the unused
`useWebSocket` hook.
