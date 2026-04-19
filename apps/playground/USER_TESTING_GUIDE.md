# Larynx Playground — User Testing Guide

*For testers poking at the playground before M6.5/M6.6 ships. Takes
~15 minutes if everything works; a little longer if you bump into
something we want to hear about.*

---

## What we're testing this cycle

All seven tabs are live: **TTS**, **Clone**, **Design**, **Library**,
**Conversation**, **Transcribe**, **Fine-tune**. Every backend feature
the gateway exposes is now reachable from the browser; there are no
"Coming soon" placeholders in this build.

Our goal is to hear about the **full platform loop**: authenticate,
browse the library, clone a voice, synthesize with it, design a voice
from a prompt, transcribe an audio clip, fine-tune a LoRA, then try
a live conversation. Sign out when you're done.

---

## Prerequisites

### 1. Get a bearer token

Ask Neel for the playground's API token in `#larynx-bench`. The
default dev token for a fresh local gateway is `change-me-please` —
if you're pointed at a local box, paste that; otherwise Neel will
give you a real one.

### 2. Get a reference audio clip

Any clean, single-speaker recording of 10–30 seconds works. If you
don't have one handy, grab one of these from the repo:

- `scripts/m0/smoke_tts.wav` — a short English sample used by the
  smoke tests (16 kHz, ~6 s).
- `packages/gateway/tests/fixtures/audio/english_reference.wav`
- `packages/gateway/tests/fixtures/audio/chinese_reference.wav`
- `packages/gateway/tests/fixtures/audio/cantonese_reference.wav`
- `packages/gateway/tests/fixtures/audio/portuguese_reference.wav`

Formats we accept: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`.
Cap on size: 50 MB (the UI gates this client-side).

### 3. Know where the playground is running

Ask Neel for the URL. If you're running it locally yourself, it's
typically `http://localhost:3000` after `npm run dev`.

---

## The flow (run these in order)

### A. Authenticate with your token

1. Open the playground URL in a recent Chrome/Firefox/Safari.
2. A dialog titled **"Set API token"** appears immediately. Paste
   your token.
3. Hit **Save token** (or press Enter).
4. The dialog will say **"Checking…"** briefly, then close.

**What should happen:** Dialog closes, the home page becomes visible
with seven tab cards. A small **"Sign out"** button appears in the
top-right of the nav.

**What to flag:**
- Did the dialog reject a valid token? (copy-paste mistake on our
  end would be good to know.)
- Did it accept an obviously bad token? (we validate against the
  gateway; if it accepts `hunter2`, that's a bug.)
- Did it show a raw JSON blob instead of a human error? (we have an
  error-humanizer; if you see `{"detail":{...}}` somewhere other than
  behind a "Show details" expander, that's a bug.)

### B. Browse the voice library

1. Click the **"Library"** tab (card 04 on the home grid, or in the
   nav).
2. You should see the list of voices already in the gateway's
   Postgres: cards with name, a colored source badge (uploaded /
   designed / seed / lora), sample rate + duration, created date.

**What should happen:**
- Loading skeleton shows briefly, then a grid of 2-column cards.
- A refresh button in the top-right re-fetches on click.
- If the library happens to be empty, you see a big "No voices yet"
  card with a "Clone a voice →" CTA.

**What to flag:**
- Did the list look half-loaded or error out? Flag the wording and
  whether the "Show details" expander gave useful info.
- Are the cards misaligned on mobile? (We target desktop, but it
  should at least not overflow.)

### C. Clone a new voice from a reference clip

1. Click **"Clone"** (tab 02).
2. Drag your `.wav` into the big dashed box, or click it to open a
   file picker. The file's name and size appears in a card below.
3. Type a **Voice name** — anything unique like `tester-<your-initials>-<date>`.
4. Optional: fill **Description** (one-line note) and **Reference
   transcript** (exactly what's being said in the clip — this
   unlocks higher-quality cloning).
5. Hit **Clone voice**.

**What should happen:**
- Button flips to "Processing reference…" for ~1–2 seconds on mock
  hardware, longer on real.
- On success: the form is replaced by a celebration card showing the
  voice id (12-char prefix), sample rate, duration, source.
- Two primary actions appear: **Test in TTS →** and **Go to library**.

**What to flag:**
- Any step where you were unsure what was happening — especially
  during the upload. A pulsing dot should narrate "Uploading and
  encoding."
- If the file looked accepted but the submit button stayed disabled,
  tell us why (size over 50 MB? wrong extension?). The meta card
  should explain; if it doesn't, that's a UX bug.
- Did the name-clash error ("A voice with that name already exists")
  surface clearly if you re-used a name?

### D. Synthesize 2–3 phrases with your new voice

1. From the Clone success card, click **Test in TTS →**. You should
   land on `/tts?voice=<id>` with your new voice pre-selected.
2. Type something into the **Text** box. Try something characterful
   so you can tell the voice is working: *"The weather in Mumbai is
   kind of unreasonable this week."*
3. Hit **Synthesize**.
4. An audio card appears with a `<audio>` player (scrub, play,
   volume) and a **Download .wav** button. Play it. Does it sound
   like what you expected?
5. Try a second phrase. Maybe expand the **Advanced** drawer and
   nudge Temperature down to 0.5 (less variance) — does the output
   sound more deterministic?
6. Try a third phrase that intentionally trips edge cases: all
   numbers (`"Area 51, interstate 95, 24601"`), emoji, non-English
   characters.

**What should happen:**
- Button reads "Synthesizing…" during synth; a pulse dot explains
  the expected wait time.
- The audio card shows duration / sample rate / gateway-side
  generation time / file size — a tiny mono row of metadata.
- Download saves as `<first-40-chars-of-text>-<timestamp>.wav`.

**What to flag:**
- Does the voice quality match what you expected? (If you used a
  noisy reference, we expect a noisy synth; if you used a clean
  reference and got garbage, that's a backend issue — flag it.)
- Did the player fail to play? That would mean either the backend
  returned malformed audio or the browser rejected the content-type.
- Did an error like "Voice no longer exists — pick another" show up
  unexpectedly (you didn't delete the voice)? That's a bug.

### E. Delete the cloned voice

1. Go back to **Library**.
2. Your new voice should be there (may need a Refresh if the list
   was cached).
3. Click **Delete** on its card. A confirmation modal appears: *"Delete
   voice? <name> and its cached latents + reference audio will be
   removed. This can't be undone."*
4. Hit **Delete** (the destructive button).

**What should happen:** Modal closes. Card disappears from the grid.
Total count at the top decrements.

**What to flag:**
- Did delete appear to succeed but the card stayed? (We optimistically
  remove on 2xx. If the row lingers, the DELETE didn't return 204.)
- Did the "Cancel" button behave? It should close the modal without
  doing anything.

### F. Design a voice from a prompt

1. Click **"Design"** (tab 03).
2. In the **Describe the voice you want** box, type a short natural-
   language prompt. Example prompts sit underneath as chips — click
   one to seed the textarea, or write your own: *"A warm British
   narrator, unhurried, mid-range male."*
3. Optional: fill **Sample text** with what the preview should say.
   Default is `Hello. This is a preview of the designed voice.`
4. Hit **Generate preview**. Button flips to "Designing voice…" while
   the gateway renders the preview clip.
5. A card appears with an `<audio>` element — play the preview.
6. If it sounds right, type a **Voice name** and hit **Save to
   library**. On success a celebration panel with **Test in TTS →**
   + **Go to library** handoffs appears.

**What should happen:**
- Editing the prompt or sample text after generating hides the save
  form — you can't accidentally commit a stale preview.
- The saved voice has `source: designed` in the library.

**What to flag:**
- Preview audio doesn't play / sounds wrong for the prompt.
- Save flow errors with a confusing message on name clash
  (we override 409 → "A voice with that name already exists").

### G. Transcribe an audio clip

1. Click **"Transcribe"** (tab 06).
2. Two modes at the top: **Upload audio** (drag/drop a file) and
   **Record** (mic capture in-browser).
3. For upload: drop `scripts/m0/smoke_tts.wav` or any fixture clip.
   For record: click the big circular mic button (grant permission
   if the browser prompts), say a few sentences, click it again to
   stop. A playback `<audio>` appears on both paths.
4. Pick a **Language** if you know it — leaving it on "Auto-detect
   (Nano)" works for English/Chinese/Japanese. Other languages route
   through Fun-ASR MLT.
5. Optional: enter **Hotwords** (comma-separated) to bias Fun-ASR
   toward proper nouns or jargon.
6. Hit **Transcribe**. Small files go through instantly; large files
   show an upload-percentage indicator before Fun-ASR runs.
7. Transcript panel below shows the text, language detected, model
   used (nano vs mlt), word count, audio duration, and RTF. **Copy**
   and **Download .txt** actions.

**What to flag:**
- Did a recording in-browser produce garbage transcripts? The
  `audio/webm;codecs=opus` container is what we send by default;
  some browsers may pick a different container.
- Was the language mis-detected on "Auto"? Try picking the code
  manually and note whether quality improves.
- Did RTF stay under 1.0 on real hardware? (Above 1.0 means Fun-ASR
  is processing slower than realtime — flag it if persistent.)

### H. Have a conversation

1. Click **"Conversation"** (tab 05).
2. Pick a **Voice** for the AI's responses (leave on default for the
   gateway's built-in voice).
3. Hit **Start conversation**. Grant mic permission.
4. Top-right pill shows **Listening**. Say something: *"Hello,
   what's your name?"* Stop talking; the pill flips to **Thinking**
   for a moment, then **Speaking** as the AI plays back its reply.
5. Your partial transcript appears in the rail in real time; the
   AI's reply streams in as `response.text_delta` tokens.
6. While the AI is speaking, try **barge-in**: start speaking over
   it. Playback fades out within a few hundred milliseconds, the
   previous assistant turn is marked `interrupted`, and the state
   pill flips back to **Listening · you**.
7. Alternatively, click **Stop AI** while it's speaking — this mutes
   local playback (note: the server keeps generating unless you
   also start speaking).
8. Hit **End conversation** to tear down the session.

**What to flag:**
- Did barge-in actually cut off the audio promptly? We target <200 ms
  end-to-end (server `interrupt` event → local fade complete).
- Did the transcript rail lose a turn or double-post one?
- Did a reconnect banner appear on a flaky network? It should
  auto-reconnect up to 5 times; failure after that should leave a
  clear error.

### I. Sign out

1. Top-right of the nav, click **Sign out**.
2. A modal asks "Clear token?" with Cancel / Sign out.
3. Hit **Sign out**.

**What should happen:** Modal closes. The AuthGate dialog reappears.
You'd have to paste your token to do anything else.

---

## Where to leave feedback

- **Slack:** `#larynx-bench` — quick reactions, screenshots.
- **Written notes:** if you have more than a few bullets, drop a
  doc in the shared "Larynx bench notes" folder with your initials
  + date in the title.
- **Bugs you're sure are bugs:** Neel will file them under `bugs/`
  in the repo. Include reproduction steps and a note about what
  you expected vs. what you got.

We're especially curious about:

1. **Was any step ambiguous?** — any moment where you didn't know
   whether something was happening or whether you should click
   again.
2. **Did any error message confuse you?** — even ones where
   functionally everything worked; wording that sounded alarming
   when it shouldn't have.
3. **Did the voice picker feel fast enough?** — it fetches on
   mount and sorts alphabetically; the page should be interactive
   within a second on a local gateway.
4. **Was the design quiet enough not to distract?** — editorial
   dark, single vermillion accent. If it felt like a bank
   dashboard or a children's app, tell us.

---

## What this build is *not* supposed to do

Symmetric flags so you don't false-positive a bug:

- **The home page kicker says "M6 · playground".** Not a bug — the
  root README will be corrected once two parallel streams (this
  and M8) both land.
- **The `NEXT_PUBLIC_GATEWAY_URL` env var is how the UI points at
  a gateway.** If unset, the UI assumes same-origin — which won't
  work if you serve playground on :3000 and gateway on :8000.
  If Neel tells you "the UI says the gateway is down," this env
  var is the first thing to double-check.
- **The mock-mode gateway returns short generic audio.** If you're
  pointed at mock mode (no GPU), the cloned voice will sound
  exactly like the default — mock TTS doesn't actually encode
  your reference. Real-hardware testing catches voice quality;
  mock mode catches UX.

Thanks for poking at it. Every weirdness you flag saves us a round
of "why did the tester not follow up?"
