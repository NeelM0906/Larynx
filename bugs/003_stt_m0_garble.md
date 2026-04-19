# Bug 003 — M0 smoke pipeline: Fun-ASR-Nano garbles VoxCPM2-synthesised audio

> **Numbering note.** bugs/002 is already taken by the real-model GPU-accumulation
> report; this is the STT-garble investigation the prompt called `bugs/002`.
> Filed as `003` to avoid colliding with an already-merged filename.

> **Scope.** This is a **diagnostic** doc. It stops after § 4. No fix code,
> no probe code, no decision on which option (FIX-DOC / FIX-PIPE / FIX-BOTH)
> is correct. The probe matrix in § 4 is the plan; running it is gated on
> your approval.

---

## § 1. Symptom

**Pipeline under investigation:** `scripts/m0/` two-stage smoke.

```
scripts/m0/smoke_tts.py          →  VoxCPM2 synth  →  scripts/m0/smoke_tts.wav
scripts/m0/smoke_stt.py          →  reads smoke_tts.wav  →  FunASRNano.inference(language="英文")
```

**Prompt asked of VoxCPM2** (`smoke_tts.py:12`):

```
"Hello from the voice platform smoke test."
```

(7 words, 42 characters.)

**Transcripts returned by Fun-ASR-Nano-2512** (three runs, all garbled):

| Run | Date | Invocation | Transcript |
|-----|------|------------|------------|
| A | 2026-04-16 17:52 | `smoke_stt.py` as-shipped: `m.inference(data_in=[WAV], language="英文", ...)` | `"As for the dish, cooks are ashamed to taste it."` |
| B | 2026-04-16 (same session, follow-up) | same but `language` override removed | `"As for the dish, cooks the dish like the fish."` |
| C | 2026-04-18 (today, re-run) | `smoke_stt.py` as-shipped | `"Oh yes, yes, it can."` |

(Run-A / Run-B transcripts are from `docs/m0_smoke_report.md:58-61`. Run-C
is as reported in the debug prompt; the on-disk `scripts/m0/logs/smoke_stt_run.log`
is still the Apr-16 copy — the Apr-18 run was not captured into that log,
so one of the § 4 probes is "re-run and capture".)

**WER / CER** (ref="Hello from the voice platform smoke test.", computed via
`jiwer==4.0.0`. Both metrics use a **single** normaliser —
`jiwer.Compose([ToLowerCase, RemovePunctuation, Strip, RemoveMultipleSpaces])`
— applied to ref and hyp before calling `jiwer.wer` and `jiwer.cer`, so the
two metrics are directly comparable):

| Run | WER | CER |
|-----|-----|-----|
| A | **1.286** | **0.800** |
| B | **1.286** | **0.775** |
| C | **1.000** | **0.850** |

Every run is effectively "no words correct." Hyp lengths in Runs A/B exceed
the reference (10 and 9 words vs. 7), so WER can — and does — exceed 1.0.

**Commands + timestamps to reproduce (as shipped):**

```bash
# Run A + B (archived, 2026-04-16):
cd scripts/m0
uv sync
uv run python smoke_tts.py       # produces smoke_tts.wav (16000 Hz, 16-bit, mono, 9.12s)
uv run python smoke_stt.py       # transcribes the above

# Run C (today, 2026-04-18) — same commands, new environment snapshot.
# Not captured into scripts/m0/logs/; will be re-captured by § 4's P1.
```

**Timings (Run A, `scripts/m0/logs/smoke_stt_run.log:2568-2569`):**

```
[stt] model+vllm loaded in 229.01s
[stt] transcript: 'As for the dish, cooks are ashamed to taste it.'
```

**Duration anomaly (flagged in the original m0 smoke report,
`docs/m0_smoke_report.md:147`, recorded as an open question but not followed
up):**

The WAV is **9.12 s long** for a 7-word sentence that a natural reader
delivers in roughly 3 s. This is a **3.04× stretch**. That number lands
within 1.5 % of 48000 / 16000 = 3.00 — the ratio you would see if VoxCPM2's
native output were 48 kHz but the WAV were written (and re-read) as if it
were 16 kHz. Recording this here because it's a concrete, measurable clue,
not a conclusion — § 3 lists this as one hypothesis among several.

---

## § 2. Context probe — what does the single-session STT test actually test?

The debug prompt claimed `test_real_model.py` passes a single-session STT
test with WER=0.00. That is **not quite accurate**, and the correction is
informative.

**Finding:** `packages/gateway/tests/integration/test_real_model.py` has **no
STT tests at all** — it only exercises voice cloning and TTS. Its
`_real_voice_bytes()` fixture synthesises a 2 s vowel-analog sinusoid at
24 kHz for upload-as-reference-clip tests; that audio is never fed to
Fun-ASR.

The STT single-session test the prompt was thinking of lives in a sibling
file: `test_real_model_stream.py::test_stt_stream_end_to_end_via_synthesized_audio`
(`packages/gateway/tests/integration/test_real_model_stream.py:268-367`).
Per bugs/001's recorded run (bugs/001 § 1: "WER=0.00"), that test does
pass cleanly on VoxCPM2-synthesised audio.

**Critical difference between the gateway test's pipeline and the M0 smoke
script's pipeline** — which is load-bearing for § 3's hypothesis list:

| Step | Gateway test path | M0 smoke path |
|------|-------------------|---------------|
| 1. TTS request | `POST /v1/tts { sample_rate: 24000, output_format: "wav" }` | `VoxCPM.from_pretrained(...).generate(target_text=TEXT)` — raw library call, no gateway |
| 2. What emerges | WAV bytes at **24 kHz** (gateway resamples worker output to the requested rate server-side — see `packages/voxcpm_worker/src/larynx_voxcpm_worker/model_manager.py:650-655`, librosa resample with `target_sr=24000`) | Python list of np arrays at VoxCPM2's **native** rate |
| 3. Normalisation | `_wav_to_pcm16_16k(wav)` (`test_real_model_stream.py:236-247`) — `sf.read` picks up the WAV's declared rate, then decimates to 16 kHz int16 | `sample_rate = getattr(server, "sample_rate", None) or getattr(server, "sr", 16000)` → `sf.write(..., samplerate=sample_rate)` — **whatever the server attribute happens to return**, fallback 16000, no resample |
| 4. What Fun-ASR sees | Genuine 16 kHz int16 PCM, passed in as `bytes` through `transcribe_rolling(pcm_s16le=..., sample_rate=16000)` | A WAV file path; Fun-ASR's own frontend (`frontend.fs`) loads it via `load_audio_text_image_video(sub_str, fs=frontend.fs, ...)` |
| 5. Language tag | Explicit `language="en"` via gateway config message | Explicit `language="英文"` (Chinese for "English") — the format Fun-ASR's Chinese prompt template expects (see `scripts/m0/Fun-ASR-vllm/model.py:560-568` and `Fun-ASR-vllm/README.md:89-94`) |

**One-paragraph synthesis.** The gateway test feeds Fun-ASR audio that is
unambiguously, and independently-verified, 16 kHz mono int16 — because
`_wav_to_pcm16_16k` re-derives the rate from the WAV header and resamples
if it isn't 16 k. The M0 smoke script writes the WAV using whatever
sample-rate attribute the `nanovllm_voxcpm.VoxCPM` instance happens to
expose; if the actual native output rate of VoxCPM2's HIFI decoder doesn't
equal that attribute, the WAV is **mis-labeled**, and when Fun-ASR's
frontend treats it as its declared rate, it's effectively operating on
time-stretched and pitch-shifted audio. The gateway test and the M0 script
are **not the same pipeline**, and the WER=0.00 on the gateway test tells
us nothing about whether the M0 smoke path is sane.

Background context that further reduces confidence in "Fun-ASR just
struggles with TTS audio": `test_real_model_stream.py:268-367` proves
Fun-ASR-Nano-2512 can transcribe VoxCPM2-synthesised English at WER=0.00
when the pipeline feeds it 16 kHz int16. So *Fun-ASR on synthesised speech*
is clearly not, in general, a disaster.

Short version: the M0-vs-gateway delta is the first place to look.

---

## § 3. Hypotheses

Listed, not ranked. No prior-weighting applied; § 4 designs probes that
distinguish them.

**H1. Fun-ASR-Nano genuinely struggles on TTS-synthesised audio.**
Prosody, absent breath noise, spectral artefacts. Not contradicted by
the garble alone, but **strongly tensioned** by the fact that
`test_stt_stream_end_to_end_via_synthesized_audio` passes with WER=0.00
on VoxCPM2-synthesised audio too. For H1 to be the cause, some property
specific to the M0 script's VoxCPM2 call (different voice, different
config, different raw-decoder output) would need to make the output
harder than the gateway's TTS output. Not impossible. Probe-able.

**H2. Sample-rate mis-labelling at the TTS-writer layer.**
`smoke_tts.py:28` reads `getattr(server, "sample_rate", None) or
getattr(server, "sr", 16000)`. If `VoxCPM.sample_rate` doesn't exist and
`VoxCPM.sr` is `None`, the fallback is 16 000. But the VoxCPM2 HIFI decoder
natively outputs at 24 kHz or 48 kHz (the worker's `DEFAULT_OUTPUT_SR` is
**48000** — `packages/voxcpm_worker/src/larynx_voxcpm_worker/model_manager.py:53`).
If the samples are really at 48 kHz but the WAV header says 16 kHz, Fun-ASR's
frontend — which trusts the header — operates on audio whose wall-clock
content has been stretched 3× and pitch-shifted down by an octave and a
fifth. *Specific circumstantial evidence:* the WAV's actual duration is
**9.12 s** for a 3-second phrase, a 3.04× stretch that lines up with
48000/16000.

**H3. Channel layout mis-handling.** The VoxCPM2 generator yields chunked
arrays; `smoke_tts.py:32-36` reshapes each chunk with `.reshape(-1)` and
concatenates. If the upstream generator ever yields a `[T, 2]` stereo shape
(e.g. duplicated-mono), this collapses it to `[2T]` interleaved samples,
turning stereo into twice-as-long silly mono. `file(1)` reports
**mono** for the current `smoke_tts.wav`, so H3 is unlikely *for this file*,
but the collapse-on-reshape pattern is the kind of thing that can silently
change under a VoxCPM upgrade, so H3 still deserves a cheap probe.

**H4. Bit-depth / encoding mis-match.** VoxCPM2's generator emits float32;
`sf.write` with no `subtype=` defaults to `PCM_16` for `.wav` — that's the
conversion we observe. Quantisation noise at PCM_16 is well below Fun-ASR
thresholds. Low-prior; probe it only to be thorough.

**H5. Amplitude clipping / normalisation.** VoxCPM2 output at float32
scale may be hot (peak > 1.0 clipped on write) or quiet enough that
Fun-ASR's log-mel normalisation misfires. Probe by measuring the WAV's
peak/RMS.

**H6. Language auto-detect misfire.** Fun-ASR-Nano-2512 is marketed as
zh / en / ja. `smoke_stt.py:39` passes `language="英文"` (Chinese for
"English") — the string the Fun-ASR prompt template expects. Run B had
the override removed and still got a garbled English-ish transcript, which
weakens H6 a lot: if language routing were the cause, we'd expect Run B
to shift to Chinese or Japanese output, but it stayed in English ("cooks
the dish like the fish"). H6 is probably not the root cause, but worth
probing because the confirmation is cheap and the alternative
explanations ("the LLM is generating fluent nonsense") are harder to
falsify without the probe.

**H7. A combination of the above.** Particularly H2 + H5 (wrong sample
rate exposed as amplitude oddity too) or H1 + H2 (model is already stressed
and the sample-rate mis-label finishes the job).

### Hypotheses added during evidence gathering

**H8. VoxCPM2 decoder mis-configuration at `from_pretrained` time.**
`smoke_tts.py:18-24` instantiates `VoxCPM.from_pretrained(model="openbmb/VoxCPM2",
devices=[0], max_num_batched_tokens=8192, max_num_seqs=16,
gpu_memory_utilization=0.80)` — no explicit `sample_rate=` or voice-selection
kwarg. If that kwarg defaults the HIFI decoder to 48 kHz internally but
the class exposes `self.sr = 16000` as a compatibility stub, H2's
mis-label condition is structural, not incidental. Probe with `dir(server)`
/ introspection (P2).

**H9. Target-text tokenisation quirk.** VoxCPM2's tokenizer is declared
`VoxCPM2Tokenizer` but loads as `LlamaTokenizerFast` (warning in
`scripts/m0/logs/smoke_tts_run.log:2-4`). If the fallback tokeniser
mis-handles the prompt text, VoxCPM2 might be generating prosodic noise
that happens to start with English-like phonemes but says something
entirely different. Would explain non-sample-rate-shaped garble (Run C
especially). Probe by listening to the WAV + visual spectrogram inspection.

---

## § 4. Diagnostic probe matrix

Each row is one probe that differentiates at least two hypotheses. Probe
code is **not yet written** — this is the plan, not the implementation.
After approval, I'll materialise these as `scripts/diagnostics/stt_garble_probe.py`.

Probes re-use already-cached VoxCPM2 + Fun-ASR weights from `scripts/m0/`
where possible; no re-download. Expected total GPU time ≤ 10 min once models
are warm.

**Ground truth** = `"Hello from the voice platform smoke test."` Threshold
for "good" WER on real speech in this repo's other suites is 0.5 (the
gateway stream test's tolerance). I'll use **WER ≤ 0.2** as "clean pass"
and **WER ≥ 0.8** as "garbled — indistinguishable from Run A/B/C".

### P1 — Re-run as-shipped M0 pipeline and capture

One-line description: run `smoke_tts.py` then `smoke_stt.py` exactly as
checked in, today, and capture both stdout and the resulting WAV's metadata.

Purpose: (a) freshen Run C's transcript into a versioned log so every
subsequent probe has a baseline to compare against — the prompt's quoted
"Oh yes, yes, it can." is second-hand; (b) confirm the bug reproduces on
today's environment rather than being a one-off.

Distinguishes: confirms the bug still exists (if Run C suddenly works, some
recent infra change fixed it and the rest of the matrix is unnecessary).

Pass/fail: WER against ground truth. Either ≤ 0.2 (bug gone — investigate
what changed) or ≥ 0.8 (confirmed, proceed).

### P2 — Metadata probe on smoke_tts.wav

One-line description: open the WAV with `soundfile.info` and print
`samplerate`, `channels`, `subtype`, `frames`; also open the VoxCPM server
object with `inspect` and print `type(server).__mro__`, `hasattr(server,
'sample_rate')`, the value of `server.sample_rate` and `server.sr` if
present.

Purpose: H2's load-bearing question — does the WAV's declared sample rate
match VoxCPM2's actual native output rate? — must be answered before any
other interpretation is valid.

Distinguishes: H2 vs. H1. If `server.sample_rate` returns 48000 (or 24000)
while the WAV is labelled 16000, H2 is essentially confirmed. If both
agree at 16000 and `server.sample_rate` really is 16000, H2 is ruled out.
H3 (channel) also drops out here: 1-channel means not a stereo folding
artefact.

Pass/fail: informational — no WER computed. Collects the raw facts that
P3 / P5 interpret.

### P3 — Resample probe: write at "true" rate, transcribe

One-line description: regenerate VoxCPM2 audio with `smoke_tts.py`, then
write **two** WAV variants: (a) one as the current code does
(whatever-server-reports, falling back to 16 000); (b) one at each
candidate native rate from P2's output (24000 and 48000 are the likely
candidates). For each variant, run `librosa.resample(..., target_sr=16000)`
to produce a known-good 16 kHz int16 WAV, and transcribe that through
Fun-ASR with `language="英文"` and the current `smoke_stt.py` sampling
params.

Purpose: direct test of H2. If the current-code variant garbles but the
"correctly-declared-then-resampled" variant transcribes cleanly, H2 is the
root cause and a sample-rate fix will solve the bug.

Distinguishes: H2 vs. H1, H3, H5. Also drops out H4 implicitly (all
variants are written as PCM_16).

Pass/fail: each variant gets a WER. "Correctly-resampled" should land
< 0.2 if H2 is the cause; "as-shipped" should stay ≥ 0.8.

### P4 — Human-speech control via Fun-ASR's own English sample

One-line description: transcribe `scripts/m0/Fun-ASR-vllm/example/en.mp3`
(shipped with the Fun-ASR checkpoint — real human English) through
**exactly the same** `FunASRNano.inference(language="英文", ...)` call the
M0 script makes. Report the Fun-ASR transcript and compute WER against
the example clip's shipped transcript (if the checkpoint includes one; if
not, eyeball-check the transcript is sensible English and use that as a
qualitative signal).

Purpose: tests H1. Human-speech input through identical Fun-ASR config →
clean transcript means Fun-ASR + its config are fine and all failures are
on the input-audio side. Garbled here too would push H1 (or an environment
problem like GPU corruption) into play.

Distinguishes: H1 vs. everything else. If P4 is clean and P3 is still
garbled, H1 is dead.

Pass/fail: qualitative sensibleness + (if reference transcript available)
WER ≤ 0.2.

### P5 — Language-override ablation on fresh-today audio

One-line description: on the WAV from P1, run Fun-ASR **three times**:
once with `language="英文"`, once with `language=None`, once with
`language="中文"` (Chinese, deliberately wrong). Print all three
transcripts; compute WER against ground truth for each.

Purpose: tests H6. The existing Run A vs. Run B data already gives us two
of the three points (both still garble to English-shaped nonsense), but
running all three on the same WAV under controlled conditions makes the
comparison tight. If `language="中文"` produces a plausible phonetic
transliteration to Chinese characters but `language="英文"` and
`language=None` produce nonsense, Fun-ASR's English decoder is being
routed out of distribution by the TTS artefacts — an H1 flavour, not H6.

Distinguishes: H6 from H1. Cheap to run on the back of P1's WAV — no new
synthesis required.

Pass/fail: tabulated WERs per language tag; look for structure.

### P6 — Spectrogram / peak-RMS inspection

One-line description: load the WAV from P1 at its **declared** rate and
print: peak amplitude, RMS dBFS, DC offset, and a rough spectral centroid
(via `librosa.feature.spectral_centroid`). Same measurements at an
interpretation-as-48-kHz rate if P2 finds a mismatch.

Purpose: tests H5 (clipping / normalisation). A WAV peaking at 32767
(saturation) or averaging below -60 dBFS (whisper-quiet) is a red flag
independent of sample-rate questions. Spectral centroid that's wildly out
of range for speech (e.g. < 500 Hz or > 5 kHz at the declared rate, when
speech usually sits 1-3 kHz) is consistent with a sample-rate
mis-label — if you mis-declare 48 k as 16 k, the centroid reported
*at 16 k* will be 1/3 of its true value.

Distinguishes: H5 vs. H2. Also cross-validates P2/P3.

Pass/fail: informational.

### P7 — Eight-session-sibling: synthesise-at-24 kHz path, smoke_stt it

One-line description: write a short throwaway script that (a) asks the
gateway (`POST /v1/tts`) for the same phrase at `sample_rate=24000`,
saves the resulting WAV to `scripts/diagnostics/outputs/`, then (b) runs
exactly `smoke_stt.py`'s transcription logic on that WAV. No streaming,
no VAD — just the Fun-ASR inference path.

Purpose: independent corroboration. The gateway's TTS path is known-good
(bugs/001 WER=0.00 evidence). If this WAV transcribes cleanly through
`smoke_stt.py`, the delta is entirely on the TTS-writer side of the M0
script — the STT-side code is fine. If this WAV **also** garbles, the
bug is on the STT-reader side, not the TTS-writer side.

Distinguishes: "writer bug" vs. "reader bug" — refines H2/H3 into a
subtle but important sub-question. Requires the gateway to be startable,
which it is (bugs/001 confirmed).

Pass/fail: WER.

### Matrix summary

| Probe | Drives out | Cost | Risk |
|-------|-----------|------|------|
| P1 | Confirms bug still reproduces today | ~1 min | None |
| P2 | H2, H3, H8 | ~30 s | None |
| P3 | H2 vs. H1/H5 | ~3 min | Uses models from M0 cache |
| P4 | H1 | ~1 min | None |
| P5 | H6 | ~2 min | None |
| P6 | H5 | ~30 s | None |
| P7 | Writer-side vs. reader-side narrows H2 | ~3 min | Requires gateway start |

After running these, the expected survivor is **one** of H1 / H2 / H6,
or an H7-hybrid (likely H1+H2). H4 and H3 are probably eliminated by P2.
H5 is cross-checked as side-info.

If a probe returns something that doesn't match any of H1-H9, I'll add
H10+ to § 3 **before** running the next probe and re-plan.

---

## § 5. Probe results

Executed via `scripts/diagnostics/stt_garble_probe.py` on 2026-04-18 at
21:17 local. Raw run logs live at
`scripts/diagnostics/outputs/probe_run_20260418_211755.log` and the JSON
summary at `probe_summary_20260418_211755.json` (both gitignored).

Models were phased (VoxCPM2 first, stopped + released, then Fun-ASR) to
fit two vLLM instances on one GPU; doesn't affect the observations. All
WER/CER use the single `NORMALISE` compose from § 1.

### § 5.1 P1 — as-shipped pipeline reproduces the bug

```
synth: 1.12s  samples=161280  declared_sr=16000
wav meta: samplerate=16000, channels=1, subtype=PCM_16, frames=161280, duration_s=10.08
transcript: "Honor, honor, yes, but don't share twice."
WER=1.000  CER=0.725
verdict: GARBLED (bug reproduced)
```

Note: the WAV is slightly longer today (10.08 s vs. April's 9.12 s) —
VoxCPM2 is non-deterministic without a fixed seed, so utterance length
jitters. Garble verdict is the same.

### § 5.2 P2 — VoxCPM server object has *no* sample-rate attribute

```
type(server).__mro__ = ['SyncVoxCPM2ServerPool', 'object']
sr-ish attributes: ['generate']      # literally the only hit — a method, not a rate
config.json sr-keys: (none)           # top-level voxcpm2 config doesn't surface them either
```

Cross-check against `nanovllm_voxcpm` source
(`third_party/nanovllm-voxcpm/nanovllm_voxcpm/models/voxcpm2/server.py`):

- `SyncVoxCPM2ServerPool` (the class `VoxCPM.from_pretrained` returns)
  never defines `sample_rate`, `sr`, `output_sample_rate`, `fs`, or
  anything similar on the pool wrapper.
- The rate lives two layers down, on the inner `VoxCPM2ServerImpl`:
  `self.encoder_sample_rate = model_runner.vae.sample_rate`,
  `self.output_sample_rate = model_runner.vae.out_sample_rate`.
- The only way to ask *the pool* for the rate is
  `await server.get_model_info()` → `ModelInfoResponse` with
  `output_sample_rate` populated from the VAE.
- `layers/audio_vae_v2.py:340`: `out_sample_rate: int = 48000`.
  `models/voxcpm2/config.py:65`: `out_sample_rate: int = 48000`.

**So VoxCPM2 natively emits at 48 000 Hz**, and the M0 smoke script's
`getattr(server, "sample_rate", None) or getattr(server, "sr", 16000)`
**always** falls through to the 16000 literal — there is nothing on the
pool object to find. The 16 kHz label on `smoke_tts.wav` has been
accidental since day one.

(My "observed stretch ≈ 4.34×" heuristic in § 1 was based on an
aggressive 0.3 s/word natural-speed assumption. Using a more realistic
~0.45 s/word → expected ≈ 3.15 s natural, observed ≈ 3 s for the actual
48 kHz-interpreted audio (161 280 ÷ 48 000 = 3.36 s). The raw 48 kHz fact
from the source is cleaner evidence than the speech-rate heuristic; the
heuristic was just an early signpost.)

### § 5.3 P3 — resample-to-16 k fixes the transcript (smoking gun)

Three variants transcribed through Fun-ASR-Nano with `language="英文"`:

| Variant | Note | Transcript | WER | CER |
|---------|------|-----------|-----|-----|
| A as-shipped | declared 16 k, no resample | "Honor, honor, yes, but don't share twice." | 1.000 | 0.725 |
| B_relabel_24000_then_16k | interpret as 24 k, librosa kaiser_best → 16 k | **"Hello from the voice platform smoke test."** | **0.000** | **0.000** |
| B_relabel_48000_then_16k | interpret as 48 k, librosa kaiser_best → 16 k | **"Hello from the voice platform smoke test."** | **0.000** | **0.000** |

Both B variants transcribe **perfectly**. The 24 k relabel "working" too
is not contradictory: the samples are just float32 values, and librosa's
kaiser-best anti-alias + decimation from either 24 k or 48 k down to 16 k
produces audio that Fun-ASR can follow. The 24 k interpretation plays
the content at 2× the 48 k interpretation's pace — still inside the
model's intelligibility window for clearly-articulated synthesised
speech. Only the **true** native rate matters for writing a correctly-
labelled WAV (and that's 48 k per § 5.2).

**Early-stop condition met: H2 confirmed.**

### § 5.4 P4 — Fun-ASR is healthy on real human speech

Input: `Fun-ASR-Nano-2512/example/en.mp3` (shipped English sample).

```
transcript: 'The tribal chieftain called for the boy and presented him with fifty pieces of gold.'
contains common English connector word: True
```

Coherent, fluent, punctuated English. The Fun-ASR model + config + vLLM
sampling params used by the M0 script are working fine — H1 is dead.

### § 5.5 P5 — language tag isn't the cause

Language ablation on today's (as-shipped, mis-labelled) WAV:

| Language tag | Transcript | WER | CER |
|--------------|-----------|-----|-----|
| 英文 (en) | "Honor, honor, yes, but you are so close." | 1.143 | 0.750 |
| None (auto) | "Honor, honor, yes, but you are so close." | 1.143 | 0.750 |
| 中文 (zh) | "安儿安儿，不要不要删，不要删。" | 1.000 | 1.000 |

`英文` and `None` produce the same English garble — auto-detect is
picking English correctly. `中文` routes the same stretched audio through
the Chinese decoder and produces phonetic-approximation Chinese nonsense
("An'er An'er, don't-don't delete, don't delete."), which is the
expected Chinese-decoder failure mode on non-Chinese input. H6 is dead:
swapping the language tag doesn't fix the transcript, and the shape of
the zh-tagged output is consistent with Fun-ASR's language routing
working correctly — there's no routing bug, the audio is simply wrong.

(Run A vs. Run C of § 1 returned different English garbles even with the
same `英文` tag. That's expected: VoxCPM2 is non-deterministic, so the
mis-labelled audio is different each run; Fun-ASR's vLLM sampling with
`top_p=0.001` then "hallucinates" whichever plausible English sentence
best matches the degraded acoustic input. This also rules out "sampling
noise" — all runs are garbled with consistently-high WER.)

### § 5.6 P6 — spectral centroid corroborates the mis-label

```
at declared sr=16000:  peak=0.8421  rms_dbfs=-18.54  dc=-0.00012  centroid_hz=1042.9
```

Peak 0.84 (no clipping — H5 is dead on the amplitude side); RMS
-18.5 dBFS (healthy level); DC offset negligible. Spectral centroid
1042.9 Hz *at declared 16 kHz* is ~1/3 of typical English-speech centroid
(~2000-3000 Hz at 16 kHz) — consistent with the samples actually being
48 kHz material being read as 16 kHz, which compresses the declared
spectrum by 3×. H5 is cleanly ruled out as a cause; the centroid is
additional H2 evidence.

---

## § 6. Diagnosis

**H2 — sample-rate mislabelling at the M0 TTS writer.** VoxCPM2's native
output is 48 kHz; the M0 `smoke_tts.py` writes the WAV as 16 kHz because
the `SyncVoxCPM2ServerPool` object has no `sample_rate` attribute and the
`getattr(..., "sr", 16000)` fallback fires. Fun-ASR's frontend trusts
the WAV header, sees 3× slowed / pitch-shifted input, and hallucinates
plausible-English garble.

Supporting evidence:

- § 5.3 P3: relabelling to the true 48 kHz rate and resampling to 16 kHz
  produces a **WER=0.000** transcript of the exact reference text.
- § 5.2 P2: `SyncVoxCPM2ServerPool` genuinely has no sample-rate attr;
  `nanovllm_voxcpm` source confirms native output is 48 000 Hz.
- § 5.6 P6: spectral centroid at declared rate is 1/3 of expected, the
  ratio implied by 48 k → 16 k mis-label.

Hypotheses ruled out:

- **H1** (Fun-ASR struggles on TTS audio): § 5.4 P4 transcribes shipped
  human English sample cleanly, and § 5.3 P3's resampled variant
  transcribes VoxCPM2 output cleanly too — the model handles both real
  and synth speech just fine when the rate is correct.
- **H3** (channel-layout mis-handling): § 5.1 P1 WAV metadata shows
  `channels=1`; no folding artefact possible.
- **H4** (bit-depth mis-match): § 5.1 P1 shows `PCM_16`; librosa
  float32→int16 round-trip is standard and correct.
- **H5** (amplitude clipping / normalisation): § 5.6 P6 peak=0.84,
  rms=-18.5 dBFS — well-behaved audio levels.
- **H6** (language auto-detect misfire): § 5.5 P5 — `英文` and `None`
  produce identical English garble; `中文` produces expected Chinese
  pinyin-shape garble. Language routing works; audio is the problem.
- **H7/H8/H9** (various speculative compositions): subsumed by the
  cleaner single-cause finding of H2; no need to invoke them.

No hypothesis beyond H1-H9 was needed — nothing in the probe output
surprised me in a way that required an H10+.

---

## § 7. Fix recommendation

**OPTION FIX-PIPE.**

Short version: the M0 `smoke_tts.py` writer must stop guessing the
sample rate via best-effort `getattr` and instead use the authoritative
rate reported by the VoxCPM2 server. The fix is code, small, and scoped
to `scripts/m0/smoke_tts.py` (and possibly `smoke_stt.py` if we want to
harden the read-side too, but that's optional since Fun-ASR re-reads the
WAV header through its own frontend). No PRD/docs update needed — this
isn't a Fun-ASR quality claim to document; it's a plain writer bug.

Justification:

1. The garble is **entirely** on the writer side. § 5.3 P3 proves that
   the same VoxCPM2 output, relabelled and resampled, transcribes at
   WER=0.000 — the same model, the same prompt, the same Fun-ASR
   call. The delta between pass and fail is a single number in the WAV
   header.
2. The gateway pipeline already does this correctly (§ 2 context probe):
   `packages/voxcpm_worker/src/larynx_voxcpm_worker/model_manager.py:650-655`
   resamples the worker's native-rate output to whatever the request
   asked for before returning WAV bytes. The M0 script is a one-off
   research helper that predates that plumbing; we should align it with
   how the production path handles the same problem.
3. FIX-DOC would frame the garble as a Fun-ASR limitation on synthetic
   speech. § 5.4 + § 5.3 prove that's not what we're seeing, so a doc
   update would be misleading to future readers.

Fix shape (sketch — does **not** commit code until the three-commit
sequence below):

- `smoke_tts.py` — replace the `getattr(server, "sample_rate", None) or
  getattr(server, "sr", 16000)` guess with a call to the authoritative
  `server.get_model_info()` (or equivalent) to obtain `output_sample_rate`
  — then either write at that rate **or** resample to 16 kHz first.
  Match what `_wav_to_pcm16_16k` does in the gateway tests so the M0
  smoke exercises the same audio-normalisation path production uses.
- `smoke_stt.py` — optional, out of scope here. Fun-ASR's own frontend
  already honours whatever rate the WAV header declares; if the header
  is correct, no change needed on the reader.

---

## § 8. (post-fix verification — written after the FIX-PIPE commits land)

(pending)

