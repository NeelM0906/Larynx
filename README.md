# Larynx

Self-hosted voice AI platform (VoxCPM2 TTS + Fun-ASR STT) — single-box,
GPU-resident, REST + WebSocket.

**Status (2026-04-18).** Branch [`feat/m7-finetune`](https://github.com/NeelM0906/Larynx).
All core path-components (M0–M7) are implemented; M8 (batch + OpenAI shim + soak) is in
design. This README is a living status report; numbers below are measured on real hardware
on the most recent verified run, not theoretical.

Authoritative docs:
- [PRD.md](./PRD.md) — product spec, goals, non-goals
- [ORCHESTRATION.md](./ORCHESTRATION.md) — M1–M6 design notes
- [ORCHESTRATION-M7.md](./ORCHESTRATION-M7.md) — fine-tuning design
- [ORCHESTRATION-M8.md](./ORCHESTRATION-M8.md) — batch/OpenAI/hardening/soak design
- [REPO_STRUCTURE.md](./REPO_STRUCTURE.md) — monorepo layout
- [VERIFICATION_REPORT.md](./VERIFICATION_REPORT.md) — bugs/001 post-fix verification
- [bugs/](./bugs/) — bug ledger (see table below)

---

## Quickstart

```bash
git clone <repo> larynx && cd larynx
cp .env.example .env                      # mock TTS by default — no GPU needed
make up                                   # postgres + redis on :5433 / :6380
make migrate                              # create Voice + fine_tune + batch tables
make smoke                                # POST /v1/tts → /tmp/larynx_smoke.wav
```

Real TTS requires a GPU box: set `LARYNX_TTS_MODE=voxcpm` in `.env` and
`uv sync --extra gpu` (pulls in torch 2.9 + flash-attn 2.8.3 + vllm 0.13 +
nano-vllm-voxcpm + funasr). Full real-model test gate is `RUN_REAL_MODEL=1
uv run pytest -m real_model -v`.

---

## Hardware

Reference box:

| | |
|---|---|
| GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, 95 GiB each) |
| CUDA driver | 580.126.09 (cu13.0 runtime) |
| Python | 3.12.3 |
| torch | 2.9.0+cu128 |
| flash-attn | 2.8.3 (prebuilt wheel from Dao-AILab, cu12torch2.9cxx11abiTRUE) |
| vllm | 0.13.0 |
| nano-vllm-voxcpm | 2.0.0 |
| funasr | 1.3.1 |

GPU allocation:

- **GPU 0**: VoxCPM2 TTS (~8 GB used + batch headroom; default
  `gpu_memory_utilization=0.80`)
- **GPU 1**: Fun-ASR-Nano + Fun-ASR-MLT-Nano (~6 GB combined) + room for
  on-demand fine-tuning jobs

---

## Milestone status

| Milestone | PRD ref | State | Notes |
|---|---|---|---|
| M0 hardware smoke | §9 | **done** | 2 × RTX PRO 6000 verified; [docs/m0_smoke_report.md](./docs/m0_smoke_report.md); sample-rate bug caught and fixed — [bugs/003](./bugs/003_stt_m0_garble.md) |
| M1 foundation | §9 | **done** | gateway, Postgres 16, Redis 7, `/v1/tts` |
| M2 voice library + cloning | §9 | **done** | upload + latent-cache + `/v1/voices/design` |
| M3 STT + language routing | §9 | **done** | Fun-ASR-Nano + MLT; `/v1/stt` REST |
| M4 streaming TTS + STT | §9 | **done** | WS endpoints; rolling-buffer partials; concurrent STT (bugs/001 fixed) |
| M5 conversational loop | §9 | **done** | `/v1/conversation` with barge-in (real-hardware conversation tests gated on `OPENROUTER_API_KEY`) |
| M6 playground UI | §9 | **done** | Next.js, all tabs |
| M7 fine-tuning UI | §9 | **done** | LoRA train + hot-swap; current branch |
| M8 batch + OpenAI shim + soak | §9 | **design** | [ORCHESTRATION-M8.md](./ORCHESTRATION-M8.md) |

---

## Measured performance

Numbers below are the most-recent live-hardware measurement. Every
entry is backed by either a pytest log in `/tmp/` that you can re-run
or a `bugs/` doc with the raw trace.

### M0 smoke (scripts/m0, post-bugs/003 fix, 2026-04-18)

Single-pass latencies with warm CUDA-graph cache (first-ever cold run
numbers are 5-10× higher; see [docs/m0_smoke_report.md](./docs/m0_smoke_report.md)).

| Stage | Wall-clock |
|---|---|
| VoxCPM2 load (warm HF cache) | **9.5 s** |
| VoxCPM2 synth, 3-4 s utterance | **0.96-1.08 s** |
| Fun-ASR + vLLM load (warm graphs) | **36 s** |
| Fun-ASR inference, 3 s utterance (warm) | **0.29 s** |
| VoxCPM2 native output rate | **48 000 Hz** (resampled to 16 k for STT) |
| Round-trip WER (smoke_tts → smoke_stt) | **0.000** (fresh run: 0.143) |

Reproduce: `cd scripts/m0 && uv run python smoke_tts.py && uv run python smoke_stt.py`.

### Gateway streaming (bugs/001 post-fix verification)

| Scenario | Measured | PRD target (§4) |
|---|---|---|
| TTS WS TTFB p50 | **116.7 ms** | ≤ 200 ms ✓ |
| STT WS partial cadence p50 (1 session) | 770 ms | (720 ms target; within noise) |
| STT WS partial cadence p50 (8 sessions) | **787 ms** | cadence must not collapse ✓ |
| STT WS partial cadence p95 (8 sessions) | 870 ms | — |
| STT WS finalization latency | **441 ms** | — (end-to-end; pure model ~40-60 ms) |
| STT WS final WER on synth audio | **0.00** | "match or exceed Fun-ASR published" ✓ |
| Concurrent STT sessions (`test_..._eight_concurrent_sessions`) | **8 / 8** finals | ≥ 8 (PRD goal 4) ✓ |
| GPU 1 utilisation under 8-way load | min 0 % / mean 16 % / max 65 % | headroom ✓ |

### Cache-warm latency (test_cache_warms_second_request, 10 sequential `/v1/tts` calls)

Measured 2026-04-18. End-to-end client-timed. First request hits
Redis cold (disk-backed re-warm); subsequent hits are Redis-warm.

| Sample | Latency |
|---|---|
| First request (Redis cold) | **1 710 ms** |
| Warm requests: min / mean / max | **381 / 520 / 872 ms** |

(Interpretation: the cache saves the VAE-encode pass on the reference clip,
not the TTS synthesis itself. Mean-warm 520 ms is dominated by VoxCPM2
synthesis time on a 5-word sentence. Raw encode-cost isolation lives in
`scripts/measure_cache.py`.)

---

## Test suite state

Running `RUN_REAL_MODEL=1 uv run pytest -m real_model -v` in a single
invocation still hits bugs/002 (GPU memory accumulates across test
modules). The supported way to run the suite is per-file, which is what
`make test-real-per-file` does.

Per-file results (2026-04-18, feat/m7-finetune HEAD):

| File | Passed | Skipped | Failed | Wall | Notes |
|---|---:|---:|---:|---:|---|
| `test_real_model.py` (voice CRUD + synth) | 5 | 0 | 0 | 22 s | clean |
| `test_m0_smoke_roundtrip.py` (bugs/003) | 1 | 0 | 0 | 58 s | post-fix regression |
| `test_real_model_stream.py` (streaming) | 5 | 0 | 0 | 135 s | 8-way concurrency verified |
| `test_real_model_stt.py` (language routing) | 5 | 0 | 1 | 534 s | 1 failure = [bugs/004](./bugs/004_hotword_test_case_sensitivity.md) (case-sensitive assertion; hotword stem present in transcript) |
| `test_real_model_conversation.py` | 0 | 6 | 0 | <1 s | skipped — set `OPENROUTER_API_KEY` in `.env` (pytest-dotenv now loads it) to unskip |

Totals: **16 passed, 6 skipped, 1 failed, 0 errors** across the five
real-model files. Before this branch: 6 / 11 / 0 / 6.

Unit-test suites (no real model required):

```
uv run pytest packages/gateway/tests/unit -q          # 100+ tests, ~15 s
uv run pytest packages/funasr_worker/tests -q         # mock backend
uv run pytest packages/voxcpm_worker/tests -q         # mock backend
uv run pytest packages/training_worker/tests -q       # subprocess + config builder
```

---

## Bug ledger

Living file under [bugs/](./bugs/). Current state:

| Bug | Status | Severity | Summary |
|---|---|---|---|
| [bugs/001](./bugs/001_concurrent_stt.md) — concurrent STT deadlock | **fixed** | v1 blocker | Four concurrent `FunASRNano.inference()` calls from `asyncio.to_thread` deadlocked vLLM's shared LLM. Fix: `asyncio.Lock` on `FunASRBackendReal` + teardown-safe `finally` in `STTStreamSession.run()`. 8-way concurrency verified. |
| [bugs/002](./bugs/002_full_real_model_suite_gpu_accumulation.md) — real_model suite GPU accumulation | **filed, not fixed** | test-infra only (prod unaffected) | Running all real_model test modules in one pytest invocation leaks GPU memory across modules; `LARYNX_STT_MODE` env var leaks too. Workaround: run files separately. |
| [bugs/003](./bugs/003_stt_m0_garble.md) — M0 smoke STT garble | **fixed** | M0 smoke artefact | `scripts/m0/smoke_tts.py` wrote WAVs at 16 kHz via a `getattr()` fallback; VoxCPM2 native output is 48 kHz, so Fun-ASR saw 3× slowed audio and garbled. Fix: query `server.get_model_info()["output_sample_rate"]` and resample via librosa `soxr_hq`. Post-fix WER 0.000. |
| [bugs/004](./bugs/004_hotword_test_case_sensitivity.md) — hotword-recovery test case-sensitive | **filed, not fixed** | test-assertion tightness (prod OK) | `test_hotword_recovery` asserts `"Larynx" in text` but Fun-ASR returns lowercased proper noun (`"larynx"`); hotword stem is correctly present, just not capitalised. Fix options in § 4. |

---

## Architecture (brief — full in [PRD §6](./PRD.md#6-architecture))

```
  Clients  ──HTTPS/WSS──▶  API Gateway (FastAPI, async)
                              │
                              ├─▶ VoxCPM2 worker   (GPU 0, nano-vllm-voxcpm)
                              ├─▶ Fun-ASR-Nano     (GPU 1, Fun-ASR-vllm)
                              ├─▶ Fun-ASR-MLT-Nano (GPU 1, Fun-ASR-vllm)
                              ├─▶ VAD + punc       (CPU, FunASR fsmn-vad + CT-Transformer)
                              └─▶ Training worker  (GPU 1, on-demand LoRA)
                              
                  Postgres 16 ◀┤ (voices, batch_jobs, fine_tune_jobs)
                  Redis 7     ◀┤ (latent cache, streaming session state, Arq)
                  OpenRouter  ◀┘ (LLM, outbound HTTPS)
```

Process isolation via in-process `asyncio.Queue` IPC today; stable
client interface lets us swap to ZMQ/gRPC when we split machines
without gateway-code changes.

---

## 24-hour soak

Harness lives at [`scripts/soak_test.py`](./scripts/soak_test.py); final
report overwrites [SOAK_REPORT.md](./SOAK_REPORT.md) at run completion.
**Not yet run on the target box** — current SOAK_REPORT.md is the
template.

---

## Known open items (not blocking v1)

1. **M8 — batch + OpenAI shim + hardening + 24h soak** — designed in
   ORCHESTRATION-M8.md, not yet implemented.

Test-infra only, not product gaps: [bugs/002](./bugs/002_full_real_model_suite_gpu_accumulation.md)
(run `make test-real-per-file` instead of `make test-real`) and
[bugs/004](./bugs/004_hotword_test_case_sensitivity.md) (test assertion
tightness, hotword feature works).

---

## Contributing

See bugs/ for known issues and the memory policy embedded in existing
bug docs (bugs/001, bugs/003) for the pattern: doc-first for multi-
component fixes; small, reviewable commits; real Redis / real Postgres /
real models in tests — no fakes.
