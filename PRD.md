# PRD: Voice Platform v1 (VoxCPM2 + Fun-ASR)

**Owner:** Neel
**Status:** Draft v2
**Target ship:** v1 in ~2 weeks of focused build time
**Codename:** *Larynx* (placeholder, rename freely)

---

## 1. Why

We need an open-source, self-hosted voice AI stack that the team owns end-to-end. External providers (ElevenLabs, OpenAI voice, etc.) cost money at scale, lock us in, can't be fine-tuned on our data, and don't let us compose the pipeline the way we want. VoxCPM2 (TTS) and Fun-ASR (STT) are both Apache-2.0, both SOTA-competitive on their tasks, both run on our hardware, and both have vLLM-accelerated serving paths.

This service is the first pillar of the broader AI-native platform. LLM gateway and DB connectors come later; today we're shipping voice as a standalone service that exposes clean APIs the rest of the stack plugs into.

## 2. What it is

A GPU-resident voice service running on a single box (2×RTX Pro 6000, 192GB total VRAM), exposing:

- **REST API** for single-shot TTS, STT, voice management, and batch jobs
- **WebSocket API** for streaming TTS, streaming STT, and real-time conversational loops
- **OpenAI-compatible endpoints** (`/v1/audio/speech`, `/v1/audio/transcriptions`) as drop-in replacements
- **Web playground** for internal experimentation (voice cloning, voice design, parameter tuning)
- **LoRA fine-tuning UI** for creating custom voices from 5–10 minutes of reference audio

Access is internal-only via Tailscale or ngrok. No public exposure in v1.

## 3. Users and use cases

**Primary user:** The 7-person Callagy Recovery team + internal engineers building agent features on top.

**Use cases that must work in v1:**
1. Developer calls REST TTS endpoint, gets a WAV back — drop-in replacement for ElevenLabs in existing code
2. Team member opens the playground, uploads a 10-second voice sample, generates a cloned version reading arbitrary text
3. Team member designs a voice from a natural-language description ("warm, middle-aged female, slight southern lilt") and saves it to the voice library
4. Developer hits the conversational WebSocket endpoint, has a spoken back-and-forth with an LLM (via OpenRouter) with sub-second perceived latency and working barge-in
5. Batch TTS job: long text (e.g., a podcast script) gets rendered with a specified voice
6. Team member uploads 5–10 minutes of audio in the fine-tuning UI, kicks off a LoRA training run, and the resulting voice appears in the library when done
7. Developer calls `/v1/audio/transcriptions` and gets back accurate, punctuated transcripts — including for Chinese dialects and noisy audio

## 4. Goals and non-goals

### Goals

- **TTS latency:** p50 TTFB ≤ 200ms
- **STT latency:** p50 finalization ≤ 80ms on a 3-second utterance
- **Conversational latency:** p50 perceived turn latency ≤ 700ms (user stops speaking → first audio byte back)
- **Concurrency:** ≥ 8 simultaneous conversational sessions without degradation
- **Quality:** TTS indistinguishable from VoxCPM2 reference playground; STT WER matches or exceeds Fun-ASR published numbers on our hardware
- **Reliability:** model workers survive 24h continuous uptime without memory leaks or quality drift
- **DX:** a new engineer can make a working TTS request in < 10 minutes from reading the README

### Non-goals for v1

- Public internet exposure, rate limiting, billing, multi-tenant auth. Internal-only, Tailscale-gated.
- Horizontal scaling / K8s. Single-box. Designed so a K8s move isn't a rewrite but not built for it.
- Semantic turn-detection (beyond acoustic VAD). v1.5.
- Echo cancellation / noise suppression on the server side. Client handles this.
- Phone / SIP integration. v2.
- Speaker diarization for multi-talker conversations. FunASR toolkit supports it but out of v1 scope.
- Migration of existing ElevenLabs-based flows in BOMBA SR. Happens after v1 proves stable.

## 5. Capabilities

### 5.1 TTS core (REST)

`POST /v1/tts`

Inputs: `text`, `voice_id` (optional), `reference_audio` (optional multipart), `prompt_audio` + `prompt_text` (optional, for "ultimate cloning"), `language` (optional, default auto), `cfg_value` (default 2.0), `inference_timesteps` (default 10), `output_format` (wav/mp3/pcm16), `sample_rate` (default 48000). Text supports VoxCPM2's parenthetical voice-design syntax.

Outputs: audio bytes, `X-Voice-ID` and `X-Generation-Time-Ms` headers.

### 5.2 Streaming TTS (WebSocket)

`WS /v1/tts/stream` — client sends text, server streams PCM frames using VoxCPM2's `generate_streaming`. 10ms crossfade between chunks to avoid clicks.

### 5.3 STT core (REST)

`POST /v1/stt`

Inputs: audio (multipart), `language` (optional, auto-detect default), `hotwords` (optional list — Fun-ASR supports this natively), `itn` (inverse text normalization, default true), `punctuate` (default true), `return_timestamps` (default false — Fun-ASR roadmap).

Outputs: `{ text, language, segments?, confidence? }`.

Model routing: auto-detects language and routes to Fun-ASR-Nano (zh/en/ja) or Fun-ASR-MLT-Nano (28 other languages). Fallback to Whisper large-v3 for languages not in either Fun-ASR model.

### 5.4 Streaming STT (WebSocket)

`WS /v1/stt/stream` — client streams PCM; server returns `{type: "partial", text}` events every ~720ms during an utterance and `{type: "final", text, punctuated_text}` when VAD segments close.

Implementation uses Fun-ASR's **rolling-buffer re-decode pattern** (see `demo.py` in the Fun-ASR repo): the model is re-run on a growing audio buffer every 720ms, with the previous partial passed as `prev_text` for context continuity. The last 5 tokens of each intermediate partial are dropped because they're most likely to be revised when more audio arrives; the final chunk keeps all tokens. This gives genuine sub-second partials (not just post-VAD finalization) while using the standard Fun-ASR checkpoint.

Cost: re-decodes the same audio N times per utterance, but at Fun-ASR-vllm's RTFx 180 each pass is ~15-20ms of GPU time — total GPU cost per utterance stays under 100ms.

fsmn-vad runs alongside to detect utterance boundaries (speech-start / speech-end), which is what closes the rolling buffer and emits `final`.

### 5.5 Voice library (REST)

- `POST /v1/voices` — upload reference audio + name → `voice_id`. Server extracts and caches `ref_audio_latents` at upload. **Key latency optimization** — avoids re-encoding per request.
- `GET /v1/voices` / `GET /v1/voices/{id}` / `DELETE /v1/voices/{id}`
- `POST /v1/voices/design` — create a voice from a text description.

### 5.6 Conversational loop (WebSocket)

`WS /v1/conversation` — full-duplex, interruptible.

Client opens connection with config (voice_id, LLM model via OpenRouter, system prompt, VAD sensitivity), then streams audio up and receives audio down.

Per-turn pipeline:
1. Client streams 20ms PCM frames up
2. Server runs fsmn-vad; on speech-start, begins buffering
3. On speech-end (VAD silence threshold + confirmation window ~300ms), buffered utterance goes to Fun-ASR
4. Fun-ASR returns punctuated transcript (~40-60ms on RTX Pro 6000)
5. Transcript → OpenRouter LLM (streaming)
6. LLM first sentence boundary → VoxCPM2 streaming TTS begins
7. TTS PCM frames stream back to client
8. **Barge-in:** if VAD detects new user speech while server is speaking, server cancels LLM + TTS streams, sends `interrupt` event so client stops playback

Control events: `session.start`, `session.end`, `input.speech_start`, `input.speech_end`, `output.speech_start`, `output.speech_end`, `interrupt`, `error`, `transcript.partial`, `transcript.final`, `response.text_delta`.

### 5.7 Batch TTS

`POST /v1/batch` — submit list of (text, voice_id) pairs → job_id. Runs at low priority behind real-time traffic. Redis + Arq for queuing.

### 5.8 Fine-tuning

Web UI at `/finetune` wraps VoxCPM's LoRA pipeline:
1. Upload dataset (audio + transcripts, or audio-only with Fun-ASR auto-transcription)
2. UI validates min duration (~5 min) and audio quality
3. Kick off training via `scripts/train_voxcpm_finetune.py`
4. Progress streaming via SSE
5. On completion, LoRA registered as a new voice with nano-vllm hot-swap (reference: `example_lora.py`)

### 5.9 OpenAI-compatible shim

- `POST /v1/audio/speech` — OpenAI TTS API shape
- `POST /v1/audio/transcriptions` — OpenAI Whisper API shape (uses Fun-ASR under the hood)

### 5.10 Playground UI

Next.js app. Tabs: **TTS**, **Clone**, **Design**, **Library**, **Conversation**, **Transcribe**, **Fine-tune**.

## 6. Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                               Clients                                       │
│   (curl / SDK / Playground UI / BOMBA SR / internal agents)                 │
└───────────────────────────┬────────────────────────────────────────────────┘
                            │  HTTPS / WSS  (via Tailscale)
                            ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  API Gateway  (FastAPI, async)                                              │
│  ─ REST + WS routes                                                         │
│  ─ Auth (bearer)                                                            │
│  ─ Voice library (Postgres + filesystem)                                    │
│  ─ Latent cache (Redis + on-disk .pt)                                       │
│  ─ Conversation orchestrator (VAD + STT + LLM + TTS coordinator)            │
└────┬──────────────┬───────────────┬────────────────┬──────────────┬────────┘
     │              │               │                │              │
     ▼              ▼               ▼                ▼              ▼
┌─────────┐  ┌────────────┐  ┌────────────┐   ┌─────────┐    ┌──────────┐
│ VoxCPM2 │  │ Fun-ASR    │  │ Fun-ASR-   │   │ fsmn-   │    │ Training │
│ worker  │  │ Nano       │  │ MLT-Nano   │   │ vad +   │    │ worker   │
│ GPU 0   │  │ (zh/en/ja) │  │ (28 langs) │   │ punc    │    │ GPU 1    │
│ ~8GB    │  │ GPU 1      │  │ GPU 1      │   │ (CPU)   │    │ on-      │
│ nano-   │  │ ~3GB       │  │ ~3GB       │   │         │    │ demand   │
│ vllm    │  │ Fun-ASR-   │  │ Fun-ASR-   │   │         │    │          │
│         │  │ vllm       │  │ vllm       │   │         │    │          │
└─────────┘  └────────────┘  └────────────┘   └─────────┘    └──────────┘

         OpenRouter (LLM)  ←─── outbound HTTPS from gateway
```

### Key architectural decisions

**Serving engines:** Nano-vLLM-VoxCPM for TTS, Fun-ASR-vllm for STT. Both give ~2-3x throughput over raw PyTorch and support batched concurrent requests.

**Dual STT models always loaded.** Both Fun-ASR variants fit easily on GPU 1 (~6GB combined). A language-router in the gateway picks between them per request based on detected language or an explicit `language` param.

**GPU allocation:**
- GPU 0: VoxCPM2 (~8GB used, rest is batch headroom)
- GPU 1: Fun-ASR-Nano + Fun-ASR-MLT-Nano (~6GB combined) + room for fine-tuning jobs
- Room on both GPUs to later colocate a local LLM if we move off OpenRouter

**VAD: fsmn-vad (FunASR toolkit)** — CPU-based, integrates natively with Fun-ASR, replaces what would've been Silero.

**Punctuation: FunASR's CT-Transformer punc model** — adds proper punctuation/capitalization to Fun-ASR output, runs on CPU. Critical for downstream LLM quality.

**Process isolation:** API gateway and each model worker are separate Python processes communicating via an in-process async queue (v1) with a clear client interface that can swap to ZMQ/gRPC when we split machines. API can restart without reloading models.

**Latent caching is the TTS perf unlock.** On voice upload, run the audio through VoxCPM's VAE encoder once, persist `ref_audio_latents` to disk, cache hot ones in memory.

**Auth:** Single bearer token in `.env` for v1. 7 people, no per-user auth needed. Rotate quarterly.

**Storage:**
- Postgres (Docker): voice metadata, batch jobs, fine-tuning runs
- Filesystem: reference audio, latent caches, generated outputs, LoRA weights
- Redis (Docker): hot latent cache, streaming session state, Arq task queue

### Conversational latency budget

Target p50 perceived latency ≤ 700ms. Breakdown:

- VAD speech-end detection: ~300ms (silence window; tuneable)
- STT finalization: ~40-60ms (Fun-ASR on 2-3s utterance, vLLM-accelerated — **note: because we're running rolling-buffer partials during the utterance, the final decode is on already-warm batches and often faster than a cold decode**)
- Punctuation: ~5-10ms (CPU, tiny model)
- OpenRouter LLM first-token: ~200-400ms (dominant factor — Claude Haiku, GPT-4o-mini, Gemini Flash are fastest)
- First-sentence chunking: ~50ms
- VoxCPM2 TTFB on RTX Pro 6000: ~100ms

Sum: ~700-900ms. **LLM choice is the bottleneck** — biggest v1 latency win is defaulting to a fast model.

**Optimization unlock from streaming STT:** because we now have partial transcripts during the utterance, M5+ can optionally **speculate the LLM call** on partials that look stable (e.g., partial hasn't changed in 500ms). If the final transcript matches the speculation, we skip waiting for speech-end and save up to 300ms. If it doesn't match, we cancel and re-issue. This is a v1.5 optimization but the architecture supports it natively from day one.

## 7. Non-functional requirements

- **Observability:** structured logs (JSON), Prometheus metrics on per-stage latencies. Grafana in v1.1.
- **Resilience:** model worker dies → gateway returns 503 → supervisord auto-restarts.
- **Security:** single bearer token. No PII retention in logs. Generated audio auto-purged after 7 days unless `retain=true`.
- **Licensing:** Apache-2.0 across the stack.

## 8. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Model ecosystem |
| TTS serving | nano-vllm-voxcpm | 2x throughput, async, batched |
| STT serving | Fun-ASR-vllm | RTFx 180, batched |
| STT models | Fun-ASR-Nano-2512 + Fun-ASR-MLT-Nano-2512 | Beats Whisper on accuracy, 31 total languages |
| VAD | fsmn-vad (FunASR toolkit) | CPU, native Fun-ASR integration |
| Punctuation | FunASR CT-Transformer | CPU, production-quality |
| API | FastAPI | Async, WS, OpenAPI auto-gen |
| LLM (v1) | OpenRouter | Model-agnostic |
| Frontend | Next.js + shadcn/ui | Fast iteration |
| DB | Postgres 16 | Familiar |
| Cache/queue | Redis + Arq | Simple async |
| Process mgmt | supervisord + Docker Compose | Single-box |
| Package mgmt | `uv` | Matches VoxCPM and Fun-ASR |

## 9. Delivery plan

**M0 — Hardware smoke test (half-day, do this before M1):** on target box, verify flash-attn installs, nano-vllm-voxcpm runs a single-shot inference, Fun-ASR-vllm runs a single-shot inference. De-risks Blackwell compat early. Exit: one TTS .wav and one STT transcript from the RTX Pro 6000 box.

**M1 — Foundation (days 1–2):** repo scaffold, Postgres + Redis up, VoxCPM2 worker runs in isolation, single-shot TTS via nano-vllm. Exit: `curl POST /v1/tts` returns valid WAV.

**M2 — Voice library + cloning (days 3–4):** voice upload, latent caching, cloning endpoints, voice design. Exit: upload voice, list it, synthesize with it.

**M3 — STT workers + REST (days 5–6):** Fun-ASR-Nano + MLT workers, language-router, punctuation stage, `/v1/stt` endpoint. Exit: transcribe audio in 3+ languages including one dialect.

**M4 — Streaming TTS + Streaming STT (days 7–8):** WS endpoints, chunked PCM, 10ms crossfade for TTS, VAD-gated segmentation for STT. Exit: < 200ms TTFB streaming TTS; streaming STT returns finals within 500ms of speech-end.

**M5 — Conversational loop (days 9–11):** `/v1/conversation` WS, OpenRouter integration, sentence-chunked TTS, barge-in with cancellation tokens. Exit: end-to-end conversation in browser playground with working interrupt, p50 ≤ 700ms.

**M6 — Playground UI (days 12–13):** Next.js with all tabs.

**M7 — Fine-tuning UI (days 14–15):** LoRA training wrapper, dataset upload, progress streaming, hot-swap registration.

**M8 — Batch + OpenAI shim + hardening (days 16–17):** remaining endpoints, metrics, auto-restart, 24h soak test.

## 10. Risks

- **Blackwell / vLLM compatibility.** M0 de-risks this. Fallback: nano-vllm-voxcpm's non-FA path and Fun-ASR's direct PyTorch path. ~30% latency regression, still within budget.
- **Fun-ASR rolling-buffer efficiency.** The re-decode-on-growing-buffer pattern works well at batch=1 but could contend with concurrent sessions since each session runs its own re-decode loop. Fun-ASR-vllm's batched inference should absorb this, but we'll measure actual GPU utilization with 8 concurrent sessions in M5 and fall back to VAD-only if needed.
- **TTS chunk-boundary clicks.** 10ms crossfade in M4.
- **Barge-in race conditions.** Cancellation token pattern, explicit integration tests in M5.
- **LoRA hot-swap with nano-vllm.** Read `example_lora.py` and `example_lora_sync.py` in M2 to de-risk.
- **Fine-tuning pauses STT on GPU 1.** Acceptable for v1; v1.1 schedules FT to off-hours.

## 11. Success metrics

- 5 of 7 team members have used the playground within a week of launch
- ≥ 3 voices created by non-engineers
- At least one BOMBA SR integration migrated from external TTS
- < 1% error rate on `/v1/tts` and `/v1/stt` over rolling 7-day window
- p50 conversational turn latency in production ≤ 800ms
- Fun-ASR WER on held-out internal test set matches published numbers within 1pp
