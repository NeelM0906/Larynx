# Repo Structure

Monorepo. Python packages + one Next.js app. `uv` workspaces. Root `docker-compose.yml` runs everything for dev. The structure is designed so LLM gateway and DB layers slot in later as sibling packages without refactoring.

```
larynx/                                    # project root
├── README.md
├── PRD.md
├── docker-compose.yml                     # postgres, redis, gateway, workers
├── docker-compose.dev.yml                 # dev overrides (hot-reload, exposed ports)
├── pyproject.toml                         # uv workspace root
├── uv.lock
├── .env.example
├── .gitignore
├── Makefile                               # up, down, test, lint, migrate, smoke
├── supervisord.conf                       # process supervision for workers
│
├── packages/
│   │
│   ├── gateway/                           # FastAPI app (HTTP + WS)
│   │   ├── pyproject.toml
│   │   ├── src/larynx_gateway/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                    # FastAPI app factory
│   │   │   ├── config.py                  # pydantic-settings, env vars
│   │   │   ├── auth.py                    # bearer token dep
│   │   │   ├── deps.py                    # shared DI (db, redis, workers)
│   │   │   ├── logging.py                 # structlog setup
│   │   │   │
│   │   │   ├── routes/
│   │   │   │   ├── tts.py                 # POST /v1/tts
│   │   │   │   ├── tts_stream.py          # WS /v1/tts/stream
│   │   │   │   ├── stt.py                 # POST /v1/stt
│   │   │   │   ├── stt_stream.py          # WS /v1/stt/stream
│   │   │   │   ├── voices.py              # /v1/voices CRUD + design
│   │   │   │   ├── conversation.py        # WS /v1/conversation
│   │   │   │   ├── batch.py               # /v1/batch
│   │   │   │   ├── finetune.py            # /v1/finetune
│   │   │   │   ├── openai_compat.py       # /v1/audio/speech, /v1/audio/transcriptions
│   │   │   │   └── health.py              # /health, /ready
│   │   │   │
│   │   │   ├── services/                  # business logic
│   │   │   │   ├── voice_library.py       # upload/list/delete + latent caching
│   │   │   │   ├── tts_service.py         # single-shot + streaming synthesis
│   │   │   │   ├── stt_service.py         # single-shot + streaming transcription
│   │   │   │   ├── language_router.py     # picks Fun-ASR-Nano vs MLT
│   │   │   │   ├── conversation_service.py# VAD+STT+LLM+TTS orchestration + barge-in
│   │   │   │   ├── llm_client.py          # OpenRouter streaming client
│   │   │   │   ├── latent_cache.py        # Redis + disk latent cache
│   │   │   │   └── batch_service.py       # Arq task producer
│   │   │   │
│   │   │   ├── workers_client/            # clients that talk to model worker processes
│   │   │   │   ├── base.py                # abstract async worker client
│   │   │   │   ├── voxcpm_client.py       # TTS worker client
│   │   │   │   ├── funasr_client.py       # STT worker client (routes to Nano or MLT)
│   │   │   │   └── vad_punc_client.py     # VAD + punctuation client (in-process, CPU)
│   │   │   │
│   │   │   ├── db/
│   │   │   │   ├── models.py              # SQLAlchemy: Voice, BatchJob, FineTuneJob
│   │   │   │   ├── session.py
│   │   │   │   └── migrations/            # alembic
│   │   │   │
│   │   │   └── schemas/                   # pydantic request/response models
│   │   │       ├── tts.py
│   │   │       ├── stt.py
│   │   │       ├── voice.py
│   │   │       ├── conversation.py
│   │   │       └── finetune.py
│   │   │
│   │   └── tests/
│   │       ├── unit/
│   │       ├── integration/               # spin up gateway against fake workers
│   │       └── fixtures/                  # sample audio, sample texts
│   │
│   ├── voxcpm_worker/                     # TTS worker (GPU 0)
│   │   ├── pyproject.toml
│   │   ├── src/larynx_voxcpm_worker/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                    # entrypoint, holds VoxCPM2 in VRAM
│   │   │   ├── server.py                  # async IPC server
│   │   │   ├── model_manager.py           # loads nano-vllm-voxcpm, handles LoRA hot-swap
│   │   │   ├── audio_utils.py             # encoding, crossfade, format conversion
│   │   │   └── latent_encoder.py          # on-upload audio→latent encoding
│   │   └── tests/
│   │
│   ├── funasr_worker/                     # STT worker (GPU 1)
│   │   ├── pyproject.toml
│   │   ├── src/larynx_funasr_worker/
│   │   │   ├── main.py                    # loads Fun-ASR-Nano + MLT via Fun-ASR-vllm
│   │   │   ├── server.py
│   │   │   ├── model_manager.py           # dual-model loader, language-aware dispatch
│   │   │   ├── audio_utils.py             # resampling, format conversion
│   │   │   └── hotword_helper.py          # hotword list handling
│   │   └── tests/
│   │
│   ├── vad_punc_worker/                   # CPU worker (VAD + punctuation)
│   │   ├── pyproject.toml
│   │   ├── src/larynx_vad_punc_worker/
│   │   │   ├── main.py
│   │   │   ├── vad.py                     # fsmn-vad wrapper
│   │   │   ├── punctuation.py             # CT-Transformer wrapper
│   │   │   └── utterance_segmenter.py     # combines VAD events into utterance boundaries
│   │   └── tests/
│   │
│   ├── training_worker/                   # LoRA fine-tuning worker (on-demand)
│   │   ├── pyproject.toml
│   │   ├── src/larynx_training_worker/
│   │   │   ├── main.py                    # Arq worker
│   │   │   ├── jobs.py                    # train_lora_job, register_lora_job
│   │   │   ├── dataset_prep.py            # validation, auto-transcription via Fun-ASR
│   │   │   └── config_builder.py          # generates VoxCPM LoRA config from UI inputs
│   │   └── tests/
│   │
│   └── shared/                            # types + utilities shared across packages
│       ├── pyproject.toml
│       ├── src/larynx_shared/
│       │   ├── ipc/                       # IPC primitives (currently in-process asyncio queue)
│       │   │   ├── messages.py            # typed request/response messages
│       │   │   └── client_base.py
│       │   ├── audio/                     # audio format helpers
│       │   │   ├── pcm.py
│       │   │   ├── wav.py
│       │   │   └── resample.py
│       │   └── tracing/                   # latency instrumentation helpers
│       └── tests/
│
├── apps/
│   └── playground/                        # Next.js playground UI
│       ├── package.json
│       ├── next.config.js
│       ├── tsconfig.json
│       ├── app/
│       │   ├── page.tsx                   # landing / nav
│       │   ├── tts/page.tsx
│       │   ├── clone/page.tsx
│       │   ├── design/page.tsx
│       │   ├── library/page.tsx
│       │   ├── conversation/page.tsx
│       │   ├── transcribe/page.tsx
│       │   └── finetune/page.tsx
│       ├── components/
│       │   ├── audio-recorder.tsx         # mic + WebAudio
│       │   ├── audio-player.tsx
│       │   ├── voice-picker.tsx
│       │   ├── voice-card.tsx
│       │   ├── conversation-transport.tsx # WS client for /v1/conversation
│       │   └── ui/                        # shadcn components
│       └── lib/
│           ├── api-client.ts
│           └── ws-client.ts
│
├── scripts/
│   ├── smoke_test.py                      # M0: one TTS + one STT end-to-end
│   ├── migrate.sh                         # alembic upgrade head
│   ├── load_demo_voices.py                # seed 3-5 example voices
│   └── soak_test.py                       # 24h reliability test
│
├── docker/
│   ├── gateway.Dockerfile
│   ├── voxcpm_worker.Dockerfile
│   ├── funasr_worker.Dockerfile
│   ├── vad_punc_worker.Dockerfile
│   ├── training_worker.Dockerfile
│   └── playground.Dockerfile
│
└── docs/
    ├── architecture.md                    # expanded architecture diagrams
    ├── deployment.md                      # how to deploy on the box
    ├── api.md                             # generated from OpenAPI + hand-written WS docs
    ├── adding_a_worker.md                 # how to add new model workers later
    └── runbook.md                         # what to do when things break
```

## Design notes

**Why separate worker processes?**
Three reasons: (1) API can restart without unloading 10+GB of GPU weights, (2) each worker can be replaced/upgraded independently, (3) the IPC interface is the same abstraction we'll use when the system eventually spans multiple boxes.

**Why `shared/` package?**
IPC message types, audio helpers, and tracing utilities are needed by every package. Making them a sibling package instead of importing from `gateway/` avoids circular dependency risk and makes the worker packages truly standalone.

**Why `apps/playground/` next to `packages/`?**
Clear separation: Python packages in `packages/`, deployable apps in `apps/`. The LLM gateway and DB connector services will go next to `gateway/` under `packages/` when they're built.

**Why Arq (not Celery, RQ, or similar)?**
Native asyncio, minimal config, small surface area. Fits the rest of the stack. Celery is overkill for a single-box deployment.

**Why supervisord on top of docker-compose?**
Docker handles containers; supervisord handles auto-restart of Python workers within the worker containers when they OOM or crash. This is standard practice for long-running GPU processes.

**Future extension points:**
- `packages/llm_gateway/` — when we unify LLM routing (OpenRouter + local models + BOMBA SR)
- `packages/db_connector/` — when we expose DB access through the platform
- `packages/auth/` — when we move beyond single bearer token
- `packages/metrics/` — when Prometheus/Grafana becomes its own service
