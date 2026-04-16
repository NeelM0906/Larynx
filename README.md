# Larynx

Self-hosted voice AI platform (VoxCPM2 TTS + Fun-ASR STT) — single-box,
GPU-resident, REST + WebSocket.

Authoritative docs:
- [PRD.md](./PRD.md) — product spec, goals, milestones
- [REPO_STRUCTURE.md](./REPO_STRUCTURE.md) — monorepo layout and rationale

## Quickstart (M1)

```bash
git clone <repo> larynx && cd larynx
cp .env.example .env                      # mock TTS by default — no GPU needed
make up                                   # postgres + redis on :5433 / :6380
make migrate                              # create Voice table
make smoke                                # POST /v1/tts -> /tmp/larynx_smoke.wav
```

That's it. `make smoke` launches the gateway, posts a test phrase, saves the
returned WAV, and validates its header.

To use the real TTS model, set `LARYNX_TTS_MODE=voxcpm` in `.env` on a box with
GPU 0 available and `nano-vllm-voxcpm` installed.

## Milestone status

Currently at **M1 — Foundation**. What works today:
- Monorepo + uv workspaces
- Postgres 16 / Redis 7 via docker-compose (non-default ports)
- Bearer-auth gateway with `/health`, `/ready`, `POST /v1/tts`
- VoxCPM2 worker runs in-process with the gateway (asyncio.Queue IPC)
- Mock TTS mode for GPU-less scaffold verification
- Alembic migration for the `voices` table (populated in M2)

Not yet implemented — see PRD §9: voice library (M2), STT (M3), streaming
(M4), conversational loop (M5), playground (M6), fine-tuning (M7), batch +
OpenAI shim (M8).
