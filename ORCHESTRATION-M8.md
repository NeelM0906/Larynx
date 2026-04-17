# M8 Design — Batch TTS, OpenAI Shim Completeness, Hardening, 24h Soak

Status: **draft v1** · owner: Neel · scope: PRD §5.7, §5.9, §7, §10 + M8 prompt

This is v1's finishing milestone. Four parts — each lands as its own
logically-scoped commit sequence:

- **A — Batch TTS** — `/v1/batch` CRUD, Arq worker at low priority, artifact
  serving, 7-day retention
- **B — OpenAI shim** — `/v1/audio/speech`, SDK-verified end-to-end
- **C — Hardening** — Prometheus on every worker, structured `/ready`,
  supervisord auto-restart, graceful drain, cleanup cron, body/timeout limits
- **D — 24h soak** — `scripts/soak_test.py` + `SOAK_REPORT.md`

The state-machine and concurrency pieces (Part A's batch worker, Part C's
graceful-shutdown drain) are the load-bearing parts of the design; the rest
is plumbing.

---

## 0. Non-negotiable invariants

1. **Real-time TTS never starves.** Batch work runs on a separate Arq queue
   (`larynx:batch`) with its own worker and a low job-level priority. The
   existing in-process `voxcpm_client` serves WS streaming + `/v1/tts` on
   the gateway as today; batch items go through the *same* client but only
   after the batch worker acquires an in-process `asyncio.Semaphore`
   bounded to **2** concurrent batch items per gateway process. Real-time
   requests are never admission-controlled. The semaphore caps only
   batch-originating calls — real-time paths bypass it.
2. **No fakes in tests** (per memory `feedback_no_fakes_in_tests.md`). Batch
   tests hit real Redis + real Postgres + the mock VoxCPM worker the rest
   of the suite already uses. The end-to-end OpenAI-SDK test hits the real
   OpenAI Python SDK against our real gateway over loopback; transport is
   live HTTP, only the model layer is mocked.
3. **Commits are small and reviewable** (per memory
   `feedback_commit_cadence.md`). This milestone splits into ≥20 commits
   along the part boundaries below. Never one "M8" bundle.
4. **Backwards-compat with M7.** `/v1/tts` and `/v1/conversation` keep
   identical shapes. Adding `response_format="mp3"` to TTS is additive.
5. **Graceful shutdown must drain in-flight work.** A SIGTERM during a
   batch item waits ≤30s for the current item to finish; remaining queued
   items stay in Redis and are picked up by the restarted worker.

---

## 1. Part A — Batch TTS

### 1.1 Data model

New SQLAlchemy models + Alembic migration `0004_batch_jobs.py`:

```python
class BatchJob(Base):
    __tablename__ = "batch_jobs"
    id: Mapped[str]                        # uuid hex (matches fine_tune_jobs)
    state: Mapped[BatchJobState]           # QUEUED|RUNNING|COMPLETED|CANCELLED|FAILED
    num_items: Mapped[int]
    num_completed: Mapped[int]
    num_failed: Mapped[int]
    retain: Mapped[bool]                   # default False → 7-day cleanup
    error_code: Mapped[str | None]
    created_at, started_at, finished_at, expires_at: timezone-aware

class BatchItem(Base):
    __tablename__ = "batch_items"
    job_id: FK to batch_jobs.id (CASCADE)
    item_idx: Mapped[int]                  # 0..N-1, preserved from request
    state: Mapped[BatchItemState]          # QUEUED|RUNNING|DONE|FAILED|CANCELLED
    voice_id, text, params_json            # snapshot of the submitted item
    output_path: Mapped[str | None]        # populated on DONE; relative to DATA_DIR
    output_format, sample_rate, duration_ms, generation_time_ms
    error_code, error_detail: Mapped[str | None]
    started_at, finished_at
```

States are `typing.Literal` strings + `VARCHAR(32)` in the DB — same
convention as `JobState`/`VoiceSource`.

`expires_at` is set at job creation to `created_at + 7 days` unless
`retain=true`. The daily cron (§3.5) deletes rows where
`finished_at IS NOT NULL AND expires_at < now() AND retain = false`,
plus their on-disk outputs.

### 1.2 HTTP surface

All under `/v1/batch`, bearer-token gated.

| Method | Path                           | Body / Response                                       |
|--------|--------------------------------|-------------------------------------------------------|
| POST   | `/v1/batch`                    | `{items: [{text, voice_id?, params?}, ...], retain?}` → `{job_id}` |
| GET    | `/v1/batch/{id}`               | full status + per-item state + signed artifact URLs when DONE |
| DELETE | `/v1/batch/{id}`               | 202 — cancels QUEUED + RUNNING items; DONE items kept |
| GET    | `/v1/batch/{id}/items/{idx}`   | serves the wav file under bearer auth                 |

Status response shape:

```json
{
  "job_id": "…",
  "state": "RUNNING",
  "progress": 0.37,
  "num_items": 100, "num_completed": 37, "num_failed": 0,
  "created_at": "…", "started_at": "…",
  "items": [
    {"idx": 0, "state": "DONE", "url": "/v1/batch/{id}/items/0",
     "duration_ms": 4230, "voice_id": "librivox-female-clear"},
    {"idx": 1, "state": "RUNNING"},
    ...
  ]
}
```

`items[]` is capped at 500 entries in a single submission (Postgres insert
batch stays fast, Arq payloads stay under 1MB). Requests over 500 return
`400 items_too_many`.

`params` per item is the *subset* of `TTSRequest` fields: `sample_rate`,
`output_format`, `cfg_value`, `temperature`, `prompt_text`. No per-item
reference-audio uploads in batch — use an existing `voice_id`.

### 1.3 Queueing

Add two new dependencies to `larynx-gateway`: `arq>=0.25`.

New file `packages/gateway/src/larynx_gateway/services/batch_queue.py`:

```python
class BatchQueue:
    """Arq producer — enqueues one job per BatchItem under the
    ``larynx:batch`` queue name. Returns an Arq job_id used to trace."""
    async def enqueue_item(self, *, batch_job_id, item_idx): ...
    async def cancel_job(self, *, batch_job_id): ...   # deletes queued items
```

Queue layout:

- `larynx:batch` — Arq queue for batch items. `job_timeout=600`,
  `max_tries=2`, `queue_read_limit=2` per worker so a single batch can't
  saturate.
- `larynx:cron` — Arq queue for the daily cleanup cron (§3.5).

Two Arq workers run in-process inside the gateway container under
supervisord (§3.3):

```
[program:batch_worker]  — runs arq larynx_gateway.workers.batch_worker
[program:cron_worker]   — runs arq larynx_gateway.workers.cron_worker
```

The batch worker imports `VoxCPMClient` via a thin HTTP loopback — the
gateway exposes an internal `POST /internal/batch_synth` that the Arq
worker calls. This avoids duplicating model-loading in a second process
and keeps the batch worker stateless. The internal route requires a
shared-secret header (`X-Internal-Token`, settings-backed; random per
boot if unset) so external callers can't reach it.

*Why not call `VoxCPMClient` directly from the Arq worker?* The worker
runs in its own process; loading a second voxcpm manager would double the
GPU footprint. Treat the batch worker as a thin scheduler that fans item
jobs back into the gateway's real-time synthesis path, which is already
bounded by the semaphore in §0.

### 1.4 State machine

```
QUEUED ─lock=ok─▶ RUNNING ─all-items-terminal─▶ COMPLETED
  │ cancel                │ cancel                      │
  └─▶ CANCELLED           └─▶ CANCELLED                 │
                          │ any item failed + halt-rule ▼
                          └─▶ FAILED (when num_failed / num_items > 0.5)
```

Cancellation semantics:

- `DELETE /v1/batch/{id}` sets the job row to `CANCELLED`. An item in
  `QUEUED` flips to `CANCELLED` in the item-pickup path. An item in
  `RUNNING` runs to completion (no mid-synthesis interruption; cheaper
  than plumbing cancel tokens into the TTS path). Already `DONE` items
  are preserved — outputs remain accessible.
- `FAILED` is used only if every item failed to even start; partial
  failures report as `COMPLETED` with `num_failed > 0` so the client can
  retry specific indices without re-submitting a whole job.

Progress is computed server-side in GET: `num_completed / num_items`.
No separate progress store.

### 1.5 Artifact storage + serving

- Output path:
  `${DATA_DIR}/batch/{job_id}/{item_idx:05d}.{ext}` — 5-digit padded so
  lexical sort == numeric sort.
- Container: WAV by default; overridable per item via `params.output_format`.
- `GET /v1/batch/{id}/items/{idx}` does a single file stream with
  `Content-Type` matching the item's `output_format` and
  `Content-Disposition: attachment`. Bearer-token gated. Symlink-follow
  is off. `ETag` = sha256 of the file (cached in DB).
- `expires_at` enforcement: the cron job (§3.5) removes expired rows and
  files; GET on an expired id returns 410.

### 1.6 Test plan (Part A)

Real-Redis + real-Postgres + mock VoxCPM. In `tests/integration/`:

- `test_batch_create_and_run.py` — submit 10 items with 3 voice_ids,
  poll GET until `COMPLETED`, assert all 10 files exist and the GET
  serves bytes that start with `RIFF`.
- `test_batch_cancel.py` — submit 20 items, delete before queue drains,
  assert: cancelled items never produce files, already-DONE items are
  preserved, subsequent GET is idempotent.
- `test_batch_priority.py` — submit batch, then fire 5 parallel
  `/v1/tts` calls, assert their p95 < 200ms while the batch is in flight
  (uses the mock worker's synthetic 50ms-per-item latency).
- `test_batch_cleanup.py` — insert a BatchJob with `expires_at` in the
  past, run the cleanup cron directly, assert rows + files are gone.
- `test_batch_item_limits.py` — >500 items → 400.

### 1.7 Files touched / added (Part A)

```
packages/gateway/src/larynx_gateway/db/models.py                        [+]
packages/gateway/src/larynx_gateway/db/migrations/versions/0004_batch.py [+]
packages/gateway/src/larynx_gateway/schemas/batch.py                    [+]
packages/gateway/src/larynx_gateway/services/batch_queue.py             [+]
packages/gateway/src/larynx_gateway/services/batch_service.py           [+]   (create/list/cancel/status logic)
packages/gateway/src/larynx_gateway/services/batch_cleanup.py           [+]
packages/gateway/src/larynx_gateway/workers/batch_worker.py             [+]   (Arq WorkerSettings)
packages/gateway/src/larynx_gateway/workers/cron_worker.py              [+]   (Arq cron)
packages/gateway/src/larynx_gateway/routes/batch.py                     [+]
packages/gateway/src/larynx_gateway/routes/internal_batch.py            [+]   (loopback endpoint)
packages/gateway/src/larynx_gateway/main.py                              [~]   (include_router + lifespan startup for Arq pool)
packages/gateway/pyproject.toml                                          [~]   (add arq)
packages/gateway/tests/integration/test_batch_*.py                        [+]
supervisord.conf                                                         [~]
```

---

## 2. Part B — OpenAI shim completeness

### 2.1 `POST /v1/audio/speech`

Matches the OpenAI TTS API shape exactly:

```json
{
  "model": "tts-1",            // accepted, ignored — we pick VoxCPM2
  "input": "Hello world",
  "voice": "alloy",            // maps to voice_id in our library
  "response_format": "mp3",    // mp3|opus|aac|flac|wav|pcm — default mp3 (OpenAI default)
  "speed": 1.0                 // 0.25..4.0
}
```

Routing logic:

1. Look up `voice` by name in the `voices` table. If no match, accept the
   six OpenAI short-names (`alloy`, `echo`, `fable`, `onyx`, `nova`,
   `shimmer`) and map them to our three seed voices (round-robin: alloy
   + echo → librivox-male-baritone, fable + onyx → librivox-male-expressive,
   nova + shimmer → librivox-female-clear). This keeps us honest about
   the shim: clients that iterate OpenAI voices get consistent output.
2. Convert the request into a regular `TTSRequest` and re-use
   `tts_service.resolve_conditioning` + `synthesize`.
3. `speed` maps to engine knobs:
   - **speed > 1.0**: reduce `inference_timesteps` proportionally (min 6)
     and increase VoxCPM `chunk_stride` (worker knob) by ~(1/speed). The
     result is a slightly compressed synth that reads as faster pacing;
     exact prosody match with OpenAI isn't the goal.
   - **speed < 1.0**: increase `inference_timesteps` up to 16 and reduce
     `chunk_stride` by ~speed. Again, "reasonable approximation is fine."
   - **speed == 1.0**: pass straight through (default path).
   - The mapping table is documented in the route's docstring so we can
     tune without grep-surprise later.
4. Output encoding:
   - `mp3`, `aac`, `opus`, `flac`: encoded via an in-process transcoder
     (`larynx_shared.audio.encode`) backed by `pyav` — added as a
     gateway dep. Chosen over shelling out to `ffmpeg` because pyav
     stays inside asyncio (we already use it elsewhere for latent
     decoding patterns).
   - `wav`: re-uses existing `pack_wav`.
   - `pcm`: raw `audio/L16`, bit-identical to current `output_format=pcm16`.
   - Content-Type mapping: `mp3→audio/mpeg`, `aac→audio/aac`,
     `opus→audio/ogg`, `flac→audio/flac`, `wav→audio/wav`, `pcm→audio/L16`.
5. On encode failure or unsupported runtime codec (e.g. missing system
   libs), return 500 with `{"error": {"type": "server_error", "code":
   "codec_unavailable"}}` — OpenAI error shape.

### 2.2 Voice pre-seeding — six real, distinct voices

`scripts/load_demo_voices.py` is extended so that after a run the voice
library always contains six voices whose **names** match the OpenAI
preset short-names exactly:

- `alloy`   — rename of `librivox-male-baritone` (upload path, seeded from LibriVox)
- `echo`    — rename of `librivox-male-expressive` (upload path, seeded from LibriVox)
- `nova`    — rename of `librivox-female-clear` (upload path, seeded from LibriVox)
- `fable`   — newly **designed** via `POST /v1/voices/design`
  prompt: "warm mid-range male storyteller, unhurried pace, British English"
- `onyx`    — newly **designed**
  prompt: "deep resonant male, measured and authoritative, American English"
- `shimmer` — newly **designed**
  prompt: "soft intimate female, breathy but clear, American English"

The seed script is idempotent — it checks `GET /v1/voices` for each name
and only creates what's missing, so re-runs on an existing library are
free. Designed voices are persisted with `source='designed'` (already a
supported VoiceSource value from M2).

With distinct rows under each of the six names, `/v1/audio/speech`
routing becomes a plain name-lookup against the `voices` table — no
alias dict, no fallback. Requests for any other voice name return 404.

### 2.3 SDK verification test

New test: `tests/integration/test_openai_sdk_speech.py`:

```python
import pytest
from openai import OpenAI

@pytest.mark.openai_sdk
def test_openai_sdk_speech_roundtrip(live_gateway_url, live_token, tmp_path):
    client = OpenAI(base_url=f"{live_gateway_url}/v1", api_key=live_token)
    resp = client.audio.speech.create(
        model="tts-1", voice="alloy", input="Hello from the shim."
    )
    resp.stream_to_file(tmp_path / "out.mp3")
    assert (tmp_path / "out.mp3").stat().st_size > 1024
    # header check: bytes start with 0xFFFB / 0xFFF3 / 0xFFF2 / 0x4944
    header = (tmp_path / "out.mp3").read_bytes()[:4]
    assert header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") or header[:3] == b"ID3"
```

Live gateway fixture: the existing `client` fixture already boots the
app via ASGI transport; we extend it with a `live_gateway_url` variant
that runs the app under uvicorn on a random port via `pytest-httpserver`
or (simpler) `asyncio.start_server` around the ASGI app. The SDK insists
on a real HTTP base_url — we can't pass the ASGI transport to OpenAI's
client directly.

Also add: `test_openai_sdk_speech_formats.py` — parametrised over
`mp3`, `wav`, `flac`, `opus` (skip `aac` if the pyav build lacks the
encoder — feature-detect at test collection time).

`openai>=1.40` goes into the dev dependency group (not runtime).

### 2.4 Files touched / added (Part B)

```
packages/gateway/src/larynx_gateway/routes/openai_compat.py              [~]  (+ speech endpoint)
packages/gateway/src/larynx_gateway/schemas/openai.py                    [+]  (OpenAISpeechRequest etc.)
packages/gateway/src/larynx_gateway/services/openai_voice_alias.py       [+]
packages/shared/src/larynx_shared/audio/encode.py                         [+]  (pyav encoders)
packages/shared/pyproject.toml                                            [~]  (pyav)
pyproject.toml                                                            [~]  (openai in dev)
packages/gateway/tests/integration/test_openai_sdk_speech.py              [+]
packages/gateway/tests/integration/test_openai_sdk_speech_formats.py      [+]
```

---

## 3. Part C — Hardening

### 3.1 Prometheus coverage

Every long-running process exposes its own `/metrics`:

| Process           | Endpoint                    | Notes                                     |
|-------------------|-----------------------------|-------------------------------------------|
| gateway           | `GET /metrics` (existing)   | re-uses current Histograms + adds more    |
| funasr_worker     | `GET :9101/metrics`         | tiny aiohttp sidecar in the worker main() |
| vad_punc_worker   | `GET :9102/metrics`         | same                                      |
| voxcpm_worker     | `GET :9103/metrics`         | same (when/if split to its own container) |
| training_worker   | `GET :9104/metrics`         | same                                      |

Metrics conventions — every endpoint emits:

- `larynx_request_duration_seconds` (Histogram, labels `endpoint`,
  `status_class`) — covers every HTTP + WS route. Wired via a thin
  `metrics_middleware` on the FastAPI app.
- `larynx_error_total` (Counter, labels `endpoint`, `error_code`) —
  incremented from a shared `record_error()` helper the existing
  `HTTPException` handlers call.
- `larynx_ws_connections` (Gauge, labels `kind` ∈ {tts, stt,
  conversation}) — set from the WS route's connect/disconnect branches.
- `larynx_queue_depth` (Gauge, labels `queue` ∈ {batch, cron}) —
  sampled every 5s by a lifespan-started task via `redis.xlen`.
- Per-worker specific:
  - voxcpm: `larynx_tts_generation_seconds` (Histogram),
    `larynx_tts_lora_active` (Gauge, value = # loaded LoRAs)
  - funasr: `larynx_stt_rtfx` (Histogram),
    `larynx_stt_language_used_total` (Counter, label `model`)
  - training: `larynx_training_step_duration_seconds` (Histogram)

`/metrics` on the gateway already exists for existing counters — extend
it (not re-implement). Add a *proxy* route `/metrics/workers` that
concatenates the per-worker scrapes (using `httpx` against the sidecar
URLs) and returns the union. This is the "gateway proxying all worker
metrics" requirement from the prompt.

### 3.2 Structured `/ready`

Replace the current `/ready` with a worker-aware implementation:

```json
200 OK or 503 Service Unavailable
{
  "status": "ready"|"degraded"|"starting",
  "workers": {
    "voxcpm":   {"state": "ready", "last_heartbeat_s_ago": 1.2},
    "funasr":   {"state": "ready", "last_heartbeat_s_ago": 0.8},
    "vad_punc": {"state": "ready", "last_heartbeat_s_ago": 0.3},
    "training": {"state": "idle",  "last_heartbeat_s_ago": 2.1}
  },
  "queues": {"batch": 12, "cron": 0},
  "version": "0.8.0"
}
```

Heartbeat mechanism: each worker client (`VoxCPMClient`, `FunASRClient`,
`VadPuncClient`) gains a `last_heartbeat_at: float` field updated on
every successful response. A lifespan-started task pings each worker via
a cheap `health_ping` IPC message every 5s to keep the timestamp fresh
when idle.

- `state=ready` iff `last_heartbeat_s_ago < 15`
- `state=degraded` iff `15 ≤ last_heartbeat_s_ago < 60`
- `state=down` iff `last_heartbeat_s_ago ≥ 60`
- `status=ready` iff every expected worker is `ready`; otherwise 503.
- `training` is special — if no job is running, `state=idle` and the
  overall status is still `ready`.

### 3.3 supervisord + restart-backoff

Rewrite `supervisord.conf`:

- Every program gets `autorestart=true`, `startsecs=5`, `startretries=3`,
  `backoff=10` (supervisord uses fixed backoff; we mimic exponential by
  starting with `startretries=3` inside 60s then a longer `autorestart`
  pause via a small sidecar). After 3 consecutive restart failures in
  60s the supervisor writes an `ALERT` line to stderr that the soak
  test can pick up (and a future pagerduty hook can watch).
- New programs: `batch_worker`, `cron_worker`, plus the per-worker
  sidecar `metrics_server` where the worker isn't already the gateway.

Because supervisord 4.x supports exponential backoff via its
`eventlistener:` mechanism, we add a **8-line Python listener**
`packages/gateway/src/larynx_gateway/ops/restart_alerter.py` that reads
PROCESS_STATE events and logs a single `critical.restart_storm` line
after 3 failures in 60s. That log line is what the soak test watches.

### 3.4 Log rotation

Two options — we pick the docker-friendly one:

- Docker already writes logs to stdout. Docker's `json-file` log driver
  supports rotation via `max-size` / `max-file` at the container level.
  Add these options to `docker-compose.yml` for every service.
- For bare-metal runs (the host uses systemd), document an
  `/etc/logrotate.d/larynx` snippet in `docs/deployment.md` that rotates
  `/var/log/larynx/*.log` daily, keeps 14, `copytruncate`.

### 3.5 Generated-audio cleanup cron

New Arq cron (runs once daily at 03:00 UTC, owned by `cron_worker`):

- Deletes `BatchJob` rows (and cascades items + files) where
  `expires_at < now()` and `retain = False`.
- Deletes bare files under `${DATA_DIR}/batch/` whose parent job_id
  isn't in the DB (orphan sweep).
- Deletes `${DATA_DIR}/generated/` files (single-shot TTS outputs, if
  they exist — the current code doesn't persist them; future-proofing).

### 3.6 Body-size limits + timeouts

Global body-size caps enforced in a FastAPI middleware:

| Route                    | Body limit |
|--------------------------|------------|
| `/v1/tts` JSON           | 64 KB      |
| `/v1/tts` multipart      | 50 MB      |
| `/v1/stt`                | 100 MB     |
| `/v1/voices` audio       | 100 MB     |
| `/v1/batch` POST         | 2 MB       |
| `/v1/audio/speech`       | 64 KB      |
| `/v1/audio/transcriptions` | 100 MB   |
| `/v1/finetune/datasets`  | 500 MB     |
| default                  | 1 MB       |

All outbound HTTP calls (OpenRouter, internal loopback) pass
`timeout=httpx.Timeout(connect=5, read=30, write=5, pool=5)`. The LLM
streaming call uses `read=60`. No bare `.get()`/`.post()` without an
explicit timeout anywhere.

### 3.7 Graceful shutdown

On SIGTERM:

1. FastAPI lifespan teardown starts.
2. Gateway flips a process-wide `app.state.shutting_down=True` flag;
   `/ready` starts returning 503 immediately so upstream LBs drain.
3. WS handlers observe the flag, stop accepting new frames, and send a
   close frame after the current turn completes.
4. Batch worker finishes the current item (no new pickups). The `_run`
   coroutine for a running item is left uninterrupted for ≤30s.
5. After 30s, any still-running tasks are cancelled; partial progress
   on batch items leaves them as `QUEUED` (the state never transitioned
   to `RUNNING` unless the worker successfully started) or, if
   transitioned, they're left `RUNNING` with `error_code='drained'` —
   the cron (§3.5) can re-queue these on next boot if we want, but for
   v1 they're simply surfaced to the client on GET.

Implementation: a `shutdown_event = asyncio.Event()` on `app.state`,
checked in the batch worker's loop; the 30s cap is enforced by a
`shield + wait_for` pattern in the lifespan teardown.

### 3.8 Files touched / added (Part C)

```
packages/gateway/src/larynx_gateway/middleware/metrics.py               [+]
packages/gateway/src/larynx_gateway/middleware/body_limits.py           [+]
packages/gateway/src/larynx_gateway/routes/metrics.py                   [~]  (existing; add worker-proxy)
packages/gateway/src/larynx_gateway/routes/health.py                    [~]  (structured /ready)
packages/gateway/src/larynx_gateway/services/worker_health.py           [+]
packages/gateway/src/larynx_gateway/ops/restart_alerter.py              [+]
packages/<each>_worker/src/larynx_<each>_worker/metrics_server.py       [+]
packages/<each>_worker/src/larynx_<each>_worker/main.py                 [~]  (add metrics sidecar)
supervisord.conf                                                        [~]
docker-compose.yml                                                      [~]
docs/deployment.md                                                      [+]  (log rotation)
```

---

## 4. Part D — 24h soak

### 4.1 `scripts/soak_test.py`

CLI invocation: `uv run python scripts/soak_test.py --gateway-url http://localhost:8000 --token $LARYNX_API_TOKEN --duration 24h --out soak-artifacts/`.

Traffic mix (ints per minute):

- 1 conversation: opens `/v1/conversation` WS, plays 3 turns of canned
  PCM, listens for audio back, closes. Voice rotates through all
  voices in the library (3 seed + any added).
- 10 single-shot TTS: `/v1/tts` with random text (30–400 char,
  pulled from a corpus file) and random voice_id.
- 5 single-shot STT: `/v1/stt` with a random clip pulled from
  `tests/fixtures/audio/*.wav` (existing fixtures).
- Every ~10 min: submit a 20-item batch job. Cancel ~10% of them
  mid-flight.

Implementation: one asyncio task per stream type, each running a
rate-limited loop. Failures are logged to `soak-artifacts/errors.jsonl`
(line-per-failure JSON) but don't stop the run.

Per-minute, the script samples:

- `psutil` process RSS for every larynx-* process on the host (gateway,
  workers, supervisord-owned children)
- `nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu`
  per GPU
- `/metrics` histograms (Prometheus text parse for selected series:
  request duration p50/p95/p99 per endpoint, error totals, queue depth,
  WS connections)
- `df -B1 ${DATA_DIR}` for disk footprint

Samples go to `soak-artifacts/timeseries.parquet` via pyarrow — one row
per minute per metric. Written append-only; if the script dies, the
parquet is still recoverable up to the last flush (flush interval: 60s).

### 4.2 `SOAK_REPORT.md`

Generated at the end by `soak_test.py` from the parquet:

- Total requests per endpoint + error rate
- p50/p95/p99 latencies per stage (TTS TTFB, STT finalization,
  conversation turn, batch-item generation)
- Memory growth: linear regression slope on RSS over 24h, per process.
  **If > 5% in 24h, fail the run and the report says so.**
- VRAM footprint: min, max, mean per GPU
- CPU utilisation: mean, 95th per core
- Disk delta: `df` at t=0 vs t=24h
- Any restarts logged by the `restart_alerter` eventlistener
- Qualitative: sample 10 random batch outputs + 10 random
  single-shot TTS outputs, report MOS-lite (RMS + spectral centroid)
  over time. If spectral centroid drifts > 5% across the run, call it
  quality degradation.
- Conclusion: one of **pass** / **fail + patch list**.

### 4.3 Files added (Part D)

```
scripts/soak_test.py
scripts/soak_utils/corpus.txt              # 1000 random English sentences
scripts/soak_utils/__init__.py
SOAK_REPORT.md                             # generated; committed after the run
```

---

## 5. Cross-cutting concerns

### 5.1 Exit criteria mapping

| Exit criterion                                          | Where it's tested                                                |
|---------------------------------------------------------|------------------------------------------------------------------|
| 24h soak passes, <5% mem growth, zero stuck workers     | §4.1 + §4.2; the soak script's pass/fail gate                    |
| OpenAI Python SDK works unchanged                       | §2.3 `test_openai_sdk_speech_roundtrip`                          |
| Grafana/`/metrics` shows per-stage latencies in budget  | §4.1 samples Prometheus; report tabulates p50 vs PRD §4 budget   |
| 100-item job with 10 voices succeeds                    | §1.6 `test_batch_create_and_run` extended to 100 items/10 voices |

### 5.2 Commit sequence

Roughly 20 commits across the four parts. Order:

1. `feat(gateway): alembic 0004 — batch jobs + items`
2. `feat(gateway): batch schemas + service stubs`
3. `feat(gateway): /v1/batch CRUD routes (mock worker)`
4. `feat(gateway): batch Arq worker + loopback synth route`
5. `feat(gateway): batch item artifact serving + ETag`
6. `feat(gateway): batch cancel semantics + tests`
7. `test(gateway): batch real-time starvation regression`
8. `feat(gateway): batch cleanup cron + Arq cron_worker`
9. `feat(gateway): OpenAI /v1/audio/speech route + schema`
10. `feat(shared): pyav encoders for mp3/aac/opus/flac`
11. `feat(gateway): OpenAI voice aliases + speed→engine-knob mapping`
12. `test(gateway): OpenAI SDK round-trip + format matrix`
13. `feat(workers): metrics sidecar servers`
14. `feat(gateway): metrics middleware + /metrics/workers proxy`
15. `feat(gateway): worker heartbeat + structured /ready`
16. `feat(ops): supervisord restart alerter`
17. `feat(gateway): body-size + timeout middleware`
18. `feat(gateway): graceful shutdown drain`
19. `feat(ops): docker-compose log rotation + logrotate doc`
20. `feat(scripts): 24h soak harness`
21. `docs: SOAK_REPORT from first clean 24h run`

Each commit keeps tests green. Sequence 1–7 unlocks §1, 8 unlocks the
cleanup, 9–12 unlock §2, 13–19 unlock §3, 20–21 unlock §4. If soak
fails in commit 21, subsequent patch commits follow before v1 declare.

### 5.3 Open questions (decisions made, called out)

1. **OpenAI voice naming.** Resolved: six real distinct voices (3
   renamed LibriVox seeds + 3 newly designed), see §2.2. No alias layer.
2. **Batch worker as HTTP loopback vs direct client.** Went with
   loopback. Avoids doubling GPU footprint. Trade-off: one extra
   serialization hop per batch item (≤1ms).
3. **pyav as transcoder.** Single additional C dep. Alternative was
   shell-out to ffmpeg; rejected because subprocess per request adds
   ~30ms and complicates shutdown drain. pyav decision committed
   unless Neel wants ffmpeg for ecosystem reasons.
4. **Shutdown drain caps at 30s.** Matches the prompt. Anything longer
   and supervisord kills us anyway (default STOPWAITSECS=10; we raise
   to 35 on the programs that need drain).
5. **Scope deliberately excluded from M8.** Rate limiting, per-user
   quotas, semantic turn detection, phone integration, multi-tenant
   auth, Grafana dashboard files. All v1.1+.

---

## 6. Rollback / safety

- Migration 0004 is additive-only (new tables). Downgrade drops them;
  no data loss for anything outside M8.
- Feature-flag: `LARYNX_BATCH_ENABLED` (default True). If the batch
  worker OOMs in production, flip to False — gateway refuses new batch
  submissions (`503 batch_disabled`) but existing single-shot paths are
  untouched.
- pyav is runtime-optional. If it fails to import, `/v1/audio/speech`
  with `response_format ∈ {mp3, aac, opus, flac}` returns 501; wav/pcm
  still work. The dep is marked required in pyproject but the import is
  wrapped so a broken install in staging doesn't take down the whole
  gateway.
