# Fine-tuning Orchestration — M7 Design (v1)

Scope: end-to-end LoRA fine-tuning for VoxCPM2. Covers the new
`training_worker`, the `/v1/finetune/*` gateway surface, LoRA hot-swap in
`voxcpm_worker`, schema + migration for LoRA-backed voices, and the
`/finetune` page in the playground. Target exit criterion is the one in the
M7 prompt: upload audio + transcripts, kick off training, watch progress,
synthesise with the resulting voice at `/v1/tts`, and have it clearly match
the training speaker on held-out text.

A training job has exactly one owner at a time — this is a single-GPU
platform and fine-tuning monopolises GPU 0 for the life of the job. The
state machine and cancellation semantics are the load-bearing parts of the
design; everything else serves them.

---

## 0. Third-party reference provenance

Two upstream repos have been cloned into `./third_party/` (gitignored) for
design reference only:

- `third_party/VoxCPM/` — `openbmb/VoxCPM` @ tag `2.0.2`. Authoritative for
  the training script (`scripts/train_voxcpm_finetune.py`), LoRA YAML
  templates (`conf/voxcpm_v2/voxcpm_finetune_{lora,all}.yaml`), dataset
  loader (`src/voxcpm/training/data.py`), and Gradio reference UI
  (`lora_ft_webui.py`).
- `third_party/nanovllm-voxcpm/` — `a710128/nanovllm-voxcpm` @ default
  branch. Authoritative for LoRA hot-swap public API
  (`nanovllm_voxcpm/models/voxcpm2/server.py`), checkpoint format
  (`nanovllm_voxcpm/models/voxcpm2/lora_loader.py`), and lifecycle
  semantics (`docs/adr/0002-lora-lifecycle-and-residency.md`).

**Rule**: our code never imports from `third_party/`. When we need to
reproduce upstream behaviour, we copy the logic into our package with a
comment citing the source file and commit/tag we copied from. Updating the
reference is a `cd third_party/<repo> && git pull` — no build impact.

---

## 1. Per-job state machine

Exactly one `FineTuneJob` row per submission. One `TrainingJobRunner` owns
the lifecycle and holds a single `_state: JobState` enum. State
transitions happen in one place — the job's `_run` coroutine in
`training_worker/jobs.py`. The job is never concurrent with another
training job because the runner acquires a process-wide `GPU_TRAIN_LOCK`
before entering `PREPARING`; a second submission observed while the lock
is held goes to `QUEUED` and waits.

```
              ┌──────────────────────────────────────────┐
              │  QUEUED                                  │
              │  (Row persisted, waiting on              │
              │   GPU_TRAIN_LOCK; cancellable)           │
              └─────────────────┬────────────────────────┘
                                │ lock acquired
                                ▼
              ┌──────────────────────────────────────────┐
              │  PREPARING                               │
              │  dataset validation + auto-transcription │
              │  (calls funasr_worker); LoRA config YAML │
              │  written to {job_dir}/train_config.yaml  │
              └─────────────────┬────────────────────────┘
                                │ config on disk, dataset vetted
                                ▼
              ┌──────────────────────────────────────────┐
              │  TRAINING                                │
              │  subprocess: train_voxcpm_finetune.py    │
              │  --config_path=...; stdout/stderr tailed │
              │  into Redis stream logs:training:{job_id}│
              │  with 24h TTL                            │
              └─────────────────┬────────────────────────┘
                                │ exit 0 AND latest/lora_weights.safetensors exists
                                ▼
              ┌──────────────────────────────────────────┐
              │  REGISTERING                             │
              │  copy LoRA checkpoint → {lora_weights}/  │
              │  {voice_id}/; INSERT Voice row with      │
              │  source='lora'; voxcpm_client.load_lora  │
              └─────────────────┬────────────────────────┘
                                │ voxcpm_worker confirms register_lora
                                ▼
                              SUCCEEDED


  Terminal edges (from any non-terminal state):
    cancel  → CANCELLED   (user called DELETE /v1/finetune/jobs/{id})
    error   → FAILED      (exception, non-zero exit, validation failure,
                           missing artifact, hot-load rejected)
```

A job is **terminal** in `SUCCEEDED`, `FAILED`, or `CANCELLED`. No retries
in v1 — the user resubmits with adjusted parameters. Retries become a
v1.5 concern once we have log-driven insight into which failures are
transient.

### 1.1 Cancellation semantics

One `asyncio.Event` per job called `job_cancel`. Every state-specific
handler awaits on this event in parallel with its primary future via
`asyncio.wait(..., return_when=FIRST_COMPLETED)`. Propagation:

- `QUEUED`: cancel sets the job straight to `CANCELLED`. No resources
  held.
- `PREPARING`: interrupts whatever validation call is in flight. Datasets
  on disk remain (users may reuse them); the provisional job dir is
  removed.
- `TRAINING`: send **SIGTERM** to the subprocess. The upstream training
  script installs a SIGTERM handler that flushes a checkpoint, then
  exits — we wait up to `TRAIN_CANCEL_GRACE_SECONDS` (default **30s**;
  sized for a single checkpoint-save pass plus slack on disk) and then
  escalate to SIGKILL. Either way the job transitions to `CANCELLED`
  and the job dir is marked `partial/` (kept for debugging — cleanup
  job deletes entries older than 7 days).
- `REGISTERING`: cancellation behaviour splits at the `load_lora` boundary.
  - **Before** `load_lora` is called: cancel wins. No Voice row is
    written. Job transitions to `CANCELLED`; the on-disk checkpoint is
    left intact under `{lora_weights}/_orphaned/{ts}/` so the user can
    recover by resubmitting without retraining.
  - **While** `load_lora` is in flight: we cannot interrupt the engine
    mid-register safely, so the cancel waits. This window is
    sub-second.
  - **After** `load_lora` returns success: too late to cancel. We
    write the Voice row and transition to `SUCCEEDED`. The API
    returns `202 accepted` on the cancel but the terminal state is
    `SUCCEEDED` — the resulting voice is surfaced in `/v1/voices`.

**Load-bearing invariant**: `CANCELLED` never leaves a Voice row behind.
`SUCCEEDED` always has one. There is no third state — the runner's
final step (before writing the Voice row) re-checks `job_cancel`; if
set and `load_lora` has not yet been called, the row is never written.
If `load_lora` has already committed, the row is always written.

### 1.2 Progress reporting

Progress has two flavours:

- **Coarse state** (`_state` + percent estimate + ETA) — polled via
  `GET /v1/finetune/jobs/{id}`. Percent is computed from
  `current_step / max_steps` scraped out of training stdout (see §6.3).
  ETA is `(max_steps - current_step) * mean_step_ms` using a 20-step
  EWMA on the observed step cadence.
- **Log stream** — `GET /v1/finetune/jobs/{id}/logs` returns an SSE
  stream of each line the subprocess emits, plus periodic structured
  events (`{"type":"step","step":N,"loss":X}` parsed out of the
  training tracker lines). Client reconnects resume from the
  `Last-Event-ID` header — which we back with a Redis Stream at
  `logs:training:{job_id}` (capped at 10_000 entries, 24h TTL).

Redis is authoritative for logs; the subprocess pipe is the producer.
We don't write logs to Postgres — too spammy, no query pattern needs
them.

---

## 2. Dataset validation rules

A dataset is a directory at `${DATA_DIR}/datasets/{dataset_id}/`
containing `audio/*.wav` (or `.flac`, `.mp3`; librosa decodes) and an
optional `transcripts.jsonl`. The upload endpoint accepts the raw files
and stages them into that layout. Dataset validation runs in two phases.

### 2.1 Phase A — structural (runs on upload, synchronous)

Rules exactly match what the upstream loader (`voxcpm/training/data.py`
→ `load_audio_text_datasets`) expects to find in the manifest, plus the
sample-rate gate the AudioVAE encoder enforces:

1. **Total duration ≥ `MIN_DATASET_SECONDS`** (default 300; PRD §5.8
   says ~5 min). Computed by `soundfile.info()` on each file — cheap,
   no decode.
2. **Sample rate**: any rate is accepted at upload (the HF `Audio`
   column resamples to `sample_rate` at training time). We log the
   distribution to help the user catch bad mixtures. We do **not**
   reject non-16kHz files; the training YAML will be written with the
   AudioVAE's required rate (16000 for VoxCPM2) and HF does the
   resample.
3. **Channel layout**: reject files with >2 channels (training pipeline
   expects mono; stereo is down-mixed by `Audio(sampling_rate=...)`
   but 5.1/7.1 is a user error).
4. **No NaN / Inf / silent-only files**: quick peak check via `soundfile`
   read — reject if peak < `SILENCE_PEAK_THRESHOLD` (default 0.001)
   across the whole file.
5. **Transcript pairing**: if `transcripts.jsonl` is present, every
   `audio` path in it must exist and every audio file in `audio/`
   must appear exactly once. Name-based pairing fallback (stem match)
   is attempted when only `transcripts.jsonl` is missing some entries —
   we don't silently drop files, we either pair all or report the gap.
6. **Transcript field shape**: each JSONL row must have `audio: str`,
   `text: str` (non-empty after strip). `duration` is optional;
   `dataset_id` is optional (defaults to 0 per upstream).

On failure: reject the upload with a 400 listing every offending file,
not just the first. No partial state persists on disk — the staging
directory is `datasets/{dataset_id}.staging/`, renamed to the final
path atomically only after Phase A passes.

### 2.2 Phase B — transcript quality (runs during PREPARING)

Phase B is opt-in via request field `validate_transcripts: bool` on the
job-create request (default `true`). We run Fun-ASR over a random
subset (default 5% of samples, capped at 20 files) and compute WER
against the provided transcript after a normalisation pass
(lowercase, strip punctuation, collapse whitespace). A sample counts
as "suspect" if `WER > TRANSCRIPT_QUALITY_WER` (default 0.40 — loose;
we want to catch *wrong file paired with wrong text*, not mild
ASR-vs-human transcript drift).

Outputs:

- `validation_report.json` in the job dir (every sample's WER + the
  subset choice, seed saved for reproducibility).
- UI diff view (Step 2) surfaces suspect samples for human review.

**Phase B is advisory**, not blocking. A job with suspect transcripts
can still proceed — the user confirms in Step 2. This matches the
reference UI's flow and avoids false negatives blocking legitimate
datasets with non-standard orthography (Chinese, proper nouns, etc.).

### 2.3 Auto-transcription

If a dataset is uploaded with **no** `transcripts.jsonl`, the runner
synthesises one during `PREPARING` by calling `funasr_worker` on every
file. Results are written to the dataset dir as `transcripts.jsonl`
before training starts. Auto-transcription is not re-run on subsequent
jobs that reuse the same `dataset_id`.

**Load-bearing invariant**: training always reads a written
`transcripts.jsonl`. The auto-transcription path produces the same
on-disk artifact as a user upload, so the training script doesn't
care how it got there.

---

## 3. LoRA hot-swap strategy

**Decision: per-request, not per-session / not LRU-managed by our
code.** The PRD speculated about an LRU cache; the upstream API doesn't
need it. This section documents why and what we do instead.

### 3.1 What nano-vllm-voxcpm actually supports

Reading `third_party/nanovllm-voxcpm/nanovllm_voxcpm/models/voxcpm2/server.py`
and ADR 0002:

- `AsyncVoxCPM2ServerPool.register_lora(name, path)` loads a LoRA into
  CPU memory **permanently** until `unregister_lora(name)`. Multiple
  LoRAs can be registered concurrently — no CPU-side cap enforced.
- `server.generate(target_text=..., lora_name=name)` selects the LoRA
  per-request. Omitting `lora_name` uses base weights.
- `max_loras` in the init-time `LoRAConfig` sizes the **GPU slot
  pool** — the number of distinct LoRAs that can be active in a single
  batch simultaneously. The engine handles LRU eviction of idle GPU
  slots automatically; callers never see this.
- `unregister_lora(name)` drains — new requests rejected, in-flight
  generates run to completion, then the adapter is freed.

### 3.2 What this means for us

We register every LoRA-type voice at worker startup (from DB) and at
fine-tune completion. Voices stay registered for the lifetime of the
worker process. There is no code we write that evicts anything. Per-
request routing is `generate(..., lora_name=voice.id)`.

GPU slot count (`max_loras`) is an init-time tuning knob. Default:
**8**. Rationale: PRD §6 gives max_num_seqs=16 concurrent generations;
realistic voice mixtures in a single batch won't exceed ~8 distinct
LoRAs. If the engine has to spill to fewer slots it transparently
queues — we'd see elevated scheduling latency but no errors. 8 slots
at rank 32 over the attention+dit LoRA layers is roughly
O(100-300 MB) of GPU memory, acceptable.

Other init-time constraints that must be superset-compatible with
every LoRA we'll serve:

- `max_lora_rank` ≥ any trained rank. Default 32 (matches upstream
  YAML). We validate at fine-tune submission that
  `lora_rank ≤ max_lora_rank`; if the user requests higher, we refuse
  the job and tell them to lower rank or restart the worker with a
  bumped env var.
- `target_modules_lm` / `target_modules_dit` must be a superset of the
  LoRA's trained modules. Default `["q_proj","k_proj","v_proj","o_proj"]`
  for both — matches the upstream reference defaults.
- `enable_lm`, `enable_dit`, `enable_proj` each must be `True` at init
  time for the corresponding LoRA components to load. Defaults:
  `enable_lm=True`, `enable_dit=True`, `enable_proj=False`. Users who
  train with `enable_proj=True` will get a validation error at job
  submission telling them to set `LARYNX_VOXCPM_LORA_ENABLE_PROJ=1`
  in the worker env and restart.

### 3.3 voxcpm_worker API additions

Three new request/response types land in `shared/ipc/messages.py`:

- `LoadLoraRequest(voice_id, lora_path)` → `LoadLoraResponse`
- `UnloadLoraRequest(voice_id)` → `UnloadLoraResponse`
- `ListLorasRequest` → `ListLorasResponse(names: list[str])` — used on
  boot to reconcile DB state vs what the engine has.

`SynthesizeRequest` + `SynthesizeStreamRequest` gain an optional
`lora_name: str | None = None` field (passed through to
`backend.generate(..., lora_name=...)`). This is a pure additive
change — existing callers omit it and get base-weight synthesis.

On worker startup, the gateway's lifespan hook calls
`list_loras` against the fresh engine (result: empty set), queries
Postgres for `Voice.source='lora'` rows, and fires one `LoadLoraRequest`
per row. If any load fails (missing file, rank-too-high, etc.) we log
and skip that voice — it gets an `unloaded` flag surfaced in
`GET /v1/voices` and synthesis requests using it fail fast with a
clear error.

### 3.4 Edge cases

**E-LoRA-1. Hot-load after successful train.** Runner's `REGISTERING`
step sends `LoadLoraRequest`. If the worker rejects (rank too high,
proj modules unsupported, etc.), the job transitions to `FAILED`
with a specific error code and the Voice row is never written. No
half-registered state.

**E-LoRA-2. Worker restart mid-job.** Training subprocess has its own
CUDA context, so it survives. The runner coroutine does not — on
gateway restart, any job in `TRAINING` has lost its parent. Policy:
orphan reaper at boot moves any non-terminal job older than
`ORPHAN_JOB_GRACE` (default 60s) to `FAILED` with code
`gateway_restarted`. Subprocess cleanup: each training subprocess
writes its PID to `{job_dir}/subprocess.pid` at spawn; the orphan
reaper reads it and SIGKILLs. PID-file stale detection uses
`/proc/{pid}/cmdline` prefix match so we never kill an unrelated PID.

**E-LoRA-3. User deletes a voice whose LoRA is in flight.** Delete
calls `UnloadLoraRequest` first. nano-vllm draining lets in-flight
generates finish naturally; new requests fail fast with
`lora_not_registered`. We do NOT try to cancel in-flight synthesis
— they're sub-second and tearing them down races with audio delivery
to the client.

**E-LoRA-4. Two jobs completing concurrently want the same voice
name.** The Voice table has a unique index on `name`. Second INSERT
fails → job transitions to `FAILED` with `voice_name_conflict`. The
on-disk LoRA checkpoint is moved to `{lora_weights}/_orphaned/{ts}/`
for recovery; does not pollute the active directory. User picks a
different name and resubmits. Collision is narrow (clients control
name at job creation).

**E-LoRA-5. Engine rejects at scheduling time.** If a batch requires
more distinct LoRAs than `max_loras` slots, nano-vllm stalls the
waiting sequence — it does NOT error. User-perceived symptom: slower
synthesis under heavy multi-voice load. Surfaced in gateway latency
metrics; no orchestration changes required.

---

## 4. DB schema changes

Migration `0003_finetune_artifacts.py`. Two shape changes.

### 4.1 Voices extension

- `source` column accepts the new value `'lora'`. No enum constraint
  exists (it's a free-form `VARCHAR(32)` today — see migration 0002);
  we add an application-level enum
  `VoiceSource = Literal['uploaded','designed','seed','lora']` in
  `db/models.py` but leave the column text-typed. Aligns with the
  existing convention.
- `lora_path: Mapped[str | None]` — absolute path on disk to the LoRA
  checkpoint directory (the one holding `lora_weights.safetensors`
  and `lora_config.json`). `NULL` for non-LoRA voices.
- `ft_job_id: Mapped[str | None]` — application-level link (nullable,
  no DB-level foreign key, no cascade) back into `fine_tune_jobs.id`.
  Indexed. Lets the UI link a voice to its training job for a
  "show training report" affordance.

### 4.2 FineTuneJob table (new)

```python
class FineTuneJob(Base):
    __tablename__ = "fine_tune_jobs"

    id: Mapped[str]                       # uuid
    name: Mapped[str]                     # target voice name (not unique
                                          # here — unique goes on Voice.name)
    dataset_id: Mapped[str]               # points at
                                          # {DATA_DIR}/datasets/{dataset_id}
    state: Mapped[str]                    # JobState enum str — see §1
    voice_id: Mapped[str | None]          # set when state=SUCCEEDED
    config_json: Mapped[str]              # serialised overrides
                                          # (rank, alpha, epochs, etc.)
    resolved_config_path: Mapped[str]     # {job_dir}/train_config.yaml
    log_key: Mapped[str]                  # Redis key; matches §1.2
    error_code: Mapped[str | None]        # populated on FAILED
    error_detail: Mapped[str | None]      # short sentence; full in logs
    current_step: Mapped[int]             # last observed step
    max_steps: Mapped[int]                # from resolved config
    created_at: Mapped[datetime]
    started_at: Mapped[datetime | None]   # PREPARING entry time
    finished_at: Mapped[datetime | None]  # terminal-state entry time
```

Indices: `idx_ftjobs_state` (for the orphan reaper), `idx_ftjobs_voice_id`
(for the voice → job back-link).

### 4.3 Migration shape

```python
# 0003_finetune_artifacts.py
def upgrade() -> None:
    op.add_column("voices",
        sa.Column("lora_path", sa.String(length=1024), nullable=True))
    op.add_column("voices",
        sa.Column("ft_job_id", sa.String(length=36), nullable=True))
    op.create_index("idx_voices_ft_job_id", "voices", ["ft_job_id"])

    op.create_table(
        "fine_tune_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("dataset_id", sa.String(64), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("voice_id", sa.String(36), nullable=True),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("resolved_config_path", sa.String(1024), nullable=False),
        sa.Column("log_key", sa.String(128), nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("current_step", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_steps", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_ftjobs_state", "fine_tune_jobs", ["state"])
    op.create_index("idx_ftjobs_voice_id", "fine_tune_jobs", ["voice_id"])

def downgrade() -> None:
    op.drop_index("idx_ftjobs_voice_id", table_name="fine_tune_jobs")
    op.drop_index("idx_ftjobs_state", table_name="fine_tune_jobs")
    op.drop_table("fine_tune_jobs")
    op.drop_index("idx_voices_ft_job_id", table_name="voices")
    op.drop_column("voices", "ft_job_id")
    op.drop_column("voices", "lora_path")
```

We don't add a DB-level foreign key from `voices.ft_job_id` →
`fine_tune_jobs.id` because the row-delete order for a cancelled job
that never produced a voice is "delete job row, voices untouched".
Adding an FK introduces `ON DELETE SET NULL` or cascade ambiguity that
we sidestep by enforcing the link application-side only.

---

## 5. Arq vs existing IPC — how training jobs run

**Decision: reuse the existing `InProcessWorkerClient` + typed
`RequestMessage` / `StreamChunk` pattern. Do not add Arq.** This is a
judgment call and the tradeoffs deserve to be written down so a future
reviewer can re-weigh them.

### 5.1 The case for Arq

- **Persistent queue**: job rows survive gateway restart; a job submitted
  seconds before a crash reappears on boot.
- **Standard toolkit**: scheduling, retries, dead-letter queues, cron,
  concurrency limits are off-the-shelf.
- **Process isolation**: the Arq worker is a separate process tree, so
  a training crash doesn't take the gateway down.

### 5.2 The case for reusing our IPC

- **Training subprocess dies with the gateway anyway.** The upstream
  training script holds a CUDA context in its own process. When our
  gateway process dies we lose the subprocess handle, the subprocess is
  orphaned, and there's no safe way to re-adopt it after restart — the
  CUDA context is already owned by a dead parent. So "queue persistence"
  buys us nothing: on restart the job is dead regardless of where the
  queue state lives. The orphan reaper in §1 already handles this; Arq's
  persistence just makes us think we're safer than we are.
- **Progress already fits `StreamChunk`/`StreamEnd`.** Training logs and
  step events are a streaming RPC by nature: one request ("train this
  LoRA"), many chunks (log lines + step events), one terminal frame
  (`{state: SUCCEEDED | FAILED, voice_id?, error?}`). Our IPC was
  designed for exactly this shape and already has `CancelStreamRequest`
  wired up.
- **Single GPU ⇒ single worker.** "Concurrency control" in Arq would
  reduce to "max_concurrency=1", which is what `GPU_TRAIN_LOCK` gives
  us for free.
- **No new service.** Arq would need a new container, a new supervisord
  entry, a new dependency, and a new set of ops questions (what
  happens if the worker pool drops a heartbeat?). v1 doesn't need any
  of that.
- **Retries are the wrong default.** A failed LoRA run almost always
  needs human eyeballs on the config or the data. Automatic retries
  obscure the signal.

### 5.3 What the `training_worker` looks like in the reuse model

It's a sibling of `voxcpm_worker` / `funasr_worker`:
`packages/training_worker/src/larynx_training_worker/`. The gateway's
lifespan hook instantiates it alongside the others; it receives a
`WorkerChannel`; its `serve()` loop reads `TrainLoraRequest` off the
request queue and streams `TrainLogChunk` / `TrainStateChunk` /
`TrainDoneFrame` back.

Cancellation re-uses `CancelStreamRequest` — the worker's `_inflight`
dict looks up the job's `TrainingJobRunner` task and calls
`.cancel()`, which propagates into `job_cancel.set()` (see §1.1). No
new protocol.

**Load-bearing invariant**: `training_worker` never imports anything
from `voxcpm_worker`. The two workers communicate only through the
gateway as a hub — the training_worker's hot-swap step goes out over
the `voxcpm_client` the gateway already exposes. Keeps the worker
dependency DAG acyclic.

### 5.4 When we'd revisit

If we move to multi-box v2 and training needs to run on a separate
machine (or against a pool of training rigs), Arq or its equivalent
becomes the natural choice. The design of our IPC messages maps to a
network transport without shape changes; the queue becomes Redis-backed
at that point. That migration is out of scope for M7.

---

## 6. Subprocess strategy

### 6.1 Spawn

The training subprocess is launched via `asyncio.create_subprocess_exec`
(the safe variant that bypasses the shell). Arguments:

- **argv** — `[sys.executable, "third_party/VoxCPM/scripts/train_voxcpm_finetune.py",
  "--config_path", <resolved_config_path>]`. The Python interpreter is
  the same `uv`-resolved one the gateway uses, so the `[gpu]` extra is
  already on sys.path.
- **stdout** = `asyncio.subprocess.PIPE`.
- **stderr** = `asyncio.subprocess.STDOUT` (merged onto stdout; the
  upstream script logs most progress to stderr and we want a single
  ordered stream for the tailer).
- **stdin** = `asyncio.subprocess.DEVNULL`.
- **env** — inherited, overridden with `CUDA_VISIBLE_DEVICES=0` and
  `TOKENIZERS_PARALLELISM=false`. `HF_HOME` and `LARYNX_DATA_DIR`
  pass through unchanged.
- **cwd** = project root, so the `third_party/VoxCPM/scripts/...` path
  resolves and the training script's own relative imports work.
- **start_new_session** = `True`, so the subprocess heads its own
  process group — `os.killpg(pid, SIG)` cleanly tears down any
  DataLoader worker forks it creates.

After the call returns, we write `proc.pid` to
`{job_dir}/subprocess.pid` for the orphan-reaper path (§3.4 E-LoRA-2).

**Rule about the upstream script**: we invoke it as a script — we do
NOT import from it. Per §0, our code does not reach into `third_party/`.
The reason it's OK to spawn the script despite the import rule is that
process isolation acts as our import firewall: the subprocess has its
own Python interpreter, its own `sys.path`, and whatever that script
imports (including from `third_party/VoxCPM/src/`) is contained within
the subprocess's module graph.

If the `third_party/` reference pattern later becomes a vendoring
pattern, we replace the script path with a vendored copy at
`scripts/train_voxcpm_finetune.py` with attribution in the header.
Neither pattern changes the subprocess protocol.

### 6.2 Log tailing

One task reads `proc.stdout` line-by-line. Each raw line is decoded
(utf-8, `errors="replace"`), right-stripped of `\n`, and then:

1. Appended to Redis stream `logs:training:{job_id}` via `XADD`
   with `maxlen=~10_000` and a 24h TTL on the key.
2. Sent to the IPC channel as a `TrainLogChunk(request_id, line)`.
3. Passed to `_parse_training_event` — a small regex set against the
   upstream `TrainingTracker.log_metrics` output format (known line
   shapes are `"step={N} loss/diff={F} loss/stop={F} lr={F} epoch={F}"`).
   Matches produce a `TrainStateChunk(request_id, step, loss_diff,
   loss_stop, lr, epoch)`. Unmatched lines flow through as raw logs;
   we don't mine them for state.

### 6.3 Completion detection

The runner awaits `proc.wait()` concurrently with `job_cancel`, bounded
by `TRAIN_MAX_WALL_SECONDS` (default 24h; configurable).

Success criteria — **all must hold**:

1. Return code is 0.
2. `{save_path}/latest/lora_weights.safetensors` exists.
3. `{save_path}/latest/lora_config.json` exists and parses.
4. The reported `training_state.json → step` is within 5% of
   `max_steps` (catches short-circuit crashes that happened to exit
   cleanly — a real finish hits the last step).

Any failure transitions to `FAILED`; the reason goes into
`error_code` / `error_detail`. Known codes: `nonzero_exit`,
`missing_artifact`, `bad_lora_config`, `early_exit`,
`wall_timeout`, `gateway_restarted`, `voice_name_conflict`,
`hot_load_rejected`.

### 6.4 Termination and kill

- **Graceful (cancel, completion)**: `proc.terminate()` sends SIGTERM.
  The upstream script's signal handler writes a final checkpoint and
  exits. We wait `TRAIN_CANCEL_GRACE_SECONDS`.
- **Hard (grace exceeded, wall timeout)**: `os.killpg(proc.pid,
  SIGKILL)` — takes the whole process group so any lingering PyTorch
  DataLoader workers die with it.
- **On gateway shutdown**: lifespan hook cancels every active
  `TrainingJobRunner`; the cancel path above runs for each. Jobs in
  `QUEUED` transition to `FAILED(gateway_shutdown)` so the orphan
  reaper at next boot has less work.

### 6.5 Subprocess I/O backpressure

If the log consumer falls behind (gateway too busy, Redis stuck), the
subprocess's stdout pipe will fill and block. That's actually the
behaviour we want — we never want to silently drop log lines. We size
the consumer to be always faster than the producer: one Redis XADD per
line at ~1k lines/second is comfortably within Redis throughput. If
the pipe ever blocks the subprocess for more than `LOG_BACKPRESSURE_
WARN_SECONDS` (default 5) we emit a warning event but do not kill.

---

## 7. Test gating

The M7 prompt suggests `FULL_INTEGRATION_TESTS=1`. **We use the
existing convention instead**: `RUN_REAL_MODEL=1` + the `real_model`
pytest marker, already wired in the root `pyproject.toml`. Don't
introduce a parallel gate.

For M7 we add one more marker to distinguish the inference vs. training
cost:

```toml
markers = [
  "real_model: requires the real VoxCPM2 / Fun-ASR model on GPU ...",
  "real_train: requires a real LoRA training run (~minutes, GPU 0)",
]
addopts = "-m 'not real_model and not real_train'"
```

Invocation:

- Unit tests (dataset validation, config builder, state-machine
  transitions with a fake subprocess, migration round-trip): default
  `pytest`.
- Integration with real voxcpm_worker + real Fun-ASR but a tiny fake
  training script that emits a valid LoRA artifact in <5s:
  `RUN_REAL_MODEL=1 pytest -m real_model`.
- Full end-to-end with the actual upstream training script on a
  30-second clip for 10 steps:
  `RUN_REAL_TRAIN=1 RUN_REAL_MODEL=1 pytest -m "real_model or real_train"`.
  This is the opt-in test the prompt's "minimal fine-tune" criterion
  asks for. It takes GPU time — tens of seconds to a couple of
  minutes — so it stays off by default.

Memory-principle-compliance: no mocked Redis, no fakeredis, no in-
memory Postgres. Tests use a real Redis container (from
`docker-compose.yml`) at the M7 non-mock test level and real aiosqlite
only for migration-shape tests.

---

## 8. Prerequisites

Must land **before** any `training_worker` code is written, because
they're cross-component changes other reviewers will want to see
isolated.

### 8.1 voxcpm_worker LoRA API

Additive changes to `packages/voxcpm_worker/`:

- `VoxCPMBackend` gains `load_lora(name, path)`, `unload_lora(name)`,
  `list_loras()` abstract methods.
- `VoxCPMBackendReal` delegates to
  `self._pool.register_lora(name, path)` /
  `self._pool.unregister_lora(name)` /
  `self._pool.list_loras()`. Use the names exactly as upstream does
  — `name`, not `lora_name`, to keep the public vocabulary clean.
- `MockVoxCPMBackend` implements the same methods over an in-memory
  dict so tests can exercise the full flow without a GPU. When
  `lora_name` is passed to the mock's `synthesize`, we shift the
  pitch by a name-derived amount — same mechanism used today for
  voice diffs, now keyed on LoRA name.
- `VoxCPMBackendReal.__init__` accepts a new `lora_config: LoRAConfig
  | None` kwarg and passes it through to `VoxCPM.from_pretrained`.
  Defaults: `max_loras=8`, `max_lora_rank=32`, `enable_lm=True`,
  `enable_dit=True`, `enable_proj=False`, target modules as in §3.2.
  Configurable via `LARYNX_VOXCPM_LORA_*` env vars.
- `SynthesizeRequest` / `SynthesizeStreamRequest` gain optional
  `lora_name: str | None = None` (see §3.3). Handler threads it
  through unchanged.

Tests:

- Unit: mock backend's load/unload/list state transitions; synthesize
  with unknown `lora_name` returns an error.
- `real_model`-gated: real backend loads a fixture LoRA (we ship one
  we trained ourselves at ~100 KB rank-8 for CI speed once we have
  the training pipeline).

### 8.2 Dataset dir contract

`${DATA_DIR}/datasets/{dataset_id}/` is formalised as a schema in
`shared/`. A small `dataset_paths.py` with typed helpers (so the
training_worker and gateway agree on where `audio/`, `transcripts.jsonl`,
and `validation_report.json` live) prevents string-path drift.

### 8.3 Redis Streams helper

We don't have a Redis Streams helper in the gateway today — only LRU
GET/SET for the latent cache. Add a thin wrapper in
`services/training_logs.py` with one method for append and one for a
bounded backward scan from a given id. Used by the SSE route and the
log tailer.

### 8.4 Voice source enum

Introduce `VoiceSource` as a `Literal` in `db/models.py` and update the
existing call sites (voice_library create paths) to use it. No runtime
change today — this is pure type discipline so the M7 change that adds
`'lora'` is one-line.

### 8.5 Lifespan boot order

`main.py`'s lifespan needs to:

1. Bring up voxcpm_worker (existing).
2. After `wait_for_ready`, list LoRA voices from Postgres.
3. Fire `LoadLoraRequest` for each — in parallel via `asyncio.gather`
   with `return_exceptions=True`. Log individual failures; mark the
   affected Voice rows `unloaded=True` in app.state (transient flag;
   not persisted).
4. Run the orphan reaper sweep (see §1.1 E-LoRA-2).
5. Start the gateway's HTTP server.

Steps 2-4 must complete before we accept TTS requests, so the
lifespan blocks on them.

---

## 9. Things this doc deliberately does NOT specify

- **UI component composition inside the `/finetune` wizard.** The steps
  are enumerated in the M7 prompt (upload → validate → configure → submit
  → watch → redirect). Component boundaries, shadcn primitives, and the
  SSE hook live in the implementation plan.
- **Specific URL shapes for sub-resources.** REST shape (§5 of PRD,
  broadly) is `/v1/finetune/{datasets,jobs}`; exact field names land
  in the pydantic schemas when we write them.
- **Arq migration details.** We're not using Arq (§5). If we ever
  change that, the migration deserves its own doc, not a paragraph
  here.
- **LoRA quality regression harness.** Detecting that a LoRA sounds
  like the target speaker is today a subjective test per the M7 exit
  criteria. We'll add a quantitative speaker-embedding similarity
  score in v1.5 once the baseline is shipping.
- **Multi-GPU training.** The upstream script supports
  `accelerator.world_size > 1`. We target single GPU 0 in v1. Adding
  multi-GPU is a configuration change plus a GPU-lock update; not
  design work.
- **Quota / rate limiting per user.** No auth model for multi-user
  yet; single-key deployments don't need quotas.

---

## 10. Runtime constants

Defaults in one place for the implementation plan to pull from:

| Constant                        | Default | Source                  |
|---------------------------------|---------|-------------------------|
| `MIN_DATASET_SECONDS`           | 300     | §2.1 / PRD §5.8         |
| `SILENCE_PEAK_THRESHOLD`        | 0.001   | §2.1                    |
| `TRANSCRIPT_QUALITY_WER`        | 0.40    | §2.2                    |
| `TRAIN_CANCEL_GRACE_SECONDS`    | 30      | §1.1 / §6.4             |
| `TRAIN_MAX_WALL_SECONDS`        | 86400   | §6.3                    |
| `ORPHAN_JOB_GRACE`              | 60      | §3.4 E-LoRA-2           |
| `LOG_BACKPRESSURE_WARN_SECONDS` | 5       | §6.5                    |
| `LORA_LOG_TTL_SECONDS`          | 86400   | §1.2                    |
| `max_loras` (engine)            | 8       | §3.2                    |
| `max_lora_rank` (engine)        | 32      | §3.2                    |
| Default LoRA rank / alpha       | 32 / 32 | upstream `voxcpm_finetune_lora.yaml` |
| Default `num_iters` / `max_steps` | 1000 / 1000 | upstream YAML     |
| Default batch_size              | 2       | upstream YAML           |
| Default grad_accum_steps        | 8       | upstream YAML (effective batch 16) |
| Default learning_rate           | 1e-4    | upstream YAML           |

All configurable via env vars prefixed `LARYNX_FT_*` (gateway-side) or
`LARYNX_VOXCPM_LORA_*` (voxcpm_worker-side). Env names land in the
implementation plan.
