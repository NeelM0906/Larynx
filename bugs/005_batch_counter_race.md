# Bug 005 — `record_item_done` counter race leaves completed batch jobs stuck in RUNNING

## § 1. Symptom

**Test:** `packages/gateway/tests/integration/test_batch.py::test_batch_create_and_run`

**Command:** `uv run pytest -q` (only fails intermittently during full-suite runs; passes reliably when `test_batch.py` is invoked on its own)

**Outcome (2026-04-19, during M8 Part C work on feat/m8):**

```
AssertionError: job 066fcf15988946d7b43c5aaec9ceab48 did not reach COMPLETED;
last={'job_id': ..., 'state': 'RUNNING', 'progress': 0.9,
      'num_items': 10, 'num_completed': 9, 'num_failed': 0,
      ...
      'items': [
        {'idx': 0, 'state': 'DONE', ...},
        ...
        {'idx': 9, 'state': 'DONE', ...}
      ]}
```

All 10 item rows are `DONE`. The aggregate `num_completed` on the
`BatchJob` row is 9. The state-machine transition to `COMPLETED` is
gated on `num_completed + num_failed >= num_items`, so it never fires
and the test times out after 30s of polling.

Flake rate in local runs on feat/m8 (branched off `origin/main` at
`0f870a8`): roughly 1 in 3 full-suite runs. 0 in N isolated
`test_batch.py` runs.

## § 2. Diagnosis

`packages/gateway/src/larynx_gateway/services/batch_service.py:260-277`
— `_bump_counters_and_maybe_finish` is a read-modify-write without
row locking:

```python
async def _bump_counters_and_maybe_finish(
    session: AsyncSession,
    job_id: str,
    *,
    delta_done: int,
    delta_failed: int,
) -> None:
    job = (await session.execute(select(BatchJob).where(BatchJob.id == job_id))).scalar_one()
    job.num_completed += delta_done
    job.num_failed += delta_failed
    if job.num_completed + job.num_failed >= job.num_items:
        job.state = "FAILED" if job.num_completed == 0 else "COMPLETED"
        job.finished_at = _now()
```

The two in-gateway batch consumers (§1.3's `asyncio.Semaphore(2)` via
two consumer tasks) can both call this concurrently when items 8 and 9
finish within a few microseconds of each other. Each consumer runs its
own SQLAlchemy session with no `FOR UPDATE` on the `SELECT`:

1. consumer A: SELECT → num_completed=8, delta=+1, compute 9
2. consumer B: SELECT → num_completed=8, delta=+1, compute 9
3. A commits: UPDATE SET num_completed=9
4. B commits: UPDATE SET num_completed=9

One increment is lost. When `num_items=10` and both consumers finish
the last two items in the same window, the counter tops out at 9 and
the terminal transition never fires.

The mock VoxCPM worker returns synthesis in ~1 ms, which is why this
reproduces on the 10-item test but not on the slower 100-item batch
case — longer per-item latency spreads the commit windows out.

## § 3. Fix options

### 3.1 UPDATE ... SET counter = counter + delta (preferred)

Replace the read-modify-write with an atomic SQL update:

```python
result = await session.execute(
    update(BatchJob)
    .where(BatchJob.id == job_id)
    .values(
        num_completed=BatchJob.num_completed + delta_done,
        num_failed=BatchJob.num_failed + delta_failed,
    )
    .returning(
        BatchJob.num_completed,
        BatchJob.num_failed,
        BatchJob.num_items,
    )
)
num_completed, num_failed, num_items = result.one()
if num_completed + num_failed >= num_items:
    # Reload for the terminal transition; this SELECT races with
    # the sibling consumer but the terminal-state assignment is
    # idempotent — both sides will converge on the same COMPLETED
    # state.
    ...
```

Postgres' `UPDATE ... RETURNING` is atomic at the row level. Both
consumers' deltas land; no lost increment. This is the cleanest fix
and doesn't require changing session isolation levels.

### 3.2 SELECT ... FOR UPDATE

```python
stmt = select(BatchJob).where(BatchJob.id == job_id).with_for_update()
job = (await session.execute(stmt)).scalar_one()
```

Serialises the two consumer writes at the database level. Simpler
diff but strictly slower than 3.1 because of the lock acquisition.

### 3.3 Per-job asyncio lock

An `asyncio.Lock` keyed by `job_id` in the consumer process. Only
helps because the two consumers share a process; would fall over if
a future v1.x sprouts a second gateway replica. 3.1 is preferred.

## § 4. Scope

- **Workaround:** existing — the test is flaky, the prod impact is a
  job stuck at `RUNNING` with `num_completed = num_items - 1`. The
  per-day cleanup cron doesn't evict `RUNNING` jobs, so the stuck row
  would persist until manually reconciled.
- **Fix owner:** next batch-service touch. Not blocking M8 ship
  because it's a pre-existing bug on `origin/main` (observed in
  `feat/m8` at branch-time but present before) and the flake rate is
  low enough to retry in CI.
- **Regression test:** a 100-consumer stress test on a 500-item batch
  would catch it; the current suite only hits 10 items with 2
  consumers which masks it except under coincidence.

## § 5. Surface where discovered

Running the full unit suite during M8 Part C implementation
(`feat/m8`, after commits `46f8c36` and `84d5b12` for the worker
metrics sidecars). The sidecar changes do not touch any batch code,
so the flake is pre-existing on `origin/main` — the commits just
changed timing slightly enough to expose it at ~1-in-3 full-suite
runs.
