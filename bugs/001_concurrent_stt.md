# Bug 001 — `test_stt_stream_four_concurrent_sessions` fails: 3 of 4 sessions hang

## § 1. Symptom

**Test:** `packages/gateway/tests/integration/test_real_model_stream.py::test_stt_stream_four_concurrent_sessions`

**Command to reproduce:**

```
RUN_REAL_MODEL=1 uv run pytest \
  packages/gateway/tests/integration/test_real_model_stream.py \
  -m real_model -v -s
```

**Outcome:** 1 of 3 tests in the file fails. Full pytest summary:

```
FAILED packages/gateway/tests/integration/test_real_model_stream.py::test_stt_stream_four_concurrent_sessions
============= 1 failed, 2 passed, 9 warnings in 482.84s (0:08:02) ==============
```

**Concurrent-sessions test output:**

```
[stt_concurrent] session raised: TimeoutError:
[stt_concurrent] session raised: TimeoutError:
[stt_concurrent] session raised: TimeoutError:
[stt_concurrent][3] kinds=['speech_start', 'partial', 'speech_end', 'final'] intervals=[]
FAILED
```

**Assertion that actually fails (verbatim):**

```
>       assert len(successful) >= 2, (
            f"expected ≥ 2 concurrent sessions to complete, got {len(successful)}"
        )
E       AssertionError: expected ≥ 2 concurrent sessions to complete, got 1
E       assert 1 >= 2
E        +  where 1 = len([(3, [], ['speech_start', 'partial', 'speech_end', 'final'])])

packages/gateway/tests/integration/test_real_model_stream.py:464: AssertionError
```

Only session index 3 completed. Sessions 0, 1, 2 raised `TimeoutError` — their
`asyncio.wait_for(ws.recv(), timeout=45.0)` calls expired because no further
WS frames arrived from the gateway after the initial `speech_start`.

**Secondary noise in the traceback** (these are the errors the prompt mentioned;
they fire *after* the test already failed, during test-fixture teardown — see
§2 for why this is not the root cause):

```
  File "packages/gateway/src/larynx_gateway/services/stt_stream_service.py", line 265, in _finalise_utterance
    roll = await self._funasr.transcribe_rolling(
  File "packages/gateway/src/larynx_gateway/workers_client/funasr_client.py", line 71, in transcribe_rolling
    return await self._rpc.request(req, TranscribeRollingResponse, timeout=self._timeout_s)
  File "packages/shared/src/larynx_shared/ipc/client_base.py", line 144, in request
    result = await asyncio.wait_for(fut, timeout=timeout)
  File "/usr/lib/python3.12/asyncio/tasks.py", line 520, in wait_for
    return await fut
asyncio.exceptions.CancelledError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "packages/gateway/src/larynx_gateway/routes/stt_stream.py", line 167, in ws_stt_stream
    await session.run(pcm_source())
  File "packages/gateway/src/larynx_gateway/services/stt_stream_service.py", line 176, in run
    await self._vad.vad_stream_close(session_id=self._session.session_id)
  File "packages/gateway/src/larynx_gateway/workers_client/vad_punc_client.py", line 70, in vad_stream_close
    return await self._rpc.request(req, VadStreamCloseResponse, timeout=self._timeout_s)
  File "packages/shared/src/larynx_shared/ipc/client_base.py", line 138, in request
    raise RuntimeError("InProcessWorkerClient.start() must be called first")
RuntimeError: InProcessWorkerClient.start() must be called first
```

**Pytest-level teardown warning** (critical clue — see §2):

```
RuntimeWarning: The executor did not finishing joining its threads within 300 seconds.
  self._context.run(self._callback, *self._args)
```

The full raw log (482s cold run, three tests, includes all four sessions'
gateway-side logs) lives at `bugs/_raw_repro.log`. The previous-passing tests
(`test_tts_stream_ttfb_distribution`: p50 TTFB 116.7 ms; `test_stt_stream_end_to_end_via_synthesized_audio`: 5 partials, WER=0.00, finalization 425ms) confirm
the TTS path, single-session STT path, and VAD/punc paths are all healthy — the
failure is specific to N≥2 concurrent STT.

---

## § 2. Root cause

**Hypothesis (one sentence):** Concurrent calls to `FunASRNano.inference()`
from multiple `asyncio.to_thread` threads deadlock inside vLLM's shared `LLM`
instance, starving 3 of 4 sessions' rolling-partial decodes and leaving their
WebSocket handlers stuck waiting on futures that will never resolve.

**The "InProcessWorkerClient.start() must be called first" RuntimeError
cited in the original bug report is a teardown artifact, not the cause.**
The prompt conflated a secondary symptom surfaced during shutdown with the
primary failure. The rest of this section demonstrates that.

### §2.1 What the code path actually does

Per-session, the streaming STT route stands up:

- One `STTStreamSession` (`services/stt_stream_service.py`) with two cooperating
  asyncio tasks: `run()` (the ingest/feed loop) and `_partials_loop()` (fires
  every 720 ms during speech and calls `funasr.transcribe_rolling`).
- Both tasks use the **shared process-wide `FunASRClient`** hung off
  `app.state.funasr_client` (created in `main.py:125` lifespan).
- `FunASRClient.transcribe_rolling` → `InProcessWorkerClient.request` →
  `asyncio.Queue[RequestMessage]` shared with `FunASRWorkerServer`.
- `FunASRWorkerServer._serve` (`packages/funasr_worker/.../server.py:62-70`)
  pulls each request off the queue and **dispatches it onto a *new*
  `asyncio.Task`** — no per-worker serialisation.
- Each dispatch task ends at
  `FunASRBackendReal.transcribe_rolling` (`model_manager.py:341-380`) which calls
  `await asyncio.to_thread(_run)` where `_run` invokes
  `loaded.handle.inference(...)` — the `FunASRNano` wrapper around vLLM.

**Critical:** `FunASRBackendReal.transcribe_rolling` has **no lock** around the
`asyncio.to_thread` call. Four concurrent sessions → four concurrent OS threads
(drawn from `asyncio`'s default `ThreadPoolExecutor`, max 32 threads) calling
`FunASRNano.inference()` against the same shared `LLM` instance on GPU 1.

Contrast with the sibling VAD backend — `FunasrStreamingVad` (which handles
the VAD side of the streaming pipeline) explicitly serialises with an
`asyncio.Lock`:

```
# streaming_vad.py:204-210  (docstring)
# Thread safety: ``AutoModel.generate`` mutates model-internal buffers
# (beyond the per-session ``cache`` dict) — concurrent calls from
# different WS sessions under ``asyncio.to_thread`` corrupt state and
# trigger ``IndexError`` inside ``GetFrameState``. We serialise every
# ``generate`` call with an async lock so concurrent sessions time-slice
# the model cleanly.
```

The VAD path recognised this risk for FunASR and guarded against it. The
transcribe path does not, and the test hits it.

### §2.2 Evidence from the live run

From `bugs/_raw_repro.log` timestamps during the concurrent test (sessions
open at `23:02:05.376`, test wall-clock begins):

```
23:02:06.213  funasr.transcribe_rolling  infer_ms=115   text_len=0
23:02:06.971  funasr.transcribe_rolling  infer_ms=873   text_len=0
23:02:07.730  funasr.transcribe_rolling  infer_ms=1631  text_len=0
23:02:08.349  funasr.transcribe_rolling  infer_ms=79    text_len=35  is_final=True
23:02:08.707  vad.stream_close   session=1ba3646...  (session 3 only)

<< 40 seconds of total silence — no further transcribe_rolling returns >>

23:02:47.646  batch.cleanup
  (test's 45-second ws.recv timeout fires around here on the three stuck sessions)
```

Four `transcribe_rolling` calls return, ever. `infer_ms` walks up 115 →
873 → 1631 for the first three (consistent with an in-flight call holding the
model while the next arrivals wait behind it, **not** with vLLM batching — a
true batch would show ≈ the first call's latency for all four). Only session 3
then gets a clean `is_final=True` decode and closes. **The other three sessions
never see another `funasr.transcribe_rolling` response at all.**

The smoking gun from pytest itself:

```
RuntimeWarning: The executor did not finishing joining its threads within 300 seconds.
```

At interpreter shutdown, the asyncio threadpool cannot join the worker threads
that were executing `_run` inside `FunASRNano.inference()`. They are **hung
inside vLLM C++ code**, holding the GIL release from `asyncio.to_thread`.
Python can't kill them. The 300-second join timeout is the `ThreadPoolExecutor`
default; after that, Python gives up and lets the process exit.

This matches vLLM's documented threading model: an `LLM` object is designed to
be driven by **one** calling thread that submits requests to its internal
scheduler, which batches them. Direct concurrent `.generate()` / `.inference()`
calls from multiple Python threads are not supported and (empirically here)
deadlock.

### §2.3 Why single-session tests pass and this one doesn't

- `test_stt_stream_end_to_end_via_synthesized_audio` has **one** WS session →
  one thread in the pool running `_run` at a time → no contention.
- `test_tts_stream_ttfb_distribution` opens 20 WS connections but
  **sequentially** (outer `for i in range(20)` around `async with
  websockets.connect(...)`) so at most one TTS in flight at a time.
- The 4-concurrent test uses `asyncio.gather(*(run_one(i) for i in range(4)))`
  — four sessions active simultaneously, four threads into vLLM, first one
  wins, the rest deadlock.

### §2.4 Why the RuntimeError about `.start()` shows up (and why it's secondary)

Reading the raw log forward from the assertion failure:

1. `23:02:50` (ish): three sessions' `ws.recv()` 45-s timeouts fire →
   `TimeoutError` propagates out of `run_one()`, `asyncio.gather(..., return_exceptions=True)` collects them.
2. The test computes `len(successful) == 1`, asserts `>= 2`, **fails**.
3. Test function returns with the assertion; pytest starts tearing down the
   `live_server` module-scoped fixture (the three hung WS handlers are still
   running server-side — they're blocked awaiting futures that will never resolve).
4. The fixture's `finally` block sets `server.should_exit = True`; uvicorn's
   lifespan starts shutting down (`23:03:10.279 voxcpm_worker.stopped`).
5. The lifespan `finally` calls `funasr_client.stop()` and `vad_punc_client.stop()`,
   which set `_dispatcher = None` on each client and cancel the dispatcher task.
6. Task cancellation cascades to the three hung WS handlers' `await
   asyncio.wait_for(fut, ...)` calls → `CancelledError` propagates up through
   `_finalise_utterance` → `_handle_vad_events` → `_feed` → `run()`.
7. `run()`'s `finally` block unconditionally calls
   `self._vad.vad_stream_close(...)`. By now the VAD client's `_dispatcher` is
   `None`. `request()` raises `RuntimeError("InProcessWorkerClient.start() must
   be called first")`. That's the stacked exception the prompt described.

Timestamps confirm the order (`23:03:11.049 vad_punc_worker.stopped` → same ms
`stt_stream.failed` errors). So:

- **Primary bug:** non-thread-safe concurrent Fun-ASR inference → sessions hang
  → ws.recv times out → test assertion fails.
- **Secondary bug:** `STTStreamSession.run()`'s `finally` block is not
  robust against a stopped worker client during teardown — it raises a
  confusing `RuntimeError` instead of logging and moving on. Visible only
  *after* primary bug fails the test and triggers fixture teardown.

The prompt identified the secondary bug and missed the primary one. Both need
to be fixed; only the primary one makes the test pass.

---

## § 3. Is this a production bug?

**Primary (thread-safe Fun-ASR): yes, unambiguously.** Any two WS clients
that hit `/v1/stt/stream` simultaneously trigger the same two `asyncio.to_thread`
calls into the same shared `LLM`. The test fixture just makes this easy to
reproduce. Production hit: gateway deadlock on the second concurrent
streaming-STT session, indefinitely. Not just "tests serialise poorly" — every
second WS client blocks the first, with no recovery path short of a
gateway restart. This blocks PRD goal 4 (≥8 simultaneous conversational
sessions) and PRD §6 concurrent-STT §. **v1 blocker.**

**Secondary (teardown-safe finally): yes, lower severity.** In production,
the lifespan runs `client.stop()` only at gateway shutdown — at which point the
process is about to exit. But a WebSocket disconnect mid-stream (a client
network hiccup, a browser tab close, a load-balancer 60-s idle timeout) can
cancel the WS handler task while an in-flight `transcribe_rolling` is pending.
The handler's `finally` then tries to `vad_stream_close`. In that scenario
the clients are *still running* so the specific `RuntimeError` we see in
tests doesn't fire, but the same finally block can leak a
`WorkerError("timeout", ...)` from `vad_stream_close` if punctuation or VAD
is also stuck. It's a minor robustness bug; worth fixing once we're in there.

---

## § 4. Fix design

I pick a **hybrid of Option D (serialise Fun-ASR at the backend) + a
small piece of Option B (defensive idempotent teardown)**, reasoning below
against A/B/C from the prompt.

### §4.1 Why A/B/C from the prompt don't apply

The prompt's A/B/C framed the bug as a worker-client lifecycle issue. That is
the **secondary** symptom. The primary bug is *inside* the Fun-ASR backend
and has nothing to do with `InProcessWorkerClient.start()`/`.stop()` at all.
A ref-counted client lifecycle, idempotent `start()`, or fixture-scope lift
would each mask the teardown-time `RuntimeError` but leave the three sessions
still hung on vLLM — the test would still fail with `TimeoutError` on
`ws.recv()`. That's why the prompt's "after the fix, 4 concurrent sessions
pass" expectation cannot be met by any of A/B/C alone.

### §4.2 Primary fix — serialise `FunASRNano.inference()`

Add an `asyncio.Lock` on `FunASRBackendReal`. Wrap the `asyncio.to_thread(_run)`
calls in `transcribe` and `transcribe_rolling` so only one inference is in
flight at a time across the whole process.

Rationale:

- Mirrors the pattern the VAD worker already uses for `AutoModel.generate`
  (`FunasrStreamingVad._model_lock`) — not a new idea in the repo.
- Single-session tests stay fast (lock is uncontended).
- Concurrent sessions *serialise* cleanly: session N waits ~ (N-1) × decode_ms
  behind its turn on the lock. For a 79 ms final decode with 8 sessions,
  worst-case wait ≈ 560 ms — inside PRD §4's budget, and well under the
  120-s client timeout.
- Known limitation: this does **not** give us batched throughput. If we
  want the claimed fun-asr-vllm batching benefit, we need a real
  coalesce-and-batch loop inside the worker — tracked as a follow-on, not
  a prerequisite for this bug. The 8-session concurrency goal (PRD goal 4)
  is reachable at "serialised" throughput: 8 × 79ms per-utterance decode ≈
  632 ms finalize, still inside budget. Partial cadence under 8-way load
  will stretch from 720ms to ~1s — the 8-session test in §5 allows that.

**Alternative considered: `threading.Lock` instead of `asyncio.Lock`.**
Rejected because the lock guards `await asyncio.to_thread(...)` — the awaiting
coroutine cooperates cleanly with `asyncio.Lock`. A `threading.Lock` would
force every asyncio task to block its event loop thread during the wait.

**Alternative considered: refactor to a batching queue inside the worker.**
Rejected for this bug because it's a much bigger design change (per-request
batching vs. per-call dispatch, new result-fan-out plumbing, new timeouts).
The bugs/ doc rule is "scope discipline — fix what the bug requires". I'll
file the batching improvement separately if we still see PRD-§10 "rolling-
buffer efficiency" concerns after this fix.

### §4.3 Secondary fix — teardown-safe finally

In `STTStreamSession.run()`'s `finally` block, the `vad_stream_close` call
(and the final-flush `vad_stream_feed(is_final=True)`) should catch
`RuntimeError` and `WorkerError` both, log once, and move on. Don't raise
over a teardown cleanup step. Same treatment for `_vad.vad_stream_feed`
and `_finalise_utterance` when the caller is already unwinding from a
`CancelledError`.

Rationale: finally blocks are for cleanup. If cleanup itself fails because
the world is being torn down around us, the right answer is to log and let
the outer exception propagate — not to raise a second confusing exception
that obscures the original failure.

### §4.4 What I am explicitly NOT doing in this bug fix

Not touching `InProcessWorkerClient.start()` / `.stop()` semantics. The
existing design is fine for normal operation: one client per worker, owned
by lifespan, single start + single stop. The prompt's Options A/B/C would
each make `.start()` more tolerant of misuse, but misuse isn't what's
happening here. If we later find ourselves repeatedly writing "did I start
this?" boilerplate, we can revisit — it's the kind of change that wants its
own design doc, not a fold-in to this bug.

Not touching the fixture scope. Module scope is correct for `live_server` —
it amortises a 90-second model load across 3 tests and matches production
(lifespan owns the client's lifetime). Lifting to session scope doesn't
help here because there's only ever one module in this test file.

---

## § 5. Test plan

All tests gated `@pytest.mark.real_model` to keep them off the mock-only CI
lane. Infrastructure per repo rule: real Redis / real Postgres / real models.

### §5.1 Regression test — pre-fix `xfail`, post-fix `pass`

A new test that closely mirrors the failure without the 482-s cold-model load
tax. Two cases, in one new file or folded into existing
`test_real_model_stream.py`:

1. **`test_stt_concurrent_transcribe_rolling_does_not_deadlock`** — opens 4
   raw `FunASRClient.transcribe_rolling` calls against the same
   `app.state.funasr_client` via `asyncio.gather`, no WS layer. Asserts all
   4 return within `4 × single_call_budget`. Reproduces the primary bug
   directly. Marked `@pytest.mark.xfail(strict=True)` in the commit before
   the fix; the xfail marker is removed in the third commit.

2. **`test_stt_stream_run_finally_tolerates_stopped_client`** — covers the
   secondary bug independently. Starts a session, stops the funasr client
   out from under it, triggers `run()`'s finally by closing the pcm source,
   asserts that the session emits a clean session_closed log instead of
   raising `RuntimeError` outward.

### §5.2 Flip the existing 4-session test from fail to pass

`test_stt_stream_four_concurrent_sessions` should pass cleanly after the fix.
Existing threshold (`>= 2` successes) is passed trivially; the PR commit
that adds the fix also tightens it to `== 4` successes since serialised
inference guarantees all four complete.

### §5.3 New 8-session test — PRD goal 4 proof

`test_stt_stream_eight_concurrent_sessions`, sibling to the 4-concurrent
test. Looser tolerances to absorb 8-way contention:

- All 8 sessions must reach `final` at least once.
- `p50` partial interval allowed to stretch to 1.0 s (from 720 ms under 1-way).
- `p95` partial interval allowed up to 1.5 s.
- Wall-clock for the whole test bounded at 90 s (8× base × safety margin).

Print the GPU-1 utilisation snapshot (`nvidia-smi --query-gpu=utilization.gpu`)
sampled every 500 ms during the run; attach to VERIFICATION_REPORT.md.

### §5.4 Regression coverage for paths we didn't touch

- `test_stt_stream_end_to_end_via_synthesized_audio` — single-session STT
  path must still show p50 partial cadence ≤ 800 ms and WER ≤ 0.5 on the
  reference phrase. Lock is uncontended here so zero expected change.
- `test_tts_stream_ttfb_distribution` — TTS unaffected (different worker,
  different lock). Zero expected change.
- `test_stt_stream_end_to_end` (the non-streaming REST STT path, if present)
  — shares the same `transcribe` method so the lock is on its hot path
  too; must still pass.

### §5.5 WS-disconnect mid-stream regression

Not currently covered. Add:
`test_stt_stream_client_disconnect_mid_partial` — opens WS, starts streaming,
closes WS abruptly while a partial decode is in flight. Asserts: no
`RuntimeError` logged, session cleans up within 2 s, `funasr_client` still
usable for a following session. Covers the secondary-bug's production-side
manifestation.

---

## § 6. Rollback plan

**One-line revert.** The primary fix is a scoped change (add a lock, wrap two
calls); a `git revert <sha>` of commit 2 restores the old behaviour and
re-introduces the bug — no schema, no IPC protocol, no env-var changes. The
secondary fix is similarly scoped (try/except around two cleanup calls) and
also trivially revertable.

No DB migration, no protocol-version bump, no cross-service coordination.
Rollback can be done blind at any time without prior verification.

If we discover after merging that the serialised inference tanks throughput
badly enough to matter for production (unlikely given §4.2 numbers, but we
should watch `larynx_stt_stream_partial_interval_seconds` after deploy), the
mitigation is the batching-coalescer follow-on, not a revert.

---

## § 7. Known unknowns / what this doc can't prove until we try

- **Whether a single `asyncio.Lock` is sufficient, or whether
  `FunASRNano.inference()` holds torch/CUDA state that still collides under
  re-entry from the same thread.** I don't think it does — the failure mode
  is multi-thread — but the fix's regression test at 8-way will tell us.
- **Whether vLLM's own internal batching gives us anything while we hold the
  Python-side lock.** It shouldn't (we only submit one request at a time), so
  throughput is fundamentally serial. If PRD partial-cadence budgets squeeze
  us later, that's the lever to pull.
- **First-ever real-hardware run of the conversation tests is still pending.**
  `OPENROUTER_API_KEY` is now set; those tests have never run on real models.
  When I run the full `RUN_REAL_MODEL=1 -m real_model` suite post-fix, any
  conversation-test failures get filed as separate bugs in `bugs/` rather
  than fixed in this branch, per the prompt's scope-discipline rule.
