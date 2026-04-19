# Bug 002 — full `real_model` suite leaks GPU memory across test modules, skips stream tests + errors stt tests

## § 1. Symptom

**Command:** `RUN_REAL_MODEL=1 uv run pytest -m real_model -v`

**Outcome** (first-ever run on RTX Pro 6000, after bugs/001 lands):

```
= 5 passed, 11 skipped, 299 deselected, 5 warnings, 6 errors in 168.54s =

ERROR packages/gateway/tests/integration/test_real_model_stt.py::test_english_wer_under_10pct
ERROR packages/gateway/tests/integration/test_real_model_stt.py::test_chinese_transcript_reasonable
ERROR packages/gateway/tests/integration/test_real_model_stt.py::test_portuguese_uses_mlt
ERROR packages/gateway/tests/integration/test_real_model_stt.py::test_cantonese_dialect
ERROR packages/gateway/tests/integration/test_real_model_stt.py::test_hotword_recovery
ERROR packages/gateway/tests/integration/test_real_model_stt.py::test_punctuation_on_vs_off
```

`test_real_model_stream.py` tests 5/5 **skipped silently** (no skip reason
surfaced in `-v` output) — but they pass cleanly when run in isolation
(see VERIFICATION_REPORT.md § 3 for the post-fix standalone run).

`test_real_model_stt.py` tests 6/6 **error** during fixture setup with
CUDA OOM:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB.
GPU 0 has a total capacity of 94.97 GiB of which 41.75 MiB is free.
Process 85817 has 1.28 GiB memory in use.
Process 2470446 has 1.60 GiB memory in use.
Process 2471447 has 74.78 GiB memory in use.   ← earlier module's vLLM
Process 2474114 has 15.02 GiB memory in use.   ← earlier module's VoxCPM2
Including non-PyTorch memory, this process has 1.74 GiB memory in use.
```

Four `Process <pid>` entries — vLLM child subprocesses from earlier
test-module fixtures that weren't reaped before the next module's
fixture tried to allocate.

Raw log: `bugs/_full_real_model.log`.

## § 2. Root-cause hypothesis

Two independent infrastructure issues that only manifest when
multiple `real_model` test modules run in the same pytest invocation:

**§2.1 GPU memory doesn't release between test-module fixture scopes.**

Each real-model test module has its own module-scoped fixture that
loads VoxCPM2 + optionally Fun-ASR onto GPUs 0/1:

- `test_real_model.py` → `real_client` (ASGI transport in-process)
- `test_real_model_stream.py` → `live_server` (uvicorn + full lifespan)
- `test_real_model_stt.py` → per-test `client` from conftest.py

When a module's tests finish, the fixture `finally` runs
`worker.stop()` / `client.stop()`. But the vLLM child subprocesses
(one per `LLM()` instance, total 3: VoxCPM2 + Fun-ASR-Nano + MLT-Nano)
aren't explicitly torn down — `FunASRBackendReal.close()` just does
`self._models.clear()` and lets Python GC handle the vLLM refs. GC
doesn't guarantee immediate CUDA release, and vLLM's own subprocess
cleanup is known-racy. Memory accumulates across module boundaries.

**§2.2 Env-var leakage between modules puts stt tests in a bad
combined state.**

`test_real_model_stream.py::live_server` sets:

```python
os.environ["LARYNX_STT_MODE"] = "funasr"
os.environ["LARYNX_VOXCPM_GPU"] = "0"
os.environ["LARYNX_FUNASR_GPU"] = "1"
```

These env vars persist after the fixture tears down. When
`test_real_model_stt.py` then runs, its `_needs_real_model()` guard
checks `LARYNX_STT_MODE == "funasr"` and proceeds. Its per-test
`client` fixture from conftest.py then stands up a fresh app with
real models — on top of the GPU memory left over from the previous
module. Result: OOM on every test.

If `LARYNX_STT_MODE` were still `mock` (conftest.py's default),
these tests would skip cleanly and never try to allocate.

**§2.3 Why the stream tests silently skipped**

Unclear without more instrumentation. Hypothesis: the module's
`live_server` fixture tries to start uvicorn → lifespan tries to load
VoxCPM2 onto GPU 0 → GPU 0 is already saturated from `test_real_model`
module's residual allocation → lifespan raises → fixture falls
through to `pytest.skip("gateway failed to become ready within 360s")`
but the log shows `SKIPPED` without the reason string (pytest may drop
skip reasons under certain failure modes).

Needs `-rs` on the pytest invocation or a print inside the fixture to
confirm. Not doing that in this bug report — filing only.

## § 3. Production impact

**None for production.** The production gateway runs a *single* set of
model workers inside one uvicorn process, with a lifespan that owns
them for the life of the process. Production never reloads models
in-process or shuffles fixtures. This is strictly a test-infrastructure
bug.

## § 4. Fix sketch (deferred)

Two independent fixes, either of which would mask the symptom but
both of which are worth landing:

- **§4a. Scope lift or explicit GPU reset between test modules.**
  Either (a) run each real_model test module in a separate pytest
  subprocess (`pytest -m real_model --forked` or a Makefile target that
  invokes pytest once per file), or (b) add a session-scoped
  teardown that explicitly kills leftover vLLM subprocesses and
  calls `torch.cuda.empty_cache()`.
- **§4b. Env-var restoration in live_server fixture.** Snapshot
  `os.environ` on fixture entry; restore on teardown. Conservative
  fix — guarantees downstream modules see the conftest defaults.

## § 5. Impact on bugs/001

**Zero.** The stream-test file runs green when invoked alone (proven by
commits `87f45fb` and `610c9ac` verification runs). The bugs/001 fix
itself is validated. This bug is a pre-existing test-harness hygiene
gap that only surfaces now because bugs/001 was the first time we ran
the `real_model_stream` suite to green in any configuration.

## § 6. Priority

**Low-to-medium.** Doesn't block v1 ship (production isn't affected).
Does block confidently running `-m real_model` as a single CI
invocation. Workaround today: run each file separately, which is what
bugs/001 verification already does.

Filing now so we don't forget it after the bugs/001 branch merges.
