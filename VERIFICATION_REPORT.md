# VERIFICATION_REPORT.md — bugs/001 concurrent-STT fix

## § 1. Bug reference

Primary writeup: **[bugs/001_concurrent_stt.md](bugs/001_concurrent_stt.md)**.

One-sentence fix summary: an `asyncio.Lock` on `FunASRBackendReal`
serialises concurrent `asyncio.to_thread` calls into vLLM's shared
`LLM`, and the streaming STT session's `run()` finally block now
tolerates `RuntimeError` alongside `WorkerError` so teardown races
don't shadow the original exception.

Commits landed on `feat/m7-finetune`:

```
610c9ac  test(gateway): 8-session concurrent STT (PRD goal 4)
87f45fb  test(gateway): bugs/001 fix verified — flip xfail to pass
bb13750  fix(funasr-worker): serialise inference + teardown-safe finally (bugs/001)
c6338db  test(gateway): regression tests for bugs/001 concurrent STT (xfail)
```

## § 2. Pre-fix state

From `bugs/001_concurrent_stt.md` § 1, the same data reproduced here
so this report stands alone.

`RUN_REAL_MODEL=1 uv run pytest packages/gateway/tests/integration/test_real_model_stream.py -m real_model -v -s` (pre-fix):

```
[stt_concurrent] session raised: TimeoutError:
[stt_concurrent] session raised: TimeoutError:
[stt_concurrent] session raised: TimeoutError:
[stt_concurrent][3] kinds=['speech_start', 'partial', 'speech_end', 'final'] intervals=[]
FAILED

>       assert len(successful) >= 2, (
            f"expected ≥ 2 concurrent sessions to complete, got {len(successful)}"
        )
E       AssertionError: expected ≥ 2 concurrent sessions to complete, got 1
E       assert 1 >= 2

============= 1 failed, 2 passed, 9 warnings in 482.84s (0:08:02) ==============
```

Three of four concurrent sessions deadlocked inside `FunASRNano.inference()`.
pytest exit also surfaced the smoking-gun warning:

```
RuntimeWarning: The executor did not finishing joining its threads within 300 seconds.
```

Confirming `asyncio.to_thread` workers were hung inside vLLM C++ code
and couldn't be joined.

## § 3. Post-fix pytest output

### § 3a. `test_real_model_stream.py` alone — primary fix verification

`RUN_REAL_MODEL=1 uv run pytest packages/gateway/tests/integration/test_real_model_stream.py -m real_model -v`:

```
collecting ... collected 4 items

test_tts_stream_ttfb_distribution                           PASSED [ 25%]
test_stt_stream_end_to_end_via_synthesized_audio            PASSED [ 50%]
test_stt_stream_four_concurrent_sessions                    PASSED [ 75%]
test_stt_concurrent_transcribe_rolling_does_not_deadlock    PASSED [100%]

================== 4 passed, 7 warnings in 127.23s (0:02:07) ===================
```

Four-for-four. The `>= 4` assertion on the 4-concurrent test holds;
the new no-deadlock regression test (added with `xfail(strict=True)`
in commit `c6338db`, flipped to expected-pass in commit `87f45fb`)
passes cleanly.

### § 3b. 8-concurrent-session test — PRD goal 4 proof

`RUN_REAL_MODEL=1 uv run pytest packages/gateway/tests/integration/test_real_model_stream.py::test_stt_stream_eight_concurrent_sessions -m real_model -v -s`:

```
[stt_8concurrent][0] kinds=['speech_start', 'partial', 'partial', 'partial', 'partial', 'speech_end', 'final'] intervals=['776ms', '769ms', '763ms']
[stt_8concurrent][1] kinds=['speech_start', 'partial', 'partial', 'partial', 'partial', 'speech_end', 'final'] intervals=['780ms', '786ms', '760ms']
[stt_8concurrent][2] kinds=['speech_start', 'partial', 'partial', 'partial', 'partial', 'speech_end', 'final'] intervals=['789ms', '797ms', '762ms']
[stt_8concurrent][3] kinds=['speech_start', 'partial', 'partial', 'partial', 'partial', 'speech_end', 'final'] intervals=['794ms', '816ms', '771ms']
[stt_8concurrent][4] kinds=['speech_start', 'partial', 'partial', 'partial', 'partial', 'speech_end', 'final'] intervals=['801ms', '833ms', '760ms']
[stt_8concurrent][5] kinds=['speech_start', 'partial', 'partial', 'partial', 'partial', 'speech_end', 'final'] intervals=['796ms', '848ms', '763ms']
[stt_8concurrent][6] kinds=['speech_start', 'partial', 'partial', 'partial', 'partial', 'speech_end', 'final'] intervals=['813ms', '858ms', '762ms']
[stt_8concurrent][7] kinds=['speech_start', 'partial', 'partial', 'partial', 'partial', 'speech_end', 'final'] intervals=['822ms', '874ms', '775ms']
[stt_8concurrent] GPU1 util min=0% mean=16% max=65% n=13
[stt_8concurrent] wall=7.0s finals=8 failed=0
[stt_8concurrent] interval p50=787ms p95=870ms n=24
PASSED
================== 1 passed, 2 warnings in 108.68s (0:01:48) ===================
```

### § 3c. Unit test suite

`uv run pytest packages/gateway/tests/unit/test_stt_stream_service.py -v`:

```
test_service_emits_full_event_sequence                                  PASSED
test_utterance_ordinal_starts_at_one_and_is_stamped_on_all_events       PASSED
test_utterance_ordinal_increments_across_utterances                     PASSED
test_partial_ordinal_matches_enclosing_utterance                        PASSED
test_stt_stream_run_finally_tolerates_stopped_client                    PASSED

========================= 5 passed, 1 warning in 2.80s =========================
```

Four pre-existing tests still pass; the new teardown-safe regression
test (commit `c6338db`, flipped in `87f45fb`) passes.

### § 3d. Full `real_model` suite — first-ever run

`RUN_REAL_MODEL=1 uv run pytest -m real_model -v`:

```
= 5 passed, 11 skipped, 299 deselected, 5 warnings, 6 errors in 168.54s =
```

Per-file breakdown:

| File | Passed | Skipped | Error | Notes |
| ---- | -----: | ------: | ----: | ----- |
| `test_real_model.py`               | 5 | 0 | 0 | TTS + voice CRUD — clean. |
| `test_real_model_conversation.py`  | 0 | 6 | 0 | Skipped: `OPENROUTER_API_KEY` not set in subshell. § 5. |
| `test_real_model_stream.py`        | 0 | 5 | 0 | Skipped in cross-module suite run. See bugs/002. Passes cleanly when run alone — see § 3a / § 3b above. |
| `test_real_model_stt.py`           | 0 | 0 | 6 | OOM during fixture setup. See bugs/002. |

The 5 errors all share the same root cause: `torch.OutOfMemoryError` on
GPU 0 during fresh app lifespan, triggered by residual vLLM subprocess
memory from earlier test modules. Captured in
**[bugs/002_full_real_model_suite_gpu_accumulation.md](bugs/002_full_real_model_suite_gpu_accumulation.md)**
and unambiguously not a regression from bugs/001's fix.

Raw log: `bugs/_full_real_model.log`.

## § 4. Concurrency ceiling

Post-fix measurements on RTX Pro 6000 Blackwell Max-Q (97,887 MiB per GPU),
pinned to one model checkpoint (`Fun-ASR-Nano`) on GPU 1 under the
serialised-inference lock added in commit `bb13750`:

| Sessions | # reached final | p50 partial | p95 partial | p99 partial | Wall     | GPU 1 min / mean / max |
| -------: | --------------: | ----------: | ----------: | ----------: | -------: | ---------------------- |
| 1        | 1 / 1           |     770 ms* |       n/a   |       n/a   |     ~2 s | n/a (not sampled)       |
| 4        | 4 / 4           | (test-level gate; individual numbers in `bugs/_post_fix_stream.log`) |
| 8        | 8 / 8           |     787 ms  |     870 ms  |     874 ms† |   7.0 s  | 0 % / 16 % / 65 %       |

<sub>\* From `test_stt_stream_end_to_end_via_synthesized_audio` `intervals` line, § 3a raw log.</sub>

<sub>† `statistics.quantiles` with n=20 on 24 samples; treat p99 ≈ p95 — sample size too small for a meaningful p99.</sub>

Interpretation:

- All 8 sessions complete; bugs/001 goal met, PRD goal 4 met for the
  STT leg.
- Partial cadence under 8-way contention (787 ms p50) is essentially
  indistinguishable from single-session cadence (770 ms p50). That's
  because actual inference time per partial is ~75–150 ms in warm
  state — the lock rarely blocks in practice at this load.
- GPU 1 utilisation mean 16 % with max 65 % indicates significant
  headroom. The serialised lock is NOT the throughput ceiling at 8
  sessions; the hardware is lightly loaded. Room for a batching
  coalescer if we ever need to scale beyond ~15 sessions — tracked
  as the follow-on in bugs/001 § 4.4.

## § 5. Conversation tests — first-ever real-hardware run

**Not run.** All 6 parameterisations of the two conversation tests
(`test_conversation_three_turn_happy_path` and
`test_conversation_barge_in_real_model`, × 3 model IDs) skipped with:

```
OPENROUTER_API_KEY not set — required for the LLM leg
```

Root cause: `OPENROUTER_API_KEY` was not present in the `uv run`
subshell environment when pytest was invoked. bugs/001's original
prompt stated the key would be available; whatever mechanism was
expected to populate it (shell rc, `.env` auto-load, etc.) did not
take effect in this session.

**This is flagged as an open item, NOT a fix target for this branch.**
Per bugs/001's scope-discipline rule, an environment-setup gap is
orthogonal to the STT-concurrency bug. The conversation tests'
first-ever real-hardware run is deferred until the env-var delivery
path is confirmed.

## § 6. What this work did NOT fix

Explicit, no hedging:

1. **`test_real_model_stt.py` is still effectively blocked** on
   missing audio fixtures (`packages/gateway/tests/fixtures/audio/`
   doesn't exist; only `.gitkeep` is in the parent). The 6 errors
   observed in § 3d are OOM-at-fixture-setup symptoms that mask the
   missing-fixture skips. Seeding fixtures is out of scope per the
   prompt's 15-minute cap.
2. **bugs/002 filed but not fixed** —
   `bugs/002_full_real_model_suite_gpu_accumulation.md`. Full
   `real_model` pytest invocations leak GPU memory across modules and
   leak env vars from the stream fixture into later modules. Only
   manifests when multiple real-model files run in the same pytest
   invocation; does NOT affect production.
3. **`OPENROUTER_API_KEY` not wired into the test subshell** — the
   conversation tests skipped rather than running for the first time
   on real hardware. Flagged in § 5.
4. **Pre-existing F841 lint error** in
   `packages/gateway/tests/integration/test_real_model_stream.py:305`
   (unused local `silence = b"\x00\x00" * (16000 * 1 // 1 * 2 // 2)`).
   Introduced by commit `d409a8e7`, pre-dates this branch. Not fixed
   per scope-discipline rule.
5. **Batching-coalescer follow-on** — bugs/001 § 4.4. The
   serialised-lock fix is correct and meets PRD goals 4 + 10, but GPU 1
   mean utilisation during 8-way load is 16 % — there's headroom to
   add a coalescing request batcher inside the Fun-ASR worker that
   would amortise vLLM scheduler overhead. Worth landing before we
   need to scale past ~15 concurrent STT streams. Not required for v1.
6. **pytest's `"The executor did not finishing joining its threads
   within 300 seconds"` warning** — no longer fires in post-fix
   stream-file runs (threads finish and join cleanly), so this is
   implicitly resolved. Included here only as a "don't regress" marker
   for future changes to `FunASRBackendReal`.

## § 7. Shippable?

**Verdict: (b) Near-shippable.**

The STT concurrency blocker that drove this work (bugs/001) is
fixed, tested end-to-end on real hardware, and meets PRD goal 4
(≥8 simultaneous sessions) with comfortable margin on the STT leg.
Conversation-path real-hardware verification is gated only on the
`OPENROUTER_API_KEY` environment-setup gap (§ 5), not on any product
defect, and the two infrastructure gaps (bugs/002, fixture seeding)
affect test-harness ergonomics rather than production correctness.
Once OPENROUTER is wired and the conversation suite runs green, this
branch is ready to merge.
