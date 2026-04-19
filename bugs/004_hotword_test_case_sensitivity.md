# Bug 004 — `test_hotword_recovery` case-sensitive assertion fails on lowercased Fun-ASR output

## § 1. Symptom

**Test:** `packages/gateway/tests/integration/test_real_model_stt.py::test_hotword_recovery`

**Command:** `RUN_REAL_MODEL=1 uv run pytest packages/gateway/tests/integration/test_real_model_stt.py::test_hotword_recovery -m real_model -v -s`

**Outcome (2026-04-18, first real-hardware run after STT fixtures landed):**

```
E       AssertionError: assert 'Larynx' in ' Please contact larynx for help .'

packages/gateway/tests/integration/test_real_model_stt.py:206: AssertionError
FAILED ... 1 failed, 5 passed, 2 warnings in 533.75s (0:08:53)
```

The transcript returned by Fun-ASR-Nano with `hotwords="Larynx"`:

```
' Please contact larynx for help .'
```

**Fun-ASR _did_ place the hotword in the transcript** — just lowercased.
The test's assertion is case-sensitive:

```python
assert "Larynx" in r.json()["text"]
```

`"Larynx" in "larynx"` is `False`, so the assertion fails even though the
hotword ability is demonstrably working (Fun-ASR produced the proper-noun
stem; without the hint it typically emits `lorex` / `laryx` / similar).

## § 2. Root-cause hypothesis

**Fun-ASR's text-normalisation lowercases proper nouns that aren't in
its training vocabulary.** The hotword injection biases the decoder
toward the term's phoneme sequence but doesn't by itself override the
downstream itn (inverse text-normalisation) / punctuation pass that
re-casts all tokens to lowercase for English output.

Two independent observations support this:

1. The infer log shows `text_len=31` → post-punctuate `text_len=33`,
   consistent with `Please contact larynx for help` + punctuation.
2. The other passing English test (`test_english_wer_under_10pct`)
   transcribes "Hello from the voice platform smoke test." cleanly
   under the same itn=True path; sentence-initial `Hello` is capitalised
   there because Fun-ASR auto-capitalises the first token. `Larynx` is
   mid-sentence, so that capitalisation rule doesn't apply.

## § 3. Production impact

**Low.** The hotword recovery _feature_ works at the phoneme level —
passing `hotwords="Larynx"` did cause Fun-ASR to emit the correct stem
verbatim (no `lorex` / `laryx` phonetic approximation). Only the case is
wrong, and a downstream consumer that wants the proper-noun capitalised
can re-case it from the hotword list. This is a test-assertion tightness
bug, not a product bug.

## § 4. Fix sketch (deferred)

Two possible fixes; pick whichever is more aligned with product intent.

- **§ 4a — relax the assertion**
  ```python
  assert "larynx" in r.json()["text"].lower()
  ```
  Asserts the stem landed in the transcript, case-insensitive. Matches
  the underlying capability (hotword gets the phoneme right).

- **§ 4b — re-apply hotword case in the gateway**
  In `packages/gateway/src/larynx_gateway/routes/stt.py` (or the STT
  service), after Fun-ASR returns, scan the hotword list against the
  transcript case-insensitively and re-capitalise matches to the
  hotword's original case. That would make the test pass as-written
  and also give every hotword caller a better client-facing transcript.
  Slightly more surface area; benefits every consumer of `/v1/stt`,
  not just this test.

§4b is the right long-term fix. Filing here to triage; not fixing in
the Item-3 "seed STT fixtures" commit per the scope-discipline rule
("file each failure as a separate bug in bugs/, don't fix inline").

## § 5. Priority

**Low.** Doesn't block v1 ship — the product returns the hotword stem,
just lowercased. The test currently fails so the suite reports
5/6 pass; once either fix lands the full STT suite goes green.

## § 6. When to fix

- Next time someone is touching the STT response-shaping code (natural
  fold-in for § 4b).
- Before the batch API (M8) ships, since batch consumers likely want
  correctly-cased proper nouns in their JSONL output.
