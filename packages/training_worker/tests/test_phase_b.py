"""Phase B — transcript-quality sanity check against Fun-ASR.

Advisory, not blocking (per ORCHESTRATION-M7.md §2.2). Validates that
the provided transcripts aren't wildly mismatched with what Fun-ASR
hears in the audio — catches "wrong audio paired with wrong text"
scenarios without rejecting legitimate transcriber drift.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import soundfile as sf
from larynx_shared.paths import DatasetPaths
from larynx_training_worker.dataset_prep import (
    PhaseBReport,
    normalise_transcript,
    validate_transcripts_phase_b,
    word_error_rate,
)

SR = 16_000


def _seed(tmp_path: pathlib.Path, pairs: list[tuple[str, str]]) -> DatasetPaths:
    dp = DatasetPaths(data_dir=tmp_path, dataset_id="ds-b")
    dp.audio_dir.mkdir(parents=True)
    rows = []
    for i, (text, _) in enumerate(pairs):
        path = dp.audio_dir / f"c{i}.wav"
        samples = np.linspace(-0.3, 0.3, SR, dtype=np.float32)
        sf.write(path, samples, SR, subtype="PCM_16")
        rows.append({"audio": str(path), "text": text})
    with dp.transcripts_jsonl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return dp


# -- WER primitive ----------------------------------------------------------


def test_word_error_rate_identical() -> None:
    assert word_error_rate("hello world", "hello world") == 0.0


def test_word_error_rate_empty_reference() -> None:
    # Convention: empty reference + non-empty hypothesis -> 1.0 (fully
    # wrong), empty both -> 0.0.
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "hello") == 1.0


def test_word_error_rate_single_substitution() -> None:
    # "the quick fox" vs "the quick brown fox" -> 1 insertion / 3 ref
    # words ≈ 0.333.
    wer = word_error_rate("the quick fox", "the quick brown fox")
    assert abs(wer - 1 / 3) < 0.01


def test_word_error_rate_case_insensitive() -> None:
    # WER itself is case-sensitive; ``normalise_transcript`` is where we
    # lowercase. Keeping them separable is intentional.
    assert word_error_rate("Hello", "hello") == 1.0
    assert word_error_rate(normalise_transcript("Hello!"), normalise_transcript("hello")) == 0.0


def test_normalise_transcript_strips_punctuation_and_case() -> None:
    assert normalise_transcript("Hello, world!  It's a test.") == "hello world it's a test"


# -- Phase B ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_b_happy_path(tmp_path: pathlib.Path) -> None:
    # Every sample has the same written text; we return the same text
    # as the ASR hypothesis, so WER = 0 uniformly and nothing is
    # flagged as suspect.
    matching_text = "the quick brown fox jumps over the lazy dog"
    dp = _seed(tmp_path, [(matching_text, "") for _ in range(20)])

    async def perfect_transcribe(pcm: bytes, sr: int) -> str:
        del pcm, sr
        return matching_text

    report = await validate_transcripts_phase_b(
        dp, transcribe=perfect_transcribe, subset_fraction=1.0, max_samples=20, seed=42
    )
    assert isinstance(report, PhaseBReport)
    assert report.num_samples == 20
    assert len(report.suspects) == 0


@pytest.mark.asyncio
async def test_phase_b_catches_mismatched_transcripts(tmp_path: pathlib.Path) -> None:
    dp = _seed(
        tmp_path,
        [("this is definitely the wrong file label", "") for _ in range(20)],
    )

    async def wrong_transcribe(pcm: bytes, sr: int) -> str:
        del pcm, sr
        return "completely different output that shares zero tokens"

    report = await validate_transcripts_phase_b(
        dp,
        transcribe=wrong_transcribe,
        subset_fraction=1.0,
        max_samples=20,
        seed=42,
        wer_threshold=0.4,
    )
    # All 20 samples have WER >> 0.4 → all flagged as suspect.
    assert report.num_samples == 20
    assert len(report.suspects) == 20
    # Each suspect records the provided text, hypothesis, and WER.
    for s in report.suspects:
        assert s.wer > 0.4
        assert s.reference.startswith("this is definitely")
        assert "different output" in s.hypothesis


@pytest.mark.asyncio
async def test_phase_b_subset_fraction(tmp_path: pathlib.Path) -> None:
    dp = _seed(
        tmp_path,
        [(f"line {i}", "") for i in range(100)],
    )

    calls = 0

    async def counting_transcribe(pcm: bytes, sr: int) -> str:
        nonlocal calls
        calls += 1
        del pcm, sr
        return ""

    report = await validate_transcripts_phase_b(
        dp,
        transcribe=counting_transcribe,
        subset_fraction=0.1,
        max_samples=20,
        seed=42,
    )
    # 10% of 100 = 10, capped by max_samples=20 (no cap hit here).
    assert report.num_samples == 10
    assert calls == 10


@pytest.mark.asyncio
async def test_phase_b_writes_validation_report(tmp_path: pathlib.Path) -> None:
    dp = _seed(tmp_path, [("a b c", "") for _ in range(5)])

    async def trans(pcm: bytes, sr: int) -> str:
        del pcm, sr
        return "a b c"

    await validate_transcripts_phase_b(
        dp, transcribe=trans, subset_fraction=1.0, max_samples=5, seed=7
    )
    assert dp.validation_report_json.is_file()
    blob = json.loads(dp.validation_report_json.read_text())
    assert blob["num_samples"] == 5
    assert blob["seed"] == 7
    assert "suspects" in blob


@pytest.mark.asyncio
async def test_phase_b_no_manifest_is_zero_samples(tmp_path: pathlib.Path) -> None:
    # Phase B requires a manifest; no file -> 0 samples, empty report,
    # non-blocking.
    dp = DatasetPaths(data_dir=tmp_path, dataset_id="ds-empty")
    dp.audio_dir.mkdir(parents=True)

    async def trans(pcm: bytes, sr: int) -> str:
        del pcm, sr
        return ""

    report = await validate_transcripts_phase_b(
        dp, transcribe=trans, subset_fraction=1.0, max_samples=10, seed=1
    )
    assert report.num_samples == 0
    assert report.suspects == []
