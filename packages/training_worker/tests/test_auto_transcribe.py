"""Auto-transcription — PREPARING-time synthesis of transcripts.jsonl.

The training_worker doesn't know about the gateway's FunASRClient; it
takes a ``transcribe`` callable so tests can inject deterministic text
and production passes a closure that wraps FunASRClient.transcribe.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import soundfile as sf
from larynx_shared.paths import DatasetPaths
from larynx_training_worker.dataset_prep import auto_transcribe_if_missing


def _write_wav(path: pathlib.Path, duration_s: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.3, 0.3, int(16_000 * duration_s), dtype=np.float32)
    sf.write(path, samples, 16_000, subtype="PCM_16")


def _make_dataset(tmp_path: pathlib.Path) -> DatasetPaths:
    dp = DatasetPaths(data_dir=tmp_path, dataset_id="ds-at")
    dp.audio_dir.mkdir(parents=True)
    for i in range(3):
        _write_wav(dp.audio_dir / f"c{i}.wav")
    return dp


async def _fake_transcribe(pcm_s16le: bytes, sample_rate: int) -> str:
    # Return a transcription that encodes the input length so tests
    # can confirm the right file was passed.
    return f"transcribed {sample_rate}Hz {len(pcm_s16le)} bytes"


@pytest.mark.asyncio
async def test_auto_transcribe_writes_manifest(tmp_path: pathlib.Path) -> None:
    dp = _make_dataset(tmp_path)
    assert not dp.has_transcripts()

    count = await auto_transcribe_if_missing(dp, transcribe=_fake_transcribe)
    assert count == 3
    assert dp.has_transcripts()

    rows = [json.loads(line) for line in dp.transcripts_jsonl.read_text().splitlines() if line]
    # One row per audio file, in sorted order.
    names = [pathlib.Path(r["audio"]).name for r in rows]
    assert names == ["c0.wav", "c1.wav", "c2.wav"]
    for r in rows:
        assert r["text"].startswith("transcribed")


@pytest.mark.asyncio
async def test_auto_transcribe_noop_if_manifest_present(tmp_path: pathlib.Path) -> None:
    dp = _make_dataset(tmp_path)
    dp.transcripts_jsonl.write_text(json.dumps({"audio": "c0.wav", "text": "pre-existing"}) + "\n")
    count = await auto_transcribe_if_missing(dp, transcribe=_fake_transcribe)
    assert count == 0
    # Manifest preserved as-is.
    rows = [json.loads(line) for line in dp.transcripts_jsonl.read_text().splitlines() if line]
    assert rows == [{"audio": "c0.wav", "text": "pre-existing"}]


@pytest.mark.asyncio
async def test_auto_transcribe_reads_audio_as_16k_s16(tmp_path: pathlib.Path) -> None:
    # File is written at 44.1kHz float; auto-transcribe must resample
    # to 16k s16 LE before calling the hook so the transcribe callable
    # sees a consistent format regardless of upload rate.
    dp = DatasetPaths(data_dir=tmp_path, dataset_id="ds-resample")
    dp.audio_dir.mkdir(parents=True)
    samples = np.linspace(-0.3, 0.3, int(44_100 * 1.0), dtype=np.float32)
    sf.write(dp.audio_dir / "high_sr.wav", samples, 44_100, subtype="PCM_16")

    seen: list[tuple[int, int]] = []

    async def spy(pcm_s16le: bytes, sample_rate: int) -> str:
        seen.append((sample_rate, len(pcm_s16le)))
        return "ok"

    await auto_transcribe_if_missing(dp, transcribe=spy)
    assert len(seen) == 1
    sr, n_bytes = seen[0]
    assert sr == 16_000
    # ~1 second @ 16k * 2 bytes/sample ≈ 32000 bytes, ±1 sample of slack.
    assert 30_000 <= n_bytes <= 34_000


@pytest.mark.asyncio
async def test_auto_transcribe_skips_failures_but_records_empty(
    tmp_path: pathlib.Path,
) -> None:
    dp = _make_dataset(tmp_path)

    async def flaky(pcm_s16le: bytes, sample_rate: int) -> str:
        # Fail on the second file.
        if len(pcm_s16le) == 32 * 2:
            raise RuntimeError("ASR blew up")
        return "ok"

    # The first iteration succeeds; the second raises and is logged
    # but does not abort the run. Subsequent files still get processed.
    count = await auto_transcribe_if_missing(dp, transcribe=flaky)
    # All 3 files produce a row (failures get empty text) — the caller
    # sees the count, not the error.
    assert count == 3
    rows = [json.loads(line) for line in dp.transcripts_jsonl.read_text().splitlines() if line]
    assert len(rows) == 3
