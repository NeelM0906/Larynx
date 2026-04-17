"""Phase-A dataset validation — structural checks before training.

Covers every rule in ORCHESTRATION-M7.md §2.1 using real WAV files
written to tmp_path. No fakes — soundfile does the decoding, the
validator runs its real code paths.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import soundfile as sf
from larynx_shared.paths import DatasetPaths
from larynx_training_worker.dataset_prep import (
    PhaseAReport,
    validate_dataset_phase_a,
)

SR = 16_000


def _write_wav(path: pathlib.Path, duration_s: float, channels: int = 1, peak: float = 0.3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-peak, peak, int(SR * duration_s), dtype=np.float32)
    if channels == 2:
        samples = np.stack([samples, samples], axis=1)
    elif channels > 2:
        samples = np.stack([samples] * channels, axis=1)
    sf.write(path, samples, SR, subtype="PCM_16")


def _write_transcripts(dp: DatasetPaths, rows: list[dict[str, str]]) -> None:
    with dp.transcripts_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _make_dataset(tmp_path: pathlib.Path, duration_s: float = 10.0) -> DatasetPaths:
    dp = DatasetPaths(data_dir=tmp_path, dataset_id="ds-test")
    dp.audio_dir.mkdir(parents=True)
    return dp


def test_happy_path(tmp_path: pathlib.Path) -> None:
    # 30 x 11s = 330s > 300s minimum.
    dp = _make_dataset(tmp_path)
    rows = []
    for i in range(30):
        path = dp.audio_dir / f"clip{i:02d}.wav"
        _write_wav(path, duration_s=11.0)
        rows.append({"audio": str(path), "text": f"sample text {i}"})
    _write_transcripts(dp, rows)

    report = validate_dataset_phase_a(dp)
    assert isinstance(report, PhaseAReport)
    assert report.ok is True
    assert report.issues == []
    assert report.total_duration_s >= 300
    assert report.num_clips == 30
    assert report.sample_rates == {SR: 30}


def test_too_short_total_duration(tmp_path: pathlib.Path) -> None:
    dp = _make_dataset(tmp_path)
    rows = []
    for i in range(5):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=10.0)  # 50s total < 300s
        rows.append({"audio": str(path), "text": f"line {i}"})
    _write_transcripts(dp, rows)

    report = validate_dataset_phase_a(dp)
    assert report.ok is False
    codes = {i.code for i in report.issues}
    assert "duration_too_short" in codes


def test_too_short_custom_minimum(tmp_path: pathlib.Path) -> None:
    # Explicit min_seconds=600 rejects a 400s dataset that would pass default.
    dp = _make_dataset(tmp_path)
    rows = []
    for i in range(40):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=10.0)
        rows.append({"audio": str(path), "text": f"line {i}"})
    _write_transcripts(dp, rows)

    report = validate_dataset_phase_a(dp, min_seconds=600)
    assert report.ok is False
    assert any(i.code == "duration_too_short" for i in report.issues)


def test_rejects_silent_file(tmp_path: pathlib.Path) -> None:
    dp = _make_dataset(tmp_path)
    # 30 real clips + 1 silent one.
    rows = []
    for i in range(30):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=11.0)
        rows.append({"audio": str(path), "text": f"line {i}"})
    silent = dp.audio_dir / "silent.wav"
    _write_wav(silent, duration_s=10.0, peak=0.0)
    rows.append({"audio": str(silent), "text": "silence"})
    _write_transcripts(dp, rows)

    report = validate_dataset_phase_a(dp)
    assert report.ok is False
    codes = {i.code for i in report.issues}
    assert "silent_audio" in codes
    assert any("silent.wav" in i.detail for i in report.issues if i.code == "silent_audio")


def test_rejects_multichannel_file(tmp_path: pathlib.Path) -> None:
    dp = _make_dataset(tmp_path)
    rows = []
    for i in range(30):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=11.0)
        rows.append({"audio": str(path), "text": f"line {i}"})
    # 5.1 surround — should be rejected.
    surround = dp.audio_dir / "surround.wav"
    _write_wav(surround, duration_s=10.0, channels=6)
    rows.append({"audio": str(surround), "text": "five point one"})
    _write_transcripts(dp, rows)

    report = validate_dataset_phase_a(dp)
    assert report.ok is False
    assert any(i.code == "too_many_channels" for i in report.issues)


def test_allows_stereo(tmp_path: pathlib.Path) -> None:
    # 2 channels is OK; the dataset loader down-mixes at training time.
    dp = _make_dataset(tmp_path)
    rows = []
    for i in range(30):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=11.0, channels=2)
        rows.append({"audio": str(path), "text": f"line {i}"})
    _write_transcripts(dp, rows)

    report = validate_dataset_phase_a(dp)
    assert report.ok is True


def test_transcripts_missing_entry_reported(tmp_path: pathlib.Path) -> None:
    dp = _make_dataset(tmp_path)
    rows = []
    for i in range(30):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=11.0)
        if i != 5:  # clip05.wav has no transcript line
            rows.append({"audio": str(path), "text": f"line {i}"})
    _write_transcripts(dp, rows)

    report = validate_dataset_phase_a(dp)
    assert report.ok is False
    codes = {i.code for i in report.issues}
    assert "audio_missing_transcript" in codes


def test_transcripts_extra_manifest_row_reported(tmp_path: pathlib.Path) -> None:
    dp = _make_dataset(tmp_path)
    rows = []
    for i in range(30):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=11.0)
        rows.append({"audio": str(path), "text": f"line {i}"})
    # An extra row pointing at a file that doesn't exist on disk.
    rows.append({"audio": str(dp.audio_dir / "ghost.wav"), "text": "this file does not exist"})
    _write_transcripts(dp, rows)

    report = validate_dataset_phase_a(dp)
    assert report.ok is False
    codes = {i.code for i in report.issues}
    assert "transcript_missing_audio" in codes


def test_transcripts_empty_text_reported(tmp_path: pathlib.Path) -> None:
    dp = _make_dataset(tmp_path)
    rows = []
    for i in range(30):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=11.0)
        text = "   " if i == 5 else f"line {i}"
        rows.append({"audio": str(path), "text": text})
    _write_transcripts(dp, rows)

    report = validate_dataset_phase_a(dp)
    assert report.ok is False
    assert any(i.code == "empty_transcript_text" for i in report.issues)


def test_missing_transcripts_file_is_ok_for_auto_transcription(
    tmp_path: pathlib.Path,
) -> None:
    # Phase A's job is structural only; auto-transcription runs later in
    # PREPARING. A dataset uploaded without transcripts is valid.
    dp = _make_dataset(tmp_path)
    for i in range(30):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=11.0)
    report = validate_dataset_phase_a(dp)
    assert report.ok is True
    assert report.num_clips == 30


def test_unsupported_suffix_rejected(tmp_path: pathlib.Path) -> None:
    # A .txt in the audio/ dir counts as a structure violation — caller
    # probably dropped the wrong kind of file.
    dp = _make_dataset(tmp_path)
    for i in range(30):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=11.0)
    (dp.audio_dir / "notes.txt").write_text("stray")

    report = validate_dataset_phase_a(dp)
    assert report.ok is False
    assert any(i.code == "unsupported_file" for i in report.issues)


def test_report_is_serialisable(tmp_path: pathlib.Path) -> None:
    dp = _make_dataset(tmp_path)
    for i in range(30):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=11.0)
    report = validate_dataset_phase_a(dp)
    # Used by routes/finetune.py to include the issue list in a 400 body.
    blob = report.model_dump_json()
    roundtrip = PhaseAReport.model_validate_json(blob)
    assert roundtrip == report


def test_issue_includes_file_path(tmp_path: pathlib.Path) -> None:
    dp = _make_dataset(tmp_path)
    for i in range(30):
        path = dp.audio_dir / f"clip{i}.wav"
        _write_wav(path, duration_s=11.0)
    silent = dp.audio_dir / "silent.wav"
    _write_wav(silent, duration_s=10.0, peak=0.0)

    report = validate_dataset_phase_a(dp)
    issue = next(i for i in report.issues if i.code == "silent_audio")
    # The UI needs the file path so the user knows which clip is bad.
    assert str(silent) in issue.detail
