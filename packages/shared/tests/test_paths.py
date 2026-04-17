"""Typed-path helpers for M7 on-disk artifacts.

Prevents string-path drift between the gateway, the training_worker, and
the upstream training script. See ORCHESTRATION-M7.md §8.2.
"""

from __future__ import annotations

import pathlib

from larynx_shared.paths import DatasetPaths, JobPaths


def test_dataset_paths_layout(tmp_path: pathlib.Path) -> None:
    dp = DatasetPaths(data_dir=tmp_path, dataset_id="ds-42")
    assert dp.base_dir == tmp_path / "datasets" / "ds-42"
    assert dp.staging_dir == tmp_path / "datasets" / "ds-42.staging"
    assert dp.audio_dir == tmp_path / "datasets" / "ds-42" / "audio"
    assert dp.transcripts_jsonl == tmp_path / "datasets" / "ds-42" / "transcripts.jsonl"
    assert dp.validation_report_json == tmp_path / "datasets" / "ds-42" / "validation_report.json"


def test_dataset_paths_audio_files_empty(tmp_path: pathlib.Path) -> None:
    dp = DatasetPaths(data_dir=tmp_path, dataset_id="ds-42")
    dp.base_dir.mkdir(parents=True)
    dp.audio_dir.mkdir()
    assert list(dp.audio_files()) == []


def test_dataset_paths_audio_files_sorted(tmp_path: pathlib.Path) -> None:
    dp = DatasetPaths(data_dir=tmp_path, dataset_id="ds-42")
    dp.audio_dir.mkdir(parents=True)
    for name in ("c.wav", "a.wav", "b.flac", "ignore.txt"):
        (dp.audio_dir / name).write_bytes(b"")
    assert [p.name for p in dp.audio_files()] == ["a.wav", "b.flac", "c.wav"]


def test_dataset_paths_has_transcripts(tmp_path: pathlib.Path) -> None:
    dp = DatasetPaths(data_dir=tmp_path, dataset_id="ds-42")
    dp.base_dir.mkdir(parents=True)
    assert not dp.has_transcripts()
    dp.transcripts_jsonl.write_text("{}\n")
    assert dp.has_transcripts()


def test_job_paths_layout(tmp_path: pathlib.Path) -> None:
    jp = JobPaths(data_dir=tmp_path, job_id="job-7")
    assert jp.root == tmp_path / "finetune_jobs" / "job-7"
    assert jp.train_config_yaml == tmp_path / "finetune_jobs" / "job-7" / "train_config.yaml"
    assert jp.save_path == tmp_path / "finetune_jobs" / "job-7" / "checkpoints"
    assert jp.logs_dir == tmp_path / "finetune_jobs" / "job-7" / "logs"
    assert jp.subprocess_pid == tmp_path / "finetune_jobs" / "job-7" / "subprocess.pid"


def test_job_paths_latest_checkpoint(tmp_path: pathlib.Path) -> None:
    jp = JobPaths(data_dir=tmp_path, job_id="job-7")
    # Upstream training script writes the final checkpoint to
    # <save_path>/latest/lora_weights.safetensors (+ lora_config.json).
    assert (
        jp.latest_checkpoint_dir == tmp_path / "finetune_jobs" / "job-7" / "checkpoints" / "latest"
    )
    assert (
        jp.latest_lora_weights
        == tmp_path
        / "finetune_jobs"
        / "job-7"
        / "checkpoints"
        / "latest"
        / "lora_weights.safetensors"
    )
    assert (
        jp.latest_lora_config
        == tmp_path / "finetune_jobs" / "job-7" / "checkpoints" / "latest" / "lora_config.json"
    )


def test_job_paths_ensure_dirs_creates_all(tmp_path: pathlib.Path) -> None:
    jp = JobPaths(data_dir=tmp_path, job_id="job-7")
    jp.ensure_dirs()
    assert jp.root.is_dir()
    assert jp.save_path.is_dir()
    assert jp.logs_dir.is_dir()


def test_lora_weights_paths_layout(tmp_path: pathlib.Path) -> None:
    # After a successful job the runner copies the LoRA checkpoint out of
    # the job-scoped save_path and into the global lora_weights tree,
    # keyed by voice_id (not job_id) so voice deletion cleanup is a
    # straightforward rmtree of one directory.
    from larynx_shared.paths import lora_weights_dir

    assert lora_weights_dir(tmp_path, "voice-abc") == tmp_path / "lora_weights" / "voice-abc"


def test_paths_accept_string_data_dir(tmp_path: pathlib.Path) -> None:
    # Convenience: callers often pass settings.data_dir as a str. Accept
    # and normalise.
    dp = DatasetPaths(data_dir=str(tmp_path), dataset_id="ds-42")
    assert dp.base_dir == tmp_path / "datasets" / "ds-42"
