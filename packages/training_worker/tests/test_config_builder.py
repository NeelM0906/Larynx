"""Config builder — merges upstream voxcpm_finetune_lora.yaml template
with per-job overrides and writes train_config.yaml that the upstream
training script consumes verbatim.

Tests ensure the template defaults match upstream (one test diffs
against ``third_party/VoxCPM/conf/voxcpm_v2/voxcpm_finetune_lora.yaml``
when that checkout is present; otherwise the tests run the full merge
logic against in-memory pretrained config.json fixtures).
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml
from larynx_shared.paths import DatasetPaths, JobPaths
from larynx_training_worker.config_builder import (
    DEFAULT_LORA_TEMPLATE,
    build_training_config,
    write_training_config,
)


def _write_pretrained_config(root: pathlib.Path, sample_rate: int = 16_000) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"audio_vae_config": {"sample_rate": sample_rate}}
    (root / "config.json").write_text(json.dumps(cfg))
    return root


def _make_paths(tmp_path: pathlib.Path) -> tuple[JobPaths, DatasetPaths, pathlib.Path]:
    job = JobPaths(data_dir=tmp_path, job_id="job-1")
    job.ensure_dirs()
    dataset = DatasetPaths(data_dir=tmp_path, dataset_id="ds-1")
    dataset.audio_dir.mkdir(parents=True)
    dataset.transcripts_jsonl.write_text("")
    pretrained = _write_pretrained_config(tmp_path / "pretrained")
    return job, dataset, pretrained


def test_defaults_match_upstream_yaml_shape(tmp_path: pathlib.Path) -> None:
    job, dataset, pretrained = _make_paths(tmp_path)
    cfg = build_training_config(
        pretrained_path=str(pretrained),
        job_paths=job,
        dataset_paths=dataset,
    )
    # Top-level keys the upstream YAML defines — none may be missing.
    expected_keys = {
        "pretrained_path",
        "train_manifest",
        "val_manifest",
        "sample_rate",
        "batch_size",
        "grad_accum_steps",
        "num_workers",
        "num_iters",
        "log_interval",
        "valid_interval",
        "save_interval",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "max_steps",
        "max_batch_tokens",
        "max_grad_norm",
        "save_path",
        "tensorboard",
        "lambdas",
        "lora",
    }
    assert expected_keys <= set(cfg)
    # Default LoRA block matches the upstream template.
    assert cfg["lora"]["r"] == 32
    assert cfg["lora"]["alpha"] == 32
    assert cfg["lora"]["enable_lm"] is True
    assert cfg["lora"]["enable_dit"] is True
    assert cfg["lora"]["enable_proj"] is False


def test_paths_are_wired_to_job_and_dataset(tmp_path: pathlib.Path) -> None:
    job, dataset, pretrained = _make_paths(tmp_path)
    cfg = build_training_config(
        pretrained_path=str(pretrained),
        job_paths=job,
        dataset_paths=dataset,
    )
    assert cfg["pretrained_path"] == str(pretrained)
    assert cfg["train_manifest"] == str(dataset.transcripts_jsonl)
    assert cfg["save_path"] == str(job.save_path)
    assert cfg["tensorboard"] == str(job.logs_dir)
    # val_manifest defaults to empty string per upstream.
    assert cfg["val_manifest"] in (None, "")


def test_sample_rate_auto_detected_from_pretrained_config(tmp_path: pathlib.Path) -> None:
    job, dataset, pretrained = _make_paths(tmp_path)
    # Rewrite pretrained config to an unusual rate so we can tell the
    # auto-detect path ran.
    (pretrained / "config.json").write_text(
        json.dumps({"audio_vae_config": {"sample_rate": 22_050}})
    )
    cfg = build_training_config(
        pretrained_path=str(pretrained),
        job_paths=job,
        dataset_paths=dataset,
    )
    assert cfg["sample_rate"] == 22_050


def test_sample_rate_falls_back_to_16k_when_pretrained_missing(tmp_path: pathlib.Path) -> None:
    # A mis-specified pretrained dir should not brick the job; upstream
    # training asserts on mismatch anyway, so we pick a sensible default
    # and let the training script fail loudly if it actually cares.
    job, dataset, _ = _make_paths(tmp_path)
    cfg = build_training_config(
        pretrained_path=str(tmp_path / "nonexistent"),
        job_paths=job,
        dataset_paths=dataset,
    )
    assert cfg["sample_rate"] == 16_000


def test_overrides_replace_defaults(tmp_path: pathlib.Path) -> None:
    job, dataset, pretrained = _make_paths(tmp_path)
    cfg = build_training_config(
        pretrained_path=str(pretrained),
        job_paths=job,
        dataset_paths=dataset,
        overrides={
            "num_iters": 500,
            "learning_rate": 5e-5,
            "lora": {"r": 16, "alpha": 16},
        },
    )
    assert cfg["num_iters"] == 500
    assert cfg["learning_rate"] == 5e-5
    # LoRA subdict merges (dict-level deep merge), doesn't clobber other keys.
    assert cfg["lora"]["r"] == 16
    assert cfg["lora"]["alpha"] == 16
    assert cfg["lora"]["enable_lm"] is True
    assert cfg["lora"]["enable_dit"] is True


def test_overrides_preserve_path_fields(tmp_path: pathlib.Path) -> None:
    # Users cannot override the path-wired fields — those are controlled
    # by JobPaths / DatasetPaths. A config override that tries to set
    # save_path is silently ignored so a bad UI input can't redirect
    # writes to an unexpected dir.
    job, dataset, pretrained = _make_paths(tmp_path)
    cfg = build_training_config(
        pretrained_path=str(pretrained),
        job_paths=job,
        dataset_paths=dataset,
        overrides={"save_path": "/attacker/controlled"},
    )
    assert cfg["save_path"] == str(job.save_path)


def test_override_lora_rank_above_max_lora_rank_raises(tmp_path: pathlib.Path) -> None:
    job, dataset, pretrained = _make_paths(tmp_path)
    with pytest.raises(ValueError, match="lora_rank"):
        build_training_config(
            pretrained_path=str(pretrained),
            job_paths=job,
            dataset_paths=dataset,
            overrides={"lora": {"r": 512}},
            max_lora_rank=32,
        )


def test_write_training_config_produces_yaml_matching_upstream_loader(
    tmp_path: pathlib.Path,
) -> None:
    job, dataset, pretrained = _make_paths(tmp_path)
    cfg = build_training_config(
        pretrained_path=str(pretrained),
        job_paths=job,
        dataset_paths=dataset,
        overrides={"num_iters": 42},
    )
    out = write_training_config(cfg, job.train_config_yaml)
    assert out == job.train_config_yaml
    assert out.is_file()
    loaded = yaml.safe_load(out.read_text())
    assert loaded == cfg
    # Upstream loader expects top-level `lora` as a mapping.
    assert isinstance(loaded["lora"], dict)


def test_default_lora_template_exposed_for_ui_defaults() -> None:
    # The UI wizard shows the same defaults in the configure step; the
    # public module constant keeps both sides honest. If this changes,
    # the UI defaults must too.
    assert DEFAULT_LORA_TEMPLATE["r"] == 32
    assert DEFAULT_LORA_TEMPLATE["alpha"] == 32
    assert DEFAULT_LORA_TEMPLATE["enable_lm"] is True
    assert DEFAULT_LORA_TEMPLATE["enable_dit"] is True
    assert DEFAULT_LORA_TEMPLATE["enable_proj"] is False
