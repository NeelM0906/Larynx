"""Build ``train_config.yaml`` for the upstream training script.

The template + defaults are copied from
``third_party/VoxCPM/conf/voxcpm_v2/voxcpm_finetune_lora.yaml`` @ tag
2.0.2. Keeping them in code (rather than reading from the third_party
checkout at runtime) means our build doesn't depend on the third_party
tree being present and the M7 design doc's §0 "never import from
third_party" rule holds without an exception.

See ORCHESTRATION-M7.md §6.1 and §10 for the values and the
auto-detected sample-rate rule.
"""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any

import yaml
from larynx_shared.paths import DatasetPaths, JobPaths

# ---------------------------------------------------------------------------
# Defaults copied from upstream third_party/VoxCPM/conf/voxcpm_v2/
# voxcpm_finetune_lora.yaml @ tag 2.0.2. The upstream YAML is the source
# of truth; anything that diverges here is a bug.
# ---------------------------------------------------------------------------

DEFAULT_LORA_TEMPLATE: dict[str, Any] = {
    "enable_lm": True,
    "enable_dit": True,
    "enable_proj": False,
    "r": 32,
    "alpha": 32,
    "dropout": 0.0,
    # target_modules_* are not in the upstream YAML (they use the library
    # default), but making them explicit here lets the M7 voxcpm_worker
    # init config (see §3.2) stay a superset no matter what rank/alpha
    # the user picks.
    "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "target_modules_dit": ["q_proj", "v_proj", "k_proj", "o_proj"],
}

_BASE_TEMPLATE: dict[str, Any] = {
    "pretrained_path": "",  # filled by caller
    "train_manifest": "",  # filled from dataset_paths.transcripts_jsonl
    "val_manifest": "",  # no val split in v1
    "sample_rate": 16_000,  # overridden if pretrained config.json has one
    "batch_size": 2,
    "grad_accum_steps": 8,
    "num_workers": 8,
    "num_iters": 1000,
    "log_interval": 10,
    "valid_interval": 500,
    "save_interval": 500,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "max_steps": 1000,
    "max_batch_tokens": 8192,
    "max_grad_norm": 1.0,
    "save_path": "",  # filled from job_paths.save_path
    "tensorboard": "",  # filled from job_paths.logs_dir
    "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
    # ``lora`` is deep-merged with the DEFAULT_LORA_TEMPLATE so partial
    # overrides (e.g. just r + alpha) don't drop the target_modules_*.
    "lora": dict(DEFAULT_LORA_TEMPLATE),
}

# Field names the caller cannot override — these are controlled by the
# JobPaths / DatasetPaths contracts so a bad UI input can't redirect
# writes or point at a different dataset.
_PATH_FIELDS: frozenset[str] = frozenset({"pretrained_path", "train_manifest", "save_path"})


def build_training_config(
    *,
    pretrained_path: str,
    job_paths: JobPaths,
    dataset_paths: DatasetPaths,
    overrides: dict[str, Any] | None = None,
    max_lora_rank: int = 32,
) -> dict[str, Any]:
    """Return the YAML-ready config dict for the upstream training script.

    ``overrides`` is merged on top of the defaults. ``lora`` is merged
    key-by-key; all other fields are replaced outright. Fields in
    :data:`_PATH_FIELDS` are re-set from ``job_paths`` / ``dataset_paths``
    after the merge so a user override can't escape the sandbox.

    ``max_lora_rank`` must be ≥ the resolved LoRA rank; otherwise a
    ``ValueError`` is raised. Matches ORCHESTRATION-M7.md §3.2 — a LoRA
    trained at rank > ``max_lora_rank`` will not load into the worker.
    """
    config = copy.deepcopy(_BASE_TEMPLATE)

    # Path wiring first — overrides may NOT override these (see below).
    config["pretrained_path"] = str(pretrained_path)
    config["train_manifest"] = str(dataset_paths.transcripts_jsonl)
    config["save_path"] = str(job_paths.save_path)
    config["tensorboard"] = str(job_paths.logs_dir)

    # Auto-detect sample rate from the pretrained config's audio_vae_config
    # (see third_party/VoxCPM/lora_ft_webui.detect_sample_rate for the
    # upstream reference). Falls back to 16 kHz on read/parse failures so
    # the job still reaches the training script where it'll error loudly
    # if the rate actually matters.
    detected_sr = _read_pretrained_sample_rate(pretrained_path)
    if detected_sr is not None:
        config["sample_rate"] = detected_sr

    if overrides:
        for key, value in overrides.items():
            if key in _PATH_FIELDS:
                continue  # silently ignored
            if key == "lora" and isinstance(value, dict):
                merged = dict(config["lora"])
                merged.update(value)
                config["lora"] = merged
            else:
                config[key] = value

    rank = int(config["lora"].get("r", DEFAULT_LORA_TEMPLATE["r"]))
    if rank > max_lora_rank:
        raise ValueError(
            f"lora_rank {rank} exceeds max_lora_rank {max_lora_rank}; either "
            f"pick a smaller rank or set LARYNX_VOXCPM_LORA_MAX_RANK higher "
            f"and restart voxcpm_worker."
        )

    return config


def write_training_config(config: dict[str, Any], path: pathlib.Path) -> pathlib.Path:
    """Serialise ``config`` to ``path`` as YAML.

    Returns the path for convenience (so the caller can inline it into
    the subprocess ``--config_path`` argv).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return path


def _read_pretrained_sample_rate(pretrained_path: str) -> int | None:
    cfg_path = pathlib.Path(pretrained_path) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        blob = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    vae = blob.get("audio_vae_config")
    if not isinstance(vae, dict):
        return None
    sr = vae.get("sample_rate")
    try:
        return int(sr) if sr is not None else None
    except (TypeError, ValueError):
        return None
