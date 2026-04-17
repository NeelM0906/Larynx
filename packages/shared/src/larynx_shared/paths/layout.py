"""On-disk layout under ``${DATA_DIR}``.

The layout this module encodes::

    ${DATA_DIR}/
      datasets/
        {dataset_id}/
          audio/              # *.wav, *.flac, *.mp3
          transcripts.jsonl   # upstream HF-dataset manifest shape
          validation_report.json
        {dataset_id}.staging/ # upload-in-progress (renamed to final on success)
      finetune_jobs/
        {job_id}/
          train_config.yaml   # built from upstream template + overrides
          checkpoints/
            step_XXXXXXX/...
            latest/
              lora_weights.safetensors
              lora_config.json
          logs/               # train.log + TensorBoard
          subprocess.pid      # for the orphan reaper
      lora_weights/
        {voice_id}/           # copied here at REGISTERING
          lora_weights.safetensors
          lora_config.json
        _orphaned/
          {ts}/               # name-conflict / cancellation debris

Only the path *shapes* live here. File-producing logic (e.g. the
training script's own output naming) is upstream; this module just
agrees with it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Audio extensions librosa / soundfile open without extra deps. Anything
# else we reject at dataset-upload time in Phase A (see ORCHESTRATION-M7
# §2.1). Keeping the set in one place so validator + helper can agree.
SUPPORTED_AUDIO_SUFFIXES: frozenset[str] = frozenset({".wav", ".flac", ".mp3"})


def _as_path(p: Path | str) -> Path:
    return p if isinstance(p, Path) else Path(p)


@dataclass(frozen=True)
class DatasetPaths:
    """Path composer for ``${DATA_DIR}/datasets/{dataset_id}/``."""

    data_dir: Path
    dataset_id: str

    def __init__(self, data_dir: Path | str, dataset_id: str) -> None:
        # Frozen dataclass + a custom __init__: coerce data_dir to Path
        # so callers don't have to. ``object.__setattr__`` is the
        # documented escape hatch for frozen dataclasses.
        object.__setattr__(self, "data_dir", _as_path(data_dir))
        object.__setattr__(self, "dataset_id", dataset_id)

    @property
    def base_dir(self) -> Path:
        return self.data_dir / "datasets" / self.dataset_id

    @property
    def staging_dir(self) -> Path:
        # Uploads land here first; the gateway renames staging -> base on
        # Phase A success so a failed upload never leaves a half-
        # populated ``base_dir`` visible to the training_worker.
        return self.data_dir / "datasets" / f"{self.dataset_id}.staging"

    @property
    def audio_dir(self) -> Path:
        return self.base_dir / "audio"

    @property
    def transcripts_jsonl(self) -> Path:
        return self.base_dir / "transcripts.jsonl"

    @property
    def validation_report_json(self) -> Path:
        return self.base_dir / "validation_report.json"

    def audio_files(self) -> Iterator[Path]:
        """Yield audio files in ``audio/`` in stable (sorted) order.

        Only files with a supported suffix are returned. Non-audio
        artifacts (READMEs, stray ``.DS_Store``) are skipped silently.
        """
        if not self.audio_dir.is_dir():
            return iter(())

        def _iter() -> Iterator[Path]:
            for path in sorted(self.audio_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
                    yield path

        return _iter()

    def has_transcripts(self) -> bool:
        return self.transcripts_jsonl.is_file()


@dataclass(frozen=True)
class JobPaths:
    """Path composer for ``${DATA_DIR}/finetune_jobs/{job_id}/``."""

    data_dir: Path
    job_id: str

    def __init__(self, data_dir: Path | str, job_id: str) -> None:
        object.__setattr__(self, "data_dir", _as_path(data_dir))
        object.__setattr__(self, "job_id", job_id)

    @property
    def root(self) -> Path:
        return self.data_dir / "finetune_jobs" / self.job_id

    @property
    def train_config_yaml(self) -> Path:
        return self.root / "train_config.yaml"

    @property
    def save_path(self) -> Path:
        # Upstream's train_voxcpm_finetune.py writes step_XXXXXXX/ and
        # latest/ under this directory. We pass it in as ``save_path``
        # in train_config.yaml.
        return self.root / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def subprocess_pid(self) -> Path:
        return self.root / "subprocess.pid"

    @property
    def latest_checkpoint_dir(self) -> Path:
        return self.save_path / "latest"

    @property
    def latest_lora_weights(self) -> Path:
        return self.latest_checkpoint_dir / "lora_weights.safetensors"

    @property
    def latest_lora_config(self) -> Path:
        return self.latest_checkpoint_dir / "lora_config.json"

    def ensure_dirs(self) -> None:
        """Create ``root``, ``save_path``, and ``logs_dir`` if missing.

        The training subprocess will make its own per-step folders, so we
        don't pre-create those.
        """
        for d in (self.root, self.save_path, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


def lora_weights_dir(data_dir: Path | str, voice_id: str) -> Path:
    """Canonical location of a registered LoRA voice's weights.

    Addressed by ``voice_id``, not ``job_id``, so deletion is one
    ``rmtree`` call and there's no back-reference into the job that
    produced them (jobs may be pruned independently).
    """
    return _as_path(data_dir) / "lora_weights" / voice_id
