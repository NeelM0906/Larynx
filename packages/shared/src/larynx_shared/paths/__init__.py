"""Typed on-disk paths for M7 fine-tuning artifacts.

The gateway, training_worker, and the upstream training script all write
or read files under the same root (``${DATA_DIR}``). Expressing the
layout as typed helpers here keeps path composition in one place —
renaming a file is a single-file edit instead of a grep-and-hope.

See ORCHESTRATION-M7.md §8.2.
"""

from larynx_shared.paths.layout import (
    DatasetPaths,
    JobPaths,
    lora_weights_dir,
)

__all__ = [
    "DatasetPaths",
    "JobPaths",
    "lora_weights_dir",
]
