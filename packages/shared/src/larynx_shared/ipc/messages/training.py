"""LoRA fine-tune IPC types (see ORCHESTRATION-M7.md §5.3).

One ``TrainLoraRequest`` kicks off a fine-tune. The worker then streams
``TrainLogChunk`` (one per stdout line), ``TrainStateChunk`` (when a tracker
step line is parseable), and ends with exactly one ``TrainDoneFrame``. A
``CancelStreamRequest`` mid-flight triggers §1.1 cancellation and the
``TrainDoneFrame`` reports state=CANCELLED.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from larynx_shared.ipc.messages.base import (
    RequestMessage,
    StreamChunk,
    StreamEnd,
)


class TrainLoraRequest(RequestMessage):
    kind: Literal["train_lora"] = "train_lora"
    job_id: str
    dataset_id: str
    voice_name: str
    # Arbitrary overrides merged onto the upstream voxcpm_finetune_lora.yaml
    # template — rank, alpha, num_iters, learning_rate, etc. Unknown keys
    # pass through unmodified so the gateway doesn't have to mirror every
    # upstream field. Serialised as JSON over the wire.
    config_overrides: dict[str, object] = Field(default_factory=dict)
    # Explicit opt-out of Phase-B transcript quality check. Default True;
    # see ORCHESTRATION-M7.md §2.2.
    validate_transcripts: bool = True


class TrainLogChunk(StreamChunk):
    kind: Literal["train_log"] = "train_log"
    line: str


class TrainStateChunk(StreamChunk):
    """Structured progress extracted from an upstream tracker line.

    Emitted opportunistically — not every training line parses to a
    state event. Callers should treat a gap between events as "no new
    progress", not "training stuck".
    """

    kind: Literal["train_state"] = "train_state"
    step: int
    loss_diff: float | None = None
    loss_stop: float | None = None
    lr: float | None = None
    epoch: float | None = None


class TrainDoneFrame(StreamEnd):
    kind: Literal["train_done"] = "train_done"
    # "SUCCEEDED" | "FAILED" | "CANCELLED". String rather than Literal so
    # adding a state here doesn't cascade into a mypy update everywhere.
    state: str
    voice_id: str | None = None
    error_code: str | None = None
    error_detail: str | None = None


__all__ = [
    "TrainDoneFrame",
    "TrainLogChunk",
    "TrainLoraRequest",
    "TrainStateChunk",
]
