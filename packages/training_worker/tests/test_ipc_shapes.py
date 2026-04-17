"""Training IPC message round-trip through pydantic + JSON.

Keeps gateway <-> training_worker wire shapes honest: if a field
renames or a type loosens, one of these asserts fails before the
runtime does.
"""

from __future__ import annotations

from larynx_shared.ipc import (
    TrainDoneFrame,
    TrainLogChunk,
    TrainLoraRequest,
    TrainStateChunk,
)


def test_train_lora_request_defaults_and_json() -> None:
    req = TrainLoraRequest(
        job_id="j1",
        dataset_id="ds1",
        voice_name="nimbus",
    )
    assert req.kind == "train_lora"
    assert req.validate_transcripts is True
    assert req.config_overrides == {}
    # Round-trip JSON.
    parsed = TrainLoraRequest.model_validate_json(req.model_dump_json())
    assert parsed == req


def test_train_lora_request_accepts_overrides() -> None:
    req = TrainLoraRequest(
        job_id="j1",
        dataset_id="ds1",
        voice_name="nimbus",
        config_overrides={"lora": {"r": 16, "alpha": 16}, "num_iters": 2000},
        validate_transcripts=False,
    )
    assert req.config_overrides["num_iters"] == 2000
    assert req.config_overrides["lora"] == {"r": 16, "alpha": 16}


def test_train_log_chunk_preserves_line() -> None:
    msg = TrainLogChunk(request_id="j1", line="step=10 loss/diff=0.25")
    parsed = TrainLogChunk.model_validate_json(msg.model_dump_json())
    assert parsed.line == "step=10 loss/diff=0.25"


def test_train_state_chunk_optional_metrics() -> None:
    msg = TrainStateChunk(request_id="j1", step=10, loss_diff=0.25, lr=1e-4)
    assert msg.loss_stop is None
    assert msg.epoch is None
    parsed = TrainStateChunk.model_validate_json(msg.model_dump_json())
    assert parsed == msg


def test_train_done_frame_succeeded_shape() -> None:
    frame = TrainDoneFrame(request_id="j1", state="SUCCEEDED", voice_id="v-42")
    assert frame.error_code is None
    assert frame.error_detail is None
    parsed = TrainDoneFrame.model_validate_json(frame.model_dump_json())
    assert parsed == frame


def test_train_done_frame_failed_shape() -> None:
    frame = TrainDoneFrame(
        request_id="j1",
        state="FAILED",
        error_code="nonzero_exit",
        error_detail="upstream script returned 137",
    )
    assert frame.voice_id is None
    assert frame.error_code == "nonzero_exit"
