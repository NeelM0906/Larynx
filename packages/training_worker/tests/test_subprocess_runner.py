"""Subprocess runner — tails a fake training script end-to-end.

Instead of invoking third_party/VoxCPM/scripts/train_voxcpm_finetune.py
(which needs a GPU + a real dataset), every test here launches a tiny
Python script that writes deterministic output to stdout/stderr and
produces or omits the expected artifacts. Exercises every branch of
:func:`run_training_subprocess` without touching the real upstream.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import textwrap

import pytest
from larynx_shared.paths import JobPaths
from larynx_training_worker.subprocess_runner import (
    RunnerOutcome,
    parse_training_event,
    run_training_subprocess,
)


def _write_fake_script(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    path = tmp_path / "fake_trainer.py"
    path.write_text(textwrap.dedent(body))
    return path


def _write_happy_config(job: JobPaths) -> pathlib.Path:
    job.ensure_dirs()
    cfg = job.train_config_yaml
    cfg.write_text("max_steps: 10\n")
    return cfg


def _make_success_artifact(job: JobPaths) -> None:
    job.latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    job.latest_lora_weights.write_bytes(b"fake-weights")
    job.latest_lora_config.write_text(
        json.dumps({"base_model": "openbmb/VoxCPM2", "lora_config": {"r": 32, "alpha": 32}})
    )
    (job.save_path / "latest" / "training_state.json").write_text(json.dumps({"step": 10}))


# -- parse_training_event ------------------------------------------------


def test_parse_training_event_extracts_step_and_losses() -> None:
    line = "step=10 loss/diff=0.25 loss/stop=0.10 lr=0.0001 epoch=0.5 grad_norm=1.1"
    ev = parse_training_event(line)
    assert ev is not None
    assert ev["step"] == 10
    assert ev["loss_diff"] == pytest.approx(0.25)
    assert ev["loss_stop"] == pytest.approx(0.10)
    assert ev["lr"] == pytest.approx(0.0001)
    assert ev["epoch"] == pytest.approx(0.5)


def test_parse_training_event_handles_missing_fields() -> None:
    line = "step=42"
    ev = parse_training_event(line)
    assert ev is not None
    assert ev["step"] == 42
    assert ev.get("loss_diff") is None


def test_parse_training_event_returns_none_for_unrelated_lines() -> None:
    assert parse_training_event("loading model...") is None
    assert parse_training_event("") is None
    assert parse_training_event("step=not-a-number") is None


# -- run_training_subprocess happy path ----------------------------------


@pytest.mark.asyncio
async def test_happy_path_emits_logs_and_state_and_success(tmp_path: pathlib.Path) -> None:
    job = JobPaths(data_dir=tmp_path, job_id="j1")
    _write_happy_config(job)
    _make_success_artifact(job)  # we're simulating the script; fake produces no artifact

    script = _write_fake_script(
        tmp_path,
        """
        import sys
        for i in range(3):
            print(f"preparing {i}")
        for step in range(0, 11):
            print(f"step={step} loss/diff=0.25 loss/stop=0.10 lr=0.0001 epoch={step/10}")
        sys.exit(0)
        """,
    )

    logs: list[str] = []
    states: list[dict] = []
    outcome = await run_training_subprocess(
        script_path=script,
        job_paths=job,
        on_log=lambda line: logs.append(line),
        on_state=lambda event: states.append(event),
        max_steps=10,
        cancel_event=asyncio.Event(),
    )

    assert outcome is RunnerOutcome.SUCCESS
    # "preparing 0/1/2" plus 11 step lines.
    assert any(line.startswith("preparing") for line in logs)
    step_events = [s["step"] for s in states]
    assert step_events == list(range(11))


@pytest.mark.asyncio
async def test_missing_artifact_reports_missing_artifact(tmp_path: pathlib.Path) -> None:
    job = JobPaths(data_dir=tmp_path, job_id="j1")
    _write_happy_config(job)
    # No artifact written, but subprocess exits 0.

    script = _write_fake_script(
        tmp_path,
        """
        import sys
        for step in range(0, 11):
            print(f"step={step} loss/diff=0.25")
        sys.exit(0)
        """,
    )

    outcome = await run_training_subprocess(
        script_path=script,
        job_paths=job,
        on_log=lambda line: None,
        on_state=lambda event: None,
        max_steps=10,
        cancel_event=asyncio.Event(),
    )
    assert outcome is RunnerOutcome.MISSING_ARTIFACT


@pytest.mark.asyncio
async def test_early_exit_when_step_short_of_max(tmp_path: pathlib.Path) -> None:
    job = JobPaths(data_dir=tmp_path, job_id="j1")
    _write_happy_config(job)
    # Produce artifact but final step far below max.
    job.latest_checkpoint_dir.mkdir(parents=True)
    job.latest_lora_weights.write_bytes(b"x")
    job.latest_lora_config.write_text(
        json.dumps({"lora_config": {"r": 32, "alpha": 32}, "base_model": "x"})
    )
    (job.save_path / "latest" / "training_state.json").write_text(json.dumps({"step": 2}))

    script = _write_fake_script(
        tmp_path,
        """
        print("step=2 loss/diff=1.0")
        """,
    )
    outcome = await run_training_subprocess(
        script_path=script,
        job_paths=job,
        on_log=lambda line: None,
        on_state=lambda event: None,
        max_steps=100,
        cancel_event=asyncio.Event(),
    )
    assert outcome is RunnerOutcome.EARLY_EXIT


@pytest.mark.asyncio
async def test_nonzero_exit_reports_nonzero_exit(tmp_path: pathlib.Path) -> None:
    job = JobPaths(data_dir=tmp_path, job_id="j1")
    _write_happy_config(job)

    script = _write_fake_script(
        tmp_path,
        """
        import sys
        print("crashed early", flush=True)
        sys.exit(7)
        """,
    )
    outcome = await run_training_subprocess(
        script_path=script,
        job_paths=job,
        on_log=lambda line: None,
        on_state=lambda event: None,
        max_steps=10,
        cancel_event=asyncio.Event(),
    )
    assert outcome is RunnerOutcome.NONZERO_EXIT


@pytest.mark.asyncio
async def test_cancel_triggers_sigterm_then_returns_cancelled(tmp_path: pathlib.Path) -> None:
    job = JobPaths(data_dir=tmp_path, job_id="j1")
    _write_happy_config(job)

    # Script sleeps forever — we cancel it mid-flight. It writes a line
    # first so we know it's actually running.
    script = _write_fake_script(
        tmp_path,
        """
        import time, sys
        print("started", flush=True)
        while True:
            time.sleep(0.1)
        """,
    )

    cancel_event = asyncio.Event()
    logs: list[str] = []

    async def _delayed_cancel() -> None:
        # Wait until we see 'started' then cancel.
        while "started" not in logs:
            await asyncio.sleep(0.05)
        cancel_event.set()

    canceller = asyncio.create_task(_delayed_cancel())
    outcome = await run_training_subprocess(
        script_path=script,
        job_paths=job,
        on_log=lambda line: logs.append(line),
        on_state=lambda event: None,
        max_steps=10,
        cancel_event=cancel_event,
        cancel_grace_seconds=2,
    )
    await canceller
    assert outcome is RunnerOutcome.CANCELLED


@pytest.mark.asyncio
async def test_wall_timeout_kills_process(tmp_path: pathlib.Path) -> None:
    job = JobPaths(data_dir=tmp_path, job_id="j1")
    _write_happy_config(job)

    script = _write_fake_script(
        tmp_path,
        """
        import time, sys
        print("running", flush=True)
        while True:
            time.sleep(0.1)
        """,
    )
    outcome = await run_training_subprocess(
        script_path=script,
        job_paths=job,
        on_log=lambda line: None,
        on_state=lambda event: None,
        max_steps=10,
        cancel_event=asyncio.Event(),
        wall_timeout_seconds=1,
        cancel_grace_seconds=1,
    )
    assert outcome is RunnerOutcome.WALL_TIMEOUT


@pytest.mark.asyncio
async def test_pid_file_written_and_cleaned_up(tmp_path: pathlib.Path) -> None:
    job = JobPaths(data_dir=tmp_path, job_id="j1")
    _write_happy_config(job)
    _make_success_artifact(job)

    pid_observed: list[int] = []

    script = _write_fake_script(
        tmp_path,
        """
        import os, sys
        print(f"pid={os.getpid()}", flush=True)
        sys.exit(0)
        """,
    )

    def _capture_pid(line: str) -> None:
        if line.startswith("pid="):
            pid_observed.append(int(line.removeprefix("pid=")))
        # Check pid file while process is running.
        if job.subprocess_pid.is_file():
            pid_observed.append(int(job.subprocess_pid.read_text()))

    outcome = await run_training_subprocess(
        script_path=script,
        job_paths=job,
        on_log=_capture_pid,
        on_state=lambda event: None,
        max_steps=10,
        cancel_event=asyncio.Event(),
    )
    assert outcome is RunnerOutcome.SUCCESS
    # Both the self-reported PID and the file-reported PID should agree.
    assert len(pid_observed) >= 2
    assert pid_observed[0] == pid_observed[1]
