"""Runs the upstream training script as a subprocess and tails its output.

One function, :func:`run_training_subprocess`, owns the full
TRAINING-state lifecycle from §6 of ORCHESTRATION-M7.md:

- spawn via ``asyncio.create_subprocess_exec`` in a new process group
  so we can SIGKILL the whole tree on hard cancel
- write subprocess.pid so the orphan reaper can find the process after
  a gateway crash
- tail stdout (stderr merged) line-by-line: every line gets passed to
  ``on_log``; parsed tracker lines also get forwarded to ``on_state``
- watch for cancel / wall-timeout concurrently with proc.wait
- on graceful shutdown: SIGTERM, wait ``cancel_grace_seconds``, SIGKILL
  the process group if still alive
- classify the outcome (SUCCESS / MISSING_ARTIFACT / EARLY_EXIT /
  NONZERO_EXIT / CANCELLED / WALL_TIMEOUT) so the caller can map it
  to a FineTuneJob error_code without re-parsing logs
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
import signal
import sys
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import structlog
from larynx_shared.paths import JobPaths

log = structlog.get_logger(__name__)

# Tracker-line regex. Matches the upstream TrainingTracker.log_metrics
# output shape (see third_party/VoxCPM/src/voxcpm/training/tracker.py).
# We don't care about exact ordering — every metric is captured by name.
_TRACKER_FIELD = re.compile(r"(\w[\w/]*)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_HAS_STEP = re.compile(r"(?:^|\s)step\s*=\s*\d+")

# Mapping from upstream field name → our TrainStateChunk field.
_FIELD_MAP: dict[str, str] = {
    "step": "step",
    "loss/diff": "loss_diff",
    "loss/stop": "loss_stop",
    "lr": "lr",
    "epoch": "epoch",
}


# Bound the spawn primitive once at import. Keeps call sites readable
# and avoids any per-call attribute lookups in the hot path.
_spawn_subprocess = asyncio.create_subprocess_exec


class RunnerOutcome(StrEnum):
    """Classification the caller maps to FineTuneJob.error_code."""

    SUCCESS = "success"
    NONZERO_EXIT = "nonzero_exit"
    MISSING_ARTIFACT = "missing_artifact"
    BAD_LORA_CONFIG = "bad_lora_config"
    EARLY_EXIT = "early_exit"
    CANCELLED = "cancelled"
    WALL_TIMEOUT = "wall_timeout"


def parse_training_event(line: str) -> dict[str, Any] | None:
    """Extract a dict of state fields from one upstream log line.

    Returns ``None`` if the line doesn't look like a tracker row (no
    ``step=N`` anywhere in it). Only fields we care about make it into
    the dict; unknown keys are dropped.
    """
    if not line or not _HAS_STEP.search(line):
        return None
    out: dict[str, Any] = {}
    for key, raw in _TRACKER_FIELD.findall(line):
        if key not in _FIELD_MAP:
            continue
        try:
            value: float | int = int(raw) if key == "step" else float(raw)
        except ValueError:
            continue
        out[_FIELD_MAP[key]] = value
    return out if "step" in out else None


async def run_training_subprocess(
    *,
    script_path: pathlib.Path,
    job_paths: JobPaths,
    on_log: Callable[[str], None],
    on_state: Callable[[dict[str, Any]], None],
    max_steps: int,
    cancel_event: asyncio.Event,
    wall_timeout_seconds: int = 86_400,
    cancel_grace_seconds: int = 30,
    extra_env: dict[str, str] | None = None,
) -> RunnerOutcome:
    """Spawn the training script and drive it to a terminal outcome.

    ``on_log`` is called once per stdout line (synchronous; must be
    fast — the caller is responsible for any Redis / IPC work).
    ``on_state`` is called once per parseable tracker line.

    The script is launched with:
    - Python interpreter = ``sys.executable`` (so the gpu extra is on
      sys.path for the subprocess).
    - ``stdout=PIPE``, ``stderr=STDOUT`` so we get one ordered stream.
    - ``start_new_session=True`` so SIGKILL via ``os.killpg`` reaches
      DataLoader worker forks.
    - ``CUDA_VISIBLE_DEVICES=0`` and ``TOKENIZERS_PARALLELISM=false``
      merged into the env.
    """
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["TOKENIZERS_PARALLELISM"] = "false"

    # PYTHONPATH scope: the subprocess — not our code — imports from
    # third_party/VoxCPM's ``src/voxcpm/`` package. We inject the src
    # dir on the subprocess env only, so the gateway process stays
    # clean of upstream voxcpm imports (ORCHESTRATION-M7.md §0).
    # Caller passes ``third_party_voxcpm_src`` via ``extra_env`` when
    # it wants this wiring; we compute a sensible default below.
    voxcpm_src = env.get("LARYNX_VOXCPM_SRC_DIR")
    if voxcpm_src and pathlib.Path(voxcpm_src).is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{voxcpm_src}:{existing}" if existing else voxcpm_src

    if extra_env:
        env.update(extra_env)

    argv = [sys.executable, str(script_path), "--config_path", str(job_paths.train_config_yaml)]

    log.info("training.subprocess.spawn", argv=argv, job=str(job_paths.root))
    proc = await _spawn_subprocess(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )

    try:
        job_paths.subprocess_pid.write_text(str(proc.pid))
    except OSError:
        # Non-fatal — the orphan reaper can still find us via the job
        # state row; we'd just miss the PID fast-path. Keep going.
        log.warning("training.subprocess.pid_write_failed", pid=proc.pid)

    try:
        return await _drive(
            proc=proc,
            on_log=on_log,
            on_state=on_state,
            cancel_event=cancel_event,
            wall_timeout_seconds=wall_timeout_seconds,
            cancel_grace_seconds=cancel_grace_seconds,
            job_paths=job_paths,
            max_steps=max_steps,
        )
    finally:
        try:
            job_paths.subprocess_pid.unlink(missing_ok=True)
        except OSError:
            pass


async def _drive(
    *,
    proc: asyncio.subprocess.Process,
    on_log: Callable[[str], None],
    on_state: Callable[[dict[str, Any]], None],
    cancel_event: asyncio.Event,
    wall_timeout_seconds: int,
    cancel_grace_seconds: int,
    job_paths: JobPaths,
    max_steps: int,
) -> RunnerOutcome:
    assert proc.stdout is not None  # redeclared for mypy

    tail_task = asyncio.create_task(_tail(proc.stdout, on_log, on_state))
    wait_task = asyncio.create_task(proc.wait())
    cancel_task = asyncio.create_task(cancel_event.wait())

    timeout_outcome: RunnerOutcome | None = None
    try:
        done, pending = await asyncio.wait(
            {wait_task, cancel_task},
            timeout=wall_timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            timeout_outcome = RunnerOutcome.WALL_TIMEOUT
        elif cancel_task in done and wait_task not in done:
            timeout_outcome = RunnerOutcome.CANCELLED

        if timeout_outcome is not None:
            await _terminate(proc, cancel_grace_seconds)

        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        # Wait for the tail to drain so on_log/on_state see every line
        # the subprocess emitted before we classify.
        try:
            await asyncio.wait_for(tail_task, timeout=5)
        except TimeoutError:
            tail_task.cancel()
            try:
                await tail_task
            except (asyncio.CancelledError, Exception):
                pass
    except BaseException:
        tail_task.cancel()
        wait_task.cancel()
        cancel_task.cancel()
        raise

    return_code = proc.returncode if proc.returncode is not None else -1
    log.info("training.subprocess.exit", return_code=return_code)

    if timeout_outcome is not None:
        return timeout_outcome

    if return_code != 0:
        return RunnerOutcome.NONZERO_EXIT

    # Success path: artifacts must exist + final step must be near max.
    if not job_paths.latest_lora_weights.is_file():
        return RunnerOutcome.MISSING_ARTIFACT
    if not job_paths.latest_lora_config.is_file():
        return RunnerOutcome.MISSING_ARTIFACT
    try:
        lora_cfg_blob = json.loads(job_paths.latest_lora_config.read_text())
    except Exception:  # noqa: BLE001
        return RunnerOutcome.BAD_LORA_CONFIG
    if not isinstance(lora_cfg_blob, dict) or "lora_config" not in lora_cfg_blob:
        return RunnerOutcome.BAD_LORA_CONFIG

    training_state_path = job_paths.latest_checkpoint_dir / "training_state.json"
    if training_state_path.is_file():
        try:
            final_step = int(json.loads(training_state_path.read_text()).get("step", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            final_step = 0
        if max_steps > 0 and final_step < max_steps * 0.95:
            return RunnerOutcome.EARLY_EXIT

    return RunnerOutcome.SUCCESS


async def _tail(
    stdout: asyncio.StreamReader,
    on_log: Callable[[str], None],
    on_state: Callable[[dict[str, Any]], None],
) -> None:
    """Read stdout line-by-line and fan out to the log + state hooks."""
    while True:
        try:
            raw = await stdout.readline()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("training.subprocess.tail_error", error=str(e))
            return
        if not raw:
            return
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        try:
            on_log(line)
        except Exception:  # noqa: BLE001
            log.exception("training.subprocess.on_log_failed", line=line)
        ev = parse_training_event(line)
        if ev is not None:
            try:
                on_state(ev)
            except Exception:  # noqa: BLE001
                log.exception("training.subprocess.on_state_failed", event=ev)


async def _terminate(proc: asyncio.subprocess.Process, grace_seconds: int) -> None:
    """SIGTERM, wait ``grace_seconds``, SIGKILL the whole process group."""
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
        return
    except TimeoutError:
        pass

    log.warning("training.subprocess.sigkill", pid=proc.pid)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        log.error("training.subprocess.refused_to_die", pid=proc.pid)
