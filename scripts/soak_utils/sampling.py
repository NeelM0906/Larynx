"""Resource samplers used by the soak harness.

Three collectors:

- :func:`sample_processes` -- RSS / CPU% per larynx-* process on the
  host, plus an overall system snapshot (``name="__system__"``).
- :func:`sample_gpus` -- parses ``nvidia-smi`` output. Returns an empty
  list (not an exception) on hosts without the tool so CI boxes can
  still dry-run the harness.
- :func:`sample_disk` -- ``shutil.disk_usage`` for ``--data-dir``.

Each sampler returns a list of :class:`SampleRow` tuples compatible
with :mod:`scripts.soak_utils.report` and the parquet writer in
``soak_test.py``. The common row schema is ``(timestamp, metric_name,
labels_json, value)``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

try:  # psutil is a hard dep of soak-test but we want a graceful import error.
    import psutil
except ImportError:  # pragma: no cover - importerror surfaced by the CLI.
    psutil = None  # type: ignore[assignment]


DEFAULT_PROCESS_PATTERN = re.compile(r"larynx|voxcpm|funasr|vad_punc|training_worker", re.I)


@dataclass(frozen=True)
class SampleRow:
    timestamp: float
    metric: str
    labels: dict[str, str]
    value: float

    def as_parquet_row(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "metric": self.metric,
            "labels": json.dumps(self.labels, sort_keys=True),
            "value": float(self.value),
        }


def sample_processes(
    *,
    now: float | None = None,
    pattern: re.Pattern[str] = DEFAULT_PROCESS_PATTERN,
) -> list[SampleRow]:
    """Return RSS + CPU% samples for every matching process.

    We prime ``cpu_percent`` on the first iteration by calling it with
    ``None`` -- psutil returns 0.0 that first time but primes the
    internal tick counter so subsequent samples are meaningful.
    """

    if psutil is None:
        return []
    ts = now if now is not None else time.time()
    rows: list[SampleRow] = []
    seen_pids: set[int] = set()

    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            name = proc.info.get("name") or ""
            cmdline = " ".join(proc.info.get("cmdline") or [])
            blob = f"{name} {cmdline}"
            if not pattern.search(blob):
                continue
            with proc.oneshot():
                rss = proc.memory_info().rss
                # cpu_percent(None) returns delta since last call. We
                # call unconditionally so subsequent samples of the
                # same pid get real numbers; the first call will be 0.
                cpu = proc.cpu_percent(interval=None)
            seen_pids.add(proc.pid)
            labels = {"pid": str(proc.pid), "name": name}
            rows.append(SampleRow(ts, "process_rss_bytes", labels, float(rss)))
            rows.append(SampleRow(ts, "process_cpu_percent", labels, float(cpu)))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:  # noqa: BLE001 - process sampling is best-effort.
            continue

    # System-wide snapshot for context.
    try:
        vm = psutil.virtual_memory()
        rows.append(SampleRow(ts, "system_mem_used_bytes", {"name": "__system__"}, float(vm.used)))
        rows.append(
            SampleRow(
                ts,
                "system_cpu_percent",
                {"name": "__system__"},
                float(psutil.cpu_percent(interval=None)),
            )
        )
    except Exception:  # noqa: BLE001
        pass
    return rows


_GPU_CMD = (
    "nvidia-smi",
    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
)


def sample_gpus(*, now: float | None = None, timeout_s: float = 5.0) -> list[SampleRow]:
    """Shell out to ``nvidia-smi`` once; return GPU rows.

    On any subprocess failure (no nvidia-smi, timeout, non-zero exit)
    returns an empty list -- the caller decides how to log it.
    """

    ts = now if now is not None else time.time()
    try:
        proc = subprocess.run(
            _GPU_CMD,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    rows: list[SampleRow] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx, util, mem_used, mem_total, temp = parts[:5]
            labels = {"gpu": idx}
            rows.append(SampleRow(ts, "gpu_utilization_percent", labels, float(util)))
            rows.append(SampleRow(ts, "gpu_memory_used_mib", labels, float(mem_used)))
            rows.append(SampleRow(ts, "gpu_memory_total_mib", labels, float(mem_total)))
            rows.append(SampleRow(ts, "gpu_temperature_c", labels, float(temp)))
        except ValueError:
            continue
    return rows


def sample_disk(path: str, *, now: float | None = None) -> list[SampleRow]:
    ts = now if now is not None else time.time()
    try:
        usage = shutil.disk_usage(path)
    except (OSError, FileNotFoundError):
        return []
    labels = {"path": path}
    return [
        SampleRow(ts, "disk_total_bytes", labels, float(usage.total)),
        SampleRow(ts, "disk_used_bytes", labels, float(usage.used)),
        SampleRow(ts, "disk_free_bytes", labels, float(usage.free)),
    ]


def flatten(batches: Iterable[list[SampleRow]]) -> list[SampleRow]:
    out: list[SampleRow] = []
    for batch in batches:
        out.extend(batch)
    return out


__all__ = [
    "DEFAULT_PROCESS_PATTERN",
    "SampleRow",
    "flatten",
    "sample_disk",
    "sample_gpus",
    "sample_processes",
]
