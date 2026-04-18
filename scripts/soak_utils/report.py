"""Turn the soak timeseries parquet into ``SOAK_REPORT.md``.

All math is numpy-only where available so the report can still render
without scipy / pandas installed on the target host. If pyarrow isn't
importable at report time we log a warning and emit a partial report
from whatever in-memory rows the caller hands us.

The verdict gate is:

- No endpoint has an error rate above the configured ceiling
  (default 1%).
- No tracked larynx-* process has an RSS growth slope above 5% of its
  starting value over the run.

If either fails the verdict is ``FAIL`` with the offending metrics
enumerated.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


@dataclass
class EndpointStats:
    route: str
    total: int = 0
    errors: int = 0
    p50: float = float("nan")
    p95: float = float("nan")
    p99: float = float("nan")

    @property
    def error_rate(self) -> float:
        return self.errors / self.total if self.total > 0 else 0.0


@dataclass
class ProcessStats:
    name: str
    pid: str
    rss_start: float = float("nan")
    rss_end: float = float("nan")
    rss_slope_pct: float = float("nan")  # percent-of-start growth over the run
    cpu_mean: float = float("nan")
    cpu_p95: float = float("nan")


@dataclass
class GpuStats:
    gpu: str
    mem_min: float = float("nan")
    mem_max: float = float("nan")
    mem_mean: float = float("nan")
    util_mean: float = float("nan")
    temp_max: float = float("nan")


@dataclass
class ReportInputs:
    """Aggregated inputs to the report.

    Callers (``soak_test.py``) assemble this from their own bookkeeping
    (e.g. success/error counts tracked per traffic stream) plus the
    parquet timeseries.
    """

    started_at: float
    ended_at: float
    endpoints: list[EndpointStats] = field(default_factory=list)
    processes: list[ProcessStats] = field(default_factory=list)
    gpus: list[GpuStats] = field(default_factory=list)
    disk_delta_bytes: int | None = None
    restart_storm_lines: list[str] = field(default_factory=list)
    quality_check: dict[str, Any] | None = None
    errors_jsonl_path: pathlib.Path | None = None
    error_rate_ceiling: float = 0.01
    rss_growth_ceiling_pct: float = 5.0


def compute_process_stats(rows: Iterable[dict[str, Any]]) -> list[ProcessStats]:
    """Fold ``process_rss_bytes`` + ``process_cpu_percent`` rows per pid."""

    if np is None:
        return []
    by_pid: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {"rss": [], "cpu": []}
    )
    names: dict[str, str] = {}
    for r in rows:
        labels = _labels_of(r)
        pid = labels.get("pid")
        if not pid:
            continue
        names.setdefault(pid, labels.get("name", "?"))
        if r["metric"] == "process_rss_bytes":
            by_pid[pid]["rss"].append((float(r["timestamp"]), float(r["value"])))
        elif r["metric"] == "process_cpu_percent":
            by_pid[pid]["cpu"].append((float(r["timestamp"]), float(r["value"])))

    out: list[ProcessStats] = []
    for pid, series in by_pid.items():
        rss = sorted(series["rss"])
        cpu = sorted(series["cpu"])
        stats = ProcessStats(name=names[pid], pid=pid)
        if rss:
            stats.rss_start = rss[0][1]
            stats.rss_end = rss[-1][1]
            stats.rss_slope_pct = _linreg_percent_growth(rss)
        if cpu:
            cpu_vals = np.array([v for _, v in cpu])
            stats.cpu_mean = float(cpu_vals.mean())
            stats.cpu_p95 = float(np.quantile(cpu_vals, 0.95))
        out.append(stats)
    out.sort(key=lambda s: (s.name, s.pid))
    return out


def compute_gpu_stats(rows: Iterable[dict[str, Any]]) -> list[GpuStats]:
    if np is None:
        return []
    by_gpu: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"mem": [], "util": [], "temp": []}
    )
    for r in rows:
        labels = _labels_of(r)
        gpu = labels.get("gpu")
        if gpu is None:
            continue
        if r["metric"] == "gpu_memory_used_mib":
            by_gpu[gpu]["mem"].append(float(r["value"]))
        elif r["metric"] == "gpu_utilization_percent":
            by_gpu[gpu]["util"].append(float(r["value"]))
        elif r["metric"] == "gpu_temperature_c":
            by_gpu[gpu]["temp"].append(float(r["value"]))

    out: list[GpuStats] = []
    for gpu, series in sorted(by_gpu.items()):
        stats = GpuStats(gpu=gpu)
        if series["mem"]:
            arr = np.array(series["mem"])
            stats.mem_min = float(arr.min())
            stats.mem_max = float(arr.max())
            stats.mem_mean = float(arr.mean())
        if series["util"]:
            stats.util_mean = float(np.array(series["util"]).mean())
        if series["temp"]:
            stats.temp_max = float(np.array(series["temp"]).max())
        out.append(stats)
    return out


def _linreg_percent_growth(series: list[tuple[float, float]]) -> float:
    """Fit a line, return slope * duration / start * 100 (percent)."""

    if np is None or len(series) < 2:
        return float("nan")
    xs = np.array([t for t, _ in series])
    ys = np.array([v for _, v in series])
    xs = xs - xs[0]
    if xs[-1] <= 0 or ys[0] <= 0:
        return float("nan")
    slope, _intercept = np.polyfit(xs, ys, 1)
    total_growth = slope * xs[-1]
    return float(total_growth / ys[0] * 100.0)


def _labels_of(row: dict[str, Any]) -> dict[str, str]:
    labels = row.get("labels")
    if isinstance(labels, str):
        try:
            return json.loads(labels)
        except json.JSONDecodeError:
            return {}
    if isinstance(labels, dict):
        return labels
    return {}


def render_report(inputs: ReportInputs) -> str:
    """Emit Markdown.

    The verdict block lives at the bottom so the pass/fail reason is
    easy to grep for (``^Verdict:``).
    """

    dur_s = max(0.0, inputs.ended_at - inputs.started_at)
    lines: list[str] = []
    lines.append("# Larynx 24h Soak Report")
    lines.append("")
    lines.append(f"- Started: `{_iso(inputs.started_at)}`")
    lines.append(f"- Ended:   `{_iso(inputs.ended_at)}`")
    lines.append(f"- Duration: `{_fmt_dur(dur_s)}`")
    if inputs.errors_jsonl_path is not None:
        lines.append(f"- Errors log: `{inputs.errors_jsonl_path}`")
    lines.append("")

    lines.append("## Endpoint traffic")
    lines.append("")
    lines.append("| Route | Total | Errors | Err% | p50 (ms) | p95 (ms) | p99 (ms) |")
    lines.append("|-------|------:|-------:|-----:|---------:|---------:|---------:|")
    for ep in inputs.endpoints:
        lines.append(
            f"| `{ep.route}` | {ep.total} | {ep.errors} | "
            f"{ep.error_rate * 100:.2f}% | {_ms(ep.p50)} | {_ms(ep.p95)} | {_ms(ep.p99)} |"
        )
    lines.append("")

    lines.append("## Process RSS / CPU")
    lines.append("")
    lines.append("| Name | PID | RSS start (MiB) | RSS end (MiB) | Growth % | CPU mean | CPU p95 |")
    lines.append("|------|-----|----------------:|--------------:|---------:|---------:|--------:|")
    for p in inputs.processes:
        lines.append(
            f"| {p.name} | {p.pid} | {_mib(p.rss_start)} | {_mib(p.rss_end)} "
            f"| {_pct(p.rss_slope_pct)} | {_pct(p.cpu_mean)} | {_pct(p.cpu_p95)} |"
        )
    lines.append("")

    lines.append("## GPUs")
    lines.append("")
    if inputs.gpus:
        lines.append("| GPU | VRAM min | VRAM max | VRAM mean | Util mean | Temp max |")
        lines.append("|-----|---------:|---------:|----------:|----------:|---------:|")
        for g in inputs.gpus:
            lines.append(
                f"| {g.gpu} | {_mib(g.mem_min)} | {_mib(g.mem_max)} "
                f"| {_mib(g.mem_mean)} | {_pct(g.util_mean)} | {_pct(g.temp_max)} |"
            )
    else:
        lines.append("_No GPU data (nvidia-smi absent or unresponsive)._")
    lines.append("")

    lines.append("## Disk")
    if inputs.disk_delta_bytes is None:
        lines.append("_No disk delta captured._")
    else:
        lines.append(f"- Delta: `{inputs.disk_delta_bytes:,}` bytes")
    lines.append("")

    lines.append("## Restart storms")
    if inputs.restart_storm_lines:
        lines.append("")
        for line in inputs.restart_storm_lines:
            lines.append(f"- `{line}`")
    else:
        lines.append("")
        lines.append("_None detected._")
    lines.append("")

    lines.append("## Quality degradation check")
    if inputs.quality_check is None:
        lines.append("_Skipped (numpy or pyarrow missing, or no batch outputs)._")
    else:
        qc = inputs.quality_check
        lines.append(
            f"- First-quartile centroid mean: `{qc.get('centroid_q1', float('nan')):.1f} Hz`"
        )
        lines.append(
            f"- Last-quartile centroid mean:  `{qc.get('centroid_q4', float('nan')):.1f} Hz`"
        )
        lines.append(f"- Drift: `{qc.get('drift_pct', float('nan')):.2f}%`")
        verdict = "DEGRADED" if qc.get("degraded") else "OK"
        lines.append(f"- Status: **{verdict}**")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    verdict, reasons = _evaluate(inputs)
    lines.append(f"Verdict: **{verdict}**")
    if reasons:
        for r in reasons:
            lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)


def _evaluate(inputs: ReportInputs) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for ep in inputs.endpoints:
        if ep.error_rate > inputs.error_rate_ceiling:
            reasons.append(
                f"route `{ep.route}` error rate "
                f"{ep.error_rate * 100:.2f}% > {inputs.error_rate_ceiling * 100:.2f}%"
            )
    for p in inputs.processes:
        if math.isfinite(p.rss_slope_pct) and p.rss_slope_pct > inputs.rss_growth_ceiling_pct:
            reasons.append(
                f"{p.name} (pid {p.pid}) RSS grew {p.rss_slope_pct:.2f}% > "
                f"{inputs.rss_growth_ceiling_pct:.2f}%"
            )
    if inputs.restart_storm_lines:
        reasons.append(f"{len(inputs.restart_storm_lines)} restart-storm events detected")
    if inputs.quality_check and inputs.quality_check.get("degraded"):
        reasons.append("spectral-centroid drift exceeded 5%")
    return ("PASS", []) if not reasons else ("FAIL", reasons)


def _iso(ts: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(ts, tz=_dt.UTC).isoformat(timespec="seconds")


def _fmt_dur(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _mib(b: float) -> str:
    if not math.isfinite(b):
        return "-"
    return f"{b / (1024 * 1024):.1f}"


def _ms(s: float) -> str:
    if not math.isfinite(s):
        return "-"
    return f"{s * 1000:.1f}"


def _pct(v: float) -> str:
    if not math.isfinite(v):
        return "-"
    return f"{v:.1f}"


__all__ = [
    "EndpointStats",
    "GpuStats",
    "ProcessStats",
    "ReportInputs",
    "compute_gpu_stats",
    "compute_process_stats",
    "render_report",
]
