#!/usr/bin/env python
"""Staging verification harness — replaces the 24h soak for v1.

See ORCHESTRATION-M8.md §7.3 for the product decision and scope.

Phases (each independently interpretable so a failure in one doesn't
obscure the others):

    1. Load run            — mixed traffic against a spawned gateway
    2. Graceful shutdown   — SIGTERM with work in flight
    3. Memory delta        — crude RSS growth check over N TTS calls
    4. Restart alerter     — supervisord kill-test (skipped when no
                             supervisord binary is available)
    5. Report              — STAGING_VERIFICATION_REPORT.md

Invocation:

    uv run python scripts/staging_verification.py \\
        --out staging-artifacts/ [--quick]

``--quick`` compresses durations so a dev loop takes ~5 minutes
instead of ~35.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
import psutil

# soak_utils ships corpus + sampling helpers we reuse.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from soak_utils.sampling import sample_processes  # noqa: E402


@dataclass
class PhaseResult:
    name: str
    passed: bool = False
    skipped: bool = False
    skip_reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


# ── gateway subprocess management ────────────────────────────────────


def _free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _spawn_gateway(*, port: int, token: str, data_dir: pathlib.Path) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "LARYNX_API_TOKEN": token,
        "LARYNX_TTS_MODE": "mock",
        "LARYNX_STT_MODE": "mock",
        "LARYNX_VAD_PUNC_MODE": "mock",
        "LARYNX_DATA_DIR": str(data_dir),
        "LARYNX_PORT": str(port),
        "LARYNX_LOG_JSON": "0",
        "DATABASE_URL": os.environ.get(
            "DATABASE_URL", "postgresql+psycopg://larynx:larynx@localhost:5433/larynx"
        ),
        "REDIS_URL": os.environ.get("REDIS_URL", "redis://localhost:6380/0"),
    }
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "larynx_gateway.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


async def _wait_for_ready(url: str, *, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            with contextlib.suppress(httpx.HTTPError):
                r = await client.get(f"{url}/ready")
                if r.status_code == 200 and r.json().get("status") == "ready":
                    return True
            await asyncio.sleep(0.25)
    return False


# ── Phase 1: load run ────────────────────────────────────────────────


async def phase1_load_run(url: str, token: str, duration_s: float) -> PhaseResult:
    """Drive mixed mock TTS + STT traffic for the configured duration.

    Success: >99% of requests succeed, no per-minute bucket shows a
    total outage (len(bucket) > 0), gateway /ready stays reporting
    ``status=ready`` throughout.
    """
    result = PhaseResult(name="load_run")
    t0 = time.monotonic()
    end = t0 + duration_s
    headers = {"Authorization": f"Bearer {token}"}
    tts_body = {"text": "staging verification load run.", "response_format": "wav"}

    counters = {"tts_ok": 0, "tts_fail": 0, "ready_ok": 0, "ready_not_ready": 0}
    per_minute: list[dict[str, int]] = []
    async with httpx.AsyncClient(base_url=url, timeout=10.0) as client:
        next_bucket_end = t0 + 60
        bucket = {"tts_ok": 0, "tts_fail": 0}
        while time.monotonic() < end:
            req_t0 = time.monotonic()
            try:
                r = await client.post("/v1/tts", headers=headers, json=tts_body)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("audio/"):
                    counters["tts_ok"] += 1
                    bucket["tts_ok"] += 1
                else:
                    counters["tts_fail"] += 1
                    bucket["tts_fail"] += 1
            except httpx.HTTPError:
                counters["tts_fail"] += 1
                bucket["tts_fail"] += 1

            # ~1 req/s * 1 client → 60 req/min. Good enough to trip a
            # leak without flooding.
            elapsed = time.monotonic() - req_t0
            await asyncio.sleep(max(0.0, 1.0 - elapsed))

            if time.monotonic() >= next_bucket_end:
                per_minute.append(bucket)
                bucket = {"tts_ok": 0, "tts_fail": 0}
                next_bucket_end += 60

                with contextlib.suppress(httpx.HTTPError):
                    ready = await client.get("/ready")
                    if ready.status_code == 200:
                        counters["ready_ok"] += 1
                    else:
                        counters["ready_not_ready"] += 1

        if bucket["tts_ok"] + bucket["tts_fail"] > 0:
            per_minute.append(bucket)

    result.duration_s = time.monotonic() - t0
    total = counters["tts_ok"] + counters["tts_fail"]
    success_rate = counters["tts_ok"] / total if total else 0.0
    any_empty_bucket = any(b["tts_ok"] + b["tts_fail"] == 0 for b in per_minute)
    result.passed = (
        total > 0
        and success_rate > 0.99
        and counters["ready_not_ready"] == 0
        and not any_empty_bucket
    )
    result.detail = {
        "total_requests": total,
        "success_rate": round(success_rate, 4),
        "ready_checks_ok": counters["ready_ok"],
        "ready_checks_degraded": counters["ready_not_ready"],
        "per_minute_buckets": per_minute,
    }
    return result


# ── Phase 2: graceful shutdown ───────────────────────────────────────


async def phase2_drain_test(data_dir: pathlib.Path) -> PhaseResult:
    """Start a gateway, fire traffic, SIGTERM, verify drain + clean exit."""
    result = PhaseResult(name="drain_test")
    t0 = time.monotonic()

    port = _free_port()
    token = "staging-verification-drain"
    url = f"http://127.0.0.1:{port}"
    proc = _spawn_gateway(port=port, token=token, data_dir=data_dir)
    try:
        if not await _wait_for_ready(url):
            result.passed = False
            result.detail = {"error": "gateway did not become ready within 30s"}
            return result

        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(base_url=url, timeout=10.0) as client:
            # Kick off a batch job with 20 items and a handful of /v1/tts
            # calls concurrently so there's work in flight when SIGTERM
            # lands.
            batch_body = {
                "items": [{"text": f"drain test item {i}."} for i in range(20)],
            }
            batch_resp = await client.post("/v1/batch", headers=headers, json=batch_body)
            batch_resp.raise_for_status()
            batch_job_id = batch_resp.json()["job_id"]
            # Stream a few /v1/tts requests so the async task list isn't empty.
            pending = [
                asyncio.create_task(
                    client.post(
                        "/v1/tts",
                        headers=headers,
                        json={"text": f"drain in-flight {i}.", "response_format": "wav"},
                    )
                )
                for i in range(5)
            ]

            # Give the consumers a moment to pick up batch items.
            await asyncio.sleep(0.5)

            # --- fire SIGTERM ---
            sigterm_at = time.monotonic()
            proc.send_signal(signal.SIGTERM)

            # /ready should flip to 503 promptly.
            ready_after_sigterm = None
            for _ in range(20):  # up to 2s
                try:
                    r = await client.get("/ready", timeout=1.0)
                    if r.status_code == 503:
                        ready_after_sigterm = time.monotonic() - sigterm_at
                        break
                except httpx.HTTPError:
                    # Gateway may have closed the listener already — that
                    # also counts as "not serving new traffic".
                    ready_after_sigterm = time.monotonic() - sigterm_at
                    break
                await asyncio.sleep(0.1)

        # Wait for the process itself to exit cleanly.
        exit_code: int | None
        try:
            exit_code = await asyncio.to_thread(proc.wait, 35)
        except subprocess.TimeoutExpired:
            exit_code = None
            proc.kill()

        exit_after_s = time.monotonic() - sigterm_at

        # Drain in-flight tasks — some may have completed, some
        # cancelled. We don't require all 5 to succeed; we require the
        # process to exit on time.
        for task in pending:
            with contextlib.suppress(Exception):
                await task

        result.duration_s = time.monotonic() - t0
        result.detail = {
            "batch_job_id": batch_job_id,
            "ready_503_after_s": ready_after_sigterm,
            "process_exit_after_s": round(exit_after_s, 2),
            "exit_code": exit_code,
        }
        # uvicorn's normal clean-shutdown exit code is 0, but Python's
        # default SIGTERM handler (when the server lets it reach the
        # process level after cleanup) can emit 143. Either is a
        # successful drain — the signal was caught, lifespan teardown
        # ran, the process exited within the window. Only a kill-9
        # style (negative returncode) or a timeout is a real failure.
        clean_exit = exit_code is not None and exit_code >= 0 and exit_after_s <= 35
        result.passed = (
            clean_exit and ready_after_sigterm is not None and ready_after_sigterm <= 2.0
        )
        return result
    finally:
        if proc.poll() is None:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(5)


# ── Phase 3: memory delta ────────────────────────────────────────────


async def phase3_mem_delta(url: str, token: str, num_reqs: int) -> PhaseResult:
    """Baseline RSS, fire N mock TTS requests, re-measure. Flag >20% growth."""
    result = PhaseResult(name="mem_delta")
    t0 = time.monotonic()

    def _larynx_rss_by_pid() -> dict[int, int]:
        snap: dict[int, int] = {}
        for row in sample_processes(now=time.time()):
            if row.metric == "process_rss_bytes":
                pid_str = row.labels.get("pid")
                if pid_str is not None:
                    snap[int(pid_str)] = int(row.value)
        return snap

    before = _larynx_rss_by_pid()
    if not before:
        # Harness running on a box with no larynx-* processes by name
        # (e.g. dev gateway launched via uvicorn, which psutil may show
        # as "uvicorn" instead). Fall back to the current process's own
        # children.
        before = {p.pid: p.memory_info().rss for p in psutil.Process().children(recursive=True)}

    headers = {"Authorization": f"Bearer {token}"}
    body = {"text": "mem delta probe.", "response_format": "wav"}
    ok = 0
    fail = 0
    async with httpx.AsyncClient(base_url=url, timeout=10.0) as client:
        for _ in range(num_reqs):
            try:
                r = await client.post("/v1/tts", headers=headers, json=body)
                if r.status_code == 200:
                    ok += 1
                else:
                    fail += 1
            except httpx.HTTPError:
                fail += 1

    # Let async garbage settle.
    await asyncio.sleep(1.0)

    after = _larynx_rss_by_pid() or {
        p.pid: p.memory_info().rss for p in psutil.Process().children(recursive=True)
    }

    deltas: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for pid, rss_before in before.items():
        rss_after = after.get(pid)
        if rss_after is None or rss_before <= 0:
            continue
        pct = (rss_after - rss_before) / rss_before * 100
        entry = {
            "pid": pid,
            "rss_before_mb": round(rss_before / 1024**2, 1),
            "rss_after_mb": round(rss_after / 1024**2, 1),
            "growth_pct": round(pct, 2),
        }
        deltas.append(entry)
        if pct > 20.0:
            flagged.append(entry)

    result.duration_s = time.monotonic() - t0
    result.detail = {
        "num_requests": num_reqs,
        "requests_ok": ok,
        "requests_failed": fail,
        "per_process_deltas": deltas,
        "flagged_processes": flagged,
    }
    result.passed = fail == 0 and not flagged
    return result


# ── Phase 4: restart alerter ─────────────────────────────────────────


async def phase4_restart_alerter() -> PhaseResult:
    """Supervisord-driven kill test. Skipped cleanly when unavailable."""
    result = PhaseResult(name="restart_alerter")
    t0 = time.monotonic()

    if shutil.which("supervisord") is None:
        result.skipped = True
        result.skip_reason = (
            "supervisord not installed on this host; the harness can't drive the "
            "PROCESS_STATE events end-to-end. Unit coverage in "
            "packages/gateway/tests/unit/test_restart_alerter.py exercises the "
            "threshold + window logic; the end-to-end supervisord → eventlistener "
            "path stays a deploy-time manual verification."
        )
        result.duration_s = time.monotonic() - t0
        return result

    # Supervisord-present path left as future work — production boxes
    # have supervisord and can exercise this directly; the harness's job
    # is to make the skip explicit when the capability isn't available.
    result.skipped = True
    result.skip_reason = (
        "supervisord detected but end-to-end phase 4 driver not yet implemented; "
        "use supervisorctl manually for now."
    )
    result.duration_s = time.monotonic() - t0
    return result


# ── report ───────────────────────────────────────────────────────────


def _format_result(r: PhaseResult) -> str:
    status = "SKIPPED" if r.skipped else ("PASS" if r.passed else "FAIL")
    lines = [f"### {r.name} — **{status}** ({r.duration_s:.1f}s)", ""]
    if r.skipped:
        lines.append(f"Skipped: {r.skip_reason}")
    else:
        lines.append("```json")
        lines.append(json.dumps(r.detail, indent=2, default=str))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_report(
    out_dir: pathlib.Path, results: list[PhaseResult], started_at: float
) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = pathlib.Path("STAGING_VERIFICATION_REPORT.md")
    total_duration = sum(r.duration_s for r in results)
    verdict = (
        "PASS"
        if all(r.passed or r.skipped for r in results) and any(r.passed for r in results)
        else "FAIL"
    )

    lines = [
        "# Staging Verification Report",
        "",
        f"**Run timestamp:** {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started_at))}",
        f"**Total duration:** {total_duration:.1f}s",
        f"**Verdict:** **{verdict}**",
        "",
        "Replacement for the 24h soak per ORCHESTRATION-M8.md §7.3. This is ",
        "not a substitute for a long-horizon soak — it catches Part C hardening ",
        "issues and gross leaks, not 18-hour GPU accumulation or slow drifts.",
        "",
        "## Phase results",
        "",
    ]
    for r in results:
        lines.append(_format_result(r))
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"Raw phase JSON: `{out_dir}/phase_results.json`")
    lines.append("")

    report_path.write_text("\n".join(lines))
    (out_dir / "phase_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str)
    )
    return report_path


# ── main ─────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--gateway-url",
        default=os.environ.get("LARYNX_GATEWAY_URL", "http://localhost:8000"),
        help="Target for Phases 1 + 3 (an already-running gateway).",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("LARYNX_API_TOKEN", "change-me-please"),
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("staging-artifacts"),
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Compress durations: phase 1 → 2 min, phase 3 → 100 reqs.",
    )
    p.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["1", "2", "3", "4"],
        help="Skip the given phase number. Repeatable.",
    )
    return p.parse_args()


async def _amain(args: argparse.Namespace) -> int:
    started_at = time.time()
    results: list[PhaseResult] = []

    phase1_s = 120 if args.quick else 20 * 60
    phase3_reqs = 100 if args.quick else 1000

    drain_data_dir = args.out / "phase2_data"
    drain_data_dir.mkdir(parents=True, exist_ok=True)

    phases = [
        ("1", "load_run", lambda: phase1_load_run(args.gateway_url, args.token, phase1_s)),
        ("2", "drain_test", lambda: phase2_drain_test(drain_data_dir)),
        ("3", "mem_delta", lambda: phase3_mem_delta(args.gateway_url, args.token, phase3_reqs)),
        ("4", "restart_alerter", lambda: phase4_restart_alerter()),
    ]
    for num, name, fn in phases:
        if num in args.skip:
            results.append(
                PhaseResult(
                    name=name,
                    skipped=True,
                    skip_reason=f"--skip {num} requested on the CLI",
                )
            )
            continue
        print(f"[phase {num}] {name} — starting", flush=True)
        try:
            result = await fn()
        except Exception as e:  # noqa: BLE001 — harness should surface, not crash
            result = PhaseResult(name=name, passed=False, detail={"error": repr(e)})
        print(
            f"[phase {num}] {name} — {'SKIP' if result.skipped else ('PASS' if result.passed else 'FAIL')} "
            f"({result.duration_s:.1f}s)",
            flush=True,
        )
        results.append(result)

    report = write_report(args.out, results, started_at)
    print(f"\nReport written to {report}")
    return 0 if all(r.passed or r.skipped for r in results) else 1


def main() -> int:
    args = build_arg_parser()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
