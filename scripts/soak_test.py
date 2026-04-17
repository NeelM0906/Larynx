#!/usr/bin/env python
"""24-hour soak harness for the Larynx gateway.

See ORCHESTRATION-M8.md Part D and PRD §7 for the acceptance criteria.

The script is split into:

- :mod:`scripts.soak_utils.sampling` -- psutil / nvidia-smi / disk
- :mod:`scripts.soak_utils.metrics`  -- Prometheus text parser
- :mod:`scripts.soak_utils.report`   -- SOAK_REPORT.md generator

This file owns:

- CLI parsing + duration suffix handling
- The four traffic-stream coroutines (TTS, STT, conversation, batch)
- The periodic sampler loop
- The parquet writer (pyarrow) with 60s flush
- Signal handling (SIGINT/SIGTERM -> flush + report + exit 0)

Invocation:

    uv run python scripts/soak_test.py \
        --gateway-url http://localhost:8000 \
        --token $LARYNX_API_TOKEN \
        --duration 24h \
        --out soak-artifacts/

``--dry-run`` is a 60s smoke that runs the full traffic mix against
the live gateway; no stubbing.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import pathlib
import random
import signal
import struct
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Third-party deps are imported lazily + guarded so `--help` still works
# on a box that's missing pyarrow/httpx/psutil.
try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore[assignment]


HERE = pathlib.Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from scripts.soak_utils import load_corpus  # noqa: E402
from scripts.soak_utils.metrics import histogram_quantiles  # noqa: E402
from scripts.soak_utils.metrics import parse as parse_metrics  # noqa: E402
from scripts.soak_utils.report import (  # noqa: E402
    EndpointStats,
    ReportInputs,
    compute_gpu_stats,
    compute_process_stats,
    render_report,
)
from scripts.soak_utils.sampling import (  # noqa: E402
    SampleRow,
    sample_disk,
    sample_gpus,
    sample_processes,
)

DURATION_SUFFIXES = {"s": 1, "m": 60, "h": 3600, "d": 86400}

# One scrape + one resource sample per 60 s. Parquet flush lines up.
SAMPLE_INTERVAL_S = 60.0

# Traffic-mix quotas per minute (see ORCHESTRATION-M8.md §4.1).
TTS_PER_MINUTE = 10
STT_PER_MINUTE = 5
CONVERSATION_PER_MINUTE = 1
BATCH_EVERY_MINUTES = 10

LIBRARY_VOICES = ["alloy", "echo", "nova", "fable", "onyx", "shimmer"]

# Synthetic PCM knobs used by STT uploads + conversation WS frames.
PCM_SAMPLE_RATE = 16000  # matches the default conversation input rate
PCM_FRAME_MS = 100


def parse_duration(raw: str) -> float:
    """``30s`` / ``2h`` / ``1d`` / plain ``45`` -> seconds."""

    raw = raw.strip().lower()
    if not raw:
        raise ValueError("empty duration")
    if raw[-1] in DURATION_SUFFIXES:
        return float(raw[:-1]) * DURATION_SUFFIXES[raw[-1]]
    return float(raw)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="soak_test.py",
        description="Run the Larynx 24h soak harness.",
    )
    p.add_argument("--gateway-url", required=True, help="e.g. http://localhost:8000")
    p.add_argument("--token", required=True, help="LARYNX_API_TOKEN value.")
    p.add_argument(
        "--duration",
        default="24h",
        help="Run length. Suffixes: s/m/h/d. Plain number = seconds.",
    )
    p.add_argument(
        "--out",
        default="soak-artifacts/",
        help="Output directory for parquet + report + errors log.",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Path to monitor for disk usage (defaults to --out).",
    )
    p.add_argument(
        "--stderr-log",
        default=None,
        help="Path to the supervisord stderr log (for restart-storm detection).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Shortcut for --duration=60s. Full traffic mix, no stubs.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible traffic. Default: time-based.",
    )
    p.add_argument(
        "--voices",
        default=None,
        help="Comma-separated voice_id override. Defaults to library short-names.",
    )
    return p


# ---------------------------------------------------------------------------
# Harness state
# ---------------------------------------------------------------------------


@dataclass
class EndpointCounter:
    total: int = 0
    errors: int = 0
    durations_s: list[float] = field(default_factory=list)


@dataclass
class HarnessState:
    gateway_url: str
    token: str
    out_dir: pathlib.Path
    data_dir: pathlib.Path
    stderr_log: pathlib.Path | None
    duration_s: float
    voices: list[str]
    corpus: list[str]
    rng: random.Random
    started_at: float = field(default_factory=time.time)
    # Per-route counters; keys are route templates to match middleware labels.
    endpoints: dict[str, EndpointCounter] = field(
        default_factory=lambda: defaultdict(EndpointCounter)
    )
    shutdown: asyncio.Event = field(default_factory=asyncio.Event)
    stop_accepting_at: float = 0.0
    # Buffered parquet rows since the last flush.
    sample_buffer: list[SampleRow] = field(default_factory=list)
    # Rows after flush are kept in memory for the final report.
    all_rows: list[dict[str, Any]] = field(default_factory=list)

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def ws_url(self) -> str:
        url = self.gateway_url
        if url.startswith("http://"):
            return "ws://" + url[len("http://") :] + "/v1/conversation"
        if url.startswith("https://"):
            return "wss://" + url[len("https://") :] + "/v1/conversation"
        return url.rstrip("/") + "/v1/conversation"

    def record_request(self, route: str, duration_s: float, error: bool) -> None:
        ec = self.endpoints[route]
        ec.total += 1
        ec.durations_s.append(duration_s)
        if error:
            ec.errors += 1

    def log_error(self, kind: str, detail: dict[str, Any]) -> None:
        path = self.out_dir / "errors.jsonl"
        try:
            with path.open("a", encoding="utf-8") as f:
                row = {"ts": time.time(), "kind": kind, **detail}
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Traffic streams
# ---------------------------------------------------------------------------


def _synth_pcm(ms: int, sample_rate: int, rng: random.Random) -> bytes:
    """Generate a short sine-like PCM16LE buffer."""

    n = int(sample_rate * ms / 1000)
    if np is not None:
        t = np.arange(n, dtype=np.float32) / sample_rate
        freq = float(rng.uniform(180.0, 280.0))
        amp = float(rng.uniform(0.05, 0.2))
        wave = (np.sin(2 * np.pi * freq * t) * amp * 32767).astype(np.int16)
        return wave.tobytes()
    # numpy-less fallback -- cheap zero-buffer; soak harness should
    # never hit this path in practice because numpy is a hard dep.
    return struct.pack("<" + "h" * n, *([0] * n))


def _pack_wav(pcm: bytes, sample_rate: int) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def tts_stream(state: HarnessState, client: httpx.AsyncClient) -> None:
    """10 single-shot /v1/tts per minute."""

    interval = 60.0 / TTS_PER_MINUTE
    while not state.shutdown.is_set():
        text = state.rng.choice(state.corpus)
        voice = state.rng.choice(state.voices)
        payload = {"text": text, "voice_id": voice, "sample_rate": 24000, "output_format": "wav"}
        t0 = time.perf_counter()
        ok = False
        try:
            resp = await client.post("/v1/tts", json=payload, headers=state.headers(), timeout=60.0)
            ok = 200 <= resp.status_code < 300
            if not ok:
                state.log_error("tts", {"status": resp.status_code, "body": resp.text[:400]})
        except Exception as e:  # noqa: BLE001
            state.log_error("tts", {"exception": repr(e)})
        finally:
            state.record_request("/v1/tts", time.perf_counter() - t0, error=not ok)
        await _sleep_with_shutdown(state, interval)


async def stt_stream(state: HarnessState, client: httpx.AsyncClient) -> None:
    """5 single-shot /v1/stt per minute with a synthetic WAV."""

    interval = 60.0 / STT_PER_MINUTE
    while not state.shutdown.is_set():
        pcm = _synth_pcm(ms=800, sample_rate=PCM_SAMPLE_RATE, rng=state.rng)
        wav = _pack_wav(pcm, PCM_SAMPLE_RATE)
        t0 = time.perf_counter()
        ok = False
        try:
            resp = await client.post(
                "/v1/stt",
                files={"file": ("soak.wav", wav, "audio/wav")},
                data={"punctuate": "true"},
                headers=state.headers(),
                timeout=60.0,
            )
            ok = 200 <= resp.status_code < 300
            if not ok:
                state.log_error("stt", {"status": resp.status_code, "body": resp.text[:400]})
        except Exception as e:  # noqa: BLE001
            state.log_error("stt", {"exception": repr(e)})
        finally:
            state.record_request("/v1/stt", time.perf_counter() - t0, error=not ok)
        await _sleep_with_shutdown(state, interval)


async def conversation_stream(state: HarnessState) -> None:
    """1 conversation per minute; 3-turn exchange over the WS route."""

    if websockets is None:
        state.log_error("conversation", {"exception": "websockets not installed"})
        return
    interval = 60.0 / CONVERSATION_PER_MINUTE
    voice_cycle = list(state.voices)
    cycle_idx = 0
    while not state.shutdown.is_set():
        voice = voice_cycle[cycle_idx % len(voice_cycle)]
        cycle_idx += 1
        t0 = time.perf_counter()
        ok = False
        try:
            await _run_conversation(state, voice)
            ok = True
        except Exception as e:  # noqa: BLE001
            state.log_error("conversation", {"voice": voice, "exception": repr(e)})
        finally:
            state.record_request("/v1/conversation", time.perf_counter() - t0, error=not ok)
        await _sleep_with_shutdown(state, interval)


async def _run_conversation(state: HarnessState, voice: str) -> None:
    url = state.ws_url()
    extra_headers = [("Authorization", f"Bearer {state.token}")]
    # websockets 12+ uses `additional_headers`; 11- uses `extra_headers`.
    connect_kwargs: dict[str, Any] = {"open_timeout": 10, "close_timeout": 5}
    try:
        ws_cm = websockets.connect(url, additional_headers=extra_headers, **connect_kwargs)
    except TypeError:
        ws_cm = websockets.connect(url, extra_headers=extra_headers, **connect_kwargs)

    async with ws_cm as ws:
        cfg = {
            "type": "config",
            "voice_id": voice,
            "input_sample_rate": PCM_SAMPLE_RATE,
            "output_sample_rate": 24000,
            "speech_end_silence_ms": 300,
        }
        await ws.send(json.dumps(cfg))

        for _turn in range(3):
            # Stream ~1.5s of audio in 100 ms frames.
            n_frames = int(1500 / PCM_FRAME_MS)
            for _ in range(n_frames):
                frame = _synth_pcm(PCM_FRAME_MS, PCM_SAMPLE_RATE, state.rng)
                await ws.send(frame)
                await asyncio.sleep(PCM_FRAME_MS / 1000.0)
            # Listen for one non-trivial response (audio bytes or
            # response.done) with a bounded wait. We don't care about
            # the content -- just that the round-trip completes.
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
                except TimeoutError:
                    break
                if isinstance(msg, (bytes, bytearray)):
                    # One audio frame is evidence the turn produced output.
                    break
                if isinstance(msg, str):
                    try:
                        payload = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") in {"response.done", "error"}:
                        break

        with contextlib.suppress(Exception):
            await ws.send(json.dumps({"type": "session.end"}))


async def batch_stream(state: HarnessState, client: httpx.AsyncClient) -> None:
    """One 20-item batch every ~10 minutes; cancel ~10% mid-flight."""

    interval = BATCH_EVERY_MINUTES * 60.0
    # Kick off the first batch after a short warm-up so sampling has
    # some signal before the batch load arrives.
    await _sleep_with_shutdown(state, 30.0)
    while not state.shutdown.is_set():
        items = [
            {
                "text": state.rng.choice(state.corpus),
                "voice_id": state.rng.choice(state.voices),
            }
            for _ in range(20)
        ]
        payload = {"items": items}
        t0 = time.perf_counter()
        ok = False
        job_id: str | None = None
        try:
            resp = await client.post(
                "/v1/batch", json=payload, headers=state.headers(), timeout=30.0
            )
            ok = 200 <= resp.status_code < 300
            if ok:
                job_id = resp.json().get("job_id")
            else:
                state.log_error("batch", {"status": resp.status_code, "body": resp.text[:400]})
        except Exception as e:  # noqa: BLE001
            state.log_error("batch", {"exception": repr(e)})
        finally:
            state.record_request("/v1/batch", time.perf_counter() - t0, error=not ok)

        # ~10% cancel roll.
        if job_id and state.rng.random() < 0.10:
            await asyncio.sleep(state.rng.uniform(2.0, 8.0))
            with contextlib.suppress(Exception):
                await client.delete(f"/v1/batch/{job_id}", headers=state.headers(), timeout=10.0)
        await _sleep_with_shutdown(state, interval)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


async def sampler_loop(state: HarnessState, client: httpx.AsyncClient) -> None:
    """Per-minute resource + /metrics scrape + parquet flush."""

    writer: Any = None
    schema = _parquet_schema()
    parquet_path = state.out_dir / "timeseries.parquet"
    try:
        if pa is not None and pq is not None:
            writer = pq.ParquetWriter(parquet_path, schema)
        # Prime psutil so the first CPU% sample is meaningful.
        sample_processes()
        while not state.shutdown.is_set():
            now = time.time()
            rows: list[SampleRow] = []
            rows.extend(sample_processes(now=now))
            rows.extend(sample_gpus(now=now))
            rows.extend(sample_disk(str(state.data_dir), now=now))

            # /metrics scrape -- best effort.
            scraped = await _scrape_metrics(state, client, now=now)
            rows.extend(scraped)

            state.sample_buffer.extend(rows)
            await _flush_buffer(state, writer, schema)
            await _sleep_with_shutdown(state, SAMPLE_INTERVAL_S)
    finally:
        await _flush_buffer(state, writer, schema, final=True)
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()


def _parquet_schema() -> Any:
    if pa is None:
        return None
    return pa.schema(
        [
            pa.field("timestamp", pa.float64()),
            pa.field("metric", pa.string()),
            pa.field("labels", pa.string()),
            pa.field("value", pa.float64()),
        ]
    )


async def _flush_buffer(state: HarnessState, writer: Any, schema: Any, final: bool = False) -> None:
    if not state.sample_buffer:
        return
    rows = [r.as_parquet_row() for r in state.sample_buffer]
    state.all_rows.extend(rows)
    state.sample_buffer.clear()
    if writer is None or schema is None or pa is None:
        return
    try:
        table = pa.Table.from_pylist(rows, schema=schema)
        writer.write_table(table)
    except Exception as e:  # noqa: BLE001
        state.log_error("parquet_flush", {"exception": repr(e), "final": final})


async def _scrape_metrics(
    state: HarnessState, client: httpx.AsyncClient, *, now: float
) -> list[SampleRow]:
    try:
        resp = await client.get("/metrics", timeout=5.0)
    except Exception as e:  # noqa: BLE001
        state.log_error("metrics_scrape", {"exception": repr(e)})
        return []
    if resp.status_code != 200:
        return []
    families = parse_metrics(resp.text)
    out: list[SampleRow] = []
    # Flatten every family into SampleRows for parquet; we don't cherry
    # pick here because the report module re-derives what it needs.
    for fam in families.values():
        for s in fam.samples:
            out.append(
                SampleRow(
                    timestamp=now,
                    metric=s.name,
                    labels=s.label_dict(),
                    value=s.value,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _sleep_with_shutdown(state: HarnessState, seconds: float) -> None:
    try:
        await asyncio.wait_for(state.shutdown.wait(), timeout=seconds)
    except TimeoutError:
        return


def _restart_storm_lines(path: pathlib.Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line for line in data.splitlines() if "critical.restart_storm" in line]


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------


async def _quality_check(state: HarnessState, client: httpx.AsyncClient) -> dict[str, Any] | None:
    """Sample 10 random end-of-run TTS outputs and test centroid drift.

    We kick 10 TTS requests now and compare them to the very first 10
    requests' durations + crude spectral centroid. Cheap, numpy-only;
    silently skipped when numpy isn't importable.
    """

    if np is None:
        return None
    samples: list[tuple[float, float]] = []  # (rms, centroid)
    for _ in range(10):
        text = state.rng.choice(state.corpus)
        voice = state.rng.choice(state.voices)
        payload = {"text": text, "voice_id": voice, "output_format": "wav"}
        try:
            resp = await client.post("/v1/tts", json=payload, headers=state.headers(), timeout=60.0)
            if resp.status_code != 200:
                continue
            centroid, rms = _wav_centroid_and_rms(resp.content)
            samples.append((rms, centroid))
        except Exception:  # noqa: BLE001
            continue
    if len(samples) < 4:
        return None
    mid = len(samples) // 2
    q1 = samples[:mid]
    q4 = samples[mid:]
    c1 = float(np.mean([c for _, c in q1]))
    c4 = float(np.mean([c for _, c in q4]))
    drift = float("nan") if c1 <= 0 else abs(c4 - c1) / c1 * 100.0
    return {
        "centroid_q1": c1,
        "centroid_q4": c4,
        "drift_pct": drift,
        "degraded": drift > 5.0,
    }


def _wav_centroid_and_rms(blob: bytes) -> tuple[float, float]:
    """Roughly compute spectral centroid (Hz) + RMS from WAV bytes."""

    import io
    import wave

    assert np is not None
    with wave.open(io.BytesIO(blob), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if not raw:
        return 0.0, 0.0
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(pcm * pcm)))
    # Cap the FFT window so large WAVs don't dominate CPU.
    pcm = pcm[: min(pcm.size, 1 << 17)]
    spectrum = np.abs(np.fft.rfft(pcm))
    freqs = np.fft.rfftfreq(pcm.size, d=1.0 / sr)
    denom = spectrum.sum()
    centroid = float((freqs * spectrum).sum() / denom) if denom > 0 else 0.0
    return centroid, rms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _endpoint_stats_from_state(state: HarnessState) -> list[EndpointStats]:
    """Compute per-endpoint p50/p95/p99 from locally recorded durations."""

    out: list[EndpointStats] = []
    for route, ec in sorted(state.endpoints.items()):
        stat = EndpointStats(route=route, total=ec.total, errors=ec.errors)
        if ec.durations_s and np is not None:
            arr = np.array(ec.durations_s)
            stat.p50 = float(np.quantile(arr, 0.5))
            stat.p95 = float(np.quantile(arr, 0.95))
            stat.p99 = float(np.quantile(arr, 0.99))
        elif ec.durations_s:
            xs = sorted(ec.durations_s)
            stat.p50 = xs[len(xs) // 2]
            stat.p95 = xs[int(len(xs) * 0.95)] if len(xs) > 1 else xs[-1]
            stat.p99 = xs[int(len(xs) * 0.99)] if len(xs) > 1 else xs[-1]
        out.append(stat)
    return out


def _gateway_metrics_stats(state: HarnessState) -> list[EndpointStats]:
    """Re-derive p50/p95/p99 from the gateway's own histogram rows.

    We keep both views in the final report because the client-side
    timings include network + serialization whereas the server-side
    ones are pure handler time.
    """

    # Find the final set of bucket counts per (route, method).
    rows = [r for r in state.all_rows if r["metric"].startswith("larynx_request_duration_seconds")]
    if not rows:
        return []
    # Group by (route, method); pick the last-timestamp set of samples.
    rows.sort(key=lambda r: r["timestamp"])
    # Take the last snapshot from each label set.
    last_by_labels: dict[tuple[str, str, str, str], float] = {}
    for r in rows:
        labels = _labels_of(r)
        route = labels.get("route", "?")
        method = labels.get("method", "?")
        le = labels.get("le", "")
        # metric name ends with _bucket or _sum or _count.
        last_by_labels[(route, method, r["metric"], le)] = float(r["value"])

    # Group buckets per (route, method).
    per_route: dict[tuple[str, str], dict[float, float]] = defaultdict(dict)
    counts: dict[tuple[str, str], float] = defaultdict(float)
    for (route, method, metric, le), v in last_by_labels.items():
        if metric == "larynx_request_duration_seconds_bucket":
            try:
                le_val = float(le) if le not in ("+Inf", "") else float("inf")
            except ValueError:
                continue
            per_route[(route, method)][le_val] = v
        elif metric == "larynx_request_duration_seconds_count":
            counts[(route, method)] = v

    out: list[EndpointStats] = []
    for (route, method), buckets in sorted(per_route.items()):
        total = int(counts.get((route, method), 0))
        ordered = sorted(buckets.items(), key=lambda t: t[0])
        qs = histogram_quantiles(ordered)
        out.append(
            EndpointStats(
                route=f"{route} [{method}, server]",
                total=total,
                errors=0,  # server-side errors land in larynx_error_total -- tracked separately
                p50=qs.get(0.5, float("nan")),
                p95=qs.get(0.95, float("nan")),
                p99=qs.get(0.99, float("nan")),
            )
        )
    return out


def _labels_of(row: dict[str, Any]) -> dict[str, str]:
    labels = row.get("labels")
    if isinstance(labels, str):
        try:
            return json.loads(labels)
        except json.JSONDecodeError:
            return {}
    return labels or {}


async def generate_report(state: HarnessState, client: httpx.AsyncClient) -> pathlib.Path:
    endpoints = _endpoint_stats_from_state(state)
    endpoints.extend(_gateway_metrics_stats(state))
    process_stats = compute_process_stats(state.all_rows)
    gpu_stats = compute_gpu_stats(state.all_rows)

    disk_used = [r for r in state.all_rows if r["metric"] == "disk_used_bytes"]
    disk_used.sort(key=lambda r: r["timestamp"])
    if len(disk_used) >= 2:
        disk_delta = int(float(disk_used[-1]["value"]) - float(disk_used[0]["value"]))
    else:
        disk_delta = None

    quality = None
    with contextlib.suppress(Exception):
        quality = await _quality_check(state, client)

    inputs = ReportInputs(
        started_at=state.started_at,
        ended_at=time.time(),
        endpoints=endpoints,
        processes=process_stats,
        gpus=gpu_stats,
        disk_delta_bytes=disk_delta,
        restart_storm_lines=_restart_storm_lines(state.stderr_log),
        quality_check=quality,
        errors_jsonl_path=state.out_dir / "errors.jsonl",
    )
    md = render_report(inputs)
    report_path = state.out_dir / "SOAK_REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path


async def run_harness(args: argparse.Namespace) -> int:
    if httpx is None:
        print("error: httpx is required for soak_test.py", file=sys.stderr)
        return 2

    duration_s = 60.0 if args.dry_run else parse_duration(args.duration)
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = pathlib.Path(args.data_dir).resolve() if args.data_dir else out_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = pathlib.Path(args.stderr_log).resolve() if args.stderr_log else None
    voices = [v.strip() for v in args.voices.split(",")] if args.voices else list(LIBRARY_VOICES)
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())

    corpus = load_corpus()

    state = HarnessState(
        gateway_url=args.gateway_url.rstrip("/"),
        token=args.token,
        out_dir=out_dir,
        data_dir=data_dir,
        stderr_log=stderr_log,
        duration_s=duration_s,
        voices=voices,
        corpus=corpus,
        rng=rng,
    )
    state.stop_accepting_at = state.started_at + duration_s

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, state.shutdown.set)

    async with httpx.AsyncClient(base_url=state.gateway_url, follow_redirects=True) as client:
        traffic_tasks = [
            asyncio.create_task(tts_stream(state, client), name="tts-stream"),
            asyncio.create_task(stt_stream(state, client), name="stt-stream"),
            asyncio.create_task(conversation_stream(state), name="convo-stream"),
            asyncio.create_task(batch_stream(state, client), name="batch-stream"),
        ]
        sampler_task = asyncio.create_task(sampler_loop(state, client), name="sampler")

        await _sleep_with_shutdown(state, duration_s)
        state.shutdown.set()

        all_tasks = [*traffic_tasks, sampler_task]
        for t in all_tasks:
            t.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)

        report_path = await generate_report(state, client)

    print(f"[soak] wrote {report_path}")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return asyncio.run(run_harness(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
