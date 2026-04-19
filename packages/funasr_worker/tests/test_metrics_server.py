"""Tests for the Fun-ASR worker's Prometheus metrics sidecar.

Covers the two invariants the §3.1 design asks for:
1. Sidecar binds the configured port and responds on ``/metrics``.
2. After a simulated observation, the response body contains at
   least one ``larynx_`` counter line.

Uses an ephemeral port (bound-then-released) so CI can run the test
without colliding with a real :9101 worker.
"""

from __future__ import annotations

import asyncio
import socket
import urllib.request

from larynx_funasr_worker.metrics import (
    REGISTRY,
    record_error,
    record_request,
    record_rtfx,
)
from larynx_funasr_worker.metrics_server import MetricsSidecar


def _free_port() -> int:
    """Return an ephemeral port the OS just assigned and closed."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


async def _fetch(url: str, timeout: float = 2.0) -> tuple[int, str, bytes]:
    def _blocking() -> tuple[int, str, bytes]:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()

    return await asyncio.to_thread(_blocking)


async def test_sidecar_serves_recorded_counter() -> None:
    port = _free_port()
    sidecar = MetricsSidecar(host="127.0.0.1", port=port)
    await sidecar.start()
    try:
        record_request(method="transcribe", duration_s=0.042, status_class="ok")
        record_error(method="transcribe", error_code="transcribe_failed")
        record_rtfx(model="nano", audio_seconds=1.0, wall_clock_seconds=0.05)

        status, content_type, body = await _fetch(f"http://127.0.0.1:{port}/metrics")
        assert status == 200
        assert "text/plain" in content_type
        text = body.decode("utf-8")
        assert "larynx_request_duration_seconds" in text
        assert "larynx_error_total" in text
        assert "larynx_stt_rtfx" in text
    finally:
        await sidecar.stop()


async def test_sidecar_stop_is_idempotent() -> None:
    sidecar = MetricsSidecar(host="127.0.0.1", port=_free_port())
    await sidecar.stop()  # never started — should be a no-op
    await sidecar.start()
    await sidecar.stop()
    await sidecar.stop()  # stopping twice should also be safe


async def test_sidecar_refuses_double_start() -> None:
    port = _free_port()
    sidecar = MetricsSidecar(host="127.0.0.1", port=port)
    await sidecar.start()
    try:
        # Second start() should be a no-op (runner already set) — not raise.
        await sidecar.start()
        status, _, _ = await _fetch(f"http://127.0.0.1:{port}/metrics")
        assert status == 200
    finally:
        await sidecar.stop()


def test_record_rtfx_ignores_zero_wall_clock() -> None:
    # Regression: record_rtfx shouldn't divide by zero when a mocked
    # backend returns instantly.
    before = _count_rtfx_samples()
    record_rtfx(model="nano", audio_seconds=1.0, wall_clock_seconds=0.0)
    record_rtfx(model="nano", audio_seconds=1.0, wall_clock_seconds=-0.1)
    assert _count_rtfx_samples() == before


def _count_rtfx_samples() -> int:
    # Sum of _count children across all label sets on STT_RTFX.
    total = 0
    for family in REGISTRY.collect():
        if family.name != "larynx_stt_rtfx":
            continue
        for sample in family.samples:
            if sample.name == "larynx_stt_rtfx_count":
                total += int(sample.value)
    return total


def test_metric_definitions_registered() -> None:
    # Prove the metrics exist on the private registry so the sidecar
    # has something to serve even before any observation. prometheus_client
    # strips the ``_total`` suffix from Counter family names, so the
    # Counter shows up as ``larynx_error`` — the exposition renders it
    # back to ``larynx_error_total``.
    names = {family.name for family in REGISTRY.collect() if family.name.startswith("larynx_")}
    assert "larynx_request_duration_seconds" in names
    assert "larynx_error" in names
    assert "larynx_stt_rtfx" in names
