"""Tests for the VAD+Punctuation worker's Prometheus metrics sidecar.

Mirror of the Fun-ASR sidecar tests — same two invariants (port bind,
larynx_ counter in body after observation), same ephemeral-port
strategy so CI runs in parallel with a real :9102 worker.
"""

from __future__ import annotations

import asyncio
import socket
import urllib.request

from larynx_vad_punc_worker.metrics import (
    REGISTRY,
    record_error,
    record_request,
)
from larynx_vad_punc_worker.metrics_server import MetricsSidecar


def _free_port() -> int:
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
        record_request(method="segment", duration_s=0.008, status_class="ok")
        record_error(method="segment", error_code="segment_failed")

        status, content_type, body = await _fetch(f"http://127.0.0.1:{port}/metrics")
        assert status == 200
        assert "text/plain" in content_type
        text = body.decode("utf-8")
        assert "larynx_request_duration_seconds" in text
        assert "larynx_error_total" in text
    finally:
        await sidecar.stop()


async def test_sidecar_stop_is_idempotent() -> None:
    sidecar = MetricsSidecar(host="127.0.0.1", port=_free_port())
    await sidecar.stop()  # never started — should be a no-op
    await sidecar.start()
    await sidecar.stop()
    await sidecar.stop()


async def test_sidecar_refuses_double_start() -> None:
    port = _free_port()
    sidecar = MetricsSidecar(host="127.0.0.1", port=port)
    await sidecar.start()
    try:
        await sidecar.start()  # no-op, not an error
        status, _, _ = await _fetch(f"http://127.0.0.1:{port}/metrics")
        assert status == 200
    finally:
        await sidecar.stop()


def test_metric_definitions_registered() -> None:
    names = {family.name for family in REGISTRY.collect() if family.name.startswith("larynx_")}
    assert "larynx_request_duration_seconds" in names
    # prometheus_client strips the ``_total`` suffix from Counter family
    # names (exposition re-adds it).
    assert "larynx_error" in names
