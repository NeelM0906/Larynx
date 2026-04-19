"""VAD+Punctuation worker metric definitions.

Mirror of ``larynx_funasr_worker.metrics`` with a tighter scope —
§7.2's amendment caps vad_punc at baseline request-duration + error
counters. VAD and punctuation are cheap enough that a rtfx histogram
would be noise.

See ORCHESTRATION-M8.md §3.1 + §7.2.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)

REGISTRY = CollectorRegistry()

REQUEST_DURATION = Histogram(
    "larynx_request_duration_seconds",
    "VAD+Punctuation worker request duration, by IPC method and status class.",
    labelnames=("method", "status_class"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

ERROR_TOTAL = Counter(
    "larynx_error_total",
    "VAD+Punctuation worker request errors, by IPC method and error code.",
    labelnames=("method", "error_code"),
    registry=REGISTRY,
)


def record_request(*, method: str, duration_s: float, status_class: str) -> None:
    REQUEST_DURATION.labels(method=method, status_class=status_class).observe(duration_s)


def record_error(*, method: str, error_code: str) -> None:
    ERROR_TOTAL.labels(method=method, error_code=error_code).inc()


def render_latest() -> bytes:
    return generate_latest(REGISTRY)


__all__ = [
    "ERROR_TOTAL",
    "REGISTRY",
    "REQUEST_DURATION",
    "record_error",
    "record_request",
    "render_latest",
]
