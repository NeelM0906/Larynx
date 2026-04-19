"""Fun-ASR worker metric definitions.

Kept separate from :mod:`metrics_server` so the gateway — which imports
:mod:`server` to run the worker in-process — doesn't pull aiohttp as a
transitive. The sidecar HTTP exposition lives in :mod:`metrics_server`
and is only loaded by :func:`larynx_funasr_worker.main.main`.

All counters live on a private :class:`CollectorRegistry`. This keeps
the worker's namespace cleanly separated from the gateway's own
``larynx_request_duration_seconds`` (which has different label
cardinality) when both run in the same Python process during
development and tests. In the standalone-process deployment the two
registries live in different processes and there is no conflict to
worry about.

See ORCHESTRATION-M8.md §3.1 + §7.2.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

_LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)

# rtfx == audio_seconds / wall_clock_seconds. 1.0 is real-time; the
# upstream Fun-ASR-vllm typically runs 20–100× real-time on batched
# utterances. Buckets go up to 200× so the common case isn't all in
# the +Inf overflow.
_RTFX_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)

REGISTRY = CollectorRegistry()

REQUEST_DURATION = Histogram(
    "larynx_request_duration_seconds",
    "Fun-ASR worker request duration, by IPC method and status class.",
    labelnames=("method", "status_class"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

ERROR_TOTAL = Counter(
    "larynx_error_total",
    "Fun-ASR worker request errors, by IPC method and error code.",
    labelnames=("method", "error_code"),
    registry=REGISTRY,
)

STT_RTFX = Histogram(
    "larynx_stt_rtfx",
    "Real-time factor (audio_seconds / wall_clock) for Fun-ASR STT inference.",
    labelnames=("model",),
    buckets=_RTFX_BUCKETS,
    registry=REGISTRY,
)


def record_request(*, method: str, duration_s: float, status_class: str) -> None:
    REQUEST_DURATION.labels(method=method, status_class=status_class).observe(duration_s)


def record_error(*, method: str, error_code: str) -> None:
    ERROR_TOTAL.labels(method=method, error_code=error_code).inc()


def record_rtfx(*, model: str, audio_seconds: float, wall_clock_seconds: float) -> None:
    if wall_clock_seconds <= 0 or audio_seconds < 0:
        return
    STT_RTFX.labels(model=model).observe(audio_seconds / wall_clock_seconds)


def render_latest() -> bytes:
    return generate_latest(REGISTRY)


__all__ = [
    "ERROR_TOTAL",
    "REGISTRY",
    "REQUEST_DURATION",
    "STT_RTFX",
    "record_error",
    "record_request",
    "record_rtfx",
    "render_latest",
]
