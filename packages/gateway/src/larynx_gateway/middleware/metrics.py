"""Per-request Prometheus metrics middleware.

Wraps every HTTP request with a Histogram on duration + a Counter on
error responses, labelled by route template (not raw path — per-id
cardinality would explode). WebSocket traffic is counted via explicit
Gauge updates in the WS route itself; Starlette's ASGI lifecycle for
WS doesn't fit cleanly into a single middleware histogram.

See ORCHESTRATION-M8.md §3.1.
"""

from __future__ import annotations

import time

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

# Histogram buckets tuned to the PRD latency budget — long-tail
# ~10s covers batch artifact GETs on large files; the busy end
# (20ms..200ms) is where TTS TTFB lives.
_LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


_request_duration = Histogram(
    "larynx_request_duration_seconds",
    "HTTP request duration, labelled by route template and status class.",
    labelnames=("method", "route", "status_class"),
    buckets=_LATENCY_BUCKETS,
)


_error_counter = Counter(
    "larynx_error_total",
    "Count of HTTP responses in the 4xx/5xx range.",
    labelnames=("method", "route", "status_code"),
)


def _route_template(request: Request) -> str:
    """Prefer the matched route's pattern over the raw path.

    Starlette stashes the matched Route on ``request.scope['route']``
    once the routing middleware runs. We fall back to the method-less
    path for unmatched (404) requests.
    """
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return route.path  # type: ignore[no-any-return]
    return request.url.path


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Observes every request; labels with a bounded-cardinality route."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed = time.perf_counter() - start
            route = _route_template(request)
            status_class = f"{status_code // 100}xx"
            _request_duration.labels(
                method=request.method, route=route, status_class=status_class
            ).observe(elapsed)
            if status_code >= 400:
                _error_counter.labels(
                    method=request.method, route=route, status_code=str(status_code)
                ).inc()


__all__ = ["PrometheusMiddleware"]
