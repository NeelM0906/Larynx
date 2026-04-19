"""Aiohttp sidecar that exposes the worker's Prometheus metrics.

Only started when the worker runs as a standalone supervisord program
(see :func:`larynx_funasr_worker.main.main`). When the gateway runs the
worker in-process the sidecar isn't started — the worker's counters
live on a private registry (see :mod:`metrics`) so they never collide
with the gateway's default-registry metrics.

See ORCHESTRATION-M8.md §3.1 + §7.2.
"""

from __future__ import annotations

import structlog
from aiohttp import web
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from larynx_funasr_worker.metrics import render_latest

log = structlog.get_logger(__name__)

DEFAULT_PORT = 9101


async def _metrics_handler(_request: web.Request) -> web.Response:
    return web.Response(body=render_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})


class MetricsSidecar:
    """Minimal aiohttp server exposing ``/metrics`` on the configured port."""

    def __init__(self, *, host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._runner: web.AppRunner | None = None

    @property
    def port(self) -> int:
        return self._port

    async def start(self) -> None:
        if self._runner is not None:
            return
        app = web.Application()
        app.router.add_get("/metrics", _metrics_handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host=self._host, port=self._port)
        await site.start()
        log.info("funasr_worker.metrics.started", host=self._host, port=self._port)

    async def stop(self) -> None:
        if self._runner is None:
            return
        await self._runner.cleanup()
        self._runner = None
        log.info("funasr_worker.metrics.stopped", port=self._port)


__all__ = ["DEFAULT_PORT", "MetricsSidecar"]
