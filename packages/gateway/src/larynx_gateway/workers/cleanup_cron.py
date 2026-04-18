"""Daily cleanup cron — runs in-process inside the gateway.

A single asyncio task. Wakes every ``interval_s`` seconds (default
86400 = 1d), opens a DB session, invokes ``batch_cleanup.run_cleanup``.
The shutdown_event check on every wait lets the lifespan tear it down
cleanly.

No Arq, no crontab — keeps the footprint inside a single process so
we don't need a second supervisord program for one task.

See ORCHESTRATION-M8.md §3.5.
"""

from __future__ import annotations

import asyncio
import pathlib

import structlog

from larynx_gateway.db.session import get_session
from larynx_gateway.services.batch_cleanup import run_cleanup

log = structlog.get_logger(__name__)


async def run_cleanup_cron(
    data_dir: pathlib.Path,
    shutdown_event: asyncio.Event,
    *,
    interval_s: int = 86400,
    initial_delay_s: int = 60,
) -> None:
    """Single long-lived task. Starts after ``initial_delay_s`` so a
    test run that only exists for a few seconds doesn't pay cleanup
    latency on teardown.
    """
    log.info("cleanup.cron_start", interval_s=interval_s)
    # Initial delay lets the gateway finish booting before we do
    # disk I/O. Short-circuits early if shutdown fires.
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=initial_delay_s)
        return
    except TimeoutError:
        pass

    while not shutdown_event.is_set():
        try:
            async for session in get_session():
                await run_cleanup(session, data_dir)
                break
        except Exception as e:  # noqa: BLE001
            # Cron failures don't crash the gateway; next tick retries.
            log.error("cleanup.cron_error", error=repr(e))

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_s)
            break
        except TimeoutError:
            continue

    log.info("cleanup.cron_stop")
