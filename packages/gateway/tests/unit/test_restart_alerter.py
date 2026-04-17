"""Unit tests for the supervisord restart-storm event listener.

The real loop reads from stdin — we exercise the windowing + threshold
logic via a small state-machine mirror that matches the production
``deque`` / ``_write_stderr`` path. Kept minimal on purpose: the
module is supervisord-specific and doesn't warrant a full
subprocess-driven integration test.
"""

from __future__ import annotations

import time
from collections import deque


def _push(queue: deque[float], now: float, window_s: int) -> int:
    queue.append(now)
    while queue and queue[0] < now - window_s:
        queue.popleft()
    return len(queue)


def test_threshold_fires_on_third_failure_within_window() -> None:
    from larynx_gateway.ops.restart_alerter import THRESHOLD, WINDOW_SECONDS

    queue: deque[float] = deque()
    t = time.time()
    assert _push(queue, t, WINDOW_SECONDS) == 1
    assert _push(queue, t + 10, WINDOW_SECONDS) == 2
    depth = _push(queue, t + 20, WINDOW_SECONDS)
    assert depth >= THRESHOLD


def test_older_failures_fall_out_of_window() -> None:
    from larynx_gateway.ops.restart_alerter import WINDOW_SECONDS

    queue: deque[float] = deque()
    t = time.time()
    _push(queue, t, WINDOW_SECONDS)
    # 2 minutes later — the first entry has aged out.
    depth = _push(queue, t + 120, WINDOW_SECONDS)
    assert depth == 1


def test_parse_header_round_trip() -> None:
    from larynx_gateway.ops.restart_alerter import _parse_header

    raw = "ver:3.0 server:supervisor serial:21 pool:listener poolserial:10 eventname:PROCESS_STATE_FATAL len:65\n"
    out = _parse_header(raw)
    assert out["eventname"] == "PROCESS_STATE_FATAL"
    assert out["len"] == "65"
    assert out["ver"] == "3.0"
