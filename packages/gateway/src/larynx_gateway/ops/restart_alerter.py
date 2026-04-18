"""Supervisord event listener — flags restart storms.

Reads ``PROCESS_STATE`` events on stdin (supervisord's eventlistener
protocol), tracks FATAL transitions per program over a rolling 60s
window, and emits a ``critical.restart_storm`` line on stderr once a
program has failed to start 3 times inside that window.

Exits only on EOF (supervisord tearing it down). All output on stderr
is captured by supervisord and written to the container log —
``journalctl`` / ``docker logs`` pick it up for free.

See ORCHESTRATION-M8.md §3.3.

Protocol reference:
    http://supervisord.org/events.html#event-listener-notification-protocol
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict, deque

# 3 failures inside 60s triggers the alert.
WINDOW_SECONDS = 60
THRESHOLD = 3


def _write_stdout(payload: str) -> None:
    """Required ack back to supervisord (READY / RESULT / OK)."""
    sys.stdout.write(payload)
    sys.stdout.flush()


def _write_stderr(payload: str) -> None:
    sys.stderr.write(payload + "\n")
    sys.stderr.flush()


def _parse_header(line: str) -> dict[str, str]:
    return dict(item.split(":", 1) for item in line.strip().split() if ":" in item)


def main() -> None:
    # Per-program rolling window of failure timestamps.
    fails: dict[str, deque[float]] = defaultdict(deque)

    while True:
        _write_stdout("READY\n")

        header_line = sys.stdin.readline()
        if not header_line:
            return
        header = _parse_header(header_line)
        data_len = int(header.get("len", "0"))
        payload = sys.stdin.read(data_len) if data_len else ""

        event = header.get("eventname", "")
        if event != "PROCESS_STATE_FATAL":
            _write_stdout("RESULT 2\nOK")
            continue

        # Payload is ``key:value key:value ...``. The ``processname``
        # field tells us which supervised program just hit FATAL.
        fields = _parse_header(payload)
        program = fields.get("processname", "unknown")

        now = time.time()
        queue = fails[program]
        queue.append(now)
        while queue and queue[0] < now - WINDOW_SECONDS:
            queue.popleft()

        if len(queue) >= THRESHOLD:
            _write_stderr(
                f"critical.restart_storm program={program} fails_in_{WINDOW_SECONDS}s={len(queue)}"
            )
            # Reset the window so a single storm logs once per cluster
            # of restarts, not once per additional failure after the
            # threshold.
            queue.clear()

        _write_stdout("RESULT 2\nOK")


if __name__ == "__main__":  # pragma: no cover — entrypoint only
    main()
