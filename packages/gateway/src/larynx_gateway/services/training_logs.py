"""Redis-Streams-backed log store for fine-tune jobs.

The subprocess runner produces lines faster than any single consumer (the
gateway's SSE route) can serve them to multiple clients. Redis Streams
are the right primitive here: append-only, efficient range reads by id,
approximate MAXLEN trimming, and the stream id doubles as the SSE
Last-Event-ID so reconnecting clients pick up exactly where they left off.

See ORCHESTRATION-M7.md §1.2 and §8.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as redis_async


@dataclass(frozen=True)
class LogEntry:
    """One line in a job's log stream, with its stream id."""

    event_id: str
    line: str


class TrainingLogStore:
    """Append + bounded-tail access to ``logs:training:{job_id}`` streams.

    ``maxlen`` sizes the approximate retention. ``ttl_s`` is refreshed on
    every append so an active stream never vanishes mid-training; an
    idle stream ages out and frees the key.
    """

    KEY_PREFIX = "logs:training:"

    def __init__(
        self,
        redis_client: redis_async.Redis,
        *,
        maxlen: int = 10_000,
        ttl_s: int = 86_400,
    ) -> None:
        self._redis = redis_client
        self._maxlen = maxlen
        self._ttl_s = ttl_s

    @classmethod
    def key(cls, job_id: str) -> str:
        return f"{cls.KEY_PREFIX}{job_id}"

    async def append(self, job_id: str, line: str) -> LogEntry:
        """Append one line; return the stream id it landed under."""
        key = self.key(job_id)
        # ``approximate=True`` (the Redis ``~`` modifier) lets Redis trim
        # in whole node increments, which is drastically cheaper than
        # exact MAXLEN. We only care about the bound, not the exact size.
        event_id_bytes = await self._redis.xadd(
            key,
            {"line": line},
            maxlen=self._maxlen,
            approximate=True,
        )
        event_id = event_id_bytes.decode() if isinstance(event_id_bytes, bytes) else event_id_bytes
        # EXPIRE refreshes the TTL on every append. An active stream
        # therefore never expires; a stream idle for ``ttl_s`` ages out.
        await self._redis.expire(key, self._ttl_s)
        return LogEntry(event_id=event_id, line=line)

    async def tail(
        self,
        job_id: str,
        after_id: str | None = None,
        count: int = 100,
    ) -> list[LogEntry]:
        """Return up to ``count`` entries strictly after ``after_id``.

        ``after_id=None`` means "from the start of the stream". ``count``
        upper-bounds the return so a far-behind reader doesn't get an
        unbounded buffer dumped on them.

        Implementation note: we use ``XREAD`` rather than ``XRANGE`` so
        the id semantics are naturally exclusive — ``XREAD`` always
        returns entries whose id is *greater than* the cursor. Callers
        that want the full history pass ``after_id='0-0'`` or ``None``.
        """
        cursor = after_id if after_id is not None else "0-0"
        # streams param is a mapping key -> last-seen id.
        result = await self._redis.xread({self.key(job_id): cursor}, count=count)
        if not result:
            return []
        # result is [(key_bytes, [(id_bytes, {b"line": b"..."}), ...])] —
        # one tuple per requested stream. We only asked for one stream
        # so grab the first.
        _, raw_entries = result[0]
        return [_decode_entry(raw_id, fields) for raw_id, fields in raw_entries]

    async def clear(self, job_id: str) -> None:
        """Delete the stream. Used in tests and by job-pruning cron work."""
        await self._redis.delete(self.key(job_id))


def _decode_entry(raw_id: bytes | str, fields: dict[bytes | str, bytes | str]) -> LogEntry:
    event_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
    # Fields may arrive with bytes or str keys depending on
    # decode_responses; our Redis client uses decode_responses=False
    # (see latent_cache.build_redis_client) so we decode defensively.
    line_raw = (
        fields.get(b"line") if isinstance(next(iter(fields), b""), bytes) else fields.get("line")
    )
    if line_raw is None:
        # Malformed entry — shouldn't happen, but give the caller a
        # deterministic empty line rather than raising mid-iteration.
        line = ""
    elif isinstance(line_raw, bytes):
        line = line_raw.decode("utf-8", errors="replace")
    else:
        line = str(line_raw)
    return LogEntry(event_id=event_id, line=line)
