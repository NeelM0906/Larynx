"""Minimal Redis-list-based queue for batch TTS items.

See ORCHESTRATION-M8.md §1.3 for the Arq-vs-this decision. In short:
Arq adds a second worker process + cron-event-loop we don't use in
v1. A plain ``LPUSH``/``BRPOP`` list + a cancel-set is ~100 lines and
inherits durability from the existing Redis instance.

Producer: the POST /v1/batch route calls ``enqueue_item`` once per
submitted item after the DB commit.

Consumer: two asyncio tasks started by the gateway lifespan (see
``workers/batch_worker.py``) run a BRPOP → check-cancel → synthesize
loop, sharing the in-process VoxCPMClient with the real-time path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import redis.asyncio as redis_async

QUEUE_KEY = "larynx:batch:queue"
CANCEL_SET_KEY = "larynx:batch:cancelled"

# Upper bound on the BRPOP timeout. Shorter = faster shutdown drain;
# longer = less Redis chatter when idle. 1s is the shortest round value
# that keeps idle load trivial.
BRPOP_TIMEOUT_S = 1


@dataclass(frozen=True)
class QueueEntry:
    batch_job_id: str
    item_idx: int


class BatchQueue:
    """Producer + consumer primitive for batch items.

    Call-site surface is intentionally tiny so a future Arq swap is a
    one-file change. ``enqueue_item`` + ``cancel_job`` are called from
    the route layer; ``dequeue`` + ``is_cancelled`` are called by the
    consumer tasks.
    """

    def __init__(self, client: redis_async.Redis) -> None:
        self._r = client

    async def enqueue_item(self, *, batch_job_id: str, item_idx: int) -> None:
        """Append one (job_id, item_idx) tuple to the queue tail."""
        payload = json.dumps({"job_id": batch_job_id, "item_idx": item_idx})
        await self._r.lpush(QUEUE_KEY, payload)

    async def cancel_job(self, *, batch_job_id: str) -> None:
        """Mark every queued item for this job as cancelled.

        Consumers do the actual DB transition on the next BRPOP pickup.
        The set has no TTL because a successful batch completion doesn't
        leave rows in it (consumers only add to the DB CANCELLED state,
        not the set). We clean up the set entry explicitly in
        :meth:`forget_cancelled` once the job reaches a terminal state.
        """
        await self._r.sadd(CANCEL_SET_KEY, batch_job_id)

    async def forget_cancelled(self, *, batch_job_id: str) -> None:
        await self._r.srem(CANCEL_SET_KEY, batch_job_id)

    async def is_cancelled(self, batch_job_id: str) -> bool:
        return bool(await self._r.sismember(CANCEL_SET_KEY, batch_job_id))

    async def dequeue(self, *, timeout_s: int = BRPOP_TIMEOUT_S) -> QueueEntry | None:
        """Block up to ``timeout_s`` for the next entry. Returns None on timeout.

        redis.asyncio's BRPOP returns (list_name, payload) or None. We
        unwrap both into our typed entry; a bad payload raises so a
        consumer task crashing on one entry doesn't silently drop it.
        """
        result = await self._r.brpop(QUEUE_KEY, timeout=timeout_s)
        if result is None:
            return None
        _key, payload = result
        data = json.loads(payload)
        return QueueEntry(batch_job_id=data["job_id"], item_idx=int(data["item_idx"]))

    async def depth(self) -> int:
        """Current queue length — surfaced in /ready and Prometheus."""
        return int(await self._r.llen(QUEUE_KEY))
