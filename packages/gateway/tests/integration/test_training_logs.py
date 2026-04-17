"""TrainingLogStore — real Redis Streams, no fakes.

Mirrors the test layout used by test_latent_cache.py: docker-compose Redis
on :6380 db 15, with a flushdb before / after so tests never see each
other's state. Skips loudly if Redis isn't up.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
import redis.asyncio as redis_async
from larynx_gateway.services.latent_cache import build_redis_client
from larynx_gateway.services.training_logs import TrainingLogStore

TEST_REDIS_URL = "redis://localhost:6380/15"


async def _redis_reachable() -> bool:
    try:
        client = build_redis_client(TEST_REDIS_URL)
        await client.ping()
        await client.aclose()
        return True
    except Exception:
        return False


@pytest_asyncio.fixture
async def redis_client() -> redis_async.Redis:
    if not await _redis_reachable():
        pytest.skip(
            "Redis not reachable at redis://localhost:6380/15. Run `docker compose up -d` first."
        )
    client = build_redis_client(TEST_REDIS_URL)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest_asyncio.fixture
async def store(redis_client: redis_async.Redis) -> TrainingLogStore:
    # Short TTL (3s) used by the TTL test; other tests don't care.
    return TrainingLogStore(redis_client, maxlen=1000, ttl_s=3)


@pytest.mark.asyncio
async def test_append_then_tail_returns_entries(store: TrainingLogStore) -> None:
    e1 = await store.append("job-1", "first line")
    e2 = await store.append("job-1", "second line")

    # No after_id -> everything from the start.
    entries = await store.tail("job-1")
    assert [(e.event_id, e.line) for e in entries] == [
        (e1.event_id, "first line"),
        (e2.event_id, "second line"),
    ]


@pytest.mark.asyncio
async def test_tail_honours_after_id(store: TrainingLogStore) -> None:
    e1 = await store.append("job-1", "alpha")
    e2 = await store.append("job-1", "bravo")
    e3 = await store.append("job-1", "charlie")

    entries = await store.tail("job-1", after_id=e1.event_id)
    assert [e.line for e in entries] == ["bravo", "charlie"]

    entries = await store.tail("job-1", after_id=e3.event_id)
    assert entries == []

    # Sanity: e2 is strictly between e1 and e3.
    assert e1.event_id < e2.event_id < e3.event_id


@pytest.mark.asyncio
async def test_tail_count_limits(store: TrainingLogStore) -> None:
    for i in range(10):
        await store.append("job-1", f"line {i}")
    entries = await store.tail("job-1", count=3)
    assert [e.line for e in entries] == ["line 0", "line 1", "line 2"]


@pytest.mark.asyncio
async def test_unknown_job_returns_empty(store: TrainingLogStore) -> None:
    entries = await store.tail("never-logged")
    assert entries == []


@pytest.mark.asyncio
async def test_maxlen_caps_stream_size(redis_client: redis_async.Redis) -> None:
    # XADD MAXLEN with `approximate=True` uses Redis' radix-tree node
    # boundary for trimming — the internal node size is ~100 entries,
    # so a MAXLEN of 10 with approximate trimming stabilises at roughly
    # one node (~100 entries) regardless of how much we append beyond it.
    # The point of this test is "doesn't grow without bound", not a
    # tight upper bound.
    store = TrainingLogStore(redis_client, maxlen=10, ttl_s=60)
    for i in range(2_000):
        await store.append("job-1", f"line {i}")
    length = await redis_client.xlen(TrainingLogStore.key("job-1"))
    # 2000 writes into a MAXLEN=10 approximate stream — if trimming is
    # wired correctly we're bounded by a small multiple of the radix-
    # tree node size (~100), never by the total write count.
    assert length < 500, f"stream grew to {length}; approximate MAXLEN not applied?"


@pytest.mark.asyncio
async def test_ttl_refreshes_on_append(
    store: TrainingLogStore, redis_client: redis_async.Redis
) -> None:
    # ttl_s=3 from the fixture. First append sets TTL; the stream must
    # be gone ~after that TTL expires (give ourselves 2s of slack).
    await store.append("job-ephemeral", "first")
    assert await redis_client.exists(TrainingLogStore.key("job-ephemeral")) == 1
    await asyncio.sleep(5)
    assert await redis_client.exists(TrainingLogStore.key("job-ephemeral")) == 0


@pytest.mark.asyncio
async def test_clear_removes_stream(
    store: TrainingLogStore, redis_client: redis_async.Redis
) -> None:
    await store.append("job-1", "x")
    assert await redis_client.exists(TrainingLogStore.key("job-1")) == 1
    await store.clear("job-1")
    assert await redis_client.exists(TrainingLogStore.key("job-1")) == 0


def test_key_scheme_is_stable() -> None:
    # Downstream clients (SSE route, ops tooling) encode this key in URLs
    # and logs. If we ever move it, this test fails loudly.
    assert TrainingLogStore.key("job-abc") == "logs:training:job-abc"
