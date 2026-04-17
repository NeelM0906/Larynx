"""LatentCache integration tests — real Redis + real filesystem.

Requires the docker-compose Redis on :6380. Skips with a loud message if
it isn't reachable. No fakes.
"""

from __future__ import annotations

import pathlib
import uuid

import pytest
import pytest_asyncio
import redis.asyncio as redis_async
from larynx_gateway.services.latent_cache import (
    CACHE_VERSION,
    LatentCache,
    LatentMetadata,
    build_redis_client,
)

TEST_REDIS_URL = "redis://localhost:6380/15"  # db 15 isolates from app data


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
    await client.flushdb()  # db 15 is dedicated to tests — safe to wipe
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest_asyncio.fixture
async def cache(redis_client: redis_async.Redis, tmp_path: pathlib.Path) -> LatentCache:
    return LatentCache(redis_client, data_dir=tmp_path, ttl_s=3600)


def _make_meta(voice_id: str, feat_dim: int = 64, num_frames: int = 50) -> LatentMetadata:
    return LatentMetadata(
        voice_id=voice_id,
        feat_dim=feat_dim,
        encoder_sample_rate=24000,
        num_frames=num_frames,
    )


@pytest.mark.asyncio
async def test_miss_returns_none(cache: LatentCache) -> None:
    assert await cache.get("nonexistent") is None


@pytest.mark.asyncio
async def test_put_then_hit_from_redis(cache: LatentCache) -> None:
    vid = uuid.uuid4().hex
    latents = b"\x00\x11\x22\x33" * 64 * 50  # 4 × feat_dim × num_frames
    meta = _make_meta(vid)

    await cache.put(vid, latents, meta)
    hit = await cache.get(vid)
    assert hit is not None
    assert hit.tier == "redis"
    assert hit.latents == latents
    assert hit.metadata.num_frames == 50


@pytest.mark.asyncio
async def test_disk_fallback_rewarms_redis(
    cache: LatentCache, redis_client: redis_async.Redis
) -> None:
    vid = uuid.uuid4().hex
    latents = b"\xab\xcd\xef\x01" * 64 * 50
    meta = _make_meta(vid)

    await cache.put(vid, latents, meta)
    # Evict from redis so only disk remains
    await redis_client.delete(f"latents:{vid}:{CACHE_VERSION}")
    await redis_client.delete(f"latents:{vid}:{CACHE_VERSION}:meta")

    # First get: disk hit, should re-warm Redis
    hit = await cache.get(vid)
    assert hit is not None
    assert hit.tier == "disk"

    # Second get: Redis hit (proves the re-warm ran)
    hit2 = await cache.get(vid)
    assert hit2 is not None
    assert hit2.tier == "redis"


@pytest.mark.asyncio
async def test_delete_removes_from_both_tiers(cache: LatentCache, tmp_path: pathlib.Path) -> None:
    vid = uuid.uuid4().hex
    latents = b"\x10\x20\x30\x40" * 64 * 50
    meta = _make_meta(vid)
    await cache.put(vid, latents, meta)

    await cache.delete(vid)

    assert await cache.get(vid) is None
    assert not (tmp_path / "voices" / vid / "latents.bin").exists()
    assert not (tmp_path / "voices" / vid / "latents.meta.json").exists()


@pytest.mark.asyncio
async def test_disk_size_mismatch_treated_as_miss(
    cache: LatentCache, redis_client: redis_async.Redis, tmp_path: pathlib.Path
) -> None:
    vid = uuid.uuid4().hex
    latents = b"\x55\xaa" * 64 * 50 * 2
    meta = _make_meta(vid, num_frames=50)
    await cache.put(vid, latents, meta)

    await redis_client.flushdb()

    # Truncate the on-disk file; meta will disagree -> cache should
    # report miss rather than serve corrupt data.
    latents_path = tmp_path / "voices" / vid / "latents.bin"
    latents_path.write_bytes(latents[:100])

    assert await cache.get(vid) is None


@pytest.mark.asyncio
async def test_ttl_set_on_redis_entries(
    cache: LatentCache, redis_client: redis_async.Redis
) -> None:
    vid = uuid.uuid4().hex
    await cache.put(vid, b"\x01\x02\x03\x04" * 64 * 50, _make_meta(vid))

    # Exact TTL isn't deterministic down to the second but both keys
    # must have >0 remaining TTL.
    ttl_main = await redis_client.ttl(f"latents:{vid}:{CACHE_VERSION}")
    ttl_meta = await redis_client.ttl(f"latents:{vid}:{CACHE_VERSION}:meta")
    assert ttl_main > 0
    assert ttl_meta > 0
