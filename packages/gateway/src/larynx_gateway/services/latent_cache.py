"""Two-tier latent cache: Redis (hot) + filesystem (persistent).

PRD §5.5 calls latent caching the key TTS latency unlock — encoding a
reference on every synthesis is the single biggest avoidable cost. We
cache exactly the bytes that the nano-vllm-voxcpm pool returns from
`encode_latents`, so cache -> model is a direct hand-off with no
serialisation.

Lookup order:
  1. Redis — key ``latents:{voice_id}:v1``, TTL bounded.
  2. Disk — ``{DATA_DIR}/voices/{voice_id}/latents.bin``. If found, re-warm
     Redis so the next hit stays in RAM.
  3. Miss — caller recomputes latents and calls ``put``.

Cache versioning: key suffix ``:v1`` lets us invalidate everything by
bumping the version without touching disk files. We also store a tiny
metadata JSON sidecar per voice so we can validate ``feat_dim`` /
``encoder_sample_rate`` when the upstream model changes.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis_async
import structlog
from prometheus_client import Counter

log = structlog.get_logger(__name__)

CACHE_VERSION = "v1"

# Exposed via /metrics (when we add the endpoint in M8); having the
# counters wired up from M2 means the observability story doesn't need a
# rewire later.
_cache_hits = Counter(
    "larynx_latent_cache_hits_total",
    "Latent cache hits",
    labelnames=("tier",),  # "redis" | "disk"
)
_cache_misses = Counter(
    "larynx_latent_cache_misses_total",
    "Latent cache misses (reference will be re-encoded by the worker)",
)
_cache_puts = Counter(
    "larynx_latent_cache_puts_total",
    "Latent cache writes",
)


@dataclass(frozen=True)
class LatentMetadata:
    voice_id: str
    feat_dim: int
    encoder_sample_rate: int
    num_frames: int
    cache_version: str = CACHE_VERSION

    def to_json(self) -> str:
        return json.dumps(
            {
                "voice_id": self.voice_id,
                "feat_dim": self.feat_dim,
                "encoder_sample_rate": self.encoder_sample_rate,
                "num_frames": self.num_frames,
                "cache_version": self.cache_version,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatentMetadata:
        return cls(
            voice_id=data["voice_id"],
            feat_dim=int(data["feat_dim"]),
            encoder_sample_rate=int(data["encoder_sample_rate"]),
            num_frames=int(data["num_frames"]),
            cache_version=data.get("cache_version", CACHE_VERSION),
        )


@dataclass(frozen=True)
class CacheResult:
    latents: bytes
    metadata: LatentMetadata
    tier: str  # "redis" | "disk"


def _redis_key(voice_id: str) -> str:
    return f"latents:{voice_id}:{CACHE_VERSION}"


def _redis_meta_key(voice_id: str) -> str:
    return f"latents:{voice_id}:{CACHE_VERSION}:meta"


def _disk_dir(data_dir: pathlib.Path, voice_id: str) -> pathlib.Path:
    return data_dir / "voices" / voice_id


def _disk_latents_path(data_dir: pathlib.Path, voice_id: str) -> pathlib.Path:
    return _disk_dir(data_dir, voice_id) / "latents.bin"


def _disk_meta_path(data_dir: pathlib.Path, voice_id: str) -> pathlib.Path:
    return _disk_dir(data_dir, voice_id) / "latents.meta.json"


class LatentCache:
    """Async Redis + sync-on-path disk cache.

    Disk writes happen on the event-loop thread intentionally — they're
    tiny (~100 KB per voice) and writing them under asyncio.to_thread only
    moves the same work to a different thread. The pay-off would matter
    if voices were multi-megabyte, which they aren't.
    """

    def __init__(
        self,
        redis_client: redis_async.Redis,
        data_dir: pathlib.Path,
        ttl_s: int = 3600,
    ) -> None:
        self._redis = redis_client
        self._data_dir = data_dir
        self._ttl_s = ttl_s

    # -- reads ---------------------------------------------------------------

    async def get(self, voice_id: str) -> CacheResult | None:
        redis_hit = await self._get_from_redis(voice_id)
        if redis_hit is not None:
            _cache_hits.labels(tier="redis").inc()
            log.info("latent_cache.hit", voice_id=voice_id, tier="redis")
            return redis_hit

        disk_hit = self._get_from_disk(voice_id)
        if disk_hit is not None:
            _cache_hits.labels(tier="disk").inc()
            log.info("latent_cache.hit", voice_id=voice_id, tier="disk")
            # Re-warm Redis so the next request hits RAM.
            await self._put_to_redis(voice_id, disk_hit.latents, disk_hit.metadata)
            return disk_hit

        _cache_misses.inc()
        log.info("latent_cache.miss", voice_id=voice_id)
        return None

    async def _get_from_redis(self, voice_id: str) -> CacheResult | None:
        latents = await self._redis.get(_redis_key(voice_id))
        meta_raw = await self._redis.get(_redis_meta_key(voice_id))
        if latents is None or meta_raw is None:
            return None
        meta = LatentMetadata.from_dict(
            json.loads(meta_raw if isinstance(meta_raw, str) else meta_raw.decode("utf-8"))
        )
        return CacheResult(latents=latents, metadata=meta, tier="redis")

    def _get_from_disk(self, voice_id: str) -> CacheResult | None:
        latents_path = _disk_latents_path(self._data_dir, voice_id)
        meta_path = _disk_meta_path(self._data_dir, voice_id)
        if not latents_path.exists() or not meta_path.exists():
            return None
        try:
            meta = LatentMetadata.from_dict(json.loads(meta_path.read_text()))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.warning("latent_cache.disk_meta_corrupt", voice_id=voice_id, error=str(e))
            return None
        latents = latents_path.read_bytes()
        if len(latents) != 4 * meta.feat_dim * meta.num_frames:
            log.warning(
                "latent_cache.disk_size_mismatch",
                voice_id=voice_id,
                expected=4 * meta.feat_dim * meta.num_frames,
                actual=len(latents),
            )
            return None
        return CacheResult(latents=latents, metadata=meta, tier="disk")

    # -- writes --------------------------------------------------------------

    async def put(self, voice_id: str, latents: bytes, meta: LatentMetadata) -> None:
        self._put_to_disk(voice_id, latents, meta)
        await self._put_to_redis(voice_id, latents, meta)
        _cache_puts.inc()
        log.info(
            "latent_cache.put",
            voice_id=voice_id,
            bytes=len(latents),
            num_frames=meta.num_frames,
        )

    def _put_to_disk(self, voice_id: str, latents: bytes, meta: LatentMetadata) -> None:
        target = _disk_dir(self._data_dir, voice_id)
        target.mkdir(parents=True, exist_ok=True)
        _disk_latents_path(self._data_dir, voice_id).write_bytes(latents)
        _disk_meta_path(self._data_dir, voice_id).write_text(meta.to_json())

    async def _put_to_redis(self, voice_id: str, latents: bytes, meta: LatentMetadata) -> None:
        # Pipeline so both keys expire together — otherwise a crash between
        # SET calls would leave one half orphaned.
        pipe = self._redis.pipeline()
        pipe.set(_redis_key(voice_id), latents, ex=self._ttl_s)
        pipe.set(_redis_meta_key(voice_id), meta.to_json(), ex=self._ttl_s)
        await pipe.execute()

    # -- eviction ------------------------------------------------------------

    async def delete(self, voice_id: str) -> None:
        await self._redis.delete(_redis_key(voice_id), _redis_meta_key(voice_id))
        for path in (
            _disk_latents_path(self._data_dir, voice_id),
            _disk_meta_path(self._data_dir, voice_id),
        ):
            if path.exists():
                path.unlink()
        log.info("latent_cache.delete", voice_id=voice_id)


def build_redis_client(redis_url: str) -> redis_async.Redis:
    """Construct the async redis client the cache will share with other
    subsystems (batch queue in M8, streaming session state in M4/M5).

    ``decode_responses=False`` is deliberate — we store raw latent bytes.
    """
    return redis_async.from_url(redis_url, decode_responses=False)
