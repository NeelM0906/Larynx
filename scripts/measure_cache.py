#!/usr/bin/env python
"""Measure latent-cache benefit against the real VoxCPM2 model.

Run this on the GPU box with docker-compose services up:

    docker compose up -d
    make migrate
    RUN_REAL_MODEL=1 uv run python scripts/measure_cache.py

The script:
  1. Starts an in-process gateway with LARYNX_TTS_MODE=voxcpm.
  2. Uploads one reference voice (which encodes latents + puts them in
     the two-tier cache at upload time).
  3. Wipes Redis. Issues one synth request: first hit after wipe is a
     disk fallback + Redis re-warm (proves the disk path).
  4. Issues 9 more synth requests with the same voice — all Redis hits.
  5. Prints per-request wall time so the cache benefit is visible.

Also runs a "cold encode" measurement for comparison: we delete BOTH
the Redis cache and the on-disk latent file for the voice and then
hit /v1/tts. That forces the gateway to re-encode from source audio —
the single expensive operation the cache is designed to avoid.
"""

from __future__ import annotations

import asyncio
import io
import os
import pathlib
import statistics
import tempfile
import time
import uuid

import numpy as np
import redis.asyncio as redis_async
import soundfile as sf
from httpx import ASGITransport, AsyncClient

TOKEN = "measure-cache-token"
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6380/13")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://larynx:larynx@localhost:5433/larynx",
)


def _realistic_reference_wav() -> bytes:
    sr = 16000  # matches VoxCPM2 encoder native rate — no resample cost
    # 15s is close to the LibriVox clips `load_demo_voices.py` uses, so the
    # encode cost we measure is representative of real voice uploads.
    t = np.linspace(0, 15.0, int(sr * 15.0), dtype=np.float32)
    samples = (
        0.3 * np.sin(2 * np.pi * 180 * t)
        + 0.15 * np.sin(2 * np.pi * 500 * t)
        + 0.1 * np.sin(2 * np.pi * 1500 * t)
    ).astype(np.float32)
    samples *= 0.5 * (1 + np.sin(2 * np.pi * 4 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def main() -> int:
    os.environ["LARYNX_API_TOKEN"] = TOKEN
    os.environ["LARYNX_TTS_MODE"] = "voxcpm"
    os.environ["LARYNX_VOXCPM_GPU"] = os.environ.get("LARYNX_VOXCPM_GPU", "0")
    os.environ["LARYNX_LOG_JSON"] = "false"
    os.environ["REDIS_URL"] = REDIS_URL
    os.environ["DATABASE_URL"] = DATABASE_URL

    data_dir = pathlib.Path(tempfile.mkdtemp(prefix="larynx-measure-"))
    os.environ["LARYNX_DATA_DIR"] = str(data_dir)

    from larynx_gateway.config import get_settings
    from larynx_gateway.main import create_app
    from larynx_gateway.services.latent_cache import CACHE_VERSION

    get_settings.cache_clear()
    app = create_app()

    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test", timeout=600) as http,
        app.router.lifespan_context(app),
    ):
        name = f"measure-{uuid.uuid4().hex[:8]}"
        print(f"[measure] uploading voice {name} ...")
        t0 = time.perf_counter()
        r = await http.post(
            "/v1/voices",
            headers={"Authorization": f"Bearer {TOKEN}"},
            files={"audio": ("ref.wav", _realistic_reference_wav(), "audio/wav")},
            data={"name": name},
        )
        upload_ms = int((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        vid = r.json()["id"]
        print(f"[measure] uploaded in {upload_ms} ms  (includes encode + cache put)")

        redis = redis_async.from_url(REDIS_URL, decode_responses=False)

        # --- warm Redis (just uploaded) ---
        print("\n[measure] --- Redis-hot sequence (10 calls) ---")
        hot_ms: list[int] = []
        for i in range(10):
            t0 = time.perf_counter()
            r = await http.post(
                "/v1/tts",
                headers={"Authorization": f"Bearer {TOKEN}"},
                json={"text": f"Latency test request number {i}.", "voice_id": vid},
            )
            t1 = time.perf_counter()
            assert r.status_code == 200
            hot_ms.append(int((t1 - t0) * 1000))
        _print_series("redis hot", hot_ms)

        # --- Redis cold, disk warm ---
        print("\n[measure] --- Redis-cold / disk-warm sequence (10 calls) ---")
        disk_ms: list[int] = []
        for i in range(10):
            await redis.delete(
                f"latents:{vid}:{CACHE_VERSION}",
                f"latents:{vid}:{CACHE_VERSION}:meta",
            )
            t0 = time.perf_counter()
            r = await http.post(
                "/v1/tts",
                headers={"Authorization": f"Bearer {TOKEN}"},
                json={"text": f"Disk hit request number {i}.", "voice_id": vid},
            )
            t1 = time.perf_counter()
            assert r.status_code == 200
            disk_ms.append(int((t1 - t0) * 1000))
        _print_series("disk hit", disk_ms)

        # --- both tiers cold (forces re-encode) ---
        print("\n[measure] --- Both tiers cold (re-encode per call, 10 calls) ---")
        cold_ms: list[int] = []
        for i in range(10):
            await redis.delete(
                f"latents:{vid}:{CACHE_VERSION}",
                f"latents:{vid}:{CACHE_VERSION}:meta",
            )
            latents_path = data_dir / "voices" / vid / "latents.bin"
            meta_path = data_dir / "voices" / vid / "latents.meta.json"
            for p in (latents_path, meta_path):
                if p.exists():
                    p.unlink()
            t0 = time.perf_counter()
            r = await http.post(
                "/v1/tts",
                headers={"Authorization": f"Bearer {TOKEN}"},
                json={"text": f"Cold re-encode request {i}.", "voice_id": vid},
            )
            t1 = time.perf_counter()
            assert r.status_code == 200
            cold_ms.append(int((t1 - t0) * 1000))
        _print_series("cold re-encode", cold_ms)

        print(
            "\n[measure] summary (median):"
            f"\n    redis hot   = {statistics.median(hot_ms)} ms"
            f"\n    disk hit    = {statistics.median(disk_ms)} ms"
            f"\n    cold encode = {statistics.median(cold_ms)} ms"
            f"\n    cache benefit (cold - hot) = "
            f"{statistics.median(cold_ms) - statistics.median(hot_ms)} ms"
        )

        await redis.aclose()

    return 0


def _print_series(label: str, series: list[int]) -> None:
    print(
        f"  {label:>12s}: "
        f"first={series[0]} ms  "
        f"median={statistics.median(series)} ms  "
        f"min={min(series)} ms  max={max(series)} ms  "
        f"p90={sorted(series)[int(len(series) * 0.9)]} ms"
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
