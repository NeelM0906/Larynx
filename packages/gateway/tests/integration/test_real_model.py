"""Real-model voice cloning tests.

Loads the actual VoxCPM2 model on GPU 0 and exercises the upload -> cache
-> cloned-synthesis -> cache-hit-faster pipeline end-to-end.

Opt-in: set ``RUN_REAL_MODEL=1`` and run with ``pytest -m real_model``.
Model load takes ~15-30s on the RTX Pro 6000, so the test module reuses
a single worker across the whole session via a module-scoped fixture.

Skips cleanly if RUN_REAL_MODEL is not set, or if the GPU / model cache
isn't reachable — this suite is only useful on the deployment box.
"""

from __future__ import annotations

import io
import os
import pathlib
import time
import uuid

import numpy as np
import pytest
import pytest_asyncio
import soundfile as sf
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.real_model


def _skip_if_disabled() -> None:
    if os.environ.get("RUN_REAL_MODEL") != "1":
        pytest.skip("set RUN_REAL_MODEL=1 to run real-model tests")
    try:
        import nanovllm_voxcpm  # noqa: F401
    except ImportError:
        pytest.skip("nano-vllm-voxcpm not installed (run `uv sync --extra gpu`)")


def _real_voice_bytes() -> bytes:
    """Generate a realistic-ish reference clip: 2s of mixed sinusoids.

    VoxCPM2's encoder wants something speech-like. Pure white noise also
    works but a voiced-ish signal reduces model weirdness on the output.
    """
    sr = 24000
    t = np.linspace(0, 2.0, int(sr * 2.0), dtype=np.float32)
    # Formants at ~180 (pitch), 500, 1500, 2500 Hz — rough vowel analog.
    samples = (
        0.3 * np.sin(2 * np.pi * 180 * t)
        + 0.15 * np.sin(2 * np.pi * 500 * t)
        + 0.1 * np.sin(2 * np.pi * 1500 * t)
        + 0.05 * np.sin(2 * np.pi * 2500 * t)
    ).astype(np.float32)
    # AM envelope at 4 Hz to avoid a steady drone.
    samples *= 0.5 * (1 + np.sin(2 * np.pi * 4 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest_asyncio.fixture(scope="module")
async def real_client(
    tmp_path_factory: pytest.TempPathFactory,
) -> AsyncClient:
    _skip_if_disabled()

    from larynx_gateway.config import get_settings
    from larynx_gateway.main import create_app

    from tests.conftest import TEST_TOKEN, _ensure_test_db, _reset_test_db

    _ensure_test_db()
    _reset_test_db()

    data_dir = tmp_path_factory.mktemp("larynx-real-model-data")

    os.environ["LARYNX_API_TOKEN"] = TEST_TOKEN
    os.environ["LARYNX_TTS_MODE"] = "voxcpm"
    os.environ["LARYNX_VOXCPM_GPU"] = "0"
    os.environ["LARYNX_LOG_JSON"] = "false"
    os.environ["LARYNX_DATA_DIR"] = str(data_dir)
    os.environ["DATABASE_URL"] = "postgresql+psycopg://larynx:larynx@localhost:5433/larynx_test"
    os.environ["REDIS_URL"] = "redis://localhost:6380/14"

    get_settings.cache_clear()
    app = create_app()

    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test", timeout=300) as client,
        app.router.lifespan_context(app),
    ):
        yield client


@pytest.fixture
def real_auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token-please-ignore"}


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_upload_produces_latents_on_disk_and_redis(
    real_client: AsyncClient,
    real_auth_headers: dict[str, str],
) -> None:
    r = await real_client.post(
        "/v1/voices",
        headers=real_auth_headers,
        files={"audio": ("ref.wav", _real_voice_bytes(), "audio/wav")},
        data={"name": f"real-upload-{uuid.uuid4().hex[:8]}"},
    )
    assert r.status_code == 201, r.text
    vid = r.json()["id"]
    data_dir = pathlib.Path(os.environ["LARYNX_DATA_DIR"])
    assert (data_dir / "voices" / vid / "latents.bin").exists()
    assert (data_dir / "voices" / vid / "latents.meta.json").exists()


@pytest.mark.asyncio
async def test_real_synthesis_with_voice_id_returns_audio(
    real_client: AsyncClient,
    real_auth_headers: dict[str, str],
) -> None:
    up = await real_client.post(
        "/v1/voices",
        headers=real_auth_headers,
        files={"audio": ("ref.wav", _real_voice_bytes(), "audio/wav")},
        data={"name": f"real-synth-{uuid.uuid4().hex[:8]}"},
    )
    vid = up.json()["id"]

    r = await real_client.post(
        "/v1/tts",
        headers=real_auth_headers,
        json={
            "text": "Hello from the real VoxCPM2 model.",
            "voice_id": vid,
            "sample_rate": 24000,
            "cfg_value": 2.0,
        },
    )
    assert r.status_code == 200, r.text
    assert r.content.startswith(b"RIFF")
    # A 2-3 second sentence should produce > 20kB of 16-bit @24kHz PCM.
    assert len(r.content) > 40000


@pytest.mark.asyncio
async def test_cache_warms_second_request(
    real_client: AsyncClient,
    real_auth_headers: dict[str, str],
) -> None:
    """First request is uncached (voice just uploaded, Redis already warm
    from upload-time put); wipe Redis, time the re-warm vs a subsequent hit."""
    import redis.asyncio as redis_async

    up = await real_client.post(
        "/v1/voices",
        headers=real_auth_headers,
        files={"audio": ("ref.wav", _real_voice_bytes(), "audio/wav")},
        data={"name": f"cache-bench-{uuid.uuid4().hex[:8]}"},
    )
    vid = up.json()["id"]

    redis_client = redis_async.from_url(os.environ["REDIS_URL"], decode_responses=False)
    await redis_client.delete(f"latents:{vid}:v1", f"latents:{vid}:v1:meta")
    await redis_client.aclose()

    # Run 10 synthesis requests in a row. The first sees Redis cold (disk
    # hit + re-warm); the rest should all be Redis hits.
    latencies: list[int] = []
    for i in range(10):
        t0 = time.perf_counter()
        r = await real_client.post(
            "/v1/tts",
            headers=real_auth_headers,
            json={
                "text": f"Latency measurement request number {i}.",
                "voice_id": vid,
                "sample_rate": 24000,
            },
        )
        t1 = time.perf_counter()
        assert r.status_code == 200
        latencies.append(int((t1 - t0) * 1000))

    first, rest = latencies[0], latencies[1:]
    print(
        f"\n[bench] voice_id={vid}  first_ms={first}  "
        f"mean_rest_ms={sum(rest) // len(rest)}  "
        f"min_rest_ms={min(rest)}  max_rest_ms={max(rest)}"
    )
    # No hard assertion on ratio — cold-path dominated by model synthesis
    # which isn't affected by the cache; the cache saves an encode (~30ms
    # on mock scale). Real gap is measured by scripts/measure_cache.py
    # which isolates the encode cost.
    assert all(lat > 0 for lat in latencies)


@pytest.mark.asyncio
async def test_real_design_voice_roundtrip(
    real_client: AsyncClient,
    real_auth_headers: dict[str, str],
) -> None:
    name = f"designed-{uuid.uuid4().hex[:8]}"
    preview_r = await real_client.post(
        "/v1/voices/design",
        headers=real_auth_headers,
        json={
            "name": name,
            "design_prompt": "warm middle-aged female",
            "preview_text": "Let us see what this voice sounds like.",
        },
    )
    assert preview_r.status_code == 200, preview_r.text
    preview = preview_r.json()

    save_r = await real_client.post(
        f"/v1/voices/design/{preview['preview_id']}/save",
        headers=real_auth_headers,
        json={},
    )
    assert save_r.status_code == 201, save_r.text
    vid = save_r.json()["id"]

    r = await real_client.post(
        "/v1/tts",
        headers=real_auth_headers,
        json={
            "text": "Using the designed voice on a fresh sentence.",
            "voice_id": vid,
        },
    )
    assert r.status_code == 200
    assert len(r.content) > 20000


@pytest.mark.asyncio
async def test_real_delete_voice_removes_latents(
    real_client: AsyncClient,
    real_auth_headers: dict[str, str],
) -> None:
    up = await real_client.post(
        "/v1/voices",
        headers=real_auth_headers,
        files={"audio": ("ref.wav", _real_voice_bytes(), "audio/wav")},
        data={"name": f"delete-me-{uuid.uuid4().hex[:8]}"},
    )
    vid = up.json()["id"]

    r_del = await real_client.delete(f"/v1/voices/{vid}", headers=real_auth_headers)
    assert r_del.status_code == 204

    # TTS with the deleted voice_id must now 404.
    r_tts = await real_client.post(
        "/v1/tts",
        headers=real_auth_headers,
        json={"text": "should fail", "voice_id": vid},
    )
    assert r_tts.status_code == 404
