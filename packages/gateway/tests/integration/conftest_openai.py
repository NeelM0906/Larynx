"""Fixtures for OpenAI-SDK round-trip tests.

The upstream ``openai`` Python client needs a real HTTP URL — it won't
accept ASGITransport via its public API — so we spin up the gateway
under uvicorn on a random loopback port and hand its URL back to the
test.

Loaded only by the two ``test_openai_sdk_speech*.py`` test modules via
``pytest_plugins`` so we don't introduce a uvicorn dependency into the
default test session. The top-level ``conftest.py`` still handles
Postgres / Redis reachability via its autouse session fixture.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text

# Duplicated from ``packages/gateway/tests/conftest.py`` to avoid a
# package-relative import — pytest's conftest files aren't guaranteed
# to be importable as normal modules outside the runner. Keep the two
# lists in sync; if the test DB URL or token changes in the top-level
# conftest, update here too.
TEST_TOKEN = "test-token-please-ignore"
TEST_DB_URL_SQLA = "postgresql+psycopg://larynx:larynx@localhost:5433/larynx_test"
TEST_REDIS_URL = "redis://localhost:6380/14"

_OPENAI_SDK_AVAILABLE: bool
try:
    import openai  # noqa: F401

    _OPENAI_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENAI_SDK_AVAILABLE = False


def _redis_reachable() -> bool:
    """Probe Redis by opening a TCP connection to localhost:6380.

    The main conftest's Postgres probe already short-circuits the whole
    session; we layer a Redis check on top so these tests skip cleanly
    if Redis has gone away between boot and test run.
    """
    host, port = "localhost", 6380
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reset_test_db_voices() -> None:
    """Wipe voices between OpenAI-SDK tests. Mirrors ``conftest._reset_test_db``."""
    eng = create_engine(TEST_DB_URL_SQLA, isolation_level="AUTOCOMMIT")
    with eng.connect() as conn:
        conn.execute(text("TRUNCATE TABLE voices RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE fine_tune_jobs RESTART IDENTITY CASCADE"))
    eng.dispose()


@pytest.fixture(scope="module")
def live_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, str]]:
    """Boot the FastAPI app under uvicorn on 127.0.0.1:<random_port>.

    Yields ``(base_url, bearer_token)`` and tears down cleanly by
    flipping ``server.should_exit`` and joining the thread. The fixture
    is module-scoped so both test files in the SDK matrix share a
    single gateway — startup is dominated by VoxCPM mock boot.
    """
    if not _OPENAI_SDK_AVAILABLE:
        pytest.skip("openai SDK not installed; `uv sync` pulls it into the dev group")
    if not _redis_reachable():
        pytest.skip("Redis not reachable at localhost:6380; docker compose up -d")

    _reset_test_db_voices()

    data_dir = tmp_path_factory.mktemp("openai-sdk-data")
    os.environ["LARYNX_API_TOKEN"] = TEST_TOKEN
    os.environ["LARYNX_TTS_MODE"] = "mock"
    os.environ["LARYNX_STT_MODE"] = "mock"
    os.environ["LARYNX_VAD_PUNC_MODE"] = "mock"
    os.environ["LARYNX_LOG_JSON"] = "false"
    os.environ["LARYNX_DATA_DIR"] = str(data_dir)
    os.environ["DATABASE_URL"] = TEST_DB_URL_SQLA
    os.environ["REDIS_URL"] = TEST_REDIS_URL

    import uvicorn

    from larynx_gateway.config import get_settings
    from larynx_gateway.main import create_app

    get_settings.cache_clear()
    app = create_app()

    port = _pick_free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)

    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, name="openai-sdk-uvicorn", daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    _wait_for_health(base_url, timeout_s=30.0)

    try:
        yield base_url, TEST_TOKEN
    finally:
        server.should_exit = True
        # uvicorn's serve() returns once should_exit flips; wait for the
        # thread to exit so we don't leak a port + an asyncio loop.
        thread.join(timeout=10.0)
        try:
            loop.close()
        except Exception:
            pass


def _wait_for_health(base_url: str, timeout_s: float) -> None:
    """Poll /health until 200 or timeout."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_s
    last_err: BaseException | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
        time.sleep(0.25)
    raise RuntimeError(f"gateway never went healthy at {base_url}: last_err={last_err!r}")


@pytest.fixture
def seed_voice_alloy(live_gateway: tuple[str, str]) -> str:
    """Upload a synthetic WAV under the name ``alloy`` and return the id.

    We reuse the existing upload path rather than re-running the real
    ``scripts/load_demo_voices.py`` — the script fetches LibriVox mp3s
    off the network, which is not a thing we want on every test run.
    The resulting Voice has ``source='uploaded'`` and a small synthetic
    reference clip; the mock VoxCPM worker only cares that latents
    exist.
    """
    base_url, token = live_gateway

    import httpx

    # Check idempotency — module-scoped fixture means earlier tests may
    # have already uploaded it.
    r = httpx.get(
        f"{base_url}/v1/voices",
        headers={"Authorization": f"Bearer {token}"},
        params={"limit": 500},
        timeout=10,
    )
    r.raise_for_status()
    for v in r.json()["voices"]:
        if v["name"] == "alloy":
            return v["id"]

    import io

    import numpy as np
    import soundfile as sf

    sr = 24000
    samples = (np.random.default_rng(1).standard_normal(sr * 2) * 0.05).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    audio = buf.getvalue()

    u = httpx.post(
        f"{base_url}/v1/voices",
        headers={"Authorization": f"Bearer {token}"},
        files={"audio": ("alloy.wav", audio, "audio/wav")},
        data={"name": "alloy", "description": "SDK-test seed voice"},
        timeout=30,
    )
    u.raise_for_status()
    return u.json()["id"]


