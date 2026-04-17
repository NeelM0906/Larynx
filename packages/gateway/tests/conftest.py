"""Shared test fixtures.

Boot the gateway with LARYNX_TTS_MODE=mock (fast) against real Postgres
and real Redis from docker-compose. Per the repo's "no fakes" rule —
if either service isn't reachable, tests skip loudly rather than fall
back to aiosqlite / fakeredis.

Test data lives in dedicated databases so the app DB is never touched:
- Postgres: larynx_test (created + migrated once per session)
- Redis: db 15
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

TEST_TOKEN = "test-token-please-ignore"
ADMIN_DB_URL = "postgresql+psycopg://larynx:larynx@localhost:5433/larynx"
TEST_DB_NAME = "larynx_test"
TEST_DB_URL_SQLA = f"postgresql+psycopg://larynx:larynx@localhost:5433/{TEST_DB_NAME}"
TEST_REDIS_URL = "redis://localhost:6380/14"  # db 14 for app; 15 reserved for unit tests


def _postgres_reachable() -> bool:
    try:
        eng = create_engine(ADMIN_DB_URL, isolation_level="AUTOCOMMIT", pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


def _ensure_test_db() -> None:
    """Create the test DB if missing, then run alembic migrations against it.

    Runs synchronously at module import via `_session_env` so tests don't
    each pay the setup cost.
    """
    admin = create_engine(ADMIN_DB_URL, isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    with admin.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": TEST_DB_NAME}
        ).first()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    admin.dispose()

    # Run migrations via the alembic command API so test schema stays in
    # lockstep with what prod will see.
    from alembic import command
    from alembic.config import Config

    ini_path = (
        pathlib.Path(__file__).resolve().parents[1] / "alembic.ini"
    )  # packages/gateway/alembic.ini
    alembic_cfg = Config(str(ini_path))
    alembic_cfg.set_main_option(
        "script_location",
        str(ini_path.parent / "src/larynx_gateway/db/migrations"),
    )
    os.environ["DATABASE_URL"] = make_url(TEST_DB_URL_SQLA).render_as_string(hide_password=False)
    command.upgrade(alembic_cfg, "head")


def _reset_test_db() -> None:
    """Truncate tables between tests so state doesn't leak.

    Much faster than dropping + recreating + remigrating per test.
    """
    eng = create_engine(TEST_DB_URL_SQLA, isolation_level="AUTOCOMMIT")
    with eng.connect() as conn:
        conn.execute(text("TRUNCATE TABLE voices RESTART IDENTITY CASCADE"))
    eng.dispose()


@pytest.fixture(autouse=True, scope="session")
def _session_env() -> None:
    if not _postgres_reachable():
        pytest.skip(
            "Postgres not reachable at localhost:5433. Run `docker compose up -d`.",
            allow_module_level=True,
        )
    _ensure_test_db()

    os.environ["LARYNX_API_TOKEN"] = TEST_TOKEN
    os.environ["LARYNX_TTS_MODE"] = "mock"
    os.environ["LARYNX_STT_MODE"] = "mock"
    os.environ["LARYNX_VAD_PUNC_MODE"] = "mock"
    os.environ["LARYNX_LOG_JSON"] = "false"
    os.environ["DATABASE_URL"] = TEST_DB_URL_SQLA
    os.environ["REDIS_URL"] = TEST_REDIS_URL


@pytest.fixture
def data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Isolated data dir per test — prevents voice-file collisions."""
    d = tmp_path / "larynx-data"
    d.mkdir()
    os.environ["LARYNX_DATA_DIR"] = str(d)
    return d


@pytest_asyncio.fixture
async def client(data_dir: pathlib.Path) -> AsyncIterator[AsyncClient]:
    _reset_test_db()

    # Re-import to pick up the fresh env overrides + DATA_DIR.
    from larynx_gateway.config import get_settings
    from larynx_gateway.main import create_app

    get_settings.cache_clear()
    app = create_app()

    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as c,
        app.router.lifespan_context(app),
    ):
        yield c


@pytest_asyncio.fixture
async def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TEST_TOKEN}"}
