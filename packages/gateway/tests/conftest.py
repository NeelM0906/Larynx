"""Test fixtures — boot the gateway with mock TTS and an in-memory SQLite DB."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

TEST_TOKEN = "test-token-please-ignore"


@pytest.fixture(autouse=True, scope="session")
def _test_env() -> None:
    os.environ["LARYNX_API_TOKEN"] = TEST_TOKEN
    os.environ["LARYNX_TTS_MODE"] = "mock"
    os.environ["LARYNX_LOG_JSON"] = "false"
    # aiosqlite is optional; the gateway's init_engine is lazy about actually
    # connecting, so a URL that never gets dialed is fine for M1 tests.
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    # Re-import to pick up the session-scoped env overrides cleanly.
    from larynx_gateway.config import get_settings
    from larynx_gateway.main import create_app

    get_settings.cache_clear()  # pydantic-settings caches; drop it
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
