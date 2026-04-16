"""SQLAlchemy engine + session factory."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _to_async_url(url: str) -> str:
    """Rewrite sync drivers to their async equivalents.

    pydantic-settings reads DATABASE_URL as-is; .env.example uses
    ``postgresql+psycopg`` which is async-capable (psycopg3). For sqlite URLs
    used in tests, rewrite to ``aiosqlite`` if needed.
    """
    if url.startswith("sqlite:///") or url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


def init_engine(database_url: str) -> AsyncEngine:
    global _engine, _sessionmaker
    _engine = create_async_engine(_to_async_url(database_url), pool_pre_ping=True)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


async def get_session() -> AsyncIterator[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("session factory not initialised (lifespan did not run?)")
    async with _sessionmaker() as session:
        yield session
