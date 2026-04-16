"""Alembic environment.

Uses the sync psycopg driver for migrations (alembic runs outside the app's
event loop). Reads the URL from ``DATABASE_URL`` so dev/CI/prod all share
one source of truth.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from larynx_gateway.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Rewrite the URL to a sync driver for alembic. Our app uses psycopg's async
# path at runtime; alembic migrations are one-shot scripts, so sync is fine.
raw_url = os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url") or ""
sync_url = raw_url.replace("postgresql+asyncpg://", "postgresql+psycopg://").replace(
    "sqlite+aiosqlite://", "sqlite://"
)
config.set_main_option("sqlalchemy.url", sync_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=sync_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
