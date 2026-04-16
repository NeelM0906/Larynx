"""create voices table

Revision ID: 0001_create_voices
Revises:
Create Date: 2026-04-16
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_create_voices"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "voices",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("ref_audio_path", sa.String(length=1024), nullable=True),
        sa.Column("latent_path", sa.String(length=1024), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_voices_name", "voices", ["name"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_voices_name", table_name="voices")
    op.drop_table("voices")
