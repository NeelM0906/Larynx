"""extend voices table for M2

Revision ID: 0002_extend_voices
Revises: 0001_create_voices
Create Date: 2026-04-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_extend_voices"
down_revision: str | None = "0001_create_voices"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing rows (none in prod yet but the migration is written defensively)
    # get source='uploaded'. We drop the server_default afterwards so new rows
    # are required to specify a source explicitly via the Voice model.
    op.add_column(
        "voices",
        sa.Column("source", sa.String(length=32), nullable=False, server_default="uploaded"),
    )
    op.alter_column("voices", "source", server_default=None)

    op.add_column("voices", sa.Column("prompt_text", sa.Text(), nullable=True))
    op.add_column("voices", sa.Column("design_prompt", sa.Text(), nullable=True))
    op.add_column("voices", sa.Column("sample_rate_hz", sa.Integer(), nullable=True))
    op.add_column("voices", sa.Column("duration_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("voices", "duration_ms")
    op.drop_column("voices", "sample_rate_hz")
    op.drop_column("voices", "design_prompt")
    op.drop_column("voices", "prompt_text")
    op.drop_column("voices", "source")
