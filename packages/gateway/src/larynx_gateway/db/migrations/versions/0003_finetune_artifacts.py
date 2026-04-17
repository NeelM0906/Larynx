"""add fine-tune artifacts for M7

- voices gains ``lora_path`` and ``ft_job_id`` (application-level link;
  no FK — see ORCHESTRATION-M7.md §4.1).
- new ``fine_tune_jobs`` table — the source of truth for job state,
  progress, and error reporting.

No DB-level enum is added for ``voices.source``; ``'lora'`` is a
string value validated at the application level via the
``VoiceSource`` Literal.

Revision ID: 0003_finetune_artifacts
Revises: 0002_extend_voices
Create Date: 2026-04-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_finetune_artifacts"
down_revision: str | None = "0002_extend_voices"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("voices", sa.Column("lora_path", sa.String(length=1024), nullable=True))
    op.add_column("voices", sa.Column("ft_job_id", sa.String(length=36), nullable=True))
    op.create_index("idx_voices_ft_job_id", "voices", ["ft_job_id"])

    op.create_table(
        "fine_tune_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("voice_id", sa.String(length=36), nullable=True),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("resolved_config_path", sa.String(length=1024), nullable=False),
        sa.Column("log_key", sa.String(length=128), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("current_step", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_steps", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_ftjobs_state", "fine_tune_jobs", ["state"])
    op.create_index("idx_ftjobs_voice_id", "fine_tune_jobs", ["voice_id"])


def downgrade() -> None:
    op.drop_index("idx_ftjobs_voice_id", table_name="fine_tune_jobs")
    op.drop_index("idx_ftjobs_state", table_name="fine_tune_jobs")
    op.drop_table("fine_tune_jobs")

    op.drop_index("idx_voices_ft_job_id", table_name="voices")
    op.drop_column("voices", "ft_job_id")
    op.drop_column("voices", "lora_path")
