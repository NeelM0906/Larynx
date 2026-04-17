"""add batch TTS jobs + items for M8

Part A of the M8 milestone — see ORCHESTRATION-M8.md §1.1.

Two new tables:
- ``batch_jobs`` — one row per submitted batch. Tracks lifecycle state,
  progress counters, retention policy, and the expiry timestamp that
  the daily cleanup cron acts on.
- ``batch_items`` — one row per (job, item_idx) tuple. Owns the
  per-item state, the TTS parameter snapshot, and the generated
  artifact path.

Application-level FK only on ``dataset_id``-style links elsewhere in
this schema; here we *do* add a DB FK from items -> jobs with CASCADE
because the lifecycle really is coupled (cleaning up a job must take
its items with it, unconditionally, so a cron sweep never leaves
dangling items).

Revision ID: 0004_batch_jobs
Revises: 0003_finetune_artifacts
Create Date: 2026-04-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_batch_jobs"
down_revision: str | None = "0003_finetune_artifacts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("num_items", sa.Integer(), nullable=False),
        sa.Column("num_completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("num_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retain", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        # expires_at is computed at create time (created_at + 7d) unless
        # retain=True. Storing it explicitly lets the cleanup cron do a
        # simple range scan instead of computing now()-interval in SQL.
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_batch_jobs_state", "batch_jobs", ["state"])
    op.create_index("idx_batch_jobs_expires_at", "batch_jobs", ["expires_at"])

    op.create_table(
        "batch_items",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "job_id",
            sa.String(length=36),
            sa.ForeignKey("batch_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("item_idx", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("voice_id", sa.String(length=36), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        # JSON-encoded snapshot of the submitted engine params. Kept as
        # TEXT (not JSONB) to match the rest of the schema's convention
        # of app-side serialisation — see FineTuneJob.config_json.
        sa.Column("params_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("output_path", sa.String(length=1024), nullable=True),
        sa.Column("output_format", sa.String(length=16), nullable=True),
        sa.Column("sample_rate", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("generation_time_ms", sa.Integer(), nullable=True),
        sa.Column("etag", sa.String(length=64), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("job_id", "item_idx", name="uq_batch_items_job_idx"),
    )
    op.create_index("idx_batch_items_job_id", "batch_items", ["job_id"])
    op.create_index("idx_batch_items_state", "batch_items", ["state"])


def downgrade() -> None:
    op.drop_index("idx_batch_items_state", table_name="batch_items")
    op.drop_index("idx_batch_items_job_id", table_name="batch_items")
    op.drop_table("batch_items")

    op.drop_index("idx_batch_jobs_expires_at", table_name="batch_jobs")
    op.drop_index("idx_batch_jobs_state", table_name="batch_jobs")
    op.drop_table("batch_jobs")
