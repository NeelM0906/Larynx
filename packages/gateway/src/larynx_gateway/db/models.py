"""SQLAlchemy models.

M1 defines only ``Voice`` because the alembic migration needs it; the table
is read/written by the voice-library endpoints which land in M2.
Batch/fine-tune models are deferred to their milestones.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Voice(Base):
    __tablename__ = "voices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)

    # Populated by M2 voice upload; path to the raw reference audio on disk.
    ref_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Path to the cached VoxCPM VAE latents (PRD §6 "Latent caching").
    latent_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
