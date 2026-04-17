"""SQLAlchemy models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Application-level enum. Kept as ``typing.Literal`` (not SQL ENUM) so the
# migration story stays simple — the DB column is plain VARCHAR(32) and a
# new variant (e.g. M7's ``'lora'``) lands without an ALTER TYPE.
VoiceSource = Literal["uploaded", "designed", "seed", "lora"]


class Base(DeclarativeBase):
    pass


class Voice(Base):
    """Persistent voice record.

    Columns populated at different points in the voice lifecycle:
    - ``ref_audio_path`` / ``latent_path`` / ``sample_rate_hz`` / ``duration_ms``:
      set when M2 upload encodes a reference.
    - ``prompt_text``: set when the voice was created with the "ultimate
      cloning" mode (transcript of the reference audio).
    - ``design_prompt``: set for voices created via POST /v1/voices/design.
    - ``source``: see :data:`VoiceSource` — distinguishes provenance.
    """

    __tablename__ = "voices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)
    source: Mapped[VoiceSource] = mapped_column(String(32), nullable=False, default="uploaded")

    ref_audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    latent_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    prompt_text: Mapped[str | None] = mapped_column(Text(), nullable=True)
    design_prompt: Mapped[str | None] = mapped_column(Text(), nullable=True)

    sample_rate_hz: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer(), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
