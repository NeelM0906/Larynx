"""Filesystem layout for voice assets.

Per PRD §5.5 / §6, every voice has three persistent artifacts:

    {DATA_DIR}/voices/{voice_id}/
        reference.wav        -- the original uploaded audio (or the
                                rendered preview for designed voices)
        latents.bin          -- cached VoxCPM2 ref-audio latents
        latents.meta.json    -- cache metadata sidecar (feat_dim,
                                encoder_sample_rate, num_frames)

Design previews live at {DATA_DIR}/voice_designs/{preview_id}/ and are
cleaned up after LARYNX_VOICE_DESIGN_TTL_S unless promoted via
POST /v1/voices/design/{preview_id}/save, which moves the directory
under voices/ with a freshly-minted voice_id.
"""

from __future__ import annotations

import pathlib
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceFiles:
    """Path resolver for one voice. All paths absolute."""

    voice_id: str
    root: pathlib.Path

    @property
    def dir(self) -> pathlib.Path:
        return self.root / "voices" / self.voice_id

    @property
    def reference_audio(self) -> pathlib.Path:
        return self.dir / "reference.wav"

    @property
    def latents(self) -> pathlib.Path:
        return self.dir / "latents.bin"

    @property
    def latents_meta(self) -> pathlib.Path:
        return self.dir / "latents.meta.json"

    def ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def delete(self) -> None:
        if self.dir.exists():
            shutil.rmtree(self.dir)


@dataclass(frozen=True)
class DesignPreviewFiles:
    preview_id: str
    root: pathlib.Path

    @property
    def dir(self) -> pathlib.Path:
        return self.root / "voice_designs" / self.preview_id

    @property
    def preview_audio(self) -> pathlib.Path:
        return self.dir / "preview.wav"

    @property
    def latents(self) -> pathlib.Path:
        return self.dir / "latents.bin"

    @property
    def latents_meta(self) -> pathlib.Path:
        return self.dir / "latents.meta.json"

    @property
    def metadata_json(self) -> pathlib.Path:
        return self.dir / "design.json"

    def ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def delete(self) -> None:
        if self.dir.exists():
            shutil.rmtree(self.dir)


def resolve_data_dir(data_dir: str | pathlib.Path) -> pathlib.Path:
    """Expand + resolve DATA_DIR once at app startup so every caller sees
    the same absolute path regardless of the cwd of the worker / migration
    / request handler."""
    return pathlib.Path(data_dir).expanduser().resolve()
