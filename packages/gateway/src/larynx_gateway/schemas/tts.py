"""Request / response schemas for /v1/tts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

OutputFormat = Literal["wav", "pcm16"]  # mp3 lands behind ffmpeg later


class TTSRequest(BaseModel):
    """JSON body for POST /v1/tts.

    Covers text-only synthesis plus the voice_id cloning path. The
    ad-hoc multipart `reference_audio` / `prompt_audio` cloning paths
    use POST /v1/tts with multipart form data (see routes/tts.py).
    """

    text: str = Field(min_length=1, max_length=5000)

    # Cloning via a pre-saved voice (library lookup -> cached latents).
    voice_id: str | None = None

    # Cloning via ad-hoc prompt audio that we pre-encoded. Rarely used
    # from JSON — callers usually upload the audio in the multipart path.
    # Left on the schema so SDK clients that cache encoded latents
    # locally (e.g. long-running worker) can bypass the upload.
    prompt_text: str | None = Field(default=None, max_length=500)

    # Engine controls. inference_timesteps moved to config — see PRD §5.1.
    sample_rate: int = Field(default=24000, ge=8000, le=48000)
    output_format: OutputFormat = "wav"
    cfg_value: float = Field(default=2.0, ge=0.0, le=10.0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)

    # Language hint — VoxCPM2 currently auto-detects; this is accepted
    # for forward compatibility with M3's STT-side language routing.
    language: str | None = None
