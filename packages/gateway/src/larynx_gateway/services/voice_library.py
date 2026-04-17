"""Voice library — upload, list, get, delete, design.

High-level responsibilities:
- Persist Voice metadata to Postgres.
- Write reference audio to disk.
- Encode reference -> latents (via the voxcpm worker) and populate the
  two-tier cache at upload time so the FIRST synthesis is fast.
- For design previews: render with the parenthetical prompt, encode the
  rendered audio as the reference, and stash everything under a
  preview_id with TTL. On save, promote it to a real Voice row + files.

The cache hit path on synthesis is:
    gateway -> LatentCache.get(voice_id) -> hands bytes to worker
Misses fall back to disk, which re-populates Redis. A true miss (no
disk file) means the voice was never uploaded or was deleted — the
route surface maps that to 404.
"""

from __future__ import annotations

import io
import json
import pathlib
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import soundfile as sf
import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from larynx_gateway.db.models import Voice
from larynx_gateway.services.latent_cache import LatentCache, LatentMetadata
from larynx_gateway.services.voice_files import DesignPreviewFiles, VoiceFiles
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient

log = structlog.get_logger(__name__)


class VoiceLibraryError(Exception):
    """Raised for business-logic failures (duplicate name, bad audio, etc.)."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True)
class UploadedVoice:
    voice: Voice
    files: VoiceFiles
    latent_meta: LatentMetadata


@dataclass(frozen=True)
class DesignPreview:
    preview_id: str
    name: str
    description: str | None
    design_prompt: str
    preview_text: str
    preview_audio: bytes
    sample_rate: int
    duration_ms: int
    expires_at: float  # epoch seconds


class VoiceLibrary:
    """Service layer. Routes instantiate one per request via deps.py."""

    def __init__(
        self,
        session: AsyncSession,
        voxcpm: VoxCPMClient,
        cache: LatentCache,
        data_dir: pathlib.Path,
        design_ttl_s: int,
    ) -> None:
        self._session = session
        self._voxcpm = voxcpm
        self._cache = cache
        self._data_dir = data_dir
        self._design_ttl_s = design_ttl_s

    # -- upload --------------------------------------------------------------

    async def upload(
        self,
        *,
        name: str,
        description: str | None,
        audio: bytes,
        wav_format: str = "wav",
        prompt_text: str | None = None,
        source: str = "uploaded",
    ) -> UploadedVoice:
        if not audio:
            raise VoiceLibraryError("empty_audio", "reference audio must not be empty")

        voice_id = uuid.uuid4().hex
        files = VoiceFiles(voice_id=voice_id, root=self._data_dir)

        # Decode once to capture sample_rate + duration + normalise to WAV
        # on disk (so re-encoding in the cache is always WAV regardless of
        # upload format).
        try:
            samples, sr = sf.read(io.BytesIO(audio), always_2d=False)
        except Exception as e:
            raise VoiceLibraryError(
                "unreadable_audio", f"could not decode audio: {e}", status=400
            ) from e
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = np.asarray(samples, dtype=np.float32)
        duration_ms = int(1000 * len(samples) / sr)
        if duration_ms < 500:
            raise VoiceLibraryError(
                "audio_too_short",
                f"reference audio must be ≥ 0.5s, got {duration_ms}ms",
            )

        # Encode reference via the worker BEFORE we commit the Voice row,
        # so a bad audio file doesn't leave an orphan DB row behind.
        enc = await self._voxcpm.encode_reference(audio, wav_format=wav_format)
        latent_meta = LatentMetadata(
            voice_id=voice_id,
            feat_dim=enc.feat_dim,
            encoder_sample_rate=enc.encoder_sample_rate,
            num_frames=enc.num_frames,
        )

        # Write audio + latents + insert row. If the insert fails (e.g.
        # duplicate name), delete the on-disk artifacts so we don't orphan.
        files.ensure_dir()
        # Normalise to a canonical 16-bit WAV. Both mock and real mode
        # accept this format back via encode_reference.
        sf.write(str(files.reference_audio), samples, sr, subtype="PCM_16")
        await self._cache.put(voice_id, enc.latents, latent_meta)

        voice = Voice(
            id=voice_id,
            name=name,
            description=description,
            source=source,
            ref_audio_path=str(files.reference_audio),
            latent_path=str(files.latents),
            prompt_text=prompt_text,
            sample_rate_hz=int(sr),
            duration_ms=duration_ms,
        )
        self._session.add(voice)
        try:
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            # Clean up files + cache so the next attempt is fresh.
            files.delete()
            await self._cache.delete(voice_id)
            raise VoiceLibraryError(
                "duplicate_name",
                f"a voice named {name!r} already exists",
                status=409,
            ) from e
        # Pull server-default columns (created_at / updated_at) back into the
        # Python object so response serialisation doesn't trigger lazy IO.
        await self._session.refresh(voice)

        log.info(
            "voice.uploaded",
            voice_id=voice_id,
            name=name,
            sample_rate=sr,
            duration_ms=duration_ms,
            source=source,
        )
        return UploadedVoice(voice=voice, files=files, latent_meta=latent_meta)

    # -- list / get / delete ------------------------------------------------

    async def list(self, *, limit: int = 50, offset: int = 0) -> tuple[list[Voice], int]:
        stmt = select(Voice).order_by(Voice.created_at.desc()).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        voices = list(result.scalars())
        count_stmt = select(Voice.id)
        total = len((await self._session.execute(count_stmt)).scalars().all())
        return voices, total

    async def get(self, voice_id: str) -> Voice | None:
        return await self._session.get(Voice, voice_id)

    async def get_by_name(self, name: str) -> Voice | None:
        stmt = select(Voice).where(Voice.name == name)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def delete(self, voice_id: str) -> bool:
        voice = await self.get(voice_id)
        if voice is None:
            return False
        await self._session.delete(voice)
        await self._session.commit()
        files = VoiceFiles(voice_id=voice_id, root=self._data_dir)
        files.delete()
        await self._cache.delete(voice_id)
        log.info("voice.deleted", voice_id=voice_id, name=voice.name)
        return True

    # -- latents on synthesis ------------------------------------------------

    async def get_latents_for_synthesis(self, voice_id: str) -> bytes | None:
        """Return cached latents for a voice, or None if the voice doesn't
        exist. Used by POST /v1/tts when voice_id is provided."""
        voice = await self.get(voice_id)
        if voice is None:
            return None
        hit = await self._cache.get(voice_id)
        if hit is not None:
            return hit.latents
        # Cache miss despite the voice existing — reference audio must be
        # present on disk (we wrote it at upload). Re-encode and re-warm.
        files = VoiceFiles(voice_id=voice_id, root=self._data_dir)
        if not files.reference_audio.exists():
            log.error(
                "voice.reference_missing",
                voice_id=voice_id,
                path=str(files.reference_audio),
            )
            return None
        audio = files.reference_audio.read_bytes()
        enc = await self._voxcpm.encode_reference(audio, wav_format="wav")
        meta = LatentMetadata(
            voice_id=voice_id,
            feat_dim=enc.feat_dim,
            encoder_sample_rate=enc.encoder_sample_rate,
            num_frames=enc.num_frames,
        )
        await self._cache.put(voice_id, enc.latents, meta)
        log.info("voice.latents_reencoded", voice_id=voice_id)
        return enc.latents

    # -- design --------------------------------------------------------------

    async def create_design_preview(
        self,
        *,
        name: str,
        description: str | None,
        design_prompt: str,
        preview_text: str,
    ) -> DesignPreview:
        """Render a preview using VoxCPM2's parenthetical voice-design syntax.

        The prompt is prepended to the synthesis text (the model picks up
        the parenthetical as a speaker-style directive). The rendered audio
        becomes the reference for the persistent voice if the user saves it.
        """
        target_text = f"({design_prompt}) {preview_text}"

        # Render to get the preview audio.
        synth = await self._voxcpm.synthesize_text(
            text=target_text,
            sample_rate=24000,
            cfg_value=2.0,
        )

        # Save the rendered PCM to disk as a canonical WAV for the preview.
        pcm16 = np.frombuffer(synth.pcm_s16le, dtype=np.int16)
        preview_id = uuid.uuid4().hex
        preview_files = DesignPreviewFiles(preview_id=preview_id, root=self._data_dir)
        preview_files.ensure_dir()
        sf.write(
            str(preview_files.preview_audio),
            pcm16.astype(np.float32) / 32768.0,
            synth.sample_rate,
            subtype="PCM_16",
        )

        # Encode the preview audio -> latents now so save-time promotion
        # doesn't require another worker round-trip.
        preview_wav_bytes = preview_files.preview_audio.read_bytes()
        enc = await self._voxcpm.encode_reference(preview_wav_bytes, wav_format="wav")
        preview_files.latents.write_bytes(enc.latents)
        meta = LatentMetadata(
            voice_id=preview_id,
            feat_dim=enc.feat_dim,
            encoder_sample_rate=enc.encoder_sample_rate,
            num_frames=enc.num_frames,
        )
        preview_files.latents_meta.write_text(meta.to_json())

        design_meta = {
            "preview_id": preview_id,
            "name": name,
            "description": description,
            "design_prompt": design_prompt,
            "preview_text": preview_text,
            "sample_rate": synth.sample_rate,
            "duration_ms": synth.duration_ms,
            "created_at": datetime.now(UTC).isoformat(),
        }
        preview_files.metadata_json.write_text(json.dumps(design_meta))

        log.info(
            "voice.design_preview",
            preview_id=preview_id,
            name=name,
            duration_ms=synth.duration_ms,
        )
        return DesignPreview(
            preview_id=preview_id,
            name=name,
            description=description,
            design_prompt=design_prompt,
            preview_text=preview_text,
            preview_audio=preview_wav_bytes,
            sample_rate=synth.sample_rate,
            duration_ms=synth.duration_ms,
            expires_at=time.time() + self._design_ttl_s,
        )

    def load_design_preview(self, preview_id: str) -> DesignPreviewFiles | None:
        preview = DesignPreviewFiles(preview_id=preview_id, root=self._data_dir)
        if not preview.metadata_json.exists():
            return None
        return preview

    async def save_design(
        self,
        preview_id: str,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
    ) -> UploadedVoice:
        preview = self.load_design_preview(preview_id)
        if preview is None:
            raise VoiceLibraryError(
                "preview_not_found",
                f"no design preview with id {preview_id!r}",
                status=404,
            )
        design_meta = json.loads(preview.metadata_json.read_text())
        name = name_override or design_meta["name"]
        description = description_override or design_meta.get("description")

        audio_bytes = preview.preview_audio.read_bytes()
        uploaded = await self.upload(
            name=name,
            description=description,
            audio=audio_bytes,
            wav_format="wav",
            source="designed",
        )

        # Annotate the voice with the design prompt for provenance.
        uploaded.voice.design_prompt = design_meta["design_prompt"]
        await self._session.commit()
        await self._session.refresh(uploaded.voice)

        preview.delete()
        log.info("voice.design_saved", voice_id=uploaded.voice.id, preview_id=preview_id)
        return uploaded
