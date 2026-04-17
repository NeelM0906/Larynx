"""Synthesis orchestrator.

Resolves voice conditioning (voice_id -> cached latents, OR inline
reference audio -> one-off encode), calls the worker, packages the
result into the caller's requested container (WAV / PCM16).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from larynx_shared.audio import pack_wav

from larynx_gateway.schemas.tts import OutputFormat, TTSRequest
from larynx_gateway.services.voice_library import VoiceLibrary
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient


@dataclass(frozen=True)
class TTSResult:
    audio: bytes
    content_type: str
    sample_rate: int
    duration_ms: int
    generation_time_ms: int
    voice_id: str


@dataclass(frozen=True)
class ResolvedConditioning:
    ref_audio_latents: bytes | None = None
    prompt_audio_latents: bytes | None = None
    prompt_text: str = ""


async def resolve_conditioning(
    req: TTSRequest,
    library: VoiceLibrary,
    *,
    inline_reference_audio: bytes | None = None,
    inline_prompt_audio: bytes | None = None,
    inline_prompt_text: str | None = None,
    voxcpm: VoxCPMClient,
) -> ResolvedConditioning | None:
    """Figure out which latents to send the worker.

    Precedence (stricter -> looser):
      1. voice_id + inline_* : not allowed -> caller should 400
      2. voice_id            : library lookup; None if voice missing
      3. inline_reference_audio : one-off encode, no caching
      4. inline_prompt_audio + inline_prompt_text : ultimate cloning mode,
         one-off encode
      5. no conditioning     : plain TTS
    """
    if req.voice_id is not None and inline_reference_audio is not None:
        raise ValueError("voice_id and reference_audio are mutually exclusive")

    if req.voice_id is not None:
        latents = await library.get_latents_for_synthesis(req.voice_id)
        if latents is None:
            return None  # signals 404
        # Voice may carry a prompt_text for ultimate-cloning mode.
        voice = await library.get(req.voice_id)
        assert voice is not None  # get_latents_for_synthesis already checked
        if voice.prompt_text:
            return ResolvedConditioning(
                ref_audio_latents=latents,
                prompt_audio_latents=latents,
                prompt_text=voice.prompt_text,
            )
        return ResolvedConditioning(ref_audio_latents=latents)

    if inline_reference_audio is not None:
        enc = await voxcpm.encode_reference(inline_reference_audio)
        return ResolvedConditioning(ref_audio_latents=enc.latents)

    if inline_prompt_audio is not None:
        if not inline_prompt_text:
            raise ValueError("prompt_audio requires prompt_text (ultimate cloning mode)")
        enc = await voxcpm.encode_reference(inline_prompt_audio)
        return ResolvedConditioning(
            ref_audio_latents=enc.latents,
            prompt_audio_latents=enc.latents,
            prompt_text=inline_prompt_text,
        )

    return ResolvedConditioning()  # plain TTS


async def synthesize(
    req: TTSRequest,
    conditioning: ResolvedConditioning,
    client: VoxCPMClient,
) -> TTSResult:
    t0 = time.perf_counter()
    resp = await client.synthesize_text(
        text=req.text,
        sample_rate=req.sample_rate,
        ref_audio_latents=conditioning.ref_audio_latents,
        prompt_audio_latents=conditioning.prompt_audio_latents,
        prompt_text=conditioning.prompt_text,
        cfg_value=req.cfg_value,
        temperature=req.temperature,
    )
    gen_ms = int((time.perf_counter() - t0) * 1000)

    audio, content_type = _package(resp.pcm_s16le, resp.sample_rate, req.output_format)

    return TTSResult(
        audio=audio,
        content_type=content_type,
        sample_rate=resp.sample_rate,
        duration_ms=resp.duration_ms,
        generation_time_ms=gen_ms,
        voice_id=req.voice_id or "default",
    )


def _package(pcm: bytes, sample_rate: int, fmt: OutputFormat) -> tuple[bytes, str]:
    if fmt == "wav":
        return pack_wav(pcm, sample_rate=sample_rate), "audio/wav"
    if fmt == "pcm16":
        return pcm, "audio/L16"
    raise ValueError(f"unsupported output_format: {fmt}")
