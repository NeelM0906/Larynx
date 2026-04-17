"""Single-shot TTS service — thin orchestrator around the VoxCPM client."""

from __future__ import annotations

import time
from dataclasses import dataclass

from larynx_shared.audio import pack_wav

from larynx_gateway.schemas.tts import TTSRequest
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient


@dataclass(frozen=True)
class TTSResult:
    audio: bytes
    content_type: str
    sample_rate: int
    duration_ms: int
    generation_time_ms: int
    voice_id: str


async def synthesize(req: TTSRequest, client: VoxCPMClient) -> TTSResult:
    t0 = time.perf_counter()
    resp = await client.synthesize_text(
        text=req.text,
        sample_rate=req.sample_rate,
        cfg_value=req.cfg_value,
    )
    gen_ms = int((time.perf_counter() - t0) * 1000)

    if req.output_format == "wav":
        audio = pack_wav(resp.pcm_s16le, sample_rate=resp.sample_rate)
        content_type = "audio/wav"
    elif req.output_format == "pcm16":
        audio = resp.pcm_s16le
        content_type = "audio/L16"
    else:  # pydantic Literal guards this at the edge
        raise ValueError(f"unsupported output_format: {req.output_format}")

    return TTSResult(
        audio=audio,
        content_type=content_type,
        sample_rate=resp.sample_rate,
        duration_ms=resp.duration_ms,
        generation_time_ms=gen_ms,
        voice_id=req.voice_id or "default",
    )
