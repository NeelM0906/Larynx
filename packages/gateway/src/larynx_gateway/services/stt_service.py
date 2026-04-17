"""Speech-to-text orchestrator.

Pipeline for ``POST /v1/stt`` (see PRD §5.3):

1. Decode uploaded audio (librosa handles wav/mp3/flac/ogg/...) -> 16 kHz
   mono float32.
2. Optionally trim leading/trailing silence using the VAD worker. This
   is the only use of VAD in M3 — streaming VAD (segment-open / segment-
   close events) lands in M4.
3. Convert to int16 LE PCM bytes and hand to the Fun-ASR client. The
   worker picks Nano vs MLT internally via the language router.
4. If ``punctuate=True`` AND Fun-ASR didn't already inline punctuation
   for this language (zh/en are the two cases ct-punc covers), route
   the transcript through the VAD+Punc worker. Otherwise the
   response's ``punctuated`` flag reflects "already handled upstream".

Timing fields returned on ``STTResult`` measure only the time spent
inside this orchestrator; the HTTP layer adds its own wall-clock on the
``X-Request-Duration`` headers later.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import librosa
import numpy as np
import structlog
from larynx_shared.audio import float32_to_int16
from numpy.typing import NDArray

from larynx_gateway.schemas.stt import ModelUsed
from larynx_gateway.workers_client.funasr_client import FunASRClient
from larynx_gateway.workers_client.vad_punc_client import VadPuncClient

log = structlog.get_logger(__name__)

FUNASR_SAMPLE_RATE = 16000
# Padding around the VAD-detected voiced region so we don't clip the
# consonants at word boundaries. 150 ms either side is plenty for the
# REST endpoint where the caller is sending a complete utterance.
VAD_TRIM_PAD_MS = 150
# Languages whose itn=True output is NOT already punctuated — we route
# them through ct-punc. Every other language Fun-ASR supports emits
# inline punctuation from the LLM decoder, so ct-punc would double up.
PUNCTUATION_LANGUAGES: frozenset[str] = frozenset({"zh", "en"})


@dataclass(frozen=True)
class STTResult:
    text: str
    language: str
    model_used: ModelUsed
    duration_ms: int  # source-audio duration
    processing_ms: int  # wall-clock inside the service
    punctuated: bool


async def transcribe(
    *,
    audio_bytes: bytes,
    filename: str | None,
    language: str | None,
    hotwords: list[str],
    itn: bool,
    punctuate: bool,
    trim_silence: bool,
    funasr: FunASRClient,
    vad_punc: VadPuncClient,
) -> STTResult:
    t0 = time.perf_counter()

    try:
        samples = _decode_to_16k_mono(audio_bytes, filename)
    except Exception as e:
        raise ValueError(f"could not decode audio: {e}") from e

    if samples.size == 0:
        raise ValueError("audio is empty after decoding")

    duration_ms = int(1000 * len(samples) / FUNASR_SAMPLE_RATE)

    if trim_silence:
        samples = await _trim_silence(samples, vad_punc)
        if samples.size == 0:
            # Entire clip was silence — return an empty transcript
            # rather than passing nothing to the STT model.
            return STTResult(
                text="",
                language=_resolve_language(language),
                model_used=_guess_model(language),
                duration_ms=duration_ms,
                processing_ms=int((time.perf_counter() - t0) * 1000),
                punctuated=False,
            )

    pcm = float32_to_int16(samples).tobytes()

    resp = await funasr.transcribe(
        pcm_s16le=pcm,
        sample_rate=FUNASR_SAMPLE_RATE,
        language=language,
        hotwords=hotwords,
        itn=itn,
    )

    text = resp.text
    punctuated = False
    if punctuate and text.strip():
        punc = await vad_punc.punctuate(text=text, language=resp.language)
        text = punc.text
        punctuated = punc.applied

    return STTResult(
        text=text,
        language=resp.language,
        model_used=resp.model_used,  # type: ignore[arg-type]  # Literal validated by IPC
        duration_ms=duration_ms,
        processing_ms=int((time.perf_counter() - t0) * 1000),
        punctuated=punctuated,
    )


def _decode_to_16k_mono(audio_bytes: bytes, filename: str | None) -> NDArray[np.float32]:
    """Decode any librosa-supported container to 16 kHz mono float32.

    ``filename`` is only used to surface a clearer error message if
    decoding fails — librosa sniffs the format itself.
    """
    _ = filename  # reserved for future structured error reporting
    samples, _ = librosa.load(io.BytesIO(audio_bytes), sr=FUNASR_SAMPLE_RATE, mono=True)
    return samples.astype(np.float32)


async def _trim_silence(
    samples: NDArray[np.float32], vad_punc: VadPuncClient
) -> NDArray[np.float32]:
    """Keep only the span from the first voiced frame to the last.

    The VAD worker returns [start_ms, end_ms] pairs; we merge into a
    single span (gateway REST path assumes a single utterance) and add
    a small pad so we don't clip the leading/trailing phoneme.
    """
    pcm = float32_to_int16(samples).tobytes()
    resp = await vad_punc.segment(pcm_s16le=pcm, sample_rate=FUNASR_SAMPLE_RATE)
    voiced = [s for s in resp.segments if s.is_speech]
    if not voiced:
        return np.zeros(0, dtype=np.float32)

    start_ms = max(0, min(s.start_ms for s in voiced) - VAD_TRIM_PAD_MS)
    end_ms = max(s.end_ms for s in voiced) + VAD_TRIM_PAD_MS
    start = int(start_ms * FUNASR_SAMPLE_RATE / 1000)
    end = int(end_ms * FUNASR_SAMPLE_RATE / 1000)
    end = min(end, len(samples))
    start = max(0, min(start, end))
    return samples[start:end]


def _resolve_language(language: str | None) -> str:
    if language is None:
        return "auto"
    return language.strip().split("-")[0].split("_")[0].lower() or "auto"


def _guess_model(language: str | None) -> ModelUsed:
    # Only used for the empty-clip shortcut where we bypass the worker;
    # duplicates (a small piece of) the router. If the router grows a
    # table, import it instead.
    code = (language or "").split("-")[0].split("_")[0].lower()
    if code in {"zh", "en", "ja", ""}:
        return "nano"
    return "mlt"
