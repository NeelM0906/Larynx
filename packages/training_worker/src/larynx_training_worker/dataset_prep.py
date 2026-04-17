"""Dataset validation and prep for LoRA fine-tuning.

Phase A (``validate_dataset_phase_a``) runs synchronously at upload
time. It's the "did the user hand us something that could possibly
train?" gate — structural only, no ASR, no network. Every rule comes
from ORCHESTRATION-M7.md §2.1.

Phase B (transcript-quality WER via Fun-ASR) and auto-transcription
live separately; they run during the ``PREPARING`` state of a job, not
on upload.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import soundfile as sf
from larynx_shared.paths import SUPPORTED_AUDIO_SUFFIXES, DatasetPaths
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Defaults (ORCHESTRATION-M7.md §10)
# ---------------------------------------------------------------------------

DEFAULT_MIN_SECONDS = 300
DEFAULT_SILENCE_PEAK = 0.001


class PhaseAIssue(BaseModel):
    """One structural failure surfaced to the UI.

    ``code`` is a stable machine identifier; ``detail`` is a human string
    safe to render in a toast. File paths in ``detail`` let users click
    straight to the broken file in the upload preview.
    """

    code: str
    detail: str


class PhaseAReport(BaseModel):
    """Result of running :func:`validate_dataset_phase_a`."""

    ok: bool
    num_clips: int = 0
    total_duration_s: float = 0.0
    # Histogram of observed input sample rates — useful for the upload
    # preview banner ("32 @ 16 kHz, 4 @ 44.1 kHz — will be resampled").
    sample_rates: dict[int, int] = Field(default_factory=dict)
    issues: list[PhaseAIssue] = Field(default_factory=list)


def validate_dataset_phase_a(
    dataset: DatasetPaths,
    *,
    min_seconds: int = DEFAULT_MIN_SECONDS,
    silence_peak: float = DEFAULT_SILENCE_PEAK,
) -> PhaseAReport:
    """Run the structural checks listed in ORCHESTRATION-M7.md §2.1.

    Never raises; every failure is an entry in ``issues``. A report with
    ``issues == []`` implies ``ok is True``.

    Side-effect-free: reads the dataset dir, writes nothing.
    """
    issues: list[PhaseAIssue] = []
    sample_rate_hist: dict[int, int] = {}
    total_duration_s = 0.0
    audio_paths: list[pathlib.Path] = []

    # -- 1. enumerate audio + catch stray non-audio files ---------------
    for entry in sorted(dataset.audio_dir.iterdir()) if dataset.audio_dir.is_dir() else []:
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            issues.append(
                PhaseAIssue(
                    code="unsupported_file",
                    detail=(
                        f"{entry} has unsupported suffix {entry.suffix!r}; "
                        f"accepted: {sorted(SUPPORTED_AUDIO_SUFFIXES)}"
                    ),
                )
            )
            continue
        audio_paths.append(entry)

    # -- 2. per-file structural checks ---------------------------------
    for path in audio_paths:
        try:
            info = sf.info(str(path))
        except Exception as e:  # noqa: BLE001
            issues.append(
                PhaseAIssue(
                    code="unreadable_audio",
                    detail=f"{path}: soundfile could not open ({e})",
                )
            )
            continue

        channels = int(getattr(info, "channels", 0))
        if channels > 2:
            issues.append(
                PhaseAIssue(
                    code="too_many_channels",
                    detail=(
                        f"{path}: {channels} channels; only mono + stereo "
                        "accepted (dataset loader downmixes at training time)."
                    ),
                )
            )
            # Still count it toward duration / SR histogram so those stats
            # reflect what the user actually uploaded.

        sr = int(getattr(info, "samplerate", 0))
        sample_rate_hist[sr] = sample_rate_hist.get(sr, 0) + 1

        frames = int(getattr(info, "frames", 0))
        if sr > 0:
            total_duration_s += frames / sr

        # Peak check: decode once and look for nonzero signal. This is the
        # expensive part of Phase A (a disk read + a float conversion),
        # but it's O(clip length) and Phase A is bounded by the MIN
        # duration anyway.
        try:
            samples, _ = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception as e:  # noqa: BLE001
            issues.append(
                PhaseAIssue(
                    code="unreadable_audio",
                    detail=f"{path}: soundfile read failed ({e})",
                )
            )
            continue

        if samples.size == 0:
            issues.append(PhaseAIssue(code="empty_audio", detail=f"{path}: zero samples"))
            continue

        peak = float(np.max(np.abs(samples)))
        if not np.isfinite(peak):
            issues.append(
                PhaseAIssue(
                    code="invalid_audio",
                    detail=f"{path}: samples contain NaN / Inf",
                )
            )
            continue
        if peak < silence_peak:
            issues.append(
                PhaseAIssue(
                    code="silent_audio",
                    detail=f"{path}: peak amplitude {peak:.5f} below {silence_peak:.5f}",
                )
            )

    # -- 3. total-duration gate ----------------------------------------
    if total_duration_s < min_seconds:
        issues.append(
            PhaseAIssue(
                code="duration_too_short",
                detail=(f"total duration {total_duration_s:.1f}s < {min_seconds}s minimum"),
            )
        )

    # -- 4. transcript pairing (only when a manifest is present) -------
    if dataset.has_transcripts():
        issues.extend(_validate_transcripts(dataset, audio_paths))

    return PhaseAReport(
        ok=not issues,
        num_clips=len(audio_paths),
        total_duration_s=round(total_duration_s, 3),
        sample_rates=sample_rate_hist,
        issues=issues,
    )


def _validate_transcripts(
    dataset: DatasetPaths,
    audio_paths: list[pathlib.Path],
) -> list[PhaseAIssue]:
    """Check shape + bidirectional pairing of ``transcripts.jsonl``."""
    issues: list[PhaseAIssue] = []
    audio_set = {p.resolve() for p in audio_paths}
    manifest_paths: set[pathlib.Path] = set()

    with dataset.transcripts_jsonl.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                issues.append(
                    PhaseAIssue(
                        code="bad_manifest_line",
                        detail=f"{dataset.transcripts_jsonl}:{lineno}: {e}",
                    )
                )
                continue
            audio = row.get("audio")
            text = row.get("text")
            if not isinstance(audio, str):
                issues.append(
                    PhaseAIssue(
                        code="bad_manifest_line",
                        detail=(
                            f"{dataset.transcripts_jsonl}:{lineno}: row missing ``audio`` string"
                        ),
                    )
                )
                continue
            if not isinstance(text, str) or not text.strip():
                issues.append(
                    PhaseAIssue(
                        code="empty_transcript_text",
                        detail=(
                            f"{dataset.transcripts_jsonl}:{lineno}: "
                            f"row for {audio!r} has empty or missing ``text``"
                        ),
                    )
                )
                # fall through so we still record the pairing
            candidate = _resolve_manifest_audio(dataset, audio)
            manifest_paths.add(candidate)
            if candidate not in audio_set:
                issues.append(
                    PhaseAIssue(
                        code="transcript_missing_audio",
                        detail=(
                            f"{dataset.transcripts_jsonl}:{lineno}: "
                            f"``audio`` path {audio!r} does not exist under "
                            f"{dataset.audio_dir}"
                        ),
                    )
                )

    for path in audio_set - manifest_paths:
        issues.append(
            PhaseAIssue(
                code="audio_missing_transcript",
                detail=f"{path} has no row in {dataset.transcripts_jsonl}",
            )
        )

    return issues


def _resolve_manifest_audio(dataset: DatasetPaths, audio: str) -> pathlib.Path:
    """Resolve a manifest ``audio`` value to an absolute path.

    Absolute paths are used as-is. Relative paths are tried in two
    locations, in order: ``audio_dir`` (the convention our upload flow
    produces) and ``base_dir`` (the convention the upstream example
    manifest uses, where entries are relative to the project root).
    Whichever exists wins; if neither does, ``audio_dir / audio`` is
    returned so the missing-file error message points at a plausible
    location.
    """
    candidate = pathlib.Path(audio)
    if candidate.is_absolute():
        return candidate.resolve()
    under_audio = (dataset.audio_dir / candidate).resolve()
    if under_audio.is_file():
        return under_audio
    under_base = (dataset.base_dir / candidate).resolve()
    if under_base.is_file():
        return under_base
    return under_audio
