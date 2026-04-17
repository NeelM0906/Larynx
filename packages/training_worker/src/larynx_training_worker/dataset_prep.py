"""Dataset validation and prep for LoRA fine-tuning.

Phase A (``validate_dataset_phase_a``) runs synchronously at upload
time. It's the "did the user hand us something that could possibly
train?" gate — structural only, no ASR, no network. Every rule comes
from ORCHESTRATION-M7.md §2.1.

Auto-transcription (``auto_transcribe_if_missing``) and Phase B
transcript-quality WER live here too but run during ``PREPARING`` on
the training_worker side, not at upload.
"""

from __future__ import annotations

import json
import logging
import pathlib
import random
import re
import string
from collections.abc import Awaitable, Callable

import numpy as np
import soundfile as sf
from larynx_shared.paths import SUPPORTED_AUDIO_SUFFIXES, DatasetPaths
from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

# Fun-ASR expects 16 kHz mono int16 PCM at the wire — we normalise
# every audio file to that shape before calling the transcribe hook so
# the caller never has to branch on input sample rate.
_TRANSCRIBE_SR = 16_000

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


TranscribeHook = Callable[[bytes, int], Awaitable[str]]


# ---------------------------------------------------------------------------
# Phase B — advisory transcript-quality sanity check
# ---------------------------------------------------------------------------


class PhaseBSuspect(BaseModel):
    """One sample flagged by Phase B for human review."""

    audio_path: str
    reference: str
    hypothesis: str
    wer: float


class PhaseBReport(BaseModel):
    num_samples: int
    subset_fraction: float
    max_samples: int
    wer_threshold: float
    seed: int
    suspects: list[PhaseBSuspect] = Field(default_factory=list)


# Preserve apostrophes and internal hyphens — "it's" and "well-being"
# are meaningful tokens in English ASR evaluation. Strip everything
# else in string.punctuation.
_STRIP_PUNCT = "".join(c for c in string.punctuation if c not in "'-")
_PUNCT_STRIP = str.maketrans("", "", _STRIP_PUNCT)


def normalise_transcript(text: str) -> str:
    """Lowercase + strip most ASCII punctuation + collapse whitespace."""
    lowered = text.lower().translate(_PUNCT_STRIP)
    return re.sub(r"\s+", " ", lowered).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Plain word-level WER via Levenshtein distance.

    Convention: empty reference + empty hypothesis -> 0.0 (nothing was
    wrong). Empty reference + non-empty hypothesis -> 1.0 (entirely
    wrong by insertion). Otherwise, edit distance / len(reference).
    """
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    # DP table of shape (len(ref)+1, len(hyp)+1).
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,  # deletion
                cur[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev = cur
    return prev[n] / m


async def validate_transcripts_phase_b(
    dataset: DatasetPaths,
    *,
    transcribe: TranscribeHook,
    subset_fraction: float = 0.05,
    max_samples: int = 20,
    wer_threshold: float = 0.4,
    seed: int = 0,
) -> PhaseBReport:
    """Advisory WER check against Fun-ASR (ORCHESTRATION-M7.md §2.2).

    Non-blocking: returns a report, writes ``validation_report.json``
    to the dataset dir. Caller decides whether suspects block the job
    (the UI shows them; the orchestrator runs regardless).

    ``subset_fraction`` is a fraction of rows to sample; ``max_samples``
    caps the absolute count. ``seed`` pins the random sample order so a
    re-run of Phase B produces the same subset (reproducibility).
    """
    if not dataset.has_transcripts():
        report = PhaseBReport(
            num_samples=0,
            subset_fraction=subset_fraction,
            max_samples=max_samples,
            wer_threshold=wer_threshold,
            seed=seed,
        )
        dataset.validation_report_json.parent.mkdir(parents=True, exist_ok=True)
        dataset.validation_report_json.write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        return report

    rows = []
    with dataset.transcripts_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        report = PhaseBReport(
            num_samples=0,
            subset_fraction=subset_fraction,
            max_samples=max_samples,
            wer_threshold=wer_threshold,
            seed=seed,
        )
    else:
        target_count = min(max_samples, max(1, int(len(rows) * subset_fraction)))
        rng = random.Random(seed)
        picks = rng.sample(rows, k=min(target_count, len(rows)))

        suspects: list[PhaseBSuspect] = []
        for row in picks:
            audio = row.get("audio", "")
            reference = row.get("text", "")
            path = _resolve_manifest_audio(dataset, audio)
            try:
                pcm, sr = _load_as_16k_s16_pcm(path)
                hypothesis = await transcribe(pcm, sr)
            except Exception as e:  # noqa: BLE001
                _log.warning("phase_b.transcribe_failed audio=%s error=%s", path, e)
                continue
            wer = word_error_rate(normalise_transcript(reference), normalise_transcript(hypothesis))
            if wer > wer_threshold:
                suspects.append(
                    PhaseBSuspect(
                        audio_path=str(path),
                        reference=reference,
                        hypothesis=hypothesis,
                        wer=round(wer, 4),
                    )
                )

        report = PhaseBReport(
            num_samples=len(picks),
            subset_fraction=subset_fraction,
            max_samples=max_samples,
            wer_threshold=wer_threshold,
            seed=seed,
            suspects=suspects,
        )

    dataset.validation_report_json.parent.mkdir(parents=True, exist_ok=True)
    dataset.validation_report_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def normalise_manifest_paths(dataset: DatasetPaths) -> int:
    """Rewrite ``transcripts.jsonl`` so every ``audio`` entry is absolute.

    HuggingFace ``datasets`` resolves relative audio paths against the
    subprocess CWD — not against the manifest file's directory — so a
    bare ``clip00.wav`` in the manifest fails with ``FileNotFoundError``
    at training start. Rewriting to absolute paths at PREPARING time
    is the simplest robust fix: no cwd trick required, and the
    manifest remains readable on disk for debugging.

    Idempotent: if every entry is already absolute, no file is
    rewritten and 0 is returned. Otherwise returns the count of rows
    that got rewritten.
    """
    if not dataset.has_transcripts():
        return 0
    rows: list[dict[str, object]] = []
    rewritten = 0
    with dataset.transcripts_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rows.append({"__raw__": line})
                continue
            audio = row.get("audio")
            if isinstance(audio, str):
                resolved = _resolve_manifest_audio(dataset, audio)
                if str(resolved) != audio:
                    row["audio"] = str(resolved)
                    rewritten += 1
            rows.append(row)
    if rewritten == 0:
        return 0
    tmp = dataset.transcripts_jsonl.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            if "__raw__" in row:
                fh.write(str(row["__raw__"]) + "\n")
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(dataset.transcripts_jsonl)
    return rewritten


async def auto_transcribe_if_missing(
    dataset: DatasetPaths,
    *,
    transcribe: TranscribeHook,
) -> int:
    """Generate ``transcripts.jsonl`` if the dataset arrived without one.

    For each audio file in sorted order:
      1. decode to 16 kHz mono int16 PCM (what Fun-ASR expects)
      2. call ``transcribe(pcm, sr)``
      3. append a row ``{"audio": <absolute path>, "text": ...}``

    Returns the number of rows written. Returns 0 (no-op) if the
    manifest already exists — see the invariant in ORCHESTRATION-M7.md
    §2.3: the training script always reads a manifest, and
    auto-transcription is idempotent on re-submission.

    Individual transcription failures are logged + recorded as an empty
    ``text`` field (rather than raising). The training script will
    then reject the job during dataset load, which is the correct
    behaviour — a dataset with a silent file deserves a visible error,
    not a rolled-back preparation step.
    """
    if dataset.has_transcripts():
        return 0

    rows: list[dict[str, str]] = []
    for audio_path in sorted(dataset.audio_files()):
        try:
            pcm, sr = _load_as_16k_s16_pcm(audio_path)
            text = await transcribe(pcm, sr)
        except Exception as e:  # noqa: BLE001
            _log.warning("auto_transcribe.file_failed path=%s error=%s", audio_path, e)
            text = ""
        rows.append({"audio": str(audio_path), "text": text})

    with dataset.transcripts_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows)


def _load_as_16k_s16_pcm(path: pathlib.Path) -> tuple[bytes, int]:
    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != _TRANSCRIBE_SR:
        # Pure-numpy linear resample — cheap, avoids dragging librosa
        # into the worker's hot path. Good enough for ASR; the actual
        # training dataset loader owns its own (higher-quality)
        # resampling via HuggingFace's Audio column.
        ratio = _TRANSCRIBE_SR / sr
        n_out = int(round(len(samples) * ratio))
        x_src = np.linspace(0.0, 1.0, len(samples), endpoint=False, dtype=np.float64)
        x_dst = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float64)
        samples = np.interp(x_dst, x_src, samples).astype(np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    return pcm, _TRANSCRIBE_SR


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
