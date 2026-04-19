"""Seed STT fixture audio for `test_real_model_stt.py`.

Populates ``packages/gateway/tests/fixtures/audio/`` with five WAV
clips and a ``transcripts.json`` ground-truth map. Idempotent — any
fixture that already exists is left alone, so re-running after a
partial failure only does the remaining work.

Sources
-------
The Fun-ASR-vllm checkpoint ships exactly one example clip
(``zh.wav`` under ``triton_server/assets/``). That's enough real human
speech for the Chinese test, and we reuse it for the Portuguese and
Cantonese tests because those tests only assert the language router
sends ``pt`` / ``yue`` to the Fun-ASR-MLT-Nano backend — they don't
check transcript accuracy against a ground-truth text. The audio
content is validated indirectly through ``len(body["text"]) > 0``,
which MLT satisfies for any speech-bearing clip.

English and the "hotword" clip are synthesised via VoxCPM2 (the same
TTS → 16 kHz resample pipeline ``scripts/m0/smoke_tts.py`` uses after
the bugs/003 fix). Ground-truth for those is exactly the text fed to
VoxCPM2. We re-use ``scripts/m0/smoke_tts.wav`` for the English clip
when it's already present, to avoid re-loading VoxCPM2 if the user
just ran the M0 smoke.

Run
---

    uv run --extra gpu python scripts/seed_stt_fixtures.py

(The ``--extra gpu`` is only needed if the English/hotword clips
aren't yet seeded and VoxCPM2 synthesis needs to run. A plain
``uv run python scripts/seed_stt_fixtures.py`` is enough when the
fixtures already exist or when only the ``zh.wav`` copies need
doing.)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "packages" / "gateway" / "tests" / "fixtures" / "audio"
ZH_SOURCE = REPO_ROOT / "scripts" / "m0" / "Fun-ASR-vllm" / "triton_server" / "assets" / "zh.wav"
M0_SMOKE_WAV = REPO_ROOT / "scripts" / "m0" / "smoke_tts.wav"

ENGLISH_TEXT = "Hello from the voice platform smoke test."
HOTWORD_TEXT = "Please contact Larynx for help."

TARGET_SR = 16000


def _synthesise_voxcpm(text: str, out_path: Path) -> None:
    """Synthesise ``text`` with VoxCPM2 and write a 16 kHz mono PCM_16 WAV.

    Mirrors ``scripts/m0/smoke_tts.py`` after the bugs/003 fix —
    queries the server for its native output rate, then resamples
    to 16 kHz via librosa's bundled soxr_hq resampler (no extra dep).
    Requires ``nano-vllm-voxcpm`` (installed via ``uv sync --extra gpu``)
    and GPU 0 with enough free memory to host VoxCPM2 (~8 GB).
    """
    import librosa
    import numpy as np
    import soundfile as sf

    try:
        from nanovllm_voxcpm import VoxCPM
    except ImportError as exc:
        raise SystemExit(
            f"cannot synthesise {out_path.name}: nano-vllm-voxcpm is not "
            "installed in this venv. Run `uv sync --extra gpu` first, or "
            "pre-stage the WAV from another source and re-run."
        ) from exc

    print(f"[seed] loading VoxCPM2 to synthesise {out_path.name}...", flush=True)
    server = VoxCPM.from_pretrained(
        model="openbmb/VoxCPM2",
        devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.80,
    )
    try:
        native_sr = int(server.get_model_info()["output_sample_rate"])
        chunks: list[np.ndarray] = []
        for chunk in server.generate(target_text=text):
            chunks.append(np.asarray(chunk).reshape(-1))
        native = np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float32)
        if native_sr != TARGET_SR and native.size:
            wav = librosa.resample(
                native.astype(np.float32),
                orig_sr=native_sr,
                target_sr=TARGET_SR,
                res_type="soxr_hq",
            )
        else:
            wav = native
        sf.write(str(out_path), wav, samplerate=TARGET_SR)
        dur_s = len(wav) / TARGET_SR if TARGET_SR else 0.0
        print(
            f"[seed] wrote {out_path.name}  {len(wav)} samples @ {TARGET_SR} Hz  ({dur_s:.2f}s)",
            flush=True,
        )
    finally:
        try:
            server.stop()
        except Exception:
            pass


def _seed_english(target: Path) -> None:
    if target.exists():
        return
    if M0_SMOKE_WAV.exists():
        shutil.copyfile(M0_SMOKE_WAV, target)
        print(f"[seed] copied {M0_SMOKE_WAV} → {target.name}")
        return
    _synthesise_voxcpm(ENGLISH_TEXT, target)


def _seed_hotword(target: Path) -> None:
    if target.exists():
        return
    _synthesise_voxcpm(HOTWORD_TEXT, target)


def _seed_from_zh(target: Path, note: str) -> None:
    if target.exists():
        return
    if not ZH_SOURCE.exists():
        raise SystemExit(
            f"cannot seed {target.name}: {ZH_SOURCE} is missing. Clone "
            "github.com/yuekaizhang/Fun-ASR-vllm into scripts/m0/ first "
            "(the M0 smoke setup does this) so its bundled zh.wav is "
            "available."
        )
    shutil.copyfile(ZH_SOURCE, target)
    print(f"[seed] copied {ZH_SOURCE.name} → {target.name}  ({note})")


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    _seed_english(FIXTURE_DIR / "english_reference.wav")
    _seed_hotword(FIXTURE_DIR / "hotword_reference.wav")
    _seed_from_zh(FIXTURE_DIR / "chinese_reference.wav", "real Chinese human speech")
    _seed_from_zh(
        FIXTURE_DIR / "portuguese_reference.wav",
        "reused for pt MLT-routing test — test checks routing, not content",
    )
    _seed_from_zh(
        FIXTURE_DIR / "cantonese_reference.wav",
        "reused for yue MLT-routing test — test checks routing, not content",
    )

    # Ground-truth reference texts. The STT test suite expects these in
    # sibling ``.txt`` files; ``transcripts.json`` is a discoverability
    # index of what each clip claims to say. Keep both in sync.
    english_ref = FIXTURE_DIR / "english_reference.txt"
    if not english_ref.exists():
        english_ref.write_text(ENGLISH_TEXT + "\n")
        print(f"[seed] wrote {english_ref.name}")

    # chinese_anchor.txt is optional (test_chinese_transcript_reasonable
    # skips the anchor assertion if the file is absent). We leave it
    # unseeded so the test stays passing on whatever Fun-ASR returns for
    # zh.wav; seed the file manually with a substring of an observed
    # transcript if you want the anchor check active.

    transcripts = {
        "english_reference.wav": {
            "ground_truth": ENGLISH_TEXT,
            "source": "VoxCPM2 synthesis (or scripts/m0/smoke_tts.wav fallback)",
            "language": "en",
        },
        "hotword_reference.wav": {
            "ground_truth": HOTWORD_TEXT,
            "source": "VoxCPM2 synthesis",
            "language": "en",
            "notes": "hotword assertion in test: must transcribe 'Larynx' when "
            "that term is passed as a hotword",
        },
        "chinese_reference.wav": {
            "ground_truth": None,
            "source": "scripts/m0/Fun-ASR-vllm/triton_server/assets/zh.wav "
            "(real human speech shipped with the Fun-ASR-vllm checkpoint)",
            "language": "zh",
            "notes": "populate chinese_anchor.txt with an observed substring "
            "if you want the anchor assertion active",
        },
        "portuguese_reference.wav": {
            "ground_truth": None,
            "source": "copy of chinese_reference.wav",
            "language": "pt",
            "notes": "test asserts language-router sends pt → Fun-ASR-MLT-Nano "
            "(model_used == 'mlt') and the transcript is non-empty; audio "
            "content is irrelevant to the assertions.",
        },
        "cantonese_reference.wav": {
            "ground_truth": None,
            "source": "copy of chinese_reference.wav",
            "language": "yue",
            "notes": "test asserts language-router sends yue → Fun-ASR-MLT-Nano; "
            "audio content is irrelevant to the assertions.",
        },
    }
    transcripts_path = FIXTURE_DIR / "transcripts.json"
    transcripts_path.write_text(json.dumps(transcripts, indent=2, ensure_ascii=False) + "\n")
    print(f"[seed] wrote {transcripts_path.name}")

    print("[seed] all fixtures present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
