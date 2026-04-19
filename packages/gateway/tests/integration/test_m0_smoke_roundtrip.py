"""bugs/003 regression — M0 smoke TTS→STT roundtrip must not garble.

The M0 smoke scripts live at scripts/m0/ (outside the uv workspace; archived
milestone artefacts — see scripts/m0/README.md). This test runs them as
subprocesses under the current interpreter's env, parses the transcript
out of smoke_stt.py's stdout, and asserts WER ≤ 0.2 against the
synthesised phrase.

Gated `@pytest.mark.real_model` (and `RUN_REAL_MODEL=1`) — model load
alone is ~230 s cold, so this is kept off the default CI lane.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import jiwer
import pytest

pytestmark = [pytest.mark.real_model]

REPO_ROOT = Path(__file__).resolve().parents[4]
M0_DIR = REPO_ROOT / "scripts" / "m0"
EXPECTED = "Hello from the voice platform smoke test."
PASS_WER = 0.2

NORMALISE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
    ]
)


def _skip_if_disabled() -> None:
    if os.environ.get("RUN_REAL_MODEL") != "1":
        pytest.skip("set RUN_REAL_MODEL=1 to run real-model tests")
    try:
        import nanovllm_voxcpm  # noqa: F401
    except ImportError:
        pytest.skip("nano-vllm-voxcpm not installed (run `uv sync --extra gpu`)")
    if not (M0_DIR / "smoke_tts.py").exists() or not (M0_DIR / "smoke_stt.py").exists():
        pytest.skip(f"M0 smoke scripts not found at {M0_DIR}")


def _run_m0(script: str) -> str:
    return subprocess.check_output(
        [sys.executable, script],
        cwd=str(M0_DIR),
        text=True,
        stderr=subprocess.STDOUT,
        timeout=900,
    )


def test_m0_smoke_roundtrip_under_20pct_wer() -> None:
    _skip_if_disabled()
    _run_m0("smoke_tts.py")
    stt_out = _run_m0("smoke_stt.py")
    m = re.search(r"\[stt\] transcript: '([^']*)'", stt_out)
    assert m is not None, f"transcript line not found in smoke_stt.py stdout:\n{stt_out}"
    hyp = m.group(1)
    ref_n = NORMALISE(EXPECTED)
    hyp_n = NORMALISE(hyp)
    wer = jiwer.wer(ref_n, hyp_n)
    cer = jiwer.cer(ref_n, hyp_n)
    print(f"[roundtrip] ref={EXPECTED!r} hyp={hyp!r} WER={wer:.3f} CER={cer:.3f}")
    assert wer <= PASS_WER, (
        f"M0 smoke roundtrip WER {wer:.3f} exceeds {PASS_WER}. "
        f"ref={EXPECTED!r} hyp={hyp!r}. See bugs/003."
    )
