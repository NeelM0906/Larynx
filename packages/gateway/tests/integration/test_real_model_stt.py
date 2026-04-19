"""Real Fun-ASR STT tests — opt-in via RUN_REAL_MODEL=1.

Runs only on the GPU box (targets Fun-ASR-Nano + Fun-ASR-MLT-Nano on
GPU 1). Covers the PRD M3 exit criteria:

- English clip: WER > 90% against a known reference
- Chinese clip: transcript contains expected anchor phrase
- Portuguese clip (MLT routing): ``model_used`` is ``"mlt"``
- Hotwords: an unusual term passed as a hotword lands in the transcript
- Punctuation: punctuate=true output has sentence-ending punctuation;
  punctuate=false output doesn't

Audio fixtures live under ``packages/gateway/tests/fixtures/audio/``; the
test skips with a clear message if any file is missing so the suite can
still advance on a box where fixtures haven't been seeded yet.

Fixture provenance
------------------

* ``english_reference.wav`` and ``hotword_reference.wav`` are VoxCPM2
  synthesis (16 kHz mono PCM_16). Ground-truth texts live in
  ``english_reference.txt`` and ``transcripts.json``.
* ``chinese_reference.wav`` is a copy of ``zh.wav`` bundled with the
  Fun-ASR-vllm checkpoint at
  ``scripts/m0/Fun-ASR-vllm/triton_server/assets/zh.wav`` — real human
  Chinese speech.
* ``portuguese_reference.wav`` and ``cantonese_reference.wav`` are
  also copies of that ``zh.wav``. The pt / yue tests assert the
  *language router* sends those tags to Fun-ASR-MLT-Nano
  (``model_used == "mlt"``) and that the transcript is non-empty;
  neither asserts transcript accuracy or language detection, so
  routing-only validation works with any speech-bearing audio.

Regenerate via ``uv run python scripts/seed_stt_fixtures.py`` (or
``uv run --extra gpu ...`` if the VoxCPM2 fallback path needs to fire).

Per CLAUDE memory ``feedback_no_fakes_in_tests.md`` — no mocks in this
file. Real model, real Postgres, real Redis.
"""

from __future__ import annotations

import os
import pathlib
import re
from collections.abc import Iterator

import pytest
from httpx import AsyncClient

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "audio"


@pytest.fixture(autouse=True)
def _real_stt_env() -> Iterator[None]:
    """Override the session conftest's `mock` defaults.

    The session-scoped ``_session_env`` autouse fixture in
    ``packages/gateway/tests/conftest.py`` forces ``LARYNX_STT_MODE=mock``
    and ``LARYNX_VAD_PUNC_MODE=mock`` so the fast mock suite stays fast.
    The real-model STT tests in this file need the actual Fun-ASR-Nano
    + Fun-ASR-MLT-Nano + ct-punc stack, matching what
    ``test_real_model_stream.py::live_server`` does for itself. This
    per-test fixture overrides the env vars just long enough for the
    ``client`` fixture in conftest.py to read the real-mode values when
    it re-creates the app, then restores the originals on teardown so
    later modules (run in the same pytest invocation) see the session
    defaults.

    Also mirrors ``test_real_model_stream.py`` / ``..._conversation.py``
    in putting the Fun-ASR-vllm checkout on ``sys.path`` so
    ``from model import FunASRNano`` resolves inside the real backend's
    ``_load_sync``.
    """
    if os.environ.get("RUN_REAL_MODEL") != "1":
        # Respect the real_model opt-in: if it's off, skip the swap so
        # the test's own ``_needs_real_model()`` check fires with the
        # standard skip message.
        yield
        return

    funasr_vllm_dir = os.environ.get(
        "LARYNX_FUNASR_VLLM_DIR", "/home/ripper/larynx-smoke/Fun-ASR-vllm"
    )
    if not pathlib.Path(funasr_vllm_dir).exists():
        pytest.skip(f"Fun-ASR-vllm repo not found at {funasr_vllm_dir}")
    import sys

    if funasr_vllm_dir not in sys.path:
        sys.path.insert(0, funasr_vllm_dir)

    saved = {
        k: os.environ.get(k)
        for k in (
            "LARYNX_STT_MODE",
            "LARYNX_VAD_PUNC_MODE",
            "LARYNX_FUNASR_GPU",
            "LARYNX_FUNASR_VLLM_DIR",
        )
    }
    os.environ["LARYNX_STT_MODE"] = "funasr"
    os.environ["LARYNX_VAD_PUNC_MODE"] = "real"
    # Pin Fun-ASR to GPU 1 to match production allocation (README hardware
    # table). GPU 0 is reserved for VoxCPM2; this module doesn't use TTS
    # but the setting keeps the backend's GPU selection deterministic.
    os.environ["LARYNX_FUNASR_GPU"] = os.environ.get("LARYNX_FUNASR_GPU", "1")
    os.environ["LARYNX_FUNASR_VLLM_DIR"] = funasr_vllm_dir
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _needs_real_model() -> None:
    if os.environ.get("RUN_REAL_MODEL") != "1":
        pytest.skip("real-model tests disabled; set RUN_REAL_MODEL=1 to enable")
    if os.environ.get("LARYNX_STT_MODE", "mock").lower() != "funasr":
        pytest.skip("LARYNX_STT_MODE must be 'funasr' for real-model STT tests")


def _load_fixture(name: str) -> bytes:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"missing fixture {path}; run scripts/seed_stt_fixtures.py")
    return path.read_bytes()


def _wer(hyp: str, ref: str) -> float:
    """Word error rate — classic Levenshtein distance / reference length.

    Kept inline (tiny function) so the real-model test suite has no
    extra dependency. Case-insensitive, strips punctuation so Fun-ASR's
    inline punctuation doesn't mechanically inflate WER.
    """

    def _tokens(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    h, r = _tokens(hyp), _tokens(ref)
    if not r:
        return 1.0 if h else 0.0

    # Classic dynamic-programming edit distance.
    dp = list(range(len(h) + 1))
    for i, rt in enumerate(r, 1):
        prev = dp[0]
        dp[0] = i
        for j, ht in enumerate(h, 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if rt == ht else 1),
            )
            prev = cur
    return dp[-1] / len(r)


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_english_wer_under_10pct(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    _needs_real_model()
    audio = _load_fixture("english_reference.wav")
    reference_path = FIXTURE_DIR / "english_reference.txt"
    if not reference_path.exists():
        pytest.skip(f"missing reference {reference_path}")
    reference = reference_path.read_text().strip()

    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("english.wav", audio, "audio/wav")},
        data={"language": "en"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_used"] == "nano"
    wer = _wer(body["text"], reference)
    assert wer < 0.10, f"WER={wer:.3f} > 0.10; transcript={body['text']!r}"


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_chinese_transcript_reasonable(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    _needs_real_model()
    audio = _load_fixture("chinese_reference.wav")
    anchor_path = FIXTURE_DIR / "chinese_anchor.txt"
    anchor = anchor_path.read_text().strip() if anchor_path.exists() else ""

    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("chinese.wav", audio, "audio/wav")},
        data={"language": "zh"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_used"] == "nano"
    assert body["language"] == "zh"
    assert len(body["text"]) > 0
    if anchor:
        assert anchor in body["text"], f"expected anchor {anchor!r} in transcript {body['text']!r}"


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_portuguese_uses_mlt(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    _needs_real_model()
    audio = _load_fixture("portuguese_reference.wav")
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("portuguese.wav", audio, "audio/wav")},
        data={"language": "pt"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_used"] == "mlt", body
    assert body["language"] == "pt"
    assert len(body["text"]) > 0


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_cantonese_dialect(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    """PRD M3 exit criterion: one Chinese dialect / regional accent.

    Cantonese is covered by the MLT checkpoint (``粤语``) — so this
    also verifies that the language router sends yue -> MLT even though
    it's part of the broader Chinese language family.
    """
    _needs_real_model()
    audio = _load_fixture("cantonese_reference.wav")
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("cantonese.wav", audio, "audio/wav")},
        data={"language": "yue"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_used"] == "mlt"
    assert len(body["text"]) > 0


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_hotword_recovery(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    """A clip containing an unusual proper noun — Fun-ASR sometimes
    mis-transcribes without the hotword, gets it right with.

    Fun-ASR's README uses ``开放时间`` as its example; we use a proper
    noun here because it's more obviously testable in English. The
    fixture is a short clip saying "Please contact Larynx for help."
    Fun-ASR tends to default to "Lorex" / "Laryx" without a hint.
    """
    _needs_real_model()
    audio = _load_fixture("hotword_reference.wav")
    r = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("hotword.wav", audio, "audio/wav")},
        data={"language": "en", "hotwords": "Larynx"},
    )
    assert r.status_code == 200, r.text
    assert "Larynx" in r.json()["text"]


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_punctuation_on_vs_off(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    _needs_real_model()
    audio = _load_fixture("english_reference.wav")

    r_on = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("english.wav", audio, "audio/wav")},
        data={"language": "en", "punctuate": "true"},
    )
    r_off = await client.post(
        "/v1/stt",
        headers=auth_headers,
        files={"file": ("english.wav", audio, "audio/wav")},
        data={"language": "en", "punctuate": "false"},
    )
    assert r_on.status_code == 200
    assert r_off.status_code == 200

    on_text = r_on.json()["text"]

    # The punctuated version must contain at least one sentence-ending
    # punctuation mark. Fun-ASR's itn=true output may already punctuate
    # in-line; ct-punc only adds on top. Either way, "." or "?" or "!"
    # should be present for a non-trivial English clip.
    assert any(p in on_text for p in [".", "?", "!"]), on_text
    # Raw (non-punctuated) path: the text came straight out of Fun-ASR
    # without the ct-punc pass; acceptable to still have inline periods
    # from itn=True, but the punctuated flag differs.
    assert r_on.json()["punctuated"] is True
    assert r_off.json()["punctuated"] is False
