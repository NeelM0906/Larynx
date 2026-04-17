"""M3 STT smoke test — three languages including one regional accent.

Boots the gateway in-process (so DB/Redis stay offline) with the real
Fun-ASR backend on GPU 1, POSTs three audio fixtures through
``/v1/stt``, prints transcripts + timings + a WER number for the
English clip.

Run::

    LARYNX_STT_MODE=funasr LARYNX_FUNASR_GPU=1 \
    uv run python scripts/smoke_stt.py

Expects audio fixtures at
``packages/gateway/tests/fixtures/audio/{english,chinese,cantonese}_reference.wav``
— seed them with scripts/seed_stt_fixtures.py (M0 leftovers or recorded
manually).
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import re
import time

import httpx

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent.parent / "packages/gateway/tests/fixtures/audio"


def _wer(hyp: str, ref: str) -> float:
    def _tokens(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    h, r = _tokens(hyp), _tokens(ref)
    if not r:
        return 1.0 if h else 0.0

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


async def _run() -> None:
    from larynx_gateway.config import get_settings
    from larynx_gateway.main import create_app

    os.environ.setdefault("LARYNX_STT_MODE", "funasr")
    os.environ.setdefault("LARYNX_VAD_PUNC_MODE", "real")
    get_settings.cache_clear()
    app = create_app()

    token = os.environ.get("LARYNX_API_TOKEN", "change-me-please")
    headers = {"Authorization": f"Bearer {token}"}

    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://smoke", timeout=120) as client,
        app.router.lifespan_context(app),
    ):
        cases = [
            ("English", "english_reference.wav", "en", "english_reference.txt"),
            ("Chinese", "chinese_reference.wav", "zh", None),
            ("Cantonese", "cantonese_reference.wav", "yue", None),
        ]
        for label, fname, lang, ref_name in cases:
            path = FIXTURE_DIR / fname
            if not path.exists():
                print(f"[stt] {label}: missing fixture {path}; skipping")
                continue
            t0 = time.perf_counter()
            r = await client.post(
                "/v1/stt",
                headers=headers,
                files={"file": (fname, path.read_bytes(), "audio/wav")},
                data={"language": lang},
            )
            wall_ms = int((time.perf_counter() - t0) * 1000)
            if r.status_code != 200:
                print(f"[stt] {label}: HTTP {r.status_code} — {r.text!r}")
                continue
            body = r.json()
            print(
                f"[stt] {label}: model={body['model_used']} lang={body['language']} "
                f"dur={body['duration_ms']}ms proc={body['processing_ms']}ms "
                f"wall={wall_ms}ms punct={body['punctuated']}"
            )
            print(f"[stt]   transcript: {body['text']!r}")
            if ref_name is not None:
                ref_path = FIXTURE_DIR / ref_name
                if ref_path.exists():
                    ref = ref_path.read_text().strip()
                    wer = _wer(body["text"], ref)
                    print(f"[stt]   reference: {ref!r}")
                    print(f"[stt]   WER: {wer:.3f}")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
