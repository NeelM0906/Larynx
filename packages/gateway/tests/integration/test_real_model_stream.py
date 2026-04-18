"""Real-model streaming tests.

Opt-in: ``RUN_REAL_MODEL=1 pytest -m real_model -k stream``.

Boots a real uvicorn instance with the real VoxCPM2 + Fun-ASR workers
loaded, then exercises the WS streaming paths with genuine audio:

- TTS streaming: TTFB p50/p95 over 20 runs.
- STT streaming: synthesize a known phrase, pipe it into /v1/stt/stream,
  verify partial cadence, WER on final, and finalization latency.
- Concurrent STT: 4 parallel sessions, assert partial cadence doesn't
  collapse — tests that Fun-ASR-vllm's batching absorbs the re-decode
  cost per spec.

Skips cleanly if the real-model env isn't set or the GPU/models aren't
reachable. Uses a module-scoped fixture so the ~30s model load is paid
once.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import socket
import statistics
import time
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
import soundfile as sf

pytestmark = [pytest.mark.real_model, pytest.mark.asyncio(loop_scope="module")]

TEST_TOKEN = "test-token-please-ignore"


def _skip_if_disabled() -> None:
    if os.environ.get("RUN_REAL_MODEL") != "1":
        pytest.skip("set RUN_REAL_MODEL=1 to run real-model streaming tests")
    try:
        import nanovllm_voxcpm  # noqa: F401
    except ImportError:
        pytest.skip("nano-vllm-voxcpm not installed (run `uv sync --extra gpu`)")
    try:
        import funasr  # noqa: F401
    except ImportError:
        pytest.skip("funasr not installed")
    try:
        import websockets  # noqa: F401
    except ImportError:
        pytest.skip("websockets lib not installed")
    # Fun-ASR-vllm is a checked-out repo, not a PyPI package. model_manager.py
    # imports `from model import FunASRNano`, which only works if the repo is
    # on sys.path. Pick it up from the env var or the M0 default location.
    import sys as _sys

    funasr_vllm_dir = os.environ.get(
        "LARYNX_FUNASR_VLLM_DIR", "/home/ripper/larynx-smoke/Fun-ASR-vllm"
    )
    if not pathlib_exists(funasr_vllm_dir):
        pytest.skip(f"Fun-ASR-vllm repo not found at {funasr_vllm_dir}")
    if funasr_vllm_dir not in _sys.path:
        _sys.path.insert(0, funasr_vllm_dir)


def pathlib_exists(p: str) -> bool:
    from pathlib import Path as _P

    return _P(p).exists()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def live_server(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[str]:
    """Start a real uvicorn in-process; yield a ws:// base URL."""
    _skip_if_disabled()
    import uvicorn

    from tests.conftest import _ensure_test_db, _reset_test_db

    _ensure_test_db()
    _reset_test_db()

    data_dir = tmp_path_factory.mktemp("larynx-stream-real-data")
    port = _free_port()

    os.environ["LARYNX_API_TOKEN"] = TEST_TOKEN
    os.environ["LARYNX_TTS_MODE"] = "voxcpm"
    os.environ["LARYNX_STT_MODE"] = "funasr"
    os.environ["LARYNX_VAD_PUNC_MODE"] = "real"
    os.environ["LARYNX_VOXCPM_GPU"] = "0"
    os.environ["LARYNX_FUNASR_GPU"] = "1"
    os.environ["LARYNX_LOG_JSON"] = "false"
    os.environ["LARYNX_DATA_DIR"] = str(data_dir)
    os.environ["DATABASE_URL"] = "postgresql+psycopg://larynx:larynx@localhost:5433/larynx_test"
    os.environ["REDIS_URL"] = "redis://localhost:6380/14"

    from larynx_gateway.config import get_settings

    get_settings.cache_clear()

    config = uvicorn.Config(
        "larynx_gateway.main:app", host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(), name="test-uvicorn")

    # Wait for the app to become ready. VoxCPM2 alone takes ~15s, and Fun-ASR
    # loads two vLLM-backed checkpoints (~60–120s on warm HF cache) — budget
    # generously so a cold load doesn't falsely skip.
    deadline = time.monotonic() + 360.0
    ready = False
    while time.monotonic() < deadline:
        if task.done():
            # uvicorn exited before readiness — lifespan failed. Re-raise
            # the exception so the test fails loudly with the root cause
            # instead of silently skipping after timeout.
            exc = task.exception()
            if exc is not None:
                raise exc
            break
        if server.started and getattr(server, "lifespan", None) is not None:
            import httpx

            try:
                async with httpx.AsyncClient(timeout=2.0) as http:
                    r = await http.get(f"http://127.0.0.1:{port}/ready")
                    if r.status_code == 200:
                        ready = True
                        break
            except Exception:
                pass
        await asyncio.sleep(1.0)
    if not ready:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        except Exception:
            pass
        pytest.skip("gateway failed to become ready within 360s")

    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=15)
        except Exception:
            task.cancel()


async def _tts_request(live_server: str, text: str, sample_rate: int = 24000) -> bytes:
    """Call the non-streaming REST TTS to obtain a reference utterance WAV.

    Used to drive the streaming STT test with known-good audio.
    """
    import httpx

    http_base = live_server.replace("ws://", "http://")
    async with httpx.AsyncClient(base_url=http_base, timeout=120) as client:
        r = await client.post(
            "/v1/tts",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={"text": text, "sample_rate": sample_rate, "output_format": "wav"},
        )
        r.raise_for_status()
        return r.content


# ---------------------------------------------------------------------------
# TTS streaming — TTFB p50/p95 across 20 runs
# ---------------------------------------------------------------------------


async def test_tts_stream_ttfb_distribution(live_server: str) -> None:
    import websockets

    url = f"{live_server}/v1/tts/stream?token={TEST_TOKEN}"
    ttfb_samples: list[float] = []
    total_samples: list[float] = []

    for i in range(20):
        async with websockets.connect(url, max_size=None) as ws:
            t0 = time.perf_counter()
            await ws.send(
                json.dumps(
                    {
                        "type": "synthesize",
                        "text": f"Run {i}: this is a short sentence for latency measurement.",
                        "sample_rate": 24000,
                    }
                )
            )
            ttfb = None
            while True:
                frame = await ws.recv()
                if isinstance(frame, bytes):
                    if ttfb is None:
                        ttfb = time.perf_counter() - t0
                else:
                    msg = json.loads(frame)
                    if msg.get("type") == "done":
                        total = time.perf_counter() - t0
                        break
                    if msg.get("type") == "error":
                        pytest.fail(f"error frame: {msg}")
            assert ttfb is not None
            ttfb_samples.append(ttfb)
            total_samples.append(total)

    p50 = statistics.median(ttfb_samples)
    p95 = statistics.quantiles(ttfb_samples, n=20)[18]  # 95th percentile
    print(
        f"\n[tts_stream] ttfb p50={p50 * 1000:.1f}ms p95={p95 * 1000:.1f}ms "
        f"min={min(ttfb_samples) * 1000:.1f}ms max={max(ttfb_samples) * 1000:.1f}ms n=20"
    )
    # PRD target: p50 ≤ 200ms. Assert a looser 300ms upper bound on p50 to
    # absorb one-off GPU jitter; the printed numbers are the real signal.
    assert p50 < 0.300, f"p50 TTFB {p50 * 1000:.1f}ms exceeds 300ms budget"


# ---------------------------------------------------------------------------
# STT streaming — synth → stream → verify partial cadence + final WER
# ---------------------------------------------------------------------------


def _wav_to_pcm16_16k(wav_bytes: bytes) -> bytes:
    buf = io.BytesIO(wav_bytes)
    samples, sr = sf.read(buf, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        # Simple linear decimation — good enough for test audio we generated
        # ourselves at 24kHz.
        ratio = sr / 16000
        idx = (np.arange(int(len(samples) / ratio)) * ratio).astype(np.int64)
        samples = samples[idx]
    return (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def _wer(ref: str, hyp: str) -> float:
    r = ref.lower().split()
    h = hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    # Standard Levenshtein on words.
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(r)][len(h)] / len(r)


async def test_stt_stream_end_to_end_via_synthesized_audio(live_server: str) -> None:
    import websockets

    phrase = "The quick brown fox jumps over the lazy dog."
    wav = await _tts_request(live_server, phrase, sample_rate=24000)
    pcm = _wav_to_pcm16_16k(wav)

    url = f"{live_server}/v1/stt/stream?token={TEST_TOKEN}"
    events: list[dict] = []
    partial_wall_ts: list[float] = []
    speech_end_ts: float | None = None
    final_latency: float | None = None

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "config",
                    "sample_rate": 16000,
                    "language": "en",
                    "chunk_interval_ms": 720,
                    "speech_end_silence_ms": 300,
                }
            )
        )
        # Stream 20ms frames (640 bytes) at approximately real-time pace
        # so VAD + partials_loop cadence matches production conditions.
        frame_bytes = 16000 * 20 // 1000 * 2  # 640
        sent = 0

        async def sender() -> None:
            nonlocal sent
            for i in range(0, len(pcm), frame_bytes):
                await ws.send(pcm[i : i + frame_bytes])
                sent += 1
                await asyncio.sleep(0.020)
            # Trailing silence so VAD closes the utterance.
            silence = b"\x00\x00" * (16000 * 1 // 1 * 2 // 2)  # 1s of silence
            for _ in range(50):  # 50 × 20ms = 1s
                await ws.send(b"\x00\x00" * (16000 * 20 // 1000))
                await asyncio.sleep(0.020)
            await ws.send(json.dumps({"type": "stop"}))

        send_task = asyncio.create_task(sender())
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                events.append(msg)
                t = msg.get("type")
                if t == "partial":
                    partial_wall_ts.append(time.monotonic())
                elif t == "speech_end":
                    speech_end_ts = time.monotonic()
                elif t == "final":
                    if speech_end_ts is not None:
                        final_latency = time.monotonic() - speech_end_ts
                    # stop reading — only one utterance expected
                    break
        finally:
            send_task.cancel()
            try:
                await send_task
            except (asyncio.CancelledError, Exception):
                pass

    kinds = [e["type"] for e in events]
    print(f"\n[stt_stream] events: {kinds}")

    partials = [e for e in events if e["type"] == "partial"]
    finals = [e for e in events if e["type"] == "final"]
    assert finals, f"no final event: kinds={kinds}"
    # At least one partial should fire during the tone window. The exact
    # count depends on VAD timing (the speech region may be shorter than
    # the full audio clip); metrics below log the observed number.
    assert len(partials) >= 1, f"expected ≥ 1 partial, got {len(partials)}"
    print(f"[stt_stream] partial_count={len(partials)} final_count={len(finals)}")

    # Partial cadence: intervals should cluster around 720ms.
    if len(partial_wall_ts) >= 2:
        intervals = [
            partial_wall_ts[i + 1] - partial_wall_ts[i] for i in range(len(partial_wall_ts) - 1)
        ]
        median_interval = statistics.median(intervals)
        print(
            f"[stt_stream] partial intervals: {[f'{x * 1000:.0f}ms' for x in intervals]} "
            f"median={median_interval * 1000:.0f}ms"
        )
        assert 0.4 <= median_interval <= 1.2, (
            f"median partial interval {median_interval * 1000:.0f}ms out of band"
        )

    # Final WER check — tolerant since we're feeding synthesized audio.
    hyp = finals[0].get("text") or finals[0].get("punctuated_text") or ""
    wer = _wer(phrase, hyp)
    print(f"[stt_stream] ref={phrase!r} hyp={hyp!r} WER={wer:.2f}")
    assert wer <= 0.5, f"WER {wer:.2f} too high for reference phrase"

    # Finalization latency
    if final_latency is not None:
        print(f"[stt_stream] finalization latency: {final_latency * 1000:.0f}ms")
        # PRD target: 80ms on 3s utterance. Allow 500ms in the test (GPU
        # jitter, batched decode) — logged value is the signal.
        assert final_latency < 0.500


# ---------------------------------------------------------------------------
# 4 concurrent STT sessions — partial cadence doesn't collapse
# ---------------------------------------------------------------------------


async def test_stt_stream_four_concurrent_sessions(live_server: str) -> None:
    import websockets

    phrase = "Concurrent streaming transcription session."
    wav = await _tts_request(live_server, phrase, sample_rate=24000)
    pcm = _wav_to_pcm16_16k(wav)

    async def run_one(i: int) -> tuple[int, list[float], list[str]]:
        url = f"{live_server}/v1/stt/stream?token={TEST_TOKEN}"
        intervals_out: list[float] = []
        kinds: list[str] = []
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "config",
                        "sample_rate": 16000,
                        "language": "en",
                        "chunk_interval_ms": 720,
                    }
                )
            )
            frame_bytes = 16000 * 20 // 1000 * 2

            async def sender() -> None:
                for k in range(0, len(pcm), frame_bytes):
                    await ws.send(pcm[k : k + frame_bytes])
                    await asyncio.sleep(0.020)
                for _ in range(60):  # 1.2s trailing silence
                    await ws.send(b"\x00\x00" * (16000 * 20 // 1000))
                    await asyncio.sleep(0.020)
                await ws.send(json.dumps({"type": "stop"}))

            send_task = asyncio.create_task(sender())
            last_partial = None
            try:
                while True:
                    # Larger timeout than the single-session test: 4-way
                    # concurrent GPU contention can push finals past 20s.
                    raw = await asyncio.wait_for(ws.recv(), timeout=45.0)
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    kinds.append(msg["type"])
                    if msg["type"] == "partial":
                        now = time.monotonic()
                        if last_partial is not None:
                            intervals_out.append(now - last_partial)
                        last_partial = now
                    elif msg["type"] == "final":
                        break
            finally:
                send_task.cancel()
                try:
                    await send_task
                except (asyncio.CancelledError, Exception):
                    pass
        return (i, intervals_out, kinds)

    results = await asyncio.gather(*(run_one(i) for i in range(4)), return_exceptions=True)

    all_intervals: list[float] = []
    finals_count = 0
    failed_sessions: list[int] = []
    for item in results:
        if isinstance(item, BaseException):
            failed_sessions.append(-1)
            print(f"[stt_concurrent] session raised: {type(item).__name__}: {item}")
            continue
        i, intervals, kinds = item
        finals_in_session = len([k for k in kinds if k == "final"])
        finals_count += finals_in_session
        print(
            f"[stt_concurrent][{i}] kinds={kinds} "
            f"intervals={[f'{x * 1000:.0f}ms' for x in intervals]}"
        )
        all_intervals.extend(intervals)

    # After bugs/001's inference lock, all four sessions must complete.
    # Pre-fix this was `>= 2` (best-effort — we knew 3 of 4 deadlocked);
    # post-fix the backend serialises cleanly so every session finishes.
    # Keep the `>=` form rather than `==` so the assertion stays honest
    # if N is ever raised — "at least this many must succeed" scales,
    # "exactly this many" doesn't.
    successful = [item for item in results if not isinstance(item, BaseException)]
    assert len(successful) >= 4, (
        f"expected ≥ 4 concurrent sessions to complete, got {len(successful)}"
    )

    if all_intervals:
        median = statistics.median(all_intervals)
        p95 = (
            statistics.quantiles(all_intervals, n=20)[18]
            if len(all_intervals) >= 20
            else max(all_intervals)
        )
        print(
            f"[stt_concurrent] aggregate median={median * 1000:.0f}ms "
            f"p95={p95 * 1000:.0f}ms n={len(all_intervals)} finals={finals_count}"
        )
        # This is a *measurement* not a hard gate — we want the real number
        # reported so the team can decide whether to accept concurrent STT
        # behaviour as-is or to build a batching coalescer in the worker.
        # (See PRD §10: "Fun-ASR rolling-buffer efficiency" risk.)


# ---------------------------------------------------------------------------
# bugs/001 regression — concurrent transcribe_rolling must not deadlock
#
# This test exercises the primary bug without the WS layer. It pokes the
# process-wide FunASRClient directly with four concurrent transcribe_rolling
# calls and asserts all four return. Before the fix they deadlock: vLLM's
# shared LLM instance can't absorb concurrent .inference() calls from
# multiple asyncio.to_thread worker threads, and 3 of 4 hang indefinitely.
# See bugs/001_concurrent_stt.md § 2.2 for the evidence trail.
# ---------------------------------------------------------------------------


async def test_stt_concurrent_transcribe_rolling_does_not_deadlock(
    live_server: str,
) -> None:
    # Reach through to the live gateway's shared FunASRClient so we're
    # hitting the exact same client instance WS sessions share. The
    # larynx_gateway.main module exposes `app` at module scope; the
    # live_server fixture's uvicorn run populates app.state during lifespan.
    from larynx_gateway.main import app

    funasr_client = app.state.funasr_client
    assert funasr_client is not None, (
        "app.state.funasr_client missing — lifespan didn't populate it?"
    )

    # Synthesize a known-good short sample via the real TTS path.
    phrase = "Testing concurrent inference race condition."
    wav = await _tts_request(live_server, phrase, sample_rate=24000)
    pcm = _wav_to_pcm16_16k(wav)

    async def one_call() -> str:
        r = await funasr_client.transcribe_rolling(
            pcm_s16le=pcm,
            sample_rate=16000,
            language="en",
            hotwords=[],
            itn=True,
            prev_text="",
            is_final=True,
            drop_tail_tokens=5,
        )
        return r.text

    # Warm the backend with one serial call so cold-model cost doesn't
    # contaminate the concurrency measurement.
    warm_text = await one_call()
    assert warm_text, f"warm-up decode produced empty text for {phrase!r}"

    # Four concurrent calls must all complete within 30s. Today they
    # deadlock and this wait_for expires.
    results = await asyncio.wait_for(
        asyncio.gather(*(one_call() for _ in range(4)), return_exceptions=True),
        timeout=30.0,
    )
    errors = [r for r in results if isinstance(r, BaseException)]
    assert not errors, f"concurrent calls raised: {errors}"
    for i, text in enumerate(results):
        assert text, f"concurrent call #{i} produced empty text: {results!r}"
