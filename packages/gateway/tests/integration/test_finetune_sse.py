"""Integration test for ``GET /v1/finetune/jobs/{id}/logs`` SSE.

Covers the Last-Event-ID reconnect semantics, terminal-state close,
and the structured ``train_state`` event decoration.
"""

from __future__ import annotations

import asyncio
import io
import json
import pathlib
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
import soundfile as sf
from httpx import ASGITransport, AsyncClient
from larynx_gateway.config import get_settings
from larynx_gateway.main import create_app
from larynx_training_worker.subprocess_runner import RunnerOutcome

SR = 16_000


def _wav_bytes(duration_s: float = 11.0) -> bytes:
    samples = np.linspace(-0.3, 0.3, int(SR * duration_s), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _multipart(num_clips: int = 30) -> list[tuple[str, tuple[str, bytes, str]]]:
    files = []
    rows = []
    for i in range(num_clips):
        name = f"clip{i:02d}.wav"
        files.append(("files", (name, _wav_bytes(), "audio/wav")))
        rows.append({"audio": name, "text": f"sample {i}"})
    manifest = "\n".join(json.dumps(r) for r in rows).encode()
    files.append(("files", ("transcripts.jsonl", manifest, "application/x-jsonlines")))
    return files


def _make_slow_hook(steps: int = 20, delay: float = 0.05):
    async def hook(**kwargs: object) -> RunnerOutcome:
        from larynx_shared.paths import JobPaths

        on_log = kwargs["on_log"]
        on_state = kwargs["on_state"]
        cancel_event = kwargs["cancel_event"]
        job_paths = kwargs["job_paths"]
        assert callable(on_log) and callable(on_state)
        assert isinstance(cancel_event, asyncio.Event)
        assert isinstance(job_paths, JobPaths)
        for step in range(steps):
            if cancel_event.is_set():
                return RunnerOutcome.CANCELLED
            on_log(f"step={step} loss/diff={1.0 - 0.01 * step} lr=0.0001")
            on_state({"step": step, "loss_diff": 1.0 - 0.01 * step, "lr": 0.0001})
            await asyncio.sleep(delay)
        job_paths.latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        job_paths.latest_lora_weights.write_bytes(b"x")
        job_paths.latest_lora_config.write_text(
            json.dumps({"base_model": "x", "lora_config": {"r": 32, "alpha": 32}})
        )
        return RunnerOutcome.SUCCESS

    return hook


@pytest_asyncio.fixture
async def client_slow(data_dir: pathlib.Path) -> AsyncIterator[AsyncClient]:
    from tests.conftest import _reset_test_db

    _reset_test_db()
    get_settings.cache_clear()
    app = create_app()
    app.state.training_subprocess_hook = _make_slow_hook(steps=30, delay=0.03)
    app.state.training_pretrained_path = str(data_dir / "pretrained")

    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as c,
        app.router.lifespan_context(app),
    ):
        yield c


async def _seed_job(client: AsyncClient, headers: dict[str, str]) -> str:
    upload = await client.post("/v1/finetune/datasets", files=_multipart(), headers=headers)
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["dataset_id"]
    create = await client.post(
        "/v1/finetune/jobs",
        json={"dataset_id": dataset_id, "name": "sse-test-voice"},
        headers=headers,
    )
    assert create.status_code == 201
    return create.json()["job_id"]


def _parse_sse(text: str) -> list[dict[str, str]]:
    """Minimal SSE parser: splits on blank lines, parses ``id:`` /
    ``event:`` / ``data:`` fields per event.
    """
    events: list[dict[str, str]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev: dict[str, str] = {}
        data_lines = []
        for line in block.split("\n"):
            if line.startswith(":"):
                continue  # comment / heartbeat
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            value = value.lstrip(" ")
            if key == "data":
                data_lines.append(value)
            else:
                ev[key] = value
        if data_lines:
            ev["data"] = "\n".join(data_lines)
        if ev:
            events.append(ev)
    return events


@pytest.mark.asyncio
async def test_sse_streams_logs_until_terminal(
    client_slow: AsyncClient, auth_headers: dict[str, str]
) -> None:
    job_id = await _seed_job(client_slow, auth_headers)

    # Open the SSE stream. The endpoint closes when the job reaches a
    # terminal state, so a full GET returns the whole stream text.
    resp = await client_slow.get(
        f"/v1/finetune/jobs/{job_id}/logs",
        headers=auth_headers,
        timeout=15.0,
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)

    log_events = [e for e in events if e.get("event") == "log"]
    state_events = [e for e in events if e.get("event") == "state"]
    terminal_events = [e for e in events if e.get("event") == "terminal"]

    assert len(log_events) > 0, "expected at least one log event"
    assert len(state_events) > 0, "expected at least one state event"
    assert len(terminal_events) == 1, "expected exactly one terminal event"

    # Each state event has a step number >= 0.
    for ev in state_events:
        payload = json.loads(ev["data"])
        assert payload["step"] >= 0

    terminal = json.loads(terminal_events[0]["data"])
    assert terminal["state"] in ("SUCCEEDED", "FAILED", "CANCELLED")


@pytest.mark.asyncio
async def test_sse_resumes_from_last_event_id(
    client_slow: AsyncClient, auth_headers: dict[str, str]
) -> None:
    job_id = await _seed_job(client_slow, auth_headers)

    # First connection: read the full stream.
    resp = await client_slow.get(
        f"/v1/finetune/jobs/{job_id}/logs",
        headers=auth_headers,
        timeout=15.0,
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    log_events = [e for e in events if e.get("event") == "log" and "id" in e]
    assert len(log_events) >= 2

    # Reconnect with Last-Event-ID set to the id of the second-to-last
    # log line; the replay must start strictly after that id.
    anchor_id = log_events[-2]["id"]
    headers2 = {**auth_headers, "Last-Event-ID": anchor_id}
    resp2 = await client_slow.get(
        f"/v1/finetune/jobs/{job_id}/logs",
        headers=headers2,
        timeout=15.0,
    )
    assert resp2.status_code == 200
    events2 = _parse_sse(resp2.text)
    # Same job, terminal state repeats (it's a snapshot of the DB row).
    terminal2 = [e for e in events2 if e.get("event") == "terminal"]
    assert len(terminal2) == 1
    # Log-event ids should all be strictly greater than anchor.
    log_ids2 = [e["id"] for e in events2 if e.get("event") == "log" and "id" in e]
    assert all(i > anchor_id for i in log_ids2), f"got ids {log_ids2} vs anchor {anchor_id}"


@pytest.mark.asyncio
async def test_sse_unknown_job_returns_404(
    client_slow: AsyncClient, auth_headers: dict[str, str]
) -> None:
    resp = await client_slow.get("/v1/finetune/jobs/does-not-exist/logs", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_sse_requires_auth(client_slow: AsyncClient) -> None:
    resp = await client_slow.get("/v1/finetune/jobs/any/logs")
    assert resp.status_code == 401
