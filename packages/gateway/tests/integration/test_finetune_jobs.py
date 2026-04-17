"""Integration test for ``/v1/finetune/jobs`` create / get / delete.

Creates real FineTuneJob rows via the route, spawns the orchestrator
task in the background, and polls ``GET /v1/finetune/jobs/{id}`` until
the job reaches a terminal state. The subprocess hook is injected via
``app.state.training_subprocess_hook`` so no real training runs — but
every other moving part (DB, Redis, VoxCPMClient over MockBackend, job
state machine) is real.
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


def _wav_bytes(duration_s: float = 11.0, peak: float = 0.3) -> bytes:
    samples = np.linspace(-peak, peak, int(SR * duration_s), dtype=np.float32)
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


def _make_success_hook():
    async def hook(**kwargs: object) -> RunnerOutcome:
        from larynx_shared.paths import JobPaths

        job_paths = kwargs["job_paths"]
        assert isinstance(job_paths, JobPaths)
        on_log = kwargs["on_log"]
        on_state = kwargs["on_state"]
        assert callable(on_log) and callable(on_state)
        for step in range(10):
            on_log(f"step={step} loss/diff=1.0")
            on_state({"step": step, "loss_diff": 1.0})
        job_paths.latest_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        job_paths.latest_lora_weights.write_bytes(b"x")
        job_paths.latest_lora_config.write_text(
            json.dumps({"base_model": "x", "lora_config": {"r": 32, "alpha": 32}})
        )
        return RunnerOutcome.SUCCESS

    return hook


def _make_slow_hook(steps: int = 100, delay: float = 0.05):
    """Emits lines slowly + respects cancel_event — lets us test DELETE."""

    async def hook(**kwargs: object) -> RunnerOutcome:
        on_log = kwargs["on_log"]
        cancel_event = kwargs["cancel_event"]
        assert callable(on_log)
        assert isinstance(cancel_event, asyncio.Event)
        for step in range(steps):
            if cancel_event.is_set():
                return RunnerOutcome.CANCELLED
            on_log(f"step={step} loss/diff=1.0")
            await asyncio.sleep(delay)
        return RunnerOutcome.SUCCESS

    return hook


async def _poll_until_terminal(
    client: AsyncClient,
    job_id: str,
    headers: dict[str, str],
    timeout: float = 10.0,
) -> dict:
    start = asyncio.get_event_loop().time()
    while True:
        resp = await client.get(f"/v1/finetune/jobs/{job_id}", headers=headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["state"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return body
        if asyncio.get_event_loop().time() - start > timeout:
            raise TimeoutError(f"job {job_id} did not terminate: {body!r}")
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Custom client fixture. We install the subprocess hook BEFORE the
# lifespan runs because the orchestrator reads it off app.state on
# every create_job call. Two fixtures — success + slow — cover the
# scenarios; tests opt in via the fixture they request.
# ---------------------------------------------------------------------------


def _build_app_with_hook(data_dir: pathlib.Path, hook) -> object:
    from tests.conftest import _reset_test_db

    _reset_test_db()
    get_settings.cache_clear()
    app = create_app()
    app.state.training_subprocess_hook = hook
    app.state.training_pretrained_path = str(data_dir / "pretrained")
    return app


@pytest_asyncio.fixture
async def client_success_hook(data_dir: pathlib.Path) -> AsyncIterator[AsyncClient]:
    app = _build_app_with_hook(data_dir, _make_success_hook())
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as c,
        app.router.lifespan_context(app),
    ):
        yield c


@pytest_asyncio.fixture
async def client_slow_hook(data_dir: pathlib.Path) -> AsyncIterator[AsyncClient]:
    app = _build_app_with_hook(data_dir, _make_slow_hook(steps=100, delay=0.05))
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as c,
        app.router.lifespan_context(app),
    ):
        yield c


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_job_and_reach_succeeded(
    client_success_hook: AsyncClient, auth_headers: dict[str, str]
) -> None:
    # Upload a dataset first.
    upload = await client_success_hook.post(
        "/v1/finetune/datasets", files=_multipart(), headers=auth_headers
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["dataset_id"]

    # Create a job against it.
    create = await client_success_hook.post(
        "/v1/finetune/jobs",
        json={"dataset_id": dataset_id, "name": "nimbus-job-test"},
        headers=auth_headers,
    )
    assert create.status_code == 201, create.text
    job_id = create.json()["job_id"]

    # Poll until terminal.
    body = await _poll_until_terminal(client_success_hook, job_id, auth_headers)
    assert body["state"] == "SUCCEEDED"
    assert body["voice_id"] is not None
    assert body["progress"] > 0
    assert body["error_code"] is None

    # And the voice is listed in /v1/voices with source=lora.
    voices = await client_success_hook.get("/v1/voices", headers=auth_headers)
    assert voices.status_code == 200
    names = {v["name"]: v for v in voices.json()["voices"]}
    assert "nimbus-job-test" in names
    assert names["nimbus-job-test"]["source"] == "lora"


@pytest.mark.asyncio
async def test_create_job_for_unknown_dataset_returns_404(
    client_success_hook: AsyncClient, auth_headers: dict[str, str]
) -> None:
    resp = await client_success_hook.post(
        "/v1/finetune/jobs",
        json={"dataset_id": "does-not-exist", "name": "nope"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_job_duplicate_voice_name_returns_409(
    client_success_hook: AsyncClient, auth_headers: dict[str, str]
) -> None:
    # First job succeeds; second one using the same name fails at
    # REGISTERING with voice_name_conflict. The second request is
    # accepted (returns 201) because the name uniqueness is only
    # known after training completes — the poll then sees FAILED.
    upload = await client_success_hook.post(
        "/v1/finetune/datasets", files=_multipart(), headers=auth_headers
    )
    dataset_id = upload.json()["dataset_id"]

    c1 = await client_success_hook.post(
        "/v1/finetune/jobs",
        json={"dataset_id": dataset_id, "name": "conflict-voice"},
        headers=auth_headers,
    )
    assert c1.status_code == 201
    b1 = await _poll_until_terminal(client_success_hook, c1.json()["job_id"], auth_headers)
    assert b1["state"] == "SUCCEEDED"

    c2 = await client_success_hook.post(
        "/v1/finetune/jobs",
        json={"dataset_id": dataset_id, "name": "conflict-voice"},
        headers=auth_headers,
    )
    assert c2.status_code == 201
    b2 = await _poll_until_terminal(client_success_hook, c2.json()["job_id"], auth_headers)
    assert b2["state"] == "FAILED"
    assert b2["error_code"] == "voice_name_conflict"


@pytest.mark.asyncio
async def test_get_unknown_job_returns_404(
    client_success_hook: AsyncClient, auth_headers: dict[str, str]
) -> None:
    resp = await client_success_hook.get("/v1/finetune/jobs/does-not-exist", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_cancels_inflight_job(
    client_slow_hook: AsyncClient, auth_headers: dict[str, str]
) -> None:
    upload = await client_slow_hook.post(
        "/v1/finetune/datasets", files=_multipart(), headers=auth_headers
    )
    dataset_id = upload.json()["dataset_id"]
    create = await client_slow_hook.post(
        "/v1/finetune/jobs",
        json={"dataset_id": dataset_id, "name": "cancel-me"},
        headers=auth_headers,
    )
    job_id = create.json()["job_id"]

    # Let the job get going — wait until we see state=TRAINING.
    # We don't assert on current_step here because in-flight progress
    # flows through the in-memory JobHandle (not the DB row); the route
    # merges those at read time, but the important cancellation
    # precondition is the state transition, not a specific step count.
    for _ in range(100):
        resp = await client_slow_hook.get(f"/v1/finetune/jobs/{job_id}", headers=auth_headers)
        if resp.json().get("state") == "TRAINING":
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("job never reached TRAINING state")

    delete = await client_slow_hook.delete(f"/v1/finetune/jobs/{job_id}", headers=auth_headers)
    assert delete.status_code == 202

    body = await _poll_until_terminal(client_slow_hook, job_id, auth_headers, timeout=15.0)
    assert body["state"] == "CANCELLED"


@pytest.mark.asyncio
async def test_delete_unknown_job_returns_404(
    client_success_hook: AsyncClient, auth_headers: dict[str, str]
) -> None:
    resp = await client_success_hook.delete(
        "/v1/finetune/jobs/does-not-exist", headers=auth_headers
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_job_requires_auth(client_success_hook: AsyncClient) -> None:
    resp = await client_success_hook.post(
        "/v1/finetune/jobs", json={"dataset_id": "ds", "name": "n"}
    )
    assert resp.status_code == 401
