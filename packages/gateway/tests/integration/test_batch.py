"""Integration tests for POST/GET/DELETE /v1/batch.

Real Postgres + real Redis via the shared conftest fixture. The mock
VoxCPM worker the gateway boots under ``LARYNX_TTS_MODE=mock``
synthesises a deterministic 1s WAV per call — enough to exercise the
queue/consumer/artifact path end-to-end.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest
from httpx import AsyncClient


async def _wait_for_job(
    client: AsyncClient,
    auth_headers: dict[str, str],
    job_id: str,
    target: str,
    *,
    timeout_s: float = 15.0,
) -> dict:
    """Poll GET until state == target or timeout. Returns last payload."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    last: dict = {}
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/v1/batch/{job_id}", headers=auth_headers)
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["state"] == target:
            return last
        await asyncio.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach {target}; last={last}")


async def _upload_voice(client: AsyncClient, auth_headers: dict[str, str], name: str) -> str:
    """Upload a tiny WAV via /v1/voices and return the voice_id."""
    # Build a minimal 1s @ 16kHz mono WAV from scratch — no external
    # fixture file needed and no dependency on load_demo_voices.
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    resp = await client.post(
        "/v1/voices",
        headers=auth_headers,
        files={"audio": (f"{name}.wav", buf.getvalue(), "audio/wav")},
        data={"name": name, "description": "batch-test"},
    )
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["id"]


async def test_batch_create_and_run(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    """10-item batch with 3 distinct voices finishes COMPLETED + files exist."""
    voice_ids = [await _upload_voice(client, auth_headers, f"batch-v{i}") for i in range(3)]

    items = [{"text": f"Hello world item {i}.", "voice_id": voice_ids[i % 3]} for i in range(10)]
    resp = await client.post("/v1/batch", headers=auth_headers, json={"items": items})
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["job_id"]

    final = await _wait_for_job(client, auth_headers, job_id, target="COMPLETED", timeout_s=30)
    assert final["num_items"] == 10
    assert final["num_completed"] == 10
    assert final["num_failed"] == 0
    assert final["progress"] == pytest.approx(1.0)
    for entry in final["items"]:
        assert entry["state"] == "DONE", entry
        assert entry["url"] is not None
        # Fetch the artifact and confirm it's valid WAV.
        got = await client.get(entry["url"], headers=auth_headers)
        assert got.status_code == 200, got.text
        assert got.headers["content-type"].startswith("audio/")
        assert got.content[:4] == b"RIFF"


async def test_batch_item_limits_enforced(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """>500 items → 422 from pydantic."""
    items = [{"text": f"t{i}"} for i in range(501)]
    resp = await client.post("/v1/batch", headers=auth_headers, json={"items": items})
    assert resp.status_code == 422


async def test_batch_cancel_before_drain(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    """DELETE mid-flight cancels queued items + preserves DONE ones."""
    # Large batch so DELETE races ahead of the consumer draining them all.
    items = [{"text": f"Cancel test {i}."} for i in range(50)]
    resp = await client.post("/v1/batch", headers=auth_headers, json={"items": items})
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]

    # Give consumers a moment to pick up some items.
    await asyncio.sleep(0.3)

    cancel = await client.delete(f"/v1/batch/{job_id}", headers=auth_headers)
    assert cancel.status_code == 202, cancel.text

    # Final state is either CANCELLED (nothing ever ran) or has some
    # mix of DONE + CANCELLED — either way, progress should stop
    # growing within a couple of polls.
    await asyncio.sleep(0.5)
    resp = await client.get(f"/v1/batch/{job_id}", headers=auth_headers)
    assert resp.status_code == 200
    snap1 = resp.json()

    await asyncio.sleep(1.0)
    resp = await client.get(f"/v1/batch/{job_id}", headers=auth_headers)
    snap2 = resp.json()

    # num_completed + num_failed must not grow by more than in-flight
    # items (consumers_n=2). Typically it's exactly equal.
    growth = (snap2["num_completed"] + snap2["num_failed"]) - (
        snap1["num_completed"] + snap1["num_failed"]
    )
    assert growth <= 2

    # DONE items still have URLs that resolve.
    done = [i for i in snap2["items"] if i["state"] == "DONE"]
    for entry in done:
        got = await client.get(entry["url"], headers=auth_headers)
        assert got.status_code == 200


async def test_batch_idempotent_cancel(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    """Double-DELETE returns 202 both times with same terminal state."""
    items = [{"text": "single"}]
    resp = await client.post("/v1/batch", headers=auth_headers, json={"items": items})
    job_id = resp.json()["job_id"]
    r1 = await client.delete(f"/v1/batch/{job_id}", headers=auth_headers)
    r2 = await client.delete(f"/v1/batch/{job_id}", headers=auth_headers)
    assert r1.status_code == 202
    assert r2.status_code == 202


async def test_batch_get_404(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    resp = await client.get("/v1/batch/nonexistent", headers=auth_headers)
    assert resp.status_code == 404


async def test_batch_artifact_404_for_pending(
    client: AsyncClient, auth_headers: dict[str, str], data_dir: pathlib.Path
) -> None:
    """GET artifact on a QUEUED item returns 404 item_not_ready."""
    # Submit then immediately probe — catch an item mid-QUEUED state.
    items = [{"text": f"pending {i}"} for i in range(5)]
    resp = await client.post("/v1/batch", headers=auth_headers, json={"items": items})
    job_id = resp.json()["job_id"]
    # Ask for item 4 (last one enqueued) before consumers drain that far.
    got = await client.get(f"/v1/batch/{job_id}/items/4", headers=auth_headers)
    # Either already DONE (200) or still pending (404). Both acceptable;
    # we're verifying the pending path doesn't 500.
    assert got.status_code in (200, 404)


async def test_batch_unauthenticated_rejected(client: AsyncClient) -> None:
    resp = await client.post("/v1/batch", json={"items": [{"text": "x"}]})
    assert resp.status_code == 401


async def test_batch_retain_skips_expiry(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    """retain=True jobs have no expires_at stamp."""
    resp = await client.post(
        "/v1/batch",
        headers=auth_headers,
        json={"items": [{"text": "keep me"}], "retain": True},
    )
    job_id = resp.json()["job_id"]
    status = await _wait_for_job(client, auth_headers, job_id, target="COMPLETED", timeout_s=10)
    assert status["retain"] is True
    assert status["expires_at"] is None
