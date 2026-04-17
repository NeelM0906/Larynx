"""Integration test for ``POST /v1/finetune/datasets``.

Exercises the full multipart upload path: staging dir creation, file
write, Phase-A validation, atomic rename to the final dataset dir.
Real Postgres + real TTS mock backend booted through the app factory
— no route-level shortcuts.
"""

from __future__ import annotations

import io
import json
import pathlib

import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient
from larynx_shared.paths import DatasetPaths

SR = 16_000


def _wav_bytes(duration_s: float = 11.0, peak: float = 0.3) -> bytes:
    samples = np.linspace(-peak, peak, int(SR * duration_s), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _build_multipart(
    num_clips: int, *, with_manifest: bool
) -> list[tuple[str, tuple[str, bytes, str]]]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    rows = []
    for i in range(num_clips):
        name = f"clip{i:02d}.wav"
        files.append(("files", (name, _wav_bytes(), "audio/wav")))
        rows.append({"audio": name, "text": f"sample {i}"})
    if with_manifest:
        manifest = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
        files.append(("files", ("transcripts.jsonl", manifest, "application/x-jsonlines")))
    return files


@pytest.mark.asyncio
async def test_dataset_upload_happy_path(
    client: AsyncClient, data_dir: pathlib.Path, auth_headers: dict[str, str]
) -> None:
    # 30 × 11s clips = 330s total > 300s min. Manifest included.
    files = _build_multipart(num_clips=30, with_manifest=True)
    resp = await client.post("/v1/finetune/datasets", files=files, headers=auth_headers)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "dataset_id" in body
    ds_id = body["dataset_id"]
    assert body["report"]["ok"] is True
    assert body["report"]["num_clips"] == 30

    # Final dataset dir exists on disk; staging is gone.
    ds = DatasetPaths(data_dir=data_dir, dataset_id=ds_id)
    assert ds.base_dir.is_dir()
    assert not ds.staging_dir.exists()
    assert len(list(ds.audio_dir.glob("*.wav"))) == 30
    assert ds.transcripts_jsonl.is_file()


@pytest.mark.asyncio
async def test_dataset_upload_without_manifest_is_allowed(
    client: AsyncClient, data_dir: pathlib.Path, auth_headers: dict[str, str]
) -> None:
    files = _build_multipart(num_clips=30, with_manifest=False)
    resp = await client.post("/v1/finetune/datasets", files=files, headers=auth_headers)
    assert resp.status_code == 201, resp.text
    ds_id = resp.json()["dataset_id"]
    ds = DatasetPaths(data_dir=data_dir, dataset_id=ds_id)
    assert ds.base_dir.is_dir()
    assert not ds.transcripts_jsonl.is_file()  # auto-transcribe runs later in PREPARING


@pytest.mark.asyncio
async def test_dataset_upload_too_short_returns_400(
    client: AsyncClient, data_dir: pathlib.Path, auth_headers: dict[str, str]
) -> None:
    # Five 10s clips = 50s total < 300s min.
    files = _build_multipart(num_clips=5, with_manifest=True)
    # Shorter per-clip duration than default.
    # _build_multipart uses 11s each — for this test override explicitly.
    files = []
    rows = []
    for i in range(5):
        name = f"clip{i}.wav"
        files.append(("files", (name, _wav_bytes(duration_s=10.0), "audio/wav")))
        rows.append({"audio": name, "text": f"line {i}"})
    manifest = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
    files.append(("files", ("transcripts.jsonl", manifest, "application/x-jsonlines")))

    resp = await client.post("/v1/finetune/datasets", files=files, headers=auth_headers)
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["code"] == "dataset_invalid"
    codes = [i["code"] for i in body["detail"]["issues"]]
    assert "duration_too_short" in codes

    # Staging dir is gone so retries don't find garbage.
    assert not any(
        p.name.endswith(".staging")
        for p in (data_dir / "datasets").iterdir()
        if (data_dir / "datasets").exists()
    )


@pytest.mark.asyncio
async def test_dataset_upload_rejects_no_files(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    resp = await client.post("/v1/finetune/datasets", headers=auth_headers)
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_dataset_upload_requires_auth(
    client: AsyncClient,
) -> None:
    files = _build_multipart(num_clips=30, with_manifest=True)
    resp = await client.post("/v1/finetune/datasets", files=files)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_dataset_upload_rejects_filename_traversal(
    client: AsyncClient, data_dir: pathlib.Path, auth_headers: dict[str, str]
) -> None:
    # Server must refuse filenames with path traversal. If we accepted
    # ``../../etc/passwd`` we'd write outside the dataset dir.
    files = [
        ("files", ("../evil.wav", _wav_bytes(), "audio/wav")),
    ]
    resp = await client.post("/v1/finetune/datasets", files=files, headers=auth_headers)
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["code"] == "bad_filename"
