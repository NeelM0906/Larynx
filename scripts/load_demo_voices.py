#!/usr/bin/env python
"""Seed the voice library with six voices named for the OpenAI presets.

Idempotent: checks the gateway's GET /v1/voices before each operation and
skips what's already present. Re-runs on an existing library are free.

The six voices:

- ``alloy``   — uploaded from a LibriVox chapter (warm male baritone)
- ``echo``    — uploaded from a LibriVox chapter (expressive older male)
- ``nova``    — uploaded from a LibriVox chapter (clear female narrator)
- ``fable``   — designed via POST /v1/voices/design
- ``onyx``    — designed
- ``shimmer`` — designed

The older seed names (``librivox-male-baritone`` / ``librivox-male-
expressive`` / ``librivox-female-clear``) are left alone when present —
the shim only looks up by the new names, and the /v1/voices API doesn't
expose a rename verb. Admins can ``DELETE`` the old rows once the new
ones are confirmed.

Requires a running gateway on ``LARYNX_GATEWAY_URL`` (default
http://localhost:8000) with ``LARYNX_API_TOKEN`` set. Run after
``make up && make migrate && make run``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys
import tempfile
import time
import urllib.request
from typing import Any

import httpx


@dataclasses.dataclass(frozen=True)
class UploadSeed:
    """A voice sourced by slicing a LibriVox chapter mp3."""

    name: str
    description: str
    url: str
    # LibriVox chapter clips are ~30 minutes; we slice 15 s out of a
    # stable offset so the sample is short + consistent across runs.
    start_s: float = 2.0
    duration_s: float = 15.0


@dataclasses.dataclass(frozen=True)
class DesignSeed:
    """A voice sourced via POST /v1/voices/design + /save."""

    name: str
    description: str
    design_prompt: str


# Three LibriVox chapter recordings, each saved under an OpenAI-preset
# short-name so /v1/audio/speech can do a direct name lookup.
UPLOAD_SEEDS: list[UploadSeed] = [
    UploadSeed(
        name="alloy",
        description="Warm male baritone, moderate pace (seeded from LibriVox).",
        url="https://www.archive.org/download/alice_in_wonderland_librivox/wonderland_ch_01_carroll.mp3",
    ),
    UploadSeed(
        name="echo",
        description="Expressive older male (seeded from LibriVox).",
        url="https://www.archive.org/download/sherlockholmes_advsh_librivox/sherlockholmes_advsh_01_doyle.mp3",
    ),
    UploadSeed(
        name="nova",
        description="Clear female narrator (seeded from LibriVox).",
        url="https://www.archive.org/download/pride_prejudice_librivox/prideandprejudice_01_austen.mp3",
    ),
]


# Three designed voices, shapes fixed by the M8 spec (ORCHESTRATION-M8.md §2.2).
DESIGN_SEEDS: list[DesignSeed] = [
    DesignSeed(
        name="fable",
        description="Warm mid-range male storyteller, unhurried pace, British English.",
        design_prompt="warm mid-range male storyteller, unhurried pace, British English",
    ),
    DesignSeed(
        name="onyx",
        description="Deep resonant male, measured and authoritative, American English.",
        design_prompt="deep resonant male, measured and authoritative, American English",
    ),
    DesignSeed(
        name="shimmer",
        description="Soft intimate female, breathy but clear, American English.",
        design_prompt="soft intimate female, breathy but clear, American English",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gateway-url",
        default=os.environ.get("LARYNX_GATEWAY_URL", "http://localhost:8000"),
    )
    p.add_argument("--token", default=os.environ.get("LARYNX_API_TOKEN"))
    p.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get("LARYNX_SEED_CACHE", tempfile.gettempdir() + "/larynx-seeds")
        ),
    )
    p.add_argument(
        "--skip-designed",
        action="store_true",
        help="Only load the three upload seeds (useful when the worker's design path is slow).",
    )
    return p.parse_args()


def download_once(url: str, dest: pathlib.Path) -> pathlib.Path:
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def slice_audio(mp3: pathlib.Path, start_s: float, duration_s: float) -> bytes:
    """Slice a short WAV-encoded clip out of the source MP3.

    Uses librosa -> soundfile to avoid shelling out to ffmpeg; librosa is
    already a worker dep.
    """
    import io

    import librosa
    import soundfile as sf

    samples, sr = librosa.load(str(mp3), sr=24000, mono=True, offset=start_s, duration=duration_s)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def list_existing(client: httpx.Client, url: str, token: str) -> set[str]:
    r = client.get(
        f"{url}/v1/voices",
        headers={"Authorization": f"Bearer {token}"},
        params={"limit": 500},
    )
    r.raise_for_status()
    return {v["name"] for v in r.json()["voices"]}


def upload(client: httpx.Client, url: str, token: str, seed: UploadSeed, audio: bytes) -> Any:
    r = client.post(
        f"{url}/v1/voices",
        headers={"Authorization": f"Bearer {token}"},
        files={"audio": (f"{seed.name}.wav", audio, "audio/wav")},
        data={"name": seed.name, "description": seed.description},
    )
    r.raise_for_status()
    return r.json()


def design(client: httpx.Client, url: str, token: str, seed: DesignSeed) -> Any:
    """Run /v1/voices/design followed by /save — two-step flow.

    ``POST /v1/voices/design`` returns a ``preview_id`` (not a voice_id)
    and a cached preview clip. We then ``POST /v1/voices/design/{id}/save``
    to promote the preview into a permanent ``Voice`` row.
    """
    preview = client.post(
        f"{url}/v1/voices/design",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": seed.name,
            "description": seed.description,
            "design_prompt": seed.design_prompt,
        },
        timeout=120,
    )
    preview.raise_for_status()
    preview_id = preview.json()["preview_id"]

    save = client.post(
        f"{url}/v1/voices/design/{preview_id}/save",
        headers={"Authorization": f"Bearer {token}"},
        json={},
        timeout=60,
    )
    save.raise_for_status()
    return save.json()


def main() -> int:
    args = parse_args()
    if not args.token:
        print("error: LARYNX_API_TOKEN env var or --token required", file=sys.stderr)
        return 2

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=120) as client:
        try:
            existing = list_existing(client, args.gateway_url, args.token)
        except httpx.HTTPError as e:
            print(f"error: cannot reach gateway at {args.gateway_url}: {e}", file=sys.stderr)
            return 3

        for seed in UPLOAD_SEEDS:
            if seed.name in existing:
                print(f"[skip] {seed.name} already exists (upload)")
                continue
            mp3 = args.cache_dir / f"{seed.name}.mp3"
            download_once(seed.url, mp3)
            audio = slice_audio(mp3, seed.start_s, seed.duration_s)
            voice = upload(client, args.gateway_url, args.token, seed, audio)
            print(f"[ok]   {seed.name}  id={voice['id']}  duration_ms={voice['duration_ms']}")
            existing.add(seed.name)

        if args.skip_designed:
            print("[info] --skip-designed set, leaving fable/onyx/shimmer unseeded")
            return 0

        for dseed in DESIGN_SEEDS:
            if dseed.name in existing:
                print(f"[skip] {dseed.name} already exists (designed)")
                continue
            t0 = time.time()
            voice = design(client, args.gateway_url, args.token, dseed)
            print(
                f"[ok]   {dseed.name}  id={voice['id']}  "
                f"source={voice.get('source')}  took={time.time() - t0:.1f}s"
            )
            existing.add(dseed.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
