#!/usr/bin/env python
"""Seed the voice library with 3 public-domain LibriVox voices.

Idempotent: checks the gateway's GET /v1/voices before each upload and
skips any voice whose name already exists. Downloads each clip once
into ``${DATA_DIR}/seed_cache/`` so re-runs are cheap.

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
import urllib.request
from typing import Any

import httpx


@dataclasses.dataclass(frozen=True)
class SeedVoice:
    name: str
    description: str
    url: str
    # LibriVox chapter clips are ~30 minutes; we slice 15 s out of a
    # stable offset so the sample is short + consistent across runs.
    start_s: float = 0.0
    duration_s: float = 15.0


# Three LibriVox chapter recordings on a range of voice qualities.
# The Internet Archive serves raw 128 kbps MP3s from librivox.org.
SEEDS: list[SeedVoice] = [
    SeedVoice(
        name="librivox-male-baritone",
        description="Seed voice — warm male baritone, moderate pace.",
        url="https://www.archive.org/download/alice_in_wonderland_librivox/wonderland_ch_01_carroll.mp3",
        start_s=2.0,
        duration_s=15.0,
    ),
    SeedVoice(
        name="librivox-female-clear",
        description="Seed voice — clear female narrator.",
        url="https://www.archive.org/download/pride_prejudice_librivox/prideandprejudice_01_austen.mp3",
        start_s=2.0,
        duration_s=15.0,
    ),
    SeedVoice(
        name="librivox-male-expressive",
        description="Seed voice — expressive older male.",
        url="https://www.archive.org/download/sherlockholmes_advsh_librivox/sherlockholmes_advsh_01_doyle.mp3",
        start_s=2.0,
        duration_s=15.0,
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
    r = client.get(f"{url}/v1/voices", headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return {v["name"] for v in r.json()["voices"]}


def upload(client: httpx.Client, url: str, token: str, seed: SeedVoice, audio: bytes) -> Any:
    r = client.post(
        f"{url}/v1/voices",
        headers={"Authorization": f"Bearer {token}"},
        files={"audio": (f"{seed.name}.wav", audio, "audio/wav")},
        data={"name": seed.name, "description": seed.description},
    )
    r.raise_for_status()
    return r.json()


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

        for seed in SEEDS:
            if seed.name in existing:
                print(f"[skip] {seed.name} already exists")
                continue
            mp3 = args.cache_dir / f"{seed.name}.mp3"
            download_once(seed.url, mp3)
            audio = slice_audio(mp3, seed.start_s, seed.duration_s)
            voice = upload(client, args.gateway_url, args.token, seed, audio)
            print(f"[ok]   {seed.name}  id={voice['id']}  duration_ms={voice['duration_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
