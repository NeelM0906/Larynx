#!/usr/bin/env python
"""End-to-end smoke test for Larynx M1.

What it does:
  1. Loads env from .env (or the process environment).
  2. Launches the gateway as a subprocess on a free port.
  3. Polls /health until it returns 200 (or times out).
  4. POSTs /v1/tts with the bearer token.
  5. Saves the WAV, prints its header + first-byte sanity check.
  6. Kills the gateway.

Exit 0 => M1 exit criteria satisfied on this machine.
"""

from __future__ import annotations

import os
import pathlib
import socket
import struct
import subprocess
import sys
import time
from urllib import request as urlrequest

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    # pydantic-settings reads .env too; python-dotenv is just convenience.
    pass


SMOKE_TEXT = "Larynx M1 smoke test: if you can hear this, the scaffold works."
OUT_PATH = pathlib.Path(os.environ.get("LARYNX_SMOKE_OUT", "/tmp/larynx_smoke.wav"))
HEALTH_TIMEOUT_S = float(os.environ.get("LARYNX_SMOKE_HEALTH_TIMEOUT", "60"))
TTS_TIMEOUT_S = float(os.environ.get("LARYNX_SMOKE_TTS_TIMEOUT", "120"))


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_health(url: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            with urlrequest.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.3)
    raise TimeoutError(f"gateway did not respond on {url} within deadline")


def main() -> int:
    token = os.environ.get("LARYNX_API_TOKEN", "change-me-please")
    port = pick_free_port()

    env = os.environ.copy()
    env.setdefault("LARYNX_TTS_MODE", "mock")
    env["LARYNX_PORT"] = str(port)
    env["LARYNX_HOST"] = "127.0.0.1"
    env["LARYNX_API_TOKEN"] = token

    print(f"[smoke] starting gateway on port {port} (mode={env['LARYNX_TTS_MODE']})")
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "larynx_gateway.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
    )

    try:
        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        wait_for_health(f"http://127.0.0.1:{port}/health", deadline)
        print("[smoke] /health OK")

        body = (
            b'{"text":"'
            + SMOKE_TEXT.encode("utf-8")
            + b'","sample_rate":24000,"output_format":"wav"}'
        )
        req = urlrequest.Request(
            f"http://127.0.0.1:{port}/v1/tts",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=TTS_TIMEOUT_S) as resp:
            if resp.status != 200:
                print(f"[smoke] /v1/tts returned {resp.status}: {resp.read()!r}")
                return 2
            audio = resp.read()
            gen_ms = resp.headers.get("X-Generation-Time-Ms", "?")
            dur_ms = resp.headers.get("X-Audio-Duration-Ms", "?")
            sr = resp.headers.get("X-Sample-Rate", "?")

        if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
            print("[smoke] response is not a RIFF/WAVE file")
            return 3

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_bytes(audio)

        # Peek at the fmt chunk for a friendlier print.
        (sample_rate,) = struct.unpack("<I", audio[24:28])
        (bits_per_sample,) = struct.unpack("<H", audio[34:36])
        print(
            f"[smoke] OK — wrote {len(audio):,} bytes to {OUT_PATH}\n"
            f"        sample_rate={sample_rate} bits={bits_per_sample} "
            f"duration_ms={dur_ms} generation_ms={gen_ms} hdr_sr={sr}"
        )
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
