"""End-to-end OpenAI Python SDK round-trip against /v1/audio/speech.

Opt-in — marked ``openai_sdk`` so the default test session skips it
(the default ``addopts`` excludes this marker). Run explicitly:

    uv run pytest packages/gateway/tests/integration/test_openai_sdk_speech.py \\
        -m openai_sdk -q

The SDK test needs a real HTTP server because ``openai.OpenAI`` won't
accept ``ASGITransport``. The ``live_gateway`` fixture (see
``conftest_openai.py``) spins up uvicorn on a random loopback port.
"""

from __future__ import annotations

import pathlib

import pytest

pytest_plugins = ["tests.integration.conftest_openai"]


@pytest.mark.openai_sdk
def test_openai_sdk_speech_roundtrip(
    live_gateway: tuple[str, str],
    seed_voice_alloy: str,
    tmp_path: pathlib.Path,
) -> None:
    """SDK -> /v1/audio/speech -> mp3 bytes."""
    import openai

    base_url, token = live_gateway
    _ = seed_voice_alloy  # ensures the ``alloy`` voice exists

    client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=token)
    resp = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="Hello from the Larynx OpenAI shim.",
    )

    out = tmp_path / "out.mp3"
    resp.stream_to_file(out)

    body = out.read_bytes()
    assert len(body) > 1024, f"mp3 too short: {len(body)} bytes"
    # Standard MP3 sync-word or ID3 tag — either is valid.
    head = body[:4]
    assert head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") or head[:3] == b"ID3", (
        f"not a valid mp3: first 4 bytes = {head!r}"
    )


@pytest.mark.openai_sdk
def test_openai_sdk_speech_unknown_voice_is_404(
    live_gateway: tuple[str, str],
) -> None:
    """Unknown voice names raise openai.NotFoundError via the SDK."""
    import openai

    base_url, token = live_gateway
    client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=token)

    with pytest.raises(openai.NotFoundError):
        client.audio.speech.create(
            model="tts-1",
            voice="definitely-does-not-exist",
            input="hello",
        )


@pytest.mark.openai_sdk
def test_openai_sdk_speech_invalid_speed_is_422(
    live_gateway: tuple[str, str],
    seed_voice_alloy: str,
) -> None:
    """speed outside [0.25, 4.0] returns a 400-family error via the SDK.

    The pydantic validator on ``OpenAISpeechRequest.speed`` rejects this
    before we touch the worker, and the route surfaces an OpenAI-shaped
    400 body. The SDK maps that to ``openai.BadRequestError``.
    """
    import openai

    base_url, token = live_gateway
    _ = seed_voice_alloy

    client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=token)
    with pytest.raises((openai.BadRequestError, openai.APIStatusError)):
        client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input="hello",
            speed=10.0,  # out of range
        )
