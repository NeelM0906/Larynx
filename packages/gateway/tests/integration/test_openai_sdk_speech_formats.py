"""Format matrix for /v1/audio/speech via the OpenAI SDK.

Parametrised over ``mp3 | wav | flac | opus | aac``. Each case asserts
the response header bytes match the codec. ``aac`` auto-skips when the
pyav install lacks an aac encoder (e.g. FFmpeg built without
libfdk/native-aac).

Opt-in via ``-m openai_sdk`` — excluded from the default session.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["tests.integration.conftest_openai"]


# Map response_format → a predicate that validates the first bytes of
# the returned container. These are deliberately permissive; we only
# want to assert "this is actually <format>", not bit-exact re-encoding.
def _is_mp3(body: bytes) -> bool:
    return body[:3] == b"ID3" or body[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")


def _is_wav(body: bytes) -> bool:
    return body[:4] == b"RIFF" and body[8:12] == b"WAVE"


def _is_flac(body: bytes) -> bool:
    return body[:4] == b"fLaC"


def _is_opus(body: bytes) -> bool:
    # Opus lives inside an Ogg container; the file starts with 'OggS'.
    return body[:4] == b"OggS"


def _is_aac(body: bytes) -> bool:
    # ADTS frame sync: 12-bit 0xFFF, next nibble high-bit is MPEG version
    # (0 or 1). Matches the 0xFFF0/0xFFF1/0xFFF8/0xFFF9 family.
    if len(body) < 2:
        return False
    return body[0] == 0xFF and (body[1] & 0xF0) == 0xF0


_FORMAT_CASES = [
    ("mp3", _is_mp3),
    ("wav", _is_wav),
    ("flac", _is_flac),
    ("opus", _is_opus),
    ("aac", _is_aac),
]


def _aac_encoder_available() -> bool:
    """Feature-detect whether the installed pyav build exposes an aac encoder."""
    try:
        import av  # type: ignore[import-not-found]

        av.codec.Codec("aac", "w")  # type: ignore[attr-defined]
        return True
    except Exception:  # pragma: no cover - depends on build
        return False


@pytest.mark.openai_sdk
@pytest.mark.parametrize("fmt,validator", _FORMAT_CASES, ids=[f for f, _ in _FORMAT_CASES])
def test_openai_sdk_speech_format_matrix(
    live_gateway: tuple[str, str],
    seed_voice_alloy: str,
    fmt: str,
    validator,
) -> None:
    import openai

    if fmt == "aac" and not _aac_encoder_available():
        pytest.skip("pyav build lacks an aac encoder")

    base_url, token = live_gateway
    _ = seed_voice_alloy

    client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=token)
    resp = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="Format matrix smoke test.",
        response_format=fmt,
    )

    # `.read()` returns the full response body; supports all formats.
    body = resp.read()
    assert len(body) > 256, f"{fmt}: body suspiciously small ({len(body)} bytes)"
    assert validator(body), f"{fmt}: first 16 bytes do not match expected header ({body[:16]!r})"


@pytest.mark.openai_sdk
def test_openai_sdk_speech_pcm_format(
    live_gateway: tuple[str, str],
    seed_voice_alloy: str,
) -> None:
    """``response_format='pcm'`` returns raw s16le PCM with no container."""
    import openai

    base_url, token = live_gateway
    _ = seed_voice_alloy

    client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=token)
    resp = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input="PCM round trip.",
        response_format="pcm",
    )
    body = resp.read()
    # Raw PCM — no magic header to check, but length must be an even
    # multiple of 2 bytes (s16) and non-trivially long.
    assert len(body) > 1024
    assert len(body) % 2 == 0
    # Sanity: headers should advertise audio/L16
    assert resp.response.headers.get("content-type", "").startswith("audio/L16")
