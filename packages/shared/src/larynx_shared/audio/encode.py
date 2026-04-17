"""Lossy/lossless audio encoders backed by pyav (FFmpeg).

Consumers pass raw int16 little-endian mono PCM and receive a ready-to-
ship container blob (mp3/aac/opus/flac). WAV + raw PCM bypass this
module — see :mod:`larynx_shared.audio.wav` and the ``pcm16`` branch in
``tts_service._package``.

pyav is wrapped behind a try so a broken install doesn't crash the
gateway on import. Routes that need these codecs should probe via
:func:`pyav_available` and surface a ``501 codec_unavailable`` error
when pyav fails to import.
"""

from __future__ import annotations

import io
from typing import Literal

try:  # pragma: no cover - exercised via pyav_available()
    import av  # type: ignore[import-not-found]
    import numpy as np

    _PYAV_IMPORT_ERROR: BaseException | None = None
except BaseException as e:  # pragma: no cover
    av = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _PYAV_IMPORT_ERROR = e


Format = Literal["mp3", "opus", "aac", "flac"]


# Container + codec + preferred sample rate per output format. Lossy codecs
# prefer 48 kHz (opus is 48k-native; mp3/aac resample cleanly to 48k);
# flac is lossless and stays at the source rate.
_FMT_TABLE: dict[str, tuple[str, str, int | None]] = {
    "mp3": ("mp3", "libmp3lame", 48000),
    "opus": ("ogg", "libopus", 48000),
    "aac": ("adts", "aac", 48000),
    "flac": ("flac", "flac", None),
}


def pyav_available() -> bool:
    """Return True iff ``import av`` succeeded at module load."""
    return av is not None


def encode(
    pcm_s16le: bytes,
    *,
    sample_rate: int,
    fmt: Format,
) -> bytes:
    """Encode mono int16 LE PCM into the requested container/codec.

    Raises ``RuntimeError`` if pyav is unavailable or the requested codec
    isn't built into the installed FFmpeg. Raises ``ValueError`` on
    unknown ``fmt``.
    """
    if not pyav_available():
        raise RuntimeError(
            "pyav not importable — install the 'av' package to enable mp3/opus/aac/flac encoding"
        ) from _PYAV_IMPORT_ERROR
    if fmt not in _FMT_TABLE:
        raise ValueError(f"unsupported fmt {fmt!r}; choose one of {sorted(_FMT_TABLE)}")
    if not pcm_s16le:
        raise ValueError("empty PCM input")

    container_name, codec_name, target_rate_opt = _FMT_TABLE[fmt]
    out_rate = target_rate_opt or sample_rate

    buf = io.BytesIO()
    # av.open returns the Container; we must close it so the trailer is
    # flushed (the 'with' form works on Container in recent pyav).
    container = av.open(buf, mode="w", format=container_name)  # type: ignore[attr-defined]
    try:
        try:
            stream = container.add_stream(codec_name, rate=out_rate)
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"codec {codec_name!r} unavailable in this FFmpeg build") from e
        stream.layout = "mono"  # type: ignore[assignment]

        # Build a single input AudioFrame from the raw int16 bytes. pyav
        # resamples behind the scenes if the codec's sample rate differs
        # from the input rate.
        samples = np.frombuffer(pcm_s16le, dtype=np.int16)
        if samples.size == 0:
            raise ValueError("PCM buffer decoded to zero samples")
        # shape (channels, n_samples) for planar, (1, n_samples) for packed mono
        arr = samples.reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(arr, format="s16", layout="mono")  # type: ignore[attr-defined]
        frame.sample_rate = sample_rate

        # Resample to the codec's preferred rate when it differs. We use a
        # pyav AudioResampler because some codecs (opus) reject non-native
        # input rates outright.
        if out_rate != sample_rate:
            resampler = av.AudioResampler(  # type: ignore[attr-defined]
                format="s16", layout="mono", rate=out_rate
            )
            resampled_frames = resampler.resample(frame)
        else:
            resampled_frames = [frame]

        for rf in resampled_frames:
            for packet in stream.encode(rf):
                container.mux(packet)

        # Flush encoder.
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()

    return buf.getvalue()


__all__ = ["Format", "encode", "pyav_available"]
