"""Minimal WAV container helpers.

We pack 16-bit PCM into a RIFF/WAVE header by hand because we only ever
need mono, one sample rate, and one sample format — pulling in soundfile or
wave-module overhead is not worth it in the hot path.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class WavHeader:
    num_channels: int
    sample_rate: int
    bits_per_sample: int
    num_frames: int

    @property
    def duration_ms(self) -> int:
        return int(1000 * self.num_frames / self.sample_rate)


def pack_wav(pcm_s16le: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap int16 little-endian PCM bytes in a canonical RIFF/WAVE header."""
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_s16le)
    riff_size = 36 + data_size

    header = b"".join(
        [
            b"RIFF",
            struct.pack("<I", riff_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),  # PCM fmt chunk size
            struct.pack("<H", 1),  # PCM format code
            struct.pack("<H", num_channels),
            struct.pack("<I", sample_rate),
            struct.pack("<I", byte_rate),
            struct.pack("<H", block_align),
            struct.pack("<H", bits_per_sample),
            b"data",
            struct.pack("<I", data_size),
        ]
    )
    return header + pcm_s16le


def parse_wav_header(data: bytes) -> WavHeader:
    """Parse a canonical WAV header. Tolerates extra chunks between fmt/data."""
    if len(data) < 44 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")

    pos = 12
    num_channels = sample_rate = bits_per_sample = 0
    data_size = 0
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (chunk_size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body = data[pos + 8 : pos + 8 + chunk_size]
        if chunk_id == b"fmt ":
            (_fmt, num_channels, sample_rate, _byte_rate, _block_align, bits_per_sample) = (
                struct.unpack("<HHIIHH", body[:16])
            )
        elif chunk_id == b"data":
            data_size = chunk_size
            break
        pos += 8 + chunk_size + (chunk_size & 1)  # chunks are word-aligned

    if sample_rate == 0 or bits_per_sample == 0:
        raise ValueError("malformed WAV: missing fmt chunk")
    if data_size == 0:
        raise ValueError("malformed WAV: missing data chunk")

    num_frames = data_size // (num_channels * bits_per_sample // 8)
    return WavHeader(
        num_channels=num_channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        num_frames=num_frames,
    )
