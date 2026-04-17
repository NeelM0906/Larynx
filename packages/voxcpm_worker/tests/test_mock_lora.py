"""Mock backend LoRA hot-swap contract.

Mirrors the upstream ``AsyncVoxCPM2ServerPool`` register/unregister/list
surface so tests that exercise the full flow work on a CPU-only box. The
mock's LoRA weights are just a name — when passed via ``lora_name``, the
mock shifts the output pitch by a name-derived amount so tests can prove
that the LoRA parameter actually reached the synthesis path.
"""

from __future__ import annotations

import numpy as np
import pytest
from larynx_voxcpm_worker.model_manager import MockVoxCPMBackend


@pytest.mark.asyncio
async def test_mock_starts_with_no_loras() -> None:
    be = MockVoxCPMBackend()
    assert await be.list_loras() == []


@pytest.mark.asyncio
async def test_mock_load_lora_adds_to_list() -> None:
    be = MockVoxCPMBackend()
    await be.load_lora("voice-alpha", "/tmp/nonexistent-path")
    await be.load_lora("voice-bravo", "/tmp/another")
    names = await be.list_loras()
    assert set(names) == {"voice-alpha", "voice-bravo"}


@pytest.mark.asyncio
async def test_mock_unload_lora_removes_from_list() -> None:
    be = MockVoxCPMBackend()
    await be.load_lora("voice-alpha", "/tmp/nonexistent-path")
    await be.unload_lora("voice-alpha")
    assert await be.list_loras() == []


@pytest.mark.asyncio
async def test_mock_duplicate_load_raises() -> None:
    be = MockVoxCPMBackend()
    await be.load_lora("voice-alpha", "/tmp/path")
    with pytest.raises(ValueError, match="already registered"):
        await be.load_lora("voice-alpha", "/tmp/path")


@pytest.mark.asyncio
async def test_mock_unload_unknown_raises() -> None:
    be = MockVoxCPMBackend()
    with pytest.raises(ValueError, match="not registered"):
        await be.unload_lora("voice-alpha")


@pytest.mark.asyncio
async def test_mock_synthesize_with_unknown_lora_raises() -> None:
    be = MockVoxCPMBackend()
    with pytest.raises(ValueError, match="not registered"):
        await be.synthesize(text="hello", lora_name="missing")


@pytest.mark.asyncio
async def test_mock_synthesize_with_lora_differs_from_base() -> None:
    be = MockVoxCPMBackend()
    await be.load_lora("voice-alpha", "/tmp/path")
    base = await be.synthesize(text="hello world")
    with_lora = await be.synthesize(text="hello world", lora_name="voice-alpha")
    # Same text + different LoRA branch -> audibly different output.
    assert not np.array_equal(base, with_lora)


@pytest.mark.asyncio
async def test_mock_synthesize_distinct_loras_differ() -> None:
    be = MockVoxCPMBackend()
    await be.load_lora("voice-alpha", "/tmp/a")
    await be.load_lora("voice-bravo", "/tmp/b")
    a = await be.synthesize(text="hello world", lora_name="voice-alpha")
    b = await be.synthesize(text="hello world", lora_name="voice-bravo")
    assert not np.array_equal(a, b)


@pytest.mark.asyncio
async def test_mock_synthesize_stream_with_lora() -> None:
    # Streaming path also threads lora_name through. We don't assert
    # on content (the one-shot test above covers determinism) — this
    # test just verifies the kwarg doesn't error out the stream.
    be = MockVoxCPMBackend()
    await be.load_lora("voice-alpha", "/tmp/a")
    chunks = [c async for c in be.synthesize_stream(text="hi", lora_name="voice-alpha")]
    assert len(chunks) > 0
    assert all(c.dtype == np.float32 for c in chunks)
