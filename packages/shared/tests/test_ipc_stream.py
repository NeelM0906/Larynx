"""Streaming RPC contract tests.

Spin up a fake worker loop against the same ``WorkerChannel`` the real
workers use, then exercise the ``stream()`` context manager end-to-end.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest
from larynx_shared.ipc import (
    CancelStreamRequest,
    InProcessWorkerClient,
    RequestMessage,
    StreamChunk,
    StreamEnd,
    WorkerChannel,
    WorkerError,
)


class _Req(RequestMessage):
    kind: Literal["fake"] = "fake"
    count: int


class _Chunk(StreamChunk):
    kind: Literal["fake_chunk"] = "fake_chunk"
    i: int


class _End(StreamEnd):
    kind: Literal["fake_end"] = "fake_end"


async def _fake_worker(
    channel: WorkerChannel, cancel_event: asyncio.Event | None = None
) -> None:
    """Pull one request, emit ``count`` chunks, then an end frame.

    If a CancelStreamRequest arrives mid-stream, stop emitting and set the
    ``cancel_event`` (used only by the cancel test to synchronize).
    """
    cancel_seen = asyncio.Event()

    async def _consume_cancels() -> None:
        while True:
            msg = await channel.requests.get()
            if isinstance(msg, CancelStreamRequest):
                cancel_seen.set()
                if cancel_event is not None:
                    cancel_event.set()
                continue
            assert isinstance(msg, _Req)
            for i in range(msg.count):
                if cancel_seen.is_set():
                    break
                await channel.responses.put(_Chunk(request_id=msg.request_id, i=i))
                await asyncio.sleep(0.01)
            if not cancel_seen.is_set():
                await channel.responses.put(_End(request_id=msg.request_id))

    await _consume_cancels()


@pytest.mark.asyncio
async def test_stream_yields_chunks_then_end() -> None:
    channel = WorkerChannel()
    client = InProcessWorkerClient(channel)
    await client.start()
    worker = asyncio.create_task(_fake_worker(channel))
    try:
        req = _Req(count=3)
        got_chunks: list[int] = []
        got_end = False
        async with client.stream(req, chunk_type=_Chunk, end_type=_End) as frames:
            async for frame in frames:
                if isinstance(frame, _Chunk):
                    got_chunks.append(frame.i)
                elif isinstance(frame, _End):
                    got_end = True
        assert got_chunks == [0, 1, 2]
        assert got_end
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await client.stop()


@pytest.mark.asyncio
async def test_stream_cancels_on_early_exit() -> None:
    channel = WorkerChannel()
    client = InProcessWorkerClient(channel)
    await client.start()
    cancel_event = asyncio.Event()
    worker = asyncio.create_task(_fake_worker(channel, cancel_event=cancel_event))
    try:
        req = _Req(count=100)  # way more than we'll consume
        async with client.stream(req, chunk_type=_Chunk, end_type=_End) as frames:
            async for frame in frames:
                if isinstance(frame, _Chunk) and frame.i >= 2:
                    break
        # Worker should see the CancelStreamRequest shortly after we exited.
        await asyncio.wait_for(cancel_event.wait(), timeout=1.0)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await client.stop()


@pytest.mark.asyncio
async def test_stream_raises_on_error_frame() -> None:
    from larynx_shared.ipc.messages import ErrorMessage

    channel = WorkerChannel()
    client = InProcessWorkerClient(channel)
    await client.start()

    async def _err_worker() -> None:
        msg = await channel.requests.get()
        await channel.responses.put(
            ErrorMessage(request_id=msg.request_id, code="boom", message="nope")
        )

    worker = asyncio.create_task(_err_worker())
    try:
        req = _Req(count=0)
        with pytest.raises(WorkerError) as exc_info:
            async with client.stream(req, chunk_type=_Chunk, end_type=_End) as frames:
                async for _ in frames:
                    pass
        assert exc_info.value.code == "boom"
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await client.stop()
