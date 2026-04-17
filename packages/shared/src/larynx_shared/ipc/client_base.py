"""Worker client abstraction.

M1 implementation is a pair of asyncio.Queues bridging a gateway coroutine
and a worker coroutine living in the same Python process. The public surface
(``AbstractWorkerClient.request`` + ``AbstractWorkerClient.stream``) is what
swaps when we move to ZMQ / gRPC — callers don't need to change.

Transport model:
- Request/response (``request``) — one request, one response. Response is
  demultiplexed by ``request_id``.
- Server streaming (``stream``) — one request, many chunk responses ended by
  a ``StreamEnd`` message. Chunks and the terminal frame share the request's
  ``request_id``. The dispatcher recognises streams by the presence of a
  queue in ``_streams`` and routes frames there instead of ``_pending``.

Cancellation is cooperative: if the consumer abandons the async iterator
(disconnect, asyncio.CancelledError, early break), the stream context
manager sends a ``CancelStreamRequest`` back to the worker so the worker
can stop producing chunks.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TypeVar

from larynx_shared.ipc.messages import (
    CancelStreamRequest,
    ErrorMessage,
    RequestMessage,
    ResponseMessage,
    StreamChunk,
    StreamEnd,
)

ResponseT = TypeVar("ResponseT", bound=ResponseMessage)
ChunkT = TypeVar("ChunkT", bound=StreamChunk)
EndT = TypeVar("EndT", bound=StreamEnd)


class WorkerError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class WorkerChannel:
    """A pair of asyncio queues — gateway -> worker, worker -> gateway.

    Created on the gateway side; handed to the worker's `serve()` loop.
    """

    requests: asyncio.Queue[RequestMessage] = field(default_factory=asyncio.Queue)
    responses: asyncio.Queue[ResponseMessage | ErrorMessage] = field(default_factory=asyncio.Queue)


class AbstractWorkerClient(ABC):
    @abstractmethod
    async def request(
        self,
        message: RequestMessage,
        response_type: type[ResponseT],
        timeout: float | None = None,
    ) -> ResponseT: ...


class InProcessWorkerClient(AbstractWorkerClient):
    """Gateway-side client that talks to a worker through a `WorkerChannel`.

    Single-flight routing: we demultiplex responses by request_id, so multiple
    concurrent callers can share one channel. The worker must echo request_id
    on every reply (base classes enforce this).
    """

    def __init__(self, channel: WorkerChannel) -> None:
        self._channel = channel
        self._pending: dict[str, asyncio.Future[ResponseMessage | ErrorMessage]] = {}
        self._streams: dict[str, asyncio.Queue[ResponseMessage | ErrorMessage]] = {}
        self._dispatcher: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._dispatcher is None or self._dispatcher.done():
                self._dispatcher = asyncio.create_task(
                    self._dispatch(), name="worker-response-dispatcher"
                )

    async def stop(self) -> None:
        async with self._lock:
            task = self._dispatcher
            self._dispatcher = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(WorkerError("client_stopped", "worker client stopped"))
        self._pending.clear()
        # Push a synthetic error into each open stream so readers unblock.
        for rid, q in list(self._streams.items()):
            q.put_nowait(
                ErrorMessage(request_id=rid, code="client_stopped", message="worker client stopped")
            )
        self._streams.clear()

    async def _dispatch(self) -> None:
        while True:
            msg = await self._channel.responses.get()
            rid = msg.request_id
            # Streams get priority — a single request_id should never have
            # both a pending future and an open stream queue, but if it
            # does (misuse) we prefer the stream to avoid losing frames.
            q = self._streams.get(rid)
            if q is not None:
                q.put_nowait(msg)
                continue
            fut = self._pending.pop(rid, None)
            if fut is not None and not fut.done():
                fut.set_result(msg)

    async def request(
        self,
        message: RequestMessage,
        response_type: type[ResponseT],
        timeout: float | None = None,
    ) -> ResponseT:
        if self._dispatcher is None:
            raise RuntimeError("InProcessWorkerClient.start() must be called first")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ResponseMessage | ErrorMessage] = loop.create_future()
        self._pending[message.request_id] = fut
        await self._channel.requests.put(message)
        try:
            result = await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError:
            self._pending.pop(message.request_id, None)
            raise WorkerError("timeout", f"worker did not reply within {timeout}s") from None
        if isinstance(result, ErrorMessage):
            raise WorkerError(result.code, result.message)
        if not isinstance(result, response_type):
            raise WorkerError(
                "type_mismatch",
                f"expected {response_type.__name__}, got {type(result).__name__}",
            )
        return result

    @asynccontextmanager
    async def stream(
        self,
        message: RequestMessage,
        *,
        chunk_type: type[ChunkT],
        end_type: type[EndT],
        idle_timeout: float | None = None,
    ) -> AsyncIterator[AsyncIterator[ChunkT | EndT]]:
        """Open a server-streaming RPC. Yields an async iterator of frames.

        Cancellation: if the caller exits the ``async with`` block before the
        stream has ended (e.g. WebSocket disconnect), a ``CancelStreamRequest``
        is sent to the worker so it stops producing. The async iterator is
        closed and any queued frames are dropped.

        ``idle_timeout`` applies between consecutive frames (ttft is not
        gated — the caller times time-to-first-chunk itself).
        """
        if self._dispatcher is None:
            raise RuntimeError("InProcessWorkerClient.start() must be called first")
        rid = message.request_id
        q: asyncio.Queue[ResponseMessage | ErrorMessage] = asyncio.Queue()
        self._streams[rid] = q
        ended = False
        await self._channel.requests.put(message)
        try:

            async def _iter() -> AsyncIterator[ChunkT | EndT]:
                nonlocal ended
                while True:
                    if idle_timeout is not None:
                        msg = await asyncio.wait_for(q.get(), timeout=idle_timeout)
                    else:
                        msg = await q.get()
                    if isinstance(msg, ErrorMessage):
                        ended = True
                        raise WorkerError(msg.code, msg.message)
                    if isinstance(msg, end_type):
                        ended = True
                        yield msg
                        return
                    if isinstance(msg, chunk_type):
                        yield msg
                        continue
                    # Unexpected frame type for this stream — treat as protocol error.
                    ended = True
                    raise WorkerError(
                        "type_mismatch",
                        f"stream expected {chunk_type.__name__}/{end_type.__name__}, "
                        f"got {type(msg).__name__}",
                    )

            yield _iter()
        finally:
            self._streams.pop(rid, None)
            if not ended:
                # Best-effort cancel. Fire-and-forget: worker handles it if
                # the stream is still producing; ignores it otherwise.
                try:
                    self._channel.requests.put_nowait(CancelStreamRequest(target_request_id=rid))
                except asyncio.QueueFull:  # pragma: no cover — unbounded queue
                    pass
