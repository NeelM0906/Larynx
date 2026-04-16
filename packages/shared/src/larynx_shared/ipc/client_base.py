"""Worker client abstraction.

M1 implementation is a pair of asyncio.Queues bridging a gateway coroutine
and a worker coroutine living in the same Python process. The public surface
(`AbstractWorkerClient.request`) is what swaps when we move to ZMQ / gRPC —
callers don't need to change.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar

from larynx_shared.ipc.messages import ErrorMessage, RequestMessage, ResponseMessage

ResponseT = TypeVar("ResponseT", bound=ResponseMessage)


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

    async def _dispatch(self) -> None:
        while True:
            msg = await self._channel.responses.get()
            fut = self._pending.pop(msg.request_id, None)
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
