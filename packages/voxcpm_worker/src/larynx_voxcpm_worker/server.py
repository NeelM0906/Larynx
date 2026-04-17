"""Worker server loop.

Pulls typed requests off the gateway->worker queue, runs them through the
model manager, pushes typed responses back on the worker->gateway queue.

Model inference is CPU/GPU-bound and release-blocking, so we run it in the
default thread pool via ``asyncio.to_thread`` to keep the event loop
responsive (and so the Queue drain order doesn't matter under load).
"""

from __future__ import annotations

import asyncio
import signal
import time

import structlog
from larynx_shared.ipc.client_base import WorkerChannel
from larynx_shared.ipc.messages import (
    ErrorMessage,
    RequestMessage,
    ResponseMessage,
    SynthesizeRequest,
    SynthesizeResponse,
)

from larynx_voxcpm_worker.audio_utils import pcm_from_float
from larynx_voxcpm_worker.model_manager import VoxCPMModelManager

log = structlog.get_logger(__name__)


class WorkerServer:
    def __init__(self, channel: WorkerChannel, manager: VoxCPMModelManager) -> None:
        self._channel = channel
        self._manager = manager
        self._task: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._shutdown.clear()
        self._task = asyncio.create_task(self._serve(), name="voxcpm-worker-loop")
        log.info("voxcpm_worker.started", mode=self._manager.mode.value)

    async def stop(self) -> None:
        self._shutdown.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await asyncio.to_thread(self._manager.close)
        log.info("voxcpm_worker.stopped")

    async def _serve(self) -> None:
        while not self._shutdown.is_set():
            try:
                msg = await self._channel.requests.get()
            except asyncio.CancelledError:
                break
            response = await self._handle(msg)
            await self._channel.responses.put(response)

    async def _handle(self, msg: RequestMessage) -> ResponseMessage | ErrorMessage:
        if isinstance(msg, SynthesizeRequest):
            return await self._synthesize(msg)
        return ErrorMessage(
            request_id=msg.request_id,
            code="unknown_kind",
            message=f"worker does not handle kind={msg.kind!r}",
        )

    async def _synthesize(self, req: SynthesizeRequest) -> ResponseMessage | ErrorMessage:
        try:
            t0 = time.perf_counter()
            samples = await asyncio.to_thread(
                self._manager.backend.synthesize,
                req.text,
                req.sample_rate,
                req.cfg_value,
            )
            pcm = pcm_from_float(samples)
            duration_ms = int(1000 * len(samples) / req.sample_rate)
            log.info(
                "voxcpm.synthesize",
                request_id=req.request_id,
                chars=len(req.text),
                sample_rate=req.sample_rate,
                duration_ms=duration_ms,
                wall_ms=int((time.perf_counter() - t0) * 1000),
            )
            return SynthesizeResponse(
                request_id=req.request_id,
                pcm_s16le=pcm,
                sample_rate=req.sample_rate,
                duration_ms=duration_ms,
            )
        except ValueError as e:
            return ErrorMessage(request_id=req.request_id, code="invalid_input", message=str(e))
        except Exception as e:  # noqa: BLE001 — surface *any* backend failure
            log.exception("voxcpm.synthesize_failed", request_id=req.request_id)
            return ErrorMessage(request_id=req.request_id, code="synthesis_failed", message=str(e))


def install_signal_handlers(server: WorkerServer) -> None:
    """Graceful shutdown on SIGTERM/SIGINT when the worker runs standalone."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))
