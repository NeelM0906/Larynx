"""VAD + Punctuation worker server loop.

Handles ``DetectSegmentsRequest`` and ``PunctuateRequest``. Same
structure as the VoxCPM + Fun-ASR workers so a single supervisord
restart policy covers all three. CPU-bound, so we don't fight for GPU
memory.
"""

from __future__ import annotations

import asyncio
import signal
import time

import structlog
from larynx_shared.ipc.client_base import WorkerChannel
from larynx_shared.ipc.messages import (
    DetectSegmentsRequest,
    DetectSegmentsResponse,
    ErrorMessage,
    PunctuateRequest,
    PunctuateResponse,
    RequestMessage,
    ResponseMessage,
    Segment,
)

from larynx_vad_punc_worker.audio_utils import pcm_to_float32
from larynx_vad_punc_worker.model_manager import VadPuncModelManager

log = structlog.get_logger(__name__)


class WorkerServer:
    def __init__(self, channel: WorkerChannel, manager: VadPuncModelManager) -> None:
        self._channel = channel
        self._manager = manager
        self._task: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()
        self._inflight: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._shutdown.clear()
        self._task = asyncio.create_task(self._serve(), name="vad-punc-worker-loop")
        log.info("vad_punc_worker.started", mode=self._manager.mode.value)

    async def stop(self) -> None:
        self._shutdown.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await self._manager.close()
        log.info("vad_punc_worker.stopped")

    async def _serve(self) -> None:
        while not self._shutdown.is_set():
            try:
                msg = await self._channel.requests.get()
            except asyncio.CancelledError:
                break
            task = asyncio.create_task(self._dispatch(msg), name=f"vp-{msg.request_id[:8]}")
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _dispatch(self, msg: RequestMessage) -> None:
        response = await self._handle(msg)
        await self._channel.responses.put(response)

    async def _handle(self, msg: RequestMessage) -> ResponseMessage | ErrorMessage:
        if isinstance(msg, DetectSegmentsRequest):
            return await self._segment(msg)
        if isinstance(msg, PunctuateRequest):
            return await self._punctuate(msg)
        return ErrorMessage(
            request_id=msg.request_id,
            code="unknown_kind",
            message=f"vad_punc_worker does not handle kind={msg.kind!r}",
        )

    async def _segment(self, req: DetectSegmentsRequest) -> ResponseMessage | ErrorMessage:
        try:
            audio = pcm_to_float32(req.pcm_s16le, req.sample_rate)
            t0 = time.perf_counter()
            segments = await self._manager.backend.segment(audio)
            vad_ms = int((time.perf_counter() - t0) * 1000)
            log.info(
                "vad.segment",
                request_id=req.request_id,
                audio_samples=len(audio),
                segments=len(segments),
                vad_ms=vad_ms,
            )
            return DetectSegmentsResponse(
                request_id=req.request_id,
                segments=[
                    Segment(start_ms=s.start_ms, end_ms=s.end_ms, is_speech=s.is_speech)
                    for s in segments
                ],
            )
        except ValueError as e:
            return ErrorMessage(
                request_id=req.request_id, code="invalid_input", message=str(e)
            )
        except Exception as e:  # noqa: BLE001
            log.exception("vad.segment_failed", request_id=req.request_id)
            return ErrorMessage(
                request_id=req.request_id, code="segment_failed", message=str(e)
            )

    async def _punctuate(self, req: PunctuateRequest) -> ResponseMessage | ErrorMessage:
        try:
            t0 = time.perf_counter()
            text, applied = await self._manager.backend.punctuate(req.text, req.language)
            punc_ms = int((time.perf_counter() - t0) * 1000)
            log.info(
                "punc.punctuate",
                request_id=req.request_id,
                language=req.language,
                applied=applied,
                in_len=len(req.text),
                out_len=len(text),
                punc_ms=punc_ms,
            )
            return PunctuateResponse(request_id=req.request_id, text=text, applied=applied)
        except Exception as e:  # noqa: BLE001
            log.exception("punc.punctuate_failed", request_id=req.request_id)
            return ErrorMessage(
                request_id=req.request_id, code="punctuate_failed", message=str(e)
            )


def install_signal_handlers(server: WorkerServer) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))
