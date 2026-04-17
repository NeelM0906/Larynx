"""Fun-ASR worker server loop.

Pulls ``TranscribeRequest`` / ``TranscribeRollingRequest`` off the
gateway->worker queue, runs them through the model manager (which has
already picked Nano vs MLT based on the request's language), and pushes
typed responses back. Mirrors the VoxCPM worker's structure exactly so
supervisord can restart either worker without the other noticing.
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
    TranscribeRequest,
    TranscribeResponse,
    TranscribeRollingRequest,
    TranscribeRollingResponse,
)

from larynx_funasr_worker.audio_utils import pcm_to_float32
from larynx_funasr_worker.language_router import UnsupportedLanguageError, resolve
from larynx_funasr_worker.model_manager import FunASRModelManager

log = structlog.get_logger(__name__)


class WorkerServer:
    def __init__(self, channel: WorkerChannel, manager: FunASRModelManager) -> None:
        self._channel = channel
        self._manager = manager
        self._task: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()
        self._inflight: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._shutdown.clear()
        self._task = asyncio.create_task(self._serve(), name="funasr-worker-loop")
        log.info("funasr_worker.started", mode=self._manager.mode.value)

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
        log.info("funasr_worker.stopped")

    async def _serve(self) -> None:
        while not self._shutdown.is_set():
            try:
                msg = await self._channel.requests.get()
            except asyncio.CancelledError:
                break
            task = asyncio.create_task(self._dispatch(msg), name=f"stt-{msg.request_id[:8]}")
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _dispatch(self, msg: RequestMessage) -> None:
        response = await self._handle(msg)
        await self._channel.responses.put(response)

    async def _handle(self, msg: RequestMessage) -> ResponseMessage | ErrorMessage:
        if isinstance(msg, TranscribeRequest):
            return await self._transcribe(msg)
        if isinstance(msg, TranscribeRollingRequest):
            return await self._transcribe_rolling(msg)
        return ErrorMessage(
            request_id=msg.request_id,
            code="unknown_kind",
            message=f"funasr_worker does not handle kind={msg.kind!r}",
        )

    async def _transcribe(self, req: TranscribeRequest) -> ResponseMessage | ErrorMessage:
        try:
            model, funasr_lang = resolve(req.language)
            audio = pcm_to_float32(req.pcm_s16le, req.sample_rate)
            t0 = time.perf_counter()
            result = await self._manager.backend.transcribe(
                audio=audio,
                sample_rate=req.sample_rate,
                model=model,
                funasr_language=funasr_lang,
                hotwords=req.hotwords,
                itn=req.itn,
                iso_language=req.language,
            )
            infer_ms = int((time.perf_counter() - t0) * 1000)
            log.info(
                "funasr.transcribe",
                request_id=req.request_id,
                model=result.model_used.value,
                language=result.language,
                iso_in=req.language,
                audio_samples=len(audio),
                infer_ms=infer_ms,
                text_len=len(result.text),
                hotwords=len(req.hotwords),
                itn=req.itn,
            )
            return TranscribeResponse(
                request_id=req.request_id,
                text=result.text,
                language=result.language,
                model_used=result.model_used.value,
            )
        except UnsupportedLanguageError as e:
            return ErrorMessage(
                request_id=req.request_id, code="unsupported_language", message=str(e)
            )
        except ValueError as e:
            return ErrorMessage(
                request_id=req.request_id, code="invalid_input", message=str(e)
            )
        except Exception as e:  # noqa: BLE001 — surface any backend failure
            log.exception("funasr.transcribe_failed", request_id=req.request_id)
            return ErrorMessage(
                request_id=req.request_id, code="transcribe_failed", message=str(e)
            )

    async def _transcribe_rolling(
        self, req: TranscribeRollingRequest
    ) -> ResponseMessage | ErrorMessage:
        try:
            model, funasr_lang = resolve(req.language)
            audio = pcm_to_float32(req.pcm_s16le, req.sample_rate)
            t0 = time.perf_counter()
            result = await self._manager.backend.transcribe_rolling(
                audio=audio,
                sample_rate=req.sample_rate,
                model=model,
                funasr_language=funasr_lang,
                hotwords=req.hotwords,
                itn=req.itn,
                prev_text=req.prev_text,
                is_final=req.is_final,
                drop_tail_tokens=req.drop_tail_tokens,
                iso_language=req.language,
            )
            infer_ms = int((time.perf_counter() - t0) * 1000)
            log.info(
                "funasr.transcribe_rolling",
                request_id=req.request_id,
                model=result.model_used.value,
                language=result.language,
                iso_in=req.language,
                is_final=req.is_final,
                prev_text_len=len(req.prev_text),
                infer_ms=infer_ms,
                text_len=len(result.text),
            )
            return TranscribeRollingResponse(
                request_id=req.request_id,
                text=result.text,
                language=result.language,
                model_used=result.model_used.value,
                is_final=req.is_final,
            )
        except UnsupportedLanguageError as e:
            return ErrorMessage(
                request_id=req.request_id, code="unsupported_language", message=str(e)
            )
        except ValueError as e:
            return ErrorMessage(
                request_id=req.request_id, code="invalid_input", message=str(e)
            )
        except Exception as e:  # noqa: BLE001
            log.exception("funasr.transcribe_rolling_failed", request_id=req.request_id)
            return ErrorMessage(
                request_id=req.request_id, code="transcribe_failed", message=str(e)
            )


def install_signal_handlers(server: WorkerServer) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))
