"""Worker server loop.

Pulls typed requests off the gateway->worker queue, runs them through the
model manager, pushes typed responses back on the worker->gateway queue.

The backend is fully async (AsyncVoxCPM2ServerPool is native async), so the
loop just `await`s backend methods directly — no thread offload. Output is
resampled to the caller's target_sr at the edge via librosa.

Streaming handlers push one or more chunk frames plus a terminal done frame
for the same ``request_id``. The ``_inflight`` dict keys tasks by request_id
so ``CancelStreamRequest`` can cancel a specific in-flight generation.
"""

from __future__ import annotations

import asyncio
import signal
import time

import structlog
from larynx_shared.ipc.client_base import WorkerChannel
from larynx_shared.ipc.messages import (
    CancelStreamRequest,
    EncodeReferenceRequest,
    EncodeReferenceResponse,
    ErrorMessage,
    ListLorasRequest,
    ListLorasResponse,
    LoadLoraRequest,
    LoadLoraResponse,
    RequestMessage,
    ResponseMessage,
    SynthesizeChunkFrame,
    SynthesizeDoneFrame,
    SynthesizeRequest,
    SynthesizeResponse,
    SynthesizeStreamRequest,
    UnloadLoraRequest,
    UnloadLoraResponse,
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
        # Keyed by request_id so CancelStreamRequest can target a specific
        # in-flight task. Tasks are kept alive here until they complete.
        self._inflight: dict[str, asyncio.Task[None]] = {}

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
        # Cancel any still-running per-request tasks.
        for task in list(self._inflight.values()):
            task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight.values(), return_exceptions=True)
        self._inflight.clear()
        await self._manager.close()
        log.info("voxcpm_worker.stopped")

    async def _serve(self) -> None:
        while not self._shutdown.is_set():
            try:
                msg = await self._channel.requests.get()
            except asyncio.CancelledError:
                break
            if isinstance(msg, CancelStreamRequest):
                task = self._inflight.get(msg.target_request_id)
                if task is not None and not task.done():
                    task.cancel()
                continue
            task = asyncio.create_task(self._dispatch(msg), name=f"req-{msg.request_id[:8]}")
            self._inflight[msg.request_id] = task
            task.add_done_callback(lambda t, rid=msg.request_id: self._inflight.pop(rid, None))

    async def _dispatch(self, msg: RequestMessage) -> None:
        # Streaming requests write many frames themselves; request/response
        # handlers return a single frame we enqueue here.
        if isinstance(msg, SynthesizeStreamRequest):
            await self._synthesize_stream(msg)
            return
        response = await self._handle(msg)
        await self._channel.responses.put(response)

    async def _handle(self, msg: RequestMessage) -> ResponseMessage | ErrorMessage:
        if isinstance(msg, SynthesizeRequest):
            return await self._synthesize(msg)
        if isinstance(msg, EncodeReferenceRequest):
            return await self._encode_reference(msg)
        if isinstance(msg, LoadLoraRequest):
            return await self._load_lora(msg)
        if isinstance(msg, UnloadLoraRequest):
            return await self._unload_lora(msg)
        if isinstance(msg, ListLorasRequest):
            return await self._list_loras(msg)
        return ErrorMessage(
            request_id=msg.request_id,
            code="unknown_kind",
            message=f"worker does not handle kind={msg.kind!r}",
        )

    async def _synthesize(self, req: SynthesizeRequest) -> ResponseMessage | ErrorMessage:
        try:
            info = await self._manager.backend.get_info()
            t0 = time.perf_counter()
            samples = await self._manager.backend.synthesize(
                text=req.text,
                ref_audio_latents=req.ref_audio_latents,
                prompt_audio_latents=req.prompt_audio_latents,
                prompt_text=req.prompt_text,
                cfg_value=req.cfg_value,
                temperature=req.temperature,
                max_generate_length=req.max_generate_length,
                lora_name=req.lora_name,
            )
            gen_ms = int((time.perf_counter() - t0) * 1000)
            samples = self._manager.resample(samples, info.output_sample_rate, req.sample_rate)
            pcm = pcm_from_float(samples)
            duration_ms = int(1000 * len(samples) / req.sample_rate) if len(samples) else 0
            log.info(
                "voxcpm.synthesize",
                request_id=req.request_id,
                chars=len(req.text),
                has_ref_latents=req.ref_audio_latents is not None,
                has_prompt_latents=req.prompt_audio_latents is not None,
                sample_rate=req.sample_rate,
                duration_ms=duration_ms,
                generate_ms=gen_ms,
            )
            return SynthesizeResponse(
                request_id=req.request_id,
                pcm_s16le=pcm,
                sample_rate=req.sample_rate,
                duration_ms=duration_ms,
            )
        except ValueError as e:
            return ErrorMessage(request_id=req.request_id, code="invalid_input", message=str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("voxcpm.synthesize_failed", request_id=req.request_id)
            return ErrorMessage(request_id=req.request_id, code="synthesis_failed", message=str(e))

    async def _synthesize_stream(self, req: SynthesizeStreamRequest) -> None:
        """Stream chunks of audio; terminate with a done frame or an error.

        Cancellation: if the gateway client cancels (e.g. WebSocket
        disconnect), the outer ``_serve`` loop cancels our task via
        ``CancelStreamRequest``. We swallow the CancelledError and let the
        task end silently — the client queue has already been torn down.
        """
        t_start = time.perf_counter()
        ttfb_ms: int | None = None
        chunk_index = 0
        total_samples = 0
        try:
            info = await self._manager.backend.get_info()
            source_sr = info.output_sample_rate
            if not req.text:
                raise ValueError("text must not be empty")
            async for chunk in self._manager.backend.synthesize_stream(
                text=req.text,
                ref_audio_latents=req.ref_audio_latents,
                prompt_audio_latents=req.prompt_audio_latents,
                prompt_text=req.prompt_text,
                cfg_value=req.cfg_value,
                temperature=req.temperature,
                max_generate_length=req.max_generate_length,
                lora_name=req.lora_name,
            ):
                if chunk.size == 0:
                    continue
                resampled = self._manager.resample(chunk, source_sr, req.sample_rate)
                pcm = pcm_from_float(resampled)
                if ttfb_ms is None:
                    ttfb_ms = int((time.perf_counter() - t_start) * 1000)
                await self._channel.responses.put(
                    SynthesizeChunkFrame(
                        request_id=req.request_id,
                        pcm_s16le=pcm,
                        sample_rate=req.sample_rate,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                total_samples += len(resampled)
            duration_ms = int(1000 * total_samples / req.sample_rate) if total_samples else 0
            log.info(
                "voxcpm.synthesize_stream",
                request_id=req.request_id,
                chars=len(req.text),
                chunks=chunk_index,
                duration_ms=duration_ms,
                ttfb_ms=ttfb_ms or 0,
                total_ms=int((time.perf_counter() - t_start) * 1000),
            )
            await self._channel.responses.put(
                SynthesizeDoneFrame(
                    request_id=req.request_id,
                    sample_rate=req.sample_rate,
                    total_duration_ms=duration_ms,
                    chunk_count=chunk_index,
                    ttfb_ms=ttfb_ms or 0,
                )
            )
        except asyncio.CancelledError:
            log.info(
                "voxcpm.synthesize_stream_cancelled",
                request_id=req.request_id,
                chunks_emitted=chunk_index,
            )
            raise
        except ValueError as e:
            await self._channel.responses.put(
                ErrorMessage(request_id=req.request_id, code="invalid_input", message=str(e))
            )
        except Exception as e:  # noqa: BLE001
            log.exception("voxcpm.synthesize_stream_failed", request_id=req.request_id)
            await self._channel.responses.put(
                ErrorMessage(request_id=req.request_id, code="synthesis_failed", message=str(e))
            )

    async def _encode_reference(
        self, req: EncodeReferenceRequest
    ) -> ResponseMessage | ErrorMessage:
        try:
            info = await self._manager.backend.get_info()
            t0 = time.perf_counter()
            latents = await self._manager.backend.encode_reference(req.audio, req.wav_format)
            encode_ms = int((time.perf_counter() - t0) * 1000)
            num_frames = len(latents) // (4 * info.feat_dim)
            log.info(
                "voxcpm.encode_reference",
                request_id=req.request_id,
                audio_bytes=len(req.audio),
                wav_format=req.wav_format,
                latent_bytes=len(latents),
                num_frames=num_frames,
                encode_ms=encode_ms,
            )
            return EncodeReferenceResponse(
                request_id=req.request_id,
                latents=latents,
                feat_dim=info.feat_dim,
                num_frames=num_frames,
                encoder_sample_rate=info.encoder_sample_rate,
            )
        except ValueError as e:
            return ErrorMessage(request_id=req.request_id, code="invalid_input", message=str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("voxcpm.encode_reference_failed", request_id=req.request_id)
            return ErrorMessage(request_id=req.request_id, code="encode_failed", message=str(e))

    # -- LoRA hot-swap handlers --------------------------------------------
    #
    # All three handlers share the same error-to-ErrorMessage shape. The
    # backend raises ValueError for name-collision / not-registered and
    # RuntimeError for LoRA-disabled / backend-not-loaded.

    async def _load_lora(self, req: LoadLoraRequest) -> ResponseMessage | ErrorMessage:
        try:
            await self._manager.backend.load_lora(req.name, req.path)
            log.info("voxcpm.load_lora", request_id=req.request_id, name=req.name, path=req.path)
            return LoadLoraResponse(request_id=req.request_id, name=req.name)
        except ValueError as e:
            return ErrorMessage(request_id=req.request_id, code="lora_invalid", message=str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("voxcpm.load_lora_failed", request_id=req.request_id, name=req.name)
            return ErrorMessage(request_id=req.request_id, code="lora_load_failed", message=str(e))

    async def _unload_lora(self, req: UnloadLoraRequest) -> ResponseMessage | ErrorMessage:
        try:
            await self._manager.backend.unload_lora(req.name)
            log.info("voxcpm.unload_lora", request_id=req.request_id, name=req.name)
            return UnloadLoraResponse(request_id=req.request_id, name=req.name)
        except ValueError as e:
            return ErrorMessage(request_id=req.request_id, code="lora_invalid", message=str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("voxcpm.unload_lora_failed", request_id=req.request_id, name=req.name)
            return ErrorMessage(
                request_id=req.request_id, code="lora_unload_failed", message=str(e)
            )

    async def _list_loras(self, req: ListLorasRequest) -> ResponseMessage | ErrorMessage:
        try:
            names = await self._manager.backend.list_loras()
            return ListLorasResponse(request_id=req.request_id, names=names)
        except Exception as e:  # noqa: BLE001
            log.exception("voxcpm.list_loras_failed", request_id=req.request_id)
            return ErrorMessage(request_id=req.request_id, code="lora_list_failed", message=str(e))


def install_signal_handlers(server: WorkerServer) -> None:
    """Graceful shutdown on SIGTERM/SIGINT when the worker runs standalone."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))
