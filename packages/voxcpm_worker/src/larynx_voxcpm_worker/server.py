"""Worker server loop.

Pulls typed requests off the gateway->worker queue, runs them through the
model manager, pushes typed responses back on the worker->gateway queue.

The backend is fully async (AsyncVoxCPM2ServerPool is native async), so the
loop just `await`s backend methods directly — no thread offload. Output is
resampled to the caller's target_sr at the edge via librosa.
"""

from __future__ import annotations

import asyncio
import signal
import time

import structlog
from larynx_shared.ipc.client_base import WorkerChannel
from larynx_shared.ipc.messages import (
    EncodeReferenceRequest,
    EncodeReferenceResponse,
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
        await self._manager.close()
        log.info("voxcpm_worker.stopped")

    async def _serve(self) -> None:
        while not self._shutdown.is_set():
            try:
                msg = await self._channel.requests.get()
            except asyncio.CancelledError:
                break
            # Fan out one task per request so concurrent callers don't block
            # each other at the worker mouth. AsyncVoxCPM2ServerPool internally
            # batches and serialises as needed.
            asyncio.create_task(self._dispatch(msg), name=f"req-{msg.request_id[:8]}")

    async def _dispatch(self, msg: RequestMessage) -> None:
        response = await self._handle(msg)
        await self._channel.responses.put(response)

    async def _handle(self, msg: RequestMessage) -> ResponseMessage | ErrorMessage:
        if isinstance(msg, SynthesizeRequest):
            return await self._synthesize(msg)
        if isinstance(msg, EncodeReferenceRequest):
            return await self._encode_reference(msg)
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
            )
            gen_ms = int((time.perf_counter() - t0) * 1000)
            # Resample to the caller's requested output SR.
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
        except Exception as e:  # noqa: BLE001 — surface *any* backend failure
            log.exception("voxcpm.synthesize_failed", request_id=req.request_id)
            return ErrorMessage(request_id=req.request_id, code="synthesis_failed", message=str(e))

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


def install_signal_handlers(server: WorkerServer) -> None:
    """Graceful shutdown on SIGTERM/SIGINT when the worker runs standalone."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))
