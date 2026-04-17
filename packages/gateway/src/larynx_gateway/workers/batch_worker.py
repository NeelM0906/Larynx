"""Batch TTS consumer loop.

Two asyncio tasks are spawned by the gateway lifespan. Each task:

1. BRPOPs from the batch queue (1s timeout).
2. Checks the cancel-set; if hit, transitions the item to CANCELLED
   (keeping the state-machine invariant) and skips synthesis.
3. Loads the item row, reconstructs the TTSRequest, resolves
   conditioning against the voice library, calls the shared
   ``voxcpm_client``, writes the artifact to disk, marks the item
   DONE via ``batch_service.record_item_done``.

Real-time starvation is prevented by having only two consumers
(§0 invariant) — new `/v1/tts` calls share the same VoxCPMClient and
interleave at the worker-side IPC queue.

Graceful shutdown: the lifespan sets ``app.state.shutdown_event``
before stopping consumers. Each loop iteration checks the event after
BRPOP timeout, so a SIGTERM takes at most 1s to observe + the current
item's synthesis time (capped by §3.7's 30s shield).
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from dataclasses import dataclass

import structlog
from sqlalchemy import select

from larynx_gateway.db.models import BatchItem, Voice
from larynx_gateway.db.session import get_session
from larynx_gateway.schemas.batch import BatchItemParams
from larynx_gateway.schemas.tts import TTSRequest
from larynx_gateway.services import batch_service, tts_service
from larynx_gateway.services.batch_queue import BatchQueue
from larynx_gateway.services.batch_service import ItemArtifact
from larynx_gateway.services.voice_library import VoiceLibrary
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient

log = structlog.get_logger(__name__)


@dataclass
class BatchWorkerDeps:
    """Everything a consumer needs, gathered at lifespan-start.

    The consumer task keeps a reference to these — the objects
    themselves are singletons owned by app.state.
    """

    queue: BatchQueue
    voxcpm: VoxCPMClient
    data_dir: pathlib.Path
    shutdown_event: asyncio.Event
    # design-preview TTL + latent cache + session factory are pulled
    # fresh per item (VoiceLibrary wraps a per-request session).


async def run_consumer(
    deps: BatchWorkerDeps,
    consumer_idx: int,
    latent_cache,
    design_ttl_s: int,
) -> None:
    """Single consumer task. Multiple of these run concurrently."""
    log.info("batch.consumer_start", idx=consumer_idx)
    while not deps.shutdown_event.is_set():
        try:
            entry = await deps.queue.dequeue()
        except Exception as e:  # noqa: BLE001
            log.error("batch.dequeue_error", idx=consumer_idx, error=repr(e))
            await asyncio.sleep(1.0)
            continue
        if entry is None:
            continue

        if await deps.queue.is_cancelled(entry.batch_job_id):
            # Record cancellation for this specific item and move on.
            async for session in get_session():
                item_row = (
                    await session.execute(
                        select(BatchItem).where(
                            BatchItem.job_id == entry.batch_job_id,
                            BatchItem.item_idx == entry.item_idx,
                        )
                    )
                ).scalar_one_or_none()
                if item_row is not None and item_row.state == "QUEUED":
                    await batch_service.record_item_failed(
                        session,
                        entry.batch_job_id,
                        entry.item_idx,
                        error_code="cancelled",
                        error_detail="job cancelled before pickup",
                    )
                break
            continue

        await _process_one(
            deps=deps,
            entry=entry,
            latent_cache=latent_cache,
            design_ttl_s=design_ttl_s,
        )

    log.info("batch.consumer_stop", idx=consumer_idx)


async def _process_one(
    *,
    deps: BatchWorkerDeps,
    entry,
    latent_cache,
    design_ttl_s: int,
) -> None:
    """Synthesize one item end-to-end, inside a single DB session."""
    async for session in get_session():
        item = (
            await session.execute(
                select(BatchItem).where(
                    BatchItem.job_id == entry.batch_job_id,
                    BatchItem.item_idx == entry.item_idx,
                )
            )
        ).scalar_one_or_none()
        if item is None:
            log.warning(
                "batch.item_missing",
                job_id=entry.batch_job_id,
                idx=entry.item_idx,
            )
            return

        # Already terminal from a prior partial run or an out-of-band
        # cancel — don't re-process.
        if item.state != "QUEUED":
            return

        await batch_service.record_item_started(session, entry.batch_job_id, entry.item_idx)

        params = BatchItemParams.model_validate(json.loads(item.params_json))
        tts_req = TTSRequest(
            text=item.text,
            voice_id=item.voice_id,
            sample_rate=params.sample_rate,
            output_format=params.output_format,
            cfg_value=params.cfg_value,
            temperature=params.temperature,
            prompt_text=params.prompt_text,
        )

        # Validate voice_id refers to something that still exists. We
        # do this here instead of at enqueue time because the voice
        # could be deleted while the item is queued.
        if tts_req.voice_id is not None:
            voice_row = (
                await session.execute(select(Voice).where(Voice.id == tts_req.voice_id))
            ).scalar_one_or_none()
            if voice_row is None:
                await batch_service.record_item_failed(
                    session,
                    entry.batch_job_id,
                    entry.item_idx,
                    error_code="voice_not_found",
                    error_detail=f"voice_id={tts_req.voice_id}",
                )
                return

        library = VoiceLibrary(
            session=session,
            voxcpm=deps.voxcpm,
            cache=latent_cache,
            data_dir=deps.data_dir,
            design_ttl_s=design_ttl_s,
        )
        try:
            conditioning = await tts_service.resolve_conditioning(
                tts_req, library, voxcpm=deps.voxcpm
            )
        except ValueError as e:
            await batch_service.record_item_failed(
                session,
                entry.batch_job_id,
                entry.item_idx,
                error_code="invalid_input",
                error_detail=str(e),
            )
            return
        if conditioning is None:
            await batch_service.record_item_failed(
                session,
                entry.batch_job_id,
                entry.item_idx,
                error_code="voice_not_found",
                error_detail=tts_req.voice_id or "",
            )
            return

        try:
            result = await tts_service.synthesize(tts_req, conditioning, deps.voxcpm)
        except Exception as e:  # noqa: BLE001
            log.error(
                "batch.synthesize_error",
                job_id=entry.batch_job_id,
                idx=entry.item_idx,
                error=repr(e),
            )
            await batch_service.record_item_failed(
                session,
                entry.batch_job_id,
                entry.item_idx,
                error_code="worker_error",
                error_detail=str(e),
            )
            return

        ext = result.content_type.split("/")[-1].lower()
        if ext == "l16":
            ext = "pcm"
        out_path = deps.data_dir / "batch" / entry.batch_job_id / f"{entry.item_idx:05d}.{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(result.audio)

        await batch_service.record_item_done(
            session=session,
            job_id=entry.batch_job_id,
            item_idx=entry.item_idx,
            artifact_path=out_path,
            artifact=ItemArtifact(
                audio_bytes=result.audio,
                output_format=params.output_format,
                sample_rate=result.sample_rate,
                duration_ms=result.duration_ms,
                generation_time_ms=result.generation_time_ms,
            ),
        )
        log.info(
            "batch.item_done",
            job_id=entry.batch_job_id,
            idx=entry.item_idx,
            duration_ms=result.duration_ms,
            gen_ms=result.generation_time_ms,
        )
        return
