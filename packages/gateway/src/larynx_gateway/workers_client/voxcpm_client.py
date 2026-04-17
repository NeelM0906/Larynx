"""Typed gateway-side client for the VoxCPM2 worker.

Wraps the generic ``InProcessWorkerClient`` with methods that match the
operations the gateway actually performs. When the worker moves to its own
process the transport swaps without touching callers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from larynx_shared.ipc import (
    EncodeReferenceRequest,
    EncodeReferenceResponse,
    InProcessWorkerClient,
    SynthesizeChunkFrame,
    SynthesizeDoneFrame,
    SynthesizeRequest,
    SynthesizeResponse,
    SynthesizeStreamRequest,
    WorkerChannel,
)


class VoxCPMClient:
    def __init__(self, channel: WorkerChannel, timeout_s: float = 300.0) -> None:
        self._rpc = InProcessWorkerClient(channel)
        self._timeout_s = timeout_s

    async def start(self) -> None:
        await self._rpc.start()

    async def stop(self) -> None:
        await self._rpc.stop()

    async def synthesize_text(
        self,
        text: str,
        sample_rate: int,
        *,
        voice_id: str | None = None,
        ref_audio_latents: bytes | None = None,
        prompt_audio_latents: bytes | None = None,
        prompt_text: str = "",
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        max_generate_length: int = 2000,
    ) -> SynthesizeResponse:
        req = SynthesizeRequest(
            text=text,
            sample_rate=sample_rate,
            voice_id=voice_id,
            ref_audio_latents=ref_audio_latents,
            prompt_audio_latents=prompt_audio_latents,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            temperature=temperature,
            max_generate_length=max_generate_length,
        )
        return await self._rpc.request(req, SynthesizeResponse, timeout=self._timeout_s)

    @asynccontextmanager
    async def synthesize_text_stream(
        self,
        text: str,
        sample_rate: int,
        *,
        voice_id: str | None = None,
        ref_audio_latents: bytes | None = None,
        prompt_audio_latents: bytes | None = None,
        prompt_text: str = "",
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        max_generate_length: int = 2000,
        idle_timeout: float | None = 60.0,
    ) -> AsyncIterator[AsyncIterator[SynthesizeChunkFrame | SynthesizeDoneFrame]]:
        """Open a streaming synthesis RPC.

        Yields frames (chunk frames then one done frame). Exiting the ``async
        with`` block early sends a cancel to the worker — callers should
        always use it as a context manager to guarantee cleanup.
        """
        req = SynthesizeStreamRequest(
            text=text,
            sample_rate=sample_rate,
            voice_id=voice_id,
            ref_audio_latents=ref_audio_latents,
            prompt_audio_latents=prompt_audio_latents,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            temperature=temperature,
            max_generate_length=max_generate_length,
        )
        async with self._rpc.stream(
            req,
            chunk_type=SynthesizeChunkFrame,
            end_type=SynthesizeDoneFrame,
            idle_timeout=idle_timeout,
        ) as frames:
            yield frames

    async def encode_reference(
        self, audio: bytes, wav_format: str = "wav"
    ) -> EncodeReferenceResponse:
        req = EncodeReferenceRequest(audio=audio, wav_format=wav_format)
        return await self._rpc.request(req, EncodeReferenceResponse, timeout=self._timeout_s)
