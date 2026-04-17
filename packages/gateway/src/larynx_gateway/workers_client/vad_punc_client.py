"""Typed gateway-side client for the VAD+Punctuation worker."""

from __future__ import annotations

from larynx_shared.ipc import (
    DetectSegmentsRequest,
    DetectSegmentsResponse,
    InProcessWorkerClient,
    PunctuateRequest,
    PunctuateResponse,
    WorkerChannel,
)


class VadPuncClient:
    def __init__(self, channel: WorkerChannel, timeout_s: float = 30.0) -> None:
        self._rpc = InProcessWorkerClient(channel)
        self._timeout_s = timeout_s

    async def start(self) -> None:
        await self._rpc.start()

    async def stop(self) -> None:
        await self._rpc.stop()

    async def segment(
        self, *, pcm_s16le: bytes, sample_rate: int = 16000
    ) -> DetectSegmentsResponse:
        req = DetectSegmentsRequest(pcm_s16le=pcm_s16le, sample_rate=sample_rate)
        return await self._rpc.request(req, DetectSegmentsResponse, timeout=self._timeout_s)

    async def punctuate(self, *, text: str, language: str | None = None) -> PunctuateResponse:
        req = PunctuateRequest(text=text, language=language)
        return await self._rpc.request(req, PunctuateResponse, timeout=self._timeout_s)
