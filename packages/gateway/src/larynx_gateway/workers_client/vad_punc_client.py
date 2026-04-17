"""Typed gateway-side client for the VAD+Punctuation worker."""

from __future__ import annotations

from larynx_shared.ipc import (
    DetectSegmentsRequest,
    DetectSegmentsResponse,
    InProcessWorkerClient,
    PunctuateRequest,
    PunctuateResponse,
    VadStreamCloseRequest,
    VadStreamCloseResponse,
    VadStreamFeedRequest,
    VadStreamFeedResponse,
    VadStreamOpenRequest,
    VadStreamOpenResponse,
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

    # -- streaming VAD session --------------------------------------------

    async def vad_stream_open(
        self,
        *,
        session_id: str,
        sample_rate: int = 16000,
        speech_end_silence_ms: int = 300,
    ) -> VadStreamOpenResponse:
        req = VadStreamOpenRequest(
            session_id=session_id,
            sample_rate=sample_rate,
            speech_end_silence_ms=speech_end_silence_ms,
        )
        return await self._rpc.request(req, VadStreamOpenResponse, timeout=self._timeout_s)

    async def vad_stream_feed(
        self,
        *,
        session_id: str,
        pcm_s16le: bytes,
        is_final: bool = False,
    ) -> VadStreamFeedResponse:
        req = VadStreamFeedRequest(
            session_id=session_id, pcm_s16le=pcm_s16le, is_final=is_final
        )
        return await self._rpc.request(req, VadStreamFeedResponse, timeout=self._timeout_s)

    async def vad_stream_close(self, *, session_id: str) -> VadStreamCloseResponse:
        req = VadStreamCloseRequest(session_id=session_id)
        return await self._rpc.request(req, VadStreamCloseResponse, timeout=self._timeout_s)
