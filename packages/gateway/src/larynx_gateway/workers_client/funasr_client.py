"""Typed gateway-side client for the Fun-ASR worker.

Mirrors ``VoxCPMClient``: wraps ``InProcessWorkerClient`` with one method
per request kind the gateway actually issues. Swap-in for ZMQ/gRPC when
the worker moves to its own process stays a one-file change.
"""

from __future__ import annotations

from larynx_shared.ipc import (
    InProcessWorkerClient,
    TranscribeRequest,
    TranscribeResponse,
    TranscribeRollingRequest,
    TranscribeRollingResponse,
    WorkerChannel,
)


class FunASRClient:
    def __init__(self, channel: WorkerChannel, timeout_s: float = 120.0) -> None:
        self._rpc = InProcessWorkerClient(channel)
        self._timeout_s = timeout_s

    async def start(self) -> None:
        await self._rpc.start()

    async def stop(self) -> None:
        await self._rpc.stop()

    async def transcribe(
        self,
        *,
        pcm_s16le: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
        hotwords: list[str] | None = None,
        itn: bool = True,
    ) -> TranscribeResponse:
        req = TranscribeRequest(
            pcm_s16le=pcm_s16le,
            sample_rate=sample_rate,
            language=language,
            hotwords=list(hotwords or []),
            itn=itn,
        )
        return await self._rpc.request(req, TranscribeResponse, timeout=self._timeout_s)

    async def transcribe_rolling(
        self,
        *,
        pcm_s16le: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
        hotwords: list[str] | None = None,
        itn: bool = True,
        prev_text: str = "",
        is_final: bool = False,
        drop_tail_tokens: int = 5,
    ) -> TranscribeRollingResponse:
        req = TranscribeRollingRequest(
            pcm_s16le=pcm_s16le,
            sample_rate=sample_rate,
            language=language,
            hotwords=list(hotwords or []),
            itn=itn,
            prev_text=prev_text,
            is_final=is_final,
            drop_tail_tokens=drop_tail_tokens,
        )
        return await self._rpc.request(req, TranscribeRollingResponse, timeout=self._timeout_s)
