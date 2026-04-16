"""Typed gateway-side client for the VoxCPM2 worker.

Wraps the generic ``InProcessWorkerClient`` with methods that match the
operations the gateway actually performs. When the worker moves to its own
process the transport swaps without touching callers.
"""

from __future__ import annotations

from larynx_shared.ipc import (
    InProcessWorkerClient,
    SynthesizeRequest,
    SynthesizeResponse,
    WorkerChannel,
)


class VoxCPMClient:
    def __init__(self, channel: WorkerChannel, timeout_s: float = 60.0) -> None:
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
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> SynthesizeResponse:
        req = SynthesizeRequest(
            text=text,
            sample_rate=sample_rate,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        return await self._rpc.request(req, SynthesizeResponse, timeout=self._timeout_s)
