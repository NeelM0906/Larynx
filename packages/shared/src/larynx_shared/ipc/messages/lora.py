"""LoRA hot-swap IPC types (see ORCHESTRATION-M7.md §3).

nano-vllm-voxcpm's ``AsyncVoxCPM2ServerPool`` exposes
register_lora / unregister_lora / list_loras; we mirror that into the
IPC protocol so the gateway can hot-load a fine-tuned LoRA after the
training_worker writes its weights to disk. Per-request selection lives
on ``SynthesizeRequest.lora_name`` in ``messages.tts``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from larynx_shared.ipc.messages.base import RequestMessage, ResponseMessage


class LoadLoraRequest(RequestMessage):
    kind: Literal["load_lora"] = "load_lora"
    name: str
    path: str  # directory holding lora_weights.safetensors + lora_config.json


class LoadLoraResponse(ResponseMessage):
    kind: Literal["load_lora"] = "load_lora"
    name: str


class UnloadLoraRequest(RequestMessage):
    kind: Literal["unload_lora"] = "unload_lora"
    name: str


class UnloadLoraResponse(ResponseMessage):
    kind: Literal["unload_lora"] = "unload_lora"
    name: str


class ListLorasRequest(RequestMessage):
    kind: Literal["list_loras"] = "list_loras"


class ListLorasResponse(ResponseMessage):
    kind: Literal["list_loras"] = "list_loras"
    names: list[str] = Field(default_factory=list)


__all__ = [
    "ListLorasRequest",
    "ListLorasResponse",
    "LoadLoraRequest",
    "LoadLoraResponse",
    "UnloadLoraRequest",
    "UnloadLoraResponse",
]
