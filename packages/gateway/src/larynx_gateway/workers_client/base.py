"""Re-export of the shared worker-client base for gateway-internal imports.

Gives the gateway a stable import path even if the shared package's internal
layout shifts later.
"""

from larynx_shared.ipc import AbstractWorkerClient, InProcessWorkerClient, WorkerChannel

__all__ = ["AbstractWorkerClient", "InProcessWorkerClient", "WorkerChannel"]
