"""
Workstation / Remote compute backend stub.
Enables offloading chunk operations to a remote high-memory or GPU workstation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import numpy as np
from chunkmirage.core import Box
from zarr_proxy.backends.base import BackendConfig, ComputeBackend


class WorkstationBackend(ComputeBackend):
    """
    Remote Workstation backend.
    Enables delegating compute tasks to a dedicated workstation over RPC/HTTP/Ray.
    """
    name = "workstation"

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self.endpoint: str = self.config.options.get("endpoint", "http://localhost:9000")

    def evaluate_box(self, array_like: Any, box: Box) -> np.ndarray:
        # Fallback to local evaluation or RPC fetch
        slices = box.slices()
        return np.asarray(array_like[slices])

    def dispatch_op(self, func: Callable[..., np.ndarray], block: np.ndarray, **kwargs: Any) -> np.ndarray:
        # Extensible hook for remote RPC dispatch
        return func(block, **kwargs)

