"""
Dask compute backend supporting threaded, multiprocessing, and distributed execution.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import dask.array as da
import numpy as np
from chunkmirage.core import Box
from zarr_proxy.backends.base import BackendConfig, ComputeBackend


class DaskBackend(ComputeBackend):
    """
    Dask-powered compute backend.
    Evaluates lazy dask graphs on-demand using configurable schedulers.
    """
    name = "dask"

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        self.scheduler: str = self.config.options.get("scheduler", "threads")
        self.address: Optional[str] = self.config.options.get("address")
        self._client: Any = None

        if self.address:
            try:
                from distributed import Client
                self._client = Client(self.address)
            except Exception:
                pass

    def evaluate_box(self, array_like: Any, box: Box) -> np.ndarray:
        slices = box.slices()
        sub = array_like[slices]
        if isinstance(sub, da.Array):
            if self._client is not None:
                return np.asarray(self._client.compute(sub).result())
            return np.asarray(sub.compute(scheduler=self.scheduler))
        return np.asarray(sub)

    def dispatch_op(self, func: Callable[..., np.ndarray], block: np.ndarray, **kwargs: Any) -> np.ndarray:
        return func(block, **kwargs)

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

