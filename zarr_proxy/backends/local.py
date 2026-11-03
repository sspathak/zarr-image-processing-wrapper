"""
Local in-process compute backend using NumPy / SciPy.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import numpy as np
from chunkmirage.core import Box
from zarr_proxy.backends.base import BackendConfig, ComputeBackend


class LocalBackend(ComputeBackend):
    """
    In-process local compute backend.
    Slices arrays directly and executes kernels within the server process threadpool.
    """
    name = "local"

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)

    def evaluate_box(self, array_like: Any, box: Box) -> np.ndarray:
        slices = box.slices()
        return np.asarray(array_like[slices])

    def dispatch_op(self, func: Callable[..., np.ndarray], block: np.ndarray, **kwargs: Any) -> np.ndarray:
        return func(block, **kwargs)

