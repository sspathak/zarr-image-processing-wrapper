"""
Base abstractions for pluggable compute backends.
Enables switching between Local (NumPy/SciPy), Dask (threads/distributed),
and future backends (Workstation RPC, Cubed, Numba, CuPy/GPU).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
import numpy as np
from pydantic import BaseModel, Field
from chunkmirage.core import Box


class BackendConfig(BaseModel):
    """
    Configuration specification for a compute backend.
    """
    name: str = Field(default="local", description="Backend identifier (e.g. local, dask, workstation, numba).")
    options: Dict[str, Any] = Field(default_factory=dict, description="Backend-specific runtime options.")


class ComputeBackend(ABC):
    """
    Abstract interface for compute backends.
    Manages resource execution for subvolume slicing and operator dispatch.
    """
    name: str = "base"

    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig(name=self.name)

    @abstractmethod
    def evaluate_box(self, array_like: Any, box: Box) -> np.ndarray:
        """
        Evaluate a demanded bounding box from an underlying array representation.
        Must return a concrete in-memory NumPy ndarray.
        """
        ...

    def dispatch_op(self, func: Callable[..., np.ndarray], block: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Execute an operator kernel on a padded block.
        Default implementation executes in-process synchronously.
        """
        return func(block, **kwargs)

    def shutdown(self) -> None:
        """
        Cleanup any active connection pools, cluster clients, or worker threads.
        """
        pass

