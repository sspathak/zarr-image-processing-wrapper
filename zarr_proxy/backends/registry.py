"""
Registry and discovery for compute backends.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type
from zarr_proxy.backends.base import BackendConfig, ComputeBackend
from zarr_proxy.backends.local import LocalBackend
from zarr_proxy.backends.dask_backend import DaskBackend
from zarr_proxy.backends.workstation import WorkstationBackend

_BACKENDS: Dict[str, Type[ComputeBackend]] = {
    "local": LocalBackend,
    "dask": DaskBackend,
    "workstation": WorkstationBackend,
}

_DEFAULT_BACKEND: Optional[ComputeBackend] = None


def register_backend(name: str, backend_cls: Type[ComputeBackend]) -> None:
    """Register a new compute backend class."""
    _BACKENDS[name.lower()] = backend_cls


def list_backends() -> List[str]:
    """List registered backend names."""
    return sorted(_BACKENDS.keys())


def get_backend(config: Optional[BackendConfig] = None) -> ComputeBackend:
    """Retrieve or instantiate a compute backend instance."""
    if config is None:
        global _DEFAULT_BACKEND
        if _DEFAULT_BACKEND is None:
            _DEFAULT_BACKEND = LocalBackend()
        return _DEFAULT_BACKEND

    name = config.name.lower()
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list_backends()}")
    return _BACKENDS[name](config)

