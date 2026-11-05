"""
Registry and discovery for compute backends.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type, Union
from zarr_proxy.backends.base import BackendConfig, ComputeBackend
from zarr_proxy.backends.dask_backend import DaskBackend
from zarr_proxy.backends.local import LocalBackend
from zarr_proxy.backends.webworker import WebWorkerBackend
from zarr_proxy.backends.workstation import WorkstationBackend

_BACKENDS: Dict[str, Type[ComputeBackend]] = {
    "local": LocalBackend,
    "dask": DaskBackend,
    "webworker": WebWorkerBackend,
    "workstation": WorkstationBackend,
}

_DEFAULT_BACKEND: Optional[ComputeBackend] = None


def register_backend(name: str, backend_cls: Type[ComputeBackend]) -> None:
    """Register a new compute backend class."""
    _BACKENDS[name.lower()] = backend_cls


def list_backends() -> List[str]:
    """List registered backend names."""
    return sorted(_BACKENDS.keys())


def set_default_backend(backend: Optional[Union[ComputeBackend, BackendConfig, str]]) -> ComputeBackend:
    """Set or reset the process-wide default compute backend."""
    global _DEFAULT_BACKEND
    if backend is None:
        _DEFAULT_BACKEND = LocalBackend()
    elif isinstance(backend, ComputeBackend):
        _DEFAULT_BACKEND = backend
    elif isinstance(backend, BackendConfig):
        _DEFAULT_BACKEND = get_backend(backend)
    elif isinstance(backend, str):
        _DEFAULT_BACKEND = get_backend(BackendConfig(name=backend))
    else:
        raise TypeError(f"Unsupported backend specification: {type(backend)}")
    return _DEFAULT_BACKEND


def get_backend(config: Optional[Union[BackendConfig, Dict, str]] = None) -> ComputeBackend:
    """Retrieve or instantiate a compute backend instance."""
    if config is None:
        global _DEFAULT_BACKEND
        if _DEFAULT_BACKEND is None:
            _DEFAULT_BACKEND = LocalBackend()
        return _DEFAULT_BACKEND

    if isinstance(config, str):
        cfg = BackendConfig(name=config)
    elif isinstance(config, dict):
        cfg = BackendConfig.model_validate(config)
    elif isinstance(config, BackendConfig):
        cfg = config
    else:
        raise TypeError(f"Expected BackendConfig, dict, or str, got {type(config)}")

    name = cfg.name.lower()
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list_backends()}")
    return _BACKENDS[name](cfg)
