"""
Compute backends package.
"""

from zarr_proxy.backends.base import BackendConfig, ComputeBackend
from zarr_proxy.backends.dask_backend import DaskBackend
from zarr_proxy.backends.local import LocalBackend
from zarr_proxy.backends.registry import get_backend, list_backends, register_backend
from zarr_proxy.backends.workstation import WorkstationBackend

__all__ = [
    "BackendConfig",
    "ComputeBackend",
    "DaskBackend",
    "LocalBackend",
    "WorkstationBackend",
    "get_backend",
    "list_backends",
    "register_backend",
]

