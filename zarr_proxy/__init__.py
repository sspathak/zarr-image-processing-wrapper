"""
zarr_proxy: High-performance HTTP server for dynamic virtual Zarr compute pipelines powered by chunkmirage.
"""

__version__ = "0.2.0"

from zarr_proxy.backends import ComputeBackend, DaskBackend, LocalBackend, get_backend
from zarr_proxy.ops import DynamicRange, Frangi, Identity, Otsu, Passthrough, Sato
from zarr_proxy.resolver import decode_payload, dynamic_pipeline_resolver, encode_payload
from zarr_proxy.server import app, registry, shared_cache
from zarr_proxy.sources import DaskSource, VirtualArraySource, build_dask_pyramid

__all__ = [
    "ComputeBackend",
    "DaskBackend",
    "DaskSource",
    "DynamicRange",
    "Frangi",
    "Identity",
    "LocalBackend",
    "Otsu",
    "Passthrough",
    "Sato",
    "VirtualArraySource",
    "__version__",
    "app",
    "build_dask_pyramid",
    "decode_payload",
    "dynamic_pipeline_resolver",
    "encode_payload",
    "get_backend",
    "registry",
    "shared_cache",
]
