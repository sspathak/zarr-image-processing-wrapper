"""
Compute backends package.
"""

from zarr_proxy.backends.base import BackendConfig, ComputeBackend
from zarr_proxy.backends.dask_backend import DaskBackend
from zarr_proxy.backends.local import LocalBackend
from zarr_proxy.backends.registry import (
    get_backend,
    list_backends,
    register_backend,
    set_default_backend,
)
from zarr_proxy.backends.webworker import (
    DEFAULT_WEBWORKER_CHUNK,
    NATIVE_CHUNKMIRAGE_OPS,
    WEBWORKER_PYTHON_TEMPLATES,
    WebWorkerBackend,
    build_ngplatform_url,
    clean_engine_source,
    compile_op_for_webworker,
    decompile_webworker_op,
    from_ngplatform_pipeline,
    get_own_python_snippets,
    get_pyworker_modules,
    get_webworker_js,
    is_trusted_webworker_python,
    to_ngplatform_pipeline,
)
from zarr_proxy.backends.workstation import (
    RemoteArrayRef,
    WorkstationBackend,
    WorkstationError,
    decode_array_payload,
    deserialize_array,
    encode_array_payload,
    evaluate_workstation_box,
    execute_workstation_ops,
    serialize_array,
)

__all__ = [
    "BackendConfig",
    "ComputeBackend",
    "DEFAULT_WEBWORKER_CHUNK",
    "DaskBackend",
    "LocalBackend",
    "NATIVE_CHUNKMIRAGE_OPS",
    "RemoteArrayRef",
    "WEBWORKER_PYTHON_TEMPLATES",
    "WebWorkerBackend",
    "WorkstationBackend",
    "WorkstationError",
    "build_ngplatform_url",
    "clean_engine_source",
    "compile_op_for_webworker",
    "decode_array_payload",
    "decompile_webworker_op",
    "deserialize_array",
    "encode_array_payload",
    "evaluate_workstation_box",
    "execute_workstation_ops",
    "from_ngplatform_pipeline",
    "get_backend",
    "get_own_python_snippets",
    "get_pyworker_modules",
    "get_webworker_js",
    "is_trusted_webworker_python",
    "list_backends",
    "register_backend",
    "serialize_array",
    "set_default_backend",
    "to_ngplatform_pipeline",
]
