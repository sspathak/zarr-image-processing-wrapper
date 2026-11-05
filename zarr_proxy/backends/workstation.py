"""
Workstation / Remote compute backend.

Enables offloading volumetric chunk operations, multi-stage fused pipelines,
and bounding-box reads to a dedicated high-memory, multi-core, or GPU
workstation running the `zarr_proxy` compute server over HTTP/RPC, as well as
generating virtual OME-Zarr URLs served by the workstation for `ngplatform`
and Neuroglancer.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import zlib

import dask.array as da
import httpx
import numpy as np

from chunkmirage import fused
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops import get_op, list_ops, op_from_spec
from chunkmirage.ops.base import Op
from chunkmirage.pipeline import PipelineSpec
import zarr_proxy.ops  # noqa: F401 - ensure domain ops are registered
from zarr_proxy.backends.base import BackendConfig, ComputeBackend
from zarr_proxy.backends.webworker import decompile_webworker_op
from zarr_proxy.resolver import encode_payload

log = logging.getLogger("zarr_proxy.backends.workstation")


class WorkstationError(RuntimeError):
    """Raised when a remote workstation RPC call fails and fallback is disabled."""


@dataclass(frozen=True)
class RemoteArrayRef:
    """
    Reference to a dataset or source URI hosted on a remote workstation.
    Pass to `WorkstationBackend.evaluate_box(ref, box)` or `VirtualArraySource`
    to evaluate subvolumes on the remote workstation.
    """

    source: str
    level: int = 0
    ops: List[Dict[str, Any]] = field(default_factory=list)
    select: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# Binary Array Wire Codec
# ---------------------------------------------------------------------------


def serialize_array(
    arr: np.ndarray,
    compression: str = "zlib",
    clevel: int = 3,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Serialize a NumPy array into little-endian C-contiguous bytes (optionally
    zlib-compressed) plus a metadata dictionary (`dtype`, `shape`, `compression`).
    """
    contig = np.ascontiguousarray(arr)
    if contig.dtype.byteorder not in ("<", "|", "="):
        contig = contig.astype(contig.dtype.newbyteorder("<"), copy=False)
    raw = contig.tobytes()
    comp = compression.lower()
    if comp == "zlib":
        payload = zlib.compress(raw, level=clevel)
    elif comp in ("none", "raw", ""):
        comp = "none"
        payload = raw
    else:
        raise ValueError(f"Unsupported array compression '{compression}'")

    meta = {
        "dtype": contig.dtype.name,
        "shape": list(contig.shape),
        "compression": comp,
    }
    return payload, meta


def deserialize_array(
    payload: bytes,
    dtype: str,
    shape: Sequence[int],
    compression: str = "zlib",
) -> np.ndarray:
    """
    Deserialize raw or zlib-compressed bytes back into a concrete NumPy ndarray.
    """
    comp = compression.lower()
    if comp == "zlib":
        raw = zlib.decompress(payload)
    elif comp in ("none", "raw", ""):
        raw = payload
    else:
        raise ValueError(f"Unsupported array compression '{compression}'")

    dt = np.dtype(dtype).newbyteorder("<")
    arr = np.frombuffer(raw, dtype=dt).reshape(tuple(int(x) for x in shape))
    return arr.astype(np.dtype(dtype), copy=True)


def encode_array_payload(
    arr: np.ndarray,
    compression: str = "zlib",
    clevel: int = 3,
) -> Dict[str, Any]:
    """
    Encode a NumPy array into a JSON-serializable envelope (`data` as base64,
    `dtype`, `shape`, `compression`).
    """
    raw, meta = serialize_array(arr, compression=compression, clevel=clevel)
    return {
        **meta,
        "data": base64.b64encode(raw).decode("ascii"),
    }


def decode_array_payload(envelope: Dict[str, Any]) -> np.ndarray:
    """
    Decode a JSON envelope produced by `encode_array_payload` into a NumPy array.
    """
    raw = base64.b64decode(envelope["data"].encode("ascii"))
    return deserialize_array(
        raw,
        dtype=str(envelope["dtype"]),
        shape=envelope["shape"],
        compression=str(envelope.get("compression", "zlib")),
    )


# ---------------------------------------------------------------------------
# Workstation Server-Side Execution Helpers
# ---------------------------------------------------------------------------


def _resolve_op_object(spec_or_op: Union[Dict[str, Any], Op]) -> Op:
    if isinstance(spec_or_op, Op):
        return spec_or_op
    # Decompile known WebWorker / ngplatform python ops into native C/SciPy ops
    normalized = decompile_webworker_op(dict(spec_or_op))
    return op_from_spec(normalized)


def execute_workstation_ops(
    block: np.ndarray,
    ops: Sequence[Union[Dict[str, Any], Op]],
    *,
    in_box: Optional[Box] = None,
    out_box: Optional[Box] = None,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    pad_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Execute one or more `chunkmirage` / `zarr_proxy` ops on `block` on the workstation.
    If `in_box` and `out_box` are supplied, runs `chunkmirage.fused.plan` and
    `chunkmirage.fused.run` (with edge padding when `pad_shape` is given) to crop
    the halo margin identically to `Pipeline`. Otherwise applies each op sequentially.
    """
    op_objs = [_resolve_op_object(s) for s in ops]
    if not op_objs:
        if in_box is not None and out_box is not None and in_box != out_box:
            rel = out_box.relative_to(in_box)
            return np.asarray(block[rel.slices()])
        return np.asarray(block)

    if in_box is not None and out_box is not None:
        full_shape = tuple(int(x) for x in (pad_shape if pad_shape is not None else in_box.stop))
        if pad_shape is not None:
            block = fused.pad_edge(block, in_box, full_shape)
        ndim = block.ndim
        v_sz = (1.0,) * max(0, ndim - 3) + tuple(float(x) for x in voxel_size)[-min(ndim, 3) :]
        info = ArrayInfo(
            shape=full_shape,
            dtype=block.dtype,
            chunk_shape=tuple(int(x) for x in out_box.shape),
            voxel_size=v_sz,
            units=("nm",) * ndim,
            axes=(("c",) if ndim == 4 else ()) + ("z", "y", "x")[-min(ndim, 3) :],
        )
        out_info, _, halo = fused.plan(info, op_objs)
        return np.asarray(fused.run(op_objs, block, in_box, out_box, out_info, halo))

    curr = block
    for op in op_objs:
        curr = op.apply(curr)
    return np.asarray(curr)


def evaluate_workstation_box(
    source_or_dataset: str,
    box: Box,
    *,
    level: int = 0,
    ops: Optional[Sequence[Dict[str, Any]]] = None,
    select: Optional[Dict[str, int]] = None,
    registry: Any = None,
    cache: Any = None,
) -> np.ndarray:
    """
    Evaluate a bounding box `box` on the workstation from a registered dataset name,
    a `dataset://<name>` reference, an encoded payload, or a storage URI (`s3://`,
    `gs://`, `https://`, `synthetic://`, local path), optionally running `ops`.
    """
    from chunkmirage.pipeline import Pipeline, select_axes
    from chunkmirage.sources import open_source

    target = source_or_dataset.strip()
    if target.startswith("dataset://"):
        target = target[len("dataset://") :].strip("/")

    pipeline = None
    if registry is not None:
        pipeline = registry.resolve(target)

    if pipeline is not None and not ops:
        return np.asarray(pipeline.read(level, box))

    if pipeline is not None and ops:
        # Chain ops on top of the resolved pipeline's source
        chained = Pipeline(pipeline.source, list(ops), cache=cache or getattr(registry, "cache", None))
        return np.asarray(chained.read(level, box))

    src = open_source(source_or_dataset, cache=cache or getattr(registry, "cache", None))
    if select:
        src = select_axes(src, select)

    if ops:
        p = Pipeline(src, list(ops), cache=cache or getattr(registry, "cache", None))
        return np.asarray(p.read(level, box))

    return np.asarray(src[level].read_padded(box, edge=True))


# ---------------------------------------------------------------------------
# Workstation ComputeBackend
# ---------------------------------------------------------------------------


def _callable_to_op_spec(
    func: Union[Callable[..., np.ndarray], Op, Dict[str, Any], str],
    kwargs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Attempt to resolve `func` + `kwargs` into a serializable `chunkmirage` / `zarr_proxy`
    op specification dict (`{"op": name, ...}`). Returns `None` if `func` is an
    unregistered arbitrary Python callable.
    """
    if isinstance(func, Op):
        return func.spec()
    if isinstance(func, dict) and "op" in func:
        return dict(func)
    if isinstance(func, str):
        return {"op": func.lower(), **kwargs}
    if isinstance(func, type) and issubclass(func, Op) and func.name:
        op_obj = func(**kwargs)
        return op_obj.spec()

    # Bound `op.apply` method on an Op instance
    self_obj = getattr(func, "__self__", None)
    if isinstance(self_obj, Op):
        return self_obj.spec()

    fn_name = getattr(func, "__name__", "").lower()
    known = set(list_ops().keys())
    if fn_name in known:
        try:
            cls = get_op(fn_name)
            valid_fields = set(cls.model_fields.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
            return cls(**filtered_kwargs).spec()
        except Exception:
            return None

    return None


class WorkstationBackend(ComputeBackend):
    """
    Remote Workstation compute backend.

    Delegates subvolume reads (`evaluate_box`), single-operator execution
    (`dispatch_op`), and fused multi-stage pipelines (`compute_pipeline`) to a
    remote `zarr_proxy` workstation server over HTTP/RPC with binary zlib
    array serialization, retry logic, and optional local fallback.
    """

    name = "workstation"

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        opts = self.config.options
        default_endpoint = os.environ.get("ZARR_WORKSTATION_URL", "http://localhost:8000")
        self.endpoint: str = str(opts.get("endpoint", default_endpoint)).rstrip("/")
        self.timeout: float = float(opts.get("timeout", 60.0))
        self.retries: int = int(opts.get("retries", 2))
        self.token: Optional[str] = opts.get("token") or os.environ.get("ZARR_WORKSTATION_TOKEN")
        self.compression: str = str(opts.get("compression", "zlib"))
        self.fallback_local: bool = bool(opts.get("fallback_local", True))
        self.remote_slicing: bool = bool(opts.get("remote_slicing", False))
        self._transport: Optional[httpx.BaseTransport] = opts.get("transport")
        self._client: Optional[httpx.Client] = opts.get("client")
        self._owns_client: bool = self._client is None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers: Dict[str, str] = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._client = httpx.Client(
                base_url=self.endpoint,
                timeout=self.timeout,
                headers=headers,
                transport=self._transport,
            )
            self._owns_client = True
        return self._client

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = self._get_client()
        last_err: Optional[Exception] = None
        for attempt in range(max(1, self.retries + 1)):
            try:
                resp = client.post(path, json=payload)
                if resp.status_code >= 400:
                    detail = resp.text
                    try:
                        detail = resp.json().get("error", detail)
                    except Exception:
                        pass
                    raise WorkstationError(
                        f"Workstation {self.endpoint}{path} returned HTTP {resp.status_code}: {detail}"
                    )
                return resp.json()
            except WorkstationError:
                raise
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(0.05 * (attempt + 1))
        raise WorkstationError(
            f"Failed to reach workstation at {self.endpoint}{path}: {last_err}"
        ) from last_err

    def evaluate_box(self, array_like: Any, box: Box) -> np.ndarray:
        """
        Evaluate a demanded bounding box `box`.
        - If `array_like` is a `RemoteArrayRef`, `str` (source URI or dataset name),
          or `dict` with `"source"` / `"dataset"`, requests the subvolume from the
          remote workstation (`POST /_workstation/evaluate_box`).
        - If `array_like` is an in-memory `np.ndarray` or `dask.array.Array`, slices
          it directly (or via the workstation if `remote_slicing=True`).
        """
        if isinstance(array_like, (RemoteArrayRef, str, dict)):
            if isinstance(array_like, RemoteArrayRef):
                req_body: Dict[str, Any] = {
                    "source": array_like.source,
                    "level": array_like.level,
                    "ops": array_like.ops,
                    "select": array_like.select,
                }
            elif isinstance(array_like, str):
                req_body = {"source": array_like, "level": 0}
            else:
                req_body = {
                    "source": array_like.get("source") or array_like.get("dataset", ""),
                    "level": int(array_like.get("level", 0)),
                    "ops": array_like.get("ops") or [],
                    "select": array_like.get("select"),
                }

            req_body.update({
                "start": list(box.start),
                "stop": list(box.stop),
                "compression": self.compression,
            })
            try:
                resp_data = self._post_json("/_workstation/evaluate_box", req_body)
                return decode_array_payload(resp_data["array"])
            except Exception as e:
                if not self.fallback_local:
                    if isinstance(e, WorkstationError):
                        raise
                    raise WorkstationError(str(e)) from e
                log.debug("Workstation evaluate_box fallback to local: %s", e)
                return evaluate_workstation_box(
                    req_body["source"],
                    box,
                    level=req_body.get("level", 0),
                    ops=req_body.get("ops"),
                    select=req_body.get("select"),
                )

        slices = box.slices()
        sub = array_like[slices]
        if isinstance(sub, da.Array):
            return np.asarray(sub.compute())
        return np.asarray(sub)

    def dispatch_op(
        self,
        func: Union[Callable[..., np.ndarray], Op, Dict[str, Any], str],
        block: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Execute an operator kernel on `block` on the remote workstation.
        Resolves `func` (an `Op`, op spec dict, op name, or filter function) into a
        serializable op spec and posts the compressed block to
        `POST /_workstation/dispatch_op`. Falls back to local execution when
        `fallback_local=True` if the workstation is unreachable or if `func` is a
        custom un-registered Python callable.
        """
        op_spec = _callable_to_op_spec(func, kwargs)
        if op_spec is not None:
            try:
                return self.compute_pipeline(block, [op_spec])
            except Exception as e:
                if not self.fallback_local:
                    if isinstance(e, WorkstationError):
                        raise
                    raise WorkstationError(str(e)) from e
                log.debug("Workstation dispatch_op fallback to local: %s", e)
                return execute_workstation_ops(block, [op_spec])

        if not self.fallback_local:
            raise WorkstationError(
                f"Callable {func!r} cannot be serialized as a registered workstation op and fallback_local=False"
            )
        if callable(func):
            return func(block, **kwargs)
        raise TypeError(f"Unsupported func type for dispatch_op: {type(func)}")

    def compute_pipeline(
        self,
        block: np.ndarray,
        ops: Sequence[Union[Dict[str, Any], Op]],
        *,
        in_box: Optional[Box] = None,
        out_box: Optional[Box] = None,
        voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
        pad_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """
        Offload a single op or a multi-stage fused pipeline of `ops` on `block`
        (with optional edge padding and halo cropping from `in_box` to `out_box`)
        to the remote workstation in one RPC call (`POST /_workstation/dispatch_op`).
        """
        serialized_ops = [s.spec() if isinstance(s, Op) else dict(s) for s in ops]
        payload: Dict[str, Any] = {
            "array": encode_array_payload(block, compression=self.compression),
            "ops": serialized_ops,
            "compression": self.compression,
            "voxel_size": [float(v) for v in voxel_size],
        }
        if in_box is not None:
            payload["in_box"] = {"start": list(in_box.start), "stop": list(in_box.stop)}
        if out_box is not None:
            payload["out_box"] = {"start": list(out_box.start), "stop": list(out_box.stop)}
        if pad_shape is not None:
            payload["pad_shape"] = [int(x) for x in pad_shape]

        try:
            resp_data = self._post_json("/_workstation/dispatch_op", payload)
            return decode_array_payload(resp_data["array"])
        except Exception as e:
            if not self.fallback_local:
                if isinstance(e, WorkstationError):
                    raise
                raise WorkstationError(str(e)) from e
            log.debug("Workstation compute_pipeline fallback to local: %s", e)
            return execute_workstation_ops(
                block,
                serialized_ops,
                in_box=in_box,
                out_box=out_box,
                voxel_size=voxel_size,
                pad_shape=pad_shape,
            )

    def build_source_url(
        self,
        spec: Union[PipelineSpec, Dict[str, Any]],
        fmt: str = "zarr3",
        compress: bool = True,
    ) -> str:
        """
        Build a Neuroglancer / `ngplatform` virtual Zarr source URL served by this
        workstation (`zarr3://{endpoint}/{payload}/{fmt}`).
        """
        if isinstance(spec, PipelineSpec):
            data = spec.model_dump(exclude_none=True)
        else:
            data = dict(spec)
        # Decompile any webworker python ops so the workstation executes native ops
        if "ops" in data and isinstance(data["ops"], list):
            data["ops"] = [decompile_webworker_op(o) for o in data["ops"]]
        if "chunk" in data and "chunk_shape" not in data:
            data["chunk_shape"] = [int(c) for c in data.pop("chunk")]

        payload = encode_payload(data, compress=compress)
        scheme = "zarr3"
        if fmt == "precomputed":
            scheme = "precomputed"
        elif fmt == "n5":
            scheme = "n5"
        elif fmt in ("zarr", "zarr2"):
            scheme = "zarr"
        return f"{scheme}://{self.endpoint}/{payload}/{fmt}"

    def register_dataset(
        self,
        name: str,
        spec: Union[PipelineSpec, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Register or update a named pipeline dataset on the remote workstation
        via `POST /api/datasets`.
        """
        spec_dict = spec.model_dump(exclude_none=True) if isinstance(spec, PipelineSpec) else dict(spec)
        if "ops" in spec_dict and isinstance(spec_dict["ops"], list):
            spec_dict["ops"] = [decompile_webworker_op(o) for o in spec_dict["ops"]]
        return self._post_json("/api/datasets", {"name": name, "spec": spec_dict})

    def check_health(self) -> Dict[str, Any]:
        """
        Check if the remote workstation server is reachable (`GET /_health`).
        """
        try:
            resp = self._get_client().get("/_health")
            if resp.status_code == 200:
                return {"online": True, "endpoint": self.endpoint, **resp.json()}
            return {"online": False, "endpoint": self.endpoint, "status_code": resp.status_code}
        except Exception as e:
            return {"online": False, "endpoint": self.endpoint, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """
        Fetch cache and engine telemetry from the remote workstation (`GET /_stats`).
        """
        resp = self._get_client().get("/_stats")
        resp.raise_for_status()
        return resp.json()

    def shutdown(self) -> None:
        if self._client is not None:
            if self._owns_client:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._client = None
