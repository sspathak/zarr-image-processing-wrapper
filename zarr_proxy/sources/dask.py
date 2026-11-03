"""
Dask and virtual array sources for chunkmirage pipelines.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple
import dask.array as da
import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source
from zarr_proxy.backends.base import ComputeBackend
from zarr_proxy.backends.dask_backend import DaskBackend
from zarr_proxy.backends.registry import get_backend


class DaskSource(Source):
    """
    Exposes a single level of a lazy Dask array to chunkmirage.
    Computes only the requested bounding box on demand.
    """

    def __init__(
        self,
        arr: da.Array,
        voxel_size: Tuple[float, ...] = (1.0, 1.0, 1.0),
        units: Tuple[str, ...] = ("nm", "nm", "nm"),
        axes: Tuple[str, ...] = ("z", "y", "x"),
        name: Optional[str] = None,
        backend: Optional[ComputeBackend] = None,
    ):
        self.arr = arr
        self._name = name or arr.name
        self.backend = backend or DaskBackend()

        ndim = arr.ndim
        if len(voxel_size) != ndim:
            voxel_size = (1.0,) * (ndim - len(voxel_size)) + voxel_size
        if len(units) != ndim:
            units = ("nm",) * ndim
        if len(axes) != ndim:
            axes = ("z", "y", "x")[-ndim:]

        chunk_shape = tuple(c[0] for c in arr.chunks)

        self._info = ArrayInfo(
            shape=arr.shape,
            dtype=arr.dtype,
            chunk_shape=chunk_shape,
            voxel_size=voxel_size,
            units=units,
            axes=axes,
            translation=tuple((v - 1.0) / 2.0 for v in voxel_size),
        )

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return f"dask:{self._name}"

    def read(self, box: Box) -> np.ndarray:
        return self.backend.evaluate_box(self.arr, box)


def build_dask_pyramid(
    arr: da.Array,
    levels: int = 4,
    downsample_factor: int = 2,
    voxel_size: Tuple[float, ...] = (1.0, 1.0, 1.0),
    units: Tuple[str, ...] = ("nm", "nm", "nm"),
    axes: Tuple[str, ...] = ("z", "y", "x"),
    name: str = "dask_pyramid",
    backend: Optional[ComputeBackend] = None,
) -> MultiscaleSource:
    """
    Construct a lazy multiscale OME-Zarr pyramid from a Dask array using da.coarsen.
    Each level is evaluated strictly on-demand.
    """
    sources = [
        DaskSource(
            arr,
            voxel_size=voxel_size,
            units=units,
            axes=axes,
            name=f"{name}_s0",
            backend=backend,
        )
    ]

    curr = arr
    curr_voxel = voxel_size
    chunk_sz = tuple(c[0] for c in arr.chunks)

    for i in range(1, levels):
        coarsen_dict = {d: downsample_factor for d in range(arr.ndim)}
        curr = da.coarsen(np.mean, curr, coarsen_dict).rechunk(chunk_sz)
        curr_voxel = tuple(v * downsample_factor for v in curr_voxel)
        sources.append(
            DaskSource(
                curr,
                voxel_size=curr_voxel,
                units=units,
                axes=axes,
                name=f"{name}_s{i}",
                backend=backend,
            )
        )

    return MultiscaleSource(sources, name=name)


class VirtualArraySource(Source):
    """
    Source wrapping an arbitrary array using a pluggable ComputeBackend.
    """

    def __init__(
        self,
        arr: Any,
        info: ArrayInfo,
        backend: Optional[ComputeBackend] = None,
        name: str = "virtual_array",
    ):
        self.arr = arr
        self._info = info
        self.backend = backend or get_backend()
        self._name = name

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return f"{self.backend.name}:{self._name}"

    def read(self, box: Box) -> np.ndarray:
        return self.backend.evaluate_box(self.arr, box)

