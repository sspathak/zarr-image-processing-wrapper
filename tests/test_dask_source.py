import dask.array as da
import numpy as np
import pytest

from chunkmirage.core import Box
from zarr_proxy.sources import DaskSource, build_dask_pyramid


def test_dask_source():
    arr = da.zeros((64, 64, 64), chunks=(32, 32, 32), dtype=np.float32)
    source = DaskSource(arr, voxel_size=(2.0, 2.0, 2.0), units=("um", "um", "um"))

    assert source.info.shape == (64, 64, 64)
    assert source.info.chunk_shape == (32, 32, 32)
    assert source.info.voxel_size == (2.0, 2.0, 2.0)
    assert source.info.units == ("um", "um", "um")

    # Read a box
    box = Box((0, 0, 0), (16, 16, 16))
    data = source.read(box)
    assert isinstance(data, np.ndarray)
    assert data.shape == (16, 16, 16)
    assert np.array_equal(data, arr[0:16, 0:16, 0:16].compute())


def test_build_dask_pyramid():
    arr = da.ones((64, 64, 64), chunks=(32, 32, 32), dtype=np.float32) * 5.0
    ms = build_dask_pyramid(arr, levels=3, downsample_factor=2, voxel_size=(1.0, 1.0, 1.0))

    assert len(ms) == 3
    assert ms.levels[0].info.shape == (64, 64, 64)
    assert ms.levels[1].info.shape == (32, 32, 32)
    assert ms.levels[2].info.shape == (16, 16, 16)

    # Check voxel sizes scale with pyramid
    assert ms.levels[0].info.voxel_size == (1.0, 1.0, 1.0)
    assert ms.levels[1].info.voxel_size == (2.0, 2.0, 2.0)
    assert ms.levels[2].info.voxel_size == (4.0, 4.0, 4.0)

    # Coarsened mean should preserve values of constant array
    coarse_box = Box((0, 0, 0), (8, 8, 8))
    coarse_data = ms.levels[1].read(coarse_box)
    assert np.allclose(coarse_data, 5.0)
