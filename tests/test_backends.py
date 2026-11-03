import dask.array as da
import numpy as np
import pytest

from chunkmirage.core import Box
from zarr_proxy.backends import (
    BackendConfig,
    ComputeBackend,
    DaskBackend,
    LocalBackend,
    get_backend,
    list_backends,
    register_backend,
)


def test_local_backend():
    backend = LocalBackend()
    data = np.arange(64).reshape((4, 4, 4))
    box = Box((1, 1, 1), (3, 3, 3))
    sliced = backend.evaluate_box(data, box)
    assert sliced.shape == (2, 2, 2)
    assert np.array_equal(sliced, data[1:3, 1:3, 1:3])

    def dummy_filter(block, val=10):
        return block + val

    res = backend.dispatch_op(dummy_filter, sliced, val=5)
    assert np.array_equal(res, sliced + 5)


def test_dask_backend():
    backend = DaskBackend(BackendConfig(name="dask", options={"scheduler": "threads"}))
    arr = da.arange(64, chunks=4).reshape((4, 4, 4))
    box = Box((0, 0, 0), (2, 2, 2))
    sub = backend.evaluate_box(arr, box)
    assert isinstance(sub, np.ndarray)
    assert sub.shape == (2, 2, 2)
    assert np.array_equal(sub, arr[0:2, 0:2, 0:2].compute())


def test_backend_registry():
    assert "local" in list_backends()
    assert "dask" in list_backends()
    assert "workstation" in list_backends()

    local_b = get_backend(BackendConfig(name="local"))
    assert isinstance(local_b, LocalBackend)

    dask_b = get_backend(BackendConfig(name="dask"))
    assert isinstance(dask_b, DaskBackend)

    class CustomBackend(ComputeBackend):
        name = "custom"
        def evaluate_box(self, array_like, box):
            return np.ones(box.shape)

    register_backend("custom", CustomBackend)
    assert "custom" in list_backends()
    custom_b = get_backend(BackendConfig(name="custom"))
    assert isinstance(custom_b, CustomBackend)

