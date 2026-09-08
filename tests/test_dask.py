import tempfile
import dask.array as da
import numpy as np
import pytest
import zarr
from fastapi.testclient import TestClient

from zarr_proxy.cache import get_raw_chunk_cache
from zarr_proxy.config import ProxyConfig
from zarr_proxy.filters import apply_filter_dask
from zarr_proxy.processor import (
    build_chunk_dask_graph,
    get_dask_array,
    get_filtered_dask_array,
    process_chunk_request,
)
from zarr_proxy.server import app


@pytest.fixture
def synthetic_zarr_dataset():
    """Create a temporary Zarr v3 test array."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = zarr.storage.LocalStore(tmpdir)
        arr = zarr.create_array(
            store=store,
            name="dask_test.zarr",
            shape=(64, 64, 64),
            chunks=(32, 32, 32),
            dtype="float32",
            zarr_format=3,
        )
        data = np.zeros((64, 64, 64), dtype=np.float32)
        data[32, :, 32] = 50.0  # Synthetic tube
        arr[:] = data

        yield f"file://{tmpdir}/dask_test.zarr"


@pytest.mark.asyncio
async def test_dask_array_lazy_caching(synthetic_zarr_dataset):
    """Verify get_dask_array returns a lazy Dask array and hits RawChunkCache."""
    raw_cache = get_raw_chunk_cache()
    raw_cache.clear()

    darr = await get_dask_array(synthetic_zarr_dataset)
    assert isinstance(darr, da.Array)
    assert darr.shape == (64, 64, 64)
    assert darr.chunks == ((32, 32), (32, 32), (32, 32))

    # Before computing, cache should not have any items from this array
    initial_hits = raw_cache.stats()["hits"]

    # Compute a single chunk [0:32, 0:32, 0:32]
    res1 = darr[0:32, 0:32, 0:32].compute()
    assert res1.shape == (32, 32, 32)
    assert raw_cache.stats()["items_count"] > 0

    # Repeat read should hit RawChunkCache
    res2 = darr[0:32, 0:32, 0:32].compute()
    assert np.array_equal(res1, res2)
    assert raw_cache.stats()["hits"] > initial_hits


@pytest.mark.asyncio
async def test_get_filtered_dask_array_full_map_overlap(synthetic_zarr_dataset):
    """Verify get_filtered_dask_array constructs full map_overlap graph."""
    cfg = ProxyConfig(
        src_url=synthetic_zarr_dataset,
        method="frangi",
        params={"sigmas": [1.0, 2.0]},
        target_dtype="uint8",
    )

    filtered_da = await get_filtered_dask_array(cfg)
    assert isinstance(filtered_da, da.Array)
    assert filtered_da.shape == (64, 64, 64)
    assert filtered_da.dtype == np.uint8

    # Extract block (1, 0, 1) where tube passes through
    block_tube = filtered_da.blocks[1, 0, 1].compute()
    assert block_tube.shape == (32, 32, 32)
    assert block_tube.max() > 0


@pytest.mark.asyncio
async def test_dask_chunk_localized_on_demand_compute(synthetic_zarr_dataset):
    """
    Verify build_chunk_dask_graph builds a localized graph that evaluates
    ONLY the requested chunk on-demand, without computing unneeded chunks.
    """
    cfg = ProxyConfig(
        src_url=synthetic_zarr_dataset,
        method="frangi",
        params={"sigmas": [1.0, 2.0]},
        target_dtype="uint8",
    )

    darr = await get_dask_array(synthetic_zarr_dataset)
    chunk_da = build_chunk_dask_graph(
        config=cfg,
        darr=darr,
        target_coords=(0, 0, 0),
        shape=[64, 64, 64],
        chunk_shape=[32, 32, 32],
        halo=[6, 6, 6],
    )

    assert isinstance(chunk_da, da.Array)
    assert chunk_da.shape == (32, 32, 32)
    assert chunk_da.dtype == np.uint8

    # Evaluate chunk
    res = chunk_da.compute()
    assert res.shape == (32, 32, 32)


@pytest.mark.asyncio
async def test_dask_edge_chunk_embedding():
    """Verify edge chunks extending beyond array bounds are properly embedded in full chunk_shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = zarr.storage.LocalStore(tmpdir)
        # Array shape 40x40x40, chunk size 32x32x32 -> 2 chunks per axis (32 and 8)
        arr = zarr.create_array(
            store=store,
            name="edge_test.zarr",
            shape=(40, 40, 40),
            chunks=(32, 32, 32),
            dtype="float32",
            zarr_format=3,
        )
        arr[:] = 10.0
        src_url = f"file://{tmpdir}/edge_test.zarr"

        cfg = ProxyConfig(
            src_url=src_url,
            method="identity",
            target_dtype="uint8",
        )

        darr = await get_dask_array(src_url)
        # Request corner chunk (1, 1, 1) covering indices [32:40, 32:40, 32:40] (size 8x8x8)
        chunk_da = build_chunk_dask_graph(
            config=cfg,
            darr=darr,
            target_coords=(1, 1, 1),
            shape=[40, 40, 40],
            chunk_shape=[32, 32, 32],
            halo=[0, 0, 0],
        )

        # Result should be embedded in full regular (32, 32, 32) chunk shape
        res = chunk_da.compute()
        assert res.shape == (32, 32, 32)
        assert res.dtype == np.uint8
        # Valid region [0:8, 0:8, 0:8] should be non-zero
        assert np.all(res[0:8, 0:8, 0:8] == 10)
        # Outer padded region [8:32, ...] should be zero
        assert np.all(res[8:32, 8:32, 8:32] == 0)


def test_dask_stats_endpoint():
    """Verify /_stats endpoint includes Dask engine information."""
    client = TestClient(app)
    resp = client.get("/_stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "dask" in data
    assert data["dask"]["engine"] == "dask.array"
    assert data["dask"]["scheduler"] in ("threads", "distributed")


def test_apply_filter_dask_multidimensional():
    """Verify apply_filter_dask works on 4D volumes (e.g. Channel x Z x Y x X)."""
    # 2 channels, 32x32x32 spatial
    data = np.zeros((2, 32, 32, 32), dtype=np.float32)
    data[0, 16, :, 16] = 20.0
    data[1, :, 16, 16] = 30.0

    darr = da.from_array(data, chunks=(1, 32, 32, 32))
    filtered_da = apply_filter_dask(
        darr=darr,
        method="gaussian",
        params={"sigma": 1.0},
        target_dtype="uint8",
    )

    assert filtered_da.shape == (2, 32, 32, 32)
    assert filtered_da.dtype == np.uint8
    computed = filtered_da.compute()
    assert computed.shape == (2, 32, 32, 32)
    assert computed[0, 16, 16, 16] > 0
    assert computed[1, 16, 16, 16] > 0
