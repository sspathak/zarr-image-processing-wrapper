import json
import tempfile
import numcodecs
import numpy as np
import pytest
import zarr
from fastapi.testclient import TestClient

from zarr_proxy.config import ProxyConfig, encode_config
from zarr_proxy.server import app


@pytest.fixture
def zarr3_dataset():
    """Create a temporary Zarr v3 test array on the local filesystem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = zarr.storage.LocalStore(tmpdir)
        # Create 3D array: 64x64x64, chunk 32x32x32
        arr = zarr.create_array(
            store=store,
            name="test_dataset.zarr",
            shape=(64, 64, 64),
            chunks=(32, 32, 32),
            dtype="float32",
            zarr_format=3,
        )
        # Create a tube structure through the volume
        data = np.zeros((64, 64, 64), dtype=np.float32)
        data[32, :, 32] = 50.0  # tube along axis 1
        arr[:] = data

        yield f"file://{tmpdir}/test_dataset.zarr"


@pytest.fixture
def client():
    return TestClient(app)


def test_health_and_stats(client):
    r_health = client.get("/_health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_stats = client.get("/_stats")
    assert r_stats.status_code == 200
    data = r_stats.json()
    assert "cache" in data
    assert "datasets" in data


def test_studio_ui_endpoints(client):
    for path, headers, expected in [
        ("/", {"Accept": "text/html"}, "Virtual Zarr Compute Studio"),
        ("/studio", {}, "Virtual Zarr Compute Studio"),
        ("/ui", {}, "chunkmirage"),
    ]:
        resp = client.get(path, headers=headers)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert expected.lower() in resp.text.lower()


def test_encode_utility(client):
    payload = {
        "src_url": "s3://my-bucket/image.zarr",
        "method": "frangi",
        "params": {"sigmas": [1.0, 2.0]},
    }
    resp = client.post("/_encode", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "payload" in body
    assert "neuroglancer_source" in body


def test_metadata_falsification_endpoint(client, zarr3_dataset):
    cfg = ProxyConfig(
        src_url=zarr3_dataset,
        method="frangi",
        params={"sigmas": [1.0, 2.0]},
        target_dtype="uint8",
    )
    payload = encode_config(cfg)

    # Request group zarr.json via chunkmirage zarr3 frontend
    resp_group = client.get(f"/{payload}/zarr3/zarr.json")
    assert resp_group.status_code == 200
    group_meta = resp_group.json()
    assert group_meta["node_type"] == "group"

    # Request array zarr.json for level s0
    resp_arr = client.get(f"/{payload}/zarr3/s0/zarr.json")
    assert resp_arr.status_code == 200
    meta = resp_arr.json()

    # Verify falsified fields
    assert meta["data_type"] == "uint8"
    assert meta["shape"] == [64, 64, 64]
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [32, 32, 32]
    # Check blosc typesize falsified to 1
    blosc = next(c for c in meta["codecs"] if c["name"] == "blosc")
    assert blosc["configuration"]["typesize"] == 1


def test_chunk_filtering_and_caching(client, zarr3_dataset):
    cfg = ProxyConfig(
        src_url=zarr3_dataset,
        method="frangi",
        params={"sigmas": [1.0, 2.0]},
        target_dtype="uint8",
        compression="blosc",
    )
    payload = encode_config(cfg)

    # 1. Fetch chunk (0, 0, 0)
    resp = client.get(f"/{payload}/zarr3/s0/c/0/0/0")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert int(resp.headers["content-length"]) > 0

    chunk_bytes = resp.content
    # Decompress using Blosc and verify shape and dtype
    blosc = numcodecs.Blosc(cname="zstd")
    decompressed = blosc.decode(chunk_bytes)
    arr = np.frombuffer(decompressed, dtype=np.uint8).reshape((32, 32, 32))
    assert arr.shape == (32, 32, 32)
    assert arr.dtype == np.uint8

    # 2. Fetch chunk (1, 0, 1) where tube passes through
    # Tube was at [32, :, 32] -> chunk (1, y, 1) contains coordinates z=32, x=32
    resp_tube = client.get(f"/{payload}/zarr3/s0/c/1/0/1")
    assert resp_tube.status_code == 200
    decomp_tube = blosc.decode(resp_tube.content)
    arr_tube = np.frombuffer(decomp_tube, dtype=np.uint8).reshape((32, 32, 32))
    # Vesselness filter should produce non-zero signal along the center
    assert arr_tube.max() > 0

    # 3. Check stats to confirm cache entries
    r_stats = client.get("/_stats")
    stats = r_stats.json()
    assert stats["cache"]["bytes"] > 0

    # Repeat request for (0, 0, 0) should hit cache
    resp_cached = client.get(f"/{payload}/zarr3/s0/c/0/0/0")
    assert resp_cached.status_code == 200
    assert resp_cached.content == chunk_bytes


def test_invalid_payload_error(client):
    resp = client.get("/not-valid-base64/zarr3/zarr.json")
    assert resp.status_code in (400, 404)


def test_missing_dataset_404(client):
    cfg = ProxyConfig(src_url="file:///nonexistent/path/data.zarr")
    payload = encode_config(cfg)
    resp = client.get(f"/{payload}/zarr3/zarr.json")
    assert resp.status_code in (400, 404, 500)


def test_multiscale_ome_zarr(client):
    """Test full OME-Zarr 0.5 multiscale hierarchy: root group and subscale arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = zarr.storage.LocalStore(tmpdir)
        # 1. Create root group with OME-Zarr 0.5 metadata
        root_group = zarr.create_group(store=store, zarr_format=3)
        root_group.attrs.put({
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "datasets": [
                            {"path": "s0"},
                        ]
                    }
                ]
            }
        })

        # 2. Create subscale array s0
        s0 = zarr.create_array(
            store=store,
            name="s0",
            shape=(32, 32, 32),
            chunks=(32, 32, 32),
            dtype="float32",
            zarr_format=3,
        )
        s0[:] = 100.0

        src_url = f"file://{tmpdir}"
        cfg = ProxyConfig(
            src_url=src_url,
            method="identity",
            target_dtype="uint8",
        )
        payload = encode_config(cfg)

        # Query root group zarr.json via chunkmirage
        r_root = client.get(f"/{payload}/zarr3/zarr.json")
        assert r_root.status_code == 200
        root_meta = r_root.json()
        assert root_meta["node_type"] == "group"

        # Query subscale s0 array zarr.json
        r_s0_meta = client.get(f"/{payload}/zarr3/s0/zarr.json")
        assert r_s0_meta.status_code == 200
        s0_meta = r_s0_meta.json()
        assert s0_meta["node_type"] == "array"
        assert s0_meta["data_type"] == "uint8"

        # Query chunk s0/c/0/0/0
        r_s0_chunk = client.get(f"/{payload}/zarr3/s0/c/0/0/0")
        assert r_s0_chunk.status_code == 200
        assert len(r_s0_chunk.content) > 0
