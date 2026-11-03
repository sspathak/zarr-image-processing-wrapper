import numpy as np
import pytest
from starlette.testclient import TestClient

from chunkmirage import Pipeline
from chunkmirage.core import ArrayInfo
from chunkmirage.sources.base import MultiscaleSource, Source
from zarr_proxy.resolver import encode_payload
from zarr_proxy.server import app, registry


class DummySource(Source):
    def __init__(self, name="dummy"):
        self._name = name
        self._info = ArrayInfo(
            shape=(32, 32, 32),
            dtype=np.dtype("float32"),
            chunk_shape=(16, 16, 16),
            voxel_size=(1.0, 1.0, 1.0),
            units=("nm", "nm", "nm"),
            axes=("z", "y", "x"),
        )

    @property
    def info(self):
        return self._info

    def cache_key(self):
        return f"dummy:{self._name}"

    def read(self, box):
        # Return synthetic tube
        vol = np.zeros(box.shape, dtype=np.float32)
        vol[box.shape[0] // 2, :, box.shape[2] // 2] = 100.0
        return vol


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


def test_ui_endpoints(client):
    for path in ["/ui", "/configure"]:
        resp = client.get(path)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


def test_encode_endpoint(client):
    payload_body = {
        "source": "s3://bucket/test.zarr",
        "ops": [{"op": "frangi", "sigmas": [1.0, 2.0]}],
    }
    resp = client.post("/_encode", json=payload_body)
    assert resp.status_code == 200
    data = resp.json()
    assert "payload" in data
    assert "sample_url" in data
    assert "neuroglancer_source" in data


def test_dynamic_zarr3_serving(client):
    # Register a named pipeline using our custom Frangi op
    src = MultiscaleSource([DummySource("raw_dummy")])
    pipeline = Pipeline(src, [{"op": "frangi", "sigmas": [1.0, 2.0]}], cache=registry.cache)
    registry.add("test_pipeline", pipeline)

    # 1. Fetch Zarr v3 group metadata
    r_meta = client.get("/test_pipeline/zarr3/zarr.json")
    assert r_meta.status_code == 200
    meta = r_meta.json()
    assert meta["zarr_format"] == 3
    assert meta["node_type"] == "group"

    # 2. Fetch array metadata for level 0
    r_arr_meta = client.get("/test_pipeline/zarr3/s0/zarr.json")
    assert r_arr_meta.status_code == 200
    arr_meta = r_arr_meta.json()
    assert arr_meta["zarr_format"] == 3
    assert arr_meta["node_type"] == "array"
    assert arr_meta["shape"] == [32, 32, 32]

    # 3. Request a chunk: s0/c/0/0/0
    r_chunk = client.get("/test_pipeline/zarr3/s0/c/0/0/0")
    assert r_chunk.status_code == 200
    assert len(r_chunk.content) > 0
    assert r_chunk.headers["content-type"] == "application/octet-stream"

    # 4. Verify cache hit in shared_cache
    stats = registry.cache.stats()
    assert stats["entries"] > 0


def test_stateless_payload_and_dataset_chaining(client):
    # Base dataset registered
    base_src = MultiscaleSource([DummySource("chain_base")])
    base_pipeline = Pipeline(base_src, [{"op": "gaussian", "sigma": 1.0}], cache=registry.cache)
    registry.add("chain_base", base_pipeline)

    # Chained pipeline referencing base dataset: dataset://chain_base
    spec = {
        "source": "dataset://chain_base",
        "ops": [
            {"op": "frangi", "sigmas": [1.0, 2.0]},
            {"op": "otsu", "value": 1},
        ],
    }
    payload = encode_payload(spec, compress=True)

    # Request chunk via stateless payload URL
    r_chunk = client.get(f"/{payload}/zarr3/s0/c/0/0/0")
    assert r_chunk.status_code == 200
    assert len(r_chunk.content) > 0
