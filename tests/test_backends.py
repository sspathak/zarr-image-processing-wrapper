import base64
import dask.array as da
import httpx
import numpy as np
import pytest

from chunkmirage.core import Box
from chunkmirage.ops import op_from_spec
from zarr_proxy.backends import (
    BackendConfig,
    ComputeBackend,
    DaskBackend,
    LocalBackend,
    RemoteArrayRef,
    WebWorkerBackend,
    WorkstationBackend,
    WorkstationError,
    build_ngplatform_url,
    clean_engine_source,
    compile_op_for_webworker,
    decode_array_payload,
    decompile_webworker_op,
    encode_array_payload,
    from_ngplatform_pipeline,
    get_backend,
    get_own_python_snippets,
    get_pyworker_modules,
    get_webworker_js,
    is_trusted_webworker_python,
    list_backends,
    register_backend,
    set_default_backend,
    to_ngplatform_pipeline,
)
from zarr_proxy.backends.webworker import NGPLATFORM_FILTERS_FRANGI, NGPLATFORM_FILTERS_MEDIAN
from zarr_proxy.ops import (
    CTWindow,
    Dilate,
    DynamicRange,
    Erode,
    Frangi,
    GammaCorrection,
    Identity,
    Laplace,
    Li,
    Median,
    Meijering,
    Otsu,
    Passthrough,
    Sato,
    Scharr,
    Sobel,
    UnsharpMask,
    WhiteTopHat,
)
from zarr_proxy.server import app, registry


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
    backend.shutdown()


def test_backend_registry():
    backends = list_backends()
    assert "local" in backends
    assert "dask" in backends
    assert "webworker" in backends
    assert "workstation" in backends

    local_b = get_backend(BackendConfig(name="local"))
    assert isinstance(local_b, LocalBackend)

    dask_b = get_backend("dask")
    assert isinstance(dask_b, DaskBackend)

    ww_b = get_backend({"name": "webworker"})
    assert isinstance(ww_b, WebWorkerBackend)
    ww_b.shutdown()

    ws_b = get_backend("workstation")
    assert isinstance(ws_b, WorkstationBackend)
    ws_b.shutdown()

    set_default_backend("local")
    assert isinstance(get_backend(), LocalBackend)

    class CustomBackend(ComputeBackend):
        name = "custom"

        def evaluate_box(self, array_like, box):
            return np.ones(box.shape)

    register_backend("custom", CustomBackend)
    assert "custom" in list_backends()
    custom_b = get_backend(BackendConfig(name="custom"))
    assert isinstance(custom_b, CustomBackend)


# ---------------------------------------------------------------------------
# WebWorker Backend Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "native_op",
    [
        Sato(sigmas=[1.0, 2.0], black_ridges=False),
        Frangi(sigmas=[1.0, 2.0], alpha=0.5, beta=0.5, black_ridges=False),
        Meijering(sigmas=[1.0, 2.0], black_ridges=False),
        Sobel(),
        Scharr(),
        Laplace(),
        UnsharpMask(radius=1.5, amount=2.0),
        Median(size=3),
        Erode(radius=1),
        Dilate(radius=1),
        WhiteTopHat(radius=2),
        CTWindow(vmin=10.0, vmax=200.0, target_dtype="uint8"),
        CTWindow(vmin=10.0, vmax=200.0, target_dtype="uint16"),
        CTWindow(vmin=10.0, vmax=200.0, target_dtype="float32"),
        DynamicRange(pmin=5.0, pmax=95.0, target_dtype="uint8"),
        GammaCorrection(gamma=0.8),
        Otsu(nbins=64, value=1),
        Li(value=1),
        Passthrough(),
        Identity(),
    ],
)
def test_webworker_op_compilation_and_numerical_equivalence(native_op):
    rng = np.random.default_rng(42)
    block = (rng.random((12, 12, 12), dtype=np.float32) * 250.0).astype(np.float32)
    block[5:7, :, 5:7] += 500.0  # synthetic bright vessel

    compiled_spec = compile_op_for_webworker(native_op)
    assert compiled_spec["op"] == "python"
    assert is_trusted_webworker_python(compiled_spec["code"])

    # Instantiate as chunkmirage PythonOp (validates AST, #halo, #output, #uicontrol)
    py_op = op_from_spec(compiled_spec)
    assert py_op.halo_for(3) == native_op.halo_for(3)
    assert py_op.output_dtype(block.dtype) == native_op.output_dtype(block.dtype)

    expected = native_op.apply(block)
    actual = py_op.apply(block)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    # Verify decompilation back to native zarr_proxy op
    decompiled = decompile_webworker_op(compiled_spec)
    expected_name = "passthrough" if native_op.name == "identity" else native_op.name
    assert decompiled["op"] == expected_name


def test_webworker_ngplatform_pipeline_roundtrip():
    spec = {
        "source": "zarr3://https://storage.googleapis.com/bucket/heart.ome.zarr|zarr3:",
        "ops": [
            {"op": "gaussian", "sigma": 1.0},
            {"op": "sato", "sigmas": [1.0, 2.0], "black_ridges": False},
            {"op": "li", "value": 1},
        ],
        "chunk_shape": [64, 64, 64],
        "select": {"c": 0},
    }

    ngp = to_ngplatform_pipeline(spec)
    assert ngp["source"] == "https://storage.googleapis.com/bucket/heart.ome.zarr"
    assert ngp["chunk"] == [64, 64, 64]
    assert ngp["select"] == {"c": 0}
    assert ngp["ops"][0] == {"op": "gaussian", "sigma": 1.0}  # native chunkmirage op preserved
    assert ngp["ops"][1]["op"] == "python"  # sato compiled to python op
    assert ngp["ops"][2]["op"] == "python"  # li compiled to python op

    url = build_ngplatform_url(spec, base_url="http://localhost:5173/")
    assert url.startswith("zarr3://http://localhost:5173/virtual/ngp/")

    recovered = from_ngplatform_pipeline(url, decompile_ops=True)
    assert recovered.source == "https://storage.googleapis.com/bucket/heart.ome.zarr"
    assert list(recovered.chunk_shape) == [64, 64, 64]
    assert recovered.select == {"c": 0}
    assert recovered.ops[0] == {"op": "gaussian", "sigma": 1.0}
    assert recovered.ops[1]["op"] == "sato"
    assert recovered.ops[2]["op"] == "li"


def test_decompile_builtin_ngplatform_filters_ops():
    # Verify compatibility with ngplatform/plugins/filters/pipeline.ts MEDIAN and FRANGI
    median_spec = {"op": "python", "code": NGPLATFORM_FILTERS_MEDIAN, "params": {"radius": 2}}
    assert decompile_webworker_op(median_spec) == {"op": "median", "size": 5}

    frangi_spec = {
        "op": "python",
        "code": NGPLATFORM_FILTERS_FRANGI,
        "params": {"sigma_low": 1.0, "sigma_high": 3.0, "bright": True},
    }
    dec_frangi = decompile_webworker_op(frangi_spec)
    assert dec_frangi["op"] == "frangi"
    assert dec_frangi["sigmas"] == [1.0, 1.5, 2.0, 2.5, 3.0]
    assert dec_frangi["black_ridges"] is False


def test_webworker_backend_rpc_protocol():
    backend = WebWorkerBackend(
        BackendConfig(name="webworker", options={"simulate_pyodide": True, "chunk_shape": [8, 8, 8]})
    )
    try:
        # 1. Plan a view with Sato + Li
        plan_msg = {
            "type": "plan",
            "reqId": 1,
            "views": {
                "v0": {
                    "ops": [
                        {"op": "sato", "sigmas": [1.0], "black_ridges": False},
                        {"op": "li", "value": 1},
                    ],
                    "shape": [16, 16, 16],
                    "dtype": "float32",
                    "chunk": [8, 8, 8],
                    "voxel": [1.0, 1.0, 1.0],
                }
            },
        }
        plan_ans = backend.handle_worker_message(plan_msg)
        assert plan_ans["reqId"] == 1
        v0_plan = plan_ans["value"]["v0"]
        assert v0_plan["dtype"] == "uint8"
        assert v0_plan["kind"] == "mask"
        assert v0_plan["halo"] == [7, 7, 7]  # ceil(3*1.0)=3 + 4 = 7

        # 2. Compute a chunk [0..8, 0..8, 0..8] with halo padded input
        raw_vol = np.zeros((15, 15, 15), dtype=np.float32)
        raw_vol[4, :, 4] = 100.0
        comp_msg = {
            "type": "compute",
            "reqId": 2,
            "view": "v0",
            "level": 0,
            "data": raw_vol.tobytes(),
            "readShape": [15, 15, 15],
            "inLo": [-7, -7, -7],
            "inHi": [15, 15, 15],
            "outLo": [0, 0, 0],
            "outHi": [8, 8, 8],
            "full": [16, 16, 16],
            "voxel": [1.0, 1.0, 1.0],
        }
        comp_ans = backend.handle_worker_message(comp_msg)
        assert comp_ans["reqId"] == 2
        chunk_arr = np.frombuffer(comp_ans["value"], dtype=np.uint8).reshape((8, 8, 8))
        assert chunk_arr.shape == (8, 8, 8)
        assert np.any(chunk_arr == 1)

        # 3. Verify modules and JS script generation
        mods = get_pyworker_modules()
        assert "zarr_proxy/ops/filters.py" in mods
        js = get_webworker_js()
        assert "loadPyodide" in js
        assert "zarr_proxy/ops/filters.py" in js
        assert len(get_own_python_snippets()) >= 17
        assert clean_engine_source("zarr3://http://localhost:8000/test|zarr3:") == "http://localhost:8000/test"
    finally:
        backend.shutdown()


# ---------------------------------------------------------------------------
# Workstation Backend Tests
# ---------------------------------------------------------------------------


def test_workstation_array_wire_codec():
    arr = np.linspace(0, 100, 1000, dtype=np.float32).reshape((10, 10, 10))
    for comp in ("zlib", "none"):
        env = encode_array_payload(arr, compression=comp)
        assert env["dtype"] == "float32"
        assert env["shape"] == [10, 10, 10]
        assert env["compression"] == comp
        decoded = decode_array_payload(env)
        np.testing.assert_array_equal(decoded, arr)


def test_workstation_backend_over_asgi_transport():
    from starlette.testclient import TestClient

    client = TestClient(app, base_url="http://workstation.local")
    ws = WorkstationBackend(
        BackendConfig(
            name="workstation",
            options={
                "endpoint": "http://workstation.local",
                "client": client,
                "fallback_local": False,
                "compression": "zlib",
            },
        )
    )
    try:
        # 1. Health & Stats
        health = ws.check_health()
        assert health["online"] is True
        assert health["status"] == "ok"

        stats = ws.get_stats()
        assert "workstation" in stats["backends"]
        assert "webworker" in stats["backends"]

        # 2. Remote dispatch_op with Op instance, op name, and apply_filter
        vol = np.zeros((12, 12, 12), dtype=np.float32)
        vol[6, :, 6] = 200.0

        sato_op = Sato(sigmas=[1.0], black_ridges=False)
        res_op = ws.dispatch_op(sato_op, vol)
        np.testing.assert_allclose(res_op, sato_op.apply(vol), rtol=1e-5, atol=1e-5)

        res_str = ws.dispatch_op("sobel", vol)
        np.testing.assert_allclose(res_str, Sobel().apply(vol), rtol=1e-5, atol=1e-5)

        res_cls = ws.dispatch_op(Sato, vol, sigmas=[1.0])
        np.testing.assert_allclose(res_cls, sato_op.apply(vol), rtol=1e-5, atol=1e-5)

        # 3. Multi-stage fused pipeline with halo cropping on the workstation
        in_box = Box((-3, -3, -3), (11, 11, 11))
        out_box = Box((0, 0, 0), (8, 8, 8))
        clipped_data = vol[:11, :11, :11]
        fused_res = ws.compute_pipeline(
            clipped_data,
            [{"op": "sato", "sigmas": [1.0]}, {"op": "li", "value": 1}],
            in_box=in_box,
            out_box=out_box,
            pad_shape=(12, 12, 12),
        )
        assert fused_res.shape == (8, 8, 8)
        assert fused_res.dtype == np.uint8

        # 4. Remote evaluate_box on a synthetic source with ops
        ref = RemoteArrayRef(
            source="synthetic://blobs?shape=32,32,32",
            level=0,
            ops=[{"op": "gaussian", "sigma": 1.0}],
        )
        sub = ws.evaluate_box(ref, Box((0, 0, 0), (16, 16, 16)))
        assert sub.shape == (16, 16, 16)
        assert sub.dtype == np.float32

        # 5. Build source URL for workstation
        src_url = ws.build_source_url({
            "source": "s3://bucket/heart.zarr",
            "ops": [compile_op_for_webworker({"op": "sato", "sigmas": [1.0, 2.0]})],
            "chunk": [64, 64, 64],
        })
        assert src_url.startswith("zarr3://http://workstation.local/")
        assert src_url.endswith("/zarr3")
    finally:
        ws.shutdown()


def test_workstation_backend_fallback_and_strict_mode():
    # Unreachable workstation with fallback_local=True
    ws_fallback = WorkstationBackend(
        BackendConfig(
            name="workstation",
            options={
                "endpoint": "http://127.0.0.1:1",
                "timeout": 0.05,
                "retries": 0,
                "fallback_local": True,
            },
        )
    )
    vol = np.ones((8, 8, 8), dtype=np.float32) * 10.0
    out = ws_fallback.dispatch_op("sobel", vol)
    assert out.shape == (8, 8, 8)
    ws_fallback.shutdown()

    # Unreachable workstation with fallback_local=False raises WorkstationError
    ws_strict = WorkstationBackend(
        BackendConfig(
            name="workstation",
            options={
                "endpoint": "http://127.0.0.1:1",
                "timeout": 0.05,
                "retries": 0,
                "fallback_local": False,
            },
        )
    )
    with pytest.raises(WorkstationError):
        ws_strict.dispatch_op("sobel", vol)
    ws_strict.shutdown()
