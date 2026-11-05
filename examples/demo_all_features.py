#!/usr/bin/env python3
"""
End-to-end feature tour and live demo for `zarr_proxy` (`zarr-image-processing-wrapper`).

Demonstrates:
1. All 17 domain-specific 3D volumetric operators (`zarr_proxy/ops`) + fused halo planning.
2. `LocalBackend` and `DaskSource` (lazy Dask array wrapping with halo-padded chunk reads).
3. `WebWorkerBackend` (`ngplatform` ComputedPipeline compilation, `#halo` / `#output` /
   `#uicontrol` Python templates, `virtual/ngp/` deflate-raw URLs, and `ToPyWorker`
   `plan` / `compute` / `describe` / `call` RPC protocol).
4. `WorkstationBackend` (binary `zlib` array serialization, remote `dispatch_op`,
   multi-stage `compute_pipeline` with halo cropping, `evaluate_box` via `RemoteArrayRef`,
   and dynamic dataset registration).
5. Multi-protocol virtual Zarr server (`zarr3`, `zarr` v2, `n5`, `precomputed`) and
   interactive Studio `/_demo/slice` endpoint.

Run with:
    pixi run demo
or:
    python examples/demo_all_features.py
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
import dask.array as da
import numpy as np
from starlette.testclient import TestClient

from chunkmirage import fused
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops import op_from_spec
from chunkmirage.pipeline import Pipeline
from chunkmirage.sources import open_source
import zarr_proxy.ops  # noqa: F401 - register custom 3D ops
from zarr_proxy.backends import (
    BackendConfig,
    RemoteArrayRef,
    WebWorkerBackend,
    WorkstationBackend,
    build_ngplatform_url,
    from_ngplatform_pipeline,
    get_backend,
    get_own_python_snippets,
    list_backends,
    to_ngplatform_pipeline,
)
from zarr_proxy.resolver import encode_payload
from zarr_proxy.server import app
from zarr_proxy.sources import DaskSource, build_dask_pyramid


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")


def demo_all_ops() -> None:
    _banner("1. All 17 Domain 3D Volumetric Ops & Automated Halo Planning")
    src = open_source("synthetic://blobs?shape=64,64,64")
    lvl = src[0]
    out_box = Box((16, 16, 16), (48, 48, 48))

    op_specs: List[Dict[str, Any]] = [
        {"op": "sato", "sigmas": [1.0, 2.0], "black_ridges": False},
        {"op": "frangi", "sigmas": [1.0, 2.0], "alpha": 0.5, "beta": 0.5},
        {"op": "meijering", "sigmas": [1.0, 2.0]},
        {"op": "sobel"},
        {"op": "scharr"},
        {"op": "laplace"},
        {"op": "unsharp_mask", "radius": 1.5, "amount": 2.0},
        {"op": "median", "size": 3},
        {"op": "erode", "radius": 1},
        {"op": "dilate", "radius": 1},
        {"op": "white_tophat", "radius": 2},
        {"op": "ct_window", "vmin": 20.0, "vmax": 200.0, "target_dtype": "uint8"},
        {"op": "dynamic_range", "pmin": 5.0, "pmax": 95.0, "target_dtype": "uint8"},
        {"op": "gamma", "gamma": 0.6},
        {"op": "otsu", "nbins": 128, "value": 1},
        {"op": "li", "value": 1},
        {"op": "passthrough"},
    ]

    print(f"{'Op Name':<16} | {'Halo (z,y,x)':<14} | {'In Shape':<14} | {'Out Shape':<12} | {'Dtype':<8} | {'Time'}")
    print("-" * 78)
    for spec in op_specs:
        op_obj = op_from_spec(spec)
        out_info, _, halo = fused.plan(lvl.info, [op_obj])
        in_box = out_box.pad(halo)
        t0 = time.perf_counter()
        padded = np.asarray(lvl.read_padded(in_box, edge=True))
        result = fused.run([op_obj], padded, in_box, out_box, out_info, halo)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"{spec['op']:<16} | {str(list(halo)):<14} | {str(padded.shape):<14} | "
            f"{str(result.shape):<12} | {result.dtype.name:<8} | {dt_ms:5.1f} ms"
        )


def demo_local_and_dask() -> None:
    _banner("2. LocalBackend & DaskSource Fused Multi-Stage Pipeline")
    rng = np.random.default_rng(42)
    raw_vol = (rng.random((64, 64, 64), dtype=np.float32) * 1000.0).astype(np.uint16)

    dask_arr = da.from_array(raw_vol, chunks=(32, 32, 32))
    dask_pyramid = build_dask_pyramid(dask_arr, levels=2, voxel_size=(8.01, 8.01, 8.01), units=("um", "um", "um"))
    dask_src = dask_pyramid[0]

    ops = [
        {"op": "median", "size": 3},
        {"op": "sato", "sigmas": [1.0, 2.0]},
        {"op": "li", "value": 1},
    ]
    pipeline = Pipeline(dask_pyramid, [op_from_spec(o) for o in ops])
    box = Box((16, 16, 16), (48, 48, 48))

    t0 = time.perf_counter()
    seg_chunk = np.asarray(pipeline.read(0, box))
    dt_ms = (time.perf_counter() - t0) * 1000.0

    local_backend = get_backend("local")
    sub = local_backend.evaluate_box(raw_vol, box)

    print(f"Registered backends : {list_backends()}")
    print(f"DaskSource info     : shape={dask_src.info.shape}, chunks={dask_src.info.chunk_shape}, voxel={dask_src.info.voxel_size}")
    print(f"Fused 3-stage ops   : median(3) -> sato([1,2]) -> li(1)")
    print(f"Output chunk        : shape={seg_chunk.shape}, dtype={seg_chunk.dtype.name}, fg_voxels={int(seg_chunk.sum())} ({dt_ms:.1f} ms)")
    print(f"LocalBackend slice  : shape={sub.shape}, dtype={sub.dtype.name}")


def demo_webworker_backend() -> None:
    _banner("3. WebWorkerBackend (ngplatform Compiler & Pyodide Worker Protocol)")
    spec = {
        "source": "zarr3://https://storage.googleapis.com/ucl-hip-ct/heart.ome.zarr",
        "ops": [
            {"op": "median", "size": 3},
            {"op": "sato", "sigmas": [1.0, 2.0], "black_ridges": False},
            {"op": "otsu", "nbins": 256, "value": 1},
        ],
    }

    ww = WebWorkerBackend(BackendConfig(backend="webworker", options={"simulate_pyodide": True}))
    ngp_pipeline = to_ngplatform_pipeline(spec, chunk=[32, 32, 32])
    ngp_url = build_ngplatform_url(spec, base_url="http://localhost:5173/")
    roundtrip_spec = from_ngplatform_pipeline(ngp_url)

    print(f"ngplatform Virtual URL : {ngp_url[:88]}...")
    print(f"Compiled stages        : {[op['op'] for op in ngp_pipeline['ops']]}")
    print(f"Decompiled round-trip  : {[op['op'] if isinstance(op, dict) else op.name for op in roundtrip_spec.ops]}")
    print(f"OWN_PYTHON templates   : {sorted(get_own_python_snippets().keys())}")
    print("\n--- Compiled Stage 2 (Sato 3D) Pyodide Script with #halo & #uicontrol ---")
    print(ngp_pipeline["ops"][1]["code"])

    # Execute ToPyWorker message protocol (plan -> compute)
    plan_ans = ww.handle_worker_message({
        "reqId": 1,
        "type": "plan",
        "views": {
            "v0": {
                "ops": spec["ops"],
                "shape": [64, 64, 64],
                "dtype": "uint16",
                "chunk": [16, 16, 16],
                "voxel": [1.0, 1.0, 1.0],
            }
        },
    })
    plan_meta = plan_ans["value"]["v0"]
    print(f"ToPyWorker 'plan'      : halo={plan_meta['halo']}, out_dtype={plan_meta['dtype']}, kind={plan_meta['kind']}")

    in_lo, in_hi = [9, 9, 9], [41, 41, 41]
    out_lo, out_hi = [16, 16, 16], [32, 32, 32]
    read_shape = [32, 32, 32]
    test_block = np.arange(int(np.prod(read_shape)), dtype=np.uint16).reshape(read_shape)

    comp_ans = ww.handle_worker_message({
        "reqId": 2,
        "type": "compute",
        "view": "v0",
        "level": 0,
        "data": test_block.tobytes(),
        "readShape": read_shape,
        "inLo": in_lo,
        "inHi": in_hi,
        "outLo": out_lo,
        "outHi": out_hi,
        "full": [64, 64, 64],
        "voxel": [1.0, 1.0, 1.0],
    })
    out_arr = np.frombuffer(comp_ans["value"], dtype=np.uint8).reshape((16, 16, 16))
    print(f"ToPyWorker 'compute'   : output shape={out_arr.shape}, dtype={out_arr.dtype.name}, unique={np.unique(out_arr).tolist()}")
    ww.shutdown()


def demo_workstation_and_server() -> None:
    _banner("4. WorkstationBackend RPC & Multi-Protocol Virtual Zarr Server")
    with TestClient(app) as client:
        ws = WorkstationBackend(
            BackendConfig(
                backend="workstation",
                options={
                    "endpoint": "http://testserver",
                    "client": client,
                    "fallback_local": False,
                },
            )
        )

        health = ws.check_health()
        print(f"Workstation health     : {health}")

        # 1. Fused multi-stage compute_pipeline with halo cropping over binary zlib RPC
        raw_block = np.random.default_rng(7).normal(100.0, 25.0, size=(28, 28, 28)).astype(np.float32)
        in_box = Box((10, 10, 10), (38, 38, 38))
        out_box = Box((16, 16, 16), (32, 32, 32))
        ops = [
            {"op": "sato", "sigmas": [1.0, 2.0]},
            {"op": "dynamic_range", "pmin": 2.0, "pmax": 98.0, "target_dtype": "uint8"},
        ]
        res = ws.compute_pipeline(raw_block, ops, in_box=in_box, out_box=out_box, pad_shape=(64, 64, 64))
        print(f"Workstation RPC pipe   : in={raw_block.shape} ({raw_block.dtype.name}) -> out={res.shape} ({res.dtype.name})")

        # 2. Remote subvolume evaluation via RemoteArrayRef
        ref = RemoteArrayRef(
            source="synthetic://blobs?shape=64,64,64",
            level=0,
            ops=[{"op": "sobel"}],
        )
        remote_sub = ws.evaluate_box(ref, Box((8, 8, 8), (24, 24, 24)))
        print(f"RemoteArrayRef eval    : shape={remote_sub.shape}, dtype={remote_sub.dtype.name}, max={float(remote_sub.max()):.2f}")

        # 3. Multi-protocol virtual Zarr endpoints (Zarr v3, Zarr v2, N5, Precomputed)
        payload = encode_payload(
            {
                "source": "synthetic://blobs?shape=64,64,64",
                "ops": [{"op": "sato", "sigmas": [1.0, 2.0]}],
            },
            compress=True,
        )
        z3_resp = client.get(f"/{payload}/zarr3/zarr.json")
        z2_resp = client.get(f"/{payload}/zarr/.zgroup")
        n5_resp = client.get(f"/{payload}/n5/attributes.json")
        pc_resp = client.get(f"/{payload}/precomputed/info")
        print(
            f"Multi-protocol status  : zarr3={z3_resp.status_code}, zarr2={z2_resp.status_code}, "
            f"n5={n5_resp.status_code}, precomputed={pc_resp.status_code}"
        )

        ws.shutdown()


def main() -> None:
    demo_all_ops()
    demo_local_and_dask()
    demo_webworker_backend()
    demo_workstation_and_server()
    print(f"\n{'=' * 78}")
    print("  All demos completed! Run `pixi run proxy-dev` and open http://localhost:8000")
    print("  to use the interactive Studio & Backend Demo Lab in your browser.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
