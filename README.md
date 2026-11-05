# Virtual Zarr Compute Server

A high-performance virtual compute server powered by **[chunkmirage](https://github.com/JaneliaSciComp/chunkmirage)** that serves on-the-fly volumetric image processing pipelines as lazy **OME-Zarr v3 / v2 / N5 / Neuroglancer Precomputed** datasets.

---

## The User Story: How & Why It Works

Imagine you are a researcher or bioimaging scientist analyzing a **500 GB whole-organ synchrotron scan**—such as the UCL Human Organ Atlas (HiP-CT) whole human heart dataset stored in cloud object storage (`gs://` or `s3://`).

### ❌ The Old Way: Slow Feedback Loops & Disk Bloat
To segment fine coronary micro-vessels or test whether a Sato/Frangi filter enhances capillary networks:
1. **Downloading a local crop** (e.g. 512³ voxels) gives a partial view, but parameters tuned on a small ROI rarely generalize across varying tissue densities in the full organ.
2. **Materializing intermediate Zarr volumes** by submitting an offline batch job takes hours to process hundreds of gigabytes—only to discover that $\sigma$ was too large or a threshold clipped the vessels.
3. Every test run generates another massive intermediate dataset on disk, rapidly consuming storage budgets and killing interactive exploration.

### ✨ The Virtual Zarr Way: Interactive Zero-Disk Exploration
With this virtual compute server powered by **chunkmirage**, each processing stage is an on-demand, stateless OME-Zarr endpoint:

```text
[Remote Cloud Storage]                  [Virtual Zarr Proxy]                     [Interactive Viewer]
     (GCS / S3)                              (chunkmirage)                          (Neuroglancer)
         │                                         │                                      │
         │                                         │ 1. Demands chunk (z, y, x)           │
         │                                         │ ◄────────────────────────────────────┤
         │ 2. Fetches raw chunk + halo margin      │                                      │
         │ ◄───────────────────────────────────────┤                                      │
         │ 3. Returns compressed raw bytes         │                                      │
         ├────────────────────────────────────────►│                                      │
         │                                         │ 4. Evaluates Fused Pipeline:         │
         │                                         │    Gaussian(1.0) ──► Sato ──► Scale  │
         │                                         │ 5. Crops halo & caches stage in RAM  │
         │                                         │                                      │
         │                                         │ 6. Streams filtered chunk to WebGL   │
         │                                         ├─────────────────────────────────────►│
         │                                         │                                      │ (Renders 3D vessels!)
```

#### Step-by-Step Experience:
1. **Connect Directly to Cloud Data**:
   You start the server (`pixi run proxy`) and open the Web Studio at `http://localhost:8000`. You select the UCL HiP-CT Heart volume directly from Google Cloud Storage without downloading a single byte ahead of time.
2. **One-Click Virtual Pipeline**:
   You click **"🌟 Sato Tubeness"** (or build a multi-stage pipeline like `Median` $\to$ `Sato` $\to$ `Li`).
3. **Instant 3D Browsing in Neuroglancer**:
   You click **"Open in Neuroglancer"**. The 4-panel 3D viewer immediately loads the heart. As you navigate the 3D volume or slice through the left ventricle, the browser requests only the chunks visible in your viewport.
4. **On-The-Fly Computation**:
   The proxy server fetches the raw CT chunks on demand, pads them with the exact spatial halo required by the filter kernel (e.g. 6 voxels for $\sigma=2.0$), evaluates the 3D Hessian eigenvalues in ~90 ms, crops the boundary margin, and streams the result directly into your browser's WebGL canvas.
5. **Instant Parameter Tweaking with Upstream Caching**:
   Want to see how an automated threshold (Li or Otsu) looks over the enhanced vessels? You add a `Li` segmentation stage. Because chunkmirage uses **hierarchical content-addressed LRU caching**, the expensive Sato filter output for chunks you've already browsed is reused directly from memory—the thresholding runs in $<2$ ms!
6. **Zero Disk Footprint**:
   You freely explore gigabytes of computed results across different scales and angles without writing a single intermediate array to disk. Once you find the perfect parameters, you export the exact pipeline JSON to run in production with total confidence.

---

## Key Capabilities

1. **Virtualization on the Compute Layer**:
   - Intercepts chunk requests on-the-fly and evaluates operations lazily.
   - Eliminates the need to materialize expensive intermediate Zarr volumes to disk when experimenting with filter parameters.
   - Provides instantaneous visual feedback in **Neuroglancer**, **Napari**, or **Fiji/BigDataViewer**.

2. **Hierarchical Content-Addressed Caching**:
   - Uses chunkmirage's per-stage LRU chunk caching keyed by pipeline stage digests.
   - When downstream parameters (e.g. Otsu threshold) are tuned, upstream computed stages (e.g. Frangi vesselness) **remain cached in memory** and are not recomputed.

3. **In-Process Pipeline Stacking (`dataset://`)**:
   - Stack multiple virtual datasets into an interactive processing graph without intermediate storage:
     ```
     Raw Volume (S3 / Local)
       └── Denoise (dataset://raw + Gaussian)
             └── Vessels (dataset://denoise + Frangi)
                   └── Mask (dataset://vessels + Otsu)
     ```
   - Opening intermediate and final layers side-by-side computes prerequisite chunks **only once**.

4. **Pluggable Compute Backends (`zarr_proxy.backends`)**:
   - **LocalBackend (`local`)**: Multicore in-process CPU execution using NumPy and SciPy.
   - **DaskBackend (`dask`)**: Lazy computation graph execution across threads, processes, or Dask Distributed clusters.
   - **WebWorkerBackend (`webworker`)**: Browser-side Pyodide WebWorker execution and pipeline compilation for [ngplatform](https://github.com/JaneliaSciComp/ngplatform) (`ctx.workspace.computed`).
   - **WorkstationBackend (`workstation`)**: Remote HTTP/RPC execution offloading heavy 3D subvolume reads and multi-stage fused pipelines to a dedicated high-memory or multi-core workstation.

5. **ngplatform & Pyodide Integration**:
   - Natively encodes and decodes the `deflate-raw + base64url` pipeline payload format used by [ngplatform](https://github.com/JaneliaSciComp/ngplatform).
   - Compiles all 17 `zarr_proxy` domain ops (`sato`, `frangi`, `meijering`, `sobel`, `scharr`, `laplace`, `unsharp_mask`, `median`, `erode`, `dilate`, `white_tophat`, `ct_window`, `dynamic_range`, `gamma`, `otsu`, `li`, `passthrough`) into self-contained `chunkmirage` `python` ops (`#halo`, `#output`, `#uicontrol`) for in-browser **Pyodide** WebWorkers, and automatically decompiles them back into native ops when resolved on a workstation server.

---

## Using the Compute Backends

All compute backends share the unified `ComputeBackend` interface in `zarr_proxy.backends` and can be retrieved by name or configuration via `get_backend()`:

```python
from zarr_proxy.backends import BackendConfig, get_backend, list_backends

print(list_backends())  # ['dask', 'local', 'webworker', 'workstation']

backend = get_backend("local")  # or BackendConfig(name="workstation", options={...})
```

### 1. WebWorker Backend (`webworker`) — Browser Pyodide & `ngplatform`

The `WebWorkerBackend` enables zero-server, in-browser execution inside `ngplatform`'s Service Worker + Pyodide WebWorker pool (`ctx.workspace.computed`), as well as local Pyodide simulation and worker-pool testing.

#### A. Compiling a Pipeline for `ngplatform` in Python

```python
from zarr_proxy.backends import (
    WebWorkerBackend,
    BackendConfig,
    build_ngplatform_url,
    from_ngplatform_pipeline,
    get_own_python_snippets,
    to_ngplatform_pipeline,
)

spec = {
    "source": "https://storage.googleapis.com/ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr",
    "ops": [
        {"op": "sato", "sigmas": [1.0, 2.0], "black_ridges": False},
        {"op": "li", "value": 1},
    ],
    "chunk_shape": [64, 64, 64],
}

# 1. Convert to an ngplatform ComputedPipeline ({source, ops, chunk, select})
#    Custom zarr_proxy ops are compiled into self-contained chunkmirage `python` ops
#    with exact #halo, #output, and #uicontrol directives.
ngp_pipeline = to_ngplatform_pipeline(spec)

# 2. Build the deflate-raw + base64url virtual source URL for ngplatform
ngp_url = build_ngplatform_url(spec, base_url="http://localhost:5173/")
# -> "zarr3://http://localhost:5173/virtual/ngp/<deflate-raw-base64url>/"

# 3. Round-trip back from an ngplatform URL/payload to a native PipelineSpec
native_spec = from_ngplatform_pipeline(ngp_url, decompile_ops=True)
```

#### B. Integrating with an `ngplatform` Plugin (`plugin.ts`)

When writing an `ngplatform` plugin (see `ngplatform/plugins/starter_ts`), register the canonical Python snippets via `ctx.workspace.computed.ownPython(code)` during `activate(ctx)` so shared links execute immediately without a 403 prompt:

```ts
// Inside activate(ctx: PlatformContext) of an ngplatform plugin:
const resp = await fetch("http://localhost:8000/_webworker/status");
const { own_python_templates } = await resp.json();
for (const code of Object.values(own_python_templates)) {
  ctx.workspace.computed.ownPython(code as string);
}

// Compile and add a virtual computed layer executed by Pyodide WebWorkers:
const compiled = await fetch("http://localhost:8000/_webworker/compile", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    source: selectedLayerSource,
    ops: [{ op: "sato", sigmas: [1.0, 2.0] }, { op: "li", value: 1 }],
    chunk: [64, 64, 64],
  }),
}).then((r) => r.json());

const url = await ctx.workspace.computed.url(compiled.pipeline);
ctx.workspace.addDataset({ source: url, name: "Sato + Li (WebWorker)", kind: "segmentation" });
```

#### C. Local Pyodide Simulation & Worker RPC Protocol

```python
from chunkmirage.core import Box
from zarr_proxy.backends import BackendConfig, WebWorkerBackend
from zarr_proxy.ops import Sato

ww = WebWorkerBackend(
    BackendConfig(name="webworker", options={"simulate_pyodide": True, "max_workers": 2})
)

# Evaluates the op by compiling it to a chunkmirage PythonOp and running it in the worker pool
out = ww.dispatch_op(Sato(sigmas=[1.0, 2.0]), volume_block)

# Or drive the chunkmirage-web ToPyWorker RPC protocol directly ("plan", "compute", "describe", "call")
plan_reply = ww.handle_worker_message({
    "type": "plan",
    "reqId": 1,
    "views": {
        "v0": {
            "ops": [{"op": "sato", "sigmas": [1.0, 2.0]}],
            "shape": [64, 64, 64],
            "dtype": "float32",
            "chunk": [32, 32, 32],
            "voxel": [1.0, 1.0, 1.0],
        }
    },
})
ww.shutdown()
```

#### D. WebWorker HTTP Endpoints
- `GET /_webworker/status` — Returns supported ops, default chunk shape, and canonical `own_python_templates`.
- `POST /_webworker/compile` — Compiles a `zarr_proxy` pipeline JSON into an `ngplatform` `ComputedPipeline` + `neuroglancer_source` URL.
- `POST /_webworker/message` — Executes a `ToPyWorker` RPC message (`plan`, `compute`, `describe`, `call`).
- `GET /_webworker/worker.js` — Serves a standalone ES module Pyodide WebWorker script with `zarr_proxy.ops` pre-mounted.

---

### 2. Workstation Backend (`workstation`) — Remote High-Memory / Multi-Core Offloading

When datasets or 3D multi-scale Hessian kernels exceed browser WebAssembly memory or CPU limits, `WorkstationBackend` offloads subvolume reads, single operators, and multi-stage fused pipelines to a remote `zarr_proxy` workstation server over HTTP/RPC with `zlib` binary array serialization.

#### A. Offloading Operations & Fused Pipelines from Python

```python
import numpy as np
from chunkmirage.core import Box
from zarr_proxy.backends import BackendConfig, RemoteArrayRef, WorkstationBackend
from zarr_proxy.ops import Sato

ws = WorkstationBackend(
    BackendConfig(
        name="workstation",
        options={
            "endpoint": "http://workstation.my-lab.org:8000",  # or env ZARR_WORKSTATION_URL
            "timeout": 60.0,
            "retries": 2,
            "compression": "zlib",       # "zlib" or "none"
            "fallback_local": True,      # fall back to local CPU if workstation is unreachable
            # "token": "my-secret-token" # or env ZARR_WORKSTATION_TOKEN
        },
    )
)

# 1. Check workstation health and cache telemetry
print(ws.check_health())  # {'online': True, 'endpoint': '...', 'status': 'ok', ...}
print(ws.get_stats())

# 2. Offload a single operator on a 3D NumPy block
block = np.random.rand(64, 64, 64).astype(np.float32)
vesselness = ws.dispatch_op(Sato(sigmas=[1.0, 2.0]), block)

# 3. Offload a multi-stage fused pipeline with edge padding and halo cropping in one RPC call
in_box = Box((-6, -6, -6), (70, 70, 70))
out_box = Box((0, 0, 0), (64, 64, 64))
segmented = ws.compute_pipeline(
    block,
    ops=[
        {"op": "sato", "sigmas": [1.0, 2.0]},
        {"op": "li", "value": 1},
    ],
    in_box=in_box,
    out_box=out_box,
    pad_shape=(64, 64, 64),
)

# 4. Evaluate a bounding box directly from a remote dataset or cloud URI on the workstation
ref = RemoteArrayRef(
    source="gs://ucl-hip-ct-35a68e99feaae8932b1d44da0358940b/LADAF-2021-17/heart/8.01um_complete-organ_bm18.ome.zarr",
    level=2,
    ops=[{"op": "sato", "sigmas": [1.0, 2.0]}],
)
subvolume = ws.evaluate_box(ref, Box((0, 0, 0), (64, 64, 64)))

# 5. Generate a virtual OME-Zarr URL served by the workstation for Neuroglancer / ngplatform
#    (Any WebWorker `python` ops in the spec are automatically decompiled to native ops!)
zarr_url = ws.build_source_url({
    "source": ref.source,
    "ops": ref.ops,
    "chunk": [64, 64, 64],
})
# -> "zarr3://http://workstation.my-lab.org:8000/<payload>/zarr3"

ws.shutdown()
```

#### B. Workstation HTTP/RPC Endpoints
Every `zarr_proxy` server instance exposes the workstation endpoints out of the box:
- `GET /_workstation/status` — Returns CPU count, cache statistics, active datasets, and registered ops.
- `POST /_workstation/dispatch_op` — Executes a list of `ops` on a base64/zlib-encoded NumPy array payload with optional `in_box`, `out_box`, and `pad_shape`.
- `POST /_workstation/evaluate_box` — Evaluates `Box(start, stop)` at `level` on a dataset or storage URI with optional `ops`.
- `GET /{payload}/zarr3/...` — Streams virtual OME-Zarr v3 metadata and chunks directly to Neuroglancer or `ngplatform` (automatically decompiling `ngplatform` WebWorker `python` ops into native SciPy/scikit-image ops).

---

### 3. Local & Dask Backends (`local`, `dask`)

For in-process array slicing or Dask graph execution across threads/clusters:

```python
import dask.array as da
from chunkmirage.core import Box
from zarr_proxy.backends import BackendConfig, DaskBackend, LocalBackend
from zarr_proxy.sources import DaskSource, build_dask_pyramid

# Local in-process NumPy/SciPy backend
local_backend = LocalBackend()

# Dask backend (supports scheduler="threads", "processes", or distributed address="tcp://...")
dask_backend = DaskBackend(
    BackendConfig(name="dask", options={"scheduler": "threads"})
)

dask_arr = da.random.random((256, 256, 256), chunks=(64, 64, 64))
pyramid = build_dask_pyramid(dask_arr, levels=3, backend=dask_backend)
chunk_data = pyramid[0].read(Box((0, 0, 0), (64, 64, 64)))
```

---

## Quickstart

### 1. Installation

Install dependencies using [Pixi](https://pixi.sh):

```bash
pixi install
```

### 2. Running the Server

Start the virtual compute server:

```bash
pixi run proxy
# or directly:
pixi run python -m zarr_proxy.main --host 0.0.0.0 --port 8000
```

### 3. Interactive Web Studio

Open your browser to:
👉 **`http://localhost:8000`**

- Configure multi-stage pipelines (e.g. Sato $\to$ Li, Median $\to$ Sato, Sobel, Unsharp Mask).
- One-click launch in Neuroglancer with synchronized virtual layers.
- Real-time cache and queue monitoring.

### 4. CLI URL / Payload Generator

Encode a pipeline specification into a URL-safe payload for either a workstation/server (`--backend workstation` / `local`) or `ngplatform`'s browser WebWorkers (`--backend webworker`):

```bash
# For Workstation / Server OME-Zarr streaming:
pixi run python -m zarr_proxy.main encode \
  --src-url "s3://my-bucket/dataset.zarr" \
  --method sato \
  --sigmas 1.0 2.0 \
  --target-dtype float32 \
  --backend workstation \
  --proxy-host "http://localhost:8000" \
  --compress

# For ngplatform Browser WebWorker (Pyodide) execution:
pixi run python -m zarr_proxy.main encode \
  --src-url "s3://my-bucket/dataset.zarr" \
  --method sato \
  --sigmas 1.0 2.0 \
  --target-dtype float32 \
  --backend webworker \
  --proxy-host "http://localhost:5173"
```

Output:
```text
Pipeline Specification JSON:
{
  "source": "s3://my-bucket/dataset.zarr",
  "ops": [
    {"op": "sato", "sigmas": [1.0, 2.0]}
  ]
}

URL-safe Payload:
eJyLrnZMLskv0svNL8hQEAAj...

Neuroglancer Layer Source URL:
zarr3://http://localhost:8000/eJyLrnZMLskv0svNL8hQEAAj.../zarr3
```

---

## Running the Tests

```bash
pixi run test
```
