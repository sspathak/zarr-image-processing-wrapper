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
   - **LocalBackend**: Multicore CPU execution using NumPy and SciPy.
   - **DaskBackend**: Lazy computation graph execution across threads, processes, or Dask Distributed clusters.
   - **Extensible Hooks**: Documented extension points for `WorkstationBackend` (remote RPC), `CubedBackend` (serverless bounded memory), and `NumbaBackend` (JIT compilation).

5. **ngplatform & Pyodide Integration**:
   - Natively decodes the `deflate-raw + base64url` pipeline payload format used by [ngplatform](https://github.com/JaneliaSciComp/ngplatform).
   - The same Python op code (`Frangi`, `Sato`, `Otsu`) runs identically in native Python on the server and inside in-browser **Pyodide** WebAssembly workers.

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

Encode a pipeline specification into a URL-safe payload:

```bash
pixi run python -m zarr_proxy.main encode \
  --src-url "s3://my-bucket/dataset.zarr" \
  --method sato \
  --sigmas 1.0 2.0 \
  --target-dtype float32 \
  --compress
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
