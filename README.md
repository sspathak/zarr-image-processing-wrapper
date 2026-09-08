# Zarr v3 Dynamic Volumetric Filtering HTTP Proxy

A high-performance HTTP proxy server written in Python (FastAPI) that sits between **Neuroglancer** and cloud storage (S3 / HTTP / Local) hosting multi-terabyte **Zarr v3** and **OME-Zarr 0.5** datasets.

The proxy intercepts chunk requests on-the-fly, fetches necessary spatial context (boundary "halos"), applies 3D volumetric filters (Frangi vesselness, Sato tubeness, Gaussian, Otsu thresholding), falsifies Zarr v3 metadata (`zarr.json`) to reflect the output data type and codec pipeline, and streams compressed chunks to Neuroglancer with bounded, thread-safe LRU caching.

---

## Key Features

1. **Dynamic Path-Based Routing:**
   - Designed specifically for Neuroglancer path conventions (`/{config_payload}/{zarr_path:path}`).
   - URL-safe Base64 config encoding carries `src_url`, `method`, filter `params`, `halo`, and codec preferences.
2. **Bounded LRU Memory Caching (Strict Byte Limits):**
   - **Raw Chunk Cache:** Caches decompressed source chunks (default 2 GB).
   - **Processed Chunk Cache:** Caches compressed filtered output chunks (default 1 GB).
   - Thread-safe concurrency using `cachetools.LRUCache` with exact byte-level tracking (`nbytes` / `len`).
3. **Zarr v3 & OME-Zarr 0.5 Metadata Falsification:**
   - Intercepts requests for `zarr.json` (at dataset root and multiscale levels like `s0/zarr.json`).
   - Automatically modifies `data_type` (e.g. `float32` -> `uint8`).
   - Adjusts codec pipeline (`bytes` codec + `blosc` with `typesize: 1` or `gzip`/uncompressed) to match outgoing streams.
4. **Dask-Powered On-Demand Volumetric Pipeline & Halo Stitching:**
   - Evaluates only the regions requested by Neuroglancer/user on a strict need basis using lazy Dask graphs (`dask.array`).
   - Automatically computes required halo padding based on filter kernel support (e.g. $\lceil 3 \times \max(\sigma) \rceil$).
   - Intersecting adjacent chunks are transparently cached through `CachedZarrArray` into `RawChunkCache`.
   - Pads outer volume boundaries with reflection/edge modes, executes filters via Dask, and crops away the halo to eliminate edge artifacts.
5. **CORS & Zero-Config Neuroglancer Integration:**
   - Full CORS headers (`Access-Control-Allow-Origin: *`).
   - Direct support for `zarr3://` source URLs in Neuroglancer.

---

## Tech Stack

* **Environment Manager:** [Pixi](https://pixi.sh)
* **Web Framework:** FastAPI, Uvicorn, Starlette
* **Data & Codecs:** `zarr` (v3+), `numcodecs`, `s3fs`, `fsspec`
* **Parallel & Distributed Computing:** `dask` (v2026+), `dask.array`, `distributed`
* **Computation:** `numpy`, `scikit-image`, `scipy`
* **Caching:** `cachetools` (thread-safe byte-bounded `LRUCache`)

---

## Quickstart

### 1. Installation

Install dependencies using `pixi`:

```bash
pixi install
```

### 2. Running the Proxy Server

Start the server on host `0.0.0.0` port `8000`:

```bash
pixi run proxy
# or directly via python:
pixi run python -m zarr_proxy.main --host 0.0.0.0 --port 8000
```

### 3. Interactive Web UI Studio (Recommended)

Open your browser to the Web Studio:

👉 **`http://128.40.89.2:8000/`** (or `http://128.40.89.2:8000/ui`)

Features:
- **Interactive Inputs & Dropdowns**: Select algorithms (`Frangi`, `Sato`, `Gaussian`, etc.), configure filter $\sigma$ scales, boundary halos, codecs, and target dtypes.
- **1-Click Launch**: Click **"Open in Neuroglancer"** to instantly open the official Neuroglancer viewer with the pre-configured layer state loaded!
- **Instant Copy**: Copy the `zarr3://` layer source URL or the URL-safe Base64 payload with a single click.
- **Live Metadata Preview**: Inspect the proxy's falsified `zarr.json` on the fly.
- **Cache Dashboard**: Real-time monitoring of Raw/Processed cache memory usage and hit rates.

### 4. CLI Configuration Generator

You can also generate URLs directly via CLI:

```bash
pixi run python -m zarr_proxy.main encode \
  --src-url "s3://my-bucket/dataset.zarr" \
  --method frangi \
  --sigmas 1.0 2.0 \
  --target-dtype uint8 \
  --proxy-host "http://128.40.89.2:8000"
```

Output:
```text
Configuration Payload:
eyJzcmNfdXJsIjoiczM6Ly9teS1idWNrZXQvZGF0YXNldC56YXJyIiwibWV0aG9kIjoiZnJhbmdpIiwicGFyYW1zIjp7InNpZ21hcyI6WzEuMCwyLjBdfSwiaGFsbyI6bnVsbCwndGFyZ2V0X2R0eXBlIjoidWludDgiLCJub3JtYWxpemUiOnRydWUsImNvbXByZXNzaW9uIjoiYmxvc2MiLCJjbmFtZSI6InpzdGQiLCJjbGV2ZWwiOjV9

Neuroglancer Layer Source URL:
zarr3://http://128.40.89.2:8000/eyJzcmNfdXJsIjoiczM6Ly9teS1idWNrZXQvZGF0YXNldC56YXJyIiwibWV0aG9kIjoiZnJhbmdpIiwicGFyYW1zIjp7InNpZ21hcyI6WzEuMCwyLjBdfSwiaGFsbyI6bnVsbCwndGFyZ2V0X2R0eXBlIjoidWludDgiLCJub3JtYWxpemUiOnRydWUsImNvbXByZXNzaW9uIjoiYmxvc2MiLCJjbmFtZSI6InpzdGQiLCJjbGV2ZWwiOjV9
```

In Neuroglancer, add a new layer and set the source to the generated `zarr3://...` URL.

---

## Configuration Payload Specification

The `config_payload` in `/{config_payload}/{zarr_path:path}` is a URL-safe Base64-encoded JSON dictionary:

| Field | Type | Default | Description |
|---|---|---|---|
| `src_url` | `str` | *required* | Source Zarr v3 dataset location (`s3://...`, `https://...`, `file://...`). |
| `method` | `str` | `"identity"` | Volumetric filter: `"frangi"`, `"sato"`, `"gaussian"`, `"threshold_otsu"`, `"constant"`, `"identity"`. |
| `params` | `dict` | `{}` | Keyword arguments passed to filter (e.g. `{"sigmas": [1.0, 2.0], "black_ridges": false}`). |
| `halo` | `int` or `list` | `null` | Halo padding radius in voxels. If omitted, automatically computed as $\lceil 3 \times \max(\sigma) \rceil$. |
| `target_dtype` | `str` | `"uint8"` | Output data type served to client (`"uint8"`, `"uint16"`, `"float32"`). |
| `normalize` | `bool` | `true` | Scale filtered float output to dynamic range of target data type. |
| `compression` | `str` | `"blosc"` | Codec compression: `"blosc"`, `"gzip"`, `"zstd"`, or `"none"`. |
| `cname` | `str` | `"zstd"` | Blosc compressor (`"zstd"`, `"lz4"`, `"zlib"`). |
| `clevel` | `int` | `5` | Compression level (0–9). |

---

## Monitoring & Endpoints

* `GET /_health`: Health status and proxy version.
* `GET /_stats`: Real-time cache utilization, item counts, byte usage, and Dask engine status.
* `POST /_encode`: Utility API to generate configuration payloads from JSON.
* `GET /{config_payload}/{zarr_path:path}`: Dynamic Neuroglancer proxy route for metadata and chunks.

---

## Dask Lazy Processing Architecture

1. **On-Demand Localized Compute:**
   Chunk requests trigger localized Dask computation graphs targeting *only* the specific spatial coordinates and halo boundaries needed. Unrequested chunks and distant subvolumes are never evaluated.
2. **Transparent Chunk Caching:**
   All chunk slices read by Dask pass through `CachedZarrArray`, which transparently populates the thread-safe `RawChunkCache`. Adjacent halo reads seamlessly reuse already-loaded chunks.
3. **Full-Volume Dask API:**
   Python clients can directly construct lazy Dask arrays for downstream analysis:
   ```python
   from zarr_proxy.processor import get_filtered_dask_array
   from zarr_proxy.config import ProxyConfig

   cfg = ProxyConfig(src_url="s3://bucket/volume.zarr", method="frangi")
   darr = await get_filtered_dask_array(cfg)  # returns da.Array
   ```
4. **Distributed Cluster Support (Optional):**
   Set `DASK_SCHEDULER_ADDRESS="tcp://scheduler:8786"` in the environment to automatically dispatch chunk filtering tasks across a distributed Dask cluster. By default, tasks execute using Dask's high-performance multi-threaded local scheduler.


---

## Running the Test Suite

Run the automated test suite covering config encoding, thread-safe LRU eviction, metadata falsification, 3D halo boundary continuity, and end-to-end API streaming:

```bash
pixi run test
```