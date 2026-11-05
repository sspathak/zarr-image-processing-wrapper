"""
Starlette server for dynamic Zarr compute pipelines powered by chunkmirage.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from chunkmirage import create_app
from chunkmirage.cache import LRUCache
from chunkmirage.core import Box
from chunkmirage.ops import list_ops
from chunkmirage.server import DatasetRegistry
import zarr_proxy.ops  # noqa: F401 - registers custom domain ops into chunkmirage
from zarr_proxy import __version__
from zarr_proxy.backends import (
    WebWorkerBackend,
    build_ngplatform_url,
    decode_array_payload,
    encode_array_payload,
    evaluate_workstation_box,
    execute_workstation_ops,
    get_own_python_snippets,
    get_webworker_js,
    list_backends,
    to_ngplatform_pipeline,
)
from zarr_proxy.resolver import dynamic_pipeline_resolver, encode_payload
from zarr_proxy.ui import UI_HTML

# Shared LRU cache for all pipeline stages (default 2 GB or from env)
CACHE_BYTES = int(os.environ.get("CACHE_GB", "2")) * (1024 ** 3)
shared_cache = LRUCache(max_bytes=CACHE_BYTES)

registry = DatasetRegistry(cache=shared_cache, resolver=dynamic_pipeline_resolver)

# Shared WebWorkerBackend instance for worker message handling
_webworker_backend = WebWorkerBackend()


async def studio_ui(request: Request) -> Response:
    """Serve the interactive pipeline studio UI."""
    if request.method == "HEAD":
        return Response(status_code=200, media_type="text/html")
    return HTMLResponse(content=UI_HTML)


async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "version": __version__})


async def stats(request: Request) -> JSONResponse:
    """Cache and engine metrics endpoint."""
    cache_info = shared_cache.stats()
    return JSONResponse({
        "status": "ok",
        "version": __version__,
        "cache": cache_info,
        "datasets": registry.names(),
        "backends": list_backends(),
        "dask": {
            "engine": "dask.array",
            "scheduler": "threads",
        },
        "webworker": {
            "engine": "pyodide",
            "default_chunk": _webworker_backend.chunk_shape,
            "templates": sorted(get_own_python_snippets().keys()),
        },
        "workstation": {
            "cpus": os.cpu_count() or 1,
            "cache_max_bytes": CACHE_BYTES,
        },
    })


async def encode_endpoint(request: Request) -> JSONResponse:
    """
    Utility endpoint to encode a pipeline specification into
    a URL-safe Base64 / ngplatform deflate payload.
    """
    try:
        body: Dict[str, Any] = await request.json()
        compress = request.query_params.get("compress", "false").lower() == "true"
        backend_mode = str(
            request.query_params.get("backend") or body.get("backend") or "local"
        ).lower()
        base_url = str(request.base_url).rstrip("/")

        if backend_mode == "webworker":
            ngp_pipeline = to_ngplatform_pipeline(body, compile_ops=True)
            payload = encode_payload(ngp_pipeline, compress=True)
            ngp_url = build_ngplatform_url(body, base_url=f"{base_url}/")
            return JSONResponse({
                "backend": "webworker",
                "payload": payload,
                "pipeline": ngp_pipeline,
                "sample_url": f"{base_url}/virtual/ngp/{payload}/",
                "neuroglancer_source": ngp_url,
            })

        payload = encode_payload(body, compress=compress)
        sample_url = f"{base_url}/{payload}/zarr3"
        return JSONResponse({
            "backend": backend_mode,
            "payload": payload,
            "sample_url": sample_url,
            "neuroglancer_source": f"zarr3://{sample_url}",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# WebWorker Backend Endpoints
# ---------------------------------------------------------------------------


async def webworker_status(request: Request) -> JSONResponse:
    """Return WebWorker backend capabilities and registered Python op templates."""
    snippets = get_own_python_snippets()
    return JSONResponse({
        "status": "ok",
        "backend": "webworker",
        "default_chunk": _webworker_backend.chunk_shape,
        "own_python_templates": snippets,
        "supported_ops": sorted(list_ops().keys()),
    })


async def webworker_compile(request: Request) -> JSONResponse:
    """
    Compile a `zarr_proxy` pipeline specification into an `ngplatform`
    `ComputedPipeline` with self-contained Pyodide `python` ops.
    """
    try:
        body: Dict[str, Any] = await request.json()
        base_url = str(body.get("base_url") or request.base_url)
        chunk = body.get("chunk") or body.get("chunk_shape")
        compile_ops = bool(body.get("compile_ops", True))
        ngp_pipeline = to_ngplatform_pipeline(body, chunk=chunk, compile_ops=compile_ops)
        payload = encode_payload(ngp_pipeline, compress=True)
        ngp_url = build_ngplatform_url(body, base_url=base_url, chunk=chunk, compile_ops=compile_ops)
        return JSONResponse({
            "status": "ok",
            "pipeline": ngp_pipeline,
            "payload": payload,
            "neuroglancer_source": ngp_url,
            "own_python": list(get_own_python_snippets().values()),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def webworker_message(request: Request) -> JSONResponse:
    """
    Execute a `chunkmirage-web` `ToPyWorker` RPC message (`plan`, `compute`, `describe`, `call`).
    Binary `data` buffers in `compute` can be passed as base64 strings and are returned as base64.
    """
    try:
        msg: Dict[str, Any] = await request.json()
        if msg.get("type") == "compute" and isinstance(msg.get("data"), str):
            msg = {**msg, "data": base64.b64decode(msg["data"].encode("ascii"))}
        answer = await run_in_threadpool(_webworker_backend.handle_worker_message, msg)
        if isinstance(answer.get("value"), (bytes, bytearray)):
            answer = {
                **answer,
                "value": base64.b64encode(bytes(answer["value"])).decode("ascii"),
                "encoding": "base64",
            }
        status_code = 400 if "error" in answer else 200
        return JSONResponse(answer, status_code=status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def webworker_script(request: Request) -> Response:
    """Serve the browser Pyodide WebWorker ES module script."""
    js_code = get_webworker_js()
    return Response(content=js_code, media_type="application/javascript")


# ---------------------------------------------------------------------------
# Workstation Backend RPC Endpoints
# ---------------------------------------------------------------------------


async def workstation_status(request: Request) -> JSONResponse:
    """Return Workstation compute node status, hardware concurrency, and cache telemetry."""
    return JSONResponse({
        "status": "ok",
        "backend": "workstation",
        "version": __version__,
        "cpus": os.cpu_count() or 1,
        "cache": shared_cache.stats(),
        "datasets": registry.names(),
        "ops": sorted(list_ops().keys()),
    })


async def workstation_dispatch_op(request: Request) -> JSONResponse:
    """
    RPC endpoint for `WorkstationBackend.dispatch_op` and `WorkstationBackend.compute_pipeline`.
    Executes one or more ops (with optional halo padding/cropping) on an input ndarray block.
    """
    try:
        body: Dict[str, Any] = await request.json()
        block = decode_array_payload(body["array"])
        ops = body.get("ops") or []
        compression = str(body.get("compression", "zlib"))
        voxel_size = body.get("voxel_size") or [1.0, 1.0, 1.0]
        pad_shape = body.get("pad_shape")

        in_box = None
        out_box = None
        if body.get("in_box"):
            in_box = Box(tuple(body["in_box"]["start"]), tuple(body["in_box"]["stop"]))
        if body.get("out_box"):
            out_box = Box(tuple(body["out_box"]["start"]), tuple(body["out_box"]["stop"]))

        result = await run_in_threadpool(
            execute_workstation_ops,
            block,
            ops,
            in_box=in_box,
            out_box=out_box,
            voxel_size=voxel_size,
            pad_shape=pad_shape,
        )
        return JSONResponse({
            "status": "ok",
            "array": encode_array_payload(result, compression=compression),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


async def workstation_evaluate_box(request: Request) -> JSONResponse:
    """
    RPC endpoint for `WorkstationBackend.evaluate_box`.
    Evaluates a bounding box `Box(start, stop)` from a source URI or registered dataset.
    """
    try:
        body: Dict[str, Any] = await request.json()
        source = str(body.get("source") or body.get("dataset") or "")
        if not source:
            return JSONResponse({"error": "Missing required 'source' field"}, status_code=400)

        box = Box(
            tuple(int(x) for x in body["start"]),
            tuple(int(x) for x in body["stop"]),
        )
        level = int(body.get("level", 0))
        ops = body.get("ops")
        select = body.get("select")
        compression = str(body.get("compression", "zlib"))

        result = await run_in_threadpool(
            evaluate_workstation_box,
            source,
            box,
            level=level,
            ops=ops,
            select=select,
            registry=registry,
            cache=shared_cache,
        )
        return JSONResponse({
            "status": "ok",
            "array": encode_array_payload(result, compression=compression),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


extra_routes = [
    Route("/studio", studio_ui, methods=["GET", "HEAD"]),
    Route("/configure", studio_ui, methods=["GET", "HEAD"]),
    Route("/_health", health, methods=["GET"]),
    Route("/healthz", health, methods=["GET"]),
    Route("/_stats", stats, methods=["GET"]),
    Route("/_encode", encode_endpoint, methods=["POST"]),
    Route("/_webworker/status", webworker_status, methods=["GET"]),
    Route("/_webworker/compile", webworker_compile, methods=["POST"]),
    Route("/_webworker/message", webworker_message, methods=["POST"]),
    Route("/_webworker/worker.js", webworker_script, methods=["GET"]),
    Route("/_workstation/status", workstation_status, methods=["GET"]),
    Route("/_workstation/dispatch_op", workstation_dispatch_op, methods=["POST"]),
    Route("/_workstation/evaluate_box", workstation_evaluate_box, methods=["POST"]),
]

# Mount self-hosted Neuroglancer client bundle at /viewer if present
static_dir = Path(__file__).parent / "static" / "neuroglancer"
if static_dir.exists():
    extra_routes.append(
        Mount("/viewer", StaticFiles(directory=str(static_dir), html=True), name="viewer")
    )

# Build the main ASGI app
app = create_app(
    datasets=registry,
    cache=shared_cache,
    extra_routes=extra_routes,
    resolver=dynamic_pipeline_resolver,
    route_plugins=True,
)


# Add browser-friendly middleware: serve Studio UI when visiting / in a browser
class BrowserRootMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/", "") and "text/html" in request.headers.get("accept", ""):
            return HTMLResponse(content=UI_HTML)
        return await call_next(request)


app.add_middleware(BrowserRootMiddleware)
