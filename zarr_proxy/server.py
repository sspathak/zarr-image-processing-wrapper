"""
Starlette server for dynamic Zarr compute pipelines powered by chunkmirage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from chunkmirage import create_app
from chunkmirage.cache import LRUCache
from chunkmirage.server import DatasetRegistry
import zarr_proxy.ops  # noqa: F401 - registers custom domain ops into chunkmirage
from zarr_proxy import __version__
from zarr_proxy.resolver import dynamic_pipeline_resolver, encode_payload
from zarr_proxy.ui import UI_HTML

# Shared LRU cache for all pipeline stages (default 2 GB or from env)
CACHE_BYTES = int(os.environ.get("CACHE_GB", "2")) * (1024 ** 3)
shared_cache = LRUCache(max_bytes=CACHE_BYTES)

registry = DatasetRegistry(cache=shared_cache, resolver=dynamic_pipeline_resolver)


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
        "backends": ["local", "dask", "workstation"],
        "dask": {
            "engine": "dask.array",
            "scheduler": "threads",
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
        payload = encode_payload(body, compress=compress)
        base_url = str(request.base_url).rstrip("/")
        sample_url = f"{base_url}/{payload}/zarr3"
        return JSONResponse({
            "payload": payload,
            "sample_url": sample_url,
            "neuroglancer_source": f"zarr3://{sample_url}",
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
