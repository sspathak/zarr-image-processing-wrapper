from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import Body, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from zarr_proxy import __version__
from zarr_proxy.cache import get_processed_chunk_cache, get_raw_chunk_cache
from zarr_proxy.config import ProxyConfig, decode_config, encode_config
from zarr_proxy.processor import (
    get_or_fetch_metadata,
    process_chunk_request,
    resolve_array_and_chunk,
)
from zarr_proxy.storage import fetch_bytes, join_url
from zarr_proxy.ui import UI_HTML


app = FastAPI(
    title="Zarr v3 Dynamic Processing Proxy",
    description="High-performance HTTP proxy sitting between Neuroglancer and S3 OME-Zarr datasets.",
    version=__version__,
)


@app.middleware("http")
async def add_cors_and_private_network_headers(request: Request, call_next):
    """
    Middleware to inject permissive CORS and Chrome Private Network Access headers.
    """
    if request.method == "OPTIONS":
        response = Response(status_code=200)
    else:
        response = await call_next(request)

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, POST, OPTIONS, PUT"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


# Mount self-hosted Neuroglancer client bundle at /viewer
static_dir = Path(__file__).parent / "static" / "neuroglancer"
if static_dir.exists():
    app.mount("/viewer", StaticFiles(directory=str(static_dir), html=True), name="viewer")


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse, tags=["Studio UI"])
@app.api_route("/ui", methods=["GET", "HEAD"], response_class=HTMLResponse, tags=["Studio UI"])
@app.api_route("/configure", methods=["GET", "HEAD"], response_class=HTMLResponse, tags=["Studio UI"])
async def studio_ui(request: Request):
    """
    Interactive web UI studio for configuring filters and opening in Neuroglancer with 1-click.
    """
    if request.method == "HEAD":
        return Response(status_code=status.HTTP_200_OK, media_type="text/html")
    return HTMLResponse(content=UI_HTML)


@app.get("/_health", tags=["Monitoring"])
@app.get("/healthz", tags=["Monitoring"])
async def health():
    return {"status": "ok", "version": __version__}


@app.get("/_stats", tags=["Monitoring"])
async def cache_stats():
    raw_cache = get_raw_chunk_cache()
    proc_cache = get_processed_chunk_cache()
    return {
        "raw_chunk_cache": raw_cache.stats(),
        "processed_chunk_cache": proc_cache.stats(),
    }


@app.post("/_encode", tags=["Utility"])
async def encode_url_payload(config: Dict[str, Any] = Body(...)):
    """
    Utility endpoint to generate a URL-safe Base64 config payload
    from a JSON body for convenient Neuroglancer URL configuration.
    """
    try:
        cfg = ProxyConfig(
            src_url=config["src_url"],
            method=config.get("method", "frangi"),
            params=config.get("params", {}),
            halo=config.get("halo"),
            target_dtype=config.get("target_dtype", "uint8"),
            normalize=config.get("normalize", True),
            compression=config.get("compression", "blosc"),
            cname=config.get("cname", "zstd"),
            clevel=config.get("clevel", 5),
        )
        payload = encode_config(cfg)
        return {
            "payload": payload,
            "sample_url": f"http://localhost:8000/{payload}/zarr.json",
            "neuroglancer_source": f"zarr3://http://localhost:8000/{payload}",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid config: {e}",
        )


@app.api_route(
    "/{config_payload}/{zarr_path:path}",
    methods=["GET", "HEAD", "OPTIONS"],
    tags=["Zarr Proxy"],
)
async def proxy_zarr(
    config_payload: str,
    zarr_path: str,
    request: Request,
):
    if request.method == "OPTIONS":
        return Response(status_code=status.HTTP_200_OK)
    """
    Dynamic routing endpoint intercepting Neuroglancer requests:
    - Decodes config_payload to determine src_url, method, params.
    - If requesting zarr.json: serves falsified Zarr v3 metadata.
    - If requesting chunk (e.g. c/0/0/0): fetches context, runs filter, and returns compressed bytes.
    """
    # 1. Decode dynamic config payload
    try:
        config = decode_config(config_payload)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid config payload: {e}",
        )

    clean_path = zarr_path.strip("/")

    # Root directory or trailing slash probing
    if not clean_path:
        return JSONResponse(
            content={"zarr_format": 3, "node_type": "group"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "public, max-age=300",
            },
        )

    # Cleanly 404 on known metadata probes from other formats (Zarr v2, OCDBT, etc.)
    base_name = os.path.basename(clean_path)
    if base_name in {
        ".zattrs",
        ".zgroup",
        ".zarray",
        "manifest.ocdbt",
        "attributes.json",
        "info",
        "ZZZZZZZZ.json",
    } or clean_path.startswith("refs/"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resource {clean_path} not found",
        )

    # 2. Metadata Request (zarr.json at root or sub-array)
    if clean_path.endswith("zarr.json") or clean_path == "zarr.json":
        array_subpath = clean_path[:-9].rstrip("/") if clean_path.endswith("/zarr.json") else ""
        try:
            # Check if this node is an array or a group
            raw_meta_bytes = await fetch_bytes(
                config.src_url,
                join_url(array_subpath, "zarr.json") if array_subpath else "zarr.json"
            )
            if raw_meta_bytes is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Metadata {clean_path} not found at source",
                )

            parsed_raw = json.loads(raw_meta_bytes.decode("utf-8"))
            if parsed_raw.get("node_type") == "group":
                # Return group metadata directly
                if request.method == "HEAD":
                    return Response(
                        status_code=status.HTTP_200_OK,
                        media_type="application/json",
                        headers={"Access-Control-Allow-Origin": "*"},
                    )
                return JSONResponse(
                    content=parsed_raw,
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Cache-Control": "public, max-age=300",
                    },
                )

            # Array node: serve falsified metadata
            falsified_meta, _, _ = await get_or_fetch_metadata(config, array_subpath)
            if request.method == "HEAD":
                return Response(
                    status_code=status.HTTP_200_OK,
                    media_type="application/json",
                    headers={"Access-Control-Allow-Origin": "*"},
                )
            return JSONResponse(
                content=falsified_meta,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "public, max-age=300",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error serving metadata: {e}",
            )

    # 3. Chunk Request
    try:
        compressed_chunk = await process_chunk_request(config, clean_path)

        headers = {
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600",
            "Content-Length": str(len(compressed_chunk)),
            "Content-Type": "application/octet-stream",
        }

        if request.method == "HEAD":
            return Response(
                status_code=status.HTTP_200_OK,
                media_type="application/octet-stream",
                headers=headers,
            )

        return Response(
            content=compressed_chunk,
            status_code=status.HTTP_200_OK,
            media_type="application/octet-stream",
            headers=headers,
        )

    except (FileNotFoundError, ValueError, KeyError) as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        msg = str(e)
        if "Not a Zarr v3 array node" in msg or "not found" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=msg,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chunk: {e}",
        )

