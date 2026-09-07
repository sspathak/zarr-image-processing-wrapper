"""
Asynchronous unified storage reader supporting S3, HTTP/HTTPS, and local files.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
from typing import Optional
from urllib.parse import urlparse
import fsspec
import httpx


# Shared httpx AsyncClient for HTTP(S) sources
_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
        )
    return _http_client


def join_url(base: str, path: str) -> str:
    """Safely join a base URL and a subpath."""
    clean_path = path.lstrip("/")
    if base.endswith("/"):
        return base + clean_path
    return f"{base}/{clean_path}"


def _read_fsspec_sync(full_url: str) -> Optional[bytes]:
    """Synchronous read via fsspec / s3fs."""
    try:
        # Check if s3 URL and handle anonymous access if standard fails
        storage_options = {}
        if full_url.startswith("s3://"):
            # If AWS credentials aren't set, default to anon=True
            if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")):
                storage_options["anon"] = True

        fs, path = fsspec.core.url_to_fs(full_url, **storage_options)
        if not fs.exists(path):
            return None
        with fs.open(path, "rb") as f:
            return f.read()
    except (FileNotFoundError, KeyError, PermissionError, OSError):
        return None
    except Exception as e:
        # Some S3 NoSuchKey errors are wrapped in Exception
        if "NoSuchKey" in str(e) or "404" in str(e) or "does not exist" in str(e):
            return None
        raise


async def fetch_bytes(base_url: str, subpath: str) -> Optional[bytes]:
    """
    Fetch raw bytes from a remote or local storage location.
    Returns None if the resource does not exist (404/NoSuchKey).
    """
    full_url = join_url(base_url, subpath)
    if full_url.startswith("gs://"):
        full_url = f"https://storage.googleapis.com/{full_url[5:].lstrip('/')}"
    elif full_url.startswith("gcs://"):
        full_url = f"https://storage.googleapis.com/{full_url[6:].lstrip('/')}"

    parsed = urlparse(full_url)

    if parsed.scheme in ("http", "https"):
        client = get_http_client()
        try:
            resp = await client.get(full_url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to fetch {full_url}: {e}") from e

    # For s3://, file://, or raw paths, run fsspec in threadpool to avoid blocking event loop
    return await asyncio.to_thread(_read_fsspec_sync, full_url)
