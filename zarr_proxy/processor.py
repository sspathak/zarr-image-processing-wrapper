"""
Zarr v3 Chunk Processing Engine:
Dask-powered lazy array representation, transparent chunk caching,
on-demand localized graph execution, volumetric filtering, and encoding.
Supports all Zarr v3 codecs (including sharding_indexed, blosc, zstd, gzip).
"""

from __future__ import annotations

import asyncio
import itertools
import os
from typing import Any, Dict, List, Optional, Tuple
import dask
import dask.array as da
import numpy as np
import zarr

from zarr_proxy.cache import get_processed_chunk_cache, get_raw_chunk_cache
from zarr_proxy.codecs_util import encode_chunk_bytes
from zarr_proxy.config import ProxyConfig
from zarr_proxy.filters import apply_filter, calculate_halo
from zarr_proxy.metadata import extract_array_info, falsify_zarr_metadata
from zarr_proxy.storage import fetch_bytes, join_url


# In-memory metadata cache to prevent repeated remote fetches of zarr.json
# key: (src_url, array_subpath, config_hash) -> (falsified_meta, original_meta, array_info)
_meta_cache: Dict[Tuple[str, str, str], Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = {}
_meta_lock = asyncio.Lock()

# Cached zarr.Array objects for high-performance subvolume slicing
_zarr_array_cache: Dict[Tuple[str, str], zarr.Array] = {}
_zarr_array_lock = asyncio.Lock()

# Cached Dask Array objects wrapping CachedZarrArray
_dask_array_cache: Dict[Tuple[str, str], da.Array] = {}
_dask_array_lock = asyncio.Lock()


def resolve_array_and_chunk(zarr_path: str) -> Tuple[str, str]:
    """
    Split a request path like '0/c/0/1/2' or 'c/0/1/2' into
    (array_subpath, chunk_key).
    """
    clean = zarr_path.strip("/")
    if "/c/" in clean:
        subpath, rest = clean.split("/c/", 1)
        return subpath, f"c/{rest}"
    if clean.startswith("c/"):
        return "", clean
    if "/c." in clean:
        subpath, rest = clean.split("/c.", 1)
        return subpath, f"c.{rest}"
    if clean.startswith("c."):
        return "", clean

    # Fallback for v2-style chunk keys (e.g., "0/1/2" or "s0/0/1/2")
    parts = clean.split("/")
    if len(parts) > 1 and not parts[0].isdigit():
        return parts[0], "/".join(parts[1:])
    return "", clean


def parse_chunk_key(
    chunk_key: str,
    key_encoding_name: str = "default",
    separator: str = "/",
) -> Tuple[int, ...]:
    """
    Parse a chunk key string into integer coordinate indices.
    Examples:
        'c/0/1/2' -> (0, 1, 2)
        'c/0/0/0' -> (0, 0, 0)
        '0/1/2'   -> (0, 1, 2)
        'c.1.2.3' -> (1, 2, 3)
    """
    s = chunk_key
    if s.startswith("c/") or s.startswith("c."):
        s = s[2:]

    # Use separator or fallback to '/' and '.'
    if separator in s:
        parts = s.split(separator)
    elif "/" in s:
        parts = s.split("/")
    elif "." in s:
        parts = s.split(".")
    else:
        parts = [s]

    return tuple(int(p) for p in parts)


def format_chunk_key(
    coords: Tuple[int, ...],
    key_encoding_name: str = "default",
    separator: str = "/",
) -> str:
    """
    Format coordinate indices into a Zarr chunk key.
    """
    coords_str = separator.join(str(c) for c in coords)
    if key_encoding_name == "default":
        return f"c{separator}{coords_str}"
    return coords_str


def calculate_bounding_boxes(
    target_coords: Tuple[int, ...],
    shape: List[int],
    chunk_shape: List[int],
    halo: List[int],
) -> Dict[str, Any]:
    """
    Calculate grid coordinates of adjacent chunks, spatial slices,
    and boundary padding requirements.
    """
    ndim = len(shape)
    c_min = []
    c_max = []
    pad_left = []
    pad_right = []
    target_start = []
    target_end = []
    valid_start = []
    valid_end = []

    for i in range(ndim):
        c_i = target_coords[i]
        d_i = shape[i]
        s_i = chunk_shape[i]
        h_i = halo[i] if i < len(halo) else 0

        t_start = c_i * s_i
        t_end = min((c_i + 1) * s_i, d_i)
        target_start.append(t_start)
        target_end.append(t_end)

        desired_start = t_start - h_i
        desired_end = t_end + h_i

        p_left = max(0, -desired_start)
        p_right = max(0, desired_end - d_i)
        pad_left.append(p_left)
        pad_right.append(p_right)

        v_start = max(0, desired_start)
        v_end = min(d_i, desired_end)
        valid_start.append(v_start)
        valid_end.append(v_end)

        chunk_idx_min = max(0, v_start // s_i)
        chunk_idx_max = max(chunk_idx_min, (v_end - 1) // s_i)
        c_min.append(chunk_idx_min)
        c_max.append(chunk_idx_max)

    return {
        "c_min": c_min,
        "c_max": c_max,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "target_start": target_start,
        "target_end": target_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
    }


async def get_or_fetch_metadata(
    config: ProxyConfig,
    array_subpath: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Fetch, parse, falsify, and cache Zarr v3 metadata for an array node.
    """
    cache_key = (config.src_url, array_subpath, config.config_hash())
    if cache_key in _meta_cache:
        return _meta_cache[cache_key]

    async with _meta_lock:
        if cache_key in _meta_cache:
            return _meta_cache[cache_key]

        meta_subpath = join_url(array_subpath, "zarr.json") if array_subpath else "zarr.json"
        raw_meta = await fetch_bytes(config.src_url, meta_subpath)
        if raw_meta is None:
            raise FileNotFoundError(f"Metadata not found at {config.src_url}/{meta_subpath}")

        falsified_meta, original_meta = falsify_zarr_metadata(raw_meta, config)
        array_info = extract_array_info(original_meta)
        result = (falsified_meta, original_meta, array_info)
        _meta_cache[cache_key] = result
        return result


def _open_zarr_array_sync(src_url: str, array_subpath: str) -> zarr.Array:
    """
    Synchronously open a Zarr v3 array with automated codec & sharding handling.
    """
    full_url = join_url(src_url, array_subpath) if array_subpath else src_url
    if full_url.startswith("gs://"):
        full_url = f"https://storage.googleapis.com/{full_url[5:].lstrip('/')}"
    elif full_url.startswith("gcs://"):
        full_url = f"https://storage.googleapis.com/{full_url[6:].lstrip('/')}"
    if full_url.startswith("s3://") and not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")):
        return zarr.open_array(full_url, mode="r", storage_options={"anon": True})
    return zarr.open_array(full_url, mode="r")


async def get_zarr_array(src_url: str, array_subpath: str) -> zarr.Array:
    """
    Retrieve or initialize the cached zarr.Array instance.
    """
    key = (src_url, array_subpath)
    if key in _zarr_array_cache:
        return _zarr_array_cache[key]
    async with _zarr_array_lock:
        if key in _zarr_array_cache:
            return _zarr_array_cache[key]
        arr = await asyncio.to_thread(_open_zarr_array_sync, src_url, array_subpath)
        _zarr_array_cache[key] = arr
        return arr


class CachedZarrArray:
    """
    Adapter wrapping a zarr.Array so that chunk reads initiated by Dask
    are transparently cached in and read from the thread-safe, byte-bounded RawChunkCache.
    """

    def __init__(self, zarr_arr: zarr.Array, src_url: str, array_subpath: str = ""):
        self.zarr_arr = zarr_arr
        self.src_url = src_url
        self.array_subpath = array_subpath
        self.raw_cache = get_raw_chunk_cache()
        self.shape = zarr_arr.shape
        self.dtype = zarr_arr.dtype
        self.ndim = zarr_arr.ndim

    def __getitem__(self, item: Any) -> np.ndarray:
        if not isinstance(item, tuple):
            item = (item,)
        normalized = []
        for i, s in enumerate(item):
            d = self.shape[i] if i < len(self.shape) else 0
            if isinstance(s, slice):
                start = 0 if s.start is None else s.start
                stop = d if s.stop is None else s.stop
                normalized.append((start, stop))
            else:
                normalized.append((int(s), int(s) + 1))

        cache_key = (self.src_url, self.array_subpath, tuple(normalized))
        cached = self.raw_cache.get(cache_key)
        if cached is not None:
            return cached

        data = np.asarray(self.zarr_arr[item])
        self.raw_cache.put(cache_key, data)
        return data


async def get_dask_array(src_url: str, array_subpath: str = "") -> da.Array:
    """
    Retrieve or initialize a lazy Dask array for the specified Zarr dataset.
    Backed by CachedZarrArray so that all chunk reads pass through the RawChunkCache.
    """
    key = (src_url, array_subpath)
    if key in _dask_array_cache:
        return _dask_array_cache[key]
    async with _dask_array_lock:
        if key in _dask_array_cache:
            return _dask_array_cache[key]
        zarr_arr = await get_zarr_array(src_url, array_subpath)
        cached_wrapper = CachedZarrArray(zarr_arr, src_url, array_subpath)
        darr = da.from_array(cached_wrapper, chunks=zarr_arr.chunks)
        _dask_array_cache[key] = darr
        return darr


async def get_filtered_dask_array(
    config: ProxyConfig,
    array_subpath: str = "",
) -> da.Array:
    """
    Construct a lazy Dask array representing the entire filtered dataset.
    Applies spatial overlap (halo) handling and volumetric filtering via Dask map_overlap/map_blocks.
    """
    darr = await get_dask_array(config.src_url, array_subpath)
    from zarr_proxy.filters import apply_filter_dask
    return apply_filter_dask(
        darr=darr,
        method=config.method,
        params=config.params,
        target_dtype=config.target_dtype,
        normalize=config.normalize,
        halo=config.halo,
    )


async def fetch_valid_subvolume(
    src_url: str,
    array_subpath: str,
    valid_slices: Tuple[slice, ...],
) -> np.ndarray:
    """
    Fetch a subvolume from the Dask-backed Zarr array, utilizing RawChunkCache.
    """
    darr = await get_dask_array(src_url, array_subpath)
    sub = darr[valid_slices]
    return await asyncio.to_thread(sub.compute)


def _pad_block(block: np.ndarray, pad_widths: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    """
    Pad outer volume boundaries using reflect mode with edge fallback.
    """
    if not any(p[0] > 0 or p[1] > 0 for p in pad_widths):
        return block.astype(np.float32, copy=False)
    b_float = block.astype(np.float32, copy=False)
    try:
        return np.pad(b_float, pad_widths, mode="reflect")
    except ValueError:
        return np.pad(b_float, pad_widths, mode="edge")


def _filter_block(
    block: np.ndarray,
    method: str,
    params: Dict[str, Any],
    target_dtype: str,
    normalize: bool,
) -> np.ndarray:
    """
    Apply volumetric filter on block in worker thread.
    Handles 2D, 3D, and leading non-spatial dimensions.
    """
    if block.ndim <= 3:
        return apply_filter(
            volume=block,
            method=method,
            params=params,
            target_dtype=target_dtype,
            normalize=normalize,
        )
    else:
        out = np.empty(block.shape, dtype=np.dtype(target_dtype))
        leading_shape = block.shape[:-3]
        for leading_idx in itertools.product(*(range(s) for s in leading_shape)):
            out[leading_idx] = apply_filter(
                volume=block[leading_idx],
                method=method,
                params=params,
                target_dtype=target_dtype,
                normalize=normalize,
            )
        return out


def _embed_block(block: np.ndarray, chunk_shape: Tuple[int, ...], target_dtype: str) -> np.ndarray:
    """
    Embed a cropped edge chunk in the full regular chunk shape per Zarr v3 specification.
    """
    if tuple(block.shape) == tuple(chunk_shape):
        return block
    full_chunk = np.zeros(chunk_shape, dtype=np.dtype(target_dtype))
    embed_slices = tuple(slice(0, s) for s in block.shape)
    full_chunk[embed_slices] = block
    return full_chunk


def _run_filter_and_crop(
    subvol: np.ndarray,
    halo: List[int],
    target_lengths: List[int],
    chunk_shape: List[int],
    config: ProxyConfig,
) -> np.ndarray:
    """
    Synchronous helper preserved for direct testing and compatibility.
    """
    filtered_subvol = _filter_block(
        subvol,
        method=config.method,
        params=config.params,
        target_dtype=config.target_dtype,
        normalize=config.normalize,
    )

    ndim = subvol.ndim
    crop_slices = []
    for i in range(ndim):
        h_i = halo[i] if i < len(halo) else 0
        t_len = target_lengths[i]
        crop_slices.append(slice(h_i, h_i + t_len))

    cropped = filtered_subvol[tuple(crop_slices)]
    return _embed_block(cropped, tuple(chunk_shape), config.target_dtype)


def build_chunk_dask_graph(
    config: ProxyConfig,
    darr: da.Array,
    target_coords: Tuple[int, ...],
    shape: List[int],
    chunk_shape: List[int],
    halo: List[int],
) -> da.Array:
    """
    Construct a localized lazy Dask graph targeting ONLY the requested chunk region + halo.
    Dask evaluates only the intersecting chunks needed for this specific region on a need basis.
    """
    ndim = len(shape)
    t_start = []
    t_end = []
    v_start = []
    v_end = []
    pad_left = []
    pad_right = []
    target_lengths = []

    for i in range(ndim):
        c_i = target_coords[i]
        d_i = shape[i]
        s_i = chunk_shape[i]
        h_i = halo[i] if i < len(halo) else 0

        start_idx = c_i * s_i
        end_idx = min((c_i + 1) * s_i, d_i)
        t_start.append(start_idx)
        t_end.append(end_idx)
        target_lengths.append(end_idx - start_idx)

        desired_start = start_idx - h_i
        desired_end = end_idx + h_i

        # External padding required at dataset boundaries
        p_left = max(0, -desired_start)
        p_right = max(0, desired_end - d_i)
        pad_left.append(p_left)
        pad_right.append(p_right)

        # Region within dataset boundaries
        valid_s = max(0, desired_start)
        valid_e = min(d_i, desired_end)
        v_start.append(valid_s)
        v_end.append(valid_e)

    valid_slices = tuple(slice(v_start[i], v_end[i]) for i in range(ndim))
    pad_widths = tuple((pad_left[i], pad_right[i]) for i in range(ndim))
    crop_slices = tuple(
        slice(halo[i] if i < len(halo) else 0, (halo[i] if i < len(halo) else 0) + target_lengths[i])
        for i in range(ndim)
    )

    # 1. Lazily slice only the required valid region from Dask array and rechunk to a unified task
    sub_da = darr[valid_slices].rechunk((-1,) * ndim)
    padded_shape = tuple(s + p[0] + p[1] for s, p in zip(sub_da.shape, pad_widths))

    # 2. Lazily pad boundaries
    padded_da = da.map_blocks(
        _pad_block,
        sub_da,
        pad_widths=pad_widths,
        chunks=tuple((s,) for s in padded_shape),
        dtype=np.float32,
    )

    # 3. Lazily apply volumetric filter and normalization/casting
    filt_da = da.map_blocks(
        _filter_block,
        padded_da,
        method=config.method,
        params=config.params,
        target_dtype=config.target_dtype,
        normalize=config.normalize,
        chunks=tuple((s,) for s in padded_shape),
        dtype=np.dtype(config.target_dtype),
    )

    # 4. Crop away the halo
    cropped_da = filt_da[crop_slices]

    # 5. Embed in regular chunk_shape if edge chunk
    if tuple(target_lengths) != tuple(chunk_shape):
        chunk_da = da.map_blocks(
            _embed_block,
            cropped_da,
            chunk_shape=tuple(chunk_shape),
            target_dtype=config.target_dtype,
            chunks=tuple((s,) for s in chunk_shape),
            dtype=np.dtype(config.target_dtype),
        )
    else:
        chunk_da = cropped_da

    return chunk_da


async def process_chunk_request(
    config: ProxyConfig,
    zarr_path: str,
) -> bytes:
    """
    Main entrypoint for chunk requests:
    1. Check ProcessedChunkCache.
    2. Resolve array and chunk coords.
    3. Calculate required halo context.
    4. Construct localized lazy Dask graph targeting ONLY the requested chunk region.
    5. Evaluate Dask graph on-demand in worker thread / scheduler.
    6. Compress using falsified codec and cache.
    """
    processed_cache = get_processed_chunk_cache()
    array_subpath, chunk_key = resolve_array_and_chunk(zarr_path)

    # Processed cache key includes config hash, subpath, and chunk key
    proc_cache_key = (config.config_hash(), array_subpath, chunk_key)
    cached_bytes = processed_cache.get(proc_cache_key)
    if cached_bytes is not None:
        return cached_bytes

    # Fetch array metadata
    falsified_meta, original_meta, array_info = await get_or_fetch_metadata(config, array_subpath)

    shape = array_info["shape"]
    chunk_shape = array_info["chunk_shape"]
    key_encoding = array_info["key_encoding_name"]
    separator = array_info["key_separator"]

    target_coords = parse_chunk_key(chunk_key, key_encoding, separator)

    # Calculate halo for spatial dimensions
    ndim = len(shape)
    spatial_ndim = min(3, ndim)
    spatial_halo = calculate_halo(config.method, config.params, config.halo, ndim=spatial_ndim)

    if ndim > 3:
        halo = [0] * (ndim - 3) + spatial_halo
    else:
        halo = spatial_halo[-ndim:]

    # Get lazy Dask array backed by CachedZarrArray
    darr = await get_dask_array(config.src_url, array_subpath)

    # Construct localized lazy Dask graph targeting ONLY this chunk
    chunk_da = build_chunk_dask_graph(
        config=config,
        darr=darr,
        target_coords=target_coords,
        shape=shape,
        chunk_shape=chunk_shape,
        halo=halo,
    )

    # Execute computation on-demand via Dask scheduler
    computed_chunk = await asyncio.to_thread(chunk_da.compute)

    # Compress using falsified codec
    compressed_bytes = await asyncio.to_thread(
        encode_chunk_bytes,
        computed_chunk,
        compression=config.compression,
        cname=config.cname,
        clevel=config.clevel,
    )

    # Cache into Processed Chunk Cache
    processed_cache.put(proc_cache_key, compressed_bytes)

    return compressed_bytes
