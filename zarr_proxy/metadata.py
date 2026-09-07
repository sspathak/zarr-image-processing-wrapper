"""
Zarr v3 metadata parsing, validation, and falsification.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple
import numpy as np

from zarr_proxy.config import ProxyConfig


def falsify_zarr_metadata(
    meta_bytes: bytes,
    config: ProxyConfig
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse original Zarr v3 metadata JSON, and produce falsified metadata
    reflecting the target data type and output codec pipeline.

    Returns:
        (falsified_metadata, original_metadata)
    """
    original_meta = json.loads(meta_bytes.decode("utf-8"))

    # If it's a group (e.g. OME-Zarr root), no array falsification is needed
    if original_meta.get("node_type") == "group":
        return original_meta, original_meta

    falsified = copy.deepcopy(original_meta)

    # If the original array used sharding_indexed, update chunk_grid to the true sub-chunk shape
    # so that Neuroglancer requests standard-sized 3D chunks (e.g. 64x64x64) instead of multi-gigabyte shards!
    for codec in original_meta.get("codecs", []):
        if codec.get("name") == "sharding_indexed":
            sub_shape = codec.get("configuration", {}).get("chunk_shape")
            if sub_shape and "chunk_grid" in falsified and "configuration" in falsified["chunk_grid"]:
                falsified["chunk_grid"]["configuration"]["chunk_shape"] = list(sub_shape)

    # 1. Update data_type
    target_dt = config.target_dtype.lower()
    falsified["data_type"] = target_dt

    itemsize = np.dtype(target_dt).itemsize

    # 2. Update codecs
    if config.compression == "blosc":
        falsified["codecs"] = [
            {
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            },
            {
                "name": "blosc",
                "configuration": {
                    "cname": config.cname,
                    "clevel": config.clevel,
                    "shuffle": "shuffle",
                    "typesize": itemsize,
                    "blocksize": 0
                }
            }
        ]
    elif config.compression == "gzip":
        falsified["codecs"] = [
            {
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            },
            {
                "name": "gzip",
                "configuration": {
                    "level": config.clevel
                }
            }
        ]
    elif config.compression == "none":
        falsified["codecs"] = [
            {
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            }
        ]
    else:
        # Default fallback to bytes + blosc
        falsified["codecs"] = [
            {
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            },
            {
                "name": "blosc",
                "configuration": {
                    "cname": "zstd",
                    "clevel": 5,
                    "shuffle": "shuffle",
                    "typesize": itemsize,
                    "blocksize": 0
                }
            }
        ]

    # 3. Update fill_value for the target dtype
    falsified["fill_value"] = 0

    return falsified, original_meta


def extract_array_info(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key shape, chunking, and encoding details from an array zarr.json.
    """
    if meta.get("node_type") != "array":
        raise ValueError("Not a Zarr v3 array node")

    shape = list(meta.get("shape", []))
    chunk_grid = meta.get("chunk_grid", {})
    chunk_shape = list(chunk_grid.get("configuration", {}).get("chunk_shape", shape))

    # If sharding_indexed is present, use sub-chunk shape for processing
    codecs = meta.get("codecs", [])
    for codec in codecs:
        if codec.get("name") == "sharding_indexed":
            sub_shape = codec.get("configuration", {}).get("chunk_shape")
            if sub_shape:
                chunk_shape = list(sub_shape)

    chunk_key_encoding = meta.get("chunk_key_encoding", {})
    key_encoding_name = chunk_key_encoding.get("name", "default")
    key_separator = chunk_key_encoding.get("configuration", {}).get("separator", "/")

    data_type = meta.get("data_type", "float32")
    fill_value = meta.get("fill_value", 0)
    codecs = meta.get("codecs", [])

    return {
        "shape": shape,
        "chunk_shape": chunk_shape,
        "key_encoding_name": key_encoding_name,
        "key_separator": key_separator,
        "data_type": data_type,
        "fill_value": fill_value,
        "codecs": codecs,
    }
