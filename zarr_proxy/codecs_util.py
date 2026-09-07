"""
Zarr v3 chunk codec decoding and encoding utilities using numcodecs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numcodecs
import numpy as np


def decode_chunk_bytes(
    raw_bytes: bytes,
    codecs_spec: List[Dict[str, Any]],
    shape: Tuple[int, ...],
    dtype_str: str,
) -> np.ndarray:
    """
    Decode raw chunk bytes into a numpy array according to the Zarr v3 codec specification.
    """
    buffer = raw_bytes
    endian = "little"

    # In Zarr v3, bytes-to-bytes codecs operate in reverse order for decoding
    for codec_dict in reversed(codecs_spec):
        name = codec_dict.get("name", "")
        cfg = codec_dict.get("configuration", {})

        if name == "blosc":
            cname = cfg.get("cname", "zstd")
            clevel = cfg.get("clevel", 5)
            shuffle_mode = cfg.get("shuffle", "shuffle")
            if shuffle_mode == "bitshuffle":
                shuffle = numcodecs.Blosc.BITSHUFFLE
            elif shuffle_mode == "shuffle":
                shuffle = numcodecs.Blosc.SHUFFLE
            else:
                shuffle = numcodecs.Blosc.NOSHUFFLE
            codec = numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
            buffer = codec.decode(buffer)

        elif name == "zstd":
            level = cfg.get("level", 0)
            codec = numcodecs.Zstd(level=level)
            buffer = codec.decode(buffer)

        elif name == "gzip":
            level = cfg.get("level", 5)
            codec = numcodecs.GZip(level=level)
            buffer = codec.decode(buffer)

        elif name == "bytes":
            endian = cfg.get("endian", "little")

    # Map dtype and endianness
    base_dtype = np.dtype(dtype_str)
    byte_prefix = "<" if endian == "little" else ">"
    target_dt = base_dtype.newbyteorder(byte_prefix)

    arr = np.frombuffer(buffer, dtype=target_dt)
    return arr.reshape(shape)


def encode_chunk_bytes(
    arr: np.ndarray,
    compression: str = "blosc",
    cname: str = "zstd",
    clevel: int = 5,
) -> bytes:
    """
    Encode a numpy array into bytes according to the falsified codec specification.
    """
    # Force little-endian contiguous bytes
    dt = arr.dtype.newbyteorder("<")
    raw_bytes = np.ascontiguousarray(arr, dtype=dt).tobytes()

    comp = compression.lower()
    if comp == "blosc":
        shuffle = numcodecs.Blosc.SHUFFLE
        codec = numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
        return bytes(codec.encode(raw_bytes))
    elif comp == "gzip":
        codec = numcodecs.GZip(level=clevel)
        return bytes(codec.encode(raw_bytes))
    elif comp == "zstd":
        codec = numcodecs.Zstd(level=clevel)
        return bytes(codec.encode(raw_bytes))
    elif comp == "none":
        return raw_bytes
    else:
        # Fallback to Blosc
        codec = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
        return bytes(codec.encode(raw_bytes))
