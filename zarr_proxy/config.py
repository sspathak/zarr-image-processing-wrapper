"""
Configuration models and URL-safe Base64 payload encoding/decoding.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True)
class ProxyConfig:
    """
    Configuration payload passed in the URL path.
    Example:
    {
        "src_url": "s3://my-bucket/dataset.zarr",
        "method": "frangi",
        "params": {"sigmas": [1.0, 2.0], "black_ridges": false},
        "halo": 6,
        "target_dtype": "uint8",
        "compression": "blosc",
        "cname": "zstd",
        "clevel": 5
    }
    """
    src_url: str
    method: str = "identity"
    params: Dict[str, Any] = field(default_factory=dict)
    halo: Optional[Union[int, List[int]]] = None
    target_dtype: str = "uint8"
    normalize: bool = True
    compression: str = "blosc"  # "blosc", "none", or "gzip"
    cname: str = "zstd"         # For blosc: "zstd", "lz4", "zlib"
    clevel: int = 5             # Compression level (0-9)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def config_hash(self) -> str:
        """Produce a stable short hash for cache keying."""
        raw = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def encode_config(config: Union[ProxyConfig, Dict[str, Any]]) -> str:
    """
    Encode a ProxyConfig object or dictionary into a URL-safe Base64 string.
    """
    if isinstance(config, ProxyConfig):
        data = config.to_dict()
    elif isinstance(config, dict):
        data = config
    else:
        raise TypeError(f"Expected ProxyConfig or dict, got {type(config)}")

    json_str = json.dumps(data, separators=(",", ":"))
    encoded = base64.urlsafe_b64encode(json_str.encode("utf-8")).decode("ascii")
    # Strip trailing padding '=' for cleaner URLs
    return encoded.rstrip("=")


@functools.lru_cache(maxsize=2048)
def decode_config(payload: str) -> ProxyConfig:
    """
    Decode a URL-safe Base64 payload string into a ProxyConfig.
    Results are cached in memory for sub-microsecond lookups during concurrent streaming.
    """
    if not payload:
        raise ValueError("Empty configuration payload")

    # Re-pad Base64 if needed
    rem = len(payload) % 4
    if rem:
        padded = payload + ("=" * (4 - rem))
    else:
        padded = payload

    try:
        decoded_bytes = base64.urlsafe_b64decode(padded.encode("ascii"))
        raw_dict = json.loads(decoded_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to decode config payload: {e}") from e

    if not isinstance(raw_dict, dict):
        raise ValueError("Config payload must be a JSON object")

    if "src_url" not in raw_dict:
        raise ValueError("Config payload missing required 'src_url' field")

    src_url = raw_dict["src_url"].strip().rstrip("/")
    if src_url.startswith("zarr://"):
        src_url = src_url[7:].strip().rstrip("/")
    elif src_url.startswith("zarr3://"):
        src_url = src_url[8:].strip().rstrip("/")

    if src_url.startswith("gs://"):
        bucket_and_path = src_url[5:].lstrip("/")
        src_url = f"https://storage.googleapis.com/{bucket_and_path}".rstrip("/")
    elif src_url.startswith("gcs://"):
        bucket_and_path = src_url[6:].lstrip("/")
        src_url = f"https://storage.googleapis.com/{bucket_and_path}".rstrip("/")

    method = str(raw_dict.get("method", "identity")).lower()
    params = raw_dict.get("params", {})
    if not isinstance(params, dict):
        params = {}

    halo = raw_dict.get("halo")
    target_dtype = str(raw_dict.get("target_dtype", "uint8")).lower()
    normalize = bool(raw_dict.get("normalize", True))
    compression = str(raw_dict.get("compression", "blosc")).lower()
    cname = str(raw_dict.get("cname", "zstd")).lower()
    clevel = int(raw_dict.get("clevel", 5))

    return ProxyConfig(
        src_url=src_url,
        method=method,
        params=params,
        halo=halo,
        target_dtype=target_dtype,
        normalize=normalize,
        compression=compression,
        cname=cname,
        clevel=clevel,
    )
