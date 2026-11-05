"""
Dynamic dataset resolver supporting URL-safe Base64 specs and ngplatform deflate-raw payloads.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, Optional
import zlib
from chunkmirage.pipeline import PipelineSpec

log = logging.getLogger("zarr_proxy.resolver")


def _unpad_base64(s: str) -> str:
    rem = len(s) % 4
    if rem:
        return s + ("=" * (4 - rem))
    return s


def decode_payload(payload: str) -> Dict[str, Any]:
    """
    Decode a URL payload string into a dictionary.
    Supports:
    1. ngplatform / Web API deflate-raw + base64url
    2. Standard URL-safe Base64 JSON
    3. Standard Base64 JSON
    """
    clean = _unpad_base64(payload.strip())
    raw_bytes = base64.urlsafe_b64decode(clean.encode("ascii"))

    # Attempt 1: ngplatform deflate-raw (RFC 1951)
    try:
        decompressed = zlib.decompress(raw_bytes, -zlib.MAX_WBITS)
        return json.loads(decompressed.decode("utf-8"))
    except Exception:
        pass

    # Attempt 2: Standard zlib / gzip compressed
    try:
        decompressed = zlib.decompress(raw_bytes)
        return json.loads(decompressed.decode("utf-8"))
    except Exception:
        pass

    # Attempt 3: Plain uncompressed JSON bytes
    return json.loads(raw_bytes.decode("utf-8"))


def encode_payload(data: Dict[str, Any], compress: bool = False) -> str:
    """
    Encode a dictionary into a URL-safe Base64 string.
    If compress=True, uses deflate-raw matching ngplatform CompressionStream('deflate-raw').
    """
    raw_json = json.dumps(data, separators=(",", ":")).encode("utf-8")
    if compress:
        # Raw deflate stream without zlib header/trailer
        raw_bytes = zlib.compress(raw_json)[2:-4]
    else:
        raw_bytes = raw_json
    encoded = base64.urlsafe_b64encode(raw_bytes).decode("ascii")
    return encoded.rstrip("=")


def convert_legacy_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert legacy single-filter ProxyConfig dictionary into a chunkmirage PipelineSpec.
    """
    src_url = data.get("src_url", "")
    method = data.get("method", "identity")
    params = data.get("params", {})
    target_dtype = data.get("target_dtype")

    ops = []
    if method and method.lower() not in ("identity", "none"):
        op_dict = {"op": method.lower(), **params}
        ops.append(op_dict)

    if target_dtype and target_dtype != "float32":
        if method.lower() in ("frangi", "sato"):
            if target_dtype == "uint8":
                ops.append({"op": "scale", "factor": 255.0})
            elif target_dtype == "uint16":
                ops.append({"op": "scale", "factor": 65535.0})
        ops.append({"op": "cast", "dtype": target_dtype})

    spec: Dict[str, Any] = {"source": src_url, "ops": ops}
    if "chunk_shape" in data:
        spec["chunk_shape"] = data["chunk_shape"]
    elif "chunk" in data:
        spec["chunk_shape"] = data["chunk"]

    return spec


def _normalize_pipeline_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an incoming pipeline dictionary (including `ngplatform` `ComputedPipeline`
    specs with `chunk` or WebWorker `python` ops) into a valid `PipelineSpec` dictionary.
    """
    from zarr_proxy.backends.webworker import decompile_webworker_op

    norm = dict(data)
    # Map ngplatform's `chunk` field to chunkmirage's `chunk_shape`
    if "chunk" in norm and "chunk_shape" not in norm:
        norm["chunk_shape"] = norm.pop("chunk")
    else:
        norm.pop("chunk", None)

    # Strip optional backend routing metadata before validating PipelineSpec
    norm.pop("backend", None)

    if "ops" in norm and isinstance(norm["ops"], list):
        norm["ops"] = [decompile_webworker_op(o) for o in norm["ops"]]

    return norm


def dynamic_pipeline_resolver(name: str) -> Optional[PipelineSpec | Dict[str, Any]]:
    """
    Resolver function plugged into DatasetRegistry.resolver.
    When a dataset is requested by a name that isn't pre-registered,
    checks if the name is an encoded payload (Base64 / deflate-raw),
    decodes it, and returns the PipelineSpec for dynamic serving.
    """
    if len(name) < 4:
        return None

    try:
        data = decode_payload(name)
        if not isinstance(data, dict):
            return None

        # Check if legacy ProxyConfig format
        if "src_url" in data and "ops" not in data:
            spec_dict = convert_legacy_config(data)
            return PipelineSpec.model_validate(spec_dict)

        # Standard PipelineSpec or ngplatform ComputedPipeline format
        if "source" in data:
            spec_dict = _normalize_pipeline_dict(data)
            return PipelineSpec.model_validate(spec_dict)

        return None
    except Exception as e:
        log.debug("Name '%s' not resolvable as pipeline spec: %s", name, e)
        return None
