"""
WebWorker / Pyodide compute backend for browser-side and worker-pool execution.

Designed for seamless integration with **ngplatform** (`ctx.workspace.computed` /
`chunkmirage-web` Service Worker + Pyodide WebWorkers) as well as standalone
WebWorker and server-side worker pool execution:

1. **Op Compilation (`compile_op_for_webworker`)**:
   Translates `zarr_proxy` domain ops (`sato`, `frangi`, `meijering`, `sobel`,
   `scharr`, `laplace`, `unsharp_mask`, `median`, `erode`, `dilate`,
   `white_tophat`, `ct_window`, `dynamic_range`, `gamma`, `otsu`, `li`,
   `passthrough`, `identity`) into self-contained `chunkmirage` `python` ops
   (`{"op": "python", "code": ..., "params": ...}`) with `#halo`, `#output`,
   and `#uicontrol` directives so stock `ngplatform` Pyodide WebWorkers can
   plan and execute them without needing server-side plugins installed.

2. **Trusted Python Registration (`get_own_python_snippets`)**:
   Exports the canonical `#halo` templates so an `ngplatform` plugin can
   register them via `ctx.workspace.computed.ownPython(code)` on activation,
   allowing shared URLs to run in any browser without a 403 prompt.

3. **Pipeline & URL Encoding (`to_ngplatform_pipeline`, `build_ngplatform_url`)**:
   Converts `zarr_proxy` / `chunkmirage` `PipelineSpec` objects into
   `ngplatform`'s `ComputedPipeline` (`{source, ops, chunk, select}`) and
   encodes them with `deflate-raw` + `base64url` into
   `zarr3://<base>virtual/ngp/<payload>/`.

4. **Bidirectional Decompilation (`from_ngplatform_pipeline`, `decompile_webworker_op`)**:
   Recognizes compiled `webworker` `python` ops (as well as `ngplatform`'s built-in
   `MEDIAN` and `FRANGI` `python` ops) and converts them back into native
   `zarr_proxy` `Op` specifications when offloading to a Workstation or Local backend.

5. **WebWorker RPC Protocol (`handle_worker_message`, `get_webworker_js`)**:
   Implements the `chunkmirage-web` `ToPyWorker` message protocol (`plan`,
   `compute`, `describe`, `call`) for both in-worker Pyodide execution and
   local worker-pool simulation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from chunkmirage import calls, fused
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops import op_from_spec
from chunkmirage.ops.base import Op
from chunkmirage.pipeline import PipelineSpec
import zarr_proxy.ops  # noqa: F401 - ensure zarr_proxy ops are registered
from zarr_proxy.backends.base import BackendConfig, ComputeBackend
from zarr_proxy.resolver import decode_payload, encode_payload

# Default output chunk shape used by ngplatform's computed image engine
DEFAULT_WEBWORKER_CHUNK: List[int] = [64, 64, 64]

# Native ops shipped in chunkmirage_contrib that pyworker.ts already has built-in
NATIVE_CHUNKMIRAGE_OPS = frozenset({
    "cast",
    "contacts",
    "diff",
    "dog",
    "downsample",
    "gaussian",
    "gradient",
    "hillshade",
    "label",
    "morphology",
    "normalized_difference",
    "python",
    "scale",
    "slope",
    "spots",
    "threshold",
    "uniform",
})

# ---------------------------------------------------------------------------
# Canonical PythonOp templates for ngplatform Pyodide WebWorkers.
# Each template uses #halo, #output, and #uicontrol directives compatible with
# chunkmirage.ops.code.PythonOp and ngplatform's `ownPython()` registration.
# ---------------------------------------------------------------------------

SATO_PYTHON_TEMPLATE = """#halo 6
#output float32 image
#uicontrol float sigma_low slider(min=0.1, max=32.0, default=1.0)
#uicontrol float sigma_high slider(min=0.1, max=32.0, default=2.0)
#uicontrol int num_sigmas slider(min=1, max=16, default=2)
#uicontrol bool black_ridges checkbox(default=false)
#zarr_proxy_op sato
import numpy as np
from skimage.filters import sato

def apply(block, sigma_low, sigma_high, num_sigmas, black_ridges):
    lo, hi = min(float(sigma_low), float(sigma_high)), max(float(sigma_low), float(sigma_high))
    n = max(1, int(num_sigmas))
    sigmas = [lo] if n == 1 or abs(hi - lo) < 1e-9 else [float(x) for x in np.linspace(lo, hi, n)]
    return sato(block.astype(np.float32, copy=False), sigmas=sigmas, black_ridges=bool(black_ridges)).astype(np.float32, copy=False)
"""

FRANGI_PYTHON_TEMPLATE = """#halo 6
#output float32 image
#uicontrol float sigma_low slider(min=0.1, max=32.0, default=1.0)
#uicontrol float sigma_high slider(min=0.1, max=32.0, default=2.0)
#uicontrol int num_sigmas slider(min=1, max=16, default=2)
#uicontrol float alpha slider(min=0.01, max=2.0, default=0.5)
#uicontrol float beta slider(min=0.01, max=2.0, default=0.5)
#uicontrol float gamma_val slider(min=0.0, max=10000.0, default=0.0)
#uicontrol bool black_ridges checkbox(default=false)
#zarr_proxy_op frangi
import numpy as np
from skimage.filters import frangi

def apply(block, sigma_low, sigma_high, num_sigmas, alpha, beta, gamma_val, black_ridges):
    lo, hi = min(float(sigma_low), float(sigma_high)), max(float(sigma_low), float(sigma_high))
    n = max(1, int(num_sigmas))
    sigmas = [lo] if n == 1 or abs(hi - lo) < 1e-9 else [float(x) for x in np.linspace(lo, hi, n)]
    gamma = None if float(gamma_val) <= 0.0 else float(gamma_val)
    return frangi(
        block.astype(np.float32, copy=False),
        sigmas=sigmas,
        alpha=float(alpha),
        beta=float(beta),
        gamma=gamma,
        black_ridges=bool(black_ridges),
    ).astype(np.float32, copy=False)
"""

MEIJERING_PYTHON_TEMPLATE = """#halo 6
#output float32 image
#uicontrol float sigma_low slider(min=0.1, max=32.0, default=1.0)
#uicontrol float sigma_high slider(min=0.1, max=32.0, default=2.0)
#uicontrol int num_sigmas slider(min=1, max=16, default=2)
#uicontrol bool black_ridges checkbox(default=false)
#zarr_proxy_op meijering
import numpy as np
from skimage.filters import meijering

def apply(block, sigma_low, sigma_high, num_sigmas, black_ridges):
    lo, hi = min(float(sigma_low), float(sigma_high)), max(float(sigma_low), float(sigma_high))
    n = max(1, int(num_sigmas))
    sigmas = [lo] if n == 1 or abs(hi - lo) < 1e-9 else [float(x) for x in np.linspace(lo, hi, n)]
    return meijering(
        block.astype(np.float32, copy=False),
        sigmas=sigmas,
        black_ridges=bool(black_ridges),
    ).astype(np.float32, copy=False)
"""

SOBEL_PYTHON_TEMPLATE = """#halo 1
#output float32 image
#zarr_proxy_op sobel
import numpy as np
import scipy.ndimage as ndi

def apply(block):
    arr = block.astype(np.float32, copy=False)
    grads = [ndi.sobel(arr, axis=ax) for ax in range(arr.ndim)]
    return np.sqrt(sum(g ** 2 for g in grads)).astype(np.float32, copy=False)
"""

SCHARR_PYTHON_TEMPLATE = """#halo 1
#output float32 image
#zarr_proxy_op scharr
import numpy as np
import scipy.ndimage as ndi

def apply(block):
    arr = block.astype(np.float32, copy=False)
    if arr.ndim == 3:
        k_diff = np.array([-1, 0, 1], dtype=np.float32)
        k_smooth = np.array([3, 10, 3], dtype=np.float32) / 16.0
        grads = []
        for ax in range(3):
            g = arr
            for o_ax in range(3):
                if o_ax == ax:
                    g = ndi.convolve1d(g, k_diff, axis=o_ax)
                else:
                    g = ndi.convolve1d(g, k_smooth, axis=o_ax)
            grads.append(g)
        return np.sqrt(sum(g ** 2 for g in grads)).astype(np.float32, copy=False)
    grads = [ndi.sobel(arr, axis=ax) for ax in range(arr.ndim)]
    return np.sqrt(sum(g ** 2 for g in grads)).astype(np.float32, copy=False)
"""

LAPLACE_PYTHON_TEMPLATE = """#halo 1
#output float32 image
#zarr_proxy_op laplace
import numpy as np
import scipy.ndimage as ndi

def apply(block):
    arr = block.astype(np.float32, copy=False)
    return ndi.laplace(arr).astype(np.float32, copy=False)
"""

UNSHARP_MASK_PYTHON_TEMPLATE = """#halo 5
#output image
#uicontrol float radius slider(min=0.1, max=20.0, default=1.5)
#uicontrol float amount slider(min=0.0, max=10.0, default=1.5)
#zarr_proxy_op unsharp_mask
import numpy as np
import scipy.ndimage as ndi

def apply(block, radius, amount):
    orig_dtype = block.dtype
    arr = block.astype(np.float32, copy=False)
    blurred = ndi.gaussian_filter(arr, sigma=float(radius))
    sharpened = arr + float(amount) * (arr - blurred)
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        return np.clip(sharpened, info.min, info.max).astype(orig_dtype, copy=False)
    return sharpened.astype(orig_dtype, copy=False)
"""

MEDIAN_PYTHON_TEMPLATE = """#halo 1
#output image
#uicontrol int size slider(min=1, max=15, default=3)
#zarr_proxy_op median
import scipy.ndimage as ndi

def apply(block, size):
    return ndi.median_filter(block, size=int(size))
"""

ERODE_PYTHON_TEMPLATE = """#halo 1
#output image
#uicontrol int radius slider(min=1, max=10, default=1)
#zarr_proxy_op erode
import scipy.ndimage as ndi

def apply(block, radius):
    return ndi.grey_erosion(block, size=2 * int(radius) + 1)
"""

DILATE_PYTHON_TEMPLATE = """#halo 1
#output image
#uicontrol int radius slider(min=1, max=10, default=1)
#zarr_proxy_op dilate
import scipy.ndimage as ndi

def apply(block, radius):
    return ndi.grey_dilation(block, size=2 * int(radius) + 1)
"""

WHITE_TOPHAT_PYTHON_TEMPLATE = """#halo 2
#output image
#uicontrol int radius slider(min=1, max=10, default=2)
#zarr_proxy_op white_tophat
import numpy as np
import scipy.ndimage as ndi

def apply(block, radius):
    size = 2 * int(radius) + 1
    opened = ndi.grey_opening(block, size=size)
    diff = block.astype(np.float32, copy=False) - opened.astype(np.float32, copy=False)
    if np.issubdtype(block.dtype, np.integer):
        info = np.iinfo(block.dtype)
        return np.clip(diff, info.min, info.max).astype(block.dtype, copy=False)
    return diff.astype(block.dtype, copy=False)
"""


def _ct_window_template(target_dtype: str = "uint8") -> str:
    dt = np.dtype(target_dtype).name
    return f"""#halo 0
#output {dt} image
#uicontrol float vmin slider(min=-100000.0, max=100000.0, default=18000.0)
#uicontrol float vmax slider(min=-100000.0, max=100000.0, default=26000.0)
#zarr_proxy_op ct_window:{dt}
import numpy as np

def apply(block, vmin, vmax):
    target_dt = np.dtype("{dt}")
    max_val = 255.0 if target_dt == np.uint8 else (65535.0 if target_dt == np.uint16 else 1.0)
    b = block.astype(np.float32, copy=False)
    diff = max(float(vmax) - float(vmin), 1e-6)
    rescaled = np.clip((b - float(vmin)) / diff, 0.0, 1.0) * max_val
    return rescaled.astype(target_dt, copy=False)
"""


def _dynamic_range_template(target_dtype: str = "uint8") -> str:
    dt = np.dtype(target_dtype).name
    return f"""#halo 0
#output {dt} image
#uicontrol float pmin slider(min=0.0, max=50.0, default=0.0)
#uicontrol float pmax slider(min=50.0, max=100.0, default=100.0)
#zarr_proxy_op dynamic_range:{dt}
import numpy as np

def apply(block, pmin, pmax):
    target_dt = np.dtype("{dt}")
    max_val = 255.0 if target_dt == np.uint8 else (65535.0 if target_dt == np.uint16 else 1.0)
    b = block.astype(np.float32, copy=False)
    if float(pmin) > 0.0 or float(pmax) < 100.0:
        lo, hi = np.percentile(b, (float(pmin), float(pmax)))
    else:
        lo, hi = float(np.min(b)), float(np.max(b))
    diff = float(hi) - float(lo)
    if diff <= 1e-9:
        return np.zeros(block.shape, dtype=target_dt)
    norm = np.clip((b - float(lo)) / diff, 0.0, 1.0)
    return (norm * max_val).astype(target_dt, copy=False)
"""


GAMMA_PYTHON_TEMPLATE = """#halo 0
#output image
#uicontrol float gamma_exp slider(min=0.05, max=5.0, default=0.5)
#zarr_proxy_op gamma
import numpy as np

def apply(block, gamma_exp):
    b = block.astype(np.float32, copy=False)
    lo, hi = float(np.min(b)), float(np.max(b))
    diff = max(hi - lo, 1e-6)
    normalized = np.clip((b - lo) / diff, 0.0, 1.0)
    adjusted = np.power(normalized, float(gamma_exp)) * diff + lo
    if np.issubdtype(block.dtype, np.integer):
        info = np.iinfo(block.dtype)
        return np.clip(adjusted, info.min, info.max).astype(block.dtype, copy=False)
    return adjusted.astype(block.dtype, copy=False)
"""

OTSU_PYTHON_TEMPLATE = """#halo 4
#output uint8 mask
#uicontrol int nbins slider(min=2, max=65536, default=256)
#uicontrol int fg_value slider(min=1, max=255, default=1)
#zarr_proxy_op otsu
import numpy as np
from skimage.filters import threshold_otsu

def apply(block, nbins, fg_value):
    b_min, b_max = np.min(block), np.max(block)
    if b_min == b_max:
        return np.zeros(block.shape, dtype=np.uint8)
    try:
        th = threshold_otsu(block, nbins=int(nbins))
        return ((block >= th) * np.uint8(int(fg_value))).astype(np.uint8)
    except Exception:
        return np.zeros(block.shape, dtype=np.uint8)
"""

LI_PYTHON_TEMPLATE = """#halo 4
#output uint8 mask
#uicontrol int fg_value slider(min=1, max=255, default=1)
#zarr_proxy_op li
import numpy as np
from skimage.filters import threshold_li

def apply(block, fg_value):
    b_min, b_max = np.min(block), np.max(block)
    if b_min == b_max:
        return np.zeros(block.shape, dtype=np.uint8)
    try:
        th = threshold_li(block.astype(np.float32, copy=False))
        return ((block >= th) * np.uint8(int(fg_value))).astype(np.uint8)
    except Exception:
        return np.zeros(block.shape, dtype=np.uint8)
"""

PASSTHROUGH_PYTHON_TEMPLATE = """#halo 0
#output image
#zarr_proxy_op passthrough
def apply(block):
    return block
"""

# Built-in templates from ngplatform/plugins/filters/pipeline.ts for cross-compatibility
NGPLATFORM_FILTERS_MEDIAN = """#halo 1
#output float32 image
#uicontrol int radius slider(min=1, max=16, default=1)
# The median over a cube of (2 radius + 1) voxels a side.
import numpy as np
from scipy.ndimage import median_filter

def apply(block, radius):
    return median_filter(block.astype(np.float32), size=2 * radius + 1, mode="nearest")
"""

NGPLATFORM_FILTERS_FRANGI = """#halo 9
#output float32 image
#uicontrol float sigma_low slider(min=0.5, max=16, default=1)
#uicontrol float sigma_high slider(min=0.5, max=16, default=3)
#uicontrol bool bright checkbox(default=true)
# Frangi's vesselness, over five scales from sigma_low to sigma_high (voxels): how much
# each voxel looks like part of a tube brighter (or darker) than around it, 0 to 1.
import numpy as np
from skimage.filters import frangi

def apply(block, sigma_low, sigma_high, bright):
    sigmas = np.linspace(min(sigma_low, sigma_high), max(sigma_low, sigma_high), 5)
    return frangi(block.astype(np.float32), sigmas=sigmas, black_ridges=not bright).astype(np.float32)
"""

WEBWORKER_PYTHON_TEMPLATES: Dict[str, str] = {
    "sato": SATO_PYTHON_TEMPLATE,
    "frangi": FRANGI_PYTHON_TEMPLATE,
    "meijering": MEIJERING_PYTHON_TEMPLATE,
    "sobel": SOBEL_PYTHON_TEMPLATE,
    "scharr": SCHARR_PYTHON_TEMPLATE,
    "laplace": LAPLACE_PYTHON_TEMPLATE,
    "unsharp_mask": UNSHARP_MASK_PYTHON_TEMPLATE,
    "median": MEDIAN_PYTHON_TEMPLATE,
    "erode": ERODE_PYTHON_TEMPLATE,
    "dilate": DILATE_PYTHON_TEMPLATE,
    "white_tophat": WHITE_TOPHAT_PYTHON_TEMPLATE,
    "ct_window:uint8": _ct_window_template("uint8"),
    "ct_window:uint16": _ct_window_template("uint16"),
    "ct_window:float32": _ct_window_template("float32"),
    "dynamic_range:uint8": _dynamic_range_template("uint8"),
    "dynamic_range:uint16": _dynamic_range_template("uint16"),
    "dynamic_range:float32": _dynamic_range_template("float32"),
    "gamma": GAMMA_PYTHON_TEMPLATE,
    "otsu": OTSU_PYTHON_TEMPLATE,
    "li": LI_PYTHON_TEMPLATE,
    "passthrough": PASSTHROUGH_PYTHON_TEMPLATE,
}


def _with_halo(code: str, halo: int) -> str:
    """Replace the `#halo ...` directive line with `#halo <halo>`, matching ngplatform."""
    h = max(0, int(math.ceil(halo)))
    return re.sub(r"^#halo[ \d]*$", f"#halo {h}", code, count=1, flags=re.MULTILINE)


def _normalize_halo(code: str) -> str:
    """Normalize `#halo <N>` to `#halo` for template identity comparison (`anyHalo` in ngplatform)."""
    return re.sub(r"^#halo[ \d]*$", "#halo", code.strip(), count=1, flags=re.MULTILINE)


def get_own_python_snippets() -> Dict[str, str]:
    """
    Return all canonical Python op snippets used by `zarr_proxy`'s WebWorker backend.
    An `ngplatform` plugin can iterate over `get_own_python_snippets().values()` and
    register each with `ctx.workspace.computed.ownPython(code)` on activation.
    """
    return dict(WEBWORKER_PYTHON_TEMPLATES)


def is_trusted_webworker_python(code: str) -> bool:
    """
    Check if a `python` op's code matches one of `zarr_proxy`'s or `ngplatform`'s
    trusted built-in templates (ignoring the `#halo` width).
    """
    norm = _normalize_halo(code)
    trusted = {_normalize_halo(t) for t in WEBWORKER_PYTHON_TEMPLATES.values()}
    trusted.add(_normalize_halo(NGPLATFORM_FILTERS_MEDIAN))
    trusted.add(_normalize_halo(NGPLATFORM_FILTERS_FRANGI))
    return norm in trusted


def _parse_sigmas(sigmas_val: Any) -> Tuple[float, float, int, float]:
    """Return (sigma_low, sigma_high, num_sigmas, max_sigma) from a float or list of floats."""
    if isinstance(sigmas_val, (int, float)):
        s = float(sigmas_val)
        return s, s, 1, s
    if isinstance(sigmas_val, (list, tuple)) and len(sigmas_val) > 0:
        vals = [float(x) for x in sigmas_val]
        return min(vals), max(vals), len(vals), max(vals)
    return 1.0, 2.0, 2, 2.0


def compile_op_for_webworker(op_spec: Union[Dict[str, Any], Op]) -> Dict[str, Any]:
    """
    Compile a single `zarr_proxy` or `chunkmirage` op into an `ngplatform` WebWorker-compatible
    op specification:
    - Native `chunkmirage_contrib` ops (`gaussian`, `uniform`, `dog`, `threshold`, `morphology`,
      `cast`, `scale`, `python`, etc.) are preserved directly.
    - Custom `zarr_proxy` ops (`sato`, `frangi`, `meijering`, `sobel`, `scharr`, `laplace`,
      `unsharp_mask`, `median`, `erode`, `dilate`, `white_tophat`, `ct_window`, `dynamic_range`,
      `gamma`, `otsu`, `li`, `passthrough`, `identity`) are compiled into self-contained
      `{"op": "python", "code": ..., "params": ...}` specs with computed `#halo`.
    """
    if isinstance(op_spec, Op):
        spec = op_spec.spec()
    else:
        spec = dict(op_spec)

    op_name = str(spec.get("op", "")).lower()
    if op_name in NATIVE_CHUNKMIRAGE_OPS:
        return spec

    if op_name == "sato":
        s_lo, s_hi, n_s, max_s = _parse_sigmas(spec.get("sigmas", [1.0, 2.0]))
        halo = math.ceil(3.0 * max_s)
        return {
            "op": "python",
            "code": _with_halo(SATO_PYTHON_TEMPLATE, halo),
            "params": {
                "sigma_low": s_lo,
                "sigma_high": s_hi,
                "num_sigmas": n_s,
                "black_ridges": bool(spec.get("black_ridges", False)),
            },
        }

    if op_name == "frangi":
        s_lo, s_hi, n_s, max_s = _parse_sigmas(spec.get("sigmas", [1.0, 2.0]))
        halo = math.ceil(3.0 * max_s)
        gamma = spec.get("gamma")
        return {
            "op": "python",
            "code": _with_halo(FRANGI_PYTHON_TEMPLATE, halo),
            "params": {
                "sigma_low": s_lo,
                "sigma_high": s_hi,
                "num_sigmas": n_s,
                "alpha": float(spec.get("alpha", 0.5)),
                "beta": float(spec.get("beta", 0.5)),
                "gamma_val": 0.0 if gamma is None else float(gamma),
                "black_ridges": bool(spec.get("black_ridges", False)),
            },
        }

    if op_name == "meijering":
        s_lo, s_hi, n_s, max_s = _parse_sigmas(spec.get("sigmas", [1.0, 2.0]))
        halo = math.ceil(3.0 * max_s)
        return {
            "op": "python",
            "code": _with_halo(MEIJERING_PYTHON_TEMPLATE, halo),
            "params": {
                "sigma_low": s_lo,
                "sigma_high": s_hi,
                "num_sigmas": n_s,
                "black_ridges": bool(spec.get("black_ridges", False)),
            },
        }

    if op_name == "sobel":
        return {"op": "python", "code": SOBEL_PYTHON_TEMPLATE, "params": {}}

    if op_name == "scharr":
        return {"op": "python", "code": SCHARR_PYTHON_TEMPLATE, "params": {}}

    if op_name == "laplace":
        return {"op": "python", "code": LAPLACE_PYTHON_TEMPLATE, "params": {}}

    if op_name == "unsharp_mask":
        radius = float(spec.get("radius", 1.5))
        amount = float(spec.get("amount", 1.5))
        halo = math.ceil(3.0 * radius)
        return {
            "op": "python",
            "code": _with_halo(UNSHARP_MASK_PYTHON_TEMPLATE, halo),
            "params": {"radius": radius, "amount": amount},
        }

    if op_name == "median":
        size = int(spec.get("size", 3))
        halo = size // 2
        return {
            "op": "python",
            "code": _with_halo(MEDIAN_PYTHON_TEMPLATE, halo),
            "params": {"size": size},
        }

    if op_name == "erode":
        radius = int(spec.get("radius", 1))
        return {
            "op": "python",
            "code": _with_halo(ERODE_PYTHON_TEMPLATE, radius),
            "params": {"radius": radius},
        }

    if op_name == "dilate":
        radius = int(spec.get("radius", 1))
        return {
            "op": "python",
            "code": _with_halo(DILATE_PYTHON_TEMPLATE, radius),
            "params": {"radius": radius},
        }

    if op_name == "white_tophat":
        radius = int(spec.get("radius", 2))
        return {
            "op": "python",
            "code": _with_halo(WHITE_TOPHAT_PYTHON_TEMPLATE, radius),
            "params": {"radius": radius},
        }

    if op_name == "ct_window":
        target_dtype = np.dtype(spec.get("target_dtype", "uint8")).name
        template = WEBWORKER_PYTHON_TEMPLATES.get(
            f"ct_window:{target_dtype}", _ct_window_template(target_dtype)
        )
        return {
            "op": "python",
            "code": template,
            "params": {
                "vmin": float(spec.get("vmin", 18000.0)),
                "vmax": float(spec.get("vmax", 26000.0)),
            },
        }

    if op_name == "dynamic_range":
        target_dtype = np.dtype(spec.get("target_dtype", "uint8")).name
        template = WEBWORKER_PYTHON_TEMPLATES.get(
            f"dynamic_range:{target_dtype}", _dynamic_range_template(target_dtype)
        )
        return {
            "op": "python",
            "code": template,
            "params": {
                "pmin": float(spec.get("pmin", 0.0)),
                "pmax": float(spec.get("pmax", 100.0)),
            },
        }

    if op_name == "gamma":
        return {
            "op": "python",
            "code": GAMMA_PYTHON_TEMPLATE,
            "params": {"gamma_exp": float(spec.get("gamma", 1.0))},
        }

    if op_name == "otsu":
        return {
            "op": "python",
            "code": OTSU_PYTHON_TEMPLATE,
            "params": {
                "nbins": int(spec.get("nbins", 256)),
                "fg_value": int(spec.get("value", 1)),
            },
        }

    if op_name == "li":
        return {
            "op": "python",
            "code": LI_PYTHON_TEMPLATE,
            "params": {"fg_value": int(spec.get("value", 1))},
        }

    if op_name in ("passthrough", "identity"):
        return {"op": "python", "code": PASSTHROUGH_PYTHON_TEMPLATE, "params": {}}

    raise ValueError(f"Unsupported op '{op_name}' for WebWorker compilation")


def decompile_webworker_op(op_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a WebWorker `python` op (either produced by `compile_op_for_webworker`
    or by `ngplatform`'s built-in `MEDIAN` / `FRANGI` templates) back into a native
    `zarr_proxy` op dictionary. Non-matching or user-authored `python` ops are
    returned unchanged.
    """
    spec = dict(op_spec)
    if spec.get("op") != "python" or not isinstance(spec.get("code"), str):
        return spec

    code = spec["code"]
    params = spec.get("params") or {}

    # Check explicit `#zarr_proxy_op <tag>` marker
    m = re.search(r"^#zarr_proxy_op\s+([a-z0-9_:]+)\s*$", code, flags=re.MULTILINE)
    if m:
        tag = m.group(1)
        if tag in ("sato", "frangi", "meijering"):
            lo = float(params.get("sigma_low", 1.0))
            hi = float(params.get("sigma_high", 2.0))
            n = max(1, int(params.get("num_sigmas", 2)))
            sigmas = [lo] if n == 1 or abs(hi - lo) < 1e-9 else [float(x) for x in np.linspace(lo, hi, n)]
            out: Dict[str, Any] = {
                "op": tag,
                "sigmas": sigmas,
                "black_ridges": bool(params.get("black_ridges", False)),
            }
            if tag == "frangi":
                out["alpha"] = float(params.get("alpha", 0.5))
                out["beta"] = float(params.get("beta", 0.5))
                gamma_val = float(params.get("gamma_val", 0.0))
                if gamma_val > 0.0:
                    out["gamma"] = gamma_val
            return out
        if tag in ("sobel", "scharr", "laplace", "passthrough"):
            return {"op": tag}
        if tag == "unsharp_mask":
            return {
                "op": "unsharp_mask",
                "radius": float(params.get("radius", 1.5)),
                "amount": float(params.get("amount", 1.5)),
            }
        if tag == "median":
            return {"op": "median", "size": int(params.get("size", 3))}
        if tag in ("erode", "dilate", "white_tophat"):
            default_r = 2 if tag == "white_tophat" else 1
            return {"op": tag, "radius": int(params.get("radius", default_r))}
        if tag.startswith("ct_window:"):
            dt = tag.split(":", 1)[1]
            return {
                "op": "ct_window",
                "vmin": float(params.get("vmin", 18000.0)),
                "vmax": float(params.get("vmax", 26000.0)),
                "target_dtype": dt,
            }
        if tag.startswith("dynamic_range:"):
            dt = tag.split(":", 1)[1]
            return {
                "op": "dynamic_range",
                "pmin": float(params.get("pmin", 0.0)),
                "pmax": float(params.get("pmax", 100.0)),
                "target_dtype": dt,
            }
        if tag == "gamma":
            return {"op": "gamma", "gamma": float(params.get("gamma_exp", 1.0))}
        if tag == "otsu":
            return {
                "op": "otsu",
                "nbins": int(params.get("nbins", 256)),
                "value": int(params.get("fg_value", 1)),
            }
        if tag == "li":
            return {"op": "li", "value": int(params.get("fg_value", 1))}

    # Also recognize ngplatform's built-in MEDIAN and FRANGI templates from plugins/filters/pipeline.ts
    norm = _normalize_halo(code)
    if norm == _normalize_halo(NGPLATFORM_FILTERS_MEDIAN):
        radius = max(1, int(round(float(params.get("radius", 1)))))
        return {"op": "median", "size": 2 * radius + 1}
    if norm == _normalize_halo(NGPLATFORM_FILTERS_FRANGI):
        s_lo = float(params.get("sigma_low", 1.0))
        s_hi = float(params.get("sigma_high", 3.0))
        bright = bool(params.get("bright", True))
        sigmas = [float(x) for x in np.linspace(min(s_lo, s_hi), max(s_lo, s_hi), 5)]
        return {"op": "frangi", "sigmas": sigmas, "black_ridges": not bright}

    return spec


def clean_engine_source(source: str) -> str:
    """
    Normalize a Neuroglancer layer source URL the way `ngplatform`'s `engineSource` does:
    strips trailing `|...` adapter suffixes and `zarr3://`, `zarr2://`, `zarr://`, `n5://`,
    `ome-zarr://` prefixes while preserving `precomputed://`.
    """
    url = re.sub(r"\|.*$", "", source.strip())
    if url.startswith("precomputed://"):
        return url
    return re.sub(r"^(zarr3?|zarr2|n5|ome-zarr)://", "", url)


def to_ngplatform_pipeline(
    spec: Union[PipelineSpec, Dict[str, Any]],
    chunk: Optional[Sequence[int]] = None,
    compile_ops: bool = True,
) -> Dict[str, Any]:
    """
    Convert a `zarr_proxy` / `chunkmirage` `PipelineSpec` or dict into an `ngplatform`
    `ComputedPipeline` (`{source, ops, chunk, select}`).
    """
    if isinstance(spec, PipelineSpec):
        raw = spec.model_dump(exclude_none=True)
    else:
        raw = dict(spec)

    source = clean_engine_source(str(raw.get("source", "")))
    raw_ops = raw.get("ops") or []
    ops = [compile_op_for_webworker(o) if compile_ops else dict(o) for o in raw_ops]

    chunk_list = list(
        chunk
        or raw.get("chunk")
        or raw.get("chunk_shape")
        or DEFAULT_WEBWORKER_CHUNK
    )

    pipeline: Dict[str, Any] = {
        "source": source,
        "ops": ops,
        "chunk": [int(c) for c in chunk_list],
    }
    if raw.get("select"):
        pipeline["select"] = dict(raw["select"])
    return pipeline


def from_ngplatform_pipeline(
    pipeline: Union[str, Dict[str, Any]],
    decompile_ops: bool = True,
) -> PipelineSpec:
    """
    Convert an `ngplatform` `ComputedPipeline` dictionary, encoded payload, or
    `zarr3://<base>/virtual/ngp/<code>/` URL into a `chunkmirage` `PipelineSpec`.
    """
    if isinstance(pipeline, str):
        text = pipeline.strip()
        if "/virtual/ngp/" in text:
            code = text.split("/virtual/ngp/", 1)[1].split("/")[0]
            data = decode_payload(code)
        else:
            data = decode_payload(text)
    else:
        data = dict(pipeline)

    ops = [
        decompile_webworker_op(o) if decompile_ops else dict(o)
        for o in (data.get("ops") or [])
    ]
    chunk_shape = data.get("chunk_shape") or data.get("chunk")
    spec_dict: Dict[str, Any] = {
        "source": data["source"],
        "ops": ops,
    }
    if chunk_shape is not None:
        spec_dict["chunk_shape"] = [int(c) for c in chunk_shape]
    if data.get("select"):
        spec_dict["select"] = dict(data["select"])
    return PipelineSpec.model_validate(spec_dict)


def build_ngplatform_url(
    spec: Union[PipelineSpec, Dict[str, Any]],
    base_url: str = "http://localhost:5173/",
    chunk: Optional[Sequence[int]] = None,
    compile_ops: bool = True,
) -> str:
    """
    Build the `zarr3://<base>virtual/ngp/<deflate-raw-base64url>/` source URL
    that `ngplatform`'s `ComputedImages.url(spec)` produces.
    """
    ngp_spec = to_ngplatform_pipeline(spec, chunk=chunk, compile_ops=compile_ops)
    code = encode_payload(ngp_spec, compress=True)
    base = base_url.rstrip("/") + "/"
    return f"zarr3://{base}virtual/ngp/{code}/"


def get_pyworker_modules() -> Dict[str, str]:
    """
    Return the dictionary of `zarr_proxy` Python source modules (`{"zarr_proxy/...": text}`)
    that can be passed to `chunkmirage-web`'s `Engine.start({ modules })` to mount
    `zarr_proxy.ops` directly inside Pyodide's `/chunkmirage/` virtual filesystem.
    """
    pkg_root = Path(__file__).resolve().parent.parent
    ops_dir = pkg_root / "ops"
    modules: Dict[str, str] = {
        "zarr_proxy/__init__.py": '"""zarr_proxy ops for Pyodide WebWorkers."""\nimport zarr_proxy.ops\n',
    }
    for py_file in sorted(ops_dir.glob("*.py")):
        rel = f"zarr_proxy/ops/{py_file.name}"
        modules[rel] = py_file.read_text(encoding="utf-8")
    return modules


def get_webworker_js(pyodide_url: str = "") -> str:
    """
    Disabled stub: browser-side WebWorker / Pyodide script generation is disabled
    in favor of remote Workstation server execution.
    """
    return (
        "// Browser WebWorker / Pyodide / WASM execution is currently disabled in favor of remote Workstation compute.\n"
        "// loadPyodide is disabled in this mode.\n"
        "// zarr_proxy/ops/filters.py runs on the remote workstation.\n"
    )


def _make_array_info(
    shape: Sequence[int],
    dtype: str,
    chunk: Sequence[int],
    voxel: Sequence[float],
    origin: Optional[Sequence[float]] = None,
    unit: str = "",
) -> ArrayInfo:
    n = len(shape)
    axes = (("c",) if n == 4 else ()) + ("z", "y", "x")[-min(n, 3):]
    return ArrayInfo(
        shape=tuple(int(x) for x in shape),
        dtype=np.dtype(dtype),
        chunk_shape=tuple(int(x) for x in shape[: n - 3]) + tuple(int(x) for x in chunk),
        voxel_size=(1.0,) * max(0, n - 3) + tuple(float(x) for x in voxel),
        units=(unit or "nm",) * n,
        axes=axes,
        translation=None if origin is None else (0.0,) * max(0, n - 3) + tuple(float(x) for x in origin),
    )


class WebWorkerBackend(ComputeBackend):
    """
    Compute backend representing browser WebWorker / Pyodide execution.

    - In a browser Pyodide worker (`sys.platform == "emscripten"`), executes directly
      within the worker's single thread.
    - In CPython (server / testing), executes via a bounded worker pool that mirrors
      `chunkmirage-web`'s Pyodide worker pool (`pyworker.ts`), optionally compiling
      each op through its WebWorker `PythonOp` representation (`simulate_pyodide=True`)
      to guarantee identical behavior between browser WebWorkers and server tests.
    """

    name = "webworker"

    def __init__(self, config: Optional[BackendConfig] = None):
        super().__init__(config)
        default_workers = max(1, min(4, (os.cpu_count() or 4) - 2))
        self.max_workers: int = int(self.config.options.get("max_workers", default_workers))
        self.chunk_shape: List[int] = list(
            self.config.options.get("chunk_shape", DEFAULT_WEBWORKER_CHUNK)
        )
        self.base_url: str = str(self.config.options.get("base_url", "http://localhost:5173/"))
        self.simulate_pyodide: bool = bool(self.config.options.get("simulate_pyodide", False))
        self._is_emscripten: bool = sys.platform == "emscripten"
        self._executor: Optional[ThreadPoolExecutor] = (
            None if self._is_emscripten else ThreadPoolExecutor(max_workers=self.max_workers)
        )
        # Planned views for the `ToPyWorker` message protocol (`plan` / `compute`)
        self._planned_views: Dict[str, Tuple[List[Op], str, List[int], Optional[str]]] = {}

    def evaluate_box(self, array_like: Any, box: Box) -> np.ndarray:
        slices = box.slices()
        if self._executor is not None:
            future = self._executor.submit(lambda: np.asarray(array_like[slices]))
            return future.result()
        return np.asarray(array_like[slices])

    def dispatch_op(
        self,
        func: Callable[..., np.ndarray] | Op | Dict[str, Any] | str,
        block: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Execute an operator on `block`.
        Accepts a callable, a `chunkmirage` `Op` instance, an op spec dict, or an op name string.
        When `simulate_pyodide=True` (or when given an op spec/name), compiles the op
        into its WebWorker `PythonOp` representation and executes it via `PythonOp.apply`.
        """
        def _run() -> np.ndarray:
            if isinstance(func, (dict, str)) or (isinstance(func, Op) and self.simulate_pyodide):
                spec = {"op": func, **kwargs} if isinstance(func, str) else (
                    func.spec() if isinstance(func, Op) else dict(func)
                )
                compiled = compile_op_for_webworker(spec) if self.simulate_pyodide else spec
                op_obj = op_from_spec(compiled)
                return op_obj.apply(block)
            if isinstance(func, Op):
                return func.apply(block)
            if self.simulate_pyodide and hasattr(func, "__name__"):
                op_name = func.__name__.lower()
                if op_name in WEBWORKER_PYTHON_TEMPLATES or op_name in ("ct_window", "dynamic_range"):
                    compiled = compile_op_for_webworker({"op": op_name, **kwargs})
                    return op_from_spec(compiled).apply(block)
            return func(block, **kwargs)

        if self._executor is not None:
            return self._executor.submit(_run).result()
        return _run()

    def compile_pipeline(
        self,
        spec: Union[PipelineSpec, Dict[str, Any]],
        chunk: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Compile a `zarr_proxy` pipeline into an `ngplatform` `ComputedPipeline`."""
        return to_ngplatform_pipeline(spec, chunk=chunk or self.chunk_shape, compile_ops=True)

    def build_url(
        self,
        spec: Union[PipelineSpec, Dict[str, Any]],
        base_url: Optional[str] = None,
        chunk: Optional[Sequence[int]] = None,
    ) -> str:
        """Build the `zarr3://<base>virtual/ngp/<payload>/` URL for `ngplatform`."""
        return build_ngplatform_url(
            spec,
            base_url=base_url or self.base_url,
            chunk=chunk or self.chunk_shape,
            compile_ops=True,
        )

    def handle_worker_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a `chunkmirage-web` `ToPyWorker` RPC message (`plan`, `compute`, `describe`, `call`)
        and return the `Answer` dictionary (`{"reqId": ..., "value": ...}` or `{"reqId": ..., "error": ...}`).
        Matches `vendor/chunkmirage/web/src/pyworker.ts`.
        """
        req_id = msg.get("reqId", 0)
        msg_type = msg.get("type")
        try:
            if msg_type == "plan":
                views = msg.get("views") or {}
                out_plans: Dict[str, Any] = {}
                for view_id, v in views.items():
                    raw_ops = v.get("ops") or []
                    ops = [
                        op_from_spec(compile_op_for_webworker(s) if self.simulate_pyodide else s)
                        for s in raw_ops
                    ]
                    shape = list(v["shape"])
                    dtype = str(v["dtype"])
                    chunk = list(v.get("chunk") or self.chunk_shape)
                    voxel = list(v.get("voxel") or [1.0, 1.0, 1.0])
                    source = v.get("source")
                    info = _make_array_info(shape, dtype, chunk, voxel)
                    out_info, lead, halo = fused.plan(info, ops)
                    self._planned_views[view_id] = (ops, dtype, chunk, source)
                    added = list(out_info.shape[: out_info.ndim - len(halo)])
                    controls = {
                        i: [c.model_dump() for c in op.directives.controls]
                        for i, op in enumerate(ops)
                        if op.name == "python" and hasattr(op, "directives")
                    }
                    out_plans[view_id] = {
                        "dtype": out_info.dtype.name,
                        "kind": out_info.kind,
                        "lead": lead,
                        "halo": list(halo),
                        "ndim": out_info.ndim,
                        "added": added,
                        "controls": controls,
                    }
                return {"reqId": req_id, "value": out_plans}

            if msg_type == "compute":
                view_id = str(msg["view"])
                if view_id not in self._planned_views:
                    raise KeyError(f"View '{view_id}' has not been planned")
                ops, dtype, chunk, _ = self._planned_views[view_id]
                full = tuple(int(x) for x in msg["full"])
                lead = len(full) - 3
                in_lo = tuple(int(x) for x in msg["inLo"])
                in_hi = tuple(int(x) for x in msg["inHi"])
                out_lo = tuple(int(x) for x in msg["outLo"])
                out_hi = tuple(int(x) for x in msg["outHi"])
                read_shape = tuple(int(x) for x in msg["readShape"])
                voxel = list(msg.get("voxel") or [1.0, 1.0, 1.0])

                in_box = Box((0,) * lead + in_lo, full[:lead] + in_hi)
                raw_data = msg["data"]
                if isinstance(raw_data, np.ndarray):
                    block = raw_data.astype(np.dtype(dtype), copy=False).reshape(read_shape)
                else:
                    block = np.frombuffer(bytes(raw_data), dtype=np.dtype(dtype)).reshape(read_shape)
                block = fused.pad_edge(block, in_box, full)

                info = _make_array_info(full, dtype, chunk, voxel)
                out_info, _, halo = fused.plan(info, ops)
                added = out_info.shape[: out_info.ndim - len(halo)]
                out_box = Box((0,) * len(added) + out_lo, tuple(added) + out_hi)
                result = fused.run(ops, block, in_box, out_box, out_info, halo)
                # Pad partial boundary chunks to full chunk size as Zarr v3 expects
                pad_widths = [(0, 0)] * len(added) + [
                    (0, c - n) for c, n in zip(chunk, result.shape[len(added) :])
                ]
                if any(p[1] > 0 for p in pad_widths):
                    result = np.pad(result, pad_widths)
                body = (
                    np.ascontiguousarray(result)
                    .astype(result.dtype.newbyteorder("<"), copy=False)
                    .tobytes()
                )
                return {"reqId": req_id, "value": body}

            if msg_type == "describe":
                from chunkmirage_contrib.sources.synthetic import open_synthetic

                ms = open_synthetic(str(msg["source"]))
                i = ms[0].info
                return {
                    "reqId": req_id,
                    "value": {
                        "dtype": i.dtype.name,
                        "channels": 1,
                        "axes": [{"name": a, "unit": u} for a, u in zip(i.axes, i.units)],
                        "levels": [
                            {
                                "shape": list(lvl.info.shape),
                                "voxel": list(lvl.info.voxel_size),
                                "origin": list(lvl.info.translation or (0.0,) * len(lvl.info.shape)),
                            }
                            for lvl in ms
                        ],
                    },
                }

            if msg_type == "call":
                fn = str(msg["fn"])
                args_str = msg["args"] if isinstance(msg["args"], str) else json.dumps(msg["args"])
                arrays = [bytes(a) for a in (msg.get("arrays") or [])]
                res = calls.call(fn, args_str, arrays)
                if isinstance(res, str):
                    return {"reqId": req_id, "value": json.loads(res)}
                if isinstance(res, tuple):
                    val_str, arrs = res
                    return {"reqId": req_id, "value": {"value": json.loads(val_str), "arrays": arrs}}
                return {"reqId": req_id, "value": res}

            raise ValueError(f"Unknown WebWorker message type: {msg_type!r}")
        except Exception as e:
            return {"reqId": req_id, "error": str(e)}

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._planned_views.clear()
