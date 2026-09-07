"""
Volumetric filter algorithms and halo size calculations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union
import numpy as np
from skimage.filters import frangi, gaussian, sato, threshold_otsu


def calculate_halo(
    method: str,
    params: Dict[str, Any],
    user_halo: Optional[Union[int, List[int]]] = None,
    ndim: int = 3,
) -> List[int]:
    """
    Compute the required halo (padding in voxels along each spatial dimension)
    needed to prevent boundary artifacts in volumetric filtering.
    """
    if user_halo is not None:
        if isinstance(user_halo, int):
            return [user_halo] * ndim
        elif isinstance(user_halo, (list, tuple)):
            if len(user_halo) == 1:
                return list(user_halo) * ndim
            return list(user_halo)[:ndim]

    m = method.lower()
    if m in ("frangi", "sato"):
        sigmas = params.get("sigmas", [1.0, 2.0])
        if isinstance(sigmas, (int, float)):
            max_sigma = float(sigmas)
        elif isinstance(sigmas, (list, tuple)) and len(sigmas) > 0:
            max_sigma = float(max(sigmas))
        else:
            max_sigma = 2.0
        # 3 sigmas covers 99.7% of Gaussian kernel support
        h = int(math.ceil(3.0 * max_sigma))
        return [h] * ndim

    elif m == "gaussian":
        sigma = params.get("sigma", 1.0)
        if isinstance(sigma, (int, float)):
            max_sigma = float(sigma)
        elif isinstance(sigma, (list, tuple)) and len(sigma) > 0:
            max_sigma = float(max(sigma))
        else:
            max_sigma = 1.0
        h = int(math.ceil(3.0 * max_sigma))
        return [h] * ndim

    # Filters like constant, identity, or pixelwise threshold don't require halos
    return [0] * ndim


def normalize_and_cast(
    arr: np.ndarray,
    target_dtype: str = "uint8",
    normalize: bool = True,
    params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Normalize filtered float output and cast to the target data type.
    """
    params = params or {}
    dt = np.dtype(target_dtype)

    # Check for thresholding / binarization request
    if "threshold" in params:
        th = float(params["threshold"])
        binary = arr >= th
        if dt == np.uint8:
            return (binary * 255).astype(np.uint8)
        elif dt == np.uint16:
            return (binary * 65535).astype(np.uint16)
        else:
            return binary.astype(dt)

    if np.issubdtype(dt, np.floating):
        return arr.astype(dt)

    if not normalize:
        return np.clip(arr, np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)

    # Dynamic or explicit range normalization
    vmin = params.get("vmin")
    vmax = params.get("vmax")

    if vmin is not None:
        min_val = float(vmin)
    else:
        min_val = float(np.nanmin(arr))

    if vmax is not None:
        max_val = float(vmax)
    else:
        max_val = float(np.nanmax(arr))

    max_target = 255.0 if dt == np.uint8 else 65535.0

    if max_val > min_val:
        scaled = (arr - min_val) / (max_val - min_val) * max_target
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=max_target, neginf=0.0)
        return np.clip(scaled, 0, max_target).astype(dt)
    else:
        # If the array is constant, preserve the constant value clipped to target range
        return np.clip(arr, 0, max_target).astype(dt)


def apply_filter(
    volume: np.ndarray,
    method: str,
    params: Dict[str, Any],
    target_dtype: str = "uint8",
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply volumetric filter to an assembled 3D subvolume.
    """
    m = method.lower()
    input_float = volume.astype(np.float32, copy=False)

    if m == "frangi":
        sigmas = params.get("sigmas", [1.0, 2.0])
        alpha = float(params.get("alpha", 0.5))
        beta = float(params.get("beta", 0.5))
        gamma = params.get("gamma")
        black_ridges = bool(params.get("black_ridges", False))
        mode = params.get("mode", "reflect")
        cval = float(params.get("cval", 0.0))

        filtered = frangi(
            input_float,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=black_ridges,
            mode=mode,
            cval=cval,
        )

    elif m == "sato":
        sigmas = params.get("sigmas", [1.0, 2.0])
        black_ridges = bool(params.get("black_ridges", False))
        mode = params.get("mode", "reflect")
        cval = float(params.get("cval", 0.0))

        filtered = sato(
            input_float,
            sigmas=sigmas,
            black_ridges=black_ridges,
            mode=mode,
            cval=cval,
        )

    elif m == "gaussian":
        sigma = params.get("sigma", 1.0)
        mode = params.get("mode", "reflect")
        cval = float(params.get("cval", 0.0))

        filtered = gaussian(
            input_float,
            sigma=sigma,
            mode=mode,
            cval=cval,
            preserve_range=True,
        )

    elif m in ("threshold_otsu", "otsu"):
        if input_float.max() > input_float.min():
            th = threshold_otsu(input_float)
            filtered = (input_float >= th).astype(np.float32)
        else:
            filtered = np.zeros_like(input_float)

    elif m == "constant":
        val = float(params.get("value", 42.0))
        filtered = np.full_like(input_float, val)

    elif m == "identity":
        filtered = input_float

    else:
        raise ValueError(f"Unknown filter method '{method}'")

    return normalize_and_cast(filtered, target_dtype=target_dtype, normalize=normalize, params=params)
