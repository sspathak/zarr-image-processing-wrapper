"""
Intensity normalization, window-level, and dynamic range rescaling ops.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from pydantic import Field

from chunkmirage.ops.base import Op, register


@register
class CTWindow(Op):
    """
    Standard medical/CT window-level contrast mapping.
    Clips voxel intensities to [vmin, vmax] and linearly normalizes to target dynamic range
    (e.g. 0-255 for uint8, or 0.0-1.0 for float32).
    Default parameters are tuned for UCL HiP-CT soft-tissue / myocardium contrast (18000-26000).
    """
    name = "ct_window"
    output_kind = "image"

    vmin: float = Field(18000.0, description="Lower intensity threshold in Hounsfield / raw CT units.")
    vmax: float = Field(26000.0, description="Upper intensity threshold in Hounsfield / raw CT units.")
    target_dtype: str = Field("uint8", description="Target output dtype: uint8, uint16, float32.")

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype(self.target_dtype)

    def apply(self, block: np.ndarray) -> np.ndarray:
        target_dt = np.dtype(self.target_dtype)
        max_val = 255.0 if target_dt == np.uint8 else (65535.0 if target_dt == np.uint16 else 1.0)
        b = block.astype(np.float32, copy=False)
        diff = max(self.vmax - self.vmin, 1e-6)
        rescaled = np.clip((b - self.vmin) / diff, 0.0, 1.0) * max_val
        return rescaled.astype(target_dt, copy=False)


@register
class DynamicRange(Op):
    """
    Rescale input intensities to target dynamic range (e.g. 0-255 or 0-65535) and cast.
    Supports percentile clipping to handle microscopy/CT outliers.
    """
    name = "dynamic_range"
    output_kind = "image"

    pmin: float = Field(0.0, ge=0.0, le=50.0, description="Lower percentile for clipping (default 0.0).")
    pmax: float = Field(100.0, ge=50.0, le=100.0, description="Upper percentile for clipping (default 100.0).")
    target_dtype: str = Field("uint8", description="Target output dtype: uint8, uint16, float32.")

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype(self.target_dtype)

    def apply(self, block: np.ndarray) -> np.ndarray:
        target_dt = np.dtype(self.target_dtype)
        if target_dt == np.uint8:
            max_val = 255.0
        elif target_dt == np.uint16:
            max_val = 65535.0
        else:
            max_val = 1.0

        b = block.astype(np.float32, copy=False)
        if self.pmin > 0.0 or self.pmax < 100.0:
            lo, hi = np.percentile(b, (self.pmin, self.pmax))
        else:
            lo, hi = float(np.min(b)), float(np.max(b))

        diff = hi - lo
        if diff <= 1e-9:
            rescaled = np.zeros_like(b)
        else:
            rescaled = np.clip((b - lo) / diff, 0.0, 1.0) * max_val

        return rescaled.astype(target_dt, copy=False)


@register
class GammaCorrection(Op):
    """
    Power-law non-linear gamma contrast adjustment: Out = ((In - min)/(max - min))^gamma * scale.
    Gamma < 1.0 brightens shadows / low-intensity vessels.
    Gamma > 1.0 darkens background and boosts high-intensity structures.
    """
    name = "gamma"
    output_kind = "image"

    gamma: float = Field(0.5, ge=0.05, le=5.0, description="Power-law exponent (e.g. 0.5 for sqrt, 2.0 for quadratic).")

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return in_dtype

    def apply(self, block: np.ndarray) -> np.ndarray:
        b = block.astype(np.float32, copy=False)
        lo, hi = float(np.min(b)), float(np.max(b))
        diff = max(hi - lo, 1e-6)
        normalized = np.clip((b - lo) / diff, 0.0, 1.0)
        adjusted = np.power(normalized, self.gamma) * diff + lo
        if np.issubdtype(block.dtype, np.integer):
            info = np.iinfo(block.dtype)
            return np.clip(adjusted, info.min, info.max).astype(block.dtype, copy=False)
        return adjusted.astype(block.dtype, copy=False)
