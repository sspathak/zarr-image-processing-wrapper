"""
Segmentation and thresholding ops (Otsu, Li, Threshold).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from pydantic import Field
from skimage.filters import threshold_li, threshold_otsu

from chunkmirage.ops.base import Op, register


@register
class Otsu(Op):
    """
    Otsu automated thresholding for binary segmentation.
    Computes an optimal threshold separating foreground from background.
    """
    name = "otsu"
    output_kind = "mask"
    packages = ("scikit-image",)

    nbins: int = Field(256, ge=2, le=65536, description="Number of histogram bins for threshold estimation.")
    value: int = Field(1, ge=1, le=255, description="Label value written for foreground voxels.")

    @property
    def halo(self) -> int:
        return 4  # Context margin for robust local histogram statistics

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        b_min, b_max = np.min(block), np.max(block)
        if b_min == b_max:
            return np.zeros(block.shape, dtype=np.uint8)
        try:
            th = threshold_otsu(block, nbins=self.nbins)
            return ((block >= th) * np.uint8(self.value)).astype(np.uint8)
        except Exception:
            return np.zeros(block.shape, dtype=np.uint8)


@register
class Li(Op):
    """
    Li minimum cross-entropy automated thresholding for binary segmentation.
    Excels when foreground (e.g. coronary vessels or cell bodies) occupies
    only a small fraction of the image volume.
    """
    name = "li"
    output_kind = "mask"
    packages = ("scikit-image",)

    value: int = Field(1, ge=1, le=255, description="Label value written for foreground voxels.")

    @property
    def halo(self) -> int:
        return 4

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        b_min, b_max = np.min(block), np.max(block)
        if b_min == b_max:
            return np.zeros(block.shape, dtype=np.uint8)
        try:
            th = threshold_li(block.astype(np.float32, copy=False))
            return ((block >= th) * np.uint8(self.value)).astype(np.uint8)
        except Exception:
            return np.zeros(block.shape, dtype=np.uint8)
