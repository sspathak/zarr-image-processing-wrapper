"""
3D morphological filtering ops (Median, Erode, Dilate, WhiteTopHat).
Registered as chunkmirage Ops with automated halo calculation.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field
import scipy.ndimage as ndi

from chunkmirage.ops.base import Op, register


@register
class Median(Op):
    """
    3D median filter for non-linear salt-and-pepper and detector noise removal.
    Preserves sharp vessel and tissue boundaries while eliminating impulse noise.
    """
    name = "median"
    output_kind = "image"
    packages = ("scipy",)

    size: int = Field(3, ge=3, le=15, description="Edge size of the 3D median cube (must be odd).")

    @property
    def halo(self) -> int:
        return self.size // 2

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return in_dtype

    def apply(self, block: np.ndarray) -> np.ndarray:
        return ndi.median_filter(block, size=self.size)


@register
class Erode(Op):
    """
    3D grayscale morphological erosion (minimum filter).
    Shrinks bright structures and enlarges dark regions.
    """
    name = "erode"
    output_kind = "image"
    packages = ("scipy",)

    radius: int = Field(1, ge=1, le=10, description="Erosion radius in voxels.")

    @property
    def halo(self) -> int:
        return self.radius

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return in_dtype

    def apply(self, block: np.ndarray) -> np.ndarray:
        size = 2 * self.radius + 1
        return ndi.grey_erosion(block, size=size)


@register
class Dilate(Op):
    """
    3D grayscale morphological dilation (maximum filter).
    Expands bright structures and fills small dark cavities.
    """
    name = "dilate"
    output_kind = "image"
    packages = ("scipy",)

    radius: int = Field(1, ge=1, le=10, description="Dilation radius in voxels.")

    @property
    def halo(self) -> int:
        return self.radius

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return in_dtype

    def apply(self, block: np.ndarray) -> np.ndarray:
        size = 2 * self.radius + 1
        return ndi.grey_dilation(block, size=size)


@register
class WhiteTopHat(Op):
    """
    3D white top-hat filter (Image - Opening(Image)).
    Isolates bright micro-structures (fine capillary vessels, calcifications, puncta)
    smaller than the structuring element on non-uniform background.
    """
    name = "white_tophat"
    output_kind = "image"
    packages = ("scipy",)

    radius: int = Field(2, ge=1, le=10, description="Top-hat structuring element radius in voxels.")

    @property
    def halo(self) -> int:
        return self.radius

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return in_dtype

    def apply(self, block: np.ndarray) -> np.ndarray:
        size = 2 * self.radius + 1
        opened = ndi.grey_opening(block, size=size)
        diff = block.astype(np.float32, copy=False) - opened.astype(np.float32, copy=False)
        if np.issubdtype(block.dtype, np.integer):
            info = np.iinfo(block.dtype)
            return np.clip(diff, info.min, info.max).astype(block.dtype, copy=False)
        return diff.astype(block.dtype, copy=False)

