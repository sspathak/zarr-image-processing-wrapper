"""
Volumetric filters for tubular structure enhancement, gradients, and sharpening.
Registered as chunkmirage Ops with automated halo calculation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union
import numpy as np
from pydantic import Field
import scipy.ndimage as ndi
from skimage.filters import frangi, sato

from chunkmirage.ops.base import Op, register


@register
class Sato(Op):
    """
    3D Sato tubeness filter for tubular structure and vessel enhancement.
    Computes sorted eigenvalues of the Hessian matrix. Fast, robust, and clean.
    """
    name = "sato"
    output_kind = "image"
    packages = ("scikit-image", "scipy")

    sigmas: Union[float, List[float]] = Field(
        default_factory=lambda: [1.0, 2.0],
        description="Filter scales in voxels. Enhances tubular structures with radius ~ sigma.",
    )
    black_ridges: bool = Field(False, description="Detect dark tubular structures if True.")

    @property
    def halo(self) -> int:
        if isinstance(self.sigmas, (int, float)):
            max_s = float(self.sigmas)
        elif isinstance(self.sigmas, (list, tuple)) and len(self.sigmas) > 0:
            max_s = float(max(self.sigmas))
        else:
            max_s = 2.0
        return math.ceil(3.0 * max_s)

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        s = [float(self.sigmas)] if isinstance(self.sigmas, (int, float)) else [float(x) for x in self.sigmas]
        result = sato(
            block.astype(np.float32, copy=False),
            sigmas=s,
            black_ridges=self.black_ridges,
        )
        return result.astype(np.float32, copy=False)


@register
class Frangi(Op):
    """
    3D Frangi vesselness filter for tubular structure enhancement.
    Computes eigenvalues of the Hessian matrix across multiple Gaussian scales.
    """
    name = "frangi"
    output_kind = "image"
    packages = ("scikit-image", "scipy")

    sigmas: Union[float, List[float]] = Field(
        default_factory=lambda: [1.0, 2.0],
        description="Filter scales in voxels. Enhances tubular structures with radius ~ sigma.",
    )
    alpha: float = Field(0.5, description="Frangi plate sensitivity constant.")
    beta: float = Field(0.5, description="Frangi blob sensitivity constant.")
    gamma: Optional[float] = Field(
        None,
        description="Background noise sensitivity threshold. If None, uses half maximum Hessian norm.",
    )
    black_ridges: bool = Field(False, description="Filter for dark tubes against bright background if True.")

    @property
    def halo(self) -> int:
        if isinstance(self.sigmas, (int, float)):
            max_s = float(self.sigmas)
        elif isinstance(self.sigmas, (list, tuple)) and len(self.sigmas) > 0:
            max_s = float(max(self.sigmas))
        else:
            max_s = 2.0
        return math.ceil(3.0 * max_s)

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        s = [float(self.sigmas)] if isinstance(self.sigmas, (int, float)) else [float(x) for x in self.sigmas]
        result = frangi(
            block.astype(np.float32, copy=False),
            sigmas=s,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            black_ridges=self.black_ridges,
        )
        return result.astype(np.float32, copy=False)


@register
class Meijering(Op):
    """
    3D Meijering neurite / ridge detection filter.
    Computes modified eigenvalues of the Hessian matrix.
    """
    name = "meijering"
    output_kind = "image"
    packages = ("scikit-image", "scipy")

    sigmas: Union[float, List[float]] = Field(
        default_factory=lambda: [1.0, 2.0],
        description="Filter scales in voxels.",
    )
    black_ridges: bool = Field(False, description="Filter for dark ridges if True.")

    @property
    def halo(self) -> int:
        if isinstance(self.sigmas, (int, float)):
            max_s = float(self.sigmas)
        elif isinstance(self.sigmas, (list, tuple)) and len(self.sigmas) > 0:
            max_s = float(max(self.sigmas))
        else:
            max_s = 2.0
        return math.ceil(3.0 * max_s)

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from skimage.filters import meijering
        s = [float(self.sigmas)] if isinstance(self.sigmas, (int, float)) else [float(x) for x in self.sigmas]
        result = meijering(
            block.astype(np.float32, copy=False),
            sigmas=s,
            black_ridges=self.black_ridges,
        )
        return result.astype(np.float32, copy=False)


@register
class Sobel(Op):
    """
    3D Sobel gradient magnitude filter for edge and boundary detection.
    Computes Euclidean norm of gradients across all spatial dimensions: sqrt(Gz^2 + Gy^2 + Gx^2).
    """
    name = "sobel"
    halo = 1
    output_kind = "image"
    packages = ("scipy",)

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        arr = block.astype(np.float32, copy=False)
        grads = [ndi.sobel(arr, axis=ax) for ax in range(arr.ndim)]
        mag = np.sqrt(sum(g ** 2 for g in grads))
        return mag.astype(np.float32, copy=False)


@register
class Scharr(Op):
    """
    3D Scharr gradient magnitude filter with enhanced rotational symmetry.
    """
    name = "scharr"
    halo = 1
    output_kind = "image"
    packages = ("scipy",)

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
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
            mag = np.sqrt(sum(g ** 2 for g in grads))
            return mag.astype(np.float32, copy=False)
        grads = [ndi.sobel(arr, axis=ax) for ax in range(arr.ndim)]
        return np.sqrt(sum(g ** 2 for g in grads)).astype(np.float32, copy=False)


@register
class Laplace(Op):
    """
    3D Laplacian second-order derivative filter for boundary and edge detection.
    """
    name = "laplace"
    halo = 1
    output_kind = "image"
    packages = ("scipy",)

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        arr = block.astype(np.float32, copy=False)
        return ndi.laplace(arr).astype(np.float32, copy=False)


@register
class UnsharpMask(Op):
    """
    3D unsharp masking to enhance high-frequency edges and microstructural detail.
    Result = Image + amount * (Image - GaussianBlur(Image, radius)).
    """
    name = "unsharp_mask"
    output_kind = "image"
    packages = ("scipy",)

    radius: float = Field(1.5, ge=0.5, le=10.0, description="Radius/sigma of Gaussian blur in voxels.")
    amount: float = Field(1.5, ge=0.1, le=10.0, description="Sharpening strength / amplification factor.")

    @property
    def halo(self) -> int:
        return math.ceil(3.0 * max(0.5, self.radius))

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return in_dtype

    def apply(self, block: np.ndarray) -> np.ndarray:
        arr = block.astype(np.float32, copy=False)
        blurred = ndi.gaussian_filter(arr, sigma=self.radius)
        sharpened = arr + self.amount * (arr - blurred)
        if np.issubdtype(block.dtype, np.integer):
            info = np.iinfo(block.dtype)
            return np.clip(sharpened, info.min, info.max).astype(block.dtype, copy=False)
        return sharpened.astype(block.dtype, copy=False)
