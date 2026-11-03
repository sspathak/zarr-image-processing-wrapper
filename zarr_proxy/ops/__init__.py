"""
Custom image processing Ops for chunkmirage pipelines.
"""

from zarr_proxy.ops.filters import (
    Frangi,
    Laplace,
    Meijering,
    Sato,
    Scharr,
    Sobel,
    UnsharpMask,
)
from zarr_proxy.ops.intensity import CTWindow, DynamicRange, GammaCorrection
from zarr_proxy.ops.morphology import Dilate, Erode, Median, WhiteTopHat
from zarr_proxy.ops.passthrough import Identity, Passthrough
from zarr_proxy.ops.segment import Li, Otsu

__all__ = [
    "CTWindow",
    "Dilate",
    "DynamicRange",
    "Erode",
    "Frangi",
    "GammaCorrection",
    "Identity",
    "Laplace",
    "Li",
    "Median",
    "Meijering",
    "Otsu",
    "Passthrough",
    "Sato",
    "Scharr",
    "Sobel",
    "UnsharpMask",
    "WhiteTopHat",
]
