"""
Passthrough / Identity filter op for chunkmirage pipelines.
Passes chunks through unchanged with zero halo and preserved data types.
Useful for testing pipeline connectivity and serving raw data through the proxy.
"""

from __future__ import annotations

import numpy as np
from chunkmirage.ops.base import Op, register


@register
class Passthrough(Op):
    """
    Passthrough / Identity operator.
    Passes chunk blocks through with zero spatial padding and zero modification.
    Preserves original data type and coordinate space.
    """
    name = "passthrough"
    output_kind = None

    @property
    def halo(self) -> int:
        return 0

    def output_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return in_dtype

    def apply(self, block: np.ndarray) -> np.ndarray:
        return block


@register
class Identity(Passthrough):
    """Alias for Passthrough."""
    name = "identity"

