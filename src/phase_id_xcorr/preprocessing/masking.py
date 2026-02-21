"""Mask generation helpers."""

from __future__ import annotations

import numpy as np


def build_max_inscribed_circle_mask(height: int, width: int) -> np.ndarray:
    """Build a centered boolean mask for the maximum inscribed circle."""

    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape for mask: {(height, width)}")

    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = min(cx, cy)

    y, x = np.ogrid[:height, :width]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    mask = dist_sq <= (radius**2)
    return mask.astype(bool)
