"""Pattern preprocessing for NCC-ready inputs."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from .masking import build_max_inscribed_circle_mask


@dataclass(slots=True)
class PreparedPattern:
    """Preprocessed pattern with mask and diagnostics."""

    array: np.ndarray
    mask: np.ndarray
    valid_pixels: int
    inside_mean: float
    inside_std: float
    inside_min: float
    inside_max: float
    saturated_fraction: float


def _normalize_inside_mask(image: np.ndarray, mask: np.ndarray, method: str = "minmax_inside_mask") -> np.ndarray:
    inside = image[mask]
    if inside.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    method = method.lower().strip()

    if method == "minmax_inside_mask":
        lo = float(np.min(inside))
        hi = float(np.max(inside))
    elif method == "percentile_inside_mask":
        lo = float(np.percentile(inside, 1.0))
        hi = float(np.percentile(inside, 99.0))
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    rng = hi - lo
    if rng <= 0:
        out = np.zeros_like(image, dtype=np.float32)
    else:
        out = (image.astype(np.float32) - lo) / rng
        out = np.clip(out, 0.0, 1.0)

    out[~mask] = 0.0
    return out.astype(np.float32, copy=False)


def prepare_pattern(
    image_float01: np.ndarray,
    normalization_method: str = "minmax_inside_mask",
    logger: logging.Logger | None = None,
) -> PreparedPattern:
    """Prepare image for NCC by masking and normalization."""

    log = logger or logging.getLogger(__name__)

    if image_float01.ndim != 2:
        raise ValueError(f"prepare_pattern expects 2D grayscale image, got shape {image_float01.shape}")

    h, w = image_float01.shape
    mask = build_max_inscribed_circle_mask(h, w)
    norm = _normalize_inside_mask(image_float01, mask=mask, method=normalization_method)

    inside = norm[mask]
    if inside.size == 0:
        stats = dict(valid_pixels=0, inside_mean=0.0, inside_std=0.0, inside_min=0.0, inside_max=0.0, saturated_fraction=0.0)
    else:
        sat = float(np.mean((inside <= 1e-6) | (inside >= 1.0 - 1e-6)))
        stats = dict(
            valid_pixels=int(inside.size),
            inside_mean=float(np.mean(inside)),
            inside_std=float(np.std(inside)),
            inside_min=float(np.min(inside)),
            inside_max=float(np.max(inside)),
            saturated_fraction=sat,
        )

    log.debug(
        (
            "Prepared pattern | shape=%s valid_pixels=%d mean=%.5f std=%.5f "
            "min=%.5f max=%.5f sat_frac=%.5f"
        ),
        norm.shape,
        stats["valid_pixels"],
        stats["inside_mean"],
        stats["inside_std"],
        stats["inside_min"],
        stats["inside_max"],
        stats["saturated_fraction"],
    )

    return PreparedPattern(array=norm, mask=mask, **stats)
