"""Masked normalized cross-correlation (NCC)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NCCResult:
    """NCC score with diagnostics."""

    score: float
    valid_pixels: int
    denom: float
    is_valid: bool
    reason: str


def masked_ncc(exp: np.ndarray, sim: np.ndarray, mask: np.ndarray, eps: float = 1e-12) -> NCCResult:
    """Compute NCC over masked region with defensive checks."""

    if exp.shape != sim.shape or exp.shape != mask.shape:
        raise ValueError(
            "Shape mismatch in masked_ncc: "
            f"exp={exp.shape} sim={sim.shape} mask={mask.shape}"
        )

    valid = mask & np.isfinite(exp) & np.isfinite(sim)
    n = int(np.count_nonzero(valid))
    if n < 2:
        return NCCResult(score=0.0, valid_pixels=n, denom=0.0, is_valid=False, reason="insufficient_valid_pixels")

    x = exp[valid].astype(np.float64, copy=False)
    y = sim[valid].astype(np.float64, copy=False)

    x_c = x - np.mean(x)
    y_c = y - np.mean(y)

    sx = float(np.sum(x_c * x_c))
    sy = float(np.sum(y_c * y_c))
    denom = float(np.sqrt(sx * sy))

    if denom <= eps:
        return NCCResult(score=0.0, valid_pixels=n, denom=denom, is_valid=False, reason="near_zero_variance")

    score = float(np.sum(x_c * y_c) / denom)
    score = float(np.clip(score, -1.0, 1.0))

    return NCCResult(score=score, valid_pixels=n, denom=denom, is_valid=True, reason="ok")
