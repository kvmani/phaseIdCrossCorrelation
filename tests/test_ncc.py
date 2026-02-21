from __future__ import annotations

import numpy as np

from phase_id_xcorr.similarity import masked_ncc


def test_masked_ncc_identity() -> None:
    x = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    mask = np.ones_like(x, dtype=bool)
    res = masked_ncc(x, x, mask)
    assert res.is_valid
    assert res.score > 0.999


def test_masked_ncc_inverted() -> None:
    x = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    y = 1.0 - x
    mask = np.ones_like(x, dtype=bool)
    res = masked_ncc(x, y, mask)
    assert res.is_valid
    assert res.score < -0.999


def test_masked_ncc_zero_variance_invalid() -> None:
    x = np.zeros((8, 8), dtype=np.float32)
    y = np.zeros((8, 8), dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    res = masked_ncc(x, y, mask)
    assert not res.is_valid
    assert res.reason == "near_zero_variance"
