"""Tests for KikuchiPy-backed Hough feature extraction."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("kikuchipy") is None:
    pytest.skip("kikuchipy not available", allow_module_level=True)

from phase_id_xcorr.features import (  # noqa: E402
    HoughTransformConfig,
    KikuchiPyHoughExtractor,
    binarize_hough_map,
)


def test_hough_extractor_shape_range_and_determinism() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[16:48, 32] = 1.0
    image[32, 12:52] = 1.0

    cfg = HoughTransformConfig(n_theta=90, n_rho=45, n_bands=6, use_convolved_map=True)
    extractor = KikuchiPyHoughExtractor(image_shape=image.shape, config=cfg)

    r1 = extractor.transform(image)
    r2 = extractor.transform(image)

    assert r1.hough_map.shape == (r1.n_rho, r1.n_theta)
    assert r1.hough_map.shape == r1.radon_raw_map.shape
    assert r1.hough_map.shape == r1.radon_conv_map.shape
    assert 0.0 <= float(np.min(r1.hough_map)) <= 1.0
    assert 0.0 <= float(np.max(r1.hough_map)) <= 1.0
    assert np.allclose(r1.hough_map, r2.hough_map)


def test_binarize_hough_map_threshold() -> None:
    arr = np.array([[0.1, 0.5], [0.7, 0.2]], dtype=np.float32)
    out = binarize_hough_map(arr, threshold=0.5)
    expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert np.array_equal(out, expected)

