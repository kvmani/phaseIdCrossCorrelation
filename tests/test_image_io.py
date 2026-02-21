from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from phase_id_xcorr.preprocessing import load_image_as_float32


def test_load_image_uint8_rgb_bmp(tmp_path: Path) -> None:
    arr = np.zeros((10, 12, 3), dtype=np.uint8)
    arr[..., 0] = 10
    arr[..., 1] = 20
    arr[..., 2] = 30
    path = tmp_path / "sample.bmp"
    Image.fromarray(arr, mode="RGB").save(path)

    loaded = load_image_as_float32(path)
    assert loaded.array.shape == (10, 12)
    assert loaded.array.dtype == np.float32
    assert 0.0 <= float(loaded.array.min()) <= float(loaded.array.max()) <= 1.0
    assert loaded.source_bit_depth == 8


def test_load_image_uint16_png(tmp_path: Path) -> None:
    arr = np.arange(100, dtype=np.uint16).reshape(10, 10) * 500
    path = tmp_path / "sample.png"
    Image.fromarray(arr, mode="I;16").save(path)

    loaded = load_image_as_float32(path)
    assert loaded.array.shape == (10, 10)
    assert loaded.source_bit_depth == 16
    assert np.isclose(float(loaded.array[0, 0]), 0.0)
    assert float(loaded.array.max()) <= 1.0
