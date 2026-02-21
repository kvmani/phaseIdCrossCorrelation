"""Image I/O with deterministic grayscale conversion and bit-depth handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}


@dataclass(slots=True)
class ImageLoadResult:
    """Loaded image and metadata in canonical float representation."""

    path: str
    array: np.ndarray
    source_dtype: str
    source_bit_depth: int
    source_shape: tuple[int, ...]
    channels: int
    value_min: float
    value_max: float


def _infer_bit_depth(array: np.ndarray) -> int:
    if array.dtype == np.uint8:
        return 8
    if array.dtype == np.uint16:
        return 16
    if np.issubdtype(array.dtype, np.integer):
        if array.size:
            min_val = int(np.min(array))
            max_val = int(np.max(array))
            if min_val >= 0 and max_val <= 255:
                return 8
            if min_val >= 0 and max_val <= 65535:
                return 16
        return int(array.dtype.itemsize * 8)
    if np.issubdtype(array.dtype, np.floating):
        return 32
    return int(array.dtype.itemsize * 8)


def _to_grayscale(array: np.ndarray) -> tuple[np.ndarray, int]:
    if array.ndim == 2:
        return array, 1

    if array.ndim == 3:
        channels = int(array.shape[2])
        if channels == 1:
            return array[..., 0], 1
        if channels >= 3:
            rgb = array[..., :3].astype(np.float64, copy=False)
            gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
            return gray.astype(array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64), channels

    raise ValueError(f"Unsupported image array shape for grayscale conversion: {array.shape}")


def _to_float01(gray: np.ndarray) -> np.ndarray:
    if np.issubdtype(gray.dtype, np.integer):
        min_val = int(np.min(gray)) if gray.size else 0
        max_val = int(np.max(gray)) if gray.size else 0
        if min_val >= 0 and max_val <= 255:
            denom = 255.0
        elif min_val >= 0 and max_val <= 65535:
            denom = 65535.0
        else:
            info = np.iinfo(gray.dtype)
            denom = float(info.max) if info.max > 0 else 1.0
        out = gray.astype(np.float32) / denom
        return np.clip(out, 0.0, 1.0)

    f = gray.astype(np.float32)
    max_val = float(np.nanmax(f)) if f.size else 0.0
    min_val = float(np.nanmin(f)) if f.size else 0.0

    # Common float input cases: already [0,1] or [0,255]/[0,65535]
    if max_val <= 1.0 and min_val >= 0.0:
        return np.clip(f, 0.0, 1.0)
    if max_val <= 255.0 and min_val >= 0.0:
        return np.clip(f / 255.0, 0.0, 1.0)
    if max_val <= 65535.0 and min_val >= 0.0:
        return np.clip(f / 65535.0, 0.0, 1.0)

    # Fallback: robust normalization to [0,1]
    rng = max_val - min_val
    if rng <= 0:
        return np.zeros_like(f, dtype=np.float32)
    return np.clip((f - min_val) / rng, 0.0, 1.0)


def load_image_as_float32(path: Path, logger: logging.Logger | None = None) -> ImageLoadResult:
    """Load image from path and return canonical grayscale float32 in [0,1]."""

    log = logger or logging.getLogger(__name__)

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension '{ext}' for file: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as im:
        arr = np.array(im)

    gray, channels = _to_grayscale(arr)
    float01 = _to_float01(gray)

    if float01.ndim != 2:
        raise ValueError(f"Expected 2D grayscale result, got shape {float01.shape}")

    result = ImageLoadResult(
        path=str(path),
        array=float01.astype(np.float32, copy=False),
        source_dtype=str(arr.dtype),
        source_bit_depth=_infer_bit_depth(arr),
        source_shape=tuple(int(v) for v in arr.shape),
        channels=channels,
        value_min=float(np.min(float01)) if float01.size else 0.0,
        value_max=float(np.max(float01)) if float01.size else 0.0,
    )

    log.debug(
        "Loaded image %s | dtype=%s bit_depth=%s shape=%s channels=%s min=%.5f max=%.5f",
        path,
        result.source_dtype,
        result.source_bit_depth,
        result.source_shape,
        result.channels,
        result.value_min,
        result.value_max,
    )

    return result
