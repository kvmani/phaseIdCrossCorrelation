"""Robust `.oh5` reader for ML dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np


QUALITY_ALIASES: dict[str, tuple[str, ...]] = {
    "confidence_index": ("CI", "Confidence Index"),
    "image_quality": ("IQ", "Image Quality"),
    "fit": ("Fit",),
    "valid": ("Valid",),
}

PATTERN_ALIASES: tuple[str, ...] = ("Pattern", "Patterns")


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _read_scalar(value: Any) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("Cannot read scalar from empty dataset")
    return float(np.ravel(arr)[0])


def _infer_bit_depth(arr: np.ndarray) -> int:
    if arr.dtype == np.uint8:
        return 8
    if arr.dtype == np.uint16:
        return 16
    if np.issubdtype(arr.dtype, np.integer):
        return int(arr.dtype.itemsize * 8)
    return 32


def _to_float01(gray: np.ndarray) -> np.ndarray:
    if np.issubdtype(gray.dtype, np.integer):
        max_v = int(np.max(gray)) if gray.size else 0
        if max_v <= 255:
            denom = 255.0
        elif max_v <= 65535:
            denom = 65535.0
        else:
            denom = float(np.iinfo(gray.dtype).max)
        out = gray.astype(np.float32) / max(1.0, denom)
        return np.clip(out, 0.0, 1.0)

    f = gray.astype(np.float32)
    lo = float(np.nanmin(f)) if f.size else 0.0
    hi = float(np.nanmax(f)) if f.size else 0.0
    if 0.0 <= lo <= hi <= 1.0:
        return np.clip(f, 0.0, 1.0)
    if 0.0 <= lo <= hi <= 255.0:
        return np.clip(f / 255.0, 0.0, 1.0)
    if 0.0 <= lo <= hi <= 65535.0:
        return np.clip(f / 65535.0, 0.0, 1.0)
    rng = hi - lo
    if rng <= 0.0:
        return np.zeros_like(f, dtype=np.float32)
    return np.clip((f - lo) / rng, 0.0, 1.0)


@dataclass(slots=True)
class Oh5ScanMeta:
    """Parsed scan-level metadata from a `.oh5` file."""

    path: str
    scan_group: str
    nx: int
    ny: int
    total_pixels: int
    pattern_present: bool
    pattern_key: str | None
    pattern_shape: tuple[int, int] | None
    pattern_bit_depth: int | None
    quality_field_map: dict[str, str]


class Oh5ScanReader:
    """Open `.oh5` scan and provide per-pixel accessors."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self._h5: h5py.File | None = None
        self.scan_group: str | None = None
        self._header: h5py.Group | None = None
        self._data: h5py.Group | None = None
        self.nx = 0
        self.ny = 0
        self.header_total_pixels = 0
        self.total_pixels = 0
        self.pattern_key: str | None = None
        self.pattern_shape: tuple[int, int] | None = None
        self.pattern_bit_depth: int | None = None
        self.quality_field_map: dict[str, str] = {}

    def __enter__(self) -> "Oh5ScanReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    @property
    def data_group(self) -> h5py.Group:
        if self._data is None:
            raise RuntimeError("Reader is not open")
        return self._data

    @property
    def header_group(self) -> h5py.Group:
        if self._header is None:
            raise RuntimeError("Reader is not open")
        return self._header

    @property
    def pattern_present(self) -> bool:
        return self.pattern_key is not None

    def open(self) -> None:
        if self._h5 is not None:
            return

        self._h5 = h5py.File(self.path, "r")
        self.scan_group = self._discover_scan_group(self._h5)
        self._header = self._h5[f"{self.scan_group}/EBSD/Header"]
        self._data = self._h5[f"{self.scan_group}/EBSD/Data"]

        self.nx = int(_read_scalar(self.header_group["nColumns"][()]))
        self.ny = int(_read_scalar(self.header_group["nRows"][()]))
        self.header_total_pixels = int(self.nx * self.ny)
        self.total_pixels = int(self.header_total_pixels)

        self.pattern_key = self._find_dataset_key(PATTERN_ALIASES)
        if self.pattern_key is not None:
            ds = self.data_group[self.pattern_key]
            self.total_pixels = self._resolve_effective_total_pixels(ds)
            h, w = self._pattern_hw(ds)
            self.pattern_shape = (h, w)
            self.pattern_bit_depth = _infer_bit_depth(np.asarray(ds[(0,) + (slice(None),) * (ds.ndim - 1)]))

        self.quality_field_map = {}
        for canonical, aliases in QUALITY_ALIASES.items():
            key = self._find_dataset_key(aliases)
            if key is not None:
                self.quality_field_map[canonical] = key

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._header = None
        self._data = None

    def meta(self) -> Oh5ScanMeta:
        if self.scan_group is None:
            raise RuntimeError("Reader is not open")
        return Oh5ScanMeta(
            path=str(self.path),
            scan_group=self.scan_group,
            nx=self.nx,
            ny=self.ny,
            total_pixels=self.total_pixels,
            pattern_present=self.pattern_present,
            pattern_key=self.pattern_key,
            pattern_shape=self.pattern_shape,
            pattern_bit_depth=self.pattern_bit_depth,
            quality_field_map=dict(self.quality_field_map),
        )

    def xy_to_flat(self, x: int, y: int) -> int:
        if x < 0 or y < 0 or x >= self.nx or y >= self.ny:
            raise ValueError(f"Pixel ({x}, {y}) out of range for grid {(self.nx, self.ny)}")
        return int(y * self.nx + x)

    def flat_to_xy(self, flat_index: int) -> tuple[int, int]:
        if flat_index < 0 or flat_index >= self.total_pixels:
            raise ValueError(f"flat_index {flat_index} out of range [0, {self.total_pixels})")
        y = flat_index // self.nx
        x = flat_index % self.nx
        return int(x), int(y)

    def read_pattern(self, *, flat_index: int | None = None, x: int | None = None, y: int | None = None) -> np.ndarray:
        if not self.pattern_present:
            raise KeyError("Pattern dataset not found in this .oh5 file")

        idx = self._resolve_flat_index(flat_index=flat_index, x=x, y=y)
        ds = self.data_group[self.pattern_key]  # type: ignore[index]

        if ds.ndim >= 3 and ds.shape[0] == self.total_pixels:
            arr = np.asarray(ds[idx], dtype=np.float32)
            return _to_float01(arr).astype(np.float32, copy=False)

        if ds.ndim >= 4 and tuple(ds.shape[:2]) == (self.ny, self.nx):
            xx, yy = self.flat_to_xy(idx)
            arr = np.asarray(ds[yy, xx], dtype=np.float32)
            return _to_float01(arr).astype(np.float32, copy=False)

        raise ValueError(f"Unsupported pattern dataset shape: {tuple(ds.shape)}")

    def read_quality_row(self, *, flat_index: int | None = None, x: int | None = None, y: int | None = None) -> dict[str, float | bool | None]:
        idx = self._resolve_flat_index(flat_index=flat_index, x=x, y=y)
        row: dict[str, float | bool | None] = {
            "confidence_index": None,
            "image_quality": None,
            "fit": None,
            "valid": None,
        }

        for canonical, key in self.quality_field_map.items():
            ds = self.data_group[key]
            value = self._read_scalar_point(ds, idx)
            if canonical == "valid":
                row[canonical] = bool(round(float(value)) != 0)
            else:
                row[canonical] = float(value)

        return row


    def discover_scalar_fields(self) -> list[str]:
        """Discover scalar-compatible fields in the Data group."""

        out: list[str] = []
        for key in self.data_group.keys():
            ds = self.data_group[key]
            if not isinstance(ds, h5py.Dataset):
                continue
            if ds.ndim == 1 and int(ds.shape[0]) >= int(self.total_pixels):
                out.append(str(key))
            elif ds.ndim == 2 and tuple(ds.shape) == (self.ny, self.nx):
                out.append(str(key))
        return sorted(out)

    def read_scalar_field_value(self, field_name: str, *, flat_index: int) -> float | bool | None:
        """Read one scalar value by field name for a pixel."""

        if field_name not in self.data_group:
            return None
        ds = self.data_group[field_name]
        if not isinstance(ds, h5py.Dataset):
            return None
        value = self._read_scalar_point(ds, int(flat_index))
        if _normalize_key(field_name) == _normalize_key("Valid"):
            return bool(round(float(value)) != 0)
        return float(value)

    def read_scalar_field_array(self, field_name: str) -> np.ndarray | None:
        """Read a full scalar field array when present."""

        if field_name not in self.data_group:
            return None
        ds = self.data_group[field_name]
        if not isinstance(ds, h5py.Dataset):
            return None
        if ds.ndim == 1 and int(ds.shape[0]) >= int(self.header_total_pixels):
            return np.asarray(ds[: self.header_total_pixels], dtype=np.float32)
        if ds.ndim == 2 and tuple(ds.shape) == (self.ny, self.nx):
            return np.asarray(ds, dtype=np.float32).reshape(-1)
        return None

    def read_scalar_row_all(self, *, flat_index: int, field_names: list[str] | None = None) -> dict[str, float | bool | None]:
        """Read all discovered scalar values for one pixel."""

        names = field_names or self.discover_scalar_fields()
        out: dict[str, float | bool | None] = {}
        for name in names:
            out[str(name)] = self.read_scalar_field_value(str(name), flat_index=int(flat_index))
        return out

    def _resolve_flat_index(self, *, flat_index: int | None, x: int | None, y: int | None) -> int:
        if flat_index is not None:
            idx = int(flat_index)
            if idx < 0 or idx >= self.total_pixels:
                raise ValueError(f"flat_index {idx} out of range [0, {self.total_pixels})")
            return idx

        if x is None or y is None:
            raise ValueError("Provide either flat_index or both x and y")
        return self.xy_to_flat(int(x), int(y))

    def _read_scalar_point(self, ds: h5py.Dataset, flat_index: int) -> float:
        if ds.ndim == 1:
            return float(ds[flat_index])
        if ds.ndim == 2 and tuple(ds.shape) == (self.ny, self.nx):
            x, y = self.flat_to_xy(flat_index)
            return float(ds[y, x])
        raise ValueError(f"Unsupported scalar dataset shape for {ds.name}: {tuple(ds.shape)}")

    def _find_dataset_key(self, candidates: tuple[str, ...]) -> str | None:
        keys = list(self.data_group.keys())
        norm_map = {_normalize_key(k): k for k in keys}
        for candidate in candidates:
            hit = norm_map.get(_normalize_key(candidate))
            if hit is not None:
                return hit
        return None

    def _pattern_hw(self, ds: h5py.Dataset) -> tuple[int, int]:
        if ds.ndim >= 3 and ds.shape[0] == self.total_pixels:
            return int(ds.shape[-2]), int(ds.shape[-1])
        if ds.ndim >= 4 and tuple(ds.shape[:2]) == (self.ny, self.nx):
            return int(ds.shape[-2]), int(ds.shape[-1])
        raise ValueError(f"Unsupported pattern dataset shape: {tuple(ds.shape)}")

    def _resolve_effective_total_pixels(self, ds: h5py.Dataset) -> int:
        header_total = int(self.nx * self.ny)
        if ds.ndim >= 4 and tuple(ds.shape[:2]) == (self.ny, self.nx):
            return header_total
        if ds.ndim >= 3 and int(ds.shape[0]) <= header_total:
            return int(ds.shape[0])
        raise ValueError(f"Unsupported pattern dataset shape: {tuple(ds.shape)}")

    @staticmethod
    def _discover_scan_group(handle: h5py.File) -> str:
        for key in handle.keys():
            if key not in {"Manufacturer", "Version"} and isinstance(handle[key], h5py.Group):
                return str(key)
        raise ValueError("No scan group found in .oh5 file")
