# `.oh5` Structure and Data Access Guide

Purpose: provide a stable, reusable reference for reading TSL/EDAX `.oh5` files in this project, so future agentic tasks do not repeatedly rediscover schema details.

## Scope

- Input format: `.oh5` (HDF5 container used by TSL/EDAX workflows).
- This project treats `.oh5` as externally produced indexing output.
- Canonical orientation/simulation pipeline in this project depends on extracting fields from these files reliably.

## Sources Used

The conventions below were extracted from:

- [kvmani/kikuchiBandAnalyzer](https://github.com/kvmani/kikuchiBandAnalyzer) (inspected at commit `3f05201` on 2026-02-15)
- Reader implementation: `kikuchiBandAnalyzer/ebsd_compare/readers/oh5_reader.py`
- Project docs: `README.md`, `HowToRunAnalysis.md`, `kikuchiBandAnalyzer/ebsd_compare/README.md`
- Example file: `testData/Test_Ti.oh5`

## Canonical High-Level Layout

Typical top-level keys:

- `Manufacturer`
- `Version`
- `<scan_name>` (primary scan group)

Most EBSD data of interest is under:

- `/<scan_name>/EBSD/Header`
- `/<scan_name>/EBSD/Data`

Notes:

- In reference code, `<scan_name>` is discovered as the first top-level group that is not `Manufacturer` or `Version`.
- `.oh5` and `.h5` are both HDF5 containers; extension differs, structure is typically comparable.

## Typical Header Paths

Common fields in `/<scan_name>/EBSD/Header` include:

- `nColumns`, `nRows` (grid dimensions)
- `Pattern Height`, `Pattern Width`
- `Pattern Bit Depth`
- `Sample Tilt`
- `Camera Azimuthal Angle`
- `Camera Elevation Angle`
- `Working Distance`
- `Step X`, `Step Y`
- `Pattern Center Calibration/*`
- `Phase/*` (phase metadata)

Pattern center calibration subgroup (commonly):

- `Pattern Center Calibration/x-star`
- `Pattern Center Calibration/y-star`
- `Pattern Center Calibration/z-star`
- plus adjustment coefficients `xAdjCoeff*`, `yAdjCoeff*`, `zAdjCoeff*`

## Typical Data Paths

Under `/<scan_name>/EBSD/Data`, common datasets include:

- `Pattern`
- `Phi1`, `Phi`, `Phi2`
- `Phase`
- `X Position`, `Y Position`
- `IQ` or `Image Quality` (naming varies)
- `CI` or `Confidence Index` (naming varies)
- `Fit`
- `SEM Signal`
- `Valid`

## Observed Example (`testData/Test_Ti.oh5`)

From the inspected sample file:

- `Pattern`: shape `(169, 230, 230)`, dtype `uint16`
- `Phi1`, `Phi`, `Phi2`: shape `(169,)`, dtype `float32`
- `Phase`: shape `(169,)`, dtype `int8`
- `X Position`, `Y Position`: shape `(169,)`, dtype `float32`
- `nColumns = 13`, `nRows = 13`

## Shape Conventions for Robust Readers

A robust reader should support these patterns:

Scalar maps:

- flattened form: `(nRows * nColumns,)`
- map form: `(nRows, nColumns)`

Pattern stacks:

- flattened stack: `(nRows * nColumns, H, W)`
- gridded stack: `(nRows, nColumns, H, W)`

Pixel-to-flat-index mapping (row-major convention used by reference reader):

- `flat_index = y * nColumns + x`

## Minimal Robust Discovery Logic

1. Open HDF5 file.
2. Discover `<scan_name>` as first group excluding `Manufacturer` and `Version`.
3. Set:
   - `header = /<scan_name>/EBSD/Header`
   - `data = /<scan_name>/EBSD/Data`
4. Read `nColumns`, `nRows` from header.
5. Classify datasets in `data` by shape into scalar vs pattern categories.
6. For scalar reads, support both flattened and 2D map layouts.
7. For pattern reads, support both flattened and 4D gridded layouts.

## Recommended Access Snippets (`h5py`)

```python
from pathlib import Path
import h5py
import numpy as np


def discover_scan_group(handle: h5py.File) -> str:
    for key in handle.keys():
        if key not in {"Manufacturer", "Version"} and isinstance(handle[key], h5py.Group):
            return key
    raise ValueError("No scan group found")


def open_ebsd_groups(path: str | Path):
    h5 = h5py.File(path, "r")
    scan = discover_scan_group(h5)
    header = h5[f"{scan}/EBSD/Header"]
    data = h5[f"{scan}/EBSD/Data"]
    nx = int(np.ravel(header["nColumns"][()])[0])
    ny = int(np.ravel(header["nRows"][()])[0])
    return h5, scan, header, data, nx, ny


def read_scalar(data_group, field: str, x: int, y: int, nx: int):
    ds = data_group[field]
    if ds.ndim == 1:
        return float(ds[y * nx + x])
    if ds.ndim == 2:
        return float(ds[y, x])
    raise ValueError(f"Unsupported scalar shape: {ds.shape}")


def read_pattern(data_group, field: str, x: int, y: int, nx: int, ny: int):
    ds = data_group[field]
    if ds.ndim >= 3 and ds.shape[0] == nx * ny:
        return np.asarray(ds[y * nx + x], dtype=np.float32)
    if ds.ndim >= 3 and ds.shape[:2] == (ny, nx):
        return np.asarray(ds[y, x], dtype=np.float32)
    raise ValueError(f"Unsupported pattern shape: {ds.shape}")
```

## Field Name Variability and Aliases

Do not hard-code a single name for common maps. Use alias mapping in configuration or code:

- `IQ` <-> `Image Quality`
- `CI` <-> `Confidence Index`
- potential capitalization/spacing variants

A practical strategy is to normalize names with:

- lower-casing,
- trimming whitespace,
- optional replacement of internal spaces/underscores.

## Read/Write Policy for This Project

- Default mode: read-only access to external `.oh5` files.
- If writing derived files, write to new output paths and preserve source files unchanged.
- Keep explicit run manifests documenting input/output file paths and reader assumptions.

## Practical Caveats

- Some datasets may deviate from expected path layout; validate and fail with clear messages.
- Some indexing-focused `.oh5` exports contain orientation/quality maps but omit `Pattern`; ML training workflows must detect this and fail or skip by explicit policy.
- Orientation units and conventions (e.g., Euler angle units/order) must be confirmed before downstream use.
- Flattening order assumptions can break map-pattern alignment if mismatched; validate with small known points.

## Validation Checklist for New `.oh5` Inputs

- [ ] Scan group correctly discovered.
- [ ] `nRows`/`nColumns` read successfully.
- [ ] `Pattern` dataset found and shape recognized.
- [ ] Euler/phase/position fields found (or mapped with aliases).
- [ ] Spot-check a few `(x, y)` accesses against known visualization or external tools.
