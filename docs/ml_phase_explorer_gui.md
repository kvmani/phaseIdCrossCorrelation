# ML Phase Explorer GUI (Raw `.oh5` Data)

This document defines the native desktop exploratory GUI for raw `.oh5` sources used in ML dataset preparation.

## Objective

Enable phase-wise, leakage-safe, reproducible data exploration for:

- cumulative Kikuchi intensity histograms,
- CDF overlays,
- discovered scalar field histograms (for example `IQ`, `Fit`, `CI`, `Valid`),
- interactive intensity-band selection with highlighted pixels on a selected Kikuchi pattern.

## Entry Point

```bash
python scripts/run_ml_phase_explorer.py --config configs/ml/dataset_prepare.v3_al_ni_cu.example.yml --debug
```

## Input Contract

The GUI accepts the same YAML contract as dataset preparation:

- legacy `sources[].oh5_path` style,
- concise v3 `data_source_folder` + `listOfFiles` style,
- `single_phase_scan_map` and `oh5_csv_labels` modes.

For v3 mode, optional `allow_filename_phase_fallback` is honored.

## Layout and Interactions

- **Phase columns** in 3-column grid for widescreen comparison.
- **Top plot per phase**: cumulative intensity histogram with optional CDF overlay.
- **Second plot per phase**: cumulative histogram for selected discovered scalar field.
- **Pattern panel per phase**: select `Pattern ID` (index over all phase patterns), view Kikuchi pattern, and highlighted pixels for selected intensity ranges.
- **Bottom log window**: detailed run/event diagnostics.

### Range Selection

- Use `+ range` to add multiple x-range selectors on the intensity plot.
- `Clear` removes all selectors.
- Only x-range is used (y ignored by design).
- Highlight mask on the current pattern is the union of selected ranges.

### Plot Gear Settings (Synced Across Phases)

A gear icon is provided for each plot type:

- intensity plot settings (bins/x/y/CDF),
- attribute plot settings (bins/x/y/CDF).

Settings are globally synchronized across phase panels to support direct visual comparison.

## Discoverable Scalar Fields

The explorer auto-discovers scalar-compatible datasets in `.oh5` Data group:

- 1D arrays of length `nPixels`,
- 2D arrays with shape `(nRows, nColumns)`.

These fields are available in the attribute histogram selector.

## Reliability and Reproducibility Notes

- `.oh5` files are read-only.
- Logging includes source loading and scalar-field discovery.
- Sampling caps are used internally for large datasets to keep GUI responsive while preserving representative distributions.

## Implementation Modules

- `src/phase_id_xcorr/ml/phase_explorer.py`: config compatibility, phase aggregation, histogram/CDF/mask utilities.
- `src/phase_id_xcorr/ml/phase_explorer_gui.py`: PySide6/PyQtGraph application.
- `scripts/run_ml_phase_explorer.py`: thin CLI launcher.

## Dependencies

Install desktop GUI dependencies in your Python environment:

- `PySide6`
- `pyqtgraph`
- existing project requirements (`numpy`, `h5py`, `pyyaml`, etc.)
