# ML Phase Explorer GUI

This GUI is the raw `.oh5` exploration tool for the ML branch. It is intended for quick phase-wise inspection before committing to dataset-prep thresholds, preprocessing choices, or training runs.

## 1. Purpose

The explorer loads `.oh5` scan sources from the same YAML-style configs used by dataset preparation and groups patterns by phase. It then exposes:

- phase-wise cumulative intensity histograms,
- optional CDF overlays,
- discovered scalar-field histograms from the `.oh5` file,
- auto-exported publication-quality histogram PNGs with matched axes across phases,
- a machine-readable JSON manifest describing exported figure paths and axis settings,
- interactive intensity-band selection that highlights pixels inside the selected range on a chosen pattern.

This tool is for exploratory analysis only. It does not write training datasets or modify `.oh5` files, but it does write figure/report artifacts into the config-defined `output_dir`.

## 2. Supported Inputs

The explorer reuses the ML dataset config conventions implemented in [`src/phase_id_xcorr/ml/phase_explorer.py`](/Users/anantatamukalaamrutha/python_projects/phaseIdCrossCorrelation/src/phase_id_xcorr/ml/phase_explorer.py):

- `input_mode: oh5_csv_labels`
  - one `.oh5` plus one per-pixel label CSV per source.
- `input_mode: single_phase_scan_map`
  - one `.oh5` per source with a file-level `phase_name` or `phase_label`.
- `schema_version: phase_id_xcorr.ml_dataset_prep.v3`
  - concise `data_source_folder` + `listOfFiles` contract, with optional filename-based phase fallback.

Required `.oh5` content:

- scan dimensions (`nColumns`, `nRows`)
- pattern dataset (`Pattern` or alias-compatible equivalent)

Optional but useful scalar fields:

- `CI` or `Confidence Index`
- `IQ` or `Image Quality`
- `Fit`
- `Valid`
- any additional scalar datasets discoverable by the reader

## 3. Main Command

```bash
python scripts/run_ml_phase_explorer.py --config configs/ml/dataset_prepare.v3_al_ni_cu.example.yml --debug
```

Use a config that points at real local `.oh5` inputs. For quick validation, the phase-explorer tests cover the single-phase scan-map path on small fixtures in [`tests/test_ml_phase_explorer.py`](/Users/anantatamukalaamrutha/python_projects/phaseIdCrossCorrelation/tests/test_ml_phase_explorer.py).

On startup, the explorer also writes PNG and JSON artifacts into the YAML `output_dir`. For a three-phase Al/Cu/Ni configuration, expected exports include:

- `Al_intensity_distribution.png`, `Cu_intensity_distribution.png`, `Ni_intensity_distribution.png`
- `Al_CI.png`, `Cu_CI.png`, `Ni_CI.png`
- `Al_IQ.png`, `Cu_IQ.png`, `Ni_IQ.png`
- `Al_Fit.png`, `Cu_Fit.png`, `Ni_Fit.png`
- `phase_explorer_exports.json`

## 4. GUI Layout

Each phase gets its own column. A column contains:

- a phase header,
- a pattern selector,
- an intensity cumulative histogram panel,
- an attribute histogram panel,
- a pattern viewer with highlighted intensity-band selections.

Intensity panel controls:

- `+ range`: add a movable selection region
- `Clear`: remove all intensity ranges
- gear button: change bin count, x-range, y-range, and CDF visibility

Attribute panel controls:

- scalar-field dropdown populated from discovered `.oh5` scalar datasets
- gear button for plot settings

Pattern viewer behavior:

- selected intensity ranges are unioned,
- matching pixels are highlighted in red on top of the grayscale pattern,
- switching pattern id refreshes the overlay immediately.

## 5. Analytical Use Cases

Use the explorer to answer questions such as:

- whether phases separate cleanly in raw intensity distribution,
- whether quality fields differ strongly by phase before thresholding,
- whether a proposed intensity mask is excluding obvious Kikuchi structure,
- whether filename-based phase fallback is assigning scans as expected.

Export guarantees:

- intensity figures share identical x/y limits across all phases,
- each attribute family (`CI`, `IQ`, `Fit`, or config-selected alternatives) shares identical x/y limits across all phases,
- exported plots use histogram counts only, even if interactive CDF display is enabled in the GUI.

This should inform dataset-prep config edits, not replace them.

## 6. Relation To The ML Pipeline

Recommended order:

1. Inspect raw `.oh5` sources in the explorer.
2. Adjust dataset-prep config fields such as source mapping, quality gates, and preprocessing.
3. Run dataset prep.
4. Run model training or the benchmark suite.
5. Use the one-go full-cycle workflow when you want a single reproducible command from `.oh5` inputs to reports and slides.

See also:

- [`docs/ml_classifier_workflow.md`](/Users/anantatamukalaamrutha/python_projects/phaseIdCrossCorrelation/docs/ml_classifier_workflow.md)
- [`docs/ml_training_inference_workflow.md`](/Users/anantatamukalaamrutha/python_projects/phaseIdCrossCorrelation/docs/ml_training_inference_workflow.md)
- [`configs/ml/README.md`](/Users/anantatamukalaamrutha/python_projects/phaseIdCrossCorrelation/configs/ml/README.md)
