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

Publication-export styling is controlled from `explorer.export`, including:

- `dpi`
- `figure_size_inches`
- `font_family`
- `font_size`
- `title_font_size`
- `label_font_size`
- `tick_label_size`
- `x_tick_label_size`
- `y_tick_label_size`
- `tick_width`
- `tick_length`
- `minor_tick_width`
- `minor_tick_length`
- `tick_direction`
- `x_tick_rotation`
- `y_tick_rotation`
- `spine_line_width`
- `grid_line_width`
- `grid_alpha`
- `title_pad`
- `label_pad`
- `figure_facecolor`
- `axes_facecolor`
- `show_minor_ticks`
- `savefig_pad_inches`

Per-plot export control is available from the phase plot blocks themselves:

- `explorer.intensity_plot`
  - `bins`
  - `x_min`, `x_max`, `y_min`, `y_max`
  - `title`, `title_template`
  - `x_label`, `x_label_template`
  - `y_label`, `y_label_template`
  - `color`
  - `edge_color`
  - `bar_line_width`
- `explorer.attribute_plot`
  - `bins`
  - `y_min`, `y_max`
  - `field_ranges`
  - `field_y_ranges`
  - `title`, `title_template`
  - `x_label`, `x_label_template`
  - `y_label`, `y_label_template`
  - `color`
  - `edge_color`
  - `bar_line_width`

`title_template`, `x_label_template`, and `y_label_template` support:

- `{phase}`
- `{attribute}`

Example:

```yaml
explorer:
  intensity_plot:
    bins: 512
    x_min: 0
    x_max: 65535
    y_min: 0
    y_max: 120000
    title_template: "{phase}"
    x_label: Intensity
    y_label: Pixel count
    color: "#1f77b4"
    edge_color: "#1f77b4"
    bar_line_width: 0.8
  attribute_plot:
    bins: 80
    y_min: 0
    field_ranges:
      CI: [0, 1]
      IQ: [0, 6000]
      Fit: [0, 2]
    field_y_ranges:
      CI: [0, 150000]
    title_template: "{phase} {attribute}"
    x_label_template: "{attribute}"
    y_label: Pixel count
    color: "#2ca02c"
    edge_color: "#2ca02c"
    bar_line_width: 0.8
  export:
    dpi: 400
    figure_size_inches: [8.5, 5.5]
    font_family: Arial
    font_size: 18
    title_font_size: 20
    label_font_size: 18
    tick_label_size: 16
    x_tick_label_size: 16
    y_tick_label_size: 16
    tick_width: 1.2
    tick_length: 6
    minor_tick_width: 0.9
    minor_tick_length: 3.5
    tick_direction: out
    x_tick_rotation: 0
    y_tick_rotation: 0
    spine_line_width: 1.2
    grid_line_width: 0.8
    grid_alpha: 0.25
    title_pad: 10
    label_pad: 8
    figure_facecolor: white
    axes_facecolor: white
    show_minor_ticks: false
    savefig_pad_inches: 0.1
    attributes: [CI, IQ, Fit]
```

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
