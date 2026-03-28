# ML Classifier Workflow

This is the overview doc for the ML branch. It summarizes the pipeline, module boundaries, and artifact contract.

Use the companion docs for details:

- `docs/ml_input_data_runbook.md`: dataset inputs and platform-specific commands
- `docs/ml_training_inference_workflow.md`: recommended experiment flow
- `docs/ml_phase_explorer_gui.md`: raw `.oh5` exploration GUI

## 1. Objective

Build a reproducible supervised classifier pipeline that:

1. ingests one or more `.oh5` scan files with config-defined labels,
2. filters low-quality pixels using configurable thresholds and safe logical expressions,
3. prepares deterministic train/val/test datasets with optional leakage-safe grouping and split caps,
4. trains configurable classifier backbones (scratch or pretrained),
5. emits machine-readable run artifacts for auditability.

## 2. Input Data Contract

Supported dataset-prep modes and schema contracts:

1. `input_mode: oh5_csv_labels` (legacy/backward compatible)
   - per source: one `.oh5` file + one CSV with per-pixel labels.
2. `input_mode: single_phase_scan_map` (legacy/backward compatible)
   - per source: one `.oh5` file + one file-level phase assignment (`phase_name` or `phase_label`).
   - all accepted pixels from that file inherit that phase label.

3. `schema_version: phase_id_xcorr.ml_dataset_prep.v3` (new concise contract)
   - top-level `data_source_folder` + `listOfFiles` entries, optional per-file metadata (`scan_id`, `phase_name`, `phase_label`).
   - optional `allow_filename_phase_fallback` to infer phase token from filename when explicitly enabled.

Required `.oh5` capabilities for ML prep:

- grid metadata: `nColumns`, `nRows`,
- pattern dataset: `Pattern` (or alias-compatible key),
- optional quality fields for filtering: `CI`/`Confidence Index`, `IQ`/`Image Quality`, `Fit`, `Valid`.

If `Pattern` is missing:

- `strict_pattern_presence=true` (default): fail fast.
- `strict_pattern_presence=false`: source is skipped with explicit reason in manifest.

CSV label requirements:

- location by either `(x, y)` or `flat_index`,
- phase by either `phase_name` or numeric `phase_label`.

## 3. Config-First Control Surface

### Dataset Prep Config

Templates:

- `configs/ml/dataset_prepare.default.yml`
- `configs/ml/dataset_prepare.single_phase_scan_map.debug.yml`
- `configs/ml/dataset_prepare.v3_al_ni_cu.example.yml`

Controls:

- phase name to label mapping,
- list of sources with mode-specific labeling fields,
- quality policy (thresholds and optional logical expression, with alias mapping),
- dataset-stage preprocessing policy (`preprocessing.resize_hw`, masking, normalization),
- optional phase balancing policy (`phase_balancing.equalize_to_min_count`) to downsample each accepted phase to the smallest accepted phase count before splitting,
- split policy (`train/val/test`, seed, stratified, `group_key`, `max_val_samples`, `max_test_samples`, optional `val_samples_per_phase`, optional `test_samples_per_phase`).

### Training Config

Templates:

- `configs/ml/train.convnextv2_nano.pretrained.debug.yml`
- `configs/ml/train.simple_cnn.debug.yml`
- `configs/ml/train.base_timm.debug.yml`

Controls:

- dataset manifest path,
- model family and model name,
- pretrained on/off,
- optimizer and runtime settings,
- input preprocessing for training (`resize_hw`, mask policy, normalization).
- if the dataset manifest already includes `preprocessing_policy`, that dataset-stage preprocessing is treated as authoritative and any training-side resize/mask overrides are ignored with an explicit warning/event.

### Benchmark Suite Config

- `configs/ml/benchmark_suite.debug.yml`

Controls:

- base train config,
- experiment list,
- per-experiment key-value overrides,
- unified summary output root.

## 4. Module Boundaries

`src/phase_id_xcorr/ml/` modules:

- `oh5_reader.py`: robust `.oh5` access, field aliasing, pattern/quality extraction.
- `labels.py`: CSV label normalization and validation.
- `quality.py`: threshold gates plus safe expression evaluation and alias resolution.
- `split.py`: deterministic split assignment with optional grouping and val/test caps.
- `preprocessing_policy.py`: dataset-stage preprocessing contract + fingerprinting.
- `dataset_builder.py`: end-to-end prep orchestration and artifact writing.
- `dataset_io.py`: NPZ/JSON/CSV helpers.
- `models.py`: classifier factory (`timm` + local `simple_cnn`).
- `metrics.py`: confusion matrix + accuracy/macro-F1 metrics.
- `training.py`: training/evaluation/checkpoint/report workflow.
- `suite.py`: multi-model benchmark orchestration.

## 5. Main Commands

Prepare dataset:

```bash
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.default.yml --debug
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.single_phase_scan_map.debug.yml --debug
```

Train single model:

```bash
python scripts/run_ml_train_classifier.py --config configs/ml/train.convnextv2_nano.pretrained.debug.yml --debug
```

Run benchmark suite:

```bash
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.debug.yml --debug
```

Run one-go full cycle (dataset prep -> suite -> HTML + optional PPTX):

```bash
python scripts/run_ml_full_cycle.py --config configs/ml/full_cycle.debug.yml --debug
```

Run raw-data phase explorer GUI:

```bash
python scripts/run_ml_phase_explorer.py --config configs/ml/dataset_prepare.v3_al_ni_cu.example.yml --debug
```

For platform-specific command notes (Windows/PyCharm, Linux, HPC), see `docs/ml_input_data_runbook.md`.
For one-command benchmark-to-PPT execution, see `docs/ml_training_inference_workflow.md`.
For raw `.oh5` exploratory GUI analytics (histograms/CDF/interactive intensity masks), see `docs/ml_phase_explorer_gui.md`.

## 6. Artifact Contract

Each run writes machine-readable artifacts first, human-readable summaries second.

Dataset prep outputs:

- `manifest.json`
- `events.jsonl`
- `records.csv`
- `splits/train.npz`, `splits/val.npz`, `splits/test.npz`
- when phase balancing is enabled, the manifest records both pre-balance qualified counts and post-balance selected counts per phase

Training outputs:

- `manifest.json`
- `events.jsonl`
- `report.json` (schema: `phase_id_xcorr.ml_training_report.v1`)
- `epoch_history.jsonl`
- `best_checkpoint.pt`, `last_checkpoint.pt`
- `report.md`

Suite outputs:

- `manifest.json`
- `events.jsonl`
- `suite_summary.json`
- `suite_summary.md`
- `suite_report.html` (interactive, artifact-linked analytics)

Full-cycle outputs:

- `manifest.json`
- `events.jsonl`
- `full_cycle_summary.json`
- `full_cycle_summary.html`
- resolved dataset/suite/train configs
- optional `.pptx`

Required reporting content:

- config path and resolved run settings,
- seed/device/runtime,
- per-split counts and phase distributions,
- confusion matrix and macro metrics,
- artifact pointers using repo-relative paths.

## 7. Reliability and Reproducibility Rules

- Default to deterministic seed control.
- Always preserve source `.oh5` read-only.
- Keep full reject reasons for filtered-out samples.
- Do not hard-code phase names in code.
- Any behavior/config/schema change must update docs and `todo_list.md` in the same change.
