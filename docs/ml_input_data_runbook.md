# ML Input Data and Pipeline Runbook

This is the operational, no-ambiguity guide for preparing `.oh5` + CSV inputs and running the ML classifier pipeline.

Use this document when creating new datasets for training.

## 1. End-to-End Flow (What Happens)

1. `run_ml_dataset_prepare.py` reads YAML config.
2. For each source pair (`.oh5` + CSV):
   - loads scan grid and pattern/quality fields from `.oh5`,
   - loads labels from CSV,
   - maps each CSV row to one `.oh5` pixel,
   - applies quality filters,
   - keeps accepted samples with full trace metadata.
3. Combines accepted samples across all sources.
4. Creates deterministic train/val/test splits.
5. Writes dataset artifacts and `manifest.json` for downstream training.
6. `run_ml_train_classifier.py` consumes the dataset manifest and trains/evaluates.
7. `run_ml_benchmark_suite.py` runs multiple training configs and writes comparative summaries.

## 2. Required Input Files

Per source scan:

- one `.oh5` file,
- one CSV label file.

### 2.1 `.oh5` Requirements

Minimum required paths:

- `/<scan>/EBSD/Header/nColumns`
- `/<scan>/EBSD/Header/nRows`
- `/<scan>/EBSD/Data/Pattern` (or alias `Patterns`)

Optional but strongly recommended quality fields:

- `CI` or `Confidence Index`
- `IQ` or `Image Quality`
- `Fit`
- `Valid`

Supported pattern layout:

- flattened stack: `(nRows*nColumns, H, W)`
- gridded stack: `(nRows, nColumns, H, W)`

If `Pattern` is missing:

- `strict_pattern_presence: true` -> run fails,
- `strict_pattern_presence: false` -> source is skipped and recorded.

## 3. CSV Label Contract

Column names are configurable (`label_csv` section in YAML).

Each row must provide:

- sample location: either `flat_index` OR both `x` and `y`,
- phase assignment: either `phase_name` OR numeric `phase_label`.

### 3.1 Coordinate-Based CSV Example

```csv
sample_id,x,y,phase_name
s001_r00c00,0,0,fe_bcc
s001_r00c01,1,0,fe3o4_magnetite
s001_r00c02,2,0,feo_wustite
```

### 3.2 Flat-Index CSV Example

```csv
sample_id,flat_index,phase_label
s001_i000000,0,0
s001_i000001,1,1
s001_i000002,2,2
```

## 4. How CSV Rows Map to `.oh5` Pixels

Mapping algorithm used in code:

1. Resolve pixel index:
   - if `flat_index` is present, use it directly,
   - else compute `flat_index = y * nColumns + x`.
2. Read quality values at that same pixel index.
3. Apply quality thresholds.
4. If accepted, read pattern at that same pixel index.
5. Convert pattern to float32 in `[0, 1]`.
6. Assign label using YAML phase mapping.
7. Store sample with ID: `"{scan_id}__{sample_id}"`.

This guarantees that pattern, quality, and label refer to exactly the same pixel.

## 5. YAML Data-Prep Example (Minimal)

```yaml
schema_version: phase_id_xcorr.ml_dataset_prep.v1
output_dir: reports/ml/datasets/my_run
strict_pattern_presence: true

target_pattern_hw: [128, 128]

phase_labels:
  - name: fe_bcc
    label: 0
  - name: fe3o4_magnetite
    label: 1
  - name: feo_wustite
    label: 2

label_csv:
  sample_id_col: sample_id
  x_col: x
  y_col: y
  flat_index_col: ""
  phase_name_col: phase_name
  phase_label_col: ""

quality_filters:
  confidence_index_min: 0.10
  image_quality_min: 0.0
  fit_max: 4.0
  valid_required: true

split:
  train: 0.70
  val: 0.15
  test: 0.15
  seed: 42
  stratified: true

sources:
  - scan_id: s001
    oh5_path: data/incoming/s001.oh5
    labels_csv_path: data/incoming/s001_labels.csv
  - scan_id: s002
    oh5_path: data/incoming/s002.oh5
    labels_csv_path: data/incoming/s002_labels.csv
```

## 6. Sanity Checks Performed

### 6.1 Dataset Preparation

- phase mapping exists and labels are unique,
- source list is non-empty,
- each source has `oh5_path` and `labels_csv_path`,
- CSV rows have valid location and valid phase assignment,
- `.oh5` has valid scan group and grid dimensions,
- pattern presence policy enforced (`strict_pattern_presence`),
- per-sample quality thresholds applied,
- all accepted patterns are 2D,
- final pattern shape is uniform across all accepted samples,
- every record gets an assigned split.

### 6.2 Training

- dataset manifest exists,
- class count and label mapping are valid,
- train/val/test splits are non-empty,
- history and checkpoints are written.

### 6.3 Benchmark Suite

- base train config exists,
- experiment list is non-empty,
- suite summaries are written.

## 7. Logging and Progress Contract

All ML operations now emit:

- human-readable logs (`INFO` level),
- structured event timeline (`events.jsonl`),
- elapsed time for each event,
- progress percentage and ETA for long loops.

### 7.1 Dataset Prep Events

Main events:

- `RUN_START`, `SOURCE_START`, `OH5_OPEN`, `LABELS_LOADED`,
- `SOURCE_PROGRESS` (with `progress_pct`, `eta_seconds`),
- `SOURCE_END`, `SPLIT_ASSIGNMENT_COMPLETE`, `SPLIT_WRITE_COMPLETE`,
- `RECORDS_WRITE_COMPLETE`, `RUN_END`.

### 7.2 Training Events

Main events:

- `RUN_START`, `SPLITS_LOADED`, `PATTERN_PREPROCESS_COMPLETE`,
- `DATALOADERS_READY`, `MODEL_READY`, `TRAIN_LOOP_START`,
- `EPOCH_START`, `EPOCH_END` (with `progress_pct`, `eta_seconds`),
- `BEST_CHECKPOINT_UPDATED`, `LAST_CHECKPOINT_SAVED`,
- `TEST_EVAL_COMPLETE`, `REPORT_WRITE_COMPLETE`, `MANIFEST_WRITE_COMPLETE`, `RUN_END`.

### 7.3 Suite Events

Main events:

- `RUN_START`,
- `EXPERIMENT_START` (with suite progress + ETA),
- `EXPERIMENT_END`,
- `RUN_END`.

## 8. Manifest Contract (For Downstream Ingestion)

### 8.1 Dataset Prep Manifest

Path: `<output_dir>/manifest.json`

Used by training script via `dataset_manifest_path`.

Includes:

- source summaries,
- split counts and per-phase counts,
- quality filter policy,
- sanity checks,
- artifact pointers (`records.csv`, split `.npz`, `events.jsonl`).

### 8.2 Training Manifest

Path: `<training_output_dir>/manifest.json`

Includes:

- model/device/config provenance,
- timing and sanity checks,
- artifact pointers (`report.json`, checkpoints, `epoch_history.jsonl`, `events.jsonl`).

### 8.3 Benchmark Suite Manifest

Path: `<suite_output_root>/manifest.json`

Includes:

- suite config provenance,
- run-level timing and sanity checks,
- artifact pointers (`suite_summary.json`, `suite_summary.md`, `events.jsonl`).

## 9. Scientific Rationale for Key Choices

- **Quality filtering before training:** low-CI/invalid patterns can inject label noise and destabilize learned decision boundaries.
- **Deterministic stratified split:** preserves class balance and makes benchmark runs reproducible and comparable.
- **Pattern normalization to `[0,1]` + optional resizing:** enables consistent model input scale across mixed bit-depth scans.
- **Circular mask option at training:** suppresses non-physical border/background artifacts common in EBSD detector frames.
- **Traceable manifests/events:** supports scientific auditability and prevents silent pipeline drift.

## 10. Assumptions and Constraints

- Source `.oh5` files are read-only.
- CSV ground truth quality is the limiting factor for supervised performance.
- Phase names are not hard-coded; they must be declared in YAML.
- Current classifier path assumes single-channel grayscale patterns.
- This branch is EBSD-only; LRS integration remains later-phase.

## 11. Run Commands (Recommended Order)

1. Prepare dataset:

```bash
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.default.yml --debug
```

2. Train single model:

```bash
python scripts/run_ml_train_classifier.py --config configs/ml/train.simple_cnn.debug.yml --debug
```

3. Run multi-model suite:

```bash
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.debug.yml --debug
```

## 12. Quick Troubleshooting

- `Pattern dataset missing`:
  - verify `.oh5` contains `EBSD/Data/Pattern`, or set `strict_pattern_presence: false` to skip that source.
- `Unknown phase_name` in CSV:
  - add that phase to `phase_labels` mapping.
- `Pattern shapes differ after preprocessing`:
  - set `target_pattern_hw` in dataset prep config.
- Too few accepted samples:
  - relax `quality_filters` thresholds and inspect `rejected_reason_counts` in manifest.
