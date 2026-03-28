# ML Input Data Runbook

Operational guide for preparing ML datasets from `.oh5` scans and launching the training workflows.

## 1. End-to-End Flow

1. `run_ml_dataset_prepare.py` reads one YAML config.
2. For each source:
   - opens `.oh5`,
   - resolves labels using the configured input mode,
   - applies quality filters (`CI`, `IQ`, `Fit`, `Valid`),
   - stores accepted pattern + label + trace metadata.
3. Merges accepted samples across all sources.
4. Optionally balances accepted phase counts down to the smallest accepted phase before splitting.
5. Creates deterministic train/val/test splits.
6. Writes dataset artifacts (`manifest.json`, `records.csv`, split `.npz`, `events.jsonl`).
7. `run_ml_train_classifier.py` trains/evaluates from dataset manifest.
8. `run_ml_benchmark_suite.py` runs repeated model experiments.
9. Dataset summaries include per-phase split composition, CI/Fit/IQ mean-median-std, and modal intensity statistics.
10. Suite summaries include best-model selection, confusion matrices, per-class metrics, and links to per-run configs and reports.

## 2. Input Modes

Dataset prep supports two modes:

1. `input_mode: oh5_csv_labels`
   - each source has one `.oh5` + one CSV with per-pixel labels.
2. `input_mode: single_phase_scan_map`
   - each source has one `.oh5` + one file-level phase assignment (`phase_name` or `phase_label`).
   - all accepted pixels from that file inherit the same phase label.

If `input_mode` is omitted, mode is inferred from source fields.

## 3. `.oh5` Requirements

Required fields:

- `/<scan>/EBSD/Header/nColumns`
- `/<scan>/EBSD/Header/nRows`
- `/<scan>/EBSD/Data/Pattern` (alias `Patterns` supported)

Quality fields (optional but recommended):

- `CI` or `Confidence Index`
- `IQ` or `Image Quality`
- `Fit`
- `Valid`

Supported pattern layouts:

- flattened stack: `(nRows*nColumns, H, W)`
- gridded stack: `(nRows, nColumns, H, W)`

Pattern-missing behavior:

- `strict_pattern_presence: true` -> fail
- `strict_pattern_presence: false` -> skip source with recorded reason

## 4. Label Contract

### 4.1 `oh5_csv_labels`

CSV columns are configurable via `label_csv`.

Each row must provide:

- location: `flat_index` OR both `x` and `y`
- class: `phase_name` OR `phase_label`

CSV example:

```csv
sample_id,x,y,phase_name
s001_r00c00,0,0,fe_bcc
s001_r00c01,1,0,fe3o4_magnetite
```

### 4.2 `single_phase_scan_map`

Each source entry must include:

- `oh5_path`
- `phase_name` OR `phase_label`

Pixel sampling in this mode:

- iterate all pixels in the scan,
- apply quality filters per pixel,
- assign the same source phase label to each accepted pixel.

## 5. YAML Examples

### 5.1 CSV-Label Mode

```yaml
schema_version: phase_id_xcorr.ml_dataset_prep.v1
input_mode: oh5_csv_labels
output_dir: reports/ml/datasets/my_run_csv
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
phase_balancing:
  equalize_to_min_count: true
  # optional exact per-phase holdout sizes; remainder goes to train
  # val_samples_per_phase: 3
  # test_samples_per_phase: 3

sources:
  - scan_id: s001
    oh5_path: data/incoming/s001.oh5
    labels_csv_path: data/incoming/s001_labels.csv
  - scan_id: s002
    oh5_path: data/incoming/s002.oh5
    labels_csv_path: data/incoming/s002_labels.csv
```

### 5.2 Single-Phase Scan Map Mode

```yaml
schema_version: phase_id_xcorr.ml_dataset_prep.v2
input_mode: single_phase_scan_map
output_dir: reports/ml/datasets/my_run_single_phase
strict_pattern_presence: true

target_pattern_hw: [128, 128]

phase_labels:
  - name: fe_bcc
    label: 0
  - name: fe3o4_magnetite
    label: 1
  - name: feo_wustite
    label: 2

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
phase_balancing:
  equalize_to_min_count: true
  # optional exact per-phase holdout sizes; for example use 100 each in larger runs
  # val_samples_per_phase: 100
  # test_samples_per_phase: 100

sources:
  - scan_id: s001
    oh5_path: data/incoming/s001_fe_bcc.oh5
    phase_name: fe_bcc
  - scan_id: s002
    oh5_path: data/incoming/s002_fe3o4.oh5
    phase_label: 1
```

## 6. Commands by Environment

Use repository root as working directory for all commands.

### 6.1 Linux/macOS Terminal

```bash
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.default.yml --debug
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.single_phase_scan_map.debug.yml --debug
python scripts/run_ml_train_classifier.py --config configs/ml/train.convnextv2_nano.pretrained.debug.yml --debug
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.debug.yml --debug
```

### 6.2 Windows

PowerShell in PyCharm:

```powershell
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.default.yml --debug
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.single_phase_scan_map.debug.yml --debug
python .\scripts\run_ml_train_classifier.py --config .\configs\ml\train.convnextv2_nano.pretrained.debug.yml --debug
python .\scripts\run_ml_benchmark_suite.py --config .\configs\ml\benchmark_suite.debug.yml --debug
```

`cmd.exe` equivalent:

```bat
python scripts\run_ml_dataset_prepare.py --config configs\ml\dataset_prepare.default.yml --debug
```

### 6.3 HPC

Example batch script:

```bash
#!/bin/bash
#SBATCH --job-name=phaseid-ml-debug
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Activate your prebuilt environment here.
# Example:
# source /path/to/venv/bin/activate

python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.single_phase_scan_map.debug.yml --debug
python scripts/run_ml_train_classifier.py --config configs/ml/train.simple_cnn.debug.yml --debug
```

Submit:

```bash
sbatch run_ml_debug.slurm
```

## 7. Outputs

Dataset prep writes:

- `<output_dir>/manifest.json`
- `<output_dir>/records.csv`
- `<output_dir>/splits/train.npz`
- `<output_dir>/splits/val.npz`
- `<output_dir>/splits/test.npz`
- `<output_dir>/events.jsonl`

Manifest highlights:

- `input_mode`
- `phase_to_label`, `label_to_phase`
- `source_summaries` and per-source reject reasons
- `accepted_per_phase` for qualified post-filter counts before balancing
- `selected_per_phase` and `phase_balancing` for the final balanced dataset used downstream
- split counts and per-phase split counts
- quality filter policy and sanity checks

Training and suite runs also emit `manifest.json`, `events.jsonl`, and their model-specific reports.

## 8. Reliability Notes

- Source `.oh5` files are read-only.
- Deterministic seeds are mandatory for reproducible comparisons.
- Quality filtering occurs before sample inclusion.
- Phase names/labels are config-defined, never hard-coded.
- Single-phase scan-map mode assumes each `.oh5` file represents one ground-truth phase.

For a practical training/evaluation/benchmark/PPT sequence, see:

- `docs/ml_training_inference_workflow.md`
