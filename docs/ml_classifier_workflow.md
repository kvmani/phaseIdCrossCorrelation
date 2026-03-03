# ML Classifier Workflow

This document defines the ML branch for EBSD phase classification from experimental Kikuchi patterns.

For detailed input preparation examples, CSV-to-`.oh5` mapping rules, sanity checks, event logging, and manifest contracts, see:

- `docs/ml_input_data_runbook.md`

## 1. Objective

Build a reproducible supervised classifier pipeline that:

1. ingests one or more `.oh5` scan files and corresponding CSV labels,
2. filters low-quality pixels using configurable metrics,
3. prepares deterministic train/val/test datasets,
4. trains configurable classifier backbones (scratch or pretrained),
5. emits machine-readable run artifacts for auditability.

## 2. Input Data Contract

Per source pair:

- `.oh5` file containing EBSD scan data.
- CSV with per-pixel labels.

Required `.oh5` capabilities for ML prep:

- grid metadata: `nColumns`, `nRows`,
- pattern dataset: `Pattern` (or alias-compatible key),
- optional quality fields for filtering: `CI`/`Confidence Index`, `IQ`/`Image Quality`, `Fit`, `Valid`.

If `Pattern` is missing:

- `strict_pattern_presence=true` (default): fail fast.
- `strict_pattern_presence=false`: source is skipped with explicit reason in manifest.

CSV label requirements (configurable column names):

- location by either `(x, y)` or `flat_index`,
- phase by either `phase_name` or numeric `phase_label`.

## 3. Config-First Control Surface

### Dataset Prep Config

Template: `configs/ml/dataset_prepare.default.yml`

Controls:

- phase name to label mapping,
- list of `.oh5` + CSV sources,
- quality thresholds,
- split policy (`train/val/test`, seed, stratified),
- optional pattern resizing (`target_pattern_hw`).

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

### Benchmark Suite Config

Template: `configs/ml/benchmark_suite.debug.yml`

Controls:

- base train config,
- experiment list,
- per-experiment key-value overrides,
- unified summary output root.

## 4. Module Boundaries

`src/phase_id_xcorr/ml/` modules:

- `oh5_reader.py`: robust `.oh5` access, field aliasing, pattern/quality extraction.
- `labels.py`: CSV label normalization and validation.
- `quality.py`: threshold gates and rejection reasons.
- `split.py`: deterministic split assignment.
- `dataset_builder.py`: end-to-end prep orchestration and artifact writing.
- `dataset_io.py`: NPZ/JSON/CSV helpers.
- `models.py`: classifier factory (`timm` + local `simple_cnn`).
- `metrics.py`: confusion matrix + accuracy/macro-F1 metrics.
- `training.py`: training/evaluation/checkpoint/report workflow.
- `suite.py`: multi-model benchmark orchestration.

## 5. Workflow Commands

Prepare dataset:

```bash
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.default.yml --debug
```

Train single model:

```bash
python scripts/run_ml_train_classifier.py --config configs/ml/train.convnextv2_nano.pretrained.debug.yml --debug
```

Run benchmark suite:

```bash
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.debug.yml --debug
```

## 6. Reporting Standard (Hydra-Inspired, Adapted)

Each run writes machine-readable artifacts first, human-readable summaries second.

Dataset prep outputs:

- `manifest.json`
- `events.jsonl`
- `records.csv`
- `splits/train.npz`, `splits/val.npz`, `splits/test.npz`

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
