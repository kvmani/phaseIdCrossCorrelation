# Proposed Architecture (Implementation Blueprint)

This is the active blueprint for EBSD phase identification with two coordinated tracks:

- NCC evidence track (existing baseline)
- ML classifier track (new supervised branch)

## Module Map

```text
src/
  phase_id_xcorr/
    intake/
      g0_validator.py
    preprocessing/
      image_io.py
      masking.py
      pattern_prep.py
    similarity/
      ncc.py
    features/
      kikuchipy_hough.py
    evaluation/
      curated_ncc.py
      curated_hough_vs_ncc.py
    reporting/
      run_manifest.py
    ml/
      config.py
      oh5_reader.py
      labels.py
      quality.py
      split.py
      dataset_builder.py
      dataset_io.py
      models.py
      metrics.py
      training.py
      suite.py
```

## ML Data Contracts

Input contracts:

- YAML data-prep config:
  - phase definitions (`name`, `label`),
  - one or more `.oh5` + CSV label pairs,
  - quality thresholds,
  - split policy.
- CSV label table (per `.oh5`):
  - row coordinates (`x`, `y`) or `flat_index`,
  - phase assignment (`phase_name` or numeric label),
  - optional sample metadata columns.

Output contracts:

- `records.csv`: one row per accepted sample with source paths, coordinates, quality values, phase mapping, and split.
- `splits/{train,val,test}.npz`: pattern tensors + labels.
- `dataset_manifest.json`: run settings, field mappings, acceptance/rejection counts.

## Workflow Boundaries

1. `scripts/run_ml_dataset_prepare.py`
- Thin CLI wrapper.
- Resolves paths and config.
- Calls `phase_id_xcorr.ml.dataset_builder.prepare_ml_dataset(...)`.

2. `scripts/run_ml_train_classifier.py`
- Thin CLI wrapper.
- Loads training config + prepared dataset manifest.
- Calls `phase_id_xcorr.ml.training.train_classifier(...)`.

3. `phase_id_xcorr.ml.suite`
- Optional benchmark orchestration over multiple model configs.
- Writes model-comparison summary reports.

## Design Constraints

- Configuration-driven (no hard-coded phase names).
- Robust `.oh5` field aliasing and clear failure modes when `Pattern` is absent.
- Deterministic split generation and reproducible seed control.
- Machine-readable run artifacts for every workflow run.
- Keep scripts orchestration-only; core logic stays in `src/` modules.
