# Architecture

Current implementation is organized around two EBSD evidence tracks:

- NCC and Hough-based curated evaluation,
- ML dataset preparation, training, and experiment reporting.

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
      dataset_builder.py
      dataset_io.py
      html_report.py
      labels.py
      metrics.py
      models.py
      oh5_reader.py
      phase_explorer.py
      phase_explorer_gui.py
      preprocessing_policy.py
      quality.py
      split.py
      suite.py
      training.py
```

## Workflow Boundaries

- `scripts/`: thin CLIs only.
- `src/phase_id_xcorr/`: reusable logic, data contracts, and artifact writing.
- `configs/`: versioned YAML templates.
- `tests/`: debug-scale unit and integration coverage.
- `reports/`: generated outputs, never canonical inputs.

## Main Data Contracts

Dataset prep inputs:

- YAML config with phase mapping, source list, quality policy, preprocessing policy, and split policy.
- `.oh5` sources with pattern arrays and optional quality fields.
- Optional per-pixel CSV labels, or file-level phase mapping for single-phase scans.

Dataset prep outputs:

- `records.csv`
- `splits/train.npz`, `splits/val.npz`, `splits/test.npz`
- `manifest.json`
- `events.jsonl`

Training outputs:

- `report.json`
- `report.md`
- `manifest.json`
- `events.jsonl`
- checkpoints and epoch history

Suite outputs:

- `suite_summary.json`
- `suite_summary.md`
- `suite_report.html`
- per-run training artifacts

## Design Constraints

- Configuration-driven (no hard-coded phase names).
- Robust `.oh5` field aliasing and clear failure modes when `Pattern` is absent.
- Deterministic split generation and reproducible seed control.
- Machine-readable run artifacts for every workflow run.
- Keep scripts orchestration-only.
