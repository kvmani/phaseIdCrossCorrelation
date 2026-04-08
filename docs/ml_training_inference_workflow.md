# ML Training and Benchmark Workflow

This is the shortest path from prepared dataset to experiment review.

## 1. Current Meaning of "Inference"

Current ML CLI support includes:

- dataset preparation (`run_ml_dataset_prepare.py`)
- model training + held-out test evaluation (`run_ml_train_classifier.py`)
- multi-model benchmark suite (`run_ml_benchmark_suite.py`)
- single-image inference CLI (`run_ml_inference.py`)
- sampled unseen-scan `.oh5` inference CLI (`run_ml_oh5_sample_inference.py`)
- desktop inference GUI (`run_ml_inference_gui.py`) with both single-image and full-scan `.oh5` modes

In this repository, "inference" includes evaluation on validation/test splits during training or suite runs, single-image saved-model inference, and sampled unseen-scan `.oh5` inference through a dedicated CLI.

## 2. Recommended Flow

1. Prepare dataset from `.oh5` scans:
   - use `oh5_csv_labels` mode (per-pixel labels), or
   - use `single_phase_scan_map` mode (one scan file = one phase).
   - inspect generated Euler/IPF diagnostics to confirm split-wise orientation coverage by phase.
2. Train one model and inspect metrics/checkpoints.
3. Run benchmark suite across model variants.
4. Auto-generate a lab-meeting PPTX from suite artifacts.
5. Use the saved best model for unknown-image inference and qualitative review.
6. Use the inference GUI for either:
   - single unknown-image review, or
   - full `.oh5` scan rendering as a predicted phase map with optional confidence shading.
7. Prefer one-go execution with full-cycle orchestration for reproducible and machine-ingestible reporting.

## 3. Commands

Run from repository root.

### 3.1 Dataset Preparation

CSV label mode:

```bash
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.default.yml --debug
```

Single-phase scan-map mode:

```bash
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.single_phase_scan_map.debug.yml --debug
```

### 3.2 Single-Model Training

```bash
python scripts/run_ml_train_classifier.py --config configs/ml/train.convnextv2_nano.pretrained.debug.yml --debug
```

Main outputs under configured `output_dir`:

- `report.json` (test metrics, confusion matrix, model metadata)
- `manifest.json`
- `best_checkpoint.pt`, `last_checkpoint.pt`
- `epoch_history.jsonl`, `events.jsonl`

### 3.3 Benchmark Suite (Multi-Model)

```bash
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.debug.yml --debug
```

Main outputs under suite `output_root`:

- `suite_summary.json`
- `suite_summary.md`
- per-experiment run folders with training artifacts

### 3.4 Auto PPTX Generation

One command to run suite and build slides:

```bash
python scripts/run_ml_suite_with_ppt.py \
  --config configs/ml/benchmark_suite.debug.yml \
  --debug \
  --deck-title "phaseIdCrossCorrelation ML Benchmark - Lab Meeting"
```

By default this script runs the suite, scans the suite output, builds a `.pptx`, and writes presentation artifacts to `reports/ml/presentations`.

Use `--skip-ppt` if you only want suite execution.

### 3.5 Lightweight Transfer Bundle

After the suite finishes, package only the lightweight artifacts for email or cross-machine review:

```bash
python scripts/package_ml_benchmark_suite.py \
  --suite-root reports/ml/benchmarks/classiication_training_data_smoke \
  --output-zip reports/ml/benchmarks/classiication_training_data_smoke/classiication_training_data_smoke_lightweight_bundle.zip \
  --extra-path reports/ml/presentations/classification-training-data-smoke-lab-meeting.pptx \
  --extra-path reports/ml/presentations/classification-training-data-smoke-lab-meeting_manifest.json
```

The bundle preserves the repo folder structure, keeps lightweight summaries such as `.json`, `.jsonl`, `.html`, `.md`, `.yml`, `.yaml`, `.csv`, and `.pptx`, excludes heavy checkpoints and tensor bundles, and auto-includes the referenced dataset manifest and dataset summary artifacts used by the suite.

### 3.6 Inference From A Saved Model

CLI:

```bash
python scripts/run_ml_inference.py \
  --run-dir reports/ml/benchmarks/ni_cu_al_production/simple_cnn_w32 \
  --image path/to/unknown_pattern.png \
  --device auto
```

GUI:

```bash
python scripts/run_ml_inference_gui.py \
  --suite-root reports/ml/benchmarks/ni_cu_al_production
```

GUI highlights:

- single-image mode preserves the original/preprocessed preview workflow
- full-scan `.oh5` mode runs inference across every pattern in the scan
- predicted scan maps use per-class colors and optional confidence-based dulling for low-score pixels

### 3.7 Sampled Inference From Unseen `.oh5` Scans

Use this when you have new scans under a folder such as `F:/PhaseID_Training_Data/Data_March2026/Different_Condition/Ni/*.oh5` and want to test a previously trained CNN by randomly sampling filtered pixels from each scan.

```bash
python scripts/run_ml_oh5_sample_inference.py \
  --config configs/ml/oh5_sample_inference.ni_different_condition.example.yml \
  --debug
```

Key config behavior:

- supports both absolute and repo-relative paths for `run_dir`, `output_dir`, `input_root`, and scan file entries
- applies the same expression-style quality filtering used during dataset prep, for example `CI > 0.5 && Fit < 1.0`
- samples `n` valid patterns per scan using a deterministic seed
- computes accuracy only for scans with `expected_phase`
- writes:
  - `sample_predictions.json`
  - `sample_predictions.csv`
  - `scan_summary.csv`
  - `summary.json`
  - `summary.md`
  - `manifest.json`
- prints a compact terminal table with `oh5_file`, `x`, `y`, `index`, `predicted_phase`, and `score`

### 3.8 Diagnostic Pattern Gallery

Use this when you want side-by-side inspection of reference phase scans and anonymous unseen scans, with direct access to the exact pattern indices that were displayed:

```bash
python scripts/run_ml_diagnostic_gallery.py \
  --config configs/ml/diagnostic_gallery.example.yml \
  --debug
```

The gallery GUI is optimized for diagnosis rather than training. It supports:

- drag-and-drop source lists for reference and unknown `.oh5` files,
- reproducible sampling with quality and prediction filters,
- manual pattern-index lookup for exact pixel inspection,
- raw vs preprocessed preview tabs,
- JSON manifest plus PNG contact-sheet export for slide assembly.

### 3.9 One-Go Full Cycle (Recommended for local conflict-free reproducibility)

```bash
python scripts/run_ml_full_cycle.py --config configs/ml/full_cycle.debug.yml --debug
```

April 2026 Cu/Ni-only balanced production run:

```bash
python scripts/run_ml_full_cycle.py --config configs/ml/full_cycle.april2026_cu_ni_balanced.yml --debug
```

This one-go workflow performs:

1. dataset prep from raw `.oh5` config,
2. benchmark suite training across configured model variants,
3. suite-level JSON/Markdown/HTML outputs,
4. optional PPTX generation,
5. full-cycle manifest + event log + concise HTML big-picture summary linking drill-down artifacts.

## 4. Platform-Specific Notes

Detailed platform command variants (Linux/macOS, Windows PyCharm terminal, HPC/SLURM) are documented in:

- `docs/ml_input_data_runbook.md`

## 5. Suggested Review Rhythm

1. Run one baseline training config and verify test metrics are stable.
2. Run benchmark suite with 3-5 model variants or preprocessing ablations.
3. Generate PPTX using `run_ml_suite_with_ppt.py`.
4. Inspect:
   - class-wise confusion trends
   - macro-F1 vs runtime tradeoffs
   - failure cases that repeat across models
5. Update next experiment batch based on slide conclusions.
