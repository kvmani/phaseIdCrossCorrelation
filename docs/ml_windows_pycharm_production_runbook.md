# ML Production Runbook (Windows 11 + PyCharm Terminal)

Use this runbook for the actual `Ni-Cu-Al` training-data run on a Windows 11 machine with a local virtual environment.

## 1. Files Prepared For You

Production configs:

- `configs/ml/phase_explorer.ni_cu_al.production.yml`
- `configs/ml/dataset_prepare.ni_cu_al.production.yml`
- `configs/ml/train.ni_cu_al.production.base.yml`
- `configs/ml/benchmark_suite.ni_cu_al.production.yml`
- `configs/ml/full_cycle.ni_cu_al.production.yml`

Portable PPT generator:

- `scripts/ml_results_presentation/generate_lab_meeting_ppt.py`

## 2. What You Need To Edit

Only edit these fields unless your workflow changes materially.

### 2.1 Explorer and dataset file names and root

Files:

- `configs/ml/phase_explorer.ni_cu_al.production.yml`
- `configs/ml/dataset_prepare.ni_cu_al.production.yml`

Check and update:

- `data_source_folder`
  - currently set to `F:\PhaseID_Training_Data\Ni-Cu-Al_Scans`
- `listOfFiles`
  - currently set to `Al-1.oh5`, `Ni.oh`, `Cu-1.oh5`
  - if the nickel filename is actually `Ni.oh5` on disk, update that one line in both files
- `scan_id`
  - optional; keep concise and stable

### 2.2 Quality filter

File: `configs/ml/dataset_prepare.ni_cu_al.production.yml`

Current filter:

```yaml
quality_filters:
  expression: "CI > 0.4 && Fit < 1.5"
```

Change only if the scientific gate changes.

### 2.3 Holdout size per phase

File: `configs/ml/dataset_prepare.ni_cu_al.production.yml`

Current production-ready holdout:

```yaml
split:
  val_samples_per_phase: 1000
  test_samples_per_phase: 1000
```

Change these two numbers only if you want different per-phase validation/test sizes.

### 2.4 Training runtime

File: `configs/ml/train.ni_cu_al.production.base.yml`

Typical knobs to change:

- `epochs`
- `batch_size`
- `device`
- `amp`
- `input.normalize`
  - kept at identity (`mean: [0.0]`, `std: [1.0]`) so there is no extra training-time dataset normalization on top of the raw intensity scaling from the `.oh5` reader

### 2.5 Benchmark model set

File: `configs/ml/benchmark_suite.ni_cu_al.production.yml`

Edit only if you want to add/remove model variants.

## 3. PyCharm Terminal Commands

Open the PyCharm terminal in the repository root.

### 3.1 Activate the local virtual environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

`cmd.exe`:

```bat
.\.venv\Scripts\activate
```

### 3.2 Install required packages

```powershell
python -m pip install --upgrade pip
python -m pip install numpy h5py pillow pyyaml torch timm python-pptx
```

If you also want the desktop explorer GUI on Windows:

```powershell
python -m pip install PySide6 pyqtgraph
```

## 4. Exact Run Commands

### 4.1 Explore the raw data first

```powershell
python scripts\run_ml_phase_explorer.py --config configs\ml\phase_explorer.ni_cu_al.production.yml --debug
```

Use this GUI to inspect:

- phase-wise intensity histograms
- CI / IQ / Fit histograms
- representative grayscale Kikuchi patterns

### 4.2 Prepare dataset only

```powershell
python scripts\run_ml_dataset_prepare.py --config configs\ml\dataset_prepare.ni_cu_al.production.yml --debug
```

Outputs to check:

- `reports/ml/datasets/ni_cu_al_production/manifest.json`
- `reports/ml/datasets/ni_cu_al_production/summary.html`
- `reports/ml/datasets/ni_cu_al_production/events.jsonl`
- `reports/ml/datasets/ni_cu_al_production/records.csv`

The dataset HTML summary includes:

- accepted and rejected scan-pixel counts and fractions
- train / val / test counts and split composition
- phase-wise percentages in each split
- mean / median / std for CI, Fit, and IQ by phase
- modal intensity values and modal pixel counts by phase

### 4.3 Train one baseline model only

```powershell
python scripts\run_ml_train_classifier.py --config configs\ml\train.ni_cu_al.production.base.yml --debug
```

### 4.4 Run the full benchmark suite

```powershell
python scripts\run_ml_benchmark_suite.py --config configs\ml\benchmark_suite.ni_cu_al.production.yml --debug
```

Outputs to check:

- `reports/ml/benchmarks/ni_cu_al_production/suite_summary.json`
- `reports/ml/benchmarks/ni_cu_al_production/suite_report.html`

The benchmark HTML summary includes:

- dataset overview and split composition
- best-model summary
- model-to-model comparison table
- validation evolution summaries
- confusion matrices
- per-class precision / recall / F1
- links to each resolved train config and training report

### 4.5 Generate the lab-meeting PPTX from benchmark artifacts

```powershell
python scripts\run_ml_suite_with_ppt.py --config configs\ml\benchmark_suite.ni_cu_al.production.yml --debug --ppt-script scripts\ml_results_presentation\generate_lab_meeting_ppt.py --deck-title "Ni-Cu-Al Phase ID - Lab Meeting"
```

Expected PPT outputs:

- `reports/ml/presentations/ni-cu-al-phase-id-lab-meeting_manifest.json`
- `reports/ml/presentations/ni-cu-al-phase-id-lab-meeting.pptx`

### 4.6 One-go full pipeline

This is the shortest path once filenames and holdout counts are correct.

```powershell
python scripts\run_ml_full_cycle.py --config configs\ml\full_cycle.ni_cu_al.production.yml --debug
```

This runs:

1. dataset prep
2. benchmark suite
3. full-cycle JSON + HTML summary
4. final PPTX generation

## 5. Minimum Check Before Starting The Real Run

Confirm these are true:

- all `.oh5` filenames in `listOfFiles` are correct
- the `Ni.oh` filename is correct on disk, or changed to `Ni.oh5` if needed
- the `F:\PhaseID_Training_Data\Ni-Cu-Al_Scans` path is reachable from the Windows machine
- `val_samples_per_phase` and `test_samples_per_phase` match the intended holdout size
- `python -m pip show python-pptx` succeeds
- one baseline training run completes before launching the full suite

## 6. Recommended First Production Sequence

1. Run the explorer GUI.
2. Run dataset prep.
3. Open `summary.html` and confirm accepted/rejected counts, split composition, CI/Fit/IQ stats, and intensity modes are scientifically reasonable.
4. Run one baseline training config.
5. Run the benchmark suite and inspect `suite_report.html`.
6. Generate the PPTX.
7. If all looks clean, use the one-go full-cycle config for repeatable reruns.
