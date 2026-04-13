# phaseIdCrossCorrelation

Accuracy-first EBSD phase identification for mixed-phase scans using two complementary evidence tracks:

- masked NCC against externally simulated or curated patterns,
- supervised ML classification from experimental Kikuchi patterns.

## Canonical Documentation

The primary user-facing documentation surface is the Sphinx site under `docs/site/`.

Install the docs dependencies:

```powershell
python -m pip install -r .\docs\requirements.txt
```

Build the HTML site:

```powershell
python .\scripts\build_docs.py --clean
```

Build and open it:

```powershell
python .\scripts\build_docs.py --clean --open
```

Main local review entry point:

- `docs/_build/html/index.html`

## Start Here

- Sphinx source home: `docs/site/index.md`
- Legacy docs bridge: `docs/README.md`
- Scientific scope and success criteria: `docs/mission_statement.md`
- Current implementation status: `docs/status.md`
- Active work queue: `todo_list.md`
- Repo rules: `AGENTS.md`

## Main Commands

### Dataset prep

```powershell
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.data_march2026.balanced.debug.yml --debug
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.data_march2026.balanced.yml --debug
```

### Benchmark suite

```powershell
python .\scripts\run_ml_benchmark_suite.py --config .\configs\ml\benchmark_suite.data_march2026.balanced.debug.yml --debug
python .\scripts\run_ml_benchmark_suite.py --config .\configs\ml\benchmark_suite.data_march2026.balanced.yml --debug
```

### Full cycle

```powershell
python .\scripts\run_ml_full_cycle.py --config .\configs\ml\full_cycle.data_march2026.balanced.debug.yml --debug
python .\scripts\run_ml_full_cycle.py --config .\configs\ml\full_cycle.data_march2026.balanced.yml --debug
```

### Inference and GUIs

```powershell
python .\scripts\run_ml_inference_gui.py --suite-root .\reports\ml\benchmarks\data_march2026_balanced_3scansEach
python .\scripts\run_oh5_crop_gui.py --input path\to\source_scan.oh5 --debug
python .\scripts\run_ml_oh5_sample_inference.py --config .\configs\ml\oh5_sample_inference.data_march2026.example.yml --debug
python .\scripts\run_ml_phase_explorer.py --config .\configs\ml\phase_explorer.ni_cu_al.production.yml --debug
python .\scripts\run_ml_diagnostic_gallery.py --config .\configs\ml\diagnostic_gallery.example.yml --debug
```

The inference GUI supports both single-image prediction and full-scan `.oh5` mapping. In full-scan mode it renders:

- the predicted phase map
- an IPF orientation reference from scan Euler angles when available
- an IPF-colored EBSD map from scan Euler angles when available
- live progress, ETA, and backend log messages for long-running scans

The dedicated `.oh5` crop GUI supports:

- rectangular crop selection on the source IQ map
- multiple crop rectangles from one source scan in a single export pass
- standalone cropped `.oh5` export named `{base_name}_crop_{row}_{col}.oh5`
- automatic original-vs-cropped review mode after export, with a selector for choosing which exported crop to inspect
- visible GUI logs and progress/status bar during load/export/reload
- side-by-side IQ/IPF/pattern validation with explicit original and cropped scan sizes shown above the panes

## Repository Layout

```text
phaseIdCrossCorrelation/
+-- AGENTS.md
+-- README.md
+-- docs/
|   +-- site/
|   +-- README.md
|   `-- ...
+-- configs/
+-- scripts/
+-- src/
+-- tests/
`-- reports/
```

## Design Principles

- correctness over speed
- reproducibility over convenience
- explicit manifests and reports for runnable workflows
- thin CLIs and modular `src/` code
- documentation treated as a first-class interface
