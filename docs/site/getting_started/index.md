# Getting Started

This section is the shortest reliable path from repository checkout to usable workflows and browsable documentation.

## 1. What This Repository Is For

The project targets **scientifically reliable phase discrimination in mixed-phase EBSD scans** where standard indexing can be unreliable. It keeps:

- an interpretable NCC baseline,
- a supervised ML branch,
- explicit provenance and machine-readable artifacts for both.

For scientific context, see {doc}`../mission/index`.

## 2. Install and Build the Documentation

Install the documentation dependencies:

```powershell
python -m pip install -r .\docs\requirements.txt
```

Build the site:

```powershell
python .\scripts\build_docs.py --clean
```

Build and open it directly:

```powershell
python .\scripts\build_docs.py --clean --open
```

## 3. Minimal Successful Workflow Sequence

For a first smoke path:

```powershell
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.single_phase_scan_map.debug.yml --debug
python .\scripts\run_ml_benchmark_suite.py --config .\configs\ml\benchmark_suite.debug.yml --debug
python .\scripts\run_ml_inference_gui.py --suite-root .\reports\ml\benchmarks\debug_suite
```

This sequence gives you:

- a prepared dataset with `manifest.json`, split `.npz`, HTML summary, Euler/IPF diagnostics
- a benchmark suite with per-run reports and a suite HTML report
- an inference GUI that can load models from the suite root

## 4. Core Command Surfaces

### Dataset preparation

```powershell
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.data_march2026.balanced.debug.yml --debug
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.data_march2026.balanced.yml --debug
```

### Benchmark suite

```powershell
python .\scripts\run_ml_benchmark_suite.py --config .\configs\ml\benchmark_suite.data_march2026.balanced.debug.yml --debug
python .\scripts\run_ml_benchmark_suite.py --config .\configs\ml\benchmark_suite.data_march2026.balanced.yml --debug
```

### Full-cycle orchestration

```powershell
python .\scripts\run_ml_full_cycle.py --config .\configs\ml\full_cycle.data_march2026.balanced.debug.yml --debug
python .\scripts\run_ml_full_cycle.py --config .\configs\ml\full_cycle.data_march2026.balanced.yml --debug
```

### Inference

```powershell
python .\scripts\run_ml_inference.py --run-dir .\reports\ml\benchmarks\data_march2026_balanced_3scansEach\simple_cnn_w32 --image path\to\pattern.png --device auto
python .\scripts\run_ml_inference_gui.py --suite-root .\reports\ml\benchmarks\data_march2026_balanced_3scansEach
python .\scripts\run_ml_oh5_sample_inference.py --config .\configs\ml\oh5_sample_inference.data_march2026.example.yml --debug
```

## 5. Where To Go Next

- For mission and scientific posture: {doc}`../mission/index`
- For theory and formulas: {doc}`../concepts/index`
- For end-to-end run instructions: {doc}`../workflows/index`
- For GUI usage: {doc}`../guis/index`
- For config and script reference: {doc}`../reference/index`
