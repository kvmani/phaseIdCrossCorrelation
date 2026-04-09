# Inference Workflows

Inference in this repository exists in three forms:

1. one-image CLI inference
2. sampled `.oh5` CLI inference
3. suite-level full-scan `.oh5` CLI inference and export
4. GUI inference, including full-scan `.oh5` map rendering

## Single-image CLI

```powershell
python .\scripts\run_ml_inference.py --run-dir .\reports\ml\benchmarks\data_march2026_balanced_3scansEach\simple_cnn_w32 --image path\to\unknown_pattern.png --device auto
```

Use this for:

- scripted checks
- isolated pattern review
- integration with other tooling

## Sampled `.oh5` CLI inference

```powershell
python .\scripts\run_ml_oh5_sample_inference.py --config .\configs\ml\oh5_sample_inference.data_march2026.example.yml --debug
```

Use this for:

- deterministic random spot-checking across unseen scans
- tabular CSV/JSON prediction outputs
- per-scan summary statistics

## Full-scan suite `.oh5` CLI inference

```powershell
python .\scripts\run_ml_inference_full_scan_suite.py `
  --suite-root .\reports\ml\benchmarks\april2026_cu_ni_balanced `
  --oh5 C:\path\to\scan.oh5 `
  --output-dir .\reports\ml\full_scan_suite_exports\scan_name `
  --device auto
```

Use this for:

- exporting full predicted maps for every trained model in a suite
- generating one machine-readable export bundle per run
- creating aggregate JSON/Markdown manifests that later automation can consume for report or presentation generation

One-command cycle when you want both the exports and the comparative HTML in one run:

```powershell
python .\scripts\run_ml_inference_full_scan_suite_cycle.py `
  --suite-root .\reports\ml\benchmarks\april2026_cu_ni_balanced `
  --oh5 C:\path\to\scan.oh5 `
  --output-dir .\reports\ml\full_scan_suite_exports\scan_name `
  --device auto
```

Use the one-command cycle when:

- the scan has not yet been exported for the suite,
- you want the final browsing surface immediately,
- you are preparing an automation or repeated evaluation workflow that should leave one complete comparison folder behind.

Top-level outputs:

- `suite_full_scan_summary.json`
- `suite_full_scan_summary.md`
- `manifest.json`
- `events.jsonl`
- `runs/<run_name>/...`

Comparative HTML generation after the exports are ready:

```powershell
python .\scripts\run_ml_inference_full_scan_suite_report.py `
  --summary-json .\reports\ml\full_scan_suite_exports\scan_name\suite_full_scan_summary.json `
  --output-html .\reports\ml\full_scan_suite_exports\scan_name\comparison_report.html
```

This report is intended for side-by-side model review. It includes:

- one shared IPF-colored scan map when available,
- one shared phase legend,
- a cross-model metric table combining training and scan-level inference metrics,
- one predicted phase-map panel per model with links to the underlying machine-readable artifacts.

Practical decision rule:

- use `run_ml_inference_full_scan_suite.py` when you only need the raw export bundles first,
- use `run_ml_inference_full_scan_suite_report.py` when exports already exist and you want to regenerate HTML only,
- use `run_ml_inference_full_scan_suite_cycle.py` when you want the full export-plus-report path in one command.

Per-run outputs under `runs/<run_name>/`:

- `summary.json`
- `summary.html`
- `manifest.json`
- `pixel_predictions.csv`
- `artifacts/predicted_phase_map.png`
- `artifacts/predicted_phase_legend.png`
- optional `artifacts/ipf_reference.png`
- optional `artifacts/ipf_colored_ebsd_map.png`

## GUI inference

```powershell
python .\scripts\run_ml_inference_gui.py --suite-root .\reports\ml\benchmarks\data_march2026_balanced_3scansEach
```

The GUI supports:

- model selection from a suite root or specific run
- single-image inference
- full-scan `.oh5` inference on all available patterns
- confidence-based color shading in map mode

## Which surface to use

| Need | Best surface |
| --- | --- |
| One unknown pattern | Single-image CLI or GUI |
| Random audit of new scans | Sampled `.oh5` CLI |
| Full-scan exports for all benchmarked models | Full-scan suite `.oh5` CLI |
| Phase-map-style scan review | GUI full-scan `.oh5` mode |
| Report-ready per-pattern table | Sampled `.oh5` CLI |

## Source docs

Legacy source files:

- `docs/ml_inference_gui.md`
- `docs/ml_training_inference_workflow.md`
