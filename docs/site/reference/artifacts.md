# Artifact Reference

## Dataset preparation artifacts

- `manifest.json`: authoritative run metadata
- `summary.html`: primary human-readable overview
- `records.csv`: accepted selected sample table
- `events.jsonl`: event trace
- split `.npz`: train/val/test tensors and IDs
- orientation exports: provenance-rich Euler tables
- IPF PNGs and `ipf_index.json`: orientation-space diagnostics

## Training artifacts

- `report.json`: main metrics and run metadata
- `report.md`: concise human-readable summary
- `best_checkpoint.pt`: checkpoint recommended for inference
- `epoch_history.jsonl`: training curve evidence

## Suite artifacts

- `suite_summary.json`: machine-readable suite comparison
- `suite_summary.md`: concise markdown table
- `suite_report.html`: main comparative browsing surface

## Full-cycle artifacts

- `full_cycle_summary.json`: orchestration-level machine-readable summary
- `full_cycle_summary.html`: landing page linking dataset and suite outputs
- resolved configs: provenance for the exact resolved run

## Inference artifacts

- CLI JSON for one-image inference
- sampled `.oh5` prediction CSV/JSON
- per-scan summary CSV/JSON/Markdown
- GUI map and probability/table panels for interactive interpretation
- suite-level full-scan export folder:
  - `suite_full_scan_summary.json`
  - `suite_full_scan_summary.md`
  - `manifest.json`
  - `events.jsonl`
  - `runs/<run_name>/...`
- per-run full-scan export bundle:
  - `summary.json`
  - `summary.html`
  - `manifest.json`
  - `pixel_predictions.csv`
  - `artifacts/predicted_phase_map.png`
  - `artifacts/predicted_phase_legend.png`
  - optional IPF PNGs
- comparative full-scan HTML:
  - `comparison_report.html`
  - side-by-side predicted phase maps across models
  - shared IPF-colored map and shared legend when available
  - training-plus-inference comparison table linked to per-run artifacts

## Transfer bundle artifacts

The transfer archive produced by `scripts/package_ml_benchmark_suite.py` is intended for cross-machine ingestion and later automated report generation.

It can include:

- benchmark suite artifacts
- referenced dataset manifests and summaries
- optional suite-level full-scan inference export folders
- optional PPTX or additional summary files passed by `--extra-path`

It intentionally excludes heavy tensor and checkpoint files such as:

- `.pt`
- `.pth`
- `.npy`
- `.npz`
