# Scripts

Thin CLI entry points live here.

## NCC and Hough

- `scripts/run_g0_data_intake_validation.py`
- `scripts/run_curated_ncc.py`
- `scripts/build_curated_ncc_inspection_html.py`
- `scripts/run_curated_hough_vs_ncc.py`

## ML

- `scripts/build_docs.py` (build, clean, and optionally open the Sphinx documentation site)
- `scripts/run_ml_dataset_prepare.py`
- `scripts/run_ml_train_classifier.py`
- `scripts/run_ml_benchmark_suite.py`
- `scripts/package_ml_benchmark_suite.py` (zip only lightweight suite artifacts such as JSON/HTML/YAML/CSV/PPTX for mail-friendly transfer)
- `scripts/run_ml_inference.py` (predict phase ID for one unknown image from a saved training run)
- `scripts/run_ml_oh5_sample_inference.py` (sample filtered patterns from unseen `.oh5` scans, run saved CNN inference, write per-pattern JSON/CSV plus per-scan summaries, and print a compact prediction table)
- `scripts/run_ml_inference_full_scan_suite.py` (run full-scan `.oh5` inference for every trained model under a benchmark suite and export one provenance-rich artifact bundle per run plus an aggregate suite summary)
- `scripts/run_ml_inference_full_scan_suite_report.py` (build one comparative HTML report from a suite-level full-scan export folder, with shared scan visuals, cross-model metric tables, and side-by-side predicted phase maps)
- `scripts/run_ml_inference_full_scan_suite_cycle.py` (one-command wrapper that runs suite-level full-scan `.oh5` exports and immediately builds the comparative HTML report in the same output folder)
- `scripts/run_ml_inference_gui.py` (desktop GUI to choose a trained model, drop an unknown image, and inspect prediction probabilities)
- `scripts/run_ml_phase_explorer.py` (native desktop GUI for raw `.oh5` phase-wise histogram/CDF exploration, interactive intensity-band highlighting, and auto-exported publication PNG/JSON histogram artifacts)
- `scripts/run_ml_diagnostic_gallery.py` (desktop GUI for cross-condition pattern diagnosis with source grouping, manual pattern lookup, and manifest/contact-sheet export)
- `scripts/run_ml_suite_with_ppt.py` (run benchmark suite and auto-build lab-meeting PPTX summary)
- `scripts/run_ml_full_cycle.py` (one-go workflow from raw `.oh5` dataset prep to multi-model suite, HTML summaries, and PPTX)
- `scripts/ml_results_presentation/generate_executive_assessment_ppt.py` (build a graphics-first lab-meeting PPTX from balanced benchmark, dataset, and diagnostic gallery artifacts)
- `scripts/generate_cnn32_architecture_svg.py` (generate a publication-ready SVG of the exact `simple_cnn_w32` architecture used in the balanced Al/Cu/Ni work)
See `docs/site/index.md` for the canonical documentation map and `docs/README.md` for the legacy-source bridge.
