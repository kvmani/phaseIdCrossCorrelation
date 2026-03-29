# Scripts

Thin CLI entry points live here.

## NCC and Hough

- `scripts/run_g0_data_intake_validation.py`
- `scripts/run_curated_ncc.py`
- `scripts/build_curated_ncc_inspection_html.py`
- `scripts/run_curated_hough_vs_ncc.py`

## ML

- `scripts/run_ml_dataset_prepare.py`
- `scripts/run_ml_train_classifier.py`
- `scripts/run_ml_benchmark_suite.py`
- `scripts/package_ml_benchmark_suite.py` (zip only lightweight suite artifacts such as JSON/HTML/YAML/CSV/PPTX for mail-friendly transfer)
- `scripts/run_ml_inference.py` (predict phase ID for one unknown image from a saved training run)
- `scripts/run_ml_oh5_sample_inference.py` (sample filtered patterns from unseen `.oh5` scans, run saved CNN inference, write per-pattern JSON/CSV plus per-scan summaries, and print a compact prediction table)
- `scripts/run_ml_inference_gui.py` (desktop GUI to choose a trained model, drop an unknown image, and inspect prediction probabilities)
- `scripts/run_ml_phase_explorer.py` (native desktop GUI for raw `.oh5` phase-wise histogram/CDF exploration, interactive intensity-band highlighting, and auto-exported publication PNG/JSON histogram artifacts)
- `scripts/run_ml_diagnostic_gallery.py` (desktop GUI for cross-condition pattern diagnosis with source grouping, manual pattern lookup, and manifest/contact-sheet export)
- `scripts/run_ml_suite_with_ppt.py` (run benchmark suite and auto-build lab-meeting PPTX summary)
- `scripts/run_ml_full_cycle.py` (one-go workflow from raw `.oh5` dataset prep to multi-model suite, HTML summaries, and PPTX)
- `scripts/ml_results_presentation/generate_executive_assessment_ppt.py` (build a graphics-first lab-meeting PPTX from balanced benchmark, dataset, and diagnostic gallery artifacts)
See `docs/README.md` for the doc map behind each runner.
