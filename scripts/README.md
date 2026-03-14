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
- `scripts/run_ml_phase_explorer.py` (native desktop GUI for raw `.oh5` phase-wise histogram/CDF exploration and interactive intensity-band highlighting)
- `scripts/run_ml_suite_with_ppt.py` (run benchmark suite and auto-build lab-meeting PPTX summary)
- `scripts/run_ml_full_cycle.py` (one-go workflow from raw `.oh5` dataset prep to multi-model suite, HTML summaries, and PPTX)
See `docs/README.md` for the doc map behind each runner.
