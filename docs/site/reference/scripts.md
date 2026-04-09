# Script Reference

This is the command-level reference for the main runnable entry points.

## Documentation

```powershell
python .\scripts\build_docs.py --clean
python .\scripts\build_docs.py --clean --open
```

## NCC and Hough

```powershell
python .\scripts\run_g0_data_intake_validation.py --debug
python .\scripts\run_curated_ncc.py --debug
python .\scripts\run_curated_hough_vs_ncc.py --debug
```

## ML dataset and training

```powershell
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.data_march2026.balanced.debug.yml --debug
python .\scripts\run_ml_train_classifier.py --config .\configs\ml\train.data_march2026.balanced.debug.base.yml --debug
python .\scripts\run_ml_benchmark_suite.py --config .\configs\ml\benchmark_suite.data_march2026.balanced.debug.yml --debug
python .\scripts\run_ml_full_cycle.py --config .\configs\ml\full_cycle.data_march2026.balanced.debug.yml --debug
```

## Inference and GUIs

```powershell
python .\scripts\run_ml_inference.py --run-dir .\reports\ml\benchmarks\data_march2026_balanced_3scansEach\simple_cnn_w32 --image path\to\pattern.png --device auto
python .\scripts\run_ml_inference_gui.py --suite-root .\reports\ml\benchmarks\data_march2026_balanced_3scansEach
python .\scripts\run_ml_oh5_sample_inference.py --config .\configs\ml\oh5_sample_inference.data_march2026.example.yml --debug
python .\scripts\run_ml_inference_full_scan_suite.py --suite-root .\reports\ml\benchmarks\april2026_cu_ni_balanced --oh5 C:\path\to\scan.oh5 --output-dir .\reports\ml\full_scan_suite_exports\scan_name --device auto
python .\scripts\run_ml_inference_full_scan_suite_report.py --summary-json .\reports\ml\full_scan_suite_exports\scan_name\suite_full_scan_summary.json --output-html .\reports\ml\full_scan_suite_exports\scan_name\comparison_report.html
python .\scripts\run_ml_inference_full_scan_suite_cycle.py --suite-root .\reports\ml\benchmarks\april2026_cu_ni_balanced --oh5 C:\path\to\scan.oh5 --output-dir .\reports\ml\full_scan_suite_exports\scan_name --device auto
python .\scripts\run_ml_phase_explorer.py --config .\configs\ml\phase_explorer.ni_cu_al.production.yml --debug
python .\scripts\run_ml_diagnostic_gallery.py --config .\configs\ml\diagnostic_gallery.example.yml --debug
```
