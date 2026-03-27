# ML Configs

YAML configs for the ML classifier branch.

Files:

- `dataset_prepare.default.yml`: build combined train/val/test dataset from `.oh5` + CSV label pairs (`input_mode: oh5_csv_labels`).
- `dataset_prepare.single_phase_scan_map.debug.yml`: build combined train/val/test dataset from `.oh5` files where each file maps to one phase (`input_mode: single_phase_scan_map`).
- `dataset_prepare.v3_al_ni_cu.example.yml`: concise v3 schema using `data_source_folder` + `listOfFiles`, expression quality filters, unified preprocessing, and capped leakage-safe splits.
- `train.convnextv2_nano.pretrained.debug.yml`: debug training run with ConvNeXtV2 pretrained init.
- `train.simple_cnn.debug.yml`: minimal fast smoke training run.
- `benchmark_suite.debug.yml`: multi-model debug benchmark suite.
- `benchmark_suite.classiication_training_data.smoke.yml`: smoke benchmark suite on local `Al/Ni/Cu` cropped `.oh5` data.
- `benchmark_suite.ni_cu_al.production.yml`: production benchmark suite template for the Windows `Ni-Cu-Al` scan folder.
- `full_cycle.debug.yml`: one-go orchestration config (dataset prep -> suite -> HTML + PPTX).
- `full_cycle.ni_cu_al.production.yml`: production full-cycle template using the Windows `Ni-Cu-Al` scan folder.
- `dataset_prepare.classiication_training_data.filtered.yml`: local `Al/Ni/Cu` cropped `.oh5` prep with `CI > 0.4 && Fit < 1.5`.
- `dataset_prepare.ni_cu_al.production.yml`: production dataset-prep template pointing at `F:\PhaseID_Training_Data\Ni-Cu-Al_Scans`.
- `phase_explorer.ni_cu_al.production.yml`: production explorer GUI config for `Al-1.oh5`, `Ni.oh`, and `Cu-1.oh5`.
- `train.classiication_training_data.smoke.yml`: smoke single-run training config on the filtered local dataset.
- `train.ni_cu_al.production.base.yml`: production baseline train config for the filtered Windows dataset.
- `oh5_sample_inference.ni_different_condition.example.yml`: example unseen-scan CNN inference config for filtered random sampling from new `.oh5` files under a Windows folder.

Run commands:

```bash
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.default.yml --debug
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.single_phase_scan_map.debug.yml --debug
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.v3_al_ni_cu.example.yml --debug
python scripts/run_ml_train_classifier.py --config configs/ml/train.convnextv2_nano.pretrained.debug.yml --debug
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.debug.yml --debug
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.classiication_training_data.smoke.yml --debug
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.ni_cu_al.production.yml --debug
python scripts/run_ml_phase_explorer.py --config configs/ml/dataset_prepare.v3_al_ni_cu.example.yml --debug
python scripts/run_ml_phase_explorer.py --config configs/ml/phase_explorer.ni_cu_al.production.yml --debug
python scripts/run_ml_oh5_sample_inference.py --config configs/ml/oh5_sample_inference.ni_different_condition.example.yml --debug
python scripts/run_ml_full_cycle.py --config configs/ml/full_cycle.debug.yml --debug
python scripts/run_ml_full_cycle.py --config configs/ml/full_cycle.ni_cu_al.production.yml --debug
```

Each workflow writes both `manifest.json` and `events.jsonl` under its output directory. Dataset HTML summaries also include phase-wise split composition, CI/Fit/IQ statistics, and modal intensity values. Benchmark HTML summaries include best-model selection, confusion matrices, per-class metrics, and links to per-run artifacts.
