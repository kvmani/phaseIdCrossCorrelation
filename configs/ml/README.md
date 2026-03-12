# ML Configs

YAML configs for the ML classifier branch.

Files:

- `dataset_prepare.default.yml`: build combined train/val/test dataset from `.oh5` + CSV label pairs (`input_mode: oh5_csv_labels`).
- `dataset_prepare.single_phase_scan_map.debug.yml`: build combined train/val/test dataset from `.oh5` files where each file maps to one phase (`input_mode: single_phase_scan_map`).
- `dataset_prepare.v3_al_ni_cu.example.yml`: concise v3 schema using `data_source_folder` + `listOfFiles`, expression quality filters, unified preprocessing, and capped leakage-safe splits.
- `train.convnextv2_nano.pretrained.debug.yml`: debug training run with ConvNeXtV2 pretrained init.
- `train.simple_cnn.debug.yml`: minimal fast smoke training run.
- `benchmark_suite.debug.yml`: multi-model debug benchmark suite.

Run commands:

```bash
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.default.yml --debug
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.single_phase_scan_map.debug.yml --debug
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.v3_al_ni_cu.example.yml --debug
python scripts/run_ml_train_classifier.py --config configs/ml/train.convnextv2_nano.pretrained.debug.yml --debug
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.debug.yml --debug
python scripts/run_ml_phase_explorer.py --config configs/ml/dataset_prepare.v3_al_ni_cu.example.yml --debug
```

Each workflow writes both `manifest.json` and `events.jsonl` under its output directory.
