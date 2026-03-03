# ML Configs

YAML configs for the ML classifier branch.

Files:

- `dataset_prepare.default.yml`: build combined train/val/test dataset from `.oh5` + CSV label pairs.
- `train.convnextv2_nano.pretrained.debug.yml`: debug training run with ConvNeXtV2 pretrained init.
- `train.simple_cnn.debug.yml`: minimal fast smoke training run.
- `benchmark_suite.debug.yml`: multi-model debug benchmark suite.

Run commands:

```bash
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.default.yml --debug
python scripts/run_ml_train_classifier.py --config configs/ml/train.convnextv2_nano.pretrained.debug.yml --debug
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.debug.yml --debug
```

Each workflow writes both `manifest.json` and `events.jsonl` under its output directory.
