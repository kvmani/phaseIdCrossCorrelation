# Inference Workflows

Inference in this repository exists in three forms:

1. one-image CLI inference
2. sampled `.oh5` CLI inference
3. GUI inference, including full-scan `.oh5` map rendering

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
| Phase-map-style scan review | GUI full-scan `.oh5` mode |
| Report-ready per-pattern table | Sampled `.oh5` CLI |

## Source docs

Legacy source files:

- `docs/ml_inference_gui.md`
- `docs/ml_training_inference_workflow.md`
