# ML Inference CLI and GUI

Use this workflow to load a trained classifier run, preprocess an unknown image so it matches training input handling, and predict the phase ID.

## 1. Inputs

The inference tools expect a training run directory containing:

- `report.json`
- `best_checkpoint.pt` or another checkpoint file

For benchmark-suite runs, each model folder under the suite root already has this structure.

Supported image formats:

- `.png`
- `.jpg` / `.jpeg`
- `.tif` / `.tiff`
- `.bmp`

Images are converted to grayscale, scaled to `[0, 1]`, then preprocessed using the stored dataset preprocessing policy from the training report.

## 2. CLI

Predict one image from one saved run:

```bash
python scripts/run_ml_inference.py \
  --run-dir reports/ml/benchmarks/ni_cu_al_production/simple_cnn_w32 \
  --image path/to/unknown_pattern.png \
  --device auto
```

This prints machine-readable JSON with:

- predicted phase
- class probabilities
- confidence
- model name
- run directory

## 3. GUI

Launch the desktop GUI on a benchmark suite root:

```bash
python scripts/run_ml_inference_gui.py \
  --suite-root reports/ml/benchmarks/ni_cu_al_production
```

GUI features:

- choose any available model from the suite
- drag-and-drop or browse an unknown image
- inspect original and preprocessed grayscale views
- view per-phase probabilities
- optionally set the known phase and compare prediction vs truth

You can also point the GUI directly at one run directory instead of the suite root.

## 4. Recommended Production Use

1. Run dataset prep and benchmark suite.
2. Select the practical winner model, not just the absolute highest score.
3. Use the inference GUI for rapid qualitative inspection of unknown patterns.
4. Use the CLI for scripted batch checks or integration with other tooling.
