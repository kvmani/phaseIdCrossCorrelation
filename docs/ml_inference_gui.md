# ML Inference CLI and GUI

Use this workflow to load a trained classifier run, preprocess unknown inputs so they match training input handling, and predict the phase ID either for one image or for every pattern in a full `.oh5` scan.

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

For full-scan `.oh5` GUI inference, the scan must contain:

- `/<scan>/EBSD/Header/nColumns`
- `/<scan>/EBSD/Header/nRows`
- `/<scan>/EBSD/Data/Pattern` or `Patterns`

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

For sampled unseen-scan `.oh5` CLI inference, use:

```bash
python scripts/run_ml_oh5_sample_inference.py \
  --config configs/ml/oh5_sample_inference.ni_different_condition.example.yml \
  --debug
```

## 3. GUI

Launch the desktop GUI on a benchmark suite root:

```bash
python scripts/run_ml_inference_gui.py \
  --suite-root reports/ml/benchmarks/ni_cu_al_production
```

GUI features:

- choose any available model from the suite
- switch between `Single image` and `Full .oh5 scan` inference modes
- drag-and-drop or browse an unknown image
- inspect original and preprocessed grayscale views in image mode
- browse a `.oh5` scan and run inference on every available pattern in full-scan mode
- render a predicted phase map on the scan grid using class colors
- click any full-scan map pixel to inspect the corresponding experimental Kikuchi pattern
- display clicked-pixel Kikuchi patterns in grayscale with optional histogram normalization and contrast stretch
- render an IPF orientation-reference panel from the scan Euler angles when present
- render a per-pixel IPF-colored EBSD map from the scan Euler angles when present
- optionally dull low-confidence pixels using `Use confidence shading`
- stream backend progress, ETA, and errors in a live log panel during large-scan processing
- view per-phase probabilities for single-image mode
- view per-phase pixel counts, fractions, and mean scores for full-scan mode
- optionally set the known phase and compare prediction vs truth or dominant predicted phase

You can also point the GUI directly at one run directory instead of the suite root.

### 3.1 Full-Scan `.oh5` Mode

Full-scan mode is intended for qualitative review of a new EBSD scan against a trained phase classifier.

Behavior:

- runs inference on all available patterns in the selected `.oh5`
- reconstructs the `nRows x nColumns` scan grid
- assigns each pixel the predicted phase color
- optionally scales color vividness by the model confidence for that pixel
- reads Euler angles from the `.oh5` when available and renders an IPF reference grouped by predicted phase
- reads Euler angles from the `.oh5` when available and renders a conventional-style IPF-colored orientation map per pixel
- lets the user click a predicted-map pixel and load the matching experimental Kikuchi pattern from the `.oh5`
- shows clicked-pattern metadata including pixel coordinate, predicted phase, confidence, and discovered quality fields when available
- keeps the pattern in grayscale and optionally applies histogram normalization and percentile-based contrast stretch for visibility tuning
- leaves missing/unavailable pixels dark
- emits progress, ETA, and backend status messages in a GUI log window

This makes it easy to spot:

- obvious phase-region boundaries
- isolated misclassified islands
- low-confidence transition regions
- scan-wide phase dominance or unexpected fragmentation
- orientation clustering or coverage differences across predicted phases
- orientation-domain structure in a familiar EBSD IPF-map view

## 4. Recommended Production Use

1. Run dataset prep and benchmark suite.
2. Select the practical winner model, not just the absolute highest score.
3. Use the inference GUI for:
   - rapid qualitative inspection of unknown single patterns,
   - full-scan `.oh5` predicted phase-map review.
4. Use the sampled `.oh5` CLI when you want tabular outputs and deterministic random spot-checking across many scans.
5. Use the single-image CLI for scripted checks or integration with other tooling.
