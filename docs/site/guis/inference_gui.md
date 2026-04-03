# Inference GUI

:::{figure} ../figures/inference_gui_schematic.svg
:alt: Inference GUI schematic
:width: 100%

Schematic layout of the inference GUI showing suite/run selection, mode switch, image or `.oh5` inputs, and the result/summary panels.
:::

:::{figure} ../figures/full_scan_map_mode.svg
:alt: Full scan mapping mode schematic
:width: 100%

Full-scan `.oh5` mode: the GUI reconstructs the scan grid and colors each pixel by predicted phase, with optional confidence shading.
:::

## Launch command

```powershell
python .\scripts\run_ml_inference_gui.py --suite-root .\reports\ml\benchmarks\data_march2026_balanced_3scansEach
```

## Modes

### Single image

Use this mode when you have one unknown image file and want:

- the predicted phase
- per-class probabilities
- original and preprocessed preview images

### Full `.oh5` scan

Use this mode when you want to infer **every available pattern in a scan** and view the result as a phase map.

The rendered map semantics are:

- hue = predicted phase
- intensity/saturation = confidence when `Use confidence shading` is enabled
- duller pixels = lower confidence

This mode is deliberately analogous to EBSD-style interpretive maps where categorical identity and scalar trust are both visible at once.

## Typical usage

1. choose the suite root or a specific trained run
2. select the desired model
3. switch to `Full .oh5 scan`
4. browse the `.oh5` file
5. enable or disable confidence shading
6. run inference
7. inspect the dominant phase, per-phase counts, fractions, and mean scores

## What the panels mean

- **top controls**: model selection and inference mode
- **left input area**: image browser or `.oh5` browser depending on mode
- **right preview area**: preprocessed image preview or predicted scan map
- **table**: probabilities for image mode, or per-phase pixel counts/fractions/mean scores for full-scan mode
- **notes**: run metadata, grid size, confidence state, and phase-color legend

## Why the GUI works this way

The GUI is not meant to be a black-box convenience toy. It is designed to make scan interpretation inspectable:

- full-scan mode avoids arbitrary sampling when a true map is desired
- confidence shading exposes low-trust transition regions
- suite-root loading reduces model-path mistakes
- the notes and table panels keep the map tied to numerical outputs rather than showing a pretty image alone

## Related source page

Legacy source file: `docs/ml_inference_gui.md`
