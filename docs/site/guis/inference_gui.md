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

## What is new in full-scan mode

The `.oh5` workflow now shows two scan-level diagnostics side by side through tabs:

- a **predicted phase map** from the trained classifier
- an **IPF orientation reference** rendered from the scan Euler angles

The window also includes a live **log panel** and progress/ETA display so large scans remain inspectable while inference is running.

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

The IPF reference semantics are:

- orientations are read from the `.oh5` Euler fields
- Euler angles are grouped by the **predicted phase**
- one IPF subplot is rendered for each model class
- this is a diagnostic reference, not ground-truth phase labeling

This mode is deliberately analogous to EBSD-style interpretive maps where categorical identity and scalar trust are both visible at once.

## Typical usage

1. choose the suite root or a specific trained run
2. select the desired model
3. switch to `Full .oh5 scan`
4. browse the `.oh5` file
5. enable or disable confidence shading
6. run inference
7. watch the live log and ETA while the scan is processed
8. inspect the predicted phase map and IPF reference tabs
9. inspect the dominant phase, per-phase counts, fractions, and mean scores

## What the panels mean

- **top controls**: model selection and inference mode
- **left input area**: image browser or `.oh5` browser depending on mode
- **right preview tabs**: preprocessed image preview, predicted scan map, and IPF reference
- **table**: probabilities for image mode, or per-phase pixel counts/fractions/mean scores for full-scan mode
- **notes**: run metadata, grid size, Euler metadata, confidence state, and phase-color legend
- **log panel**: backend progress, scan-opening messages, Euler/IPF status, and errors
- **progress row**: processed pixels plus ETA/elapsed time

## Why the GUI works this way

The GUI is not meant to be a black-box convenience toy. It is designed to make scan interpretation inspectable:

- full-scan mode avoids arbitrary sampling when a true map is desired
- confidence shading exposes low-trust transition regions
- the IPF reference panel gives orientation-space context for the same scan
- the log pane exposes long-running backend work instead of hiding it behind a frozen window
- suite-root loading reduces model-path mistakes
- the notes and table panels keep the map tied to numerical outputs rather than showing a pretty image alone

## Related source page

Legacy source file: `docs/ml_inference_gui.md`
