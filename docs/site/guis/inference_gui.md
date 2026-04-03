# Inference GUI

:::{figure} ../figures/inference_gui_schematic.svg
:alt: Inference GUI schematic
:width: 100%

Schematic layout of the inference GUI showing suite/run selection, mode switch, image or `.oh5` inputs, a clicked-pixel Kikuchi inspector, and the result/summary panels.
:::

:::{figure} ../figures/full_scan_map_mode.svg
:alt: Full scan mapping mode schematic
:width: 100%

Full-scan `.oh5` mode: the GUI reconstructs the scan grid and colors each pixel by predicted phase, with optional confidence shading.
:::

## What is new in full-scan mode

The `.oh5` workflow now shows two scan-level diagnostics side by side through tabs:

- a **predicted phase map** from the trained classifier
- a **clicked-pixel Kikuchi inspector** in the left panel
- an **IPF orientation reference** rendered from the scan Euler angles
- an **IPF-colored EBSD map** rendered per pixel from the scan Euler angles

The clicked-pixel inspector loads the experimental pattern for any map pixel, keeps the display grayscale, and offers optional **histogram normalization** plus **contrast stretch** to improve band visibility without changing the underlying inference result.

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
- click selection = the map pixel whose experimental Kikuchi pattern is shown in the inspector panel

The IPF reference semantics are:

- orientations are read from the `.oh5` Euler fields
- Euler angles are grouped by the **predicted phase**
- one IPF subplot is rendered for each model class
- this is a diagnostic reference, not ground-truth phase labeling

The IPF-colored EBSD map semantics are:

- each pixel is colored directly from its Euler orientation
- IPF color keys are applied per predicted phase symmetry
- the result is intended to resemble a conventional EBSD IPF map for orientation texture review

This mode is deliberately analogous to EBSD-style interpretive maps where categorical identity and scalar trust are both visible at once.

## Typical usage

1. choose the suite root or a specific trained run
2. select the desired model
3. switch to `Full .oh5 scan`
4. browse the `.oh5` file
5. enable or disable confidence shading
6. run inference
7. watch the live log and ETA while the scan is processed
8. click any map pixel to load its experimental Kikuchi pattern
9. optionally enable histogram normalization and/or contrast stretch for the clicked pattern
10. inspect the predicted phase map, IPF reference, and IPF-colored EBSD map tabs
11. inspect the dominant phase, per-phase counts, fractions, and mean scores

## What the panels mean

- **top controls**: model selection and inference mode
- **left input area**: image browser in image mode, or `.oh5` browser plus clicked-pixel Kikuchi inspector in full-scan mode
- **right preview tabs**: preprocessed image preview, predicted scan map, IPF reference, and IPF-colored EBSD map
- **table**: probabilities for image mode, or per-phase pixel counts/fractions/mean scores for full-scan mode
- **notes**: run metadata, grid size, Euler metadata, confidence state, phase-color legend, and currently selected pixel details
- **log panel**: backend progress, scan-opening messages, Euler/IPF status, and errors
- **progress row**: processed pixels plus ETA/elapsed time

## Why the GUI works this way

The GUI is not meant to be a black-box convenience toy. It is designed to make scan interpretation inspectable:

- full-scan mode avoids arbitrary sampling when a true map is desired
- confidence shading exposes low-trust transition regions
- clicked-pixel pattern review ties each map decision back to the raw experimental evidence
- grayscale-only display respects the native character of Kikuchi patterns
- histogram normalization and contrast stretch help reveal weak band structure during qualitative review
- the IPF reference panel gives orientation-space context for the same scan
- the IPF-colored EBSD map gives a familiar orientation microstructure view derived from the same Euler data
- the log pane exposes long-running backend work instead of hiding it behind a frozen window
- suite-root loading reduces model-path mistakes
- the notes and table panels keep the map tied to numerical outputs rather than showing a pretty image alone

## Related source page

Legacy source file: `docs/ml_inference_gui.md`
