# Phase Explorer GUI

:::{figure} ../figures/phase_explorer_gui.svg
:alt: Phase explorer GUI schematic
:width: 100%

Schematic layout of the raw `.oh5` phase explorer GUI with field distributions, filtering, and export surfaces.
:::

## Launch command

```powershell
python .\scripts\run_ml_phase_explorer.py --config .\configs\ml\phase_explorer.ni_cu_al.production.yml --debug
```

## Purpose

The phase explorer is for **raw `.oh5` interrogation before model decisions**. It helps answer:

- what scalar fields exist in this scan
- how `CI`, `IQ`, `Fit`, or intensity distributions differ by source or phase
- where suspicious low-quality regimes live before they become training or inference artifacts

## Use it when

- you are setting or revising quality filters
- you want publication-style histograms and JSON manifests
- you need to inspect phase-specific scalar distributions from raw scan data

## Related source page

Legacy source file: `docs/ml_phase_explorer_gui.md`
