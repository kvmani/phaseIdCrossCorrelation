# Diagnostic Gallery GUI

:::{figure} ../figures/diagnostic_gallery_gui.svg
:alt: Diagnostic gallery GUI schematic
:width: 100%

Schematic layout of the diagnostic gallery GUI for cross-condition pattern comparison and manual pixel lookup.
:::

## Launch command

```powershell
python .\scripts\run_ml_diagnostic_gallery.py --config .\configs\ml\diagnostic_gallery.example.yml --debug
```

## Purpose

The diagnostic gallery exists for **cross-condition pattern diagnosis**, not just inference convenience. It is the right tool when you need:

- side-by-side reference and unknown patterns
- reproducible sampling
- manual exact-pattern lookup
- contact-sheet export and manifest tracking

## Related source page

Legacy source file: `docs/ml_diagnostic_gallery_gui.md`
