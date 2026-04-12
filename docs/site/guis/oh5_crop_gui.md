# OH5 Crop GUI

The OH5 crop GUI is a dedicated desktop workflow for extracting a rectangular EBSD subscan from one pattern-bearing `.oh5` file and immediately validating the result visually.

## Main command

```powershell
python .\scripts\run_oh5_crop_gui.py --input path\to\source_scan.oh5 --debug
```

## Capabilities

- display the source `IQ` / `Image Quality` map as a grayscale scan view
- define one crop rectangle by draggable overlay or numeric row/column/width/height fields
- export a cropped standalone `.oh5`
- automatically switch into review mode after export
- keep a visible log console and progress/status bar during load, export, reload, and pixel review
- show original and cropped scan sizes directly above the IQ and IPF comparison panes
- compare original vs cropped scans in side-by-side `IQ Maps`, `IPF Maps`, and `Patterns + Pixel Data` tabs
- click the cropped scan to inspect mapped original and cropped pixel metadata plus both Kikuchi patterns
- include a `Metadata Audit` tab listing fields that changed and fields that remained unaltered, with scalar values or dataset-shape summaries

```{mermaid}
flowchart LR
    A["Source .oh5"] --> B["Crop rectangle on IQ map"]
    B --> C["Write cropped .oh5"]
    C --> D["Auto-open review mode"]
    D --> E["IQ and IPF side-by-side"]
    D --> F["Pixel metadata compare"]
    D --> G["Kikuchi pattern compare"]
```

## Review-mode expectations

- clicking a cropped pixel highlights the corresponding original-source location automatically
- scalar values should match except for intentionally rebased local-position fields such as `X Position` and `Y Position`
- Kikuchi patterns should be visually identical for corresponding original/cropped pixels
- the GUI exposes the original grid size and cropped grid size above the view panels for immediate visual confirmation

## Built-in crop verification

After writing the cropped file, the exporter performs an internal source-vs-cropped verification pass.

It asserts that:

- group paths match between source and cropped files
- dataset paths match between source and cropped files
- unchanged metadata fields remain identical
- crop-affected fields differ only in crop-consistent ways such as:
  - `nColumns`
  - `nRows`
  - scan-shaped data arrays
  - rebased local-position arrays

The `Metadata Audit` tab exposes the resulting summary so the user can inspect:

- changed fields with source and cropped values/shapes
- unchanged fields with matching values/shapes

If Euler-angle fields or usable phase metadata are not present, the `IPF Maps` tab remains visible but shows an explicit unavailable message instead of failing.
