# OH5 Crop GUI

This dedicated desktop GUI crops one pattern-bearing `.oh5` scan by one or more rectangles on the `IQ` / `Image Quality` map and writes standalone cropped `.oh5` files intended for independent reopening.

## Main command

```bash
python scripts/run_oh5_crop_gui.py --input path/to/source_scan.oh5 --debug
```

Optional startup export directory:

```bash
python scripts/run_oh5_crop_gui.py --input path/to/source_scan.oh5 --output-dir reports/oh5_crops
```

## Workflow

1. Load one `.oh5` containing:
   - `Pattern` or `Patterns`
   - `IQ` or `Image Quality`
2. Start from the centered default rectangle or add extra rectangles in the `Crop Regions` list.
3. Select a rectangle, then draw or numerically define that rectangle on the grayscale IQ map.
4. Export all defined rectangles in one pass as `{base_name}_crop_{row}_{col}.oh5`.
5. The GUI automatically switches into review mode and compares:
   - original IQ vs cropped IQ
   - original IPF vs cropped IPF when Euler/phase metadata are available
   - original pixel metadata/pattern vs cropped pixel metadata/pattern
   - source-vs-cropped metadata audit showing fields that changed and fields that remained identical

Visible operator feedback:

- a shared GUI log console remains visible throughout crop and review modes
- a progress/status bar reports source loading, crop writing, review reload, and pixel-selection state
- explicit original and cropped scan sizes are shown above the relevant view panes
- a dedicated metadata-audit tab lists field-by-field source and cropped summaries, including scalar values and dataset shapes
- review mode includes a crop selector so any exported crop from the current batch can be inspected without reopening the tool

## Review behavior

- Click in the cropped IQ or IPF pane.
- The GUI maps that local cropped pixel back to the original scan coordinate.
- It highlights both locations and shows:
  - IQ
  - CI
  - Fit
  - Valid
  - X Position
  - Y Position
  - Euler angles when present
  - side-by-side Kikuchi patterns

The review workspace also keeps the original and cropped scan sizes visible above the IQ and IPF panes so the exported grid can be checked immediately.

## Metadata audit and built-in checks

The exporter now performs an internal verification pass after writing the cropped `.oh5`:

- source and cropped group paths must match
- source and cropped dataset paths must match
- each dataset must either:
  - remain identical to the source, or
  - differ only in a crop-expected way such as `nColumns`, `nRows`, scan-shaped arrays, or rebased local `X Position` / `Y Position`

The GUI exposes this verification in a `Metadata Audit` tab with:

- `Changed Fields`
- `Unchanged Fields`

For scalar datasets the tab shows values directly. For array datasets it shows shape and dtype summaries such as pattern stack sizes before and after crop.

This is intended as an immediate visual integrity check after export.

## Export artifacts

The export writes:

- cropped `.oh5`
- sibling crop manifest JSON with crop origin/size and comparison metadata

The current implementation rebases `X Position` and `Y Position` to the cropped local origin when those fields are present.
