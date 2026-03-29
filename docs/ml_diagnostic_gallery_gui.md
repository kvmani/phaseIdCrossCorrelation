# Diagnostic Pattern Gallery GUI

This GUI is the cross-condition inspection tool for the ML phase-classification track. It is intended for cases where training metrics look good, but you want to inspect whether unseen scans from another day, another preparation state, or another working distance still resemble the training data.

## 1. What It Does

The gallery loads a trained classifier run and one or more `.oh5` scans grouped into:

- reference scans, typically the training-phase scans such as `Al`, `Cu`, and `Ni`,
- unknown scans, such as anonymized `data1`, `data2`, `data3`.

It then:

- samples a fixed number of patterns per source,
- applies the same quality gate used during ML work,
- filters by prediction confidence and top-1/top-2 margin,
- shows raw and preprocessed views for each selected pattern,
- records every displayed tile in a manifest so the exact session can be reproduced later.

## 2. Why This GUI Exists

This is for diagnosis, not training. The main questions it answers are:

- are all unknown scans being pushed to the same class because the data distribution shifted,
- is the preprocessing mismatch hiding phase-specific structure,
- are the selected patterns actually representative of the scan,
- does a manual index lookup confirm what the auto-sampler is showing.

## 3. Layout

The layout keeps controls narrow and gives most of the screen to the pattern tiles.

```text
+----------------------+--------------------------------------------------------------+
| Control rail         |  Source sections                                              |
| - model run          |  Reference bank                                               |
| - output dir         |  [tile] [tile] [tile] [tile] [tile]                           |
| - sampling/filter    |  Unknown bank                                                 |
| - file lists         |  [tile] [tile] [tile] [tile] [tile]                           |
| - manual index       |                                                               |
+----------------------+--------------------------------------------------------------+
| Collapsible detail drawer: raw preview, preprocessed preview, metadata, probabilities |
+--------------------------------------------------------------------------------------+
```

The intended interaction is:

1. Load a config or drag `.oh5` files into the reference/unknown lists.
2. Set the sample count, quality expression, and prediction thresholds.
3. Build the session.
4. Click any tile to inspect the raw and preprocessed images.
5. Type a source and pattern index to force-display an exact pixel.
6. Export the manifest and contact sheets.

## 4. Controls

The GUI exposes:

- `Run dir` and `Checkpoint` for loading the trained model,
- `Output dir` for export artifacts,
- `Patterns/source` for deterministic auto-sampling,
- `Seed` and `Strategy` for repeatable selection,
- `Quality expr` for the `.oh5` quality gate,
- `Min confidence` and `Min margin` for prediction filtering,
- `Source` and `Index` for manual pattern addition.

## 5. Outputs

The export step writes:

- `manifest.json`
- `combined_contact_sheet.png`
- `reference_contact_sheet.png`
- `unknown_contact_sheet.png`
- `session_config.json`

The manifest records:

- source files,
- selected pattern indices,
- `x/y` pixel coordinates,
- confidence and margin,
- quality values such as CI/IQ/Fit/Valid,
- class probabilities.

## 6. Recommended Use

Use this GUI after a model looks good on validation/test data but fails on new scans. In practice, it is the quickest way to tell whether the problem is:

- a real generalization failure,
- a preprocessing mismatch,
- or a source-data change such as working distance, background correction, or acquisition day.

For scripted review, use the diagnostic gallery manifest and contact sheets as the artifact source for slides or reports.

