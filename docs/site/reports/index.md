# Reports and Artifact Interpretation

## What to read first

For most workflows, inspect artifacts in this order:

1. HTML summary
2. JSON summary or manifest
3. detailed CSV/JSON tables
4. images, plots, and subordinate reports

## Dataset reports

The dataset `summary.html` is the best first surface. It answers:

- how many raw pixels were available
- how many qualified
- how many selected after balancing
- how splits landed by phase
- where orientation/IPF artifacts are

## Suite reports

The suite `suite_report.html` is the best comparative review surface. It answers:

- which models ran
- which completed or failed
- best validation macro-F1
- held-out test metrics
- links to each run's `report.json`

## Full-cycle reports

The full-cycle `full_cycle_summary.html` is the orchestration-level landing page. It should link cleanly to:

- dataset manifest and dataset HTML
- suite summary JSON and suite HTML
- optional presentation output

## Presentation authoring guidance

When building new repository presentations, use:

- `ppt_template.pptx`
- [Presentation Authoring](presentation_authoring.md)

This guidance defines the standing slide doctrine for future `.pptx` work in this repository.

## Inference outputs

### Single-image inference

Primary outputs:

- terminal JSON
- GUI probability table and preview panels

### Sampled `.oh5` inference

Primary outputs:

- `sample_predictions.csv`
- `sample_predictions.json`
- `scan_summary.csv`
- `summary.json`
- `summary.md`

### Full-scan GUI inference

Primary outputs:

- predicted phase map
- per-phase count/fraction/mean-score table
- dominant phase and mean confidence summary

## Common interpretation rule

No single headline metric should be read in isolation. Always read:

- counts
- per-class behavior
- confidence or runner-up structure
- provenance and split context

before drawing conclusions.
