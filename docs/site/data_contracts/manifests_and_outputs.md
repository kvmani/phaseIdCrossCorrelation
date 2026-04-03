# Manifests and Outputs

The repository writes machine-readable manifests because reproducibility is a core requirement, not an optional report decoration.

## Dataset-prep outputs

- `manifest.json`
- `events.jsonl`
- `records.csv`
- split `.npz` bundles
- orientation CSV/JSON exports
- IPF plot index JSON
- HTML summary

## Training outputs

- `manifest.json`
- `events.jsonl`
- `report.json`
- `report.md`
- checkpoints
- epoch history

## Benchmark outputs

- `manifest.json`
- `suite_summary.json`
- `suite_summary.md`
- `suite_report.html`

## Full-cycle outputs

- `manifest.json`
- `full_cycle_summary.json`
- `full_cycle_summary.html`
- resolved configs
- optional PPTX

## Design intent

Every workflow must answer:

- what config was used
- what inputs were consumed
- what outputs were generated
- what provenance or summary counts explain the result

This is why manifests use repo-relative paths by default and why HTML summaries link back to machine-readable artifacts.
