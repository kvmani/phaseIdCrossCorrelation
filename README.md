# phaseIdCrossCorrelation

Accuracy-first EBSD phase identification for mixed-phase scans using two evidence tracks:

- masked NCC against externally simulated patterns,
- supervised ML classification from experimental Kikuchi patterns.

## What This Repo Does

The current target problem is phase discrimination among magnetite, wustite, and iron when conventional indexing is unstable. The repository keeps both evidence tracks reproducible and traceable so they can later be compared or fused.

Implemented today:

- curated NCC workflow,
- curated image-vs-Hough comparison workflow,
- ML dataset preparation from `.oh5`,
- ML training and benchmark-suite runners,
- raw `.oh5` phase explorer GUI,
- diagnostic pattern gallery GUI for cross-condition model inspection.

## Start Here

- New to the project: `docs/README.md`
- Scientific scope and success criteria: `docs/mission_statement.md`
- Current state and near-term risks: `docs/status.md`
- Active task list: `todo_list.md`
- Repo working rules: `AGENTS.md`

## Common Entry Points

### NCC and Hough

- `docs/curated_ncc_workflow.md`
- `docs/curated_hough_vs_ncc_workflow.md`
- `docs/mcc_vs_hough_full_cycle_runbook.md`

### ML

- `docs/ml_classifier_workflow.md`
- `docs/ml_input_data_runbook.md`
- `docs/ml_training_inference_workflow.md`
- `docs/ml_phase_explorer_gui.md`
- `docs/ml_diagnostic_gallery_gui.md`
- `docs/ml_model_selection.md`

### Data Contracts

- `docs/oh5_structure.md`
- `docs/g0_data_intake_validation.md`
- `configs/ml/README.md`

## Repository Layout

```text
phaseIdCrossCorrelation/
├─ AGENTS.md
├─ README.md
├─ todo_list.md
├─ docs/
├─ configs/
├─ data/
├─ scripts/
├─ src/
├─ tests/
└─ reports/
```

## Working Principles

- correctness over speed,
- reproducibility over convenience,
- explicit artifacts and manifests for runnable workflows,
- thin CLIs and modular `src/` code,
- docs updated with behavior changes.

## External Interfaces

- `.oh5` files are the source-of-truth EBSD container.
- TSL indexing and external simulation stay outside this repository.
- ML labels come from config-defined CSV labels or single-phase scan mapping.
