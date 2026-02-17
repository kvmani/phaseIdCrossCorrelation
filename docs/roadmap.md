# Roadmap

This roadmap is the long-horizon plan. For daily execution, use `todo_list.md`.
For post-data phase-gated implementation detail, use `docs/action_plan_post_data_intake.md`.

## Phase 0: Documentation and Architecture Freeze (Current)

Objectives:

- Finalize mission statement, governance policy, and task tracking.
- Freeze initial repository layout and naming conventions.
- Document `.oh5` structure and access patterns.

Deliverables:

- `README.md`, `AGENTS.md`, `docs/mission_statement.md`, `docs/oh5_structure.md`, `todo_list.md`.

## Phase 1: EBSD Baseline Pipeline (NCC)

Objectives:

- Build EBSD-only modular pipeline.
- Ingest TSL `.oh5` outputs for phase-isolated indexing candidates.
- Compute masked NCC between experimental and externally simulated candidate patterns.
- Select phase by highest NCC and store confidence evidence.

Deliverables:

- Minimal end-to-end debug workflow using in-repo test data.
- Unit tests for `.oh5` read layer and NCC scoring.
- CLI script for single-case and batch-case evaluation.

## Phase 2: Validation Framework

Objectives:

- Define and curate manually verified benchmark cases.
- Quantify classification correctness and confidence separation.
- Document failure modes and ambiguity cases.

Deliverables:

- Evaluation protocol doc.
- Case-level report generation utilities.
- Baseline result snapshot in `docs/results_baseline.md`.

## Phase 3: Alternative Strategies and Ablations

Objectives:

- Add plug-in scoring methods beyond NCC (if needed).
- Evaluate sensitivity to masking, normalization, and candidate quality.
- Compare decision rules (max NCC, margin thresholds, consensus rules).

Deliverables:

- Ablation scripts and reports.
- Updated recommendations for default pipeline settings.

## Phase 4: LRS-Ready Extension Layer

Objectives:

- Add interfaces for sparse Raman labels and optional fusion logic.
- Keep EBSD-only path unchanged and stable.
- Prototype weak/sparse supervision and calibration ideas.

Deliverables:

- LRS data-contract specification.
- Optional fusion module with controlled experiments.

## Phase 5: Manuscript-Ready Packaging

Objectives:

- Consolidate methods, results, and reproducibility artifacts.
- Standardize figure/table generation workflows.
- Prepare draft manuscript assets from pipeline outputs.

Deliverables:

- Methods and results docs synced with implementation.
- Reproducibility bundle and report scripts.
