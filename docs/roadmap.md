# Roadmap

This roadmap is the long-horizon plan. For daily execution, use `todo_list.md`.
For post-data phase-gated implementation detail, use `docs/action_plan_post_data_intake.md`.

## Phase 0: Documentation and Architecture Sync (Current)

Objectives:

- Update mission, governance, and task tracking for dual-track phase ID (NCC + ML).
- Freeze ML module boundaries and config contracts.
- Document `.oh5` pattern-extraction assumptions and CSV label contract.

Deliverables:

- Updated: `README.md`, `AGENTS.md`, `docs/mission_statement.md`, `todo_list.md`, `docs/status.md`.
- New ML design docs and YAML config templates.

## Phase 1A: EBSD Baseline Pipeline (NCC)

Objectives:

- Maintain and harden EBSD-only NCC baseline workflow.
- Ingest TSL `.oh5` outputs for phase-isolated indexing candidates.
- Compute masked NCC between experimental and externally simulated candidate patterns.
- Select phase by highest NCC and store confidence evidence.

Deliverables:

- End-to-end debug workflow using in-repo test data.
- Unit tests for `.oh5` read layer and NCC scoring.
- CLI workflow for curated single-case and batch-case evaluation.

## Phase 1B: EBSD ML Classifier Pipeline (New)

Objectives:

- Build supervised classifier workflow for phase labels from Kikuchi patterns.
- Ingest multiple `.oh5` + CSV label pairs and combine into one dataset.
- Apply configurable quality filters and deterministic split policy.
- Train/evaluate configurable backbones with scratch and pretrained options.

Deliverables:

- ML data preparation CLI (`.oh5` extraction + filtering + split artifacts).
- ML training CLI (config-driven model, optimizer, reporting, checkpoints).
- Reproducible run artifacts (`manifest.json`, metrics report, split stats).

## Phase 2: Comparative Validation Framework

Objectives:

- Define manually verified benchmark protocol for NCC vs ML comparison.
- Quantify correctness, calibration, and failure modes per phase.
- Establish acceptance gates for promoting a default classifier model.

Deliverables:

- Evaluation protocol and benchmark manifest.
- Case-level and aggregate reports (accuracy, macro-F1, confusion matrix).
- Baseline comparison snapshot in docs.

## Phase 3: Model Selection and Ablations

Objectives:

- Compare at least five open-source classifier backbones.
- Evaluate scratch vs pretrained initialization effects.
- Test sensitivity to quality filters, split seeds, and preprocessing choices.

Deliverables:

- Model-selection report with rationale and residual risks.
- Ranked benchmark table and recommended default model family.
- Reproducible benchmark suite config for repeated runs.

## Phase 4: Hybrid Decision Track (NCC + ML)

Objectives:

- Define fusion strategy between NCC evidence and ML probabilities.
- Add uncertainty policy for disagreement and low-confidence regions.
- Preserve auditable evidence trail for each decision.

Deliverables:

- Hybrid decision module and policy config.
- Comparative report for NCC-only, ML-only, and hybrid tracks.

## Phase 5: LRS-Ready Extension Layer

Objectives:

- Add interfaces for sparse Raman labels and optional fusion logic.
- Keep EBSD-only path unchanged and stable.
- Prototype weak/sparse supervision and calibration ideas.

Deliverables:

- LRS data-contract specification.
- Optional fusion module with controlled experiments.

## Phase 6: Manuscript-Ready Packaging

Objectives:

- Consolidate methods, results, and reproducibility artifacts.
- Standardize figure/table generation workflows.
- Prepare draft manuscript assets from pipeline outputs.

Deliverables:

- Methods/results docs synced with implementation.
- Reproducibility bundle and report scripts.
