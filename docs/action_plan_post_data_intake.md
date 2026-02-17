# Post-Data Intake Action Plan (Phase-Gated)

Last updated: 2026-02-17

## Purpose

Define the execution plan after receiving the completed student data packet. This plan converts current documentation scaffolding into a validated EBSD phase-identification pipeline based on masked NCC between experimental and simulated patterns.

This document is operational. It is intended to be used as the implementation playbook and gate checklist.

## Current Status Report

### What is already in place

- Mission, governance, roadmap, and architecture docs are established.
- `.oh5` structure reference is documented in `docs/oh5_structure.md`.
- Student-facing data packet and templates are prepared (`student_data_packet_phaseid/` and zip).
- Test data specification and naming/metadata policy are documented in `docs/test_data_setup_plan.md`.

### What is missing

- No production `src/` modules are implemented yet.
- No CLI workflows exist yet for curated-case scoring or scan-level extraction.
- No unit/integration tests exist yet for the target algorithm.
- No baseline result report exists yet.

### Immediate dependency

- Receive completed data packet with:
  - experimental patterns,
  - simulated pattern candidates,
  - three phase-isolated `.oh5` files,
  - completed JSON templates.

## Execution Principles

- Accuracy and traceability first.
- Every phase ends with a gate decision: `GO`, `HOLD`, or `REWORK`.
- No phase advances without gate evidence captured in report artifacts.
- All implementation changes must update docs and `todo_list.md`.

## Gate Overview

| Gate | Phase | Objective |
| --- | --- | --- |
| G0 | Data Intake and Integrity | Confirm received data is complete, consistent, and machine-readable. |
| G1 | Core Scoring Foundation | Implement deterministic preprocessing and masked NCC primitives. |
| G2 | Curated Case Pipeline | Score curated experimental/simulated sets and produce per-case ranking reports. |
| G3 | `.oh5` Candidate Extraction | Extract per-pixel phase-isolated candidates from three `.oh5` runs. |
| G4 | Integrated Decision Workflow | Combine extraction + scoring + winner selection on benchmark points. |
| G5 | Validation and Metrics | Quantify correctness and confidence separation on manual benchmark sets. |
| G6 | Hardening and Reproducibility | Add robust tests, debug workflows, manifests, and failure handling. |
| G7 | Baseline Freeze | Freeze baseline method, docs, and outputs for future ablations. |

## Phase Details and Gate Criteria

### G0: Data Intake and Integrity

### Entry

- Student packet received and unzipped.

### Tasks

1. Validate JSON syntax and required keys in all template files.
2. Validate file existence for every referenced path.
3. Validate phase labels are restricted to:
   - `fe_bcc`
   - `fe3o4_magnetite`
   - `feo_wustite`
4. Confirm each curated record has exactly 3 simulated candidates.
5. Confirm `.oh5` triad belongs to same scan grid (`nx`, `ny`, pattern shape).

### Deliverables

- CLI: `scripts/run_g0_data_intake_validation.py`
- `reports/data_intake_validation.md`
- `reports/data_intake_manifest.json`

### Exit Criteria (Gate Decision)

- 100% required files/keys pass.
- No unresolved missing-path errors.
- All schema checks pass.

If failed: stop and issue a data-fix request; do not start coding phases dependent on invalid inputs.

### G1: Core Scoring Foundation

### Entry

- G0 is `GO`.

### Tasks

1. Implement module skeleton under `src/phase_id_xcorr/` based on `docs/architecture.md`.
2. Implement deterministic preprocessing:
   - dtype conversion,
   - circular mask,
   - normalization policy from processing JSON.
3. Implement masked NCC with explicit edge-case handling (zero variance, all-masked).
4. Add logging utilities and run metadata writer.

### Deliverables

- Core modules in:
  - `src/phase_id_xcorr/preprocessing/`
  - `src/phase_id_xcorr/similarity/`
  - `src/phase_id_xcorr/reporting/`
- Unit tests for preprocessing + NCC.

### Exit Criteria (Gate Decision)

- Unit tests pass for NCC edge cases.
- Deterministic repeatability confirmed on same inputs.

### G2: Curated Case Pipeline

### Entry

- G1 is `GO`.

### Tasks

1. Implement curated manifest loader and validator.
2. Implement case runner:
   - load one experimental pattern,
   - load 3 simulated candidates,
   - compute NCC for all,
   - rank candidates and pick winner.
3. Persist detailed per-case evidence table.

### Deliverables

- CLI: `scripts/run_curated_ncc.py` with `--debug`.
- Report artifacts:
  - `reports/curated_case_scores.csv`
  - `reports/curated_summary.json`
  - `reports/curated_error_cases.md`

### Exit Criteria (Gate Decision)

- End-to-end curated run completes without manual intervention.
- Every case produces ranked 3-candidate output.
- Failure cases are explicit and traceable.

### G3: `.oh5` Candidate Extraction

### Entry

- G2 is `GO`.

### Tasks

1. Implement `.oh5` reader with scan-group discovery and field aliasing.
2. Extract orientation candidates (`Phi1`, `Phi`, `Phi2`) per pixel for each phase-isolated file.
3. Implement consistency checks across triad `.oh5` files.
4. Build benchmark-point extractor for manual validation points.

### Deliverables

- `src/phase_id_xcorr/io/oh5_reader.py`
- `src/phase_id_xcorr/indexing/tsl_candidates.py`
- CLI: `scripts/extract_scan_candidates.py`
- Output tables under `reports/scan_candidates/`.

### Exit Criteria (Gate Decision)

- Candidate extraction succeeds for all benchmark points.
- Grid/shape consistency checks pass or fail with explicit diagnostics.

### G4: Integrated Decision Workflow

### Entry

- G3 is `GO`.

### Tasks

1. Join candidate orientations with corresponding simulated patterns.
2. Run NCC-based winner selection per point/case.
3. Compute confidence margin:
   - `margin = best_ncc - second_best_ncc`.
4. Add decision confidence flags (for example: low-margin ambiguity).

### Deliverables

- `src/phase_id_xcorr/decision/selector.py`
- `scripts/run_phase_decision.py`
- `reports/decision_results.csv`

### Exit Criteria (Gate Decision)

- Workflow runs deterministically on benchmark set.
- Output includes winner, runner-up, all NCC values, and margins.

### G5: Validation and Metrics

### Entry

- G4 is `GO`.

### Tasks

1. Define baseline validation metrics:
   - top-1 phase accuracy,
   - confusion matrix,
   - mean margin by predicted phase,
   - ambiguous-case rate (below margin threshold),
   - fallback-orientation usage rate.
2. Compare predictions against manual benchmark labels.
3. Document failure modes and likely root causes.

### Deliverables

- `reports/baseline_metrics.json`
- `reports/baseline_validation.md`
- `docs/results_baseline.md`

### Exit Criteria (Gate Decision)

- Metrics are reproducible across repeated runs.
- Error analysis is complete for misclassified and low-margin cases.

### G6: Hardening and Reproducibility

### Entry

- G5 is `GO`.

### Tasks

1. Add integration test for full debug workflow.
2. Ensure every CLI supports:
   - `--debug`,
   - structured logging,
   - machine-readable run manifest.
3. Add robust input validation and user-facing error messages.
4. Update usage docs and method assumptions.

### Deliverables

- Unit + integration tests in `tests/`.
- Stable debug scripts and manifests.
- Updated docs (`status`, `todo`, methods, usage).

### Exit Criteria (Gate Decision)

- Full debug pipeline passes in one command sequence.
- Reports and manifests are generated for every run.

### G7: Baseline Freeze

### Entry

- G6 is `GO`.

### Tasks

1. Freeze baseline configuration and processing settings.
2. Tag baseline outputs for future comparisons.
3. Archive reproducibility bundle:
   - configs,
   - manifests,
   - metrics,
   - error analysis.

### Deliverables

- `reports/baseline_freeze/`
- `docs/status.md` and `todo_list.md` updated to next phase.

### Exit Criteria (Gate Decision)

- Baseline can be rerun and reproduced from repository artifacts only.

## Metric Definition (Initial Baseline)

- `top1_accuracy`: fraction of points where predicted phase equals manual expected phase.
- `mean_ncc_winner`: average highest NCC across points.
- `mean_margin`: average `(best - second_best)` NCC.
- `low_margin_rate`: fraction with margin below threshold (threshold recorded in config).
- `fallback_candidate_rate`: fraction where winning candidate came from fallback orientation.

Thresholds are provisional in first pass and can be adjusted only with documented rationale.

## Strategic Evolution Policy (Vision/Goal Adjustment)

The mission and objectives may evolve based on evidence. Changes are allowed only when:

1. A concrete failure mode is documented.
2. An alternative strategy is proposed with expected impact.
3. Evaluation criteria are updated before implementation.
4. `docs/mission_statement.md`, `docs/roadmap.md`, and `todo_list.md` are synchronized.

This prevents scope drift while keeping the project scientifically adaptive.

## Future Work (After Baseline Freeze)

1. Add alternative similarity metrics (ZNCC variants, masked phase correlation).
2. Explore decision ensembles combining NCC + indexing confidence features.
3. Add sparse LRS integration interfaces and calibration experiments.
4. Expand benchmark set and perform phase-wise robustness analysis.
5. Build manuscript-ready automated figures/tables from report artifacts.
