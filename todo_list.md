# TODO List

Operational task list for this repository. Keep this file updated as priorities evolve.

## Priority Now

- [ ] Review and finalize all scaffold documents (`README.md`, `AGENTS.md`, mission, roadmap, status, `.oh5` guide).
- [ ] Approve initial repository module boundaries under `src/`.
- [x] Define minimum debug dataset contents and naming conventions in `data/test/` (`docs/test_data_setup_plan.md`).
- [x] Prepare student-facing data packet and templates (`student_data_packet_phaseid/`, zip archive).
- [ ] Populate `data/test/` using the approved naming and manifest scheme in `docs/test_data_setup_plan.md`.
- [ ] Define the first benchmark set of manually verified pixels/cases.

## Post-Data Intake Gates (Execution Order)

Reference: `docs/action_plan_post_data_intake.md`

- [ ] G0: Validate incoming student data package (JSON + file paths + triad `.oh5` consistency).
- [ ] G1: Implement preprocessing and masked NCC foundation modules.
- [ ] G2: Implement curated-case NCC runner and ranking outputs.
- [ ] G3: Implement `.oh5` candidate extraction from phase-isolated scan files.
- [ ] G4: Implement integrated decision workflow (winner + margin + evidence table).
- [ ] G5: Implement baseline validation metrics and benchmark report.
- [ ] G6: Add end-to-end debug integration tests and run manifests.
- [ ] G7: Freeze baseline configuration and reproducibility bundle.

## Phase 1: EBSD Baseline Implementation (Work Packages)

- [ ] Implement `.oh5` reader layer with robust scan-group discovery and field aliasing.
- [ ] Implement data model for per-pixel candidate orientations from phase-isolated TSL runs.
- [ ] Implement masked NCC scoring module with deterministic preprocessing.
- [ ] Implement decision module selecting winning phase/orientation and confidence margin.
- [ ] Add CLI workflow for single-case and batch-case runs.
- [ ] Add run `manifest.json` outputs for reproducibility.

## Testing and Validation

- [x] Implement G0 validator CLI + report generation (`scripts/run_g0_data_intake_validation.py`).
- [x] Add unit tests for G0 validator behavior (`tests/test_g0_intake_validator.py`).
- [ ] Unit tests for `.oh5` dataset-path discovery and shape handling.
- [ ] Unit tests for NCC correctness and edge cases (zero variance, masked regions).
- [ ] Integration test for debug pipeline end-to-end.
- [ ] Build first correctness report on manually identified benchmark cases.

## Documentation and Reporting

- [ ] Add methods doc for exact preprocessing + masked NCC formula used in code.
- [ ] Add usage doc for baseline CLI workflows.
- [ ] Add results baseline document with case-level evidence tables.
- [x] Add phase-gated execution plan for post-data implementation (`docs/action_plan_post_data_intake.md`).
- [x] Add G0 validation usage documentation (`docs/g0_data_intake_validation.md`).

## Later

- [ ] Add optional alternative similarity metrics for ablation.
- [ ] Add LRS fusion-ready interfaces and sparse label handling.
- [ ] Build manuscript asset generation workflow (figures, tables, supplementary).
