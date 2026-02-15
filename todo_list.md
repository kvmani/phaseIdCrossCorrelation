# TODO List

Operational task list for this repository. Keep this file updated as priorities evolve.

## Priority Now

- [ ] Review and finalize all scaffold documents (`README.md`, `AGENTS.md`, mission, roadmap, status, `.oh5` guide).
- [ ] Approve initial repository module boundaries under `src/`.
- [x] Define minimum debug dataset contents and naming conventions in `data/test/` (`docs/test_data_setup_plan.md`).
- [ ] Populate `data/test/` using the approved naming and manifest scheme in `docs/test_data_setup_plan.md`.
- [ ] Define the first benchmark set of manually verified pixels/cases.

## Phase 1: EBSD Baseline Implementation

- [ ] Implement `.oh5` reader layer with robust scan-group discovery and field aliasing.
- [ ] Implement data model for per-pixel candidate orientations from phase-isolated TSL runs.
- [ ] Implement masked NCC scoring module with deterministic preprocessing.
- [ ] Implement decision module selecting winning phase/orientation and confidence margin.
- [ ] Add CLI workflow for single-case and batch-case runs.
- [ ] Add run `manifest.json` outputs for reproducibility.

## Testing and Validation

- [ ] Unit tests for `.oh5` dataset-path discovery and shape handling.
- [ ] Unit tests for NCC correctness and edge cases (zero variance, masked regions).
- [ ] Integration test for debug pipeline end-to-end.
- [ ] Build first correctness report on manually identified benchmark cases.

## Documentation and Reporting

- [ ] Add methods doc for exact preprocessing + masked NCC formula used in code.
- [ ] Add usage doc for baseline CLI workflows.
- [ ] Add results baseline document with case-level evidence tables.

## Later

- [ ] Add optional alternative similarity metrics for ablation.
- [ ] Add LRS fusion-ready interfaces and sparse label handling.
- [ ] Build manuscript asset generation workflow (figures, tables, supplementary).
