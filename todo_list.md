# TODO List

Operational task list for this repository. Keep this file updated as priorities evolve.

## Priority Now

- [ ] Review and finalize all scaffold documents (`README.md`, `AGENTS.md`, mission, roadmap, status, `.oh5` guide).
- [ ] Approve initial repository module boundaries under `src/`.
- [x] Define minimum debug dataset contents and naming conventions in `data/test/` (`docs/test_data_setup_plan.md`).
- [x] Prepare student-facing data packet and templates (`student_data_packet_phaseid/`, zip archive).
- [x] Populate `data/test/` using the approved naming and manifest scheme in `docs/test_data_setup_plan.md`.
- [ ] Define the first benchmark set of manually verified pixels/cases.

## Post-Data Intake Gates (Execution Order)

Reference: `docs/action_plan_post_data_intake.md`

- [x] G0: Validate incoming student data package (JSON + file paths + triad `.oh5` consistency).
- [x] G1: Implement preprocessing and masked NCC foundation modules.
- [x] G2: Implement curated-case NCC runner and ranking outputs.
- [ ] G3: Implement `.oh5` candidate extraction from phase-isolated scan files.
- [ ] G4: Implement integrated decision workflow (winner + margin + evidence table).
- [ ] G5: Implement baseline validation metrics and benchmark report.
- [ ] G6: Add end-to-end debug integration tests and run manifests.
- [ ] G7: Freeze baseline configuration and reproducibility bundle.

## Scientific Recovery Plan (Band-Aware)

Reference: `docs/scientific_strategy_band_aware_phase_id.md`

- [ ] B1: Add fallback-aware winner policy and `uncertain` decision gate.
- [ ] B2: Add gradient/band-aware metrics (`S_grad`, `S_ori`, `S_edge`) with ablation report.
- [ ] B3: Add robustness stress tests (noise/gain/blur/partial visibility) and confidence calibration.
- [ ] B4: Integrate improved decision stack with `.oh5` candidate ingestion for manual benchmark pixels.

## Hough-Space Branch Plan

Reference: `docs/hough_space_ncc_action_plan.md`

- [x] H0: Freeze dual-method comparison protocol on current curated set.
- [x] H1: Implement Hough feature extraction module using KikuchiPy Hough/Radon plan.
- [x] H2: Implement Hough-space similarity metrics (continuous-map NCC + thresholded binary-map NCC).
- [x] H3a: Extend curated workflow to emit image-vs-hough decision tracks and single HTML report.
- [ ] H3b: Add hybrid decision track and explicit fusion policy.
- [ ] H4: Run robustness/overfitting checks and select stable Hough parameters.
- [ ] H5: Publish scientific comparison report and choose keep/reject/default strategy.

## Phase 1: EBSD Baseline Implementation (Work Packages)

- [x] Implement multi-format image reader (`.bmp`, `.png`, `.tif/.tiff`, `.jpg/.jpeg`) with consistent grayscale conversion.
- [x] Support both 8-bit and 16-bit inputs in preprocessing (normalize to canonical float range before NCC).
- [ ] Implement `.oh5` reader layer with robust scan-group discovery and field aliasing.
- [ ] Implement data model for per-pixel candidate orientations from phase-isolated TSL runs.
- [x] Implement masked NCC scoring module with deterministic preprocessing.
- [x] Implement decision module selecting winning phase/orientation and confidence margin.
- [x] Add CLI workflow for single-case and batch-case runs.
- [x] Add run `manifest.json` outputs for reproducibility.

## Testing and Validation

- [x] Implement G0 validator CLI + report generation (`scripts/run_g0_data_intake_validation.py`).
- [x] Add unit tests for G0 validator behavior (`tests/test_g0_intake_validator.py`).
- [x] Add unit tests for image loading + bit-depth handling (`tests/test_image_io.py`).
- [x] Add unit tests for masked NCC metric behavior (`tests/test_ncc.py`).
- [x] Add curated NCC workflow integration test (`tests/test_curated_ncc.py`).
- [x] Add KikuchiPy Hough feature extraction unit tests (`tests/test_kikuchipy_hough.py`).
- [x] Add curated image-vs-hough workflow integration test (`tests/test_curated_hough_vs_ncc.py`).
- [ ] Unit tests for `.oh5` dataset-path discovery and shape handling.
- [ ] Integration test for debug pipeline end-to-end.
- [ ] Build first correctness report on manually identified benchmark cases.

## Documentation and Reporting

- [ ] Add methods doc for exact preprocessing + masked NCC formula used in code.
- [x] Add usage doc for curated NCC CLI workflow (`docs/curated_ncc_workflow.md`).
- [x] Add usage doc for curated image-vs-hough workflow (`docs/curated_hough_vs_ncc_workflow.md`).
- [x] Add full-cycle one-go runbook for baseline NCC vs Hough comparison (`docs/mcc_vs_hough_full_cycle_runbook.md`).
- [ ] Add results baseline document with case-level evidence tables.
- [x] Add phase-gated execution plan for post-data implementation (`docs/action_plan_post_data_intake.md`).
- [x] Add G0 validation usage documentation (`docs/g0_data_intake_validation.md`).

## Later

- [ ] Add optional alternative similarity metrics for ablation.
- [ ] Add LRS fusion-ready interfaces and sparse label handling.
- [ ] Build manuscript asset generation workflow (figures, tables, supplementary).
