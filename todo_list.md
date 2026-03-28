# TODO List

Operational task list for this repository. Keep this file updated as priorities evolve.

## Priority Now

- [x] Synchronize mission/roadmap/status docs for dual-track EBSD phase ID (NCC + ML).
- [x] Define dedicated ML classifier module boundaries and configuration-first workflow scope.
- [x] Implement ML dataset preparation runner with quality filters, deterministic splits, and dual input modes (`.oh5` + CSV labels, single-phase scan map).
- [x] Implement ML classifier training runner with scratch/pretrained support and reproducible reports.
- [ ] Define the first benchmark set of manually verified pixels/cases for NCC-vs-ML comparison.

## Post-Data Intake Gates (NCC Track)

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

## ML Classifier Track (Phase 1B)

- [x] M0: Define ML scope, repository layout, and documentation contract.
- [x] M1: Add `.oh5` pattern extractor with robust field aliasing and pattern-presence checks.
- [x] M2: Add label ingestor modes supporting configurable CSV per-pixel labels and scan-level phase mapping.
- [x] M3: Add quality gating (CI/IQ/Fit/Valid) with YAML thresholds and per-file stats.
- [x] M4: Build combined dataset artifacts across multiple configured sources.
- [x] M5: Add deterministic stratified train/val/test splitting with seed control.
- [x] M6: Implement configurable classifier training/evaluation runner (timm backbones).
- [x] M7: Add scratch vs pretrained initialization mode and checkpoint/report artifacts.
- [ ] M8: Publish model-selection rationale for at least five open-source backbones.
- [x] M9: Add benchmark-suite runner for repeated multi-model experiments.
- [x] M10: Upgrade dataset-prep/training contract to v3 schema with expression quality filters, preprocessing fingerprints, leakage-safe capped splits, and HTML benchmark analytics.
- [x] M11: Add native desktop raw `.oh5` exploratory GUI for phase-wise cumulative histograms/CDF, discovered field distributions, and interactive intensity-band pixel highlighting.
- [x] M11b: Export publication-quality phase explorer histogram PNGs and a machine-readable JSON manifest with synchronized axes for intensity/IQ/Fit/CI comparisons.
- [x] M12: Add one-go ML full-cycle workflow with robust error checking, extensive logs, machine-ingestible manifests, concise HTML summary, and optional PPTX generation.
- [x] M13: Add production Ni-Cu-Al run scaffolding with richer dataset/suite analytics, Windows runbook, and lab-meeting-ready reporting.
- [x] M14: Add lightweight benchmark-suite packaging for mail-friendly transfer of summaries and manifests without checkpoints.
- [x] M15: Add saved-model inference CLI and desktop GUI for unknown-image phase prediction.
- [x] M16: Add sampled unseen-scan `.oh5` CNN inference runner with YAML-configured quality filtering, random per-scan sampling, and per-pattern/per-scan summary outputs.
- [x] M17: Add optional per-phase balancing during dataset prep by downsampling accepted samples to the smallest accepted phase count before split assignment.

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
- [x] ML unit tests for `.oh5` pattern extraction, label merge, split determinism, and single-phase scan-map input mode.
- [x] ML training smoke test on tiny deterministic dataset.

## Documentation and Reporting

- [x] Add documentation hub and tighten top-level navigation for workflow discovery.
- [ ] Add methods doc for exact preprocessing + masked NCC formula used in code.
- [x] Add usage doc for curated NCC CLI workflow (`docs/curated_ncc_workflow.md`).
- [x] Add usage doc for curated image-vs-hough workflow (`docs/curated_hough_vs_ncc_workflow.md`).
- [x] Add full-cycle one-go runbook for baseline NCC vs Hough comparison (`docs/mcc_vs_hough_full_cycle_runbook.md`).
- [ ] Add results baseline document with case-level evidence tables.
- [x] Add phase-gated execution plan for post-data implementation (`docs/action_plan_post_data_intake.md`).
- [x] Add G0 validation usage documentation (`docs/g0_data_intake_validation.md`).
- [x] Add ML workflow usage docs (dataset prep, training, benchmark suite, and auto-PPT reporting).
- [ ] Add ML model-catalog doc with pretrained provenance and citations.

## Later

- [ ] Add optional alternative similarity metrics for ablation.
- [ ] Add LRS fusion-ready interfaces and sparse label handling.
- [ ] Build manuscript asset generation workflow (figures, tables, supplementary).
