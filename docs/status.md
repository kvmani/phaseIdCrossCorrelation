# Project Status Snapshot

Last updated: 2026-02-21

## Purpose

Build a modular EBSD phase-identification workflow that improves discrimination among magnetite/wustite/iron using phase-isolated indexing candidates and masked NCC against externally simulated patterns.

## Current State

- Repository scaffold, governance docs, and architecture blueprint are in place.
- Student-facing data collection packet has been prepared:
  - `student_data_packet_phaseid/`
  - `student_data_packet_phaseid.zip`
- Test data contract and templates are now defined and documented.
- G0 validator tooling is implemented and runnable:
  - `scripts/run_g0_data_intake_validation.py`
  - `src/phase_id_xcorr/intake/g0_validator.py`
- Initial G0 run on template packet correctly returns `HOLD` because real files are not populated yet.
- Imported student packet in repo test data now passes G0 with `GO`.
- Curated NCC implementation is now available:
  - `scripts/run_curated_ncc.py`
  - `src/phase_id_xcorr/preprocessing/*`
  - `src/phase_id_xcorr/similarity/ncc.py`
  - `src/phase_id_xcorr/evaluation/curated_ncc.py`
- Curated workflow artifacts are generated under `reports/curated_ncc/`.
- Single-file inspection artifact is available:
  - `reports/curated_ncc/inspection_report.html`
- Scientific recovery strategy for band-aware phase identification is documented:
  - `docs/scientific_strategy_band_aware_phase_id.md`
- Hough-space NCC branch critical analysis and execution plan is documented:
  - `docs/hough_space_ncc_action_plan.md`
- Initial feasibility probe artifact is available:
  - `reports/curated_ncc/hough_feasibility_probe.md`
- KikuchiPy Hough comparison workflow is implemented:
  - `scripts/run_curated_hough_vs_ncc.py`
  - `src/phase_id_xcorr/features/kikuchipy_hough.py`
  - `src/phase_id_xcorr/evaluation/curated_hough_vs_ncc.py`
- Hough comparison usage doc is available:
  - `docs/curated_hough_vs_ncc_workflow.md`
- Curated image-vs-hough artifacts are generated under:
  - `reports/curated_hough_vs_ncc/`

## Confirmed Constraints

- Raman data is sparse and partial; LRS integration is later-phase only.
- Spatial registration can be assumed acceptable for now.
- TSL indexing and EMSoft simulations are external and provided to this repo.
- Baseline similarity metric is masked NCC.
- CPU target environment.
- In-repo simple test data should be used for debug/development.
- Success criterion is correctness on manually identified benchmark cases.

## Current Risks

- Intensity-only NCC can collapse predictions to a single phase on curated cases.
- Failed/fallback candidates can dominate winner selection if unpenalized.
- Band visibility and contrast mismatch between experiment and simulation can reduce discriminative power.
- Hough-space similarity can overfit tiny curated sets if parameters are tuned without robustness gates.

## Immediate Next Steps

1. Run robustness/overfitting checks (H4) on the new image-vs-hough workflow and lock stable defaults.
2. Implement fallback-aware winner policy with uncertainty gate (B1).
3. Add hybrid decision track (H3b) and choose default (`image`, `hough`, or `hybrid`) before expanding data volume.

## Future Work Summary

- Baseline freeze with reproducible metrics and error analysis.
- Alternative scoring/decision ablations.
- LRS-ready extension interfaces.
- Manuscript-oriented reporting workflows after baseline stabilization.
