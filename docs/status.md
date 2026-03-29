# Project Status Snapshot

Last updated: 2026-03-29

## Purpose

Build a modular EBSD phase-identification workflow that improves discrimination among magnetite/wustite/iron using reproducible evidence tracks:

- masked NCC against external simulations,
- supervised ML classification on experimental Kikuchi patterns.

## Current State

- Curated NCC baseline is implemented and runnable.
- Curated image-vs-Hough comparison is implemented and runnable.
- G0 intake validation and `.oh5` structure guidance are in place.
- ML dataset preparation, training, benchmark suite, and reporting are implemented.
- A native raw `.oh5` phase explorer GUI is implemented for exploratory analysis before dataset freezing, with auto-exported publication PNG histograms and JSON metadata in the configured output directory.
- A diagnostic pattern gallery GUI is implemented for cross-condition inspection, manual index lookup, and reproducible JSON/contact-sheet export.
- Debug-scale tests cover NCC, Hough, `.oh5` ingestion, dataset prep, splitting, quality gating, training smoke, and suite orchestration.

## Confirmed Constraints

- Raman data is sparse and partial; LRS integration remains later-phase only.
- TSL indexing and EMSoft simulations are external inputs to this repository.
- `.oh5` naming/field variability must be handled with alias-aware readers.
- Some `.oh5` scan exports may omit `Pattern` datasets; ML data prep must fail clearly or skip by policy.
- Priority order remains correctness > reproducibility > maintainability > speed.

## Current Risks

- Intensity-only NCC can fail to separate visually similar phases on curated cases.
- Ground-truth CSV quality and coverage directly constrain ML training reliability.
- Data leakage risk exists if split policy is not deterministic and scan-aware.
- Small dataset regimes can overfit quickly; reporting and uncertainty analysis are mandatory.

## Immediate Next Steps

1. Define and freeze the first manually verified NCC-vs-ML benchmark set.
2. Publish baseline comparative reports instead of workflow-only docs.
3. Finish remaining scan-scale NCC candidate ingestion and integrated decision logic.
4. Expand calibration and robustness reporting for small-data ML runs.
5. Freeze a reproducible baseline config bundle for comparative reruns.

## Future Work Summary

- Comparative NCC vs ML benchmark on manually verified cases.
- Multi-backbone ablation and default model freeze.
- Hybrid NCC+ML decision policy and uncertainty gating.
- LRS-ready multimodal extension after EBSD ML baseline stabilization.
