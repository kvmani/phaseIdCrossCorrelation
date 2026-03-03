# Project Status Snapshot

Last updated: 2026-03-02

## Purpose

Build a modular EBSD phase-identification workflow that improves discrimination among magnetite/wustite/iron using reproducible evidence tracks:

- masked NCC against external simulations,
- supervised ML classification on experimental Kikuchi patterns.

## Current State

- Repository scaffold, governance docs, and baseline NCC workflows are in place.
- Curated NCC and image-vs-Hough comparison workflows are implemented and runnable.
- Test data packet and G0 validation tooling are implemented.
- `.oh5` structure guide is documented (`docs/oh5_structure.md`).
- ML classifier expansion has been formally approved and moved into implementation scope.
- Dedicated ML architecture and workflow documentation is being added with config-first contracts.

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

1. Finalize ML docs/config contracts and dedicated module scaffold.
2. Implement `.oh5` + CSV dataset preparation with quality filters and deterministic splits.
3. Implement configurable training/evaluation runner with scratch/pretrained options.
4. Add tests for extraction, filtering, split determinism, and training smoke path.
5. Run first debug benchmark and publish run artifacts in `reports/`.

## Future Work Summary

- Comparative NCC vs ML benchmark on manually verified cases.
- Multi-backbone ablation and default model freeze.
- Hybrid NCC+ML decision policy and uncertainty gating.
- LRS-ready multimodal extension after EBSD ML baseline stabilization.
