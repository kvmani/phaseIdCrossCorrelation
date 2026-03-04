# Project Status Snapshot

Last updated: 2026-03-04

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
- ML dataset preparation is implemented with two label-input modes:
  - `.oh5` + per-pixel CSV labels,
  - single-phase scan-map (`.oh5` file-level phase mapping).
- ML runbooks now include platform-specific execution guidance for Linux/macOS, Windows (PyCharm terminal), and SLURM-based HPC environments.

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

1. Complete NCC-vs-ML benchmark protocol on manually verified cases.
2. Harden training/evaluation defaults and publish baseline benchmark reports.
3. Add explicit scan-aware split options to reduce leakage risk in small-data regimes.
4. Expand robustness checks and uncertainty calibration reporting.
5. Freeze a reproducible baseline config bundle for comparative runs.

## Future Work Summary

- Comparative NCC vs ML benchmark on manually verified cases.
- Multi-backbone ablation and default model freeze.
- Hybrid NCC+ML decision policy and uncertainty gating.
- LRS-ready multimodal extension after EBSD ML baseline stabilization.
