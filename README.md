# phaseIdCrossCorrelation

Accuracy-first EBSD phase identification for multi-phase scans using phase-isolated indexing candidates and normalized cross-correlation (NCC) against externally simulated patterns.

## Mission

This project addresses a practical failure mode in conventional EBSD indexing for mixed microstructures containing similar phases (currently: magnetite, wustite, and iron). In these cases, standard Hough/dictionary indexing can return unstable or incorrect phase labels, especially for magnetite vs wustite.

The core strategy in this repository is:

1. Run indexing externally in TSL/EDAX assuming one phase at a time.
2. For each pixel `(x, y)`, collect candidate orientation solutions `O1`, `O2`, `O3` (one per assumed phase).
3. Generate (externally) a simulated EBSD pattern for each candidate orientation.
4. Compute masked NCC between each simulated pattern and the experimental pattern.
5. Select the phase/orientation with the highest correlation and report confidence.

Current development scope is EBSD-only. Laser Raman Spectroscopy (LRS) integration is planned in later phases and the architecture is designed to support sparse Raman labels without major refactoring.

## Current Project Phase

This repository is currently in **data-intake-to-implementation transition** mode.

- No production algorithm implementation is committed yet.
- Student-facing data packet and templates are prepared.
- Immediate execution follows phase gates defined in `docs/action_plan_post_data_intake.md`.

## Documentation Map

- `docs/mission_statement.md`: scientific objective, assumptions, and success criteria.
- `AGENTS.md`: canonical engineering and agent workflow rules.
- `docs/roadmap.md`: phased delivery plan.
- `docs/status.md`: read-only snapshot of current state.
- `docs/action_plan_post_data_intake.md`: phase-gated implementation playbook after data intake.
- `docs/g0_data_intake_validation.md`: how to run G0 data-intake gate checks.
- `todo_list.md`: operational task list (living document).
- `docs/oh5_structure.md`: dedicated guide for TSL `.oh5` layout and data access.
- `docs/test_data_setup_plan.md`: canonical test data acquisition, naming, and manifest specification.
- `docs/agents.md`: pointer to root `AGENTS.md`.

## Intended Repository Layout

```text
phaseIdCrossCorrelation/
├─ AGENTS.md
├─ README.md
├─ todo_list.md
├─ docs/
├─ configs/
├─ data/
│  └─ test/
├─ scripts/
├─ src/
├─ tests/
└─ reports/
```

## Design Principles

- Accuracy and scientific defensibility over throughput.
- Modular architecture with explicit interfaces.
- CPU-first, deterministic, reproducible workflows.
- Debug-friendly scripts using in-repo test data.
- Documentation-first changes with synchronized task tracking.

## External Dependencies and Interfaces

- TSL indexing and EMSoft/TSL simulation are external inputs to this pipeline.
- TSL analysis outputs are expected as `.oh5` (HDF5) files.
- See `docs/oh5_structure.md` for canonical field paths and access patterns.

## Reference Repositories

- DeepImageDeconvolution (format inspiration): [kvmani/DeepImageDeconvolution](https://github.com/kvmani/DeepImageDeconvolution)
- OH5 handling reference: [kvmani/kikuchiBandAnalyzer](https://github.com/kvmani/kikuchiBandAnalyzer)
