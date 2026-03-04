# phaseIdCrossCorrelation

Accuracy-first EBSD phase identification for multi-phase scans using two complementary tracks:

- phase-isolated indexing + normalized cross-correlation (NCC),
- supervised ML classification on experimental Kikuchi patterns.

## Mission

This project addresses a practical failure mode in conventional EBSD indexing for mixed microstructures containing similar phases (currently: magnetite, wustite, and iron). In these cases, standard Hough/dictionary indexing can return unstable or incorrect phase labels, especially for magnetite vs wustite.

The repository now maintains two evidence tracks:

1. NCC track: phase-isolated candidate generation and masked NCC scoring against simulated patterns.
2. ML track: quality-filtered `.oh5` pattern extraction plus supervised classifier training from config-defined labels (per-pixel CSV or single-phase scan mapping).

Both tracks are designed to remain traceable, reproducible, and fusion-ready.

## Current Project Phase

This repository is currently in **baseline NCC/hough implementation + ML classifier scaffold implementation** mode.

- Curated masked-NCC baseline is implemented and runnable.
- KikuchiPy Hough-space comparison workflow is implemented and runnable.
- ML classifier branch is now in active implementation:
  - modular package scaffold,
  - config-driven dataset prep (`.oh5` + CSV or single-phase scan-map),
  - config-driven training/evaluation workflow.

## Documentation Map

- `docs/mission_statement.md`: scientific objective, assumptions, and success criteria.
- `AGENTS.md`: canonical engineering and agent workflow rules.
- `docs/roadmap.md`: phased delivery plan.
- `docs/status.md`: read-only snapshot of current state.
- `docs/action_plan_post_data_intake.md`: phase-gated implementation playbook after data intake.
- `docs/g0_data_intake_validation.md`: how to run G0 data-intake gate checks.
- `docs/curated_ncc_workflow.md`: curated experimental-vs-simulated NCC workflow and artifacts.
- `docs/curated_hough_vs_ncc_workflow.md`: curated image-NCC vs KikuchiPy Hough-NCC comparison workflow.
- `docs/mcc_vs_hough_full_cycle_runbook.md`: one-go command sequence to run G0 + curated NCC + Hough comparison and print headline metrics.
- `docs/oh5_structure.md`: dedicated guide for TSL `.oh5` layout and data access.
- `docs/ml_classifier_workflow.md`: ML classifier architecture and end-to-end workflow.
- `docs/ml_input_data_runbook.md`: detailed ML input-data contract (CSV labels + single-phase scan-map mode), sanity checks, logging events, manifests, and platform run commands.
- `docs/ml_training_inference_workflow.md`: practical run sequence for training/evaluation ("inference"), benchmarking, and auto-generating lab-meeting PPTX summaries.
- `docs/ml_model_selection.md`: candidate model shortlist (>=5), pretrained options, and selection rationale.
- `docs/test_data_setup_plan.md`: canonical test data acquisition, naming, and manifest specification.
- `todo_list.md`: operational task list (living document).

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

- Correctness and scientific defensibility over throughput.
- Reproducibility over convenience.
- Modular architecture with explicit interfaces.
- CPU-first deterministic debug workflows.
- Documentation-first changes with synchronized task tracking.

## External Dependencies and Interfaces

- TSL indexing and EMSoft/TSL simulation are external inputs to this pipeline.
- TSL analysis outputs are expected as `.oh5` (HDF5) files.
- Ground-truth labels for ML workflows are config-defined (per-pixel CSV or single-phase file mapping).
- See `docs/oh5_structure.md` for canonical field paths and access patterns.

## Reference Repositories

- DeepImageDeconvolution (format inspiration): [kvmani/DeepImageDeconvolution](https://github.com/kvmani/DeepImageDeconvolution)
- OH5 handling reference: [kvmani/kikuchiBandAnalyzer](https://github.com/kvmani/kikuchiBandAnalyzer)
