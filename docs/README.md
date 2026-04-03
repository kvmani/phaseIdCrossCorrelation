# Documentation Guide

Use this file as the bridge between the canonical Sphinx site and the legacy Markdown source material.

## Canonical Surface

The canonical user-facing documentation now lives in:

- source: `docs/site/`
- built HTML: `docs/_build/html/index.html`

Build it with:

```powershell
python -m pip install -r .\docs\requirements.txt
python .\scripts\build_docs.py --clean
```

## Start in Order

1. `docs/site/index.md`
2. `docs/site/getting_started/index.md`
3. `docs/site/workflows/index.md`
4. `docs/site/guis/index.md`
5. `docs/site/reference/index.md`

## By Task

### Understand the repo

- `README.md`: high-level project summary.
- `AGENTS.md`: repo rules and contribution constraints.
- `docs/roadmap.md`: longer-horizon plan.
- `docs/site/mission/index.md`: mission, principles, and scientific posture.
- `docs/site/concepts/index.md`: concepts, formulations, and rationale.

### Run NCC and Hough workflows

- `docs/g0_data_intake_validation.md`: validate incoming packet structure.
- `docs/curated_ncc_workflow.md`: run the masked NCC baseline.
- `docs/curated_hough_vs_ncc_workflow.md`: compare image-space NCC to Hough-space NCC.
- `docs/mcc_vs_hough_full_cycle_runbook.md`: one-pass runbook for the full curated cycle.

### Run ML workflows

- `docs/site/workflows/dataset_preparation.md`
- `docs/site/workflows/benchmark_and_full_cycle.md`
- `docs/site/workflows/inference_workflows.md`
- `docs/site/guis/inference_gui.md`
- `docs/site/guis/phase_explorer_gui.md`
- `docs/site/guis/diagnostic_gallery_gui.md`

### Understand data contracts

- `docs/site/data_contracts/oh5_contracts.md`
- `docs/site/data_contracts/manifests_and_outputs.md`
- `docs/test_data_setup_plan.md`: in-repo fixture and packet setup expectations.

### Understand scientific planning

- `docs/action_plan_post_data_intake.md`: gated NCC implementation plan.
- `docs/hough_space_ncc_action_plan.md`: Hough branch plan.
- `docs/scientific_strategy_band_aware_phase_id.md`: recovery strategy for weak NCC separation.
- `docs/references.md`: external references.

## Fast Lookup

- Want the current default workflow surface: `scripts/README.md`
- Want module ownership: `src/README.md`
- Want test coverage map: `tests/README.md`
- Want generated outputs: `reports/README.md`
- Want old Markdown source docs directly: browse `docs/`
