# Documentation Guide

Use this file as the entry point to the documentation set.

## Start in Order

1. `docs/mission_statement.md`: scientific objective, scope, and success criteria.
2. `docs/status.md`: current implementation snapshot, risks, and next steps.
3. `todo_list.md`: active work queue.
4. `docs/architecture.md`: code layout and workflow boundaries.

## By Task

### Understand the repo

- `README.md`: high-level project summary.
- `AGENTS.md`: repo rules and contribution constraints.
- `docs/roadmap.md`: longer-horizon plan.

### Run NCC and Hough workflows

- `docs/g0_data_intake_validation.md`: validate incoming packet structure.
- `docs/curated_ncc_workflow.md`: run the masked NCC baseline.
- `docs/curated_hough_vs_ncc_workflow.md`: compare image-space NCC to Hough-space NCC.
- `docs/mcc_vs_hough_full_cycle_runbook.md`: one-pass runbook for the full curated cycle.

### Run ML workflows

- `docs/ml_classifier_workflow.md`: ML pipeline overview and artifact contract.
- `docs/ml_input_data_runbook.md`: dataset config modes, source requirements, and environment-specific commands.
- `docs/ml_training_inference_workflow.md`: recommended experiment sequence and reporting flow.
- `docs/ml_phase_explorer_gui.md`: raw `.oh5` exploration GUI.
- `docs/ml_model_selection.md`: backbone shortlist and selection rationale.

### Understand data contracts

- `docs/oh5_structure.md`: canonical `.oh5` layout and aliasing notes.
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
