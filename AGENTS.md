# AGENTS.md — EBSD phase identification (NCC + ML)

This document defines how developers and automation agents (Codex, Claude, Gemini, …) work in
this repository; it is the only root agent-instruction file.

Always align changes with `docs/mission_statement.md`. If a change improves speed but reduces scientific reliability, prefer reliability unless explicitly instructed otherwise.

Guideline hierarchy: root `AGENTS.md` is the default policy. A deeper `AGENTS.md` in a subdirectory (for example `docs/AGENTS.md`) may add stricter local rules for that area.

## Cardinal rules (owner-mandated, every repository)

1. **Fetch before you change anything.** Start every task with
   `git fetch origin && git pull --rebase --autostash origin main`. Several people and agents push
   to this repository; never build on a stale checkout.
2. **Work on `main` only.** Commit and push directly to `main` unless the owner names another
   branch for the task. No feature branches or pull requests by default. Fetch again before
   pushing; if the push is rejected, `git pull --rebase origin main`, rerun the affected tests,
   then push. Never force-push or rewrite published history.
3. **Keep the progress ledger current:** [`todo_list.md`](todo_list.md). Before coding, add (or claim)
   the task entry: goal, plan, who is working on it, date. After each meaningful step record what
   was done, the commit, test evidence, decisions, blockers and the next step. Commit the ledger
   with the work it describes. A new session must be able to resume from the ledger and git
   history alone, without the chat that started the work.
4. **Targeted tests for ordinary changes.** Run only the tests that cover what you touched (see
   [Tests](#13-tests)). Run the full suite only when preparing a release or when the owner asks.
5. **CI runs only on request.** GitHub Actions workflows start by `workflow_dispatch` only;
   tag-triggered release workflows are the single exception. Never add push, pull-request or
   schedule triggers, and never start a workflow unless the owner asks.
6. **Be honest.** Never report a check as passed unless you ran it and saw it pass; record skipped
   checks and the reason. Never weaken, skip or delete a test to make it pass.
7. **Keep git clean.** Commit sources and canonical assets only: no secrets, real personal data,
   virtual environments, caches, logs or build output.

The cardinal rules decide *how work is done* (sync, branch, ledger, test scope, CI). The project
rules in this file decide *what this repository must achieve and protect*; nothing generic here
relaxes them. If a project rule and a cardinal rule genuinely conflict, stop and ask the owner.

## 1. Project Mandate

- Primary objective: robust phase discrimination in mixed-phase EBSD patterns using complementary evidence tracks:
  - phase-isolated indexing + NCC scoring against externally simulated patterns,
  - supervised ML classification from experimental Kikuchi patterns.
- Current scope: EBSD-only algorithm development.
- Future scope: sparse LRS integration (architecture must remain integration-ready).
- Priority order: correctness > reproducibility > maintainability > speed.

## 2. Current Operating Mode

- The project began in documentation-first scaffolding (early 2026); the implemented state is in `docs/status.md`.
- Do not start major algorithm code unless mission docs and task plan are reviewed.
- Keep `todo_list.md` current whenever priorities or assumptions change.

## 3. Architecture and Modularity Rules

- Keep code layered and modular; avoid monolithic scripts.
- Suggested module boundaries under `src/`:
  - `io/`: `.oh5` and metadata readers.
  - `indexing/`: candidate orientation ingestion from external TSL runs.
  - `simulation/`: interfaces for externally simulated patterns.
  - `similarity/`: NCC and other scoring methods.
  - `ml/`: dataset preparation, model training, evaluation, and benchmark orchestration.
  - `decision/`: phase/orientation selection and confidence logic.
  - `evaluation/`: manually curated benchmark case evaluation.
  - `workflows/`: orchestration pipelines.
- Keep scripts under `scripts/` thin; they should orchestrate, not contain core logic.

## 4. Data and File Format Policy

- `.oh5` files are treated as source-of-truth EBSD containers from external tools.
- Read-only by default unless a task explicitly requires writing modified HDF5 outputs.
- Use robust field discovery and aliasing because naming can vary (`CI` vs `Confidence Index`, `IQ` vs `Image Quality`).
- Canonical reference for `.oh5` structure in this repo: `docs/oh5_structure.md`.

## 5. Algorithm Policy (Initial)

- Baseline phase decision metric: masked normalized cross-correlation (NCC).
- ML branch: supervised classification of experimental Kikuchi patterns using YAML-configured phase labels and deterministic train/val/test splits.
- Baseline candidate generation: one-phase-at-a-time indexing externally (phase-isolated assumptions).
- For each pixel, compare all candidate phase simulations to the same experimental pattern under consistent preprocessing.
- Always store intermediate evidence for traceability:
  - candidate list, NCC scores, selected winner (NCC track),
  - sample lineage, quality-filter decisions, split assignment, model predictions/metrics (ML track).

## 6. Reproducibility and Logging

- Every runnable workflow must support `--debug`.
- Debug mode must use small in-repo test data and deterministic seeds.
- Log must include: run configuration, input file paths, field mappings, pixel counts, and summary metrics.
- Write machine-readable run metadata (`manifest.json`) for each workflow run.

## 7. Documentation Synchronization (Critical)

Any change affecting behavior, inputs/outputs, assumptions, or CLI flags must update documentation in the same change:

- user-facing usage docs,
- algorithm/method assumptions,
- roadmap/status/todo entries.
- the canonical Sphinx site under `docs/site/`.

Documentation is a foundational repository interface. Users should be able to operate the repository primarily from the documentation.

When changes materially affect workflows or GUIs, prefer version-controlled documentation figures:

- SVG for schematic screenshots and architecture/workflow diagrams,
- Mermaid for flow sheets embedded in docs pages.

When changes materially affect scientific interpretation, notation, conventions, or algorithm meaning, update the relevant mathematical and citation-bearing docs pages in the same change.

## 8. Testing Expectations

- Unit tests for data access, NCC scoring, candidate ranking, and edge cases.
- At least one debug integration test for end-to-end workflow.
- For ML track, include tests for `.oh5` pattern extraction, label mapping, quality filtering, split determinism, and training smoke path.
- Use in-repo small fixtures for deterministic tests.

## 9. Contribution Workflow (Single User, Agent-Assisted)

1. Sync (cardinal rule 1) and define or claim scope in `todo_list.md`.
2. Implement in small, reviewable increments.
3. Run targeted tests and debug workflow.
4. Update docs and status snapshot.
5. Record residual risks and assumptions in the `todo_list.md` task log; commit and push to `main`.

## 10. Non-goals (for now)

- Throughput optimization and distributed scaling.
- Full Raman registration pipeline.
- Manuscript finalization.

These are later roadmap items; current focus is establishing a correct, extensible EBSD core pipeline.

## 11. Path Convention

- In repository documents, JSON/YAML configs, reports, and generated metadata, use repository-relative paths by default.
- Do not store absolute machine-specific paths (for example `<home>/...`) unless explicitly required.
- If a CLI accepts absolute paths, any persisted output should still be normalized to repo-relative form where possible.

## 12. Quality bar

- Typed, layered Python with docstrings on every public module, class and function (purpose,
  parameters with units/shapes, returns, raised errors); scripts stay thin.
- The smallest change that fully solves the task; no dead code; fail clearly on missing `.oh5`
  fields or patterns rather than guessing.
- Every behaviour change ships with tests on the in-repo fixtures in the same commit.

## 13. Tests

```bash
python -m pytest tests/test_<area>.py -q     # targeted (ordinary changes) + the workflow's --debug run
python -m pytest -q                          # full suite (release or on request)
```

| You changed | Start with |
| --- | --- |
| `io/`, `.oh5` readers | `tests/test_ml_oh5_reader.py tests/test_image_io.py` |
| `intake/` | `tests/test_g0_intake_validator.py` |
| `similarity/` (NCC), `preprocessing/` | `tests/test_ncc.py tests/test_curated_ncc.py` |
| Hough features | `tests/test_kikuchipy_hough.py tests/test_curated_hough_vs_ncc.py` |
| `ml/` dataset, quality, split | `tests/test_ml_dataset_builder.py tests/test_ml_dataset_io.py tests/test_ml_quality.py tests/test_ml_split.py` |
| `ml/` training, inference | `tests/test_ml_training_smoke.py tests/test_ml_inference.py tests/test_ml_oh5_sample_inference.py` |
| ML suite, packaging, full cycle | `tests/test_ml_suite.py tests/test_ml_suite_packaging.py tests/test_ml_full_cycle.py` |
| `gui/` explorer, gallery, crop | `tests/test_ml_phase_explorer.py tests/test_ml_diagnostic_gallery.py tests/test_oh5_crop.py` |

## 14. Map

| Need | Read |
| --- | --- |
| Mission and priorities | [docs/mission_statement.md](docs/mission_statement.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| `.oh5` structure | [docs/oh5_structure.md](docs/oh5_structure.md) |
| Scientific strategy, references | [docs/scientific_strategy_band_aware_phase_id.md](docs/scientific_strategy_band_aware_phase_id.md), [docs/references.md](docs/references.md) |
| NCC workflows | [docs/curated_ncc_workflow.md](docs/curated_ncc_workflow.md), [docs/curated_hough_vs_ncc_workflow.md](docs/curated_hough_vs_ncc_workflow.md) |
| ML workflows and GUIs | [docs/ml_classifier_workflow.md](docs/ml_classifier_workflow.md), [docs/ml_training_inference_workflow.md](docs/ml_training_inference_workflow.md), `docs/ml_*_gui.md` |
| User documentation (Sphinx) | [docs/site/](docs/site/) and [docs/AGENTS.md](docs/AGENTS.md) for its rules |
| Roadmap and status | [docs/roadmap.md](docs/roadmap.md), [docs/status.md](docs/status.md) |
| Task list and task log (ledger) | [todo_list.md](todo_list.md) |
