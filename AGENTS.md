# AGENTS.md

This document defines how developers and automation agents should work in this repository.

Always align changes with `docs/mission_statement.md`. If a change improves speed but reduces scientific reliability, prefer reliability unless explicitly instructed otherwise.

Guideline hierarchy: root `AGENTS.md` is the default policy. A deeper `agents.md`/`AGENTS.md` in a subdirectory may add stricter local rules for that area.

## 1. Project Mandate

- Primary objective: robust phase discrimination in mixed-phase EBSD patterns using phase-isolated indexing candidates and NCC scoring against externally simulated patterns.
- Current scope: EBSD-only algorithm development.
- Future scope: sparse LRS integration (architecture must remain integration-ready).
- Priority order: correctness > reproducibility > maintainability > speed.

## 2. Current Operating Mode

- The project is currently in documentation-first scaffolding.
- Do not start major algorithm code unless mission docs and task plan are reviewed.
- Keep `todo_list.md` current whenever priorities or assumptions change.

## 3. Architecture and Modularity Rules

- Keep code layered and modular; avoid monolithic scripts.
- Suggested module boundaries under `src/`:
  - `io/`: `.oh5` and metadata readers.
  - `indexing/`: candidate orientation ingestion from external TSL runs.
  - `simulation/`: interfaces for externally simulated patterns.
  - `similarity/`: NCC and other scoring methods.
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
- Baseline candidate generation: one-phase-at-a-time indexing externally (phase-isolated assumptions).
- For each pixel, compare all candidate phase simulations to the same experimental pattern under consistent preprocessing.
- Always store intermediate evidence for traceability (candidate list, NCC scores, selected winner).

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

## 8. Testing Expectations

- Unit tests for data access, NCC scoring, candidate ranking, and edge cases.
- At least one debug integration test for end-to-end workflow.
- Use in-repo small fixtures for deterministic tests.

## 9. Contribution Workflow (Single User, Agent-Assisted)

1. Define scope in `todo_list.md`.
2. Implement in small, reviewable increments.
3. Run targeted tests and debug workflow.
4. Update docs and status snapshot.
5. Record residual risks and assumptions.

## 10. Non-goals (for now)

- Throughput optimization and distributed scaling.
- Full Raman registration pipeline.
- Manuscript finalization.

These are later roadmap items; current focus is establishing a correct, extensible EBSD core pipeline.

## 11. Path Convention

- In repository documents, JSON/YAML configs, reports, and generated metadata, use repository-relative paths by default.
- Do not store absolute machine-specific paths (for example `<home>/...`) unless explicitly required.
- If a CLI accepts absolute paths, any persisted output should still be normalized to repo-relative form where possible.
