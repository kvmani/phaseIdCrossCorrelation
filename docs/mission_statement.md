# Mission Statement: Phase Identification by Cross-Correlation in Multi-Phase EBSD

## Vision

Develop a scientifically rigorous, modular, and reproducible EBSD analysis framework that improves phase discrimination in mixed microstructures where standard Hough/dictionary indexing is unreliable.

## Problem Context

In the current target system, three phases coexist with fine spatial intermixing:

- magnetite (`Fe3O4`),
- wustite (`FeO`),
- iron (`Fe`).

Conventional indexing can misclassify phases, especially magnetite vs wustite, due to similarity in Kikuchi geometry and interplanar-angle relationships.

## Core Hypothesis

At each scan pixel, if we obtain phase-conditional orientation candidates and compare externally simulated patterns against the experimental pattern using a consistent masked NCC metric, the best-matching candidate should provide a more reliable phase decision than direct single-pass indexing alone.

## Baseline Algorithm (Phase 1)

For each EBSD pixel `(x, y)`:

1. Perform external TSL indexing with phase-isolated assumptions (run 1: only phase A, run 2: only phase B, run 3: only phase C).
2. Extract orientation candidates `O1`, `O2`, `O3` from corresponding `.oh5` outputs.
3. Generate simulated patterns externally for each candidate orientation under consistent geometry.
4. Compute masked normalized cross-correlation:
   - `NCC(exp, sim_phase_A)`
   - `NCC(exp, sim_phase_B)`
   - `NCC(exp, sim_phase_C)`
5. Select highest NCC as predicted phase/orientation.
6. Record confidence and runner-up margin for interpretability.

## Scope and Boundaries

### In Scope Now

- EBSD-only baseline pipeline.
- `.oh5` ingestion and candidate extraction.
- NCC-based decision framework.
- Reproducible debug workflows with in-repo test data.
- Documentation and architecture foundation for future growth.

### Deferred to Later Phases

- LRS registration and fusion (Raman available only sparsely and for Raman-active phases).
- Throughput and large-scale performance tuning.
- Manuscript final drafting and journal-specific formatting.

## Scientific and Engineering Principles

- Accuracy and interpretability before speed.
- Modular design with clear interfaces and replaceable components.
- Deterministic runs for reproducibility.
- Explicit assumptions and traceable evidence for each decision.
- Architecture must be extensible for sparse multimodal fusion (LRS) later.

## Success Criteria (Current)

Primary success criterion:

- Correct phase identification with clear discrimination for manually verified benchmark cases.

Supporting criteria:

- Stable reproducible results on CPU.
- Clear confidence reporting per pixel/case.
- Clean, maintainable documentation and task tracking.

## Deliverables for Current Stage

- Mission/governance/task documents finalized.
- Stable project scaffold and workflow conventions.
- `.oh5` access reference to avoid repeated rediscovery effort.
- Ready-to-implement architecture plan for coding phase.
