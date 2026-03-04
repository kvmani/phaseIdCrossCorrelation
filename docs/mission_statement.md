# Mission Statement: Phase Identification by Cross-Correlation and ML Classification in Multi-Phase EBSD

## Vision

Develop a scientifically rigorous, modular, and reproducible EBSD analysis framework that improves phase discrimination in mixed microstructures where standard indexing is unreliable.

## Problem Context

In the current target system, three phases coexist with fine spatial intermixing:

- magnetite (`Fe3O4`),
- wustite (`FeO`),
- iron (`Fe`).

Conventional indexing can misclassify phases, especially magnetite vs wustite, due to similarity in Kikuchi geometry and interplanar-angle relationships.

## Core Hypotheses

1. NCC evidence path: if we compare phase-conditioned simulated patterns against experimental patterns under consistent preprocessing, NCC-based evidence remains an interpretable baseline for phase decisions.
2. ML classification path: if we train a supervised classifier on labeled experimental Kikuchi patterns from `.oh5` scans, the model can learn discriminative phase cues that are difficult to encode with correlation-only metrics.
3. Hybrid readiness: storing full per-sample evidence (quality filters, labels, scores, predictions) enables future fusion of NCC and ML decision tracks.

## Baseline Algorithm (Track A: NCC)

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

## New Algorithm Branch (Track B: ML Classifier)

For supervised phase classification from experimental patterns:

1. Ingest one or more `.oh5` scan files with config-defined labels:
   - per-pixel CSV labels, or
   - single-phase scan mapping (one `.oh5` file corresponds to one phase).
2. Extract Kikuchi patterns and scan quality fields from `.oh5` using robust field aliasing (`CI` vs `Confidence Index`, `IQ` vs `Image Quality`).
3. Filter low-quality patterns using configurable thresholds (for example CI/IQ/Fit/Valid gates).
4. Map phase names to class labels from YAML configuration (no hard-coded material names).
5. Build a combined dataset across all configured sources.
6. Create deterministic train/val/test splits from YAML policy.
7. Train and evaluate configurable classifier backbones with optional pretrained initialization.
8. Persist run metadata and metrics for traceability (`manifest.json`, report artifacts, split/evidence tables).

## Scope and Boundaries

### In Scope Now

- EBSD-only phase discrimination pipeline with two tracks:
  - NCC evidence track (existing baseline).
  - ML classifier track (new branch: data prep + training + evaluation).
- `.oh5` ingestion and quality-aware pattern extraction.
- YAML-configurable phase labels and split policies.
- Reproducible debug workflows with in-repo test data.
- Documentation and architecture foundation for future growth.

### Deferred to Later Phases

- Sparse LRS registration and multimodal fusion.
- Throughput and large-scale distributed training optimization.
- Manuscript final drafting and journal-specific formatting.

## Scientific and Engineering Principles

- Correctness and interpretability before speed.
- Reproducibility before convenience.
- Modular design with explicit interfaces and replaceable components.
- Deterministic debug runs with machine-readable manifests.
- Architecture must remain fusion-ready for future NCC + ML + sparse LRS decisions.

## Success Criteria (Current)

Primary success criterion:

- Improved phase identification correctness on manually verified benchmark cases versus correlation-only baseline.

Supporting criteria:

- Stable reproducible runs on debug data.
- Clear confidence reporting and per-class performance summaries.
- Traceable lineage from raw `.oh5` + label-source configs to trained model artifacts.
- Clean, maintainable documentation synchronized with implementation.

## Deliverables for Current Stage

- Mission/governance/task documents synchronized with ML expansion.
- Dedicated modular ML package scaffold under `src/phase_id_xcorr/ml`.
- Config-driven dataset preparation workflow from `.oh5` sources with dual label modes (CSV labels and single-phase scan map).
- Config-driven classifier training workflow with pretrained/scratch options.
- Run-level reporting artifacts and deterministic debug tests.
