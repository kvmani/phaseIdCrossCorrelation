# Proposed Architecture (Implementation Blueprint)

This is the initial module blueprint for Phase 1 implementation (EBSD-only baseline).

## Module Map

```text
src/
  phase_id_xcorr/
    io/
      oh5_reader.py
      field_aliases.py
    indexing/
      tsl_candidates.py
    simulation/
      external_patterns.py
    preprocessing/
      pattern_prep.py
      masking.py
    similarity/
      ncc.py
    decision/
      selector.py
      confidence.py
    evaluation/
      benchmark_cases.py
      metrics.py
    workflows/
      run_case.py
      run_batch.py
    reporting/
      manifest.py
      tables.py
```

## Data Contracts

- Candidate orientation record:
  - `pixel_x`, `pixel_y`, `phase_name`, `euler`, `source_file`, `quality_fields`.
- Simulated pattern record:
  - `pixel_x`, `pixel_y`, `phase_name`, `orientation_id`, `pattern_path`.
- Decision result record:
  - `pixel_x`, `pixel_y`, `winner_phase`, `winner_ncc`, `runner_up_ncc`, `margin`, `all_scores`.

## Baseline Workflow

1. Read `.oh5`-derived candidate orientations for each assumed phase.
2. Load experimental pattern and corresponding simulated candidate patterns.
3. Apply consistent preprocessing/mask.
4. Compute NCC per candidate.
5. Select winner and confidence margin.
6. Persist evidence and summaries.

## Design Constraints

- CPU-first implementation.
- Deterministic outputs in debug mode.
- Pluggable scoring functions for future ablations.
- No hidden hard-coded paths; use configs.
