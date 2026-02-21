# Scientific Strategy: Band-Aware Phase Identification

Last updated: 2026-02-21

## Why This Strategy Is Needed

Current curated baseline (`reports/curated_ncc/summary.json`) reaches 33.33% (1/3) with phase collapse to `fe_bcc` for all cases. Two wrong winners are candidates marked:

- `indexing_status=failed`
- `is_fallback_orientation=true`

This indicates two immediate scientific issues:

1. Decision logic is currently too permissive for failed/fallback candidates.
2. Pure intensity NCC is not sufficiently phase-discriminative for noisy EBSD-vs-simulation comparisons.

## Clarification on Masking

- Scoring path already uses mask-based preprocessing and masked NCC (`src/phase_id_xcorr/preprocessing/pattern_prep.py`, `src/phase_id_xcorr/similarity/ncc.py`).
- The HTML inspection initially displayed raw images for readability; this does not mean scoring was unmasked.
- Diagnostic check (masked vs full-frame NCC) shows full-frame NCC increases all scores and reduces separability, so lack of display masking is not the root cause.

## Scientific Objective (Near Term)

Build a robust phase-decision algorithm that prioritizes **band structure agreement** over raw intensity agreement, while handling:

- noisy experimental patterns,
- missing/weak bands,
- simulation-to-experiment contrast mismatch,
- candidate orientations from failed indexing.

## Core Design Principles

1. Correctness-first: prioritize phase discrimination quality over runtime.
2. Evidence-preserving: keep per-candidate intermediate scores and diagnostics.
3. Modular metrics: multiple similarity metrics evaluated side-by-side.
4. Explicit uncertainty: allow `uncertain` instead of forced incorrect labels.
5. Strict reproducibility: deterministic seeds and run manifests for every workflow.

## Proposed Algorithm Stack

For each experimental pattern and each candidate simulated pattern:

1. **Common geometric support**
   - Use shared mask and shape checks (already implemented).

2. **Band-sensitive preprocessing branch**
   - Background suppression (large-kernel subtraction or equivalent high-pass path).
   - Robust normalization (percentile-based option).
   - Optional denoise branch (non-local or neighborhood-based) with controlled parameters.

3. **Multi-metric similarity (not intensity-only)**
   - `S_int`: masked intensity NCC (baseline).
   - `S_grad`: NCC on gradient magnitude images.
   - `S_ori`: weighted gradient-orientation agreement (cosine of orientation difference).
   - `S_edge`: edge/band overlap score (binary band map overlap or chamfer-style distance score).

4. **Decision policy with reliability constraints**
   - Penalize failed/fallback candidates.
   - Select best phase by combined score:
     - `S_total = w1*S_int + w2*S_grad + w3*S_ori + w4*S_edge - P_failed`
   - If margin is below threshold, emit `uncertain`.

5. **Confidence estimation**
   - Margin-based confidence.
   - Stability score under perturbations (already partially available via `flip_rate`).
   - Report confidence only when quality and margin pass thresholds.

## Phase Gates (Execution Plan)

### Gate B1: Decision Hygiene
Goal: Stop obviously unreliable winners.

- Add failed/fallback penalty and an uncertainty gate.
- Add side-by-side report comparing old vs new decisions.

Exit criteria:

- No case is won by failed/fallback candidate unless it exceeds best non-failed candidate by strict override threshold.

### Gate B2: Band-Aware Metrics
Goal: Improve discrimination with structural features.

- Implement `S_grad`, `S_ori`, `S_edge`.
- Produce per-record metric table and rank deltas.

Exit criteria:

- Combined metric outperforms intensity-only baseline on curated set.
- At least one ablation indicates additive value beyond intensity NCC.

### Gate B3: Robustness and Calibration
Goal: Make scores trustworthy under noise/variance.

- Stress tests: noise, gain/offset, mild blur, partial occlusion.
- Tune thresholds (`margin`, failed override, uncertainty gate).

Exit criteria:

- Ranking stability improves.
- False confident errors are reduced.

### Gate B4: `.oh5` Integration
Goal: Apply decision logic to scan-scale candidate ingestion.

- Connect candidate extraction from phase-isolated `.oh5`.
- Run on selected manually verified pixels.

Exit criteria:

- End-to-end reproducible run with traceable evidence per pixel.

## Evaluation and Reporting Requirements

Each run must produce:

- `scores.csv` with all metric components per candidate.
- `decisions.csv` with winner, margin, penalties, confidence, uncertainty flag.
- confusion summary and per-phase performance.
- error-case list with links to case visuals.
- single inspection HTML artifact for bird’s-eye review.

## Current Scientific Hypotheses to Test

1. Penalizing failed/fallback candidates will remove major misclassifications.
2. Gradient/band-aware metrics will reduce sensitivity to intensity mismatch.
3. Some cases are inherently ambiguous with current candidate quality and must be marked `uncertain`.
4. Additional curated orientations per phase are needed before freezing thresholds.
5. Hough-space similarity may improve band-location robustness but must pass robustness-gated comparison before adoption.

## Companion Plan for Hough Branch

- Detailed Hough-space plan and gate criteria: `docs/hough_space_ncc_action_plan.md`.

## Minimum Data Expansion Needed

Before threshold freeze:

- At least 2 orientations per phase (minimum immediate target),
- Preferably 5+ curated cases per phase for robust calibration.

## Reference Anchors (Primary)

- Dictionary indexing concept for EBSD patterns: [Chen et al., Integrating spherical indexing and dictionary indexing, J. Appl. Cryst. 48 (2015)](https://journals.iucr.org/paper?ff5093)
- Phase differentiation by dictionary indexing in difficult cases: [Lenthe et al., Acta Materialia 144 (2018)](https://www.sciencedirect.com/science/article/abs/pii/S1359645417309369)
- PPHT/Radon for EBSD band finding and overlap handling: [A. T. Tassen et al., Pattern indexing by PPHT, J. Microscopy 286 (2022)](https://pubmed.ncbi.nlm.nih.gov/35638570/)
- Open-source EBSD tooling and preprocessing context: [KikuchiPy (JOSS, 2024)](https://joss.theoj.org/papers/10.21105/joss.06724)
- Pattern preprocessing guidance (static/dynamic background): [KikuchiPy pattern processing tutorial](https://kikuchipy.org/en/stable/tutorials/pattern_processing.html)
- Open-source high-speed indexing and denoising references: [PyEBSDIndex README](https://github.com/USNavalResearchLaboratory/PyEBSDIndex)
- Non-local EBSD denoising for indexing robustness: [Brewick et al., Ultramicroscopy 200 (2019)](https://pubmed.ncbi.nlm.nih.gov/30825718/)
- Mixture/overlap-aware indexing direction: [Singh et al., Spherical indexing for multiphase overlap patterns (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364039/)
- Residual-based multiphase diagnostics direction (preprint): [Geisenhof et al. (2026)](https://arxiv.org/abs/2601.17727)
