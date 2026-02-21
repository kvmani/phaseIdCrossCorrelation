# Hough-Space NCC Branch: Critical Analysis and Action Plan

Last updated: 2026-02-21

## Why This Document

You proposed adding a Hough-based similarity branch to reduce sensitivity to raw intensity mismatch between experimental and simulated EBSD patterns. This plan defines how to add that branch alongside current masked image-space NCC and compare both methods rigorously.

## Quick Feasibility Check (Current Curated Set)

Dataset used: `data/test/student_data_packet_phaseid` (3 curated cases).

### Observed baseline

- Current masked image-space NCC accuracy: `1/3` (`33.33%`).

### Rapid prototype observations (not yet production code)

1. Hough accumulator NCC (`NCC(H_exp, H_sim)`) with fixed transform settings:
   - Accuracy stayed `1/3`.
   - Scores were all very high and close (`~0.97`), indicating poor separability.

2. Binarized/peak Hough comparison (`NCC(P_exp, P_sim)` with peak maps):
   - Default trial improved to `2/3`.
   - Some parameter sets reached `3/3` on this tiny set.
   - But parameter sensitivity is high; one tuned setting flipped under a mild gain perturbation in one case.

Conclusion: your idea is promising, but must be implemented with strict anti-overfitting controls and robustness checks before we trust it.

## Critical Analysis of the Idea

## Strengths

- Better physics alignment: compares band geometry/location rather than raw brightness.
- Less sensitive to global intensity scaling and background mismatch.
- In Hough space, small image translations/rotations can be handled as shifts in `(rho, theta)` domain.
- Peak-binarized option directly reduces dependence on band intensity quality.

## Risks and Limitations

- Standard line Hough can be sensitive to edge extraction parameters (Canny thresholds, smoothing).
- Magnetite vs wustite may still produce similar dominant bands; geometry alone may remain ambiguous in some cases.
- Peak-only representation can discard useful secondary information (band width/strength context).
- With only 3 curated cases, parameter tuning can overfit very easily.
- Hough accumulators can become uniformly similar if transform settings are too coarse or edge maps are too dense.

## Scientific Position

- Add Hough-space scoring as a **parallel branch**, not an immediate replacement.
- Compare against current NCC with shared preprocessing and identical candidate sets.
- Use uncertainty gating and fallback penalties in both methods.

## Proposed Dual-Branch Method

For each `(experimental, simulated candidate)` pair, compute:

1. `S_img`: masked image-space NCC (existing baseline).
2. `S_hacc`: NCC of normalized Hough accumulators.
3. `S_hpeak`: NCC of binarized (or softened-binary) Hough peak maps.

Implementation rule:

- Use KikuchiPy's PyEBSDIndex-backed Hough/Radon path for transform generation; do not introduce an independent custom Hough implementation for the baseline branch.

Optional invariance step:

- `S_hacc_shifted`: best NCC after constrained shift search in Hough space:
  - circular shift in `theta`,
  - local shift in `rho`.

Decision outputs:

- `pred_img` (image-space only),
- `pred_hough` (Hough-only),
- `pred_hybrid` (weighted fusion with reliability penalties),
- `uncertain` flag if margins are below threshold.

## Phase-Gated Execution Plan

## H0: Freeze Comparison Protocol

Tasks:

- Keep current curated dataset fixed.
- Freeze one baseline run for direct before/after comparison.
- Define acceptance metrics before coding (accuracy, margin, flip-rate, uncertain-rate).

Gate H0 exit:

- Baseline artifacts and metric definitions are fixed.

## H1: Hough Feature Extraction Module

Tasks:

- Add `src/phase_id_xcorr/features/hough_features.py`:
  - masked edge extraction,
  - fixed-theta Hough transform,
  - peak-map generation (binary/soft).
- Keep all params configurable and logged.

Gate H1 exit:

- Deterministic outputs for same input and settings.
- Unit tests for output shape, value range, determinism.

## H2: Hough Similarity Metrics

Tasks:

- Add `src/phase_id_xcorr/similarity/hough_ncc.py`:
  - accumulator NCC,
  - peak-map NCC,
  - optional shifted Hough NCC.
- Add fallback-safe behavior for sparse/no-peak cases.

Gate H2 exit:

- Unit tests for known synthetic cases (identical, shifted, low-signal).

## H3: Curated Dual-Scoring Runner

Tasks:

- Extend curated workflow to produce both image and Hough scores per candidate.
- Write artifacts:
  - `reports/curated_dual_similarity/scores.csv`
  - `reports/curated_dual_similarity/decisions_img.csv`
  - `reports/curated_dual_similarity/decisions_hough.csv`
  - `reports/curated_dual_similarity/decisions_hybrid.csv`
  - `reports/curated_dual_similarity/summary.json`

Gate H3 exit:

- One command reproduces all three decision tracks (`img`, `hough`, `hybrid`).

## H4: Robustness and Overfitting Control

Tasks:

- Parameter sensitivity sweep with strict limits (small grid, pre-declared).
- Perturbation tests (noise/gain/blur/partial visibility).
- Select settings by robustness criteria, not only best raw accuracy.

Gate H4 exit:

- Chosen settings have stable winners and acceptable flip-rate under perturbations.

## H5: Scientific Comparison and Go/No-Go

Tasks:

- Produce direct comparison report:
  - where Hough wins over image NCC,
  - where it fails,
  - ambiguity cases.
- Decide:
  - keep as optional branch,
  - use hybrid default,
  - or reject Hough branch for now.

Gate H5 exit:

- Explicit method decision documented with evidence.

## Required Artifacts for Inspection

- Single HTML bird's-eye report containing, per record:
  - masked experimental image,
  - masked simulated candidates,
  - edge maps,
  - Hough accumulators,
  - peak maps,
  - `S_img`, `S_hacc`, `S_hpeak`, hybrid score table,
  - winner and margin under each method.

## Practical Guardrails

- Do not increase dataset size yet (as requested) until method behavior is clear on current curated set.
- Do not accept a model that improves only by exploiting failed/fallback candidates.
- Keep `uncertain` as a valid output to avoid false confidence.

## Immediate Next Implementation Step

Implement H1 + H2 first, then run H3 comparison with fixed defaults before any parameter sweep.
