# Hough Feasibility Probe (Curated Set)

Date: 2026-02-21
Dataset: `data/test/student_data_packet_phaseid` (3 curated records)

## Goal
Quickly test whether Hough-space comparisons can improve phase discrimination relative to current masked image-space NCC.

## Methods Tested (prototype)

1. `image_ncc`
- Current masked image-space NCC baseline.

2. `hacc_ncc`
- Canny edges -> line Hough accumulator -> NCC in accumulator space.

3. `hpeak_ncc`
- Hough peaks converted to sparse peak map (softened by Gaussian) -> NCC in peak-map space.

## Immediate Observations

### Baseline run (single default prototype setting)

- `image_ncc`: 1/3 correct
- `hacc_ncc`: 1/3 correct (scores saturated and close; poor separability)
- `hpeak_ncc`: 2/3 correct

### Parameter sweep (small exploratory grid)

- Some `hpeak_ncc` settings reached 3/3 on this tiny set.
- This is not enough evidence of robustness due to strong overfitting risk with only 3 cases.

### Perturbation check on one high-performing `hpeak_ncc` setting

- Base curated set: 3/3
- Under mild gain perturbation, one case flipped winner (flip-rate observed for that record: 0.25 over tested perturbations).

## Scientific Takeaway

- Hough accumulator NCC alone is not sufficient in current form.
- Peak-based Hough similarity is promising and aligns with band-location matching goals.
- However, stability and overfitting are major concerns; this branch must be evaluated with strict robustness gates before adoption.

## Next Step

Proceed with `docs/hough_space_ncc_action_plan.md` (H0->H5), and keep image NCC + Hough branch in parallel until a robustness-backed decision is made.
