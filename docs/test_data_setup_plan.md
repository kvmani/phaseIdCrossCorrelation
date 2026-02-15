# Test Data Acquisition and Setup Plan

This document defines the required test data package for Phase 1 implementation and validation of EBSD phase identification by masked NCC.

## 1. Objectives

- Provide a small, deterministic in-repo dataset that supports rapid development and debug runs.
- Support both unit-level validation (single pattern comparisons) and workflow-level validation (pixel-level candidates from `.oh5`).
- Preserve enough metadata so future agents do not need to infer assumptions from file names alone.

## 2. Scope of Test Data

Two complementary datasets are required.

### Dataset A: Curated Single-Pattern Cases

Purpose:

- Validate preprocessing, NCC computation, and phase discrimination on manually verified patterns.

Expected volume:

- 6 to 10 experimental patterns total.
- Each phase (`fe_bcc`, `fe3o4_magnetite`, `feo_wustite`) should have at least 2 orientations.
- For each experimental pattern, include 3 simulated candidate patterns (one per assumed phase).

### Dataset B: Phase-Isolated Scan Outputs (`.oh5`)

Purpose:

- Validate per-pixel candidate extraction and downstream cross-correlation decision logic on real scan structure.

Required files:

- One experimental scan represented by 3 phase-isolated indexing outputs:
  - `scan_<scan_id>__assume_fe_bcc.oh5`
  - `scan_<scan_id>__assume_fe3o4_magnetite.oh5`
  - `scan_<scan_id>__assume_feo_wustite.oh5`

## 3. Critical Gaps in Current Proposal and Required Fixes

Your proposal is strong, but these additions are necessary to avoid future ambiguity.

1. File-name-only metadata is insufficient.
Use a manifest CSV/JSON as source of truth; filenames remain human-friendly only.

2. `(0,0,0)` as fallback orientation is ambiguous.
A true orientation can be near zero. Keep `(0,0,0)` if needed for compatibility, but add explicit `indexing_status` and `is_fallback_orientation` flags.

3. Missing coordinate linkage for scan-scale evaluation.
For `.oh5` workflows, store the pixel keys (`x`, `y`, and flat index convention) explicitly in manifests.

4. Missing preprocessing policy lock.
NCC is highly sensitive to normalization, masking, and bit depth. Record exact preprocessing parameters in metadata.

5. Missing simulation provenance.
For each simulated pattern, capture simulation source settings (external tool, detector geometry snapshot, and generation timestamp/hash where available).

6. Missing expected-label definitions.
For curated cases, include `expected_phase` based on manual judgment to compute correctness.

7. Missing orientation convention lock.
Euler angles are unusable without a declared convention and units. Record convention explicitly (for example Bunge ZXZ, degrees).

8. Missing exp-sim compatibility checks.
NCC comparisons require same detector-space framing (shape, cropping, mask frame, and effective geometry assumptions). Capture these compatibility constraints in metadata.

## 4. Canonical Directory Layout

```text
data/test/
  README.md
  manifests/
    curated_cases.csv
    curated_pairs.csv
    scan_bundle_manifest.csv
    preprocessing_policy.yaml
  curated_patterns/
    exp/
      <phase_slug>_Ori_<id>.<ext>
    sim/
      assume_fe_bcc/
        <source_phase_slug>_Ori_<id>.<ext>
      assume_fe3o4_magnetite/
        <source_phase_slug>_Ori_<id>.<ext>
      assume_feo_wustite/
        <source_phase_slug>_Ori_<id>.<ext>
  scans/
    scan_<scan_id>/
      scan_<scan_id>__assume_fe_bcc.oh5
      scan_<scan_id>__assume_fe3o4_magnetite.oh5
      scan_<scan_id>__assume_feo_wustite.oh5
```

## 5. Naming Conventions

Use lowercase phase slugs only:

- `fe_bcc`
- `fe3o4_magnetite`
- `feo_wustite`

Case ID convention:

- `c001`, `c002`, ... for curated single-pattern cases.

Scan ID convention:

- `s001`, `s002`, ... for scan bundles.

Curated pattern naming:

- Experimental: `{phase}_Ori_{id}.{ext}` (example `fe3o4_magnetite_Ori_2.png`).
- Simulated: same base filename as experimental, but stored under phase-assumption folder:
  - `curated_patterns/sim/assume_fe_bcc/{phase}_Ori_{id}.{ext}`
  - `curated_patterns/sim/assume_fe3o4_magnetite/{phase}_Ori_{id}.{ext}`
  - `curated_patterns/sim/assume_feo_wustite/{phase}_Ori_{id}.{ext}`

Filename is for convenience only. Orientation truth and candidate metadata must come from manifests.

Extension policy:

- Preferred: `.png` 16-bit or `.tif`.
- `.bmp` allowed only if unavoidable for source compatibility.
- All mixed-format inputs must include normalized dtype metadata in manifest.

## 6. Required Manifest Files

Preferred format for student editing is JSON (self-documented examples in `data/test/manifests/*.example.json`). CSV/YAML templates remain for compatibility.

## 6.1 `manifests/curated_cases.csv`

One row per experimental curated pattern.

Required columns:

- `case_id`
- `exp_path`
- `expected_phase`
- `phi1_deg`
- `PHI_deg`
- `phi2_deg`
- `manual_label_source`
- `bit_depth`
- `height`
- `width`
- `notes`

## 6.2 `manifests/curated_pairs.csv`

One row per experimental-simulated comparison candidate.

Required columns:

- `case_id`
- `exp_path`
- `sim_path`
- `assumed_phase`
- `candidate_phi1_deg`
- `candidate_PHI_deg`
- `candidate_phi2_deg`
- `indexing_status` (`ok` or `failed`)
- `is_fallback_orientation` (`true` or `false`)
- `fallback_reason`
- `sim_source`

## 6.3 `manifests/scan_bundle_manifest.csv`

One row per scan bundle.

Required columns:

- `scan_id`
- `assume_fe_bcc_oh5_path`
- `assume_fe3o4_magnetite_oh5_path`
- `assume_feo_wustite_oh5_path`
- `nx`
- `ny`
- `pattern_height`
- `pattern_width`
- `flat_index_rule` (set `row_major_y_times_nx_plus_x`)
- `notes`

## 6.4 `manifests/preprocessing_policy.yaml`

Must freeze preprocessing choices for NCC comparability.

Required keys:

- `dtype_target`
- `normalization_method`
- `mask_method`
- `mask_parameters`
- `resize_policy`
- `intensity_clip_policy`
- `ncc_variant`
- `euler_convention`
- `angle_units`
- `exp_sim_alignment_policy`

## 7. Minimum Acceptance Criteria for Dataset A (Curated)

- At least 2 cases per phase.
- Every experimental case has 3 simulation candidates.
- Dimensions match across each candidate trio.
- All cases have `expected_phase` and manual evidence notes.
- Failed-index candidates are flagged explicitly in manifest.

## 8. Minimum Acceptance Criteria for Dataset B (Scan Bundle)

- All 3 `.oh5` files correspond to the same physical scan and grid.
- `nRows`, `nColumns`, and pattern shapes match across phase-isolated `.oh5` files.
- Candidate extraction can retrieve orientation/quality fields for all three assumptions.
- At least 10 manually verified pixel locations are listed separately for qualitative validation in early development.

## 9. Recommended Example Filenames

Experimental curated pattern:

- `fe3o4_magnetite_Ori_1.png`

Simulated candidates for the same case:

- `curated_patterns/sim/assume_fe_bcc/fe3o4_magnetite_Ori_1.png`
- `curated_patterns/sim/assume_fe3o4_magnetite/fe3o4_magnetite_Ori_1.png`
- `curated_patterns/sim/assume_feo_wustite/fe3o4_magnetite_Ori_1.png`

Indexing-failed candidate still represented with explicit flags in manifest:

- filename does not need special encoding, but manifest must set:
  - `indexing_status=failed`
  - `is_fallback_orientation=true`

## 10. Data Ingestion Readiness Checklist

- [ ] Directory structure matches this document.
- [ ] All required manifest files exist and parse cleanly.
- [ ] Paths in manifests are valid relative to repo root.
- [ ] No missing candidate rows for any curated case.
- [ ] Preprocessing policy file is present and locked.
- [ ] `.oh5` files pass schema discovery checks described in `docs/oh5_structure.md`.

## 11. How This Supports Coding Phases

Phase 1 code can start immediately with this package by implementing:

1. curated-case NCC evaluator,
2. `.oh5` candidate extractor,
3. decision reporter with confidence margin.

This plan intentionally separates experimental truth labels, candidate provenance, and preprocessing policy to keep the implementation auditable and easy to extend.
