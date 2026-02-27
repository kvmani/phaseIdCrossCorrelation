# Curated NCC Workflow

This document describes the implemented curated-case workflow for phase identification using masked NCC.

For direct comparison against KikuchiPy Hough-space scoring, see:

- `docs/curated_hough_vs_ncc_workflow.md`

For a full copy-paste run cycle (G0 validation + baseline NCC + Hough-space comparison), see:

- `docs/mcc_vs_hough_full_cycle_runbook.md`

## Scope

The workflow compares each curated experimental pattern against three simulated candidates (one per assumed phase):

- `fe_bcc`
- `fe3o4_magnetite`
- `feo_wustite`

The top NCC candidate is selected as the predicted phase.

## Input Files

Required files under a packet directory (example: `data/test/student_data_packet_phaseid`):

- `01_experimental_patterns_template.json`
- `02_simulated_patterns_template.json`
- `04_processing_template.json`
- referenced image files under `experimental_patterns/` and `simulated_patterns/`

## Run Command

```bash
python3 scripts/run_curated_ncc.py \
  --packet-dir data/test/student_data_packet_phaseid \
  --out-dir reports/curated_ncc \
  --debug
```

Build a single-file inspection artifact (embedded images + metadata + NCC tables):

```bash
python3 scripts/build_curated_ncc_inspection_html.py \
  --packet-dir data/test/student_data_packet_phaseid \
  --results-dir reports/curated_ncc \
  --out-html reports/curated_ncc/inspection_report.html
```

Inspection HTML now includes:

- explicit per-record discrimination summary:
  - `Winner NCC: phase (score)`
  - `Other NCCs: [phase (score), ...]`
- optional interactive pattern match viewer:
  - `Overlay` mode with adjustable alpha
  - `Split 50/50` mode (left experimental, right simulated)

## Algorithm Steps

1. Load experimental/simulated mapping JSON files.
2. For each curated record:
   - load experimental pattern,
   - load 3 simulated patterns.
3. Convert any supported image format to canonical grayscale float32 in `[0,1]`.
4. Build centered maximum-inscribed circular mask.
5. Normalize intensities inside mask (`minmax_inside_mask` from processing settings).
6. Compute masked NCC for each candidate.
7. Rank NCC scores and choose winner.
8. Compute confidence margin (`top1 - top2`).
9. Run robustness probes with mild perturbations (noise + gain jitter) and estimate winner flip rate.

## Supported Image Formats and Bit Depth

Supported formats:

- `.bmp`, `.png`, `.tif`, `.tiff`, `.jpg`, `.jpeg`

Bit depth handling:

- accepts both 8-bit and 16-bit sources,
- converts to canonical float32 `[0,1]` before masking/normalization/NCC.

## Output Artifacts

Outputs under the specified `--out-dir`:

- `scores.csv`: one row per experimental-candidate pair.
- `decisions.csv`: one row per experimental record with predicted phase and confidence fields.
- `summary.json`: aggregate metrics and per-phase breakdown.
- `error_cases.md`: compact list of misclassified records.
- `manifest.json`: run metadata for traceability.
- `cases/<record_id>_panel.png`: visual comparison panels (exp + 3 sims + NCC labels).
- `inspection_report.html`: standalone bird's-eye inspection artifact (all records in one file).
  Includes per-record winner-vs-others score breakdown and optional interactive blend/split viewer.

## Key Fields

From `decisions.csv`:

- `pred_phase`
- `is_correct`
- `top_ncc`
- `second_ncc`
- `margin`
- `top_indexing_status`
- `top_is_fallback_orientation`
- `flip_rate`

## Notes

- Current robustness probe is intentionally lightweight and deterministic.
- This workflow is curated-case focused (single experimental patterns + mapped simulations).
- Scan-scale `.oh5` candidate extraction remains a separate phase-gated task.
