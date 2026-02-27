# Curated Workflow: Image NCC vs KikuchiPy Hough-NCC

This workflow compares phase decisions from:

1. masked image-space NCC (`image_ncc`),
2. KikuchiPy Hough-space continuous-map NCC (`hough_ncc_raw`),
3. KikuchiPy Hough-space binarized-map NCC across threshold sweep (`hough_ncc_bin_t...`).

The Hough transform path uses KikuchiPy's PyEBSDIndex-backed Radon/Hough plan (not a custom Hough implementation).

For a one-go end-to-end command sequence (G0 gate + curated NCC + Hough comparison + summary printout), see:

- `docs/mcc_vs_hough_full_cycle_runbook.md`

## Run

From repository root:

```bash
PYTHONPATH=src python scripts/run_curated_hough_vs_ncc.py \
  --packet-dir data/test/student_data_packet_phaseid \
  --out-dir reports/curated_hough_vs_ncc \
  --binary-thresholds 0.35,0.45,0.55,0.65,0.75 \
  --hough-n-theta 180 \
  --hough-n-rho 90 \
  --hough-n-bands 9 \
  --hough-use-convolved-map \
  --html-out reports/curated_hough_vs_ncc/inspection_report.html
```

Optional:

- add `--debug` for verbose logging.
- pass `--no-hough-use-convolved-map` to compare against non-convolved Radon map.

## Primary Artifacts

- `reports/curated_hough_vs_ncc/summary.json`:
  method-level metrics, threshold sweep summary, best binary threshold.
- `reports/curated_hough_vs_ncc/scores.csv`:
  per record and candidate scores for all methods.
- `reports/curated_hough_vs_ncc/decisions.csv`:
  winner/margin per method per record.
- `reports/curated_hough_vs_ncc/report_data.json`:
  structured metadata used to compose HTML report.
- `reports/curated_hough_vs_ncc/inspection_report.html`:
  single visual report with annotated patterns, Hough maps, candidate tables, and method comparison.
  It includes per-method score discrimination breakdown (`winner_score` and `other_scores`) and an optional
  interactive exp-vs-sim pattern viewer (`Overlay` with alpha slider or `Split 50/50`).
- `reports/curated_hough_vs_ncc/manifest.json`:
  run metadata for reproducibility.

## Notes

- Persisted paths in JSON/CSV outputs are repository-relative.
- Experimental and simulated image preprocessing follows the same normalization policy from `04_processing_template.json`.
- This workflow is curated-case validation and method comparison; it does not yet run scan-wide `.oh5` extraction.
