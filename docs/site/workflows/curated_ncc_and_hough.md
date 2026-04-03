# Curated NCC and Hough Workflows

The curated workflows cover the repository's interpretable evidence branch. They are the right place to start when you need:

- case-by-case evidence inspection,
- direct comparison between image-space and Hough-space similarity behavior,
- a baseline against which ML behavior can be discussed scientifically.

## Main commands

```powershell
python .\scripts\run_g0_data_intake_validation.py --debug
python .\scripts\run_curated_ncc.py --debug
python .\scripts\run_curated_hough_vs_ncc.py --debug
```

## Why these workflows exist

ML can be powerful, but it is not self-explanatory. The curated NCC/Hough workflows preserve a track where:

- individual cases can be inspected directly,
- runner-up margins remain interpretable,
- preprocessing changes can be evaluated without retraining a classifier.

## Source docs

Legacy source files:

- `docs/curated_ncc_workflow.md`
- `docs/curated_hough_vs_ncc_workflow.md`
- `docs/mcc_vs_hough_full_cycle_runbook.md`
