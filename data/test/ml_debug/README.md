# ML Debug Fixture

Small deterministic `.oh5` + CSV fixtures for ML dataset-preparation and training smoke tests.

Contents:

- `scan_ml_debug_s001.oh5`
- `scan_ml_debug_s001_labels.csv`
- `scan_ml_debug_s002.oh5`
- `scan_ml_debug_s002_labels.csv`

Notes:

- Files are synthetic and intended only for debug/testing.
- Each `.oh5` includes `Pattern`, `CI`, `IQ`, `Fit`, `Valid`, `Phi1`, `Phi`, `Phi2`, `Phase`, `X Position`, `Y Position`.
- CSV labels are phase-name based and reference pixels by `(x, y)`.
