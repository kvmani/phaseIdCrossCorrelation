# Full Cycle Runbook: MCC vs Hough-Space NCC

This runbook provides a single, copy-pasteable execution cycle for:

1. data-intake gate validation (G0),
2. curated baseline masked NCC (`curated_ncc`),
3. curated image-space vs Hough-space comparison (`curated_hough_vs_ncc`),
4. printed headline metrics for quick method comparison.

Terminology used in outputs:

- "simple MCC" / image-space MCC corresponds to `image_ncc`.
- baseline curated MCC summary is `curated_ncc/summary.json` (`top1_accuracy`).
- Hough-space MCC variants are `hough_ncc_raw` and `hough_ncc_bin_t...`.

## Preconditions

- Run from repository root.
- Packet path used here: `data/test/student_data_packet_phaseid`.
- Python 3.10+ recommended.

Optional one-time Linux environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy pillow h5py kikuchipy orix
```

## One-Go Command Block (Recommended)

```bash
set -euo pipefail

PACKET_DIR="data/test/student_data_packet_phaseid"
RUN_ROOT="reports/mcc_vs_hough_cycle_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

python3 scripts/run_g0_data_intake_validation.py \
  --packet-dir "$PACKET_DIR" \
  --out-dir "$RUN_ROOT" \
  --strict

python3 scripts/run_curated_ncc.py \
  --packet-dir "$PACKET_DIR" \
  --out-dir "$RUN_ROOT/curated_ncc"

python3 scripts/build_curated_ncc_inspection_html.py \
  --packet-dir "$PACKET_DIR" \
  --results-dir "$RUN_ROOT/curated_ncc" \
  --out-html "$RUN_ROOT/curated_ncc/inspection_report.html"

python3 scripts/run_curated_hough_vs_ncc.py \
  --packet-dir "$PACKET_DIR" \
  --out-dir "$RUN_ROOT/curated_hough_vs_ncc" \
  --binary-thresholds 0.35,0.45,0.55,0.65,0.75 \
  --hough-n-theta 180 \
  --hough-n-rho 90 \
  --hough-n-bands 9 \
  --hough-use-convolved-map \
  --html-out "$RUN_ROOT/curated_hough_vs_ncc/inspection_report.html"

RUN_ROOT="$RUN_ROOT" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
base = json.loads((root / "curated_ncc" / "summary.json").read_text(encoding="utf-8"))
cmp_ = json.loads((root / "curated_hough_vs_ncc" / "summary.json").read_text(encoding="utf-8"))
head = cmp_["headline_comparison"]

print("=== MCC vs Hough-NCC headline ===")
print(f"run_root={root.as_posix()}")
print(f"baseline_top1_accuracy={base['top1_accuracy']:.6f}")
print(f"image_ncc_accuracy={head['image_ncc_accuracy']:.6f}")
print(f"hough_ncc_raw_accuracy={head['hough_ncc_raw_accuracy']:.6f}")
print(f"best_hough_binary_method={head['best_hough_binary_method']}")
print(f"best_hough_binary_accuracy={head['best_hough_binary_accuracy']:.6f}")
print(f"best_binary_threshold={cmp_['best_binary_threshold']:.3f}")
print()
print("Per-method metrics:")
for method_name, m in cmp_["metrics_by_method"].items():
    print(
        f"{method_name:18s} "
        f"acc={float(m['accuracy']):.6f} "
        f"mean_margin={float(m['mean_margin']):.6f} "
        f"mean_top={float(m['mean_top_score']):.6f}"
    )
print()
print("Key artifacts:")
print((root / "data_intake_validation.md").as_posix())
print((root / "curated_ncc" / "inspection_report.html").as_posix())
print((root / "curated_hough_vs_ncc" / "inspection_report.html").as_posix())
print((root / "curated_hough_vs_ncc" / "summary.json").as_posix())
PY
```

## Expected Outputs

Under `$RUN_ROOT`:

- `data_intake_validation.md`
- `data_intake_manifest.json`
- `curated_ncc/scores.csv`
- `curated_ncc/decisions.csv`
- `curated_ncc/summary.json`
- `curated_ncc/inspection_report.html`
- `curated_ncc/manifest.json`
- `curated_hough_vs_ncc/scores.csv`
- `curated_hough_vs_ncc/decisions.csv`
- `curated_hough_vs_ncc/summary.json`
- `curated_hough_vs_ncc/report_data.json`
- `curated_hough_vs_ncc/inspection_report.html`
- `curated_hough_vs_ncc/manifest.json`

## Optional Debug Variant

If you need debug logs, add `--debug` to the runner scripts. Use it selectively because Hough debug mode can produce very large logs:

```bash
python3 scripts/run_curated_ncc.py \
  --packet-dir data/test/student_data_packet_phaseid \
  --out-dir reports/curated_ncc_debug \
  --debug
```

