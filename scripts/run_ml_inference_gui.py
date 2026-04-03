#!/usr/bin/env python3
"""Launch desktop GUI for ML phase-classifier inference."""

from __future__ import annotations

import argparse
import faulthandler
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML phase-classifier inference GUI")
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=Path("reports/ml/benchmarks/ni_cu_al_production"),
        help="Benchmark suite root or one specific run directory.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    faulthandler.enable(all_threads=True)
    logging.getLogger("ml_inference_gui").info("Starting inference GUI. suite_root=%s debug=%s", args.suite_root, bool(args.debug))
    try:
        from phase_id_xcorr.ml.inference_gui import run_inference_gui
    except Exception as exc:
        logging.getLogger("ml_inference_gui").error(
            "Failed to import GUI dependencies. Install PySide6. error=%s",
            exc,
        )
        return 2

    suite_root = args.suite_root if args.suite_root.is_absolute() else (ROOT / args.suite_root)
    return int(run_inference_gui(repo_root=ROOT, suite_root=suite_root, debug=bool(args.debug)))


if __name__ == "__main__":
    raise SystemExit(main())
