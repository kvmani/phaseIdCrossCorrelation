#!/usr/bin/env python3
"""Run native desktop `.oh5` phase explorer GUI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML phase OH5 exploratory desktop GUI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ml/dataset_prepare.v3_al_ni_cu.example.yml"),
        help="Dataset YAML config path (repo-relative by default)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        from phase_id_xcorr.ml.phase_explorer_gui import run_phase_explorer_app
    except Exception as exc:
        logging.getLogger("ml_phase_explorer").error(
            "Failed to import GUI dependencies. Install PySide6 and pyqtgraph. error=%s", exc
        )
        return 2

    cfg = args.config if args.config.is_absolute() else (ROOT / args.config)
    return int(run_phase_explorer_app(config_path=cfg, repo_root=ROOT, debug=bool(args.debug)))


if __name__ == "__main__":
    raise SystemExit(main())
