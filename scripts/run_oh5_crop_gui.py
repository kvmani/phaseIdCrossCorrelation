#!/usr/bin/env python3
"""Launch dedicated `.oh5` crop and post-export review GUI."""

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
    parser = argparse.ArgumentParser(description="Run dedicated OH5 crop and review desktop GUI")
    parser.add_argument("--input", type=Path, default=None, help="Optional source .oh5 file to load at startup.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional default directory for exported cropped .oh5 files.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    try:
        from phase_id_xcorr.gui.oh5_crop_gui import run_oh5_crop_gui
    except Exception as exc:
        logging.getLogger("oh5_crop_gui").error(
            "Failed to import crop GUI dependencies. Install PySide6 and pyqtgraph. error=%s",
            exc,
        )
        return 2

    input_path = None if args.input is None else (args.input if args.input.is_absolute() else (ROOT / args.input))
    output_dir = None if args.output_dir is None else (
        args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    )
    return int(run_oh5_crop_gui(repo_root=ROOT, debug=bool(args.debug), input_path=input_path, output_dir=output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
