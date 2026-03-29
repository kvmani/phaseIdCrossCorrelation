#!/usr/bin/env python3
"""Run the diagnostic pattern gallery GUI for `.oh5` inspection."""

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
    parser = argparse.ArgumentParser(description="Run the diagnostic pattern gallery desktop GUI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ml/diagnostic_gallery.example.yml"),
        help="Diagnostic gallery YAML config path (repo-relative by default)",
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
        from phase_id_xcorr.ml.diagnostic_gallery_gui import run_diagnostic_gallery_app
    except Exception as exc:
        logging.getLogger("ml_diagnostic_gallery").error(
            "Failed to import GUI dependencies. Install PySide6. error=%s", exc
        )
        return 2

    cfg = args.config if args.config.is_absolute() else (ROOT / args.config)
    return int(run_diagnostic_gallery_app(repo_root=ROOT, config_path=cfg, debug=bool(args.debug)))


if __name__ == "__main__":
    raise SystemExit(main())
