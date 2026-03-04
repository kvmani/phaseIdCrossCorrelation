#!/usr/bin/env python3
"""Build ML-ready dataset from `.oh5` scans using configured label mode."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.ml import prepare_ml_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ML dataset from .oh5 scans (CSV labels or single-phase scan map mode)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ml/dataset_prepare.default.yml"),
        help="Dataset-prep YAML config path (repo-relative by default).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("ml_dataset_prepare")

    cfg = args.config if args.config.is_absolute() else (ROOT / args.config)

    result = prepare_ml_dataset(
        config_path=cfg,
        repo_root=ROOT,
        debug=bool(args.debug),
        logger=log,
    )

    manifest_rel = result.manifest_path.relative_to(ROOT) if result.manifest_path.is_relative_to(ROOT) else result.manifest_path
    log.info("ML dataset preparation complete | manifest=%s", manifest_rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
