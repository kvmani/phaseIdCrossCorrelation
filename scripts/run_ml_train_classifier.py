#!/usr/bin/env python3
"""Train ML phase classifier from prepared dataset manifest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.ml import train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phase classifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ml/train.convnextv2_nano.pretrained.debug.yml"),
        help="Training YAML config path (repo-relative by default).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("ml_train")

    cfg = args.config if args.config.is_absolute() else (ROOT / args.config)

    result = train_classifier(
        config_path=cfg,
        repo_root=ROOT,
        debug=bool(args.debug),
        logger=log,
    )

    report_rel = result.report_path.relative_to(ROOT) if result.report_path.is_relative_to(ROOT) else result.report_path
    log.info("ML training complete | report=%s", report_rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
