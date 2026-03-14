#!/usr/bin/env python3
"""Run end-to-end ML workflow: dataset prep -> benchmark suite -> HTML summary -> PPTX."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.ml.full_cycle import run_full_cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML full-cycle workflow")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ml/full_cycle.debug.yml"),
        help="Full-cycle YAML config path (repo-relative by default).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("ml_full_cycle")

    cfg = args.config if args.config.is_absolute() else (ROOT / args.config)
    result = run_full_cycle(workflow_config_path=cfg, repo_root=ROOT, debug=bool(args.debug), logger=log)
    summary_rel = result.summary_json.relative_to(ROOT) if result.summary_json.is_relative_to(ROOT) else result.summary_json
    log.info("ML full cycle complete | summary=%s", summary_rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
