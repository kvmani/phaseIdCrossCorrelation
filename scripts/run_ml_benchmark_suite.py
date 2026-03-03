#!/usr/bin/env python3
"""Run multi-model ML benchmark suite from YAML."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.ml import run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML benchmark suite")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ml/benchmark_suite.debug.yml"),
        help="Suite YAML config path (repo-relative by default).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail immediately on first failed run.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("ml_benchmark_suite")

    cfg = args.config if args.config.is_absolute() else (ROOT / args.config)

    result = run_benchmark_suite(
        suite_config_path=cfg,
        repo_root=ROOT,
        debug=bool(args.debug),
        logger=log,
        strict=bool(args.strict),
    )

    summary_rel = result.summary_json.relative_to(ROOT) if result.summary_json.is_relative_to(ROOT) else result.summary_json
    log.info("Benchmark suite complete | summary=%s", summary_rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
