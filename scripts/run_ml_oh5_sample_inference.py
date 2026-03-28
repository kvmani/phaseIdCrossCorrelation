#!/usr/bin/env python3
"""Run sampled `.oh5` inference for a trained CNN phase-classifier model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.ml.oh5_inference import run_oh5_sample_inference
from phase_id_xcorr.ml.dataset_io import read_json


def _prediction_table(predictions: list[dict[str, Any]]) -> str:
    columns = [
        ("oh5_file", "oh5_file"),
        ("x", "x"),
        ("y", "y"),
        ("index", "index"),
        ("predicted_phase", "predicted_phase"),
        ("score", "score"),
    ]
    widths: dict[str, int] = {}
    for key, header in columns:
        values = [str(row.get(key, "")) for row in predictions]
        widths[key] = max(len(header), max((len(value) for value in values), default=0))

    header = " | ".join(header.ljust(widths[key]) for key, header in columns)
    divider = "-+-".join("-" * widths[key] for key, _ in columns)
    lines = [header, divider]
    for row in predictions:
        lines.append(
            " | ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns)
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample patterns from .oh5 scans, run saved CNN inference, and write per-pattern/summary reports."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("ml_oh5_inference")

    cfg = args.config if args.config.is_absolute() else (ROOT / args.config)
    result = run_oh5_sample_inference(
        config_path=cfg,
        repo_root=ROOT,
        debug=bool(args.debug),
        logger=log,
    )

    summary_rel = result.summary_json.relative_to(ROOT) if result.summary_json.is_relative_to(ROOT) else result.summary_json
    log.info(
        "Sampled .oh5 inference complete | scans=%d patterns=%d summary=%s",
        result.processed_scans,
        result.sampled_patterns,
        summary_rel,
    )
    predictions_payload = read_json(result.predictions_json)
    predictions = predictions_payload.get("predictions", [])
    if isinstance(predictions, list) and predictions:
        print(_prediction_table(predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
