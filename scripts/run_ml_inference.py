#!/usr/bin/env python3
"""Run one-off inference for a trained ML phase-classifier model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.ml.inference import load_trained_model, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict phase ID for one unknown image.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing report.json and checkpoint.")
    parser.add_argument("--image", type=Path, required=True, help="Unknown image path.")
    parser.add_argument("--checkpoint", type=str, default="best_checkpoint.pt", help="Checkpoint filename inside run-dir.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    loaded = load_trained_model(run_dir=(ROOT / args.run_dir) if not args.run_dir.is_absolute() else args.run_dir, repo_root=ROOT, checkpoint_name=str(args.checkpoint), device=str(args.device))
    result = predict_image(loaded=loaded, image_path=(ROOT / args.image) if not args.image.is_absolute() else args.image)
    print(json.dumps(
        {
            "image_path": str(result.image_path),
            "predicted_phase": result.predicted_phase,
            "predicted_index": result.predicted_index,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "model_name": loaded.model_name,
            "run_dir": str(loaded.run_dir),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
