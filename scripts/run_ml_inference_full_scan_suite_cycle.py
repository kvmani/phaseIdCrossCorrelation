#!/usr/bin/env python3
"""Run suite-level full-scan `.oh5` inference exports and build the comparative HTML report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.ml.html_report import generate_full_scan_suite_html_report
from phase_id_xcorr.ml.oh5_inference import run_suite_full_scan_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command full-scan suite cycle: export full .oh5 inference for every run in a benchmark suite, then build the comparative HTML report."
    )
    parser.add_argument("--suite-root", type=Path, required=True, help="Benchmark suite root.")
    parser.add_argument("--oh5", type=Path, required=True, help="Target .oh5 scan path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for suite exports and the comparative HTML.")
    parser.add_argument("--scan-name", type=str, default="", help="Optional logical scan name override. Defaults to .oh5 stem.")
    parser.add_argument("--checkpoint", type=str, default="best_checkpoint.pt", help="Checkpoint filename to load from each run directory.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda.")
    parser.add_argument("--no-confidence-shading", action="store_true", help="Disable confidence shading in exported predicted phase maps.")
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("comparison_report.html"),
        help="Comparative HTML filename or path. Relative paths are resolved under --output-dir.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("ml_full_scan_suite_cycle")

    suite_root = args.suite_root if args.suite_root.is_absolute() else (ROOT / args.suite_root)
    oh5_path = args.oh5 if args.oh5.is_absolute() else (ROOT / args.oh5)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    output_html = (
        args.output_html
        if args.output_html.is_absolute()
        else (output_dir / args.output_html)
    )

    inference_result = run_suite_full_scan_inference(
        suite_root=suite_root,
        oh5_path=oh5_path,
        output_dir=output_dir,
        repo_root=ROOT,
        checkpoint_name=str(args.checkpoint),
        device=str(args.device),
        scan_name=str(args.scan_name).strip() or None,
        use_confidence_shading=not bool(args.no_confidence_shading),
        logger=log,
    )

    report_path = generate_full_scan_suite_html_report(
        summary_json_path=inference_result.summary_json,
        output_html=output_html,
        repo_root=ROOT,
    )

    summary_rel = inference_result.summary_json.relative_to(ROOT) if inference_result.summary_json.is_relative_to(ROOT) else inference_result.summary_json
    report_rel = report_path.relative_to(ROOT) if report_path.is_relative_to(ROOT) else report_path
    log.info(
        "Full-scan suite cycle complete | completed=%d failed=%d summary=%s html=%s",
        inference_result.processed_runs,
        inference_result.failed_runs,
        summary_rel,
        report_rel,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
