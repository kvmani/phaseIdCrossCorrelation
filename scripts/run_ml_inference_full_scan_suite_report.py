#!/usr/bin/env python3
"""Generate comparative HTML for suite-level full-scan `.oh5` exports."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one comparative HTML report from a suite-level full-scan export folder."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Path to suite_full_scan_summary.json.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("comparison_report.html"),
        help="Output HTML path.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("ml_full_scan_suite_report")
    summary_json = args.summary_json if args.summary_json.is_absolute() else (ROOT / args.summary_json)
    output_html = args.output_html if args.output_html.is_absolute() else (ROOT / args.output_html)
    path = generate_full_scan_suite_html_report(
        summary_json_path=summary_json,
        output_html=output_html,
        repo_root=ROOT,
    )
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    log.info("Full-scan comparative HTML written to %s", rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
