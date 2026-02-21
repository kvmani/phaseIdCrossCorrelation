#!/usr/bin/env python3
"""Run curated comparison: image-space NCC vs KikuchiPy Hough-space NCC."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.evaluation.curated_hough_vs_ncc import (  # noqa: E402
    build_curated_hough_vs_ncc_html,
    run_curated_hough_vs_ncc,
)
from phase_id_xcorr.features import HoughTransformConfig  # noqa: E402


def _parse_thresholds(value: str) -> list[float]:
    out: list[float] = []
    for part in value.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    if not out:
        raise argparse.ArgumentTypeError("Expected comma-separated threshold values")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curated image NCC vs Hough NCC runner")
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=Path("data/test/student_data_packet_phaseid"),
        help="Input packet directory (repo-relative by default).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/curated_hough_vs_ncc"),
        help="Output report directory (repo-relative by default).",
    )
    parser.add_argument(
        "--binary-thresholds",
        type=_parse_thresholds,
        default=[0.35, 0.45, 0.55, 0.65, 0.75],
        help="Comma-separated Hough binarization thresholds in [0,1].",
    )
    parser.add_argument("--hough-n-theta", type=int, default=180, help="Hough theta bins.")
    parser.add_argument("--hough-n-rho", type=int, default=90, help="Hough rho bins.")
    parser.add_argument("--hough-n-bands", type=int, default=9, help="Bands in KikuchiPy band-detect plan.")
    parser.add_argument(
        "--hough-use-convolved-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use convolved Hough map from KikuchiPy/PyEBSDIndex pipeline.",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=Path("reports/curated_hough_vs_ncc/inspection_report.html"),
        help="Single HTML report output path.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode/logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("curated_hough_vs_ncc")

    packet_dir = args.packet_dir if args.packet_dir.is_absolute() else (ROOT / args.packet_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    html_out = args.html_out if args.html_out.is_absolute() else (ROOT / args.html_out)

    hough_cfg = HoughTransformConfig(
        n_theta=int(args.hough_n_theta),
        n_rho=int(args.hough_n_rho),
        n_bands=int(args.hough_n_bands),
        use_convolved_map=bool(args.hough_use_convolved_map),
    )

    log.info("Running curated image NCC vs Hough NCC comparison")
    log.info("Packet directory: %s", packet_dir.relative_to(ROOT) if packet_dir.is_relative_to(ROOT) else packet_dir)
    log.info("Output directory: %s", out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir)
    log.info("Binary thresholds: %s", args.binary_thresholds)

    outputs = run_curated_hough_vs_ncc(
        packet_dir=packet_dir,
        out_dir=out_dir,
        repo_root=ROOT,
        binary_thresholds=[float(t) for t in args.binary_thresholds],
        hough_config=hough_cfg,
        debug=bool(args.debug),
        logger=log,
    )

    html_path = build_curated_hough_vs_ncc_html(
        report_data_json=outputs["report_data_json"],
        out_html=html_out,
        repo_root=ROOT,
    )
    rel_html = html_path.relative_to(ROOT) if html_path.is_relative_to(ROOT) else html_path
    log.info("Wrote HTML report: %s", rel_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

