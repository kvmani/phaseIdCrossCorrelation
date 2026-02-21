#!/usr/bin/env python3
"""Run curated experimental-vs-simulated NCC workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.evaluation import run_curated_ncc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curated NCC runner")
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=Path("data/test/student_data_packet_phaseid"),
        help="Input packet directory (repo-relative by default).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/curated_ncc"),
        help="Output report directory (repo-relative by default).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode/logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("curated_ncc")

    packet_dir = args.packet_dir if args.packet_dir.is_absolute() else (ROOT / args.packet_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)

    log.info("Running curated NCC")
    log.info("Packet directory: %s", packet_dir.relative_to(ROOT) if packet_dir.is_relative_to(ROOT) else packet_dir)
    log.info("Output directory: %s", out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir)

    run_curated_ncc(
        packet_dir=packet_dir,
        out_dir=out_dir,
        repo_root=ROOT,
        debug=args.debug,
        logger=log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
