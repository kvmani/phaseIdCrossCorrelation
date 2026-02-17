#!/usr/bin/env python3
"""Run G0 data-intake validation for a student packet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.intake.g0_validator import validate_data_packet, write_g0_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G0 data-intake validator")
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=Path("student_data_packet_phaseid"),
        help="Path to the student packet directory (repo-relative by default).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for generated validation reports (repo-relative by default).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code unless gate status is GO.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("g0_validator")

    packet_dir = args.packet_dir if args.packet_dir.is_absolute() else (ROOT / args.packet_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)

    log.info("Starting G0 validation")
    log.info("Packet directory: %s", packet_dir.relative_to(ROOT) if packet_dir.is_relative_to(ROOT) else packet_dir)
    log.info("Output directory: %s", out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir)

    result = validate_data_packet(packet_dir, logger=log, repo_root=ROOT)
    md_path, json_path = write_g0_reports(result, out_dir)

    log.info("Gate status: %s", result.gate_status)
    log.info("Findings: %s", result.counts)
    log.info("Markdown report: %s", md_path.relative_to(ROOT) if md_path.is_relative_to(ROOT) else md_path)
    log.info("JSON manifest: %s", json_path.relative_to(ROOT) if json_path.is_relative_to(ROOT) else json_path)

    if args.strict and result.gate_status != "GO":
        log.error("Strict mode enabled and gate is not GO.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
