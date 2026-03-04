#!/usr/bin/env python3
"""Run ML benchmark suite and auto-generate a lab-meeting PPTX summary."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.ml import run_benchmark_suite


DEFAULT_SKILL_SCRIPT = (
    Path.home()
    / ".codex"
    / "skills"
    / "ml-results-presentation"
    / "scripts"
    / "generate_lab_meeting_ppt.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ML benchmark suite and compile results into a PPTX deck."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ml/benchmark_suite.debug.yml"),
        help="Suite YAML config path (repo-relative by default).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail immediately on first failed run.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--skip-ppt",
        action="store_true",
        help="Run suite only; do not generate PPTX.",
    )
    parser.add_argument(
        "--ppt-output-dir",
        type=Path,
        default=Path("reports/ml/presentations"),
        help="Directory for generated PPT manifest and deck.",
    )
    parser.add_argument(
        "--deck-title",
        type=str,
        default=None,
        help="Optional PPT deck title.",
    )
    parser.add_argument(
        "--ppt-max-results",
        type=int,
        default=8,
        help="Max number of result slides discovered by the presentation generator.",
    )
    parser.add_argument(
        "--ppt-script",
        type=Path,
        default=DEFAULT_SKILL_SCRIPT,
        help="Path to presentation skill script generate_lab_meeting_ppt.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("ml_suite_with_ppt")

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

    if args.skip_ppt:
        return 0

    ppt_script = args.ppt_script if args.ppt_script.is_absolute() else (ROOT / args.ppt_script)
    if not ppt_script.exists():
        raise FileNotFoundError(
            f"Presentation generator script not found: {ppt_script}. "
            "Ensure ml-results-presentation skill is installed."
        )

    ppt_output_dir = args.ppt_output_dir if args.ppt_output_dir.is_absolute() else (ROOT / args.ppt_output_dir)
    ppt_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ppt_script),
        "--scan-root",
        str(result.output_root),
        "--output-dir",
        str(ppt_output_dir),
        "--max-results",
        str(max(1, int(args.ppt_max_results))),
    ]
    if args.deck_title:
        cmd.extend(["--deck-title", args.deck_title])

    subprocess.run(cmd, check=True)
    log.info("PPT generation complete | output_dir=%s", ppt_output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
