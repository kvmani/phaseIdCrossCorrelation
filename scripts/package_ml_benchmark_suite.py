#!/usr/bin/env python3
"""Package lightweight ML benchmark-suite and inference-export artifacts into a transfer zip."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.ml.suite_packaging import package_benchmark_suite_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package transfer-friendly ML benchmark and inference artifacts into a zip."
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        required=True,
        help="Benchmark suite output directory.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=None,
        help="Output zip path. Defaults to <suite-root>/<suite-name>_lightweight_bundle.zip",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        action="append",
        default=[],
        help="Optional suite-level full-scan inference export root(s) to include.",
    )
    parser.add_argument(
        "--extra-path",
        type=Path,
        action="append",
        default=[],
        help="Optional extra file or directory to include, e.g. PPTX or PPT manifest.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=25.0,
        help="Maximum file size allowed into the archive.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    log = logging.getLogger("ml_suite_package")

    suite_root = args.suite_root if args.suite_root.is_absolute() else (ROOT / args.suite_root)
    suite_root = suite_root.resolve()
    output_zip = args.output_zip
    if output_zip is None:
        output_zip = suite_root / f"{suite_root.name}_lightweight_bundle.zip"
    elif not output_zip.is_absolute():
        output_zip = (ROOT / output_zip).resolve()

    extra_paths = [p if p.is_absolute() else (ROOT / p) for p in args.extra_path]
    inference_roots = [p if p.is_absolute() else (ROOT / p) for p in args.inference_root]

    result = package_benchmark_suite_artifacts(
        suite_root=suite_root,
        repo_root=ROOT,
        output_zip=output_zip,
        inference_roots=inference_roots,
        extra_paths=extra_paths,
        max_file_size_mb=float(args.max_file_size_mb),
    )
    log.info("Wrote lightweight suite bundle | zip=%s manifest=%s included=%d excluded=%d", result.zip_path, result.manifest_path, len(result.included_files), len(result.excluded_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
