#!/usr/bin/env python3
"""Build, clean, and optionally open the Sphinx documentation site."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

from sphinx.cmd.build import main as sphinx_main


ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = ROOT / "docs" / "site"
DOCS_BUILD = ROOT / "docs" / "_build"
HTML_INDEX = DOCS_BUILD / "html" / "index.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or clean the repository Sphinx docs site.")
    parser.add_argument("--clean", action="store_true", help="Remove docs/_build before building.")
    parser.add_argument("--open", action="store_true", help="Open docs/_build/html/index.html after a successful build.")
    parser.add_argument("--builder", default="html", help="Sphinx builder to use. Default: html.")
    parser.add_argument("--warning-is-error", action="store_true", help="Fail build on Sphinx warnings.")
    return parser.parse_args()


def _open_index(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Built documentation index not found: {path}")
    os.startfile(path)  # type: ignore[attr-defined]


def main() -> int:
    args = parse_args()
    if args.clean and DOCS_BUILD.exists():
        shutil.rmtree(DOCS_BUILD)

    DOCS_BUILD.mkdir(parents=True, exist_ok=True)
    argv = [
        "-b",
        str(args.builder),
        str(DOCS_SOURCE),
        str(DOCS_BUILD / args.builder),
    ]
    if args.warning_is_error:
        argv.insert(0, "-W")

    rc = int(sphinx_main(argv))
    if rc != 0:
        return rc

    if args.open and args.builder == "html":
        _open_index(HTML_INDEX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
