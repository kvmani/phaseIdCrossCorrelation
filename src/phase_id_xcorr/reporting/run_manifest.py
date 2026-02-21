"""Run-manifest helper."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rel(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        return out
    except Exception:
        return "unknown"


def build_run_manifest(*, repo_root: Path, packet_dir: Path, out_dir: Path, debug: bool, extra: dict | None = None) -> dict:
    """Construct a machine-readable run manifest."""

    payload = {
        "timestamp_utc": _now_iso_utc(),
        "git_commit": _git_commit(repo_root),
        "repo_root": _rel(repo_root, repo_root),
        "packet_dir": _rel(packet_dir, repo_root),
        "out_dir": _rel(out_dir, repo_root),
        "debug": bool(debug),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if extra:
        payload.update(extra)
    return payload
