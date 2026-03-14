"""Dataset artifact I/O helpers for ML workflows."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def rel_path(path: Path, root: Path) -> str:
    """Convert path to root-relative POSIX representation when possible.

    On Windows, input data may live on a different drive than the repository.
    In that case `os.path.relpath()` raises `ValueError`; fall back to the
    absolute path so persisted manifests remain usable.
    """

    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return Path(os.path.relpath(resolved_path, resolved_root)).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON mapping."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_records_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write tabular sample-level records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_split_npz(path: Path, *, patterns: np.ndarray, labels: np.ndarray, sample_ids: list[str]) -> None:
    """Persist one split tensor bundle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        patterns=patterns.astype(np.float32, copy=False),
        labels=labels.astype(np.int64, copy=False),
        sample_ids=np.asarray(sample_ids, dtype=object),
    )


def load_split_npz(path: Path) -> dict[str, Any]:
    """Load split tensor bundle."""

    with np.load(path, allow_pickle=True) as data:
        return {
            "patterns": np.asarray(data["patterns"], dtype=np.float32),
            "labels": np.asarray(data["labels"], dtype=np.int64),
            "sample_ids": [str(x) for x in data["sample_ids"].tolist()],
        }
