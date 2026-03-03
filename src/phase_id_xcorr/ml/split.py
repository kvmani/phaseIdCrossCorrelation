"""Deterministic dataset splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(slots=True)
class SplitConfig:
    """Train/val/test split policy."""

    train: float
    val: float
    test: float
    seed: int = 42
    stratified: bool = True


def split_config_from_yaml(payload: dict[str, Any] | None) -> SplitConfig:
    cfg = payload or {}
    train = float(cfg.get("train", 0.7))
    val = float(cfg.get("val", 0.15))
    test = float(cfg.get("test", 0.15))

    total = train + val + test
    if total <= 0:
        raise ValueError("Split ratios must sum to > 0")

    train /= total
    val /= total
    test /= total

    return SplitConfig(
        train=train,
        val=val,
        test=test,
        seed=int(cfg.get("seed", 42)),
        stratified=bool(cfg.get("stratified", True)),
    )


def _counts_for_n(n: int, cfg: SplitConfig) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0

    n_train = int(math.floor(n * cfg.train))
    n_val = int(math.floor(n * cfg.val))
    n_test = n - n_train - n_val

    # Ensure each split gets at least one sample when feasible.
    if n >= 3:
        if n_train == 0:
            n_train += 1
            n_test -= 1
        if n_val == 0:
            n_val += 1
            n_test -= 1
        if n_test == 0:
            n_test += 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

    return n_train, n_val, n_test


def build_split_assignments(labels: list[int], cfg: SplitConfig) -> list[str]:
    """Return split name for each item index."""

    n = len(labels)
    if n == 0:
        return []

    rng = np.random.default_rng(cfg.seed)
    out = ["" for _ in range(n)]

    if cfg.stratified:
        label_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(int(label), []).append(idx)

        for _, idxs in sorted(label_to_indices.items(), key=lambda kv: kv[0]):
            arr = np.asarray(idxs, dtype=np.int64)
            rng.shuffle(arr)
            n_train, n_val, _ = _counts_for_n(len(arr), cfg)
            train_idx = arr[:n_train]
            val_idx = arr[n_train : n_train + n_val]
            test_idx = arr[n_train + n_val :]

            for i in train_idx:
                out[int(i)] = "train"
            for i in val_idx:
                out[int(i)] = "val"
            for i in test_idx:
                out[int(i)] = "test"
    else:
        arr = np.arange(n, dtype=np.int64)
        rng.shuffle(arr)
        n_train, n_val, _ = _counts_for_n(n, cfg)
        train_idx = arr[:n_train]
        val_idx = arr[n_train : n_train + n_val]
        test_idx = arr[n_train + n_val :]

        for i in train_idx:
            out[int(i)] = "train"
        for i in val_idx:
            out[int(i)] = "val"
        for i in test_idx:
            out[int(i)] = "test"

    if any(s == "" for s in out):
        raise RuntimeError("Internal error: unassigned split entries found")

    return out
