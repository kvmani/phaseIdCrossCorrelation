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
    group_key: str | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    val_samples_per_phase: int | None = None
    test_samples_per_phase: int | None = None


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
        group_key=str(cfg.get("group_key")).strip() if cfg.get("group_key") else None,
        max_val_samples=int(cfg["max_val_samples"]) if cfg.get("max_val_samples") is not None else None,
        max_test_samples=int(cfg["max_test_samples"]) if cfg.get("max_test_samples") is not None else None,
        val_samples_per_phase=int(cfg["val_samples_per_phase"]) if cfg.get("val_samples_per_phase") is not None else None,
        test_samples_per_phase=int(cfg["test_samples_per_phase"]) if cfg.get("test_samples_per_phase") is not None else None,
    )


def _counts_for_n(n: int, cfg: SplitConfig) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0

    n_train = int(math.floor(n * cfg.train))
    n_val = int(math.floor(n * cfg.val))
    n_test = n - n_train - n_val

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


def _apply_caps(assignments: list[str], cfg: SplitConfig, labels: list[int], rng: np.random.Generator) -> list[str]:
    out = list(assignments)

    def _cap_split(split_name: str, max_samples: int | None) -> None:
        if max_samples is None or max_samples < 0:
            return
        idxs = [i for i, s in enumerate(out) if s == split_name]
        if len(idxs) <= max_samples:
            return
        if cfg.stratified:
            idx_by_label: dict[int, list[int]] = {}
            for i in idxs:
                idx_by_label.setdefault(int(labels[i]), []).append(i)
            kept: list[int] = []
            for label in sorted(idx_by_label):
                arr = np.asarray(idx_by_label[label], dtype=np.int64)
                rng.shuffle(arr)
                quota = int(round(max_samples * len(arr) / max(1, len(idxs))))
                kept.extend(arr[:quota].tolist())
            if len(kept) < max_samples:
                rem = [i for i in idxs if i not in set(kept)]
                arr = np.asarray(rem, dtype=np.int64)
                rng.shuffle(arr)
                kept.extend(arr[: max_samples - len(kept)].tolist())
            kept_set = set(kept[:max_samples])
        else:
            arr = np.asarray(idxs, dtype=np.int64)
            rng.shuffle(arr)
            kept_set = set(arr[:max_samples].tolist())
        for i in idxs:
            if i not in kept_set:
                out[i] = "train"

    _cap_split("val", cfg.max_val_samples)
    _cap_split("test", cfg.max_test_samples)
    return out


def build_split_assignments(labels: list[int], cfg: SplitConfig, *, groups: list[str] | None = None) -> list[str]:
    n = len(labels)
    if n == 0:
        return []
    if groups is not None and len(groups) != n:
        raise ValueError("groups must have same length as labels")

    rng = np.random.default_rng(cfg.seed)
    out = ["" for _ in range(n)]

    if cfg.val_samples_per_phase is not None or cfg.test_samples_per_phase is not None:
        if groups:
            raise ValueError("val_samples_per_phase/test_samples_per_phase are not supported with group-based splitting")
        if not cfg.stratified:
            raise ValueError("val_samples_per_phase/test_samples_per_phase require stratified=true")
        label_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(int(label), []).append(idx)

        val_n = max(0, int(cfg.val_samples_per_phase or 0))
        test_n = max(0, int(cfg.test_samples_per_phase or 0))

        for _, idxs in sorted(label_to_indices.items(), key=lambda kv: kv[0]):
            arr = np.asarray(idxs, dtype=np.int64)
            rng.shuffle(arr)
            n_val = min(val_n, int(arr.size))
            remaining_after_val = max(0, int(arr.size) - n_val)
            n_test = min(test_n, remaining_after_val)

            val_idx = arr[:n_val]
            test_idx = arr[n_val : n_val + n_test]
            train_idx = arr[n_val + n_test :]

            for i in train_idx:
                out[int(i)] = "train"
            for i in val_idx:
                out[int(i)] = "val"
            for i in test_idx:
                out[int(i)] = "test"

        if any(s == "" for s in out):
            raise RuntimeError("Internal error: unassigned split entries found")
        return out

    if groups:
        group_to_indices: dict[str, list[int]] = {}
        for idx, group in enumerate(groups):
            group_to_indices.setdefault(str(group), []).append(idx)

        # Majority label per group for approximate stratification.
        grouped: list[tuple[str, int, np.ndarray]] = []
        for gid, idxs in sorted(group_to_indices.items()):
            arr = np.asarray(idxs, dtype=np.int64)
            vals, cnts = np.unique(np.asarray([labels[i] for i in idxs], dtype=np.int64), return_counts=True)
            maj = int(vals[int(np.argmax(cnts))])
            grouped.append((gid, maj, arr))

        if cfg.stratified:
            by_label: dict[int, list[tuple[str, int, np.ndarray]]] = {}
            for rec in grouped:
                by_label.setdefault(rec[1], []).append(rec)
            for label in sorted(by_label):
                chunks = by_label[label]
                order = np.arange(len(chunks), dtype=np.int64)
                rng.shuffle(order)
                n_train, n_val, _ = _counts_for_n(len(chunks), cfg)
                for p, idx_chunk in enumerate(order.tolist()):
                    _, _, members = chunks[int(idx_chunk)]
                    split = "train" if p < n_train else "val" if p < n_train + n_val else "test"
                    for i in members.tolist():
                        out[i] = split
        else:
            order = np.arange(len(grouped), dtype=np.int64)
            rng.shuffle(order)
            n_train, n_val, _ = _counts_for_n(len(grouped), cfg)
            for p, idx_group in enumerate(order.tolist()):
                _, _, members = grouped[int(idx_group)]
                split = "train" if p < n_train else "val" if p < n_train + n_val else "test"
                for i in members.tolist():
                    out[i] = split
    elif cfg.stratified:
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

    out = _apply_caps(out, cfg, labels, rng)
    return out
