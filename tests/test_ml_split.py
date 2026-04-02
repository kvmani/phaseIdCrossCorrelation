from __future__ import annotations

from phase_id_xcorr.ml.split import SplitConfig, build_split_assignments


def test_split_assignments_stratified_deterministic() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    cfg = SplitConfig(train=0.5, val=0.25, test=0.25, seed=123, stratified=True)

    a = build_split_assignments(labels, cfg)
    b = build_split_assignments(labels, cfg)

    assert a == b
    assert len(a) == len(labels)
    assert set(a) == {"train", "val", "test"}


def test_split_group_leakage_safe() -> None:
    labels = [0, 0, 1, 1, 0, 1, 0, 1]
    groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"]
    cfg = SplitConfig(train=0.5, val=0.25, test=0.25, seed=7, stratified=True)
    split = build_split_assignments(labels, cfg, groups=groups)
    by_group: dict[str, set[str]] = {}
    for g, s in zip(groups, split, strict=True):
        by_group.setdefault(g, set()).add(s)
    assert all(len(v) == 1 for v in by_group.values())


def test_split_caps_move_remainder_to_train() -> None:
    labels = [0] * 20 + [1] * 20
    cfg = SplitConfig(
        train=0.6,
        val=0.2,
        test=0.2,
        seed=1,
        stratified=True,
        max_val_samples=4,
        max_test_samples=5,
    )
    split = build_split_assignments(labels, cfg)
    assert split.count("val") <= 4
    assert split.count("test") <= 5
    assert split.count("train") == len(labels) - split.count("val") - split.count("test")


def test_split_exact_samples_per_phase_move_remainder_to_train() -> None:
    labels = [0] * 10 + [1] * 11 + [2] * 12
    cfg = SplitConfig(
        train=0.7,
        val=0.15,
        test=0.15,
        seed=5,
        stratified=True,
        val_samples_per_phase=3,
        test_samples_per_phase=3,
    )
    split = build_split_assignments(labels, cfg)

    for label in (0, 1, 2):
        idxs = [i for i, y in enumerate(labels) if y == label]
        label_splits = [split[i] for i in idxs]
        assert label_splits.count("val") == 3
        assert label_splits.count("test") == 3
        assert label_splits.count("train") == len(idxs) - 6


def test_split_exact_train_val_test_samples_per_phase() -> None:
    labels = [0] * 600 + [1] * 610 + [2] * 620
    cfg = SplitConfig(
        train=0.8,
        val=0.1,
        test=0.1,
        seed=42,
        stratified=True,
        train_samples_per_phase=500,
        val_samples_per_phase=20,
        test_samples_per_phase=20,
    )
    split = build_split_assignments(labels, cfg)

    for label in (0, 1, 2):
        idxs = [i for i, y in enumerate(labels) if y == label]
        label_splits = [split[i] for i in idxs]
        assert label_splits.count("val") == 20
        assert label_splits.count("test") == 20
        assert label_splits.count("train") == len(idxs) - 40
