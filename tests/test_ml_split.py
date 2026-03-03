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

    # Each class contributes at least one sample to each split for n=4/class.
    for cls in (0, 1, 2):
        cls_idx = [i for i, y in enumerate(labels) if y == cls]
        splits = {a[i] for i in cls_idx}
        assert splits == {"train", "val", "test"}
