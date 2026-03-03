from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from phase_id_xcorr.ml.dataset_io import save_split_npz, write_json, read_json
from phase_id_xcorr.ml.training import train_classifier


def _make_patterns(n: int, h: int, w: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    patterns = np.zeros((n, h, w), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        cls = i % 3
        labels[i] = cls
        arr = np.zeros((h, w), dtype=np.float32)
        if cls == 0:
            arr[6:18, 6:18] = 0.8
        elif cls == 1:
            arr[:, ::3] = 0.7
        else:
            rr = np.arange(h)[:, None]
            cc = np.arange(w)[None, :]
            arr[((rr - h // 2) ** 2 + (cc - w // 2) ** 2) < 36] = 0.9
        arr += rng.normal(0.0, 0.03, size=(h, w)).astype(np.float32)
        patterns[i] = np.clip(arr, 0.0, 1.0)
    return patterns, labels


def test_ml_training_simple_cnn_smoke(tmp_path: Path) -> None:
    repo_root = tmp_path
    out_dataset = tmp_path / "dataset"
    out_dataset.mkdir(parents=True, exist_ok=True)

    train_p, train_y = _make_patterns(30, 24, 24, seed=1)
    val_p, val_y = _make_patterns(12, 24, 24, seed=2)
    test_p, test_y = _make_patterns(12, 24, 24, seed=3)

    train_npz = out_dataset / "train.npz"
    val_npz = out_dataset / "val.npz"
    test_npz = out_dataset / "test.npz"

    save_split_npz(train_npz, patterns=train_p, labels=train_y, sample_ids=[f"tr_{i}" for i in range(len(train_y))])
    save_split_npz(val_npz, patterns=val_p, labels=val_y, sample_ids=[f"va_{i}" for i in range(len(val_y))])
    save_split_npz(test_npz, patterns=test_p, labels=test_y, sample_ids=[f"te_{i}" for i in range(len(test_y))])

    manifest = {
        "schema_version": "phase_id_xcorr.ml_dataset_manifest.v1",
        "phase_to_label": {"fe_bcc": 0, "fe3o4_magnetite": 1, "feo_wustite": 2},
        "artifacts": {
            "train_npz": "dataset/train.npz",
            "val_npz": "dataset/val.npz",
            "test_npz": "dataset/test.npz",
        },
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    write_json(manifest_path, manifest)

    train_cfg = {
        "dataset_manifest_path": "dataset_manifest.json",
        "output_dir": "runs/smoke",
        "seed": 5,
        "device": "cpu",
        "amp": False,
        "batch_size": 8,
        "epochs": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "input": {
            "resize_hw": [32, 32],
            "apply_circular_mask": True,
            "normalize": {"mean": [0.5], "std": [0.25]},
        },
        "model": {
            "family": "simple_cnn",
            "width": 8,
            "in_chans": 1,
        },
    }
    cfg_path = tmp_path / "train.yml"
    cfg_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")

    result = train_classifier(
        config_path=cfg_path,
        repo_root=repo_root,
        debug=True,
    )

    assert result.report_path.exists()
    assert result.last_checkpoint.exists()
    assert (result.out_dir / "epoch_history.jsonl").exists()
    assert (result.out_dir / "manifest.json").exists()
    assert (result.out_dir / "events.jsonl").exists()

    report = read_json(result.report_path)
    assert report["artifacts"]["event_log_jsonl"].endswith("events.jsonl")
    manifest = read_json(result.out_dir / "manifest.json")
    assert manifest["sanity_checks"]["history_written"] is True
    assert manifest["sanity_checks"]["non_empty_train_split"] is True
