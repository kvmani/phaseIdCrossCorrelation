from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from phase_id_xcorr.ml.dataset_io import save_split_npz, write_json
from phase_id_xcorr.ml.inference import list_model_runs, load_trained_model, predict_image
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
            arr[4:14, 4:14] = 0.8
        elif cls == 1:
            arr[:, ::3] = 0.7
        else:
            rr = np.arange(h)[:, None]
            cc = np.arange(w)[None, :]
            arr[((rr - h // 2) ** 2 + (cc - w // 2) ** 2) < 25] = 0.9
        arr += rng.normal(0.0, 0.02, size=(h, w)).astype(np.float32)
        patterns[i] = np.clip(arr, 0.0, 1.0)
    return patterns, labels


def test_inference_loads_trained_run_and_predicts_image(tmp_path: Path) -> None:
    repo_root = tmp_path
    out_dataset = tmp_path / "dataset"
    out_dataset.mkdir(parents=True, exist_ok=True)

    train_p, train_y = _make_patterns(30, 18, 18, seed=1)
    val_p, val_y = _make_patterns(12, 18, 18, seed=2)
    test_p, test_y = _make_patterns(12, 18, 18, seed=3)

    save_split_npz(out_dataset / "train.npz", patterns=train_p, labels=train_y, sample_ids=[f"tr_{i}" for i in range(len(train_y))])
    save_split_npz(out_dataset / "val.npz", patterns=val_p, labels=val_y, sample_ids=[f"va_{i}" for i in range(len(val_y))])
    save_split_npz(out_dataset / "test.npz", patterns=test_p, labels=test_y, sample_ids=[f"te_{i}" for i in range(len(test_y))])

    manifest = {
        "schema_version": "phase_id_xcorr.ml_dataset_manifest.v1",
        "phase_to_label": {"Al": 0, "Ni": 1, "Cu": 2},
        "preprocessing_policy": {
            "resize_hw": [24, 24],
            "apply_circular_mask": True,
            "normalize_mode": "none",
        },
        "artifacts": {
            "train_npz": "dataset/train.npz",
            "val_npz": "dataset/val.npz",
            "test_npz": "dataset/test.npz",
        },
    }
    write_json(tmp_path / "dataset_manifest.json", manifest)

    train_cfg = {
        "dataset_manifest_path": "dataset_manifest.json",
        "output_dir": "suite_run/simple_cnn_w8",
        "seed": 5,
        "device": "cpu",
        "amp": False,
        "batch_size": 8,
        "epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "input": {
            "resize_hw": [24, 24],
            "apply_circular_mask": True,
            "normalize": {"mean": [0.0], "std": [1.0]},
        },
        "model": {
            "family": "simple_cnn",
            "width": 8,
            "in_chans": 1,
        },
    }
    cfg_path = tmp_path / "train.yml"
    cfg_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")
    train_classifier(config_path=cfg_path, repo_root=repo_root, debug=True)

    suite_root = tmp_path / "suite_run"
    run_dirs = list_model_runs(suite_root)
    assert len(run_dirs) == 1

    loaded = load_trained_model(run_dir=run_dirs[0], repo_root=repo_root, device="cpu")
    image_path = tmp_path / "unknown.png"
    Image.fromarray((test_p[0] * 255.0).round().astype(np.uint8), mode="L").save(image_path)

    result = predict_image(loaded=loaded, image_path=image_path)
    assert result.predicted_phase in {"Al", "Ni", "Cu"}
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
    assert result.preprocessed_image.shape == (24, 24)
