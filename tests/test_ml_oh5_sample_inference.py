from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import yaml

from phase_id_xcorr.ml.dataset_io import read_json, save_split_npz, write_json
from phase_id_xcorr.ml.oh5_inference import run_oh5_sample_inference
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
            arr[3:12, 3:12] = 0.8
        elif cls == 1:
            arr[:, ::3] = 0.75
        else:
            rr = np.arange(h)[:, None]
            cc = np.arange(w)[None, :]
            arr[((rr - h // 2) ** 2 + (cc - w // 2) ** 2) < 20] = 0.9
        arr += rng.normal(0.0, 0.01, size=(h, w)).astype(np.float32)
        patterns[i] = np.clip(arr, 0.0, 1.0)
    return patterns, labels


def _write_minimal_oh5(path: Path, *, patterns: np.ndarray, ci: np.ndarray, fit: np.ndarray, iq: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_pixels, h, w = patterns.shape
    with h5py.File(path, "w") as h5:
        scan = h5.create_group("Scan 1")
        ebsd = scan.create_group("EBSD")
        header = ebsd.create_group("Header")
        data = ebsd.create_group("Data")
        header.create_dataset("nColumns", data=np.asarray([total_pixels], dtype=np.int32))
        header.create_dataset("nRows", data=np.asarray([1], dtype=np.int32))
        data.create_dataset("Pattern", data=np.round(patterns * 255.0).astype(np.uint8).reshape(total_pixels, h, w))
        data.create_dataset("CI", data=np.asarray(ci, dtype=np.float32))
        data.create_dataset("Fit", data=np.asarray(fit, dtype=np.float32))
        data.create_dataset("IQ", data=np.asarray(iq, dtype=np.float32))


def test_run_oh5_sample_inference_supports_relative_and_absolute_paths(tmp_path: Path) -> None:
    repo_root = tmp_path
    out_dataset = tmp_path / "dataset"
    out_dataset.mkdir(parents=True, exist_ok=True)

    train_p, train_y = _make_patterns(36, 16, 16, seed=1)
    val_p, val_y = _make_patterns(18, 16, 16, seed=2)
    test_p, test_y = _make_patterns(18, 16, 16, seed=3)

    save_split_npz(out_dataset / "train.npz", patterns=train_p, labels=train_y, sample_ids=[f"tr_{i}" for i in range(len(train_y))])
    save_split_npz(out_dataset / "val.npz", patterns=val_p, labels=val_y, sample_ids=[f"va_{i}" for i in range(len(val_y))])
    save_split_npz(out_dataset / "test.npz", patterns=test_p, labels=test_y, sample_ids=[f"te_{i}" for i in range(len(test_y))])

    manifest = {
        "schema_version": "phase_id_xcorr.ml_dataset_manifest.v1",
        "phase_to_label": {"Al": 0, "Ni": 1, "Cu": 2},
        "preprocessing_policy": {
            "resize_hw": [16, 16],
            "apply_circular_mask": False,
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
        "seed": 7,
        "device": "cpu",
        "amp": False,
        "batch_size": 12,
        "epochs": 3,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "input": {
            "resize_hw": [16, 16],
            "apply_circular_mask": False,
            "normalize": {"mean": [0.0], "std": [1.0]},
        },
        "model": {
            "family": "simple_cnn",
            "width": 8,
            "in_chans": 1,
        },
    }
    train_cfg_path = tmp_path / "train.yml"
    train_cfg_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")
    train_classifier(config_path=train_cfg_path, repo_root=repo_root, debug=True)

    ni_patterns = np.stack([train_p[idx] for idx, label in enumerate(train_y) if label == 1][:6], axis=0)
    ci = np.asarray([0.9, 0.92, 0.3, 0.95, 0.85, 0.88], dtype=np.float32)
    fit = np.asarray([0.4, 0.5, 0.6, 1.4, 0.45, 0.48], dtype=np.float32)
    iq = np.asarray([120, 125, 115, 122, 128, 121], dtype=np.float32)

    data_root = tmp_path / "incoming_data"
    rel_scan_path = data_root / "Different_Condition" / "Ni" / "WD14.oh5"
    abs_scan_path = data_root / "Different_Condition" / "Ni" / "WD18.oh5"
    _write_minimal_oh5(rel_scan_path, patterns=ni_patterns, ci=ci, fit=fit, iq=iq)
    _write_minimal_oh5(abs_scan_path, patterns=ni_patterns, ci=ci, fit=fit, iq=iq)

    inference_cfg = {
        "schema_version": "phase_id_xcorr.ml_oh5_inference.v1",
        "run_dir": "suite_run/simple_cnn_w8",
        "checkpoint": "best_checkpoint.pt",
        "device": "cpu",
        "output_dir": "reports/ml/oh5_inference/test_run",
        "input_root": str(data_root.resolve()),
        "sampling": {
            "samples_per_scan": 2,
            "seed": 11,
        },
        "quality_filters": {
            "expression": "CI > 0.5 && Fit < 1.0",
        },
        "scans": [
            {
                "file": "Different_Condition/Ni/WD14.oh5",
                "expected_phase": "Ni",
            },
            {
                "file": str(abs_scan_path.resolve()),
                "expected_phase": "Ni",
            },
        ],
    }
    inference_cfg_path = tmp_path / "infer.yml"
    inference_cfg_path.write_text(yaml.safe_dump(inference_cfg, sort_keys=False), encoding="utf-8")

    result = run_oh5_sample_inference(config_path=inference_cfg_path, repo_root=repo_root, debug=True)

    assert result.processed_scans == 2
    assert result.sampled_patterns == 4
    assert result.labeled_accuracy is not None
    assert 0.0 <= result.labeled_accuracy <= 1.0

    summary = read_json(result.summary_json)
    assert summary["processed_scans"] == 2
    assert summary["sampled_patterns"] == 4
    assert summary["labeled_patterns"] == 4

    pattern_lines = result.patterns_csv.read_text(encoding="utf-8").strip().splitlines()
    scan_lines = result.scans_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(pattern_lines) == 5
    assert len(scan_lines) == 3
    assert "WD14" in result.summary_md.read_text(encoding="utf-8")
