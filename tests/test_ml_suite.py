from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from phase_id_xcorr.ml.dataset_io import save_split_npz, write_json, read_json
from phase_id_xcorr.ml.suite import run_benchmark_suite


def _make_patterns(n: int, h: int, w: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    patterns = np.zeros((n, h, w), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        cls = i % 3
        labels[i] = cls
        arr = np.zeros((h, w), dtype=np.float32)
        arr[:, cls :: 3] = 0.8
        arr += rng.normal(0.0, 0.02, size=(h, w)).astype(np.float32)
        patterns[i] = np.clip(arr, 0.0, 1.0)
    return patterns, labels


def _prepare_dataset(tmp_path: Path) -> Path:
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)

    train_p, train_y = _make_patterns(21, 20, 20, seed=10)
    val_p, val_y = _make_patterns(9, 20, 20, seed=11)
    test_p, test_y = _make_patterns(9, 20, 20, seed=12)

    save_split_npz(ds_dir / "train.npz", patterns=train_p, labels=train_y, sample_ids=[f"tr_{i}" for i in range(len(train_y))])
    save_split_npz(ds_dir / "val.npz", patterns=val_p, labels=val_y, sample_ids=[f"va_{i}" for i in range(len(val_y))])
    save_split_npz(ds_dir / "test.npz", patterns=test_p, labels=test_y, sample_ids=[f"te_{i}" for i in range(len(test_y))])

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
    return manifest_path


def test_ml_benchmark_suite_smoke(tmp_path: Path) -> None:
    manifest_path = _prepare_dataset(tmp_path)

    base_cfg = {
        "dataset_manifest_path": str(manifest_path),
        "output_dir": "outputs/base",
        "seed": 2,
        "device": "cpu",
        "amp": False,
        "batch_size": 8,
        "epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "input": {
            "resize_hw": [24, 24],
            "apply_circular_mask": True,
            "normalize": {"mean": [0.5], "std": [0.25]},
        },
        "model": {
            "family": "simple_cnn",
            "width": 8,
            "in_chans": 1,
        },
    }
    base_cfg_path = tmp_path / "base_train.yml"
    base_cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=False), encoding="utf-8")

    suite_cfg = {
        "output_root": "bench_out",
        "base_train_config": str(base_cfg_path),
        "experiments": [
            {
                "name": "run_a",
                "overrides": [
                    "output_dir=bench_out/run_a",
                    "seed=2",
                ],
            },
            {
                "name": "run_b",
                "overrides": [
                    "output_dir=bench_out/run_b",
                    "seed=3",
                ],
            },
        ],
    }
    suite_cfg_path = tmp_path / "suite.yml"
    suite_cfg_path.write_text(yaml.safe_dump(suite_cfg, sort_keys=False), encoding="utf-8")

    result = run_benchmark_suite(
        suite_config_path=suite_cfg_path,
        repo_root=tmp_path,
        debug=True,
        strict=True,
    )

    assert result.summary_json.exists()
    assert result.summary_md.exists()
    assert result.manifest_json.exists()

    payload = read_json(result.summary_json)
    assert payload["runs_total"] == 2
    assert payload["runs_completed"] == 2
    assert payload["runs_failed"] == 0
    assert payload["timing"]["total_elapsed_seconds"] >= 0.0

    manifest = read_json(result.manifest_json)
    assert manifest["sanity_checks"]["suite_summary_written"] is True
    assert manifest["artifacts"]["event_log_jsonl"].endswith("events.jsonl")
    assert manifest["artifacts"]["suite_report_html"].endswith("suite_report.html")
    events_path = tmp_path / manifest["artifacts"]["event_log_jsonl"]
    html_path = tmp_path / manifest["artifacts"]["suite_report_html"]
    assert events_path.exists()
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "href='run_a/report.json'" in html_text
    assert "href='run_b/report.json'" in html_text
