from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import yaml

from phase_id_xcorr.ml.dataset_builder import prepare_ml_dataset
from phase_id_xcorr.ml.dataset_io import read_json


def _write_fixture(oh5_path: Path, csv_path: Path) -> None:
    nx, ny = 4, 3
    n = nx * ny
    h, w = 16, 16

    with h5py.File(oh5_path, "w") as f:
        f.create_dataset("Manufacturer", data=np.asarray([b"EDAX"]))
        f.create_dataset("Version", data=np.asarray([b"TEST"]))
        g = f.create_group("scan")
        ebsd = g.create_group("EBSD")
        header = ebsd.create_group("Header")
        data = ebsd.create_group("Data")

        header.create_dataset("nColumns", data=np.asarray([nx], dtype=np.int32))
        header.create_dataset("nRows", data=np.asarray([ny], dtype=np.int32))

        patt = np.zeros((n, h, w), dtype=np.uint16)
        ci = np.zeros((n,), dtype=np.float32)
        iq = np.zeros((n,), dtype=np.float32)
        fit = np.zeros((n,), dtype=np.float32)
        valid = np.ones((n,), dtype=np.int8)

        for i in range(n):
            patt[i, 4:12, 4:12] = np.uint16(12000 + 1000 * (i % 3))
            ci[i] = 0.8
            iq[i] = 40.0
            fit[i] = 1.2

        ci[1] = 0.01  # quality reject
        valid[5] = 0  # quality reject

        data.create_dataset("Pattern", data=patt)
        data.create_dataset("CI", data=ci)
        data.create_dataset("IQ", data=iq)
        data.create_dataset("Fit", data=fit)
        data.create_dataset("Valid", data=valid)

    phases = ["fe_bcc", "fe3o4_magnetite", "feo_wustite"]
    lines = ["sample_id,x,y,phase_name\n"]
    for y in range(ny):
        for x in range(nx):
            idx = y * nx + x
            lines.append(f"r{idx},{x},{y},{phases[idx % 3]}\n")
    csv_path.write_text("".join(lines), encoding="utf-8")


def test_prepare_ml_dataset_end_to_end(tmp_path: Path) -> None:
    repo_root = tmp_path
    oh5 = tmp_path / "scan.oh5"
    labels_csv = tmp_path / "labels.csv"
    _write_fixture(oh5, labels_csv)

    cfg = {
        "output_dir": "reports/ml/dataset_test",
        "strict_pattern_presence": True,
        "target_pattern_hw": [32, 32],
        "phase_labels": [
            {"name": "fe_bcc", "label": 0},
            {"name": "fe3o4_magnetite", "label": 1},
            {"name": "feo_wustite", "label": 2},
        ],
        "label_csv": {
            "sample_id_col": "sample_id",
            "x_col": "x",
            "y_col": "y",
            "phase_name_col": "phase_name",
            "flat_index_col": "",
            "phase_label_col": "",
        },
        "quality_filters": {
            "confidence_index_min": 0.1,
            "fit_max": 2.0,
            "valid_required": True,
        },
        "split": {
            "train": 0.6,
            "val": 0.2,
            "test": 0.2,
            "seed": 11,
            "stratified": True,
        },
        "sources": [
            {
                "scan_id": "s001",
                "oh5_path": str(oh5),
                "labels_csv_path": str(labels_csv),
            }
        ],
    }

    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = prepare_ml_dataset(
        config_path=cfg_path,
        repo_root=repo_root,
        debug=True,
    )

    assert result.manifest_path.exists()
    assert result.records_csv.exists()
    assert result.split_npz["train"].exists()
    assert result.split_npz["val"].exists()
    assert result.split_npz["test"].exists()

    manifest = read_json(result.manifest_path)
    assert manifest["num_samples_total"] == 10  # 12 input - 2 filtered
    assert manifest["split_counts"]["train"] > 0
    assert manifest["split_counts"]["val"] > 0
    assert manifest["split_counts"]["test"] > 0
    assert manifest["sanity_checks"]["phase_label_mapping_unique"] is True
    assert manifest["sanity_checks"]["all_records_assigned_split"] is True
    assert "event_log_jsonl" in manifest["artifacts"]
    event_log = tmp_path / manifest["artifacts"]["event_log_jsonl"]
    assert event_log.exists()
    lines = [line for line in event_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines
