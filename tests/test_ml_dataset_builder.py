from __future__ import annotations

import csv
import json
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
        data.create_dataset("Phi1", data=np.linspace(0.0, 110.0, num=n, dtype=np.float32))
        data.create_dataset("Phi", data=np.linspace(5.0, 80.0, num=n, dtype=np.float32))
        data.create_dataset("Phi2", data=np.linspace(10.0, 120.0, num=n, dtype=np.float32))

    phases = ["fe_bcc", "fe3o4_magnetite", "feo_wustite"]
    lines = ["sample_id,x,y,phase_name\n"]
    for y in range(ny):
        for x in range(nx):
            idx = y * nx + x
            lines.append(f"r{idx},{x},{y},{phases[idx % 3]}\n")
    csv_path.write_text("".join(lines), encoding="utf-8")


def _write_single_phase_fixture(
    oh5_path: Path,
    *,
    ci_bad_idx: int | None = None,
    ci_bad_indices: list[int] | None = None,
    pattern_base: int,
    euler_unit: str = "degree",
) -> None:
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
        ci = np.full((n,), 0.9, dtype=np.float32)
        iq = np.full((n,), 35.0, dtype=np.float32)
        fit = np.full((n,), 1.0, dtype=np.float32)
        valid = np.ones((n,), dtype=np.int8)

        for i in range(n):
            patt[i, 3:13, 3:13] = np.uint16(pattern_base + 500 * (i % 3))

        bad_indices = list(ci_bad_indices or [])
        if ci_bad_idx is not None:
            bad_indices.append(int(ci_bad_idx))
        for idx in bad_indices:
            ci[int(idx)] = 0.01  # quality reject

        data.create_dataset("Pattern", data=patt)
        data.create_dataset("CI", data=ci)
        data.create_dataset("IQ", data=iq)
        data.create_dataset("Fit", data=fit)
        data.create_dataset("Valid", data=valid)
        if euler_unit == "degree":
            data.create_dataset("Phi1", data=np.linspace(0.0, 110.0, num=n, dtype=np.float32))
            data.create_dataset("Phi", data=np.linspace(5.0, 80.0, num=n, dtype=np.float32))
            data.create_dataset("Phi2", data=np.linspace(10.0, 120.0, num=n, dtype=np.float32))
        elif euler_unit == "radian":
            data.create_dataset("Phi1", data=np.linspace(0.0, 3.0, num=n, dtype=np.float32))
            data.create_dataset("Phi", data=np.linspace(0.1, 1.5, num=n, dtype=np.float32))
            data.create_dataset("Phi2", data=np.linspace(0.2, 2.5, num=n, dtype=np.float32))
        else:
            raise ValueError(f"Unsupported euler_unit {euler_unit}")


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
    assert manifest["input_mode"] == "oh5_csv_labels"
    assert manifest["num_samples_total"] == 10  # 12 input - 2 filtered
    assert manifest["split_counts"]["train"] > 0
    assert manifest["split_counts"]["val"] > 0
    assert manifest["split_counts"]["test"] > 0
    assert manifest["sanity_checks"]["phase_label_mapping_unique"] is True
    assert manifest["sanity_checks"]["all_records_assigned_split"] is True
    assert "event_log_jsonl" in manifest["artifacts"]
    assert manifest["orientation_exports"]["counts"]["qualified_records"] == 10
    assert manifest["orientation_exports"]["counts"]["selected_records"] == 10
    event_log = tmp_path / manifest["artifacts"]["event_log_jsonl"]
    assert event_log.exists()
    summary_html = tmp_path / manifest["artifacts"]["summary_html"]
    assert summary_html.exists()
    qualified_csv = tmp_path / manifest["artifacts"]["qualified_orientations_csv"]
    qualified_json = tmp_path / manifest["artifacts"]["qualified_orientations_json"]
    selected_csv = tmp_path / manifest["artifacts"]["selected_orientations_csv"]
    selected_json = tmp_path / manifest["artifacts"]["selected_orientations_json"]
    ipf_index = tmp_path / manifest["artifacts"]["ipf_index_json"]
    assert qualified_csv.exists()
    assert qualified_json.exists()
    assert selected_csv.exists()
    assert selected_json.exists()
    assert ipf_index.exists()
    selected_payload = json.loads(selected_json.read_text(encoding="utf-8"))
    assert selected_payload["euler_convention"] == "Bunge ZXZ"
    assert selected_payload["euler_export_unit"] == "degree"
    assert selected_payload["record_count"] == 10
    lines = [line for line in event_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines


def test_prepare_ml_dataset_single_phase_scan_map_mode(tmp_path: Path) -> None:
    repo_root = tmp_path
    oh5_a = tmp_path / "scan_a.oh5"
    oh5_b = tmp_path / "scan_b.oh5"
    _write_single_phase_fixture(oh5_a, ci_bad_idx=1, pattern_base=10000)
    _write_single_phase_fixture(oh5_b, ci_bad_idx=9, pattern_base=20000)

    cfg = {
        "input_mode": "single_phase_scan_map",
        "output_dir": "reports/ml/dataset_single_phase_test",
        "strict_pattern_presence": True,
        "target_pattern_hw": [24, 24],
        "phase_labels": [
            {"name": "fe_bcc", "label": 0},
            {"name": "feo_wustite", "label": 1},
        ],
        "quality_filters": {
            "confidence_index_min": 0.1,
            "fit_max": 2.0,
            "valid_required": True,
        },
        "split": {
            "train": 0.6,
            "val": 0.2,
            "test": 0.2,
            "seed": 17,
            "stratified": True,
        },
        "sources": [
            {
                "scan_id": "s001",
                "oh5_path": str(oh5_a),
                "phase_name": "fe_bcc",
            },
            {
                "scan_id": "s002",
                "oh5_path": str(oh5_b),
                "phase_label": 1,
            },
        ],
    }

    cfg_path = tmp_path / "config_single_phase.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = prepare_ml_dataset(
        config_path=cfg_path,
        repo_root=repo_root,
        debug=True,
    )

    manifest = read_json(result.manifest_path)
    assert manifest["input_mode"] == "single_phase_scan_map"
    assert manifest["source_mode_counts"]["single_phase_scan_map"] == 2
    assert manifest["raw_input_rows_total"] == 24
    assert manifest["num_samples_total"] == 22  # 24 input - 2 filtered by CI
    assert manifest["accepted_per_phase"]["fe_bcc"] == 11
    assert manifest["accepted_per_phase"]["feo_wustite"] == 11
    assert manifest["split_counts"]["train"] > 0
    assert manifest["split_counts"]["val"] > 0
    assert manifest["split_counts"]["test"] > 0
    assert manifest["orientation_exports"]["counts"]["qualified_records"] == 22
    assert manifest["orientation_exports"]["counts"]["selected_records"] == 22

    with result.records_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert all(row["source_mode"] == "single_phase_scan_map" for row in rows)
    assert all(row["labels_csv_path"] == "" for row in rows)
    assert {row["phase_name"] for row in rows} == {"fe_bcc", "feo_wustite"}
    assert all(row["euler_export_unit"] == "degree" for row in rows)


def test_prepare_ml_dataset_single_phase_mode_inferred_from_sources(tmp_path: Path) -> None:
    repo_root = tmp_path
    oh5 = tmp_path / "scan_single.oh5"
    _write_single_phase_fixture(oh5, ci_bad_idx=0, pattern_base=9000, euler_unit="radian")

    cfg = {
        "output_dir": "reports/ml/dataset_single_phase_infer",
        "strict_pattern_presence": True,
        "phase_labels": [
            {"name": "fe_bcc", "label": 0},
        ],
        "quality_filters": {
            "confidence_index_min": 0.1,
            "fit_max": 2.0,
            "valid_required": True,
        },
        "split": {
            "train": 0.6,
            "val": 0.2,
            "test": 0.2,
            "seed": 3,
            "stratified": True,
        },
        "sources": [
            {
                "scan_id": "s001",
                "oh5_path": str(oh5),
                "phase_name": "fe_bcc",
            }
        ],
    }

    cfg_path = tmp_path / "config_single_phase_infer.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = prepare_ml_dataset(
        config_path=cfg_path,
        repo_root=repo_root,
        debug=True,
    )
    manifest = read_json(result.manifest_path)
    assert manifest["input_mode"] == "single_phase_scan_map"
    assert manifest["num_samples_total"] == 11
    assert manifest["orientation_exports"]["source_units_by_scan"] == {"s001": "radian"}


def test_prepare_ml_dataset_v3_list_of_files_with_caps_and_provenance(tmp_path: Path) -> None:
    repo_root = tmp_path
    data_dir = tmp_path / "scan_folder"
    data_dir.mkdir(parents=True, exist_ok=True)
    oh5_a = data_dir / "scan_a__fe_bcc.oh5"
    oh5_b = data_dir / "scan_b__feo_wustite.oh5"
    _write_single_phase_fixture(oh5_a, ci_bad_idx=0, pattern_base=10000)
    _write_single_phase_fixture(oh5_b, ci_bad_idx=1, pattern_base=15000)

    cfg = {
        "schema_version": "phase_id_xcorr.ml_dataset_prep.v3",
        "output_dir": "reports/ml/dataset_v3",
        "data_source_folder": str(data_dir),
        "allow_filename_phase_fallback": True,
        "phase_to_label": {"fe_bcc": 0, "feo_wustite": 1},
        "quality_filters": {"expression": "CI > 0.05 && Fit < 2.0"},
        "preprocessing": {"resize_hw": [20, 20], "apply_circular_mask": True, "normalize_mode": "per_pattern_minmax"},
        "split": {
            "train": 0.6,
            "val": 0.2,
            "test": 0.2,
            "seed": 5,
            "stratified": True,
            "group_key": "scan_id",
            "max_val_samples": 3,
            "max_test_samples": 4,
        },
        "listOfFiles": [
            {"file": "scan_a__fe_bcc.oh5", "scan_id": "scan_a"},
            {"file": "scan_b__feo_wustite.oh5", "scan_id": "scan_b"},
        ],
    }

    cfg_path = tmp_path / "config_v3.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = prepare_ml_dataset(config_path=cfg_path, repo_root=repo_root, debug=True)
    manifest = read_json(result.manifest_path)

    assert manifest["split_policy"]["group_key"] == "scan_id"
    assert manifest["split_counts"]["val"] <= 3
    assert manifest["split_counts"]["test"] <= 4
    assert manifest["preprocessing_policy"]["resize_hw"] == [20, 20]
    assert manifest["quality_filters"]["resolved_expression"]
    assert manifest["provenance"]["input_file_hashes"]
    assert manifest["artifacts"]["resolved_config_json"].endswith("resolved_config.json")
    assert manifest["artifacts"]["ipf_index_json"].endswith("ipf_index.json")


def test_prepare_ml_dataset_balances_single_phase_sources_to_min_count(tmp_path: Path) -> None:
    repo_root = tmp_path
    oh5_a = tmp_path / "al.oh5"
    oh5_b = tmp_path / "cu.oh5"
    oh5_c = tmp_path / "ni.oh5"
    _write_single_phase_fixture(oh5_a, ci_bad_indices=[10, 11], pattern_base=10000)
    _write_single_phase_fixture(oh5_b, ci_bad_indices=[11], pattern_base=15000)
    _write_single_phase_fixture(oh5_c, ci_bad_indices=[9, 10, 11], pattern_base=20000)

    cfg = {
        "input_mode": "single_phase_scan_map",
        "output_dir": "reports/ml/dataset_balanced_single_phase_test",
        "strict_pattern_presence": True,
        "target_pattern_hw": [24, 24],
        "phase_labels": [
            {"name": "Al", "label": 0},
            {"name": "Cu", "label": 1},
            {"name": "Ni", "label": 2},
        ],
        "quality_filters": {
            "confidence_index_min": 0.1,
            "fit_max": 2.0,
            "valid_required": True,
        },
        "phase_balancing": {
            "equalize_to_min_count": True,
        },
        "split": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
            "seed": 23,
            "stratified": True,
        },
        "sources": [
            {
                "scan_id": "al_scan",
                "oh5_path": str(oh5_a),
                "phase_name": "Al",
            },
            {
                "scan_id": "cu_scan",
                "oh5_path": str(oh5_b),
                "phase_name": "Cu",
            },
            {
                "scan_id": "ni_scan",
                "oh5_path": str(oh5_c),
                "phase_name": "Ni",
            },
        ],
    }

    cfg_path = tmp_path / "config_balanced.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = prepare_ml_dataset(config_path=cfg_path, repo_root=repo_root, debug=True)
    manifest = read_json(result.manifest_path)

    assert manifest["accepted_per_phase"] == {"Al": 10, "Cu": 11, "Ni": 9}
    assert manifest["selected_per_phase"] == {"Al": 9, "Cu": 9, "Ni": 9}
    assert manifest["phase_balancing"]["enabled"] is True
    assert manifest["phase_balancing"]["strategy"] == "min_phase_count"
    assert manifest["phase_balancing"]["target_per_phase"] == 9
    assert manifest["num_samples_total"] == 27
    assert manifest["split_phase_counts"]["train"] == {"Al": 7, "Cu": 7, "Ni": 7}
    assert manifest["split_phase_counts"]["val"] == {"Al": 1, "Cu": 1, "Ni": 1}
    assert manifest["split_phase_counts"]["test"] == {"Al": 1, "Cu": 1, "Ni": 1}
    assert manifest["orientation_exports"]["counts"]["qualified_records"] == 30
    assert manifest["orientation_exports"]["counts"]["selected_records"] == 27
    assert len(manifest["orientation_exports"]["ipf_plots"]) == 12

    ipf_index = read_json(tmp_path / manifest["artifacts"]["ipf_index_json"])
    assert len(ipf_index["plots"]) == 12
    expected_paths = {
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/qualified/Al_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/qualified/Cu_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/qualified/Ni_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/selected/train/Al_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/selected/train/Cu_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/selected/train/Ni_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/selected/val/Al_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/selected/val/Cu_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/selected/val/Ni_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/selected/test/Al_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/selected/test/Cu_ipf.png",
        "reports/ml/dataset_balanced_single_phase_test/orientation_exports/ipf/selected/test/Ni_ipf.png",
    }
    actual_paths = {plot["path"] for plot in ipf_index["plots"]}
    assert actual_paths == expected_paths
    for rel in actual_paths:
        assert (tmp_path / rel).exists()
