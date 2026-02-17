from __future__ import annotations

import json
from pathlib import Path

import pytest

from phase_id_xcorr.intake.g0_validator import validate_data_packet


@pytest.fixture()
def temp_packet(tmp_path: Path) -> Path:
    packet = tmp_path / "packet"
    (packet / "experimental_patterns").mkdir(parents=True)
    (packet / "simulated_patterns" / "assume_fe_bcc").mkdir(parents=True)
    (packet / "simulated_patterns" / "assume_fe3o4_magnetite").mkdir(parents=True)
    (packet / "simulated_patterns" / "assume_feo_wustite").mkdir(parents=True)
    (packet / "scan_files").mkdir(parents=True)

    # Minimal image placeholders
    (packet / "experimental_patterns" / "fe_bcc_Ori_1.png").write_bytes(b"x")
    (packet / "simulated_patterns" / "assume_fe_bcc" / "fe_bcc_Ori_1.png").write_bytes(b"x")
    (packet / "simulated_patterns" / "assume_fe3o4_magnetite" / "fe_bcc_Ori_1.png").write_bytes(b"x")
    (packet / "simulated_patterns" / "assume_feo_wustite" / "fe_bcc_Ori_1.png").write_bytes(b"x")

    exp = {
        "records": [
            {
                "record_id": "r001",
                "image_file": "experimental_patterns/fe_bcc_Ori_1.png",
                "true_phase": "fe_bcc",
                "orientation_angles_degrees": {"phi1": 10.0, "PHI": 20.0, "phi2": 30.0},
                "image_info": {"bit_depth": 16, "height": 230, "width": 230, "format": "png"},
                "label_source": "manual",
            }
        ]
    }

    sim = {
        "required_assumed_phases": ["fe_bcc", "fe3o4_magnetite", "feo_wustite"],
        "records": [
            {
                "record_id": "r001",
                "experimental_image": "experimental_patterns/fe_bcc_Ori_1.png",
                "true_phase": "fe_bcc",
                "simulated_candidates": [
                    {
                        "assumed_phase": "fe_bcc",
                        "simulated_image": "simulated_patterns/assume_fe_bcc/fe_bcc_Ori_1.png",
                        "candidate_angles_degrees": {"phi1": 10.0, "PHI": 20.0, "phi2": 30.0},
                        "indexing_status": "ok",
                        "is_fallback_orientation": False,
                    },
                    {
                        "assumed_phase": "fe3o4_magnetite",
                        "simulated_image": "simulated_patterns/assume_fe3o4_magnetite/fe_bcc_Ori_1.png",
                        "candidate_angles_degrees": {"phi1": 40.0, "PHI": 50.0, "phi2": 60.0},
                        "indexing_status": "ok",
                        "is_fallback_orientation": False,
                    },
                    {
                        "assumed_phase": "feo_wustite",
                        "simulated_image": "simulated_patterns/assume_feo_wustite/fe_bcc_Ori_1.png",
                        "candidate_angles_degrees": {"phi1": 70.0, "PHI": 80.0, "phi2": 90.0},
                        "indexing_status": "ok",
                        "is_fallback_orientation": False,
                    },
                ],
            }
        ],
    }

    scan = {
        "scan_records": [
            {
                "scan_id": "s001",
                "scan_files": {
                    "assume_fe_bcc": "scan_files/a.oh5",
                    "assume_fe3o4_magnetite": "scan_files/b.oh5",
                    "assume_feo_wustite": "scan_files/c.oh5",
                },
                "grid_info": {
                    "nx": 2,
                    "ny": 2,
                    "pattern_height": 2,
                    "pattern_width": 2,
                    "flat_index_rule": "row_major_y_times_nx_plus_x",
                },
                "manual_check_points": [
                    {"point_id": f"p{i}", "x": 0, "y": 0, "expected_phase": "fe_bcc"}
                    for i in range(10)
                ],
            }
        ]
    }

    proc = {
        "settings": {
            "dtype_target": "uint16_to_float32_0_1",
            "normalization_method": "minmax_inside_mask",
            "mask_method": "circular",
            "mask_parameters": {"center_mode": "image_center", "radius_mode": "max_inscribed"},
            "resize_policy": "none",
            "intensity_clip_policy": "none",
            "ncc_variant": "masked_ncc",
            "euler_convention": "bunge_zxz",
            "angle_units": "degrees",
            "exp_sim_alignment_policy": "require_same_shape_and_mask_frame",
        }
    }

    (packet / "01_experimental_patterns_template.json").write_text(json.dumps(exp), encoding="utf-8")
    (packet / "02_simulated_patterns_template.json").write_text(json.dumps(sim), encoding="utf-8")
    (packet / "03_scan_files_template.json").write_text(json.dumps(scan), encoding="utf-8")
    (packet / "04_processing_template.json").write_text(json.dumps(proc), encoding="utf-8")

    # Build tiny OH5 triad
    h5py = pytest.importorskip("h5py")

    def build_oh5(path: Path) -> None:
        with h5py.File(path, "w") as h5f:
            scan_grp = h5f.create_group("Scan1")
            ebsd = scan_grp.create_group("EBSD")
            header = ebsd.create_group("Header")
            data = ebsd.create_group("Data")
            header.create_dataset("nColumns", data=[2])
            header.create_dataset("nRows", data=[2])
            header.create_dataset("Pattern Height", data=[2])
            header.create_dataset("Pattern Width", data=[2])
            data.create_dataset("Pattern", data=[[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]])

    build_oh5(packet / "scan_files" / "a.oh5")
    build_oh5(packet / "scan_files" / "b.oh5")
    build_oh5(packet / "scan_files" / "c.oh5")

    return packet


def test_validate_data_packet_go(temp_packet: Path) -> None:
    result = validate_data_packet(temp_packet)
    assert result.gate_status == "GO"
    assert result.counts["errors"] == 0


def test_validate_data_packet_missing_simulated_file(temp_packet: Path) -> None:
    (temp_packet / "simulated_patterns" / "assume_fe_bcc" / "fe_bcc_Ori_1.png").unlink()

    result = validate_data_packet(temp_packet)
    assert result.gate_status == "HOLD"
    assert result.counts["errors"] >= 1
    assert any(f.code == "missing_file" for f in result.findings)
