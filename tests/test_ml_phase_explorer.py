from __future__ import annotations

from pathlib import Path

import yaml

from phase_id_xcorr.ml.dataset_io import read_json
from phase_id_xcorr.ml.phase_explorer import (
    build_intensity_mask,
    cdf_from_counts,
    export_phase_explorer_artifacts,
    histogram,
    load_explorer_dataset,
)
import h5py
import numpy as np


def _write_single_phase_fixture(
    oh5_path: Path,
    *,
    ci_bad_idx: int,
    pattern_base: int,
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

        ci[ci_bad_idx] = 0.01

        data.create_dataset("Pattern", data=patt)
        data.create_dataset("CI", data=ci)
        data.create_dataset("IQ", data=iq)
        data.create_dataset("Fit", data=fit)
        data.create_dataset("Valid", data=valid)


def test_histogram_and_cdf_shapes() -> None:
    values = np.asarray([0.0, 0.2, 0.4, 0.6, 0.9], dtype=np.float32)
    counts, edges = histogram(values, bins=5, x_min=0.0, x_max=1.0)
    cdf = cdf_from_counts(np.cumsum(counts))
    assert counts.shape == (5,)
    assert edges.shape == (6,)
    assert cdf[-1] == 1.0


def test_build_intensity_mask_union_ranges() -> None:
    pattern = np.asarray([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)
    mask = build_intensity_mask(pattern, [(0.15, 0.25), (0.85, 0.95)])
    assert mask.tolist() == [[False, True], [False, True]]


def test_load_explorer_dataset_single_phase(tmp_path: Path) -> None:
    repo_root = tmp_path
    oh5_a = tmp_path / "scan_a__fe_bcc.oh5"
    oh5_b = tmp_path / "scan_b__feo_wustite.oh5"
    _write_single_phase_fixture(oh5_a, ci_bad_idx=0, pattern_base=10000)
    _write_single_phase_fixture(oh5_b, ci_bad_idx=1, pattern_base=15000)

    cfg = {
        "schema_version": "phase_id_xcorr.ml_dataset_prep.v3",
        "data_source_folder": str(tmp_path),
        "allow_filename_phase_fallback": True,
        "phase_to_label": {"fe_bcc": 0, "feo_wustite": 1},
        "listOfFiles": [
            {"file": "scan_a__fe_bcc.oh5", "scan_id": "scan_a"},
            {"file": "scan_b__feo_wustite.oh5", "scan_id": "scan_b"},
        ],
    }
    cfg_path = tmp_path / "explorer.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    ds = load_explorer_dataset(config_path=cfg_path, repo_root=repo_root)
    assert set(ds.phase_names) == {"fe_bcc", "feo_wustite"}
    assert ds.pattern_count("fe_bcc") == 12
    assert ds.pattern_count("feo_wustite") == 12
    assert ds.phases["fe_bcc"].intensity_values.size > 0
    assert "CI" in ds.phases["fe_bcc"].scalar_fields
    assert ds.phases["fe_bcc"].scalar_fields["CI"].size == 12


def test_export_phase_explorer_artifacts_writes_pngs_and_json(tmp_path: Path) -> None:
    repo_root = tmp_path
    phase_names = ["Al", "Cu", "Ni"]
    cfg = {
        "schema_version": "phase_id_xcorr.ml_dataset_prep.v3",
        "output_dir": "reports/ml/datasets/test_explorer",
        "data_source_folder": str(tmp_path),
        "phase_to_label": {"Al": 0, "Cu": 1, "Ni": 2},
        "listOfFiles": [],
        "explorer": {
            "intensity_plot": {
                "bins": 16,
                "x_min": 0,
                "x_max": 65535,
                "y_min": 0,
                "y_max": 3000,
                "title_template": "{phase}",
                "x_label": "Intensity",
                "y_label": "Pixel count",
                "color": "#123456",
                "edge_color": "#123456",
                "bar_line_width": 0.6,
                "show_cdf": False,
            },
            "attribute_plot": {
                "bins": 8,
                "y_min": 0,
                "field_ranges": {"CI": [0, 1], "IQ": [0, 100], "Fit": [0, 2]},
                "field_y_ranges": {"CI": [0, 20]},
                "title_template": "{phase} {attribute}",
                "x_label_template": "{attribute}",
                "y_label": "Pixel count",
                "color": "#654321",
                "edge_color": "#654321",
                "bar_line_width": 0.5,
            },
            "export": {
                "attributes": ["CI", "IQ", "Fit"],
                "dpi": 120,
                "figure_size_inches": [4, 3],
                "font_family": "Arial",
                "font_size": 14,
                "title_font_size": 16,
                "label_font_size": 15,
                "tick_label_size": 13,
                "x_tick_label_size": 12,
                "y_tick_label_size": 11,
                "tick_width": 1.0,
                "tick_length": 5,
                "minor_tick_width": 0.8,
                "minor_tick_length": 3,
                "tick_direction": "out",
                "x_tick_rotation": 15,
                "y_tick_rotation": 0,
                "spine_line_width": 1.1,
                "grid_line_width": 0.7,
                "grid_alpha": 0.2,
                "title_pad": 9,
                "label_pad": 7,
                "figure_facecolor": "white",
                "axes_facecolor": "white",
                "show_minor_ticks": True,
                "savefig_pad_inches": 0.2,
            },
        },
    }

    file_rows = []
    for idx, phase_name in enumerate(phase_names):
        oh5_path = tmp_path / f"{phase_name}.oh5"
        _write_single_phase_fixture(oh5_path, ci_bad_idx=idx, pattern_base=10000 + (idx * 2000))
        file_rows.append({"file": oh5_path.name, "scan_id": phase_name.lower(), "phase_name": phase_name})
    cfg["listOfFiles"] = file_rows

    cfg_path = tmp_path / "explorer_export.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    ds = load_explorer_dataset(config_path=cfg_path, repo_root=repo_root)
    manifest_path = export_phase_explorer_artifacts(dataset=ds, repo_root=repo_root)

    manifest = read_json(manifest_path)
    assert manifest["output_dir"] == "reports/ml/datasets/test_explorer"
    assert len(manifest["exports"]) == 12
    assert manifest["export_style"]["x_tick_label_size"] == 12.0
    assert manifest["export_style"]["show_minor_ticks"] is True

    intensity_exports = [row for row in manifest["exports"] if row["plot_type"] == "intensity_distribution"]
    assert len(intensity_exports) == 3
    intensity_x_limits = {tuple(row["x_limits"]) for row in intensity_exports}
    intensity_y_limits = {tuple(row["y_limits"]) for row in intensity_exports}
    assert len(intensity_x_limits) == 1
    assert len(intensity_y_limits) == 1
    assert intensity_y_limits == {(0.0, 3000.0)}
    assert {row["title"] for row in intensity_exports} == {"Al", "Cu", "Ni"}
    assert {row["color"] for row in intensity_exports} == {"#123456"}

    for attribute in ("CI", "IQ", "Fit"):
        rows = [row for row in manifest["exports"] if row["attribute"] == attribute]
        assert len(rows) == 3
        assert len({tuple(row["x_limits"]) for row in rows}) == 1
        if attribute == "CI":
            assert {tuple(row["y_limits"]) for row in rows} == {(0.0, 20.0)}
        else:
            assert len({tuple(row["y_limits"]) for row in rows}) == 1
        for row in rows:
            assert (repo_root / row["path"]).exists()

    for row in manifest["exports"]:
        assert (repo_root / row["path"]).exists()
