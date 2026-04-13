from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
from PySide6 import QtWidgets

from phase_id_xcorr.gui.oh5_crop_gui import Oh5CropMainWindow
from phase_id_xcorr.io.oh5_crop import (
    CropSpec,
    compare_cropped_pixel,
    export_cropped_oh5,
    load_review_session,
)
from phase_id_xcorr.ml.dataset_io import read_json


def _write_crop_fixture(path: Path, *, with_euler: bool = True) -> None:
    nx, ny = 5, 4
    n = nx * ny
    h, w = 12, 10

    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.asarray([b"EDAX"]))
        f.create_dataset("Version", data=np.asarray([b"TEST"]))
        scan = f.create_group("scan")
        ebsd = scan.create_group("EBSD")
        header = ebsd.create_group("Header")
        data = ebsd.create_group("Data")

        header.create_dataset("nColumns", data=np.asarray([nx], dtype=np.int32))
        header.create_dataset("nRows", data=np.asarray([ny], dtype=np.int32))
        header.create_dataset("Pattern Height", data=np.asarray([h], dtype=np.int32))
        header.create_dataset("Pattern Width", data=np.asarray([w], dtype=np.int32))
        header.create_dataset("Step X", data=np.asarray([0.5], dtype=np.float32))
        header.create_dataset("Step Y", data=np.asarray([0.75], dtype=np.float32))

        phase_header = header.create_group("Phase")
        phase0 = phase_header.create_group("0")
        phase0.create_dataset("MaterialName", data=np.asarray([b"Ni"]))
        phase0.create_dataset("SpaceGroupNumber", data=np.asarray([225], dtype=np.int32))

        patterns = np.zeros((n, h, w), dtype=np.uint16)
        ci = np.linspace(0.1, 0.9, num=n, dtype=np.float32)
        iq = np.linspace(50.0, 200.0, num=n, dtype=np.float32)
        fit = np.linspace(1.0, 3.0, num=n, dtype=np.float32)
        valid = np.ones((n,), dtype=np.int8)
        phase = np.zeros((n,), dtype=np.int8)
        x_pos = np.tile(np.arange(nx, dtype=np.float32) * 0.5, ny)
        y_pos = np.repeat(np.arange(ny, dtype=np.float32) * 0.75, nx)

        for idx in range(n):
            yy = idx // nx
            xx = idx % nx
            block = np.zeros((h, w), dtype=np.uint16)
            block[2:10, 2:8] = np.uint16(1000 + 100 * yy + 10 * xx)
            patterns[idx] = block

        data.create_dataset("Pattern", data=patterns)
        data.create_dataset("CI", data=ci)
        data.create_dataset("IQ", data=iq)
        data.create_dataset("Fit", data=fit)
        data.create_dataset("Valid", data=valid)
        data.create_dataset("Phase", data=phase)
        data.create_dataset("X Position", data=x_pos)
        data.create_dataset("Y Position", data=y_pos)
        if with_euler:
            data.create_dataset("Phi1", data=np.linspace(0.0, 90.0, num=n, dtype=np.float32))
            data.create_dataset("Phi", data=np.linspace(10.0, 40.0, num=n, dtype=np.float32))
            data.create_dataset("Phi2", data=np.linspace(20.0, 110.0, num=n, dtype=np.float32))

        sem = scan.create_group("SEM-PRIAS Images")
        sem_data = sem.create_group("Data")
        sem_data.create_dataset("SEM Image", data=np.arange(100, dtype=np.uint8).reshape(10, 10))


def test_export_cropped_oh5_rewrites_grid_and_preserves_aux_group(tmp_path: Path) -> None:
    src = tmp_path / "source.oh5"
    _write_crop_fixture(src)
    crop = CropSpec(row=1, column=1, width=3, height=2)
    out = tmp_path / "cropped.oh5"

    result = export_cropped_oh5(source_path=src, crop_spec=crop, output_path=out, repo_root=tmp_path)

    assert result.output_path == out.resolve()
    payload = read_json(result.manifest_path)
    assert payload["comparison"]["crop_origin_row"] == 1
    assert payload["comparison"]["crop_origin_column"] == 1
    assert payload["cropped_grid"] == {"nx": 3, "ny": 2}
    changed_paths = {row["path"] for row in payload["verification"]["changed_fields"]}
    unchanged_paths = {row["path"] for row in payload["verification"]["unchanged_fields"]}
    assert "scan/EBSD/Header/nColumns" in changed_paths
    assert "scan/EBSD/Header/nRows" in changed_paths
    assert "scan/EBSD/Data/Pattern" in changed_paths
    assert "scan/EBSD/Header/Phase/0/MaterialName" in unchanged_paths
    with h5py.File(out, "r") as f:
        scan = "scan"
        assert int(np.ravel(f[f"{scan}/EBSD/Header/nColumns"][()])[0]) == 3
        assert int(np.ravel(f[f"{scan}/EBSD/Header/nRows"][()])[0]) == 2
        assert tuple(f[f"{scan}/EBSD/Data/Pattern"].shape) == (6, 12, 10)
        assert tuple(f[f"{scan}/SEM-PRIAS Images/Data/SEM Image"].shape) == (10, 10)


def test_compare_cropped_pixel_matches_source_data(tmp_path: Path) -> None:
    src = tmp_path / "source.oh5"
    _write_crop_fixture(src)
    crop = CropSpec(row=1, column=1, width=3, height=2)
    out = tmp_path / "cropped.oh5"
    export_cropped_oh5(source_path=src, crop_spec=crop, output_path=out, repo_root=tmp_path)

    source_record, cropped_record = compare_cropped_pixel(
        source_path=src,
        cropped_path=out,
        crop_spec=crop,
        local_x=1,
        local_y=0,
    )

    assert source_record.x == 2
    assert source_record.y == 1
    assert cropped_record.x == 1
    assert cropped_record.y == 0
    assert np.allclose(source_record.pattern, cropped_record.pattern)
    assert source_record.quality_row["image_quality"] == cropped_record.quality_row["image_quality"]
    assert source_record.quality_row["confidence_index"] == cropped_record.quality_row["confidence_index"]
    assert source_record.scalar_values["X Position"] != cropped_record.scalar_values["X Position"]
    assert cropped_record.scalar_values["X Position"] == 0.5
    assert cropped_record.scalar_values["Y Position"] == 0.0


def test_load_review_session_renders_ipf_when_phase_metadata_present(tmp_path: Path) -> None:
    src = tmp_path / "source.oh5"
    _write_crop_fixture(src, with_euler=True)
    crop = CropSpec(row=1, column=1, width=3, height=2)
    out = tmp_path / "cropped.oh5"
    result = export_cropped_oh5(source_path=src, crop_spec=crop, output_path=out, repo_root=tmp_path)

    session = load_review_session(result)
    assert session.source.ipf_map is not None
    assert session.cropped.ipf_map is not None
    assert session.source.iq_map.shape == (4, 5)
    assert session.cropped.iq_map.shape == (2, 3)
    changed_paths = {row.path for row in session.verification_report.changed_fields}
    assert "scan/EBSD/Header/nColumns" in changed_paths
    assert "scan/EBSD/Data/Pattern" in changed_paths


def test_crop_gui_switches_to_review_and_maps_selection(tmp_path: Path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    src = tmp_path / "source.oh5"
    _write_crop_fixture(src, with_euler=True)
    crop = CropSpec(row=1, column=1, width=3, height=2)
    out = tmp_path / "cropped.oh5"
    result = export_cropped_oh5(source_path=src, crop_spec=crop, output_path=out, repo_root=tmp_path)

    window = Oh5CropMainWindow(repo_root=tmp_path, logger=__import__("logging").getLogger("test_oh5_crop_gui"))
    window.open_source_oh5(src)
    window.open_review_from_export(result)
    app.processEvents()

    assert window.mode_stack.currentWidget() == window.review_page
    assert "Original scan size: 4 rows x 5 columns" == window.original_iq_size_label.text()
    assert "Cropped scan size: 2 rows x 3 columns" == window.cropped_iq_size_label.text()
    window._handle_review_click(1, 1)
    app.processEvents()
    assert window.review_selection is not None
    assert window.review_selection.source_x == 2
    assert window.review_selection.source_y == 2
    assert window.original_info_group.labels["image_quality"].text() != "-"
    assert window.cropped_info_group.labels["image_quality"].text() != "-"
    assert window.progress_bar.value() == 100
    assert "Selected cropped pixel (1, 1)" in window.progress_label.text()
    assert "Loaded review session" in window.log_output.toPlainText()
    assert "Changed fields:" in window.audit_summary_label.text()
    changed_paths = {
        window.changed_fields_table.item(row, 0).text()
        for row in range(window.changed_fields_table.rowCount())
        if window.changed_fields_table.item(row, 0) is not None
    }
    unchanged_paths = {
        window.unchanged_fields_table.item(row, 0).text()
        for row in range(window.unchanged_fields_table.rowCount())
        if window.unchanged_fields_table.item(row, 0) is not None
    }
    assert "scan/EBSD/Header/nColumns" in changed_paths
    assert "scan/EBSD/Header/Phase/0/MaterialName" in unchanged_paths
    window.close()


def test_crop_gui_shows_ipf_unavailable_when_euler_missing(tmp_path: Path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    src = tmp_path / "source_no_euler.oh5"
    _write_crop_fixture(src, with_euler=False)
    crop = CropSpec(row=0, column=0, width=2, height=2)
    out = tmp_path / "cropped_no_euler.oh5"
    result = export_cropped_oh5(source_path=src, crop_spec=crop, output_path=out, repo_root=tmp_path)

    window = Oh5CropMainWindow(repo_root=tmp_path, logger=__import__("logging").getLogger("test_oh5_crop_gui_no_euler"))
    window.open_review_from_export(result)
    app.processEvents()

    assert "IPF unavailable" in window.original_ipf_label.text()
    assert "IPF unavailable" in window.cropped_ipf_label.text()
    assert window.original_ipf_size_label.text() == "Original scan size: 4 rows x 5 columns"
    assert window.cropped_ipf_size_label.text() == "Cropped scan size: 2 rows x 2 columns"
    window.close()


def test_crop_gui_open_source_populates_original_size_and_logs(tmp_path: Path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    src = tmp_path / "source.oh5"
    _write_crop_fixture(src, with_euler=True)

    window = Oh5CropMainWindow(repo_root=tmp_path, logger=__import__("logging").getLogger("test_oh5_crop_gui_open"))
    window.open_source_oh5(src)
    app.processEvents()

    assert window.crop_source_size_label.text() == "Original scan size: 4 rows x 5 columns"
    assert "centered starting rectangle covering about 50% of the scan" in window.crop_instructions_label.text()
    assert "Add additional rectangles" in window.crop_instructions_label.text()
    assert window.row_spin.value() == 1
    assert window.col_spin.value() == 1
    assert window.width_spin.value() == 2
    assert window.height_spin.value() == 2
    assert window.region_list.count() == 1
    assert "Rectangle 1" in window.region_list.item(0).text()
    assert window.progress_bar.value() == 100
    assert window.progress_label.text() == "Crop mode ready"
    assert "Loaded source scan" in window.log_output.toPlainText()
    window.close()


def test_crop_gui_multiple_regions_track_selected_rectangle(tmp_path: Path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    src = tmp_path / "source.oh5"
    _write_crop_fixture(src, with_euler=True)

    window = Oh5CropMainWindow(repo_root=tmp_path, logger=__import__("logging").getLogger("test_oh5_crop_gui_regions"))
    window.open_source_oh5(src)
    app.processEvents()

    window._add_region()
    app.processEvents()
    assert window.region_list.count() == 2
    assert window.region_list.currentRow() == 1

    window.row_spin.setValue(0)
    window.col_spin.setValue(2)
    window.width_spin.setValue(2)
    window.height_spin.setValue(2)
    app.processEvents()

    assert window.crop_regions[1].spec.row == 0
    assert window.crop_regions[1].spec.column == 2
    assert window.crop_regions[0].spec.row == 1
    assert window.crop_regions[0].spec.column == 1

    window.region_list.setCurrentRow(0)
    app.processEvents()
    assert window.row_spin.value() == 1
    assert window.col_spin.value() == 1
    window.close()


def test_crop_gui_batch_export_opens_review_selector(tmp_path: Path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    src = tmp_path / "source.oh5"
    _write_crop_fixture(src, with_euler=True)

    window = Oh5CropMainWindow(repo_root=tmp_path, logger=__import__("logging").getLogger("test_oh5_crop_gui_batch"))
    window.open_source_oh5(src)
    app.processEvents()

    window._add_region()
    app.processEvents()
    window.row_spin.setValue(0)
    window.col_spin.setValue(2)
    window.width_spin.setValue(2)
    window.height_spin.setValue(2)
    app.processEvents()

    window.output_path_edit.setText(str(tmp_path / "batch_output.oh5"))
    window._output_path_user_edited = True
    window._export_crop()
    app.processEvents()

    assert window.mode_stack.currentWidget() == window.review_page
    assert window.review_crop_selector.count() == 2
    assert window.review_crop_selector.currentIndex() == 0
    assert window.review_session is not None
    assert window.review_session.export.output_path.name == "batch_output_crop_1_1.oh5"

    window.review_crop_selector.setCurrentIndex(1)
    app.processEvents()
    assert window.review_session is not None
    assert window.review_session.export.output_path.name == "batch_output_crop_0_2.oh5"
    window.close()
