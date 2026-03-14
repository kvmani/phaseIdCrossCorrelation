from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from phase_id_xcorr.ml.oh5_reader import Oh5ScanReader


def _write_oh5(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.asarray([b"EDAX"]))
        f.create_dataset("Version", data=np.asarray([b"OH5_TEST"]))

        g = f.create_group("scan_test")
        ebsd = g.create_group("EBSD")
        header = ebsd.create_group("Header")
        data = ebsd.create_group("Data")

        header.create_dataset("nColumns", data=np.asarray([3], dtype=np.int32))
        header.create_dataset("nRows", data=np.asarray([2], dtype=np.int32))

        patt = np.zeros((6, 8, 8), dtype=np.uint16)
        patt[4, 2:6, 2:6] = 50000
        data.create_dataset("Pattern", data=patt)

        data.create_dataset("Confidence Index", data=np.linspace(0.1, 0.6, num=6, dtype=np.float32))
        data.create_dataset("Image Quality", data=np.linspace(10.0, 20.0, num=6, dtype=np.float32))
        data.create_dataset("Fit", data=np.linspace(1.0, 2.0, num=6, dtype=np.float32))
        data.create_dataset("Valid", data=np.asarray([1, 1, 0, 1, 1, 1], dtype=np.int8))


def test_oh5_reader_pattern_and_quality_aliases(tmp_path: Path) -> None:
    oh5 = tmp_path / "sample.oh5"
    _write_oh5(oh5)

    with Oh5ScanReader(oh5) as reader:
        meta = reader.meta()
        assert meta.pattern_present is True
        assert meta.nx == 3
        assert meta.ny == 2
        assert meta.pattern_shape == (8, 8)
        assert meta.quality_field_map["confidence_index"] == "Confidence Index"
        assert meta.quality_field_map["image_quality"] == "Image Quality"

        flat = reader.xy_to_flat(1, 1)
        assert flat == 4

        pattern = reader.read_pattern(flat_index=flat)
        assert pattern.shape == (8, 8)
        assert pattern.dtype == np.float32
        assert float(pattern.max()) <= 1.0

        q = reader.read_quality_row(flat_index=flat)
        assert q["confidence_index"] is not None
        assert q["image_quality"] is not None
        assert q["fit"] is not None
        assert q["valid"] is True


def test_oh5_reader_accepts_cropped_flat_pattern_stack(tmp_path: Path) -> None:
    oh5 = tmp_path / "cropped.oh5"

    with h5py.File(oh5, "w") as f:
        f.create_dataset("Manufacturer", data=np.asarray([b"EDAX"]))
        f.create_dataset("Version", data=np.asarray([b"OH5_TEST"]))

        g = f.create_group("scan_test")
        ebsd = g.create_group("EBSD")
        header = ebsd.create_group("Header")
        data = ebsd.create_group("Data")

        header.create_dataset("nColumns", data=np.asarray([4], dtype=np.int32))
        header.create_dataset("nRows", data=np.asarray([3], dtype=np.int32))

        patt = np.zeros((10, 8, 8), dtype=np.uint16)
        patt[9, 1:7, 1:7] = 40000
        data.create_dataset("Pattern", data=patt)
        data.create_dataset("CI", data=np.linspace(0.1, 0.9, num=12, dtype=np.float32))

    with Oh5ScanReader(oh5) as reader:
        meta = reader.meta()
        assert meta.nx == 4
        assert meta.ny == 3
        assert meta.total_pixels == 10
        assert "CI" in reader.discover_scalar_fields()
        pattern = reader.read_pattern(flat_index=9)
        assert pattern.shape == (8, 8)
        assert float(pattern.max()) <= 1.0
