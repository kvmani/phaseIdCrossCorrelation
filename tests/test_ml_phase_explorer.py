from __future__ import annotations

from pathlib import Path

import yaml

from phase_id_xcorr.ml.phase_explorer import build_intensity_mask, cdf_from_counts, histogram, load_explorer_dataset
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
