from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from phase_id_xcorr.ml.diagnostic_gallery import (
    build_diagnostic_gallery_session_from_config,
    export_diagnostic_gallery_artifacts,
    add_manual_record,
)
from phase_id_xcorr.ml.inference import LoadedModel
from phase_id_xcorr.ml.preprocessing_policy import PreprocessingPolicy
from phase_id_xcorr.ml.dataset_io import read_json


class MeanLogitModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2, 3))
        logits = torch.stack(
            [
                mean * 4.0,
                (1.0 - mean) * 3.0,
                torch.zeros_like(mean),
            ],
            dim=1,
        )
        return logits


def _write_fixture_oh5(path: Path, *, offset: float) -> None:
    nx, ny = 3, 2
    total = nx * ny
    h, w = 16, 16
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.asarray([b"EDAX"]))
        f.create_dataset("Version", data=np.asarray([b"TEST"]))
        g = f.create_group("scan")
        ebsd = g.create_group("EBSD")
        header = ebsd.create_group("Header")
        data = ebsd.create_group("Data")
        header.create_dataset("nColumns", data=np.asarray([nx], dtype=np.int32))
        header.create_dataset("nRows", data=np.asarray([ny], dtype=np.int32))

        patterns = np.zeros((total, h, w), dtype=np.uint16)
        ci = np.full((total,), 0.9, dtype=np.float32)
        iq = np.full((total,), 45.0, dtype=np.float32)
        fit = np.full((total,), 0.8, dtype=np.float32)
        valid = np.ones((total,), dtype=np.int8)

        for idx in range(total):
            level = np.clip((offset + (idx * 0.08)), 0.0, 1.0)
            patterns[idx, 3:13, 3:13] = np.uint16(level * 65535)

        data.create_dataset("Pattern", data=patterns)
        data.create_dataset("CI", data=ci)
        data.create_dataset("IQ", data=iq)
        data.create_dataset("Fit", data=fit)
        data.create_dataset("Valid", data=valid)


def _make_loaded_model(tmp_path: Path) -> LoadedModel:
    model = MeanLogitModel()
    return LoadedModel(
        run_dir=tmp_path / "run",
        report_path=tmp_path / "run" / "report.json",
        checkpoint_path=tmp_path / "run" / "best_checkpoint.pt",
        dataset_manifest_path=tmp_path / "dataset_manifest.json",
        class_names=["Al", "Cu", "Ni"],
        preprocessing_policy=PreprocessingPolicy(resize_hw=None, apply_circular_mask=False, normalize_mode="none"),
        input_mean=0.0,
        input_std=1.0,
        device=torch.device("cpu"),
        model=model,
        model_family="simple_cnn",
        model_name="simple_cnn_w8",
    )


def test_diagnostic_gallery_builds_exports_and_accepts_manual_add(tmp_path: Path) -> None:
    repo_root = tmp_path
    reference_paths: list[Path] = []
    unknown_paths: list[Path] = []

    for name, offset in (("Al", 0.15), ("Cu", 0.30), ("Ni", 0.45)):
        path = tmp_path / f"{name}.oh5"
        _write_fixture_oh5(path, offset=offset)
        reference_paths.append(path)

    for idx, offset in enumerate((0.10, 0.22, 0.38), start=1):
        path = tmp_path / f"data_{idx}.oh5"
        _write_fixture_oh5(path, offset=offset)
        unknown_paths.append(path)

    cfg = {
        "gallery_title": "Test Diagnostic Gallery",
        "output_dir": "reports/ml/diagnostic_gallery/test",
        "run_dir": str((tmp_path / "run").as_posix()),
        "checkpoint": "best_checkpoint.pt",
        "device": "cpu",
        "sampling": {
            "patterns_per_source": 2,
            "seed": 3,
            "strategy": "top_confidence",
        },
        "quality_filters": {
            "expression": "CI > 0.5 && Fit < 1.5 && Valid == True",
        },
        "prediction_filters": {
            "min_confidence": 0.2,
            "min_margin": 0.05,
        },
        "source_groups": {
            "reference": [{"file": path.as_posix(), "scan_id": path.stem, "phase_name": path.stem} for path in reference_paths],
            "unknown": [{"file": path.as_posix(), "scan_id": path.stem} for path in unknown_paths],
        },
    }

    loaded_model = _make_loaded_model(tmp_path)
    config_path = tmp_path / "diagnostic_gallery.yml"
    config_path.write_text("gallery_title: test\n", encoding="utf-8")

    session = build_diagnostic_gallery_session_from_config(
        cfg=cfg,
        config_path=config_path,
        repo_root=repo_root,
        debug=True,
        loaded_model=loaded_model,
    )

    assert session.tile_count == 12
    assert len(session.source_order) == 6
    assert all(len(result.display_records) == 2 for result in session.source_results.values())

    manual = add_manual_record(session=session, source_key="reference:Al", flat_index=5)
    assert manual.selected_by == "manual"
    assert session.tile_count == 13

    manifest_path = export_diagnostic_gallery_artifacts(session=session, repo_root=repo_root)
    manifest = read_json(manifest_path)

    assert manifest["schema_version"] == "phase_id_xcorr.ml_diagnostic_gallery.v1"
    assert len(manifest["sources"]) == 6
    assert len(manifest["records"]) == 13
    assert manifest["artifacts"]["combined_contact_sheet"].endswith("combined_contact_sheet.png")
    assert (repo_root / manifest["artifacts"]["combined_contact_sheet"]).exists()
    assert (repo_root / manifest["artifacts"]["reference_contact_sheet"]).exists()
    assert (repo_root / manifest["artifacts"]["unknown_contact_sheet"]).exists()
    assert session.manifest_path == manifest_path
