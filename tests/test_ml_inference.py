from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
from PIL import Image
from PySide6 import QtWidgets
import torch
import yaml

from phase_id_xcorr.ml.dataset_io import save_split_npz, write_json
from phase_id_xcorr.ml.dataset_io import read_json
from phase_id_xcorr.ml.inference import LoadedModel, list_model_runs, load_trained_model, predict_image, predict_pattern_array
from phase_id_xcorr.ml.inference_gui import InferenceMainWindow, _PatternCompareWidget, _contrast_stretch_gray, _prepare_display_gray
from phase_id_xcorr.ml.oh5_inference import FullScanInferenceResult, export_full_scan_artifacts
from phase_id_xcorr.ml.preprocessing_policy import PreprocessingPolicy
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
            arr[4:14, 4:14] = 0.8
        elif cls == 1:
            arr[:, ::3] = 0.7
        else:
            rr = np.arange(h)[:, None]
            cc = np.arange(w)[None, :]
            arr[((rr - h // 2) ** 2 + (cc - w // 2) ** 2) < 25] = 0.9
        arr += rng.normal(0.0, 0.02, size=(h, w)).astype(np.float32)
        patterns[i] = np.clip(arr, 0.0, 1.0)
    return patterns, labels


def test_inference_loads_trained_run_and_predicts_image(tmp_path: Path) -> None:
    repo_root = tmp_path
    out_dataset = tmp_path / "dataset"
    out_dataset.mkdir(parents=True, exist_ok=True)

    train_p, train_y = _make_patterns(30, 18, 18, seed=1)
    val_p, val_y = _make_patterns(12, 18, 18, seed=2)
    test_p, test_y = _make_patterns(12, 18, 18, seed=3)

    save_split_npz(out_dataset / "train.npz", patterns=train_p, labels=train_y, sample_ids=[f"tr_{i}" for i in range(len(train_y))])
    save_split_npz(out_dataset / "val.npz", patterns=val_p, labels=val_y, sample_ids=[f"va_{i}" for i in range(len(val_y))])
    save_split_npz(out_dataset / "test.npz", patterns=test_p, labels=test_y, sample_ids=[f"te_{i}" for i in range(len(test_y))])

    manifest = {
        "schema_version": "phase_id_xcorr.ml_dataset_manifest.v1",
        "phase_to_label": {"Al": 0, "Ni": 1, "Cu": 2},
        "preprocessing_policy": {
            "resize_hw": [24, 24],
            "apply_circular_mask": True,
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
        "seed": 5,
        "device": "cpu",
        "amp": False,
        "batch_size": 8,
        "epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "input": {
            "resize_hw": [24, 24],
            "apply_circular_mask": True,
            "normalize": {"mean": [0.0], "std": [1.0]},
        },
        "model": {
            "family": "simple_cnn",
            "width": 8,
            "in_chans": 1,
        },
    }
    cfg_path = tmp_path / "train.yml"
    cfg_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")
    train_classifier(config_path=cfg_path, repo_root=repo_root, debug=True)

    suite_root = tmp_path / "suite_run"
    run_dirs = list_model_runs(suite_root)
    assert len(run_dirs) == 1

    loaded = load_trained_model(run_dir=run_dirs[0], repo_root=repo_root, device="cpu")
    image_path = tmp_path / "unknown.png"
    Image.fromarray((test_p[0] * 255.0).round().astype(np.uint8), mode="L").save(image_path)

    result = predict_image(loaded=loaded, image_path=image_path)
    assert result.predicted_phase in {"Al", "Ni", "Cu"}
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
    assert result.preprocessed_image.shape == (24, 24)


def test_contrast_stretch_gray_expands_dynamic_range() -> None:
    arr = np.asarray(
        [
            [0.20, 0.25, 0.30],
            [0.35, 0.40, 0.45],
        ],
        dtype=np.float32,
    )
    out = _contrast_stretch_gray(arr, lower_pct=0.0, upper_pct=100.0)
    assert np.isclose(float(out.min()), 0.0)
    assert np.isclose(float(out.max()), 1.0)
    assert out.shape == arr.shape


def test_prepare_display_gray_keeps_values_bounded() -> None:
    arr = np.asarray(
        [
            [0.05, 0.05, 0.10, 0.20],
            [0.20, 0.30, 0.35, 0.40],
            [0.50, 0.60, 0.80, 0.95],
        ],
        dtype=np.float32,
    )
    out = _prepare_display_gray(arr, histogram_normalization=True, contrast_stretch=True)
    assert out.shape == arr.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert not np.allclose(out, arr)


def test_pattern_compare_widget_keeps_zoom_synchronized() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = _PatternCompareWidget()
    widget.resize(900, 700)
    widget.show()
    app.processEvents()

    raw = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)
    processed = np.flipud(raw)
    widget.set_patterns(raw, processed)
    app.processEvents()

    widget._zoom_views(1.15)
    app.processEvents()

    raw_scale = widget.raw_pane.view.transform().m11()
    processed_scale = widget.processed_pane.view.transform().m11()
    assert raw_scale > 1.0
    assert np.isclose(raw_scale, processed_scale)

    widget.raw_pane.view.horizontalScrollBar().setValue(12)
    widget.raw_pane.view.verticalScrollBar().setValue(18)
    app.processEvents()

    assert widget.processed_pane.view.horizontalScrollBar().value() == widget.raw_pane.view.horizontalScrollBar().value()
    assert widget.processed_pane.view.verticalScrollBar().value() == widget.raw_pane.view.verticalScrollBar().value()
    widget.close()


def test_predict_pattern_array_applies_circular_mask_for_inference() -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.zeros((tensor.shape[0], 2), dtype=torch.float32, device=tensor.device)

    loaded = LoadedModel(
        run_dir=Path("."),
        report_path=Path("report.json"),
        checkpoint_path=Path("best_checkpoint.pt"),
        dataset_manifest_path=Path("dataset_manifest.json"),
        class_names=["A", "B"],
        preprocessing_policy=PreprocessingPolicy(resize_hw=(32, 32), apply_circular_mask=True, normalize_mode="none"),
        input_mean=0.0,
        input_std=1.0,
        device=torch.device("cpu"),
        model=DummyModel(),
        model_family="dummy",
        model_name="dummy",
    )
    pattern = np.ones((32, 32), dtype=np.float32)
    result = predict_pattern_array(loaded=loaded, pattern=pattern)
    assert np.isclose(float(result.preprocessed_image[0, 0]), 0.0)
    assert np.isclose(float(result.preprocessed_image[-1, -1]), 0.0)
    assert float(result.preprocessed_image[16, 16]) > 0.0


def test_export_full_scan_artifacts_writes_manifest_bundle(tmp_path: Path) -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.zeros((tensor.shape[0], 2), dtype=torch.float32, device=tensor.device)

    loaded = LoadedModel(
        run_dir=tmp_path / "run",
        report_path=tmp_path / "run" / "report.json",
        checkpoint_path=tmp_path / "run" / "best_checkpoint.pt",
        dataset_manifest_path=tmp_path / "dataset_manifest.json",
        class_names=["Ni", "Cu"],
        preprocessing_policy=PreprocessingPolicy(resize_hw=(32, 32), apply_circular_mask=True, normalize_mode="none"),
        input_mean=0.0,
        input_std=1.0,
        device=torch.device("cpu"),
        model=DummyModel(),
        model_family="dummy",
        model_name="dummy",
    )
    result = FullScanInferenceResult(
        oh5_path=tmp_path / "scan.oh5",
        scan_name="scan",
        nx=2,
        ny=2,
        total_pixels=4,
        header_total_pixels=4,
        class_names=["Ni", "Cu"],
        predicted_indices=np.asarray([0, 1, 0, 1], dtype=np.int32),
        confidences=np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        phase_counts={"Ni": 2, "Cu": 2},
        phase_fractions={"Ni": 0.5, "Cu": 0.5},
        mean_confidence=0.75,
        euler_rows_deg=None,
        euler_source_unit=None,
        euler_convention=None,
        rows=[
            {"pattern_index": 0, "x": 0, "y": 0, "predicted_phase": "Ni", "predicted_index": 0, "confidence": 0.9},
            {"pattern_index": 1, "x": 1, "y": 0, "predicted_phase": "Cu", "predicted_index": 1, "confidence": 0.8},
        ],
    )
    map_image = np.zeros((2, 2, 3), dtype=np.float32)
    output_dir = tmp_path / "export"
    manifest_path = export_full_scan_artifacts(
        repo_root=tmp_path,
        loaded=loaded,
        result=result,
        output_dir=output_dir,
        predicted_map_image=map_image,
        ipf_reference_image=None,
        ipf_colored_map_image=None,
        use_confidence_shading=True,
    )
    assert manifest_path.exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.html").exists()
    assert (output_dir / "pixel_predictions.csv").exists()
    assert (output_dir / "artifacts" / "predicted_phase_map.png").exists()
    assert (output_dir / "artifacts" / "predicted_phase_legend.png").exists()
    summary = read_json(output_dir / "summary.json")
    assert summary["artifacts"]["predicted_phase_map_png"] == "artifacts/predicted_phase_map.png"
    assert summary["artifacts"]["predicted_phase_legend_png"] == "artifacts/predicted_phase_legend.png"
    assert summary["artifacts"]["summary_html"] == "summary.html"
    assert summary["artifacts"]["manifest_json"] == "manifest.json"


def test_inference_gui_export_uses_selected_directory_without_appending_suffix(tmp_path: Path, monkeypatch) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.zeros((tensor.shape[0], 2), dtype=torch.float32, device=tensor.device)

    loaded = LoadedModel(
        run_dir=tmp_path / "run",
        report_path=tmp_path / "run" / "report.json",
        checkpoint_path=tmp_path / "run" / "best_checkpoint.pt",
        dataset_manifest_path=tmp_path / "dataset_manifest.json",
        class_names=["Ni", "Cu"],
        preprocessing_policy=PreprocessingPolicy(resize_hw=(32, 32), apply_circular_mask=True, normalize_mode="none"),
        input_mean=0.0,
        input_std=1.0,
        device=torch.device("cpu"),
        model=DummyModel(),
        model_family="dummy",
        model_name="dummy",
    )
    result = FullScanInferenceResult(
        oh5_path=tmp_path / "scan.oh5",
        scan_name="scan",
        nx=2,
        ny=2,
        total_pixels=4,
        header_total_pixels=4,
        class_names=["Ni", "Cu"],
        predicted_indices=np.asarray([0, 1, 0, 1], dtype=np.int32),
        confidences=np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        phase_counts={"Ni": 2, "Cu": 2},
        phase_fractions={"Ni": 0.5, "Cu": 0.5},
        mean_confidence=0.75,
        euler_rows_deg=None,
        euler_source_unit=None,
        euler_convention=None,
        rows=[],
    )

    window = InferenceMainWindow(repo_root=tmp_path, initial_root=None, logger=logging.getLogger("test_inference_gui"))
    window.state.inference_mode = "full_scan"
    window.state.loaded_model = loaded
    window.state.full_scan_result = result

    selected_dir = tmp_path / "chosen_export_dir"
    selected_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Path] = {}

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *args, **kwargs: str(selected_dir)),
    )

    def _fake_export_full_scan_artifacts(**kwargs):
        output_dir = kwargs["output_dir"]
        captured["output_dir"] = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return manifest_path

    monkeypatch.setattr("phase_id_xcorr.ml.inference_gui.export_full_scan_artifacts", _fake_export_full_scan_artifacts)

    window._export_full_scan_results()

    assert captured["output_dir"] == selected_dir.resolve()
    assert window.status_label.text() == f"Exported full-scan artifacts to {selected_dir.resolve()}"
    window.close()


def test_inference_gui_phase_map_legend_shows_phase_entries(tmp_path: Path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = InferenceMainWindow(repo_root=tmp_path, initial_root=None, logger=logging.getLogger("test_inference_gui_legend"))
    window._refresh_phase_map_legend(["Cu", "Ni"])
    app.processEvents()

    legend_labels = window.map_legend_widget.findChildren(QtWidgets.QLabel)
    legend_texts = {label.text() for label in legend_labels}
    assert "Cu" in legend_texts
    assert "Ni" in legend_texts
    window.close()
