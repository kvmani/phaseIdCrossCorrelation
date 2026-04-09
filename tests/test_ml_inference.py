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
from phase_id_xcorr.ml.html_report import generate_full_scan_suite_html_report
from phase_id_xcorr.ml.inference_gui import InferenceMainWindow, _PatternCompareWidget, _contrast_stretch_gray, _prepare_display_gray
from phase_id_xcorr.ml.oh5_inference import FullScanInferenceResult, export_full_scan_artifacts, run_suite_full_scan_inference
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


def test_run_suite_full_scan_inference_writes_aggregate_and_per_run_outputs(tmp_path: Path, monkeypatch) -> None:
    suite_root = tmp_path / "suite"
    run_a = suite_root / "run_a"
    run_b = suite_root / "run_b"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    oh5_path = tmp_path / "scan.oh5"
    oh5_path.write_text("placeholder", encoding="utf-8")

    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.zeros((tensor.shape[0], 2), dtype=torch.float32, device=tensor.device)

    def _loaded(run_dir: Path) -> LoadedModel:
        return LoadedModel(
            run_dir=run_dir,
            report_path=run_dir / "report.json",
            checkpoint_path=run_dir / "best_checkpoint.pt",
            dataset_manifest_path=tmp_path / "dataset_manifest.json",
            class_names=["Cu", "Ni"],
            preprocessing_policy=PreprocessingPolicy(resize_hw=(32, 32), apply_circular_mask=True, normalize_mode="none"),
            input_mean=0.0,
            input_std=1.0,
            device=torch.device("cpu"),
            model=DummyModel(),
            model_family="simple_cnn",
            model_name=f"{run_dir.name}_model",
        )

    monkeypatch.setattr("phase_id_xcorr.ml.oh5_inference.list_model_runs", lambda root: [run_a, run_b])
    monkeypatch.setattr(
        "phase_id_xcorr.ml.oh5_inference.load_trained_model",
        lambda run_dir, repo_root, checkpoint_name, device: _loaded(run_dir),
    )

    def _fake_full_scan(loaded, oh5_path, scan_name=None, progress_callback=None, log_callback=None):
        return FullScanInferenceResult(
            oh5_path=oh5_path,
            scan_name=scan_name or oh5_path.stem,
            nx=2,
            ny=2,
            total_pixels=4,
            header_total_pixels=4,
            class_names=["Cu", "Ni"],
            predicted_indices=np.asarray([0, 1, 0, 1], dtype=np.int32),
            confidences=np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
            phase_counts={"Cu": 2, "Ni": 2},
            phase_fractions={"Cu": 0.5, "Ni": 0.5},
            mean_confidence=0.75,
            euler_rows_deg=None,
            euler_source_unit=None,
            euler_convention=None,
            rows=[],
        )

    monkeypatch.setattr("phase_id_xcorr.ml.oh5_inference.run_oh5_full_scan_inference", _fake_full_scan)
    monkeypatch.setattr("phase_id_xcorr.ml.oh5_inference._render_ipf_artifacts", lambda result, logger=None: (None, None))

    def _fake_export_full_scan_artifacts(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text("{}", encoding="utf-8")
        (output_dir / "summary.html").write_text("<html></html>", encoding="utf-8")
        (output_dir / "pixel_predictions.csv").write_text("", encoding="utf-8")
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "predicted_phase_map.png").write_text("png", encoding="utf-8")
        (artifacts_dir / "predicted_phase_legend.png").write_text("png", encoding="utf-8")
        manifest = output_dir / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return manifest

    monkeypatch.setattr("phase_id_xcorr.ml.oh5_inference.export_full_scan_artifacts", _fake_export_full_scan_artifacts)

    output_dir = tmp_path / "suite_exports"
    result = run_suite_full_scan_inference(
        suite_root=suite_root,
        oh5_path=oh5_path,
        output_dir=output_dir,
        repo_root=tmp_path,
        logger=logging.getLogger("test_full_scan_suite"),
    )

    assert result.processed_runs == 2
    assert result.failed_runs == 0
    assert result.summary_json.exists()
    assert result.summary_md.exists()
    assert result.manifest_json.exists()
    assert (output_dir / "events.jsonl").exists()
    assert (output_dir / "runs" / "run_a" / "manifest.json").exists()
    assert (output_dir / "runs" / "run_b" / "manifest.json").exists()

    summary = read_json(result.summary_json)
    assert summary["runs_total"] == 2
    assert summary["runs_completed"] == 2
    assert summary["runs_failed"] == 0
    assert [row["run_name"] for row in summary["rows"]] == ["run_a", "run_b"]


def test_generate_full_scan_suite_html_report_writes_comparative_page(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "ml" / "full_scan_suite_exports" / "scan_1"
    run_a = report_dir / "runs" / "run_a"
    run_b = report_dir / "runs" / "run_b"
    (run_a / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_b / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_a / "artifacts" / "predicted_phase_map.png").write_text("png", encoding="utf-8")
    (run_a / "artifacts" / "predicted_phase_legend.png").write_text("png", encoding="utf-8")
    (run_a / "artifacts" / "ipf_colored_ebsd_map.png").write_text("png", encoding="utf-8")
    (run_a / "artifacts" / "ipf_reference.png").write_text("png", encoding="utf-8")
    (run_b / "artifacts" / "predicted_phase_map.png").write_text("png", encoding="utf-8")
    (run_b / "artifacts" / "predicted_phase_legend.png").write_text("png", encoding="utf-8")
    (run_a / "summary.html").write_text("<html></html>", encoding="utf-8")
    (run_b / "summary.html").write_text("<html></html>", encoding="utf-8")
    (run_a / "summary.json").write_text("{}", encoding="utf-8")
    (run_b / "summary.json").write_text("{}", encoding="utf-8")
    (run_a / "pixel_predictions.csv").write_text("", encoding="utf-8")
    (run_b / "pixel_predictions.csv").write_text("", encoding="utf-8")
    (run_a / "manifest.json").write_text("{}", encoding="utf-8")
    (run_b / "manifest.json").write_text("{}", encoding="utf-8")

    bench_root = tmp_path / "reports" / "ml" / "benchmarks" / "suite_x"
    bench_run_a = bench_root / "run_a"
    bench_run_b = bench_root / "run_b"
    bench_run_a.mkdir(parents=True, exist_ok=True)
    bench_run_b.mkdir(parents=True, exist_ok=True)
    dataset_manifest = tmp_path / "reports" / "ml" / "datasets" / "ds" / "manifest.json"
    dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        dataset_manifest,
        {
            "num_samples_total": 100,
            "raw_input_rows_total": 120,
            "split_counts": {"train": 80, "val": 10, "test": 10},
            "phase_statistics": {
                "Cu": {"accepted_count": 50, "accepted_fraction_of_dataset": 0.5, "train_count": 40, "val_count": 5, "test_count": 5, "confidence_index": {"mean": 0.8}, "fit": {"mean": 0.7}, "image_quality": {"mean": 100.0}, "intensity_distribution": {"mode_intensity_value": 10}},
                "Ni": {"accepted_count": 50, "accepted_fraction_of_dataset": 0.5, "train_count": 40, "val_count": 5, "test_count": 5, "confidence_index": {"mean": 0.7}, "fit": {"mean": 0.8}, "image_quality": {"mean": 110.0}, "intensity_distribution": {"mode_intensity_value": 20}},
            },
        },
    )
    report_payload_a = {
        "best_val_macro_f1": 0.91,
        "test_metrics": {"accuracy": 0.90, "macro_f1": 0.89},
        "dataset_manifest_path": "reports/ml/datasets/ds/manifest.json",
    }
    report_payload_b = {
        "best_val_macro_f1": 0.93,
        "test_metrics": {"accuracy": 0.92, "macro_f1": 0.91},
        "dataset_manifest_path": "reports/ml/datasets/ds/manifest.json",
    }
    write_json(bench_run_a / "report.json", report_payload_a)
    write_json(bench_run_b / "report.json", report_payload_b)

    summary_json = report_dir / "suite_full_scan_summary.json"
    write_json(
        summary_json,
        {
            "suite_root": "reports/ml/benchmarks/suite_x",
            "oh5_path": "C:/scan.oh5",
            "runs_total": 2,
            "runs_completed": 2,
            "runs_failed": 0,
            "rows": [
                {
                    "run_name": "run_a",
                    "status": "completed",
                    "run_dir": "reports/ml/benchmarks/suite_x/run_a",
                    "model_name": "model_a",
                    "mean_confidence": 0.88,
                    "dominant_phase": "Cu",
                    "phase_fractions": {"Cu": 0.6, "Ni": 0.4},
                    "artifacts": {
                        "summary_json": "reports/ml/full_scan_suite_exports/scan_1/runs/run_a/summary.json",
                        "summary_html": "reports/ml/full_scan_suite_exports/scan_1/runs/run_a/summary.html",
                        "manifest_json": "reports/ml/full_scan_suite_exports/scan_1/runs/run_a/manifest.json",
                        "pixel_predictions_csv": "reports/ml/full_scan_suite_exports/scan_1/runs/run_a/pixel_predictions.csv",
                        "predicted_phase_map_png": "reports/ml/full_scan_suite_exports/scan_1/runs/run_a/artifacts/predicted_phase_map.png",
                        "predicted_phase_legend_png": "reports/ml/full_scan_suite_exports/scan_1/runs/run_a/artifacts/predicted_phase_legend.png",
                        "ipf_colored_ebsd_map_png": "reports/ml/full_scan_suite_exports/scan_1/runs/run_a/artifacts/ipf_colored_ebsd_map.png",
                        "ipf_reference_png": "reports/ml/full_scan_suite_exports/scan_1/runs/run_a/artifacts/ipf_reference.png",
                    },
                },
                {
                    "run_name": "run_b",
                    "status": "completed",
                    "run_dir": "reports/ml/benchmarks/suite_x/run_b",
                    "model_name": "model_b",
                    "mean_confidence": 0.83,
                    "dominant_phase": "Ni",
                    "phase_fractions": {"Cu": 0.45, "Ni": 0.55},
                    "artifacts": {
                        "summary_json": "reports/ml/full_scan_suite_exports/scan_1/runs/run_b/summary.json",
                        "summary_html": "reports/ml/full_scan_suite_exports/scan_1/runs/run_b/summary.html",
                        "manifest_json": "reports/ml/full_scan_suite_exports/scan_1/runs/run_b/manifest.json",
                        "pixel_predictions_csv": "reports/ml/full_scan_suite_exports/scan_1/runs/run_b/pixel_predictions.csv",
                        "predicted_phase_map_png": "reports/ml/full_scan_suite_exports/scan_1/runs/run_b/artifacts/predicted_phase_map.png",
                        "predicted_phase_legend_png": "reports/ml/full_scan_suite_exports/scan_1/runs/run_b/artifacts/predicted_phase_legend.png",
                    },
                },
            ],
        },
    )

    output_html = report_dir / "comparison_report.html"
    result = generate_full_scan_suite_html_report(
        summary_json_path=summary_json,
        output_html=output_html,
        repo_root=tmp_path,
    )
    text = result.read_text(encoding="utf-8")
    assert result.exists()
    assert "Full-Scan Model Comparison" in text
    assert "run_a" in text
    assert "run_b" in text
    assert "Shared Scan Visuals" in text
    assert "Predicted Phase Maps" in text
