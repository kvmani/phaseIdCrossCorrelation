"""Sampled `.oh5` inference workflow for trained CNN phase classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from phase_id_xcorr.reporting import build_run_manifest

from .config import get_required, load_yaml, resolve_path
from .dataset_io import rel_path, write_json, write_records_csv
from .inference import LoadedModel, load_trained_model, predict_pattern_array
from .oh5_reader import Oh5ScanReader
from .quality import QualityPolicy, evaluate_quality, quality_policy_from_config


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class ScanInferenceSpec:
    """One configured `.oh5` scan to evaluate."""

    file_path: Path
    scan_name: str
    expected_phase: str | None
    samples_per_scan: int | None


@dataclass(slots=True)
class Oh5InferenceResult:
    """Output artifact locations for one sampled `.oh5` inference run."""

    output_dir: Path
    patterns_csv: Path
    predictions_json: Path
    scans_csv: Path
    summary_json: Path
    manifest_json: Path
    summary_md: Path
    processed_scans: int
    sampled_patterns: int
    labeled_accuracy: float | None


@dataclass(slots=True)
class FullScanInferenceResult:
    """In-memory full-scan inference result for GUI rendering."""

    oh5_path: Path
    scan_name: str
    nx: int
    ny: int
    total_pixels: int
    header_total_pixels: int
    class_names: list[str]
    predicted_indices: np.ndarray
    confidences: np.ndarray
    phase_counts: dict[str, int]
    phase_fractions: dict[str, float]
    mean_confidence: float
    euler_rows_deg: np.ndarray | None
    euler_source_unit: str | None
    euler_convention: str | None
    rows: list[dict[str, Any]]


def _save_rgb_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(arr8, mode="RGB").save(path)


def _save_phase_legend_png(path: Path, class_names: list[str]) -> None:
    palette = {
        "Al": (235, 76, 64),
        "Ni": (41, 163, 87),
        "Cu": (51, 112, 224),
    }
    fallback = (
        (220, 68, 55),
        (55, 126, 34),
        (56, 99, 214),
        (213, 160, 33),
        (118, 84, 172),
        (33, 163, 163),
    )
    row_h = 28
    width = 340
    height = max(48, 16 + row_h * max(1, len(class_names)))
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), "Predicted phase legend", fill=(25, 25, 25))
    for idx, phase_name in enumerate(class_names):
        rgb = palette.get(phase_name, fallback[idx % len(fallback)])
        y = 16 + idx * row_h
        draw.rounded_rectangle((14, y + 6, 58, y + 22), radius=3, fill=rgb, outline=(80, 80, 80))
        draw.text((70, y + 5), str(phase_name), fill=(25, 25, 25))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _html_rel(target: Path, html_path: Path) -> str:
    return Path(os.path.relpath(target.resolve(), html_path.parent.resolve())).as_posix()


def _bundle_rel(path: Path, bundle_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), bundle_root.resolve())).as_posix()


def export_full_scan_artifacts(
    *,
    repo_root: Path,
    loaded: LoadedModel,
    result: FullScanInferenceResult,
    output_dir: Path,
    predicted_map_image: np.ndarray,
    ipf_reference_image: np.ndarray | None,
    ipf_colored_map_image: np.ndarray | None,
    use_confidence_shading: bool,
) -> Path:
    """Export GUI full-scan artifacts with machine-readable provenance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    predicted_map_png = artifacts_dir / "predicted_phase_map.png"
    phase_legend_png = artifacts_dir / "predicted_phase_legend.png"
    ipf_reference_png = artifacts_dir / "ipf_reference.png"
    ipf_colored_map_png = artifacts_dir / "ipf_colored_ebsd_map.png"
    rows_csv = output_dir / "pixel_predictions.csv"
    summary_json = output_dir / "summary.json"
    summary_html = output_dir / "summary.html"
    manifest_json = output_dir / "manifest.json"

    _save_rgb_png(predicted_map_png, predicted_map_image)
    _save_phase_legend_png(phase_legend_png, result.class_names)
    if ipf_reference_image is not None:
        _save_rgb_png(ipf_reference_png, ipf_reference_image)
    if ipf_colored_map_image is not None:
        _save_rgb_png(ipf_colored_map_png, ipf_colored_map_image)

    write_records_csv(rows_csv, result.rows)

    dominant_phase = max(result.phase_counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if result.phase_counts else ""
    summary_payload = {
        "schema_version": "phase_id_xcorr.ml_full_scan_gui_export.v1",
        "created_utc": _now_iso_utc(),
        "workflow": "ml_inference_gui_full_scan_export",
        "model": {
            "run_dir": rel_path(loaded.run_dir, repo_root),
            "report_path": rel_path(loaded.report_path, repo_root),
            "checkpoint_path": rel_path(loaded.checkpoint_path, repo_root),
            "dataset_manifest_path": rel_path(loaded.dataset_manifest_path, repo_root),
            "model_family": loaded.model_family,
            "model_name": loaded.model_name,
            "class_names": loaded.class_names,
            "preprocessing_policy": loaded.preprocessing_policy.to_dict(),
            "input_mean": loaded.input_mean,
            "input_std": loaded.input_std,
        },
        "scan": {
            "oh5_path": rel_path(result.oh5_path, repo_root),
            "scan_name": result.scan_name,
            "grid": {"nx": result.nx, "ny": result.ny},
            "total_pixels": result.total_pixels,
            "header_total_pixels": result.header_total_pixels,
            "mean_confidence": result.mean_confidence,
            "dominant_phase": dominant_phase,
            "phase_counts": result.phase_counts,
            "phase_fractions": result.phase_fractions,
            "confidence_shading": bool(use_confidence_shading),
            "euler_convention": result.euler_convention,
            "euler_source_unit": result.euler_source_unit,
        },
        "artifacts": {
            "predicted_phase_map_png": _bundle_rel(predicted_map_png, output_dir),
            "predicted_phase_legend_png": _bundle_rel(phase_legend_png, output_dir),
            "ipf_reference_png": None if ipf_reference_image is None else _bundle_rel(ipf_reference_png, output_dir),
            "ipf_colored_ebsd_map_png": None if ipf_colored_map_image is None else _bundle_rel(ipf_colored_map_png, output_dir),
            "pixel_predictions_csv": _bundle_rel(rows_csv, output_dir),
            "summary_json": _bundle_rel(summary_json, output_dir),
            "summary_html": _bundle_rel(summary_html, output_dir),
            "manifest_json": _bundle_rel(manifest_json, output_dir),
        },
        "provenance": {
            "repo_root": rel_path(repo_root, repo_root),
            "run_dir": rel_path(loaded.run_dir, repo_root),
            "report_path": rel_path(loaded.report_path, repo_root),
            "checkpoint_path": rel_path(loaded.checkpoint_path, repo_root),
            "dataset_manifest_path": rel_path(loaded.dataset_manifest_path, repo_root),
            "oh5_path": rel_path(result.oh5_path, repo_root),
        },
        "legend": [
            {"phase": phase_name, "count": int(result.phase_counts.get(phase_name, 0)), "fraction": float(result.phase_fractions.get(phase_name, 0.0))}
            for phase_name in result.class_names
        ],
    }
    write_json(summary_json, summary_payload)

    manifest_payload = build_run_manifest(
        repo_root=repo_root,
        packet_dir=loaded.run_dir,
        out_dir=output_dir,
        debug=False,
        extra={
            "workflow": "ml_inference_gui_full_scan_export",
            "run_dir": rel_path(loaded.run_dir, repo_root),
            "checkpoint_path": rel_path(loaded.checkpoint_path, repo_root),
            "oh5_path": rel_path(result.oh5_path, repo_root),
            "summary_json": _bundle_rel(summary_json, output_dir),
            "summary_html": _bundle_rel(summary_html, output_dir),
            "pixel_predictions_csv": _bundle_rel(rows_csv, output_dir),
            "predicted_phase_map_png": _bundle_rel(predicted_map_png, output_dir),
            "predicted_phase_legend_png": _bundle_rel(phase_legend_png, output_dir),
        },
    )
    write_json(manifest_json, manifest_payload)

    ipf_reference_block = (
        f'<figure><img src="{_html_rel(ipf_reference_png, summary_html)}" alt="IPF reference"><figcaption>IPF reference grouped by predicted phase.</figcaption></figure>'
        if ipf_reference_image is not None
        else "<p>IPF reference unavailable for this scan.</p>"
    )
    ipf_colored_block = (
        f'<figure><img src="{_html_rel(ipf_colored_map_png, summary_html)}" alt="IPF-colored EBSD map"><figcaption>IPF-colored EBSD map derived from scan Euler angles.</figcaption></figure>'
        if ipf_colored_map_image is not None
        else "<p>IPF-colored EBSD map unavailable for this scan.</p>"
    )
    rows_html = "\n".join(
        f"<tr><td>{phase}</td><td>{int(result.phase_counts.get(phase, 0))}</td><td>{float(result.phase_fractions.get(phase, 0.0)):.4f}</td></tr>"
        for phase in result.class_names
    )
    summary_html.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                "<html><head><meta charset='utf-8'><title>Full-Scan GUI Export</title>",
                "<style>body{font-family:Arial,sans-serif;margin:20px;} img{max-width:100%;border:1px solid #bbb;} figure{margin:0 0 20px 0;} table{border-collapse:collapse;margin-top:12px;} th,td{border:1px solid #999;padding:6px 10px;text-align:left;} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px;}</style>",
                "</head><body>",
                f"<h1>Full-scan GUI export: {result.scan_name}</h1>",
                f"<p><strong>.oh5:</strong> {rel_path(result.oh5_path, repo_root)}<br>",
                f"<strong>Run:</strong> {rel_path(loaded.run_dir, repo_root)}<br>",
                f"<strong>Checkpoint:</strong> {rel_path(loaded.checkpoint_path, repo_root)}<br>",
                f"<strong>Preprocessing:</strong> {loaded.preprocessing_policy.to_dict()}<br>",
                f"<strong>Mean confidence:</strong> {result.mean_confidence:.4f}<br>",
                f"<strong>Confidence shading:</strong> {'on' if use_confidence_shading else 'off'}</p>",
                "<div class='grid'>",
                f"<figure><img src='{_html_rel(predicted_map_png, summary_html)}' alt='Predicted phase map'><figcaption>Predicted phase map.</figcaption></figure>",
                f"<figure><img src='{_html_rel(phase_legend_png, summary_html)}' alt='Predicted phase legend'><figcaption>Phase legend.</figcaption></figure>",
                ipf_reference_block,
                ipf_colored_block,
                "</div>",
                "<h2>Per-phase summary</h2>",
                "<table><thead><tr><th>Phase</th><th>Pixels</th><th>Fraction</th></tr></thead><tbody>",
                rows_html,
                "</tbody></table>",
                f"<p>Machine-readable summary: <code>{_html_rel(summary_json, summary_html)}</code><br>",
                f"Per-pixel predictions: <code>{_html_rel(rows_csv, summary_html)}</code><br>",
                f"Manifest: <code>{_html_rel(manifest_json, summary_html)}</code></p>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_json


def _as_int(value: Any, *, field_name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return out


def _resolve_scan_file(
    file_value: str | Path,
    *,
    input_root: Path | None,
    config_dir: Path,
    repo_root: Path,
) -> Path:
    file_path = Path(file_value)
    if file_path.is_absolute():
        return file_path.resolve()
    if input_root is not None:
        candidate = (input_root / file_path).resolve()
        if candidate.exists():
            return candidate
    return resolve_path(file_path, base_dir=config_dir, repo_root=repo_root)


def _load_scan_specs(
    cfg: dict[str, Any],
    *,
    config_dir: Path,
    repo_root: Path,
) -> tuple[Path | None, list[ScanInferenceSpec], int, int]:
    input_root_cfg = cfg.get("input_root")
    input_root = None if input_root_cfg in (None, "") else resolve_path(str(input_root_cfg), base_dir=config_dir, repo_root=repo_root)

    sampling_cfg = cfg.get("sampling") if isinstance(cfg.get("sampling"), dict) else {}
    default_samples_per_scan = _as_int(
        sampling_cfg.get("samples_per_scan", cfg.get("samples_per_scan", 1)),
        field_name="sampling.samples_per_scan",
    )
    seed = int(sampling_cfg.get("seed", cfg.get("seed", 0)))

    specs: list[ScanInferenceSpec] = []

    discovery_cfg = cfg.get("scan_discovery") if isinstance(cfg.get("scan_discovery"), dict) else {}
    glob_pattern = str(discovery_cfg.get("glob", "")).strip()
    if glob_pattern:
        if input_root is None:
            raise ValueError("input_root is required when scan_discovery.glob is used")
        for path in sorted(input_root.glob(glob_pattern)):
            if path.is_file():
                specs.append(
                    ScanInferenceSpec(
                        file_path=path.resolve(),
                        scan_name=path.stem,
                        expected_phase=None,
                        samples_per_scan=None,
                    )
                )

    scans_cfg = cfg.get("scans", [])
    if scans_cfg:
        if not isinstance(scans_cfg, list):
            raise ValueError("scans must be a list")
        for idx, item in enumerate(scans_cfg):
            if not isinstance(item, dict):
                raise ValueError(f"scans[{idx}] must be a mapping")
            file_value = get_required(item, "file", where=f"scans[{idx}]")
            file_path = _resolve_scan_file(file_value, input_root=input_root, config_dir=config_dir, repo_root=repo_root)
            scan_name_raw = str(item.get("scan_name", "")).strip()
            expected_phase_raw = item.get("expected_phase")
            samples_override = item.get("samples_per_scan")
            specs.append(
                ScanInferenceSpec(
                    file_path=file_path,
                    scan_name=scan_name_raw or file_path.stem,
                    expected_phase=None if expected_phase_raw in (None, "") else str(expected_phase_raw),
                    samples_per_scan=int(samples_override) if samples_override is not None else None,
                )
            )

    if not specs:
        raise ValueError("No .oh5 scans resolved. Provide scan_discovery.glob and/or scans entries.")

    deduped: list[ScanInferenceSpec] = []
    seen: dict[tuple[str, str], int] = {}
    for spec in specs:
        key = (str(spec.file_path), str(spec.scan_name))
        if key in seen:
            deduped[seen[key]] = spec
            continue
        seen[key] = len(deduped)
        deduped.append(spec)

    return input_root, deduped, default_samples_per_scan, seed


def _resolve_output_dir(cfg: dict[str, Any], *, config_dir: Path, repo_root: Path) -> Path:
    output_dir_cfg = get_required(cfg, "output_dir", where="oh5 inference config")
    return resolve_path(str(output_dir_cfg), base_dir=config_dir, repo_root=repo_root)


def _read_quality_values(reader: Oh5ScanReader) -> dict[str, np.ndarray | None]:
    out: dict[str, np.ndarray | None] = {}
    for canonical, dataset_key in reader.quality_field_map.items():
        arr = reader.read_scalar_field_array(dataset_key)
        if arr is None:
            out[canonical] = None
            continue
        if canonical == "valid":
            out[canonical] = np.asarray(arr != 0, dtype=bool)
        else:
            out[canonical] = np.asarray(arr, dtype=np.float32)
    return out


def _eligible_indices(reader: Oh5ScanReader, policy: QualityPolicy) -> tuple[list[int], dict[str, int]]:
    arrays = _read_quality_values(reader)
    counts = {
        "total_pixels": int(reader.total_pixels),
        "eligible_pixels": 0,
    }
    eligible: list[int] = []
    for flat_index in range(reader.total_pixels):
        values = {
            "confidence_index": None if arrays.get("confidence_index") is None else float(arrays["confidence_index"][flat_index]),
            "image_quality": None if arrays.get("image_quality") is None else float(arrays["image_quality"][flat_index]),
            "fit": None if arrays.get("fit") is None else float(arrays["fit"][flat_index]),
            "valid": None if arrays.get("valid") is None else bool(arrays["valid"][flat_index]),
        }
        decision = evaluate_quality(values, policy)
        if decision.accept:
            eligible.append(flat_index)
    counts["eligible_pixels"] = len(eligible)
    return eligible, counts


def _prediction_row(
    *,
    loaded: LoadedModel,
    reader: Oh5ScanReader,
    spec: ScanInferenceSpec,
    flat_index: int,
    repo_root: Path,
) -> dict[str, Any]:
    x, y = reader.flat_to_xy(flat_index)
    quality_row = reader.read_quality_row(flat_index=flat_index)
    pattern = reader.read_pattern(flat_index=flat_index)
    prediction = predict_pattern_array(loaded=loaded, pattern=pattern)
    is_correct = None
    if spec.expected_phase:
        is_correct = prediction.predicted_phase == spec.expected_phase

    row: dict[str, Any] = {
        "scan_name": spec.scan_name,
        "oh5_path": rel_path(spec.file_path, repo_root),
        "pattern_index": int(flat_index),
        "x": int(x),
        "y": int(y),
        "expected_phase": spec.expected_phase or "",
        "predicted_phase": prediction.predicted_phase,
        "predicted_index": prediction.predicted_index,
        "confidence": round(float(prediction.confidence), 6),
        "is_correct": "" if is_correct is None else bool(is_correct),
        "confidence_index": quality_row.get("confidence_index"),
        "image_quality": quality_row.get("image_quality"),
        "fit": quality_row.get("fit"),
        "valid": quality_row.get("valid"),
    }
    for phase_name, prob in prediction.probabilities.items():
        row[f"prob_{phase_name}"] = round(float(prob), 6)
    return row


def run_oh5_full_scan_inference(
    *,
    loaded: LoadedModel,
    oh5_path: Path,
    scan_name: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    log_callback: Callable[[str, str], None] | None = None,
) -> FullScanInferenceResult:
    """Run prediction on every available pattern in one `.oh5` scan."""

    backend_log = logging.getLogger("ml_inference_gui.backend")

    def _emit_log(level: str, message: str) -> None:
        level_value = getattr(logging, str(level).upper(), logging.INFO)
        backend_log.log(level_value, message)
        if log_callback is not None:
            log_callback(level, message)

    def _emit_progress(processed: int, total: int, start_time: float, *, stage: str) -> None:
        if progress_callback is None:
            return
        elapsed = max(0.0, time.perf_counter() - start_time)
        fraction = (float(processed) / float(total)) if total > 0 else 0.0
        rate = (float(processed) / elapsed) if elapsed > 1e-9 else 0.0
        remaining = max(0, total - processed)
        eta_seconds = (float(remaining) / rate) if rate > 1e-9 else None
        progress_callback(
            {
                "stage": stage,
                "processed": int(processed),
                "total": int(total),
                "fraction": float(fraction),
                "elapsed_seconds": float(elapsed),
                "eta_seconds": None if eta_seconds is None else float(eta_seconds),
            }
        )

    resolved_path = oh5_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f".oh5 file not found: {resolved_path}")
    if resolved_path.suffix.lower() != ".oh5":
        raise ValueError(f"Expected .oh5 file, got: {resolved_path}")

    with Oh5ScanReader(resolved_path) as reader:
        if not reader.pattern_present:
            raise ValueError(f"Pattern dataset missing in .oh5 file: {resolved_path}")

        predicted_indices = np.full((reader.header_total_pixels,), -1, dtype=np.int32)
        confidences = np.full((reader.header_total_pixels,), np.nan, dtype=np.float32)
        euler_rows_deg = (
            np.full((reader.header_total_pixels, 3), np.nan, dtype=np.float32)
            if reader.euler_present
            else None
        )
        rows: list[dict[str, Any]] = []
        start_time = time.perf_counter()
        progress_stride = max(1, min(500, reader.total_pixels // 50 if reader.total_pixels > 0 else 1))

        _emit_log(
            "info",
            (
                f"Opened scan '{scan_name or resolved_path.stem}' from {resolved_path}. "
                f"Grid={reader.nx}x{reader.ny}, patterns={reader.total_pixels}, "
                f"Euler={'present' if reader.euler_present else 'missing'}."
            ),
        )
        if reader.euler_present:
            _emit_log(
                "info",
                f"Euler fields detected with convention {reader.euler_convention} and source unit {reader.euler_unit}.",
            )
        else:
            _emit_log("warning", "Euler angle fields were not found; IPF reference plot will be unavailable.")
        _emit_progress(0, reader.total_pixels, start_time, stage="scan_open")

        for flat_index in range(reader.total_pixels):
            x, y = reader.flat_to_xy(flat_index)
            quality_row = reader.read_quality_row(flat_index=flat_index)
            pattern = reader.read_pattern(flat_index=flat_index)
            prediction = predict_pattern_array(loaded=loaded, pattern=pattern)
            predicted_indices[flat_index] = int(prediction.predicted_index)
            confidences[flat_index] = float(prediction.confidence)
            euler_row = None
            if euler_rows_deg is not None:
                euler_row = reader.read_euler_row(flat_index=flat_index, degrees=True)
                euler_rows_deg[flat_index, 0] = float(euler_row["phi1"])
                euler_rows_deg[flat_index, 1] = float(euler_row["Phi"])
                euler_rows_deg[flat_index, 2] = float(euler_row["phi2"])

            row: dict[str, Any] = {
                "pattern_index": int(flat_index),
                "x": int(x),
                "y": int(y),
                "predicted_phase": prediction.predicted_phase,
                "predicted_index": prediction.predicted_index,
                "confidence": round(float(prediction.confidence), 6),
                "confidence_index": quality_row.get("confidence_index"),
                "image_quality": quality_row.get("image_quality"),
                "fit": quality_row.get("fit"),
                "valid": quality_row.get("valid"),
            }
            if euler_row is not None:
                row["euler_phi1"] = round(float(euler_row["phi1"]), 6)
                row["euler_Phi"] = round(float(euler_row["Phi"]), 6)
                row["euler_phi2"] = round(float(euler_row["phi2"]), 6)
            for phase_name, prob in prediction.probabilities.items():
                row[f"prob_{phase_name}"] = round(float(prob), 6)
            rows.append(row)

            processed = flat_index + 1
            if processed == reader.total_pixels or processed % progress_stride == 0:
                _emit_progress(processed, reader.total_pixels, start_time, stage="infer")
                _emit_log(
                    "info",
                    (
                        f"Inference progress: {processed}/{reader.total_pixels} pixels "
                        f"({100.0 * processed / max(1, reader.total_pixels):.1f}%)."
                    ),
                )

    valid_mask = predicted_indices >= 0
    class_names = list(loaded.class_names)
    phase_counts = {phase: 0 for phase in class_names}
    for idx in predicted_indices[valid_mask]:
        phase_counts[class_names[int(idx)]] += 1

    inferred_total = int(np.sum(valid_mask))
    phase_fractions = {
        phase: (float(count) / float(inferred_total) if inferred_total > 0 else 0.0)
        for phase, count in phase_counts.items()
    }
    mean_confidence = float(np.nanmean(confidences[valid_mask])) if inferred_total > 0 else 0.0
    _emit_log(
        "info",
        (
            f"Full-scan inference complete. Inferred {inferred_total} pixels with mean confidence "
            f"{mean_confidence:.4f}."
        ),
    )
    _emit_progress(inferred_total, inferred_total, start_time, stage="complete")

    return FullScanInferenceResult(
        oh5_path=resolved_path,
        scan_name=scan_name or resolved_path.stem,
        nx=reader.nx,
        ny=reader.ny,
        total_pixels=reader.total_pixels,
        header_total_pixels=reader.header_total_pixels,
        class_names=class_names,
        predicted_indices=predicted_indices,
        confidences=confidences,
        phase_counts=phase_counts,
        phase_fractions=phase_fractions,
        mean_confidence=mean_confidence,
        euler_rows_deg=euler_rows_deg,
        euler_source_unit=reader.euler_unit,
        euler_convention=reader.euler_convention,
        rows=rows,
    )


def _scan_summary_row(
    spec: ScanInferenceSpec,
    *,
    file_path: Path,
    repo_root: Path,
    total_pixels: int,
    eligible_pixels: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    confidences = [float(row["confidence"]) for row in rows]
    predicted_counts: dict[str, int] = {}
    for row in rows:
        predicted = str(row["predicted_phase"])
        predicted_counts[predicted] = predicted_counts.get(predicted, 0) + 1

    labeled_rows = [row for row in rows if str(row.get("expected_phase", "")).strip()]
    accuracy = None
    correct_predictions = None
    if labeled_rows:
        correct_predictions = sum(1 for row in labeled_rows if bool(row["is_correct"]))
        accuracy = correct_predictions / max(1, len(labeled_rows))

    dominant_phase = ""
    if predicted_counts:
        dominant_phase = sorted(predicted_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    return {
        "scan_name": spec.scan_name,
        "oh5_path": rel_path(file_path, repo_root),
        "expected_phase": spec.expected_phase or "",
        "total_pixels": int(total_pixels),
        "eligible_pixels": int(eligible_pixels),
        "sampled_patterns": int(len(rows)),
        "correct_predictions": "" if correct_predictions is None else int(correct_predictions),
        "accuracy": "" if accuracy is None else round(float(accuracy), 6),
        "mean_confidence": "" if not confidences else round(float(np.mean(confidences)), 6),
        "dominant_predicted_phase": dominant_phase,
        "predicted_phase_counts": "; ".join(f"{phase}:{count}" for phase, count in sorted(predicted_counts.items())),
    }


def _summary_markdown(
    *,
    result: Oh5InferenceResult,
    model: LoadedModel,
    pattern_rows: list[dict[str, Any]],
    scan_rows: list[dict[str, Any]],
    quality_policy: QualityPolicy,
) -> str:
    lines = [
        "# Sampled .oh5 Inference Summary",
        "",
        f"- Model run: `{model.run_dir}`",
        f"- Model family: `{model.model_family}`",
        f"- Model name: `{model.model_name}`",
        f"- Processed scans: {result.processed_scans}",
        f"- Sampled patterns: {result.sampled_patterns}",
        f"- Quality expression: `{quality_policy.expression or 'none'}`",
    ]
    if result.labeled_accuracy is not None:
        lines.append(f"- Overall labeled accuracy: {result.labeled_accuracy:.4f}")
    else:
        lines.append("- Overall labeled accuracy: not available (no expected phases provided)")

    if scan_rows:
        lines.extend(["", "## Per-Scan Summary", "", "| Scan | Expected | Sampled | Accuracy | Dominant Prediction |", "| --- | --- | ---: | ---: | --- |"])
        for row in scan_rows:
            accuracy = row["accuracy"] if row["accuracy"] != "" else "n/a"
            lines.append(
                f"| {row['scan_name']} | {row['expected_phase'] or 'n/a'} | {row['sampled_patterns']} | {accuracy} | {row['dominant_predicted_phase'] or 'n/a'} |"
            )
    if pattern_rows:
        lines.extend(["", f"Detailed per-pattern rows are in `{result.patterns_csv}`."])
    return "\n".join(lines) + "\n"


def _prediction_json_rows(pattern_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pattern_rows:
        rows.append(
            {
                "oh5_file": Path(str(row["oh5_path"])).name,
                "x": int(row["x"]),
                "y": int(row["y"]),
                "index": int(row["pattern_index"]),
                "predicted_phase": str(row["predicted_phase"]),
                "score": round(float(row["confidence"]), 6),
            }
        )
    return rows


def run_oh5_sample_inference(
    *,
    config_path: Path,
    repo_root: Path,
    debug: bool,
    logger: logging.Logger | None = None,
) -> Oh5InferenceResult:
    """Run sampled inference across `.oh5` scans using a trained CNN model."""

    log = logger or logging.getLogger("ml_oh5_inference")
    cfg = load_yaml(config_path)
    config_dir = config_path.resolve().parent

    run_dir = resolve_path(str(get_required(cfg, "run_dir", where="oh5 inference config")), base_dir=config_dir, repo_root=repo_root)
    checkpoint_name = str(cfg.get("checkpoint", "best_checkpoint.pt"))
    device = str(cfg.get("device", "auto"))
    output_dir = _resolve_output_dir(cfg, config_dir=config_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_model(run_dir=run_dir, repo_root=repo_root, checkpoint_name=checkpoint_name, device=device)
    if model.model_family != "simple_cnn":
        raise ValueError(
            f"This workflow supports only simple_cnn models for now; loaded model family '{model.model_family}'."
        )

    input_root, scan_specs, default_samples_per_scan, seed = _load_scan_specs(cfg, config_dir=config_dir, repo_root=repo_root)
    quality_policy = quality_policy_from_config(cfg.get("quality_filters") if isinstance(cfg.get("quality_filters"), dict) else {})
    strict_sampling = bool(cfg.get("strict_sampling", False))
    rng = np.random.default_rng(seed)

    pattern_rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    skipped_scans: list[dict[str, Any]] = []

    for spec in scan_specs:
        if not spec.file_path.exists():
            raise FileNotFoundError(f".oh5 file not found: {spec.file_path}")
        if spec.file_path.suffix.lower() != ".oh5":
            raise ValueError(f"Expected .oh5 file, got: {spec.file_path}")

        requested_samples = spec.samples_per_scan if spec.samples_per_scan is not None else default_samples_per_scan
        if requested_samples <= 0:
            raise ValueError(f"samples_per_scan must be > 0 for scan {spec.scan_name}")

        log.info("Scanning %s", spec.file_path)
        with Oh5ScanReader(spec.file_path) as reader:
            if not reader.pattern_present:
                skipped_scans.append({"scan_name": spec.scan_name, "reason": "pattern_dataset_missing", "oh5_path": str(spec.file_path)})
                log.warning("Skipping %s: pattern dataset missing", spec.file_path)
                continue

            eligible, counts = _eligible_indices(reader, quality_policy)
            if len(eligible) < requested_samples:
                if strict_sampling:
                    raise ValueError(
                        f"Scan {spec.scan_name} has only {len(eligible)} eligible pixels but {requested_samples} were requested"
                    )
                log.warning(
                    "Scan %s has only %d eligible pixels; sampling all available instead of requested %d",
                    spec.scan_name,
                    len(eligible),
                    requested_samples,
                )

            if not eligible:
                skipped_scans.append({"scan_name": spec.scan_name, "reason": "no_pixels_pass_quality_filters", "oh5_path": str(spec.file_path)})
                log.warning("Skipping %s: no pixels passed quality filters", spec.file_path)
                continue

            selected = sorted(rng.choice(np.asarray(eligible, dtype=np.int64), size=min(requested_samples, len(eligible)), replace=False).tolist())
            scan_pattern_rows = [
                _prediction_row(loaded=model, reader=reader, spec=spec, flat_index=int(flat_index), repo_root=repo_root)
                for flat_index in selected
            ]
            pattern_rows.extend(scan_pattern_rows)
            scan_rows.append(
                _scan_summary_row(
                    spec,
                    file_path=spec.file_path,
                    repo_root=repo_root,
                    total_pixels=counts["total_pixels"],
                    eligible_pixels=counts["eligible_pixels"],
                    rows=scan_pattern_rows,
                )
            )

    patterns_csv = output_dir / "sample_predictions.csv"
    predictions_json = output_dir / "sample_predictions.json"
    scans_csv = output_dir / "scan_summary.csv"
    summary_json = output_dir / "summary.json"
    manifest_json = output_dir / "manifest.json"
    summary_md = output_dir / "summary.md"

    labeled_rows = [row for row in pattern_rows if str(row.get("expected_phase", "")).strip()]
    labeled_accuracy = None
    if labeled_rows:
        labeled_accuracy = sum(1 for row in labeled_rows if bool(row["is_correct"])) / max(1, len(labeled_rows))

    result = Oh5InferenceResult(
        output_dir=output_dir,
        patterns_csv=patterns_csv,
        predictions_json=predictions_json,
        scans_csv=scans_csv,
        summary_json=summary_json,
        manifest_json=manifest_json,
        summary_md=summary_md,
        processed_scans=len(scan_rows),
        sampled_patterns=len(pattern_rows),
        labeled_accuracy=labeled_accuracy,
    )

    write_records_csv(patterns_csv, pattern_rows)
    write_records_csv(scans_csv, scan_rows)
    prediction_rows = _prediction_json_rows(pattern_rows)
    write_json(
        predictions_json,
        {
            "schema_version": "phase_id_xcorr.ml_oh5_sample_predictions.v1",
            "created_utc": _now_iso_utc(),
            "model": {
                "run_dir": rel_path(model.run_dir, repo_root),
                "checkpoint_path": rel_path(model.checkpoint_path, repo_root),
                "model_family": model.model_family,
                "model_name": model.model_name,
                "class_names": model.class_names,
            },
            "quality_filters": {
                "expression": quality_policy.expression,
                "resolved_expression": quality_policy.resolved_expression,
            },
            "sampling": {
                "seed": seed,
                "default_samples_per_scan": default_samples_per_scan,
                "strict_sampling": strict_sampling,
            },
            "processed_scans": len(scan_rows),
            "sampled_patterns": len(pattern_rows),
            "predictions": prediction_rows,
        },
    )

    summary_payload = {
        "schema_version": "phase_id_xcorr.ml_oh5_inference_summary.v1",
        "created_utc": _now_iso_utc(),
        "model": {
            "run_dir": rel_path(model.run_dir, repo_root),
            "checkpoint_path": rel_path(model.checkpoint_path, repo_root),
            "model_family": model.model_family,
            "model_name": model.model_name,
            "class_names": model.class_names,
        },
        "input_root": None if input_root is None else rel_path(input_root, repo_root),
        "quality_filters": {
            "expression": quality_policy.expression,
            "resolved_expression": quality_policy.resolved_expression,
        },
        "sampling": {
            "seed": seed,
            "default_samples_per_scan": default_samples_per_scan,
            "strict_sampling": strict_sampling,
        },
        "processed_scans": len(scan_rows),
        "skipped_scans": skipped_scans,
        "sampled_patterns": len(pattern_rows),
        "labeled_patterns": len(labeled_rows),
        "overall_labeled_accuracy": labeled_accuracy,
        "artifacts": {
            "sample_predictions_csv": rel_path(patterns_csv, repo_root),
            "sample_predictions_json": rel_path(predictions_json, repo_root),
            "scan_summary_csv": rel_path(scans_csv, repo_root),
            "summary_md": rel_path(summary_md, repo_root),
        },
        "scan_rows": scan_rows,
    }
    write_json(summary_json, summary_payload)

    manifest_payload = build_run_manifest(
        repo_root=repo_root,
        packet_dir=input_root or output_dir,
        out_dir=output_dir,
        debug=debug,
        extra={
            "workflow": "ml_oh5_sample_inference",
            "config_path": rel_path(config_path.resolve(), repo_root),
            "run_dir": rel_path(model.run_dir, repo_root),
            "checkpoint_name": checkpoint_name,
            "device": device,
            "processed_scans": len(scan_rows),
            "sampled_patterns": len(pattern_rows),
            "summary_json": rel_path(summary_json, repo_root),
            "sample_predictions_csv": rel_path(patterns_csv, repo_root),
            "sample_predictions_json": rel_path(predictions_json, repo_root),
            "scan_summary_csv": rel_path(scans_csv, repo_root),
        },
    )
    write_json(manifest_json, manifest_payload)
    summary_md.write_text(
        _summary_markdown(result=result, model=model, pattern_rows=pattern_rows, scan_rows=scan_rows, quality_policy=quality_policy),
        encoding="utf-8",
    )

    return result
