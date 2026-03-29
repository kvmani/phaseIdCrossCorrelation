"""Diagnostic gallery backend for cross-condition `.oh5` pattern inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import get_required, load_yaml, resolve_path
from .dataset_io import rel_path, write_json
from .inference import LoadedModel, load_trained_model, predict_pattern_array
from .oh5_reader import Oh5ScanReader
from .quality import QualityPolicy, evaluate_quality, quality_policy_from_config


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class DiagnosticPredictionFilters:
    """Model-evidence filters used to decide which patterns are eligible for display."""

    min_confidence: float | None = None
    min_margin: float | None = None


@dataclass(slots=True)
class DiagnosticSelectionPolicy:
    """Deterministic auto-selection policy for the gallery."""

    patterns_per_source: int = 5
    seed: int = 0
    strategy: str = "random"


@dataclass(slots=True)
class DiagnosticSourceSpec:
    """One `.oh5` source shown in the diagnostic gallery."""

    group_name: str
    file_path: Path
    scan_id: str
    phase_name: str | None = None
    expected_phase: str | None = None
    manual_indices: list[int] = field(default_factory=list)

    @property
    def source_key(self) -> str:
        return f"{self.group_name}:{self.scan_id}"

    @property
    def display_name(self) -> str:
        return self.phase_name or self.scan_id or self.file_path.stem


@dataclass(slots=True)
class DiagnosticPatternRecord:
    """One displayed pattern tile and its evidence payload."""

    record_id: str
    source_key: str
    group_name: str
    source_label: str
    scan_id: str
    phase_name: str | None
    expected_phase: str | None
    file_path: Path
    flat_index: int
    x: int
    y: int
    selected_by: str
    filter_pass: bool
    filter_reasons: list[str]
    confidence_index: float | None
    image_quality: float | None
    fit: float | None
    valid: bool | None
    predicted_phase: str
    predicted_index: int
    confidence: float
    margin: float
    probabilities: dict[str, float]
    raw_pattern: np.ndarray
    preprocessed_pattern: np.ndarray


@dataclass(slots=True)
class DiagnosticSourceResult:
    """Rendered patterns and source-level summary for one `.oh5` file."""

    spec: DiagnosticSourceSpec
    total_pixels: int
    eligible_pixels: int
    candidate_pixels: int
    selected_pixels: int
    skipped_reason: str | None = None
    records: list[DiagnosticPatternRecord] = field(default_factory=list)
    manual_records: list[DiagnosticPatternRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def display_records(self) -> list[DiagnosticPatternRecord]:
        merged: dict[tuple[str, int], DiagnosticPatternRecord] = {}
        for record in self.records:
            merged[(record.source_key, record.flat_index)] = record
        for record in self.manual_records:
            merged[(record.source_key, record.flat_index)] = record
        return sorted(
            merged.values(),
            key=lambda rec: (rec.group_name, rec.source_label, rec.flat_index, rec.selected_by != "manual"),
        )


@dataclass(slots=True)
class DiagnosticGallerySession:
    """Resolved diagnostic gallery session."""

    config_path: Path
    output_dir: Path
    loaded_model: LoadedModel
    quality_policy: QualityPolicy
    prediction_filters: DiagnosticPredictionFilters
    selection_policy: DiagnosticSelectionPolicy
    source_results: dict[str, DiagnosticSourceResult]
    source_order: list[str]
    gallery_title: str
    manifest_path: Path | None = None
    contact_sheet_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def records(self) -> list[DiagnosticPatternRecord]:
        out: list[DiagnosticPatternRecord] = []
        for source_key in self.source_order:
            result = self.source_results.get(source_key)
            if result is None:
                continue
            out.extend(result.display_records)
        return out

    @property
    def tile_count(self) -> int:
        return len(self.records)


def _as_int(value: Any, *, field_name: str) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return out


def _top2_margin(probabilities: dict[str, float]) -> float:
    values = sorted((float(v) for v in probabilities.values()), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(values[0] - values[1])


def _resolve_model_bundle(
    cfg: dict[str, Any],
    *,
    config_dir: Path,
    repo_root: Path,
    debug: bool,
    logger: logging.Logger,
) -> LoadedModel:
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    run_dir_value = cfg.get("run_dir")
    if run_dir_value in (None, ""):
        run_dir_value = model_cfg.get("run_dir")
    if run_dir_value is None:
        raise ValueError("Diagnostic gallery config requires run_dir or model.run_dir")
    run_dir = resolve_path(str(run_dir_value), base_dir=config_dir, repo_root=repo_root)
    checkpoint_value = cfg.get("checkpoint")
    if checkpoint_value in (None, ""):
        checkpoint_value = model_cfg.get("checkpoint", "best_checkpoint.pt")
    device_value = cfg.get("device")
    if device_value in (None, ""):
        device_value = model_cfg.get("device", "auto")
    checkpoint_name = str(checkpoint_value)
    device = str(device_value)
    loaded = load_trained_model(
        run_dir=run_dir,
        repo_root=repo_root,
        checkpoint_name=checkpoint_name,
        device=device,
    )
    logger.info(
        "Loaded diagnostic model run=%s family=%s name=%s",
        loaded.run_dir,
        loaded.model_family,
        loaded.model_name,
    )
    return loaded


def _resolve_output_dir(cfg: dict[str, Any], *, config_dir: Path, repo_root: Path) -> Path:
    output_dir_value = get_required(cfg, "output_dir", where="diagnostic gallery config")
    return resolve_path(str(output_dir_value), base_dir=config_dir, repo_root=repo_root)


def _parse_selection_policy(cfg: dict[str, Any]) -> DiagnosticSelectionPolicy:
    sampling_cfg = cfg.get("sampling") if isinstance(cfg.get("sampling"), dict) else {}
    patterns_per_source = int(sampling_cfg.get("patterns_per_source", sampling_cfg.get("samples_per_scan", cfg.get("samples_per_scan", 5))))
    seed = int(sampling_cfg.get("seed", cfg.get("seed", 0)))
    strategy = str(sampling_cfg.get("strategy", cfg.get("selection_strategy", "random"))).strip().lower() or "random"
    if patterns_per_source <= 0:
        raise ValueError("sampling.patterns_per_source must be > 0")
    if strategy not in {"random", "top_confidence", "top_margin"}:
        raise ValueError("sampling.strategy must be one of: random, top_confidence, top_margin")
    return DiagnosticSelectionPolicy(patterns_per_source=patterns_per_source, seed=seed, strategy=strategy)


def _parse_prediction_filters(cfg: dict[str, Any]) -> DiagnosticPredictionFilters:
    pred_cfg = cfg.get("prediction_filters") if isinstance(cfg.get("prediction_filters"), dict) else {}
    min_confidence = pred_cfg.get("min_confidence")
    min_margin = pred_cfg.get("min_margin")
    return DiagnosticPredictionFilters(
        min_confidence=None if min_confidence in (None, "") else float(min_confidence),
        min_margin=None if min_margin in (None, "") else float(min_margin),
    )


def _normalize_source_entry(group_name: str, entry: dict[str, Any], *, idx: int, config_dir: Path, repo_root: Path) -> DiagnosticSourceSpec:
    file_value = get_required(entry, "file", where=f"{group_name}[{idx}]")
    file_path = resolve_path(str(file_value), base_dir=config_dir, repo_root=repo_root)
    scan_id = str(entry.get("scan_id") or Path(str(file_value)).stem).strip() or file_path.stem
    phase_name = entry.get("phase_name")
    expected_phase = entry.get("expected_phase")
    manual_indices_raw = entry.get("manual_indices", [])
    manual_indices = [
        _as_int(value, field_name=f"{group_name}[{idx}].manual_indices[{manual_idx}]")
        for manual_idx, value in enumerate(manual_indices_raw)
    ] if isinstance(manual_indices_raw, list) else []
    return DiagnosticSourceSpec(
        group_name=group_name,
        file_path=file_path,
        scan_id=scan_id,
        phase_name=None if phase_name in (None, "") else str(phase_name).strip(),
        expected_phase=None if expected_phase in (None, "") else str(expected_phase).strip(),
        manual_indices=manual_indices,
    )


def _load_source_specs(cfg: dict[str, Any], *, config_dir: Path, repo_root: Path) -> list[DiagnosticSourceSpec]:
    groups_cfg = cfg.get("source_groups") if isinstance(cfg.get("source_groups"), dict) else {}
    reference_rows = groups_cfg.get("reference") if isinstance(groups_cfg.get("reference"), list) else cfg.get("reference_sources")
    unknown_rows = groups_cfg.get("unknown") if isinstance(groups_cfg.get("unknown"), list) else cfg.get("unknown_sources")

    source_specs: list[DiagnosticSourceSpec] = []
    for group_name, rows in (("reference", reference_rows), ("unknown", unknown_rows)):
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{group_name}[{idx}] must be a mapping")
            source_specs.append(_normalize_source_entry(group_name, row, idx=idx, config_dir=config_dir, repo_root=repo_root))

    if not source_specs:
        raise ValueError("No source_groups/reference_sources/unknown_sources entries found")
    return source_specs


def _quality_values(reader: Oh5ScanReader, flat_index: int) -> dict[str, float | bool | None]:
    row = reader.read_quality_row(flat_index=flat_index)
    return {
        "confidence_index": row.get("confidence_index"),
        "image_quality": row.get("image_quality"),
        "fit": row.get("fit"),
        "valid": row.get("valid"),
    }


def _prediction_passes(
    prediction: Any,
    *,
    filters: DiagnosticPredictionFilters,
) -> tuple[bool, list[str], float]:
    reasons: list[str] = []
    probabilities = dict(prediction.probabilities)
    margin = _top2_margin(probabilities)
    if filters.min_confidence is not None and float(prediction.confidence) < float(filters.min_confidence):
        reasons.append("confidence_below_min")
    if filters.min_margin is not None and margin < float(filters.min_margin):
        reasons.append("margin_below_min")
    return len(reasons) == 0, reasons, margin


def _make_record(
    *,
    loaded_model: LoadedModel,
    reader: Oh5ScanReader,
    source: DiagnosticSourceSpec,
    flat_index: int,
    selected_by: str,
    quality_policy: QualityPolicy,
    prediction_filters: DiagnosticPredictionFilters,
    record_id: str,
) -> DiagnosticPatternRecord:
    quality = _quality_values(reader, flat_index)
    decision = evaluate_quality(quality, quality_policy)
    pattern = reader.read_pattern(flat_index=flat_index)
    prediction = predict_pattern_array(loaded=loaded_model, pattern=pattern)
    passes_prediction, prediction_reasons, margin = _prediction_passes(prediction, filters=prediction_filters)
    filter_reasons = list(decision.reasons)
    if not passes_prediction:
        filter_reasons.extend(prediction_reasons)
    filter_pass = decision.accept and passes_prediction
    x, y = reader.flat_to_xy(flat_index)
    return DiagnosticPatternRecord(
        record_id=record_id,
        source_key=source.source_key,
        group_name=source.group_name,
        source_label=source.display_name,
        scan_id=source.scan_id,
        phase_name=source.phase_name,
        expected_phase=source.expected_phase,
        file_path=source.file_path,
        flat_index=int(flat_index),
        x=int(x),
        y=int(y),
        selected_by=selected_by,
        filter_pass=filter_pass,
        filter_reasons=filter_reasons,
        confidence_index=None if quality["confidence_index"] is None else float(quality["confidence_index"]),
        image_quality=None if quality["image_quality"] is None else float(quality["image_quality"]),
        fit=None if quality["fit"] is None else float(quality["fit"]),
        valid=None if quality["valid"] is None else bool(quality["valid"]),
        predicted_phase=prediction.predicted_phase,
        predicted_index=int(prediction.predicted_index),
        confidence=float(prediction.confidence),
        margin=float(margin),
        probabilities=dict(prediction.probabilities),
        raw_pattern=np.asarray(pattern, dtype=np.float32),
        preprocessed_pattern=np.asarray(prediction.preprocessed_image, dtype=np.float32),
    )


def _select_auto_records(
    *,
    records: list[DiagnosticPatternRecord],
    selection_policy: DiagnosticSelectionPolicy,
) -> list[DiagnosticPatternRecord]:
    if not records:
        return []
    if len(records) <= selection_policy.patterns_per_source:
        return sorted(records, key=lambda rec: rec.flat_index)

    if selection_policy.strategy == "top_confidence":
        ordered = sorted(records, key=lambda rec: (-rec.confidence, -rec.margin, rec.flat_index))
        return ordered[: selection_policy.patterns_per_source]
    if selection_policy.strategy == "top_margin":
        ordered = sorted(records, key=lambda rec: (-rec.margin, -rec.confidence, rec.flat_index))
        return ordered[: selection_policy.patterns_per_source]

    rng = np.random.default_rng(selection_policy.seed)
    chosen = sorted(rng.choice(len(records), size=selection_policy.patterns_per_source, replace=False).tolist())
    return [records[idx] for idx in chosen]


def build_diagnostic_gallery_session(
    *,
    config_path: Path,
    repo_root: Path,
    debug: bool = False,
    logger: logging.Logger | None = None,
    loaded_model: LoadedModel | None = None,
) -> DiagnosticGallerySession:
    """Resolve the diagnostic gallery session from config and source files."""

    log = logger or logging.getLogger("ml_diagnostic_gallery")
    cfg = load_yaml(config_path)
    return build_diagnostic_gallery_session_from_config(
        cfg=cfg,
        config_path=config_path,
        repo_root=repo_root,
        debug=debug,
        logger=log,
        loaded_model=loaded_model,
    )


def build_diagnostic_gallery_session_from_config(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    repo_root: Path,
    debug: bool = False,
    logger: logging.Logger | None = None,
    loaded_model: LoadedModel | None = None,
) -> DiagnosticGallerySession:
    """Build a diagnostic gallery session from an already-loaded config mapping."""

    log = logger or logging.getLogger("ml_diagnostic_gallery")
    config_dir = config_path.resolve().parent

    model = loaded_model or _resolve_model_bundle(cfg, config_dir=config_dir, repo_root=repo_root, debug=debug, logger=log)
    output_dir = _resolve_output_dir(cfg, config_dir=config_dir, repo_root=repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_policy = quality_policy_from_config(cfg.get("quality_filters") if isinstance(cfg.get("quality_filters"), dict) else {})
    selection_policy = _parse_selection_policy(cfg)
    prediction_filters = _parse_prediction_filters(cfg)
    source_specs = _load_source_specs(cfg, config_dir=config_dir, repo_root=repo_root)
    source_results: dict[str, DiagnosticSourceResult] = {}
    source_order: list[str] = []

    for source in source_specs:
        log.info("Loading source %s", source.file_path)
        if not source.file_path.exists():
            raise FileNotFoundError(f".oh5 file not found: {source.file_path}")
        if source.file_path.suffix.lower() != ".oh5":
            raise ValueError(f"Expected .oh5 file, got {source.file_path}")

        with Oh5ScanReader(source.file_path) as reader:
            if not reader.pattern_present:
                result = DiagnosticSourceResult(
                    spec=source,
                    total_pixels=int(reader.total_pixels),
                    eligible_pixels=0,
                    candidate_pixels=0,
                    selected_pixels=0,
                    skipped_reason="pattern_dataset_missing",
                    meta=asdict(reader.meta()),
                )
                source_results[source.source_key] = result
                source_order.append(source.source_key)
                continue

            total_pixels = int(reader.total_pixels)
            eligible_indices: list[int] = []
            eligible_records: list[DiagnosticPatternRecord] = []
            for flat_index in range(total_pixels):
                quality = _quality_values(reader, flat_index)
                if not evaluate_quality(quality, quality_policy).accept:
                    continue
                eligible_indices.append(flat_index)
                record = _make_record(
                    loaded_model=model,
                    reader=reader,
                    source=source,
                    flat_index=flat_index,
                    selected_by="auto",
                    quality_policy=quality_policy,
                    prediction_filters=prediction_filters,
                    record_id=f"{source.source_key}:{flat_index:06d}",
                )
                if record.filter_pass:
                    eligible_records.append(record)
            selected_auto = _select_auto_records(records=eligible_records, selection_policy=selection_policy)
            if selection_policy.strategy == "random" and selected_auto:
                source_seed = selection_policy.seed + sum(ord(ch) for ch in source.source_key)
                selected_auto = _select_auto_records(
                    records=eligible_records,
                    selection_policy=DiagnosticSelectionPolicy(
                        patterns_per_source=selection_policy.patterns_per_source,
                        seed=source_seed,
                        strategy=selection_policy.strategy,
                    ),
                )
            result = DiagnosticSourceResult(
                spec=source,
                total_pixels=total_pixels,
                eligible_pixels=len(eligible_indices),
                candidate_pixels=len(eligible_records),
                selected_pixels=len(selected_auto) + len(source.manual_indices),
                records=selected_auto,
                meta=asdict(reader.meta()),
            )

            for manual_index in source.manual_indices:
                manual_record = _make_record(
                    loaded_model=model,
                    reader=reader,
                    source=source,
                    flat_index=int(manual_index),
                    selected_by="manual",
                    quality_policy=quality_policy,
                    prediction_filters=prediction_filters,
                    record_id=f"{source.source_key}:manual:{int(manual_index):06d}",
                )
                result.manual_records.append(manual_record)

            source_results[source.source_key] = result
            source_order.append(source.source_key)

    gallery_title = str(cfg.get("gallery_title", "Diagnostic Pattern Gallery")).strip() or "Diagnostic Pattern Gallery"
    return DiagnosticGallerySession(
        config_path=config_path.resolve(),
        output_dir=output_dir.resolve(),
        loaded_model=model,
        quality_policy=quality_policy,
        prediction_filters=prediction_filters,
        selection_policy=selection_policy,
        source_results=source_results,
        source_order=source_order,
        gallery_title=gallery_title,
    )


def _tile_label(record: DiagnosticPatternRecord) -> str:
    confidence = f"{record.confidence:.3f}"
    margin = f"{record.margin:.3f}"
    return f"{record.source_label} #{record.flat_index}  p={record.predicted_phase}  c={confidence}  m={margin}"


def _ensure_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default()
    except Exception:  # pragma: no cover
        return ImageFont.load_default()


def _render_sheet(
    *,
    title: str,
    rows: list[DiagnosticSourceResult],
    output_path: Path,
    tile_size: tuple[int, int],
) -> Path:
    font = _ensure_font()
    title_font = _ensure_font()
    tile_w, tile_h = tile_size
    label_h = 46
    row_header_w = 190
    gap = 14
    margin = 18

    max_cols = max((len(result.display_records) for result in rows), default=0)
    total_w = margin * 2 + row_header_w + max_cols * tile_w + max(0, max_cols - 1) * gap
    total_h = margin * 2 + 52 + sum(tile_h + label_h for _ in rows) + max(0, len(rows) - 1) * gap
    canvas = Image.new("RGB", (max(1, total_w), max(1, total_h)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill="black", font=title_font)

    y = margin + 34
    for result in rows:
        header = f"{result.spec.display_name} [{result.spec.group_name}]"
        draw.text((margin, y + 12), header, fill="black", font=font)
        draw.text((margin, y + 28), f"{result.spec.file_path.name}", fill="gray", font=font)
        x = margin + row_header_w
        for record in result.display_records:
            arr = record.preprocessed_pattern if record.preprocessed_pattern is not None else record.raw_pattern
            img = Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="L").convert("RGB")
            img = img.resize((tile_w, tile_h), resample=Image.BILINEAR)
            canvas.paste(img, (x, y))
            draw.rectangle([x, y, x + tile_w, y + tile_h], outline="black", width=1)
            label = _tile_label(record)
            label_y = y + tile_h + 4
            draw.multiline_text((x, label_y), label, fill="black", font=font, spacing=2)
            x += tile_w + gap
        y += tile_h + label_h + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _prediction_rows(records: list[DiagnosticPatternRecord], *, repo_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        out.append(
            {
                "record_id": record.record_id,
                "group_name": record.group_name,
                "source_key": record.source_key,
                "source_label": record.source_label,
                "scan_id": record.scan_id,
                "phase_name": record.phase_name or "",
                "expected_phase": record.expected_phase or "",
                "oh5_path": rel_path(record.file_path, repo_root),
                "pattern_index": int(record.flat_index),
                "x": int(record.x),
                "y": int(record.y),
                "selected_by": record.selected_by,
                "filter_pass": bool(record.filter_pass),
                "filter_reasons": list(record.filter_reasons),
                "confidence_index": record.confidence_index,
                "image_quality": record.image_quality,
                "fit": record.fit,
                "valid": record.valid,
                "predicted_phase": record.predicted_phase,
                "predicted_index": int(record.predicted_index),
                "confidence": round(float(record.confidence), 6),
                "margin": round(float(record.margin), 6),
                "probabilities": {phase: round(float(prob), 6) for phase, prob in sorted(record.probabilities.items())},
            }
        )
    return out


def export_diagnostic_gallery_artifacts(
    *,
    session: DiagnosticGallerySession,
    repo_root: Path,
    logger: logging.Logger | None = None,
) -> Path:
    """Write manifest JSON and contact-sheet PNGs for a diagnostic session."""

    log = logger or logging.getLogger("ml_diagnostic_gallery")
    output_dir = session.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_results = [session.source_results[key] for key in session.source_order if session.source_results[key].spec.group_name == "reference"]
    unknown_results = [session.source_results[key] for key in session.source_order if session.source_results[key].spec.group_name == "unknown"]
    combined_results = [session.source_results[key] for key in session.source_order]

    tile_size = (220, 220)
    if session.selection_policy.patterns_per_source <= 3:
        tile_size = (240, 240)

    contact_sheet_paths: dict[str, Path] = {}
    if reference_results:
        contact_sheet_paths["reference"] = _render_sheet(
            title=f"{session.gallery_title} - Reference Bank",
            rows=reference_results,
            output_path=output_dir / "reference_contact_sheet.png",
            tile_size=tile_size,
        )
    if unknown_results:
        contact_sheet_paths["unknown"] = _render_sheet(
            title=f"{session.gallery_title} - Unknown Bank",
            rows=unknown_results,
            output_path=output_dir / "unknown_contact_sheet.png",
            tile_size=tile_size,
        )
    contact_sheet_paths["combined"] = _render_sheet(
        title=session.gallery_title,
        rows=combined_results,
        output_path=output_dir / "combined_contact_sheet.png",
        tile_size=tile_size,
    )

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": "phase_id_xcorr.ml_diagnostic_gallery.v1",
        "created_utc": _now_iso_utc(),
        "config_path": rel_path(session.config_path, repo_root),
        "gallery_title": session.gallery_title,
        "model": {
            "run_dir": rel_path(session.loaded_model.run_dir, repo_root),
            "checkpoint_path": rel_path(session.loaded_model.checkpoint_path, repo_root),
            "model_family": session.loaded_model.model_family,
            "model_name": session.loaded_model.model_name,
            "class_names": list(session.loaded_model.class_names),
            "preprocessing_policy": session.loaded_model.preprocessing_policy.to_dict(),
            "input_mean": float(session.loaded_model.input_mean),
            "input_std": float(session.loaded_model.input_std),
        },
        "quality_filters": {
            "expression": session.quality_policy.expression,
            "resolved_expression": session.quality_policy.resolved_expression,
            "thresholds": {
                "confidence_index_min": session.quality_policy.thresholds.confidence_index_min,
                "image_quality_min": session.quality_policy.thresholds.image_quality_min,
                "fit_max": session.quality_policy.thresholds.fit_max,
                "valid_required": session.quality_policy.thresholds.valid_required,
            },
        },
        "prediction_filters": {
            "min_confidence": session.prediction_filters.min_confidence,
            "min_margin": session.prediction_filters.min_margin,
        },
        "sampling": {
            "patterns_per_source": session.selection_policy.patterns_per_source,
            "seed": session.selection_policy.seed,
            "strategy": session.selection_policy.strategy,
        },
        "source_order": list(session.source_order),
        "sources": [
            {
                "source_key": result.spec.source_key,
                "group_name": result.spec.group_name,
                "scan_id": result.spec.scan_id,
                "phase_name": result.spec.phase_name or "",
                "expected_phase": result.spec.expected_phase or "",
                "oh5_path": rel_path(result.spec.file_path, repo_root),
                "total_pixels": int(result.total_pixels),
                "eligible_pixels": int(result.eligible_pixels),
                "candidate_pixels": int(result.candidate_pixels),
                "selected_pixels": int(len(result.display_records)),
                "skipped_reason": result.skipped_reason or "",
                "meta": result.meta,
            }
            for result in combined_results
        ],
        "records": _prediction_rows(session.records, repo_root=repo_root),
        "artifacts": {
            "combined_contact_sheet": rel_path(contact_sheet_paths["combined"], repo_root),
        },
    }
    if "reference" in contact_sheet_paths:
        manifest["artifacts"]["reference_contact_sheet"] = rel_path(contact_sheet_paths["reference"], repo_root)
    if "unknown" in contact_sheet_paths:
        manifest["artifacts"]["unknown_contact_sheet"] = rel_path(contact_sheet_paths["unknown"], repo_root)

    write_json(manifest_path, manifest)

    session.manifest_path = manifest_path
    session.contact_sheet_paths = contact_sheet_paths
    log.info("Exported diagnostic gallery manifest to %s", manifest_path)
    return manifest_path


def add_manual_record(
    *,
    session: DiagnosticGallerySession,
    source_key: str,
    flat_index: int,
) -> DiagnosticPatternRecord:
    """Create and attach one manual gallery record from a selected source."""

    result = session.source_results.get(source_key)
    if result is None:
        raise KeyError(f"Unknown source key: {source_key}")

    with Oh5ScanReader(result.spec.file_path) as reader:
        if not reader.pattern_present:
            raise ValueError(f"Pattern dataset missing in {result.spec.file_path}")
        record = _make_record(
            loaded_model=session.loaded_model,
            reader=reader,
            source=result.spec,
            flat_index=int(flat_index),
            selected_by="manual",
            quality_policy=session.quality_policy,
            prediction_filters=session.prediction_filters,
            record_id=f"{result.spec.source_key}:manual:{int(flat_index):06d}",
        )

    result.manual_records.append(record)
    return record
