"""Build ML-ready datasets from `.oh5` + CSV label pairs."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image

from .config import get_required, load_yaml, resolve_path
from .dataset_io import rel_path, save_split_npz, write_json, write_records_csv
from .labels import load_label_csv
from .oh5_reader import Oh5ScanReader
from .quality import evaluate_quality, thresholds_from_config
from .split import build_split_assignments, split_config_from_yaml


@dataclass(slots=True)
class PrepareDatasetResult:
    """Paths to generated ML dataset artifacts."""

    out_dir: Path
    manifest_path: Path
    records_csv: Path
    split_npz: dict[str, Path]


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _parse_phase_map(cfg: dict[str, Any]) -> dict[str, int]:
    phase_to_label: dict[str, int] = {}

    if isinstance(cfg.get("phase_labels"), list):
        for row in cfg["phase_labels"]:
            if not isinstance(row, dict):
                raise ValueError("phase_labels entries must be mappings with keys: name, label")
            name = str(get_required(row, "name", where="phase_labels[]")).strip()
            label = int(get_required(row, "label", where="phase_labels[]"))
            phase_to_label[name] = label

    if isinstance(cfg.get("phase_to_label"), dict):
        for name, label in cfg["phase_to_label"].items():
            phase_to_label[str(name).strip()] = int(label)

    if not phase_to_label:
        raise ValueError("Config must define phase labels via 'phase_labels' or 'phase_to_label'")

    labels = list(phase_to_label.values())
    if len(labels) != len(set(labels)):
        raise ValueError("Phase labels must be unique")

    return phase_to_label


def _parse_target_hw(cfg: dict[str, Any]) -> tuple[int, int] | None:
    raw = cfg.get("target_pattern_hw")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("target_pattern_hw must be [height, width]")
    h, w = int(raw[0]), int(raw[1])
    if h <= 0 or w <= 0:
        raise ValueError("target_pattern_hw values must be positive")
    return (h, w)


def _resize_pattern(pattern: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if tuple(pattern.shape) == (h, w):
        return pattern.astype(np.float32, copy=False)

    arr8 = np.clip(pattern, 0.0, 1.0)
    arr8 = (arr8 * 255.0).round().astype(np.uint8)
    im = Image.fromarray(arr8, mode="L")
    rs = im.resize((w, h), resample=Image.BILINEAR)
    out = np.asarray(rs, dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=False) + "\n")


def _safe_eta_seconds(*, processed: int, total: int, elapsed: float) -> float | None:
    if processed <= 0 or total <= processed or elapsed <= 0:
        return None
    rate = processed / elapsed
    if rate <= 0:
        return None
    return float((total - processed) / rate)


def prepare_ml_dataset(
    *,
    config_path: Path,
    repo_root: Path,
    debug: bool,
    logger: logging.Logger | None = None,
) -> PrepareDatasetResult:
    """Run dataset preparation from config and write artifacts."""

    log = logger or logging.getLogger(__name__)

    cfg_path = config_path.resolve()
    cfg_dir = cfg_path.parent
    cfg = load_yaml(cfg_path)

    out_dir = resolve_path(
        cfg.get("output_dir", "reports/ml/datasets/default"),
        base_dir=cfg_dir,
        repo_root=repo_root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    event_log = out_dir / "events.jsonl"
    event_log.write_text("", encoding="utf-8")
    run_t0 = time.monotonic()

    def emit(event: str, **fields: Any) -> None:
        elapsed = float(time.monotonic() - run_t0)
        payload = {
            "timestamp_utc": _now_iso_utc(),
            "elapsed_seconds": elapsed,
            "event": event,
        }
        payload.update(fields)
        _append_event(event_log, payload)

    phase_to_label = _parse_phase_map(cfg)
    label_to_phase = {v: k for k, v in phase_to_label.items()}
    split_cfg = split_config_from_yaml(cfg.get("split"))
    quality_th = thresholds_from_config(cfg.get("quality_filters"))
    target_hw = _parse_target_hw(cfg)

    csv_cfg = cfg.get("label_csv", {})
    if not isinstance(csv_cfg, dict):
        raise ValueError("label_csv must be a mapping")

    sources = cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Config must provide non-empty 'sources' list")
    emit(
        "RUN_START",
        config_path=rel_path(cfg_path, repo_root),
        output_dir=rel_path(out_dir, repo_root),
        source_count=len(sources),
        strict_pattern_presence=bool(cfg.get("strict_pattern_presence", True)),
    )

    strict_pattern_presence = bool(cfg.get("strict_pattern_presence", True))
    source_summaries: list[dict[str, Any]] = []

    patterns: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    records: list[dict[str, Any]] = []

    reject_reason_counts: dict[str, int] = collections.Counter()
    accepted_per_phase: dict[str, int] = collections.Counter()
    raw_rows_total = 0

    for src_idx, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            raise ValueError("Each source entry must be a mapping")

        scan_id = str(source.get("scan_id", f"scan_{src_idx:03d}"))
        oh5_path = resolve_path(
            get_required(source, "oh5_path", where=f"sources[{src_idx - 1}]"),
            base_dir=cfg_dir,
            repo_root=repo_root,
        )
        labels_csv_path = resolve_path(
            get_required(source, "labels_csv_path", where=f"sources[{src_idx - 1}]"),
            base_dir=cfg_dir,
            repo_root=repo_root,
        )

        log.info("Source %s | oh5=%s labels=%s", scan_id, oh5_path, labels_csv_path)
        source_t0 = time.monotonic()
        emit(
            "SOURCE_START",
            scan_id=scan_id,
            source_index=src_idx,
            source_total=len(sources),
            oh5_path=rel_path(oh5_path, repo_root),
            labels_csv_path=rel_path(labels_csv_path, repo_root),
        )

        with Oh5ScanReader(oh5_path) as reader:
            meta = reader.meta()
            emit(
                "OH5_OPEN",
                scan_id=scan_id,
                nx=meta.nx,
                ny=meta.ny,
                pattern_present=bool(meta.pattern_present),
                quality_field_map=meta.quality_field_map,
            )

            if not meta.pattern_present:
                msg = f"Pattern dataset missing in {oh5_path}"
                if strict_pattern_presence:
                    emit("SOURCE_ERROR", scan_id=scan_id, reason="pattern_missing", strict_pattern_presence=True)
                    raise KeyError(msg)
                log.warning("%s; skipping source due to strict_pattern_presence=false", msg)
                source_summaries.append(
                    {
                        "scan_id": scan_id,
                        "oh5_path": rel_path(oh5_path, repo_root),
                        "labels_csv_path": rel_path(labels_csv_path, repo_root),
                        "rows_total": 0,
                        "rows_accepted": 0,
                        "rows_rejected": 0,
                        "reason": "pattern_missing",
                        "elapsed_seconds": float(time.monotonic() - source_t0),
                        "grid": {"nx": meta.nx, "ny": meta.ny},
                        "quality_field_map": meta.quality_field_map,
                    }
                )
                emit(
                    "SOURCE_END",
                    scan_id=scan_id,
                    status="skipped",
                    reason="pattern_missing",
                    elapsed_seconds=float(time.monotonic() - source_t0),
                )
                continue

            label_rows, label_summary = load_label_csv(
                csv_path=labels_csv_path,
                phase_to_label=phase_to_label,
                csv_config=csv_cfg,
            )
            raw_rows_total += label_summary.rows_total
            emit(
                "LABELS_LOADED",
                scan_id=scan_id,
                rows_total=label_summary.rows_total,
                rows_loaded=label_summary.rows_loaded,
                phase_counts=label_summary.phase_counts,
            )

            accepted = 0
            rejected = 0
            phase_counts: dict[str, int] = collections.Counter()
            source_reason_counts: dict[str, int] = collections.Counter()
            processed = 0
            progress_interval = max(1, label_summary.rows_total // 10) if label_summary.rows_total > 0 else 1

            for row in label_rows:
                processed += 1
                flat_index = row.flat_index
                if flat_index is None:
                    if row.x is None or row.y is None:
                        source_reason_counts["missing_coordinates"] += 1
                        reject_reason_counts["missing_coordinates"] += 1
                        rejected += 1
                        if processed % progress_interval == 0 or processed == label_summary.rows_total:
                            source_elapsed = float(time.monotonic() - source_t0)
                            eta = _safe_eta_seconds(
                                processed=processed,
                                total=label_summary.rows_total,
                                elapsed=source_elapsed,
                            )
                            pct = (100.0 * processed / label_summary.rows_total) if label_summary.rows_total else 100.0
                            log.info(
                                "Source %s progress %.1f%% (%d/%d) accepted=%d rejected=%d elapsed=%.2fs eta=%.2fs",
                                scan_id,
                                pct,
                                processed,
                                label_summary.rows_total,
                                accepted,
                                rejected,
                                source_elapsed,
                                eta if eta is not None else 0.0,
                            )
                            emit(
                                "SOURCE_PROGRESS",
                                scan_id=scan_id,
                                processed=processed,
                                total=label_summary.rows_total,
                                progress_pct=pct,
                                accepted=accepted,
                                rejected=rejected,
                                source_elapsed_seconds=source_elapsed,
                                eta_seconds=eta,
                            )
                        continue
                    flat_index = reader.xy_to_flat(row.x, row.y)

                quality_row = reader.read_quality_row(flat_index=flat_index)
                decision = evaluate_quality(quality_row, quality_th)
                if not decision.accept:
                    for reason in decision.reasons:
                        source_reason_counts[reason] += 1
                        reject_reason_counts[reason] += 1
                    rejected += 1
                    if processed % progress_interval == 0 or processed == label_summary.rows_total:
                        source_elapsed = float(time.monotonic() - source_t0)
                        eta = _safe_eta_seconds(
                            processed=processed,
                            total=label_summary.rows_total,
                            elapsed=source_elapsed,
                        )
                        pct = (100.0 * processed / label_summary.rows_total) if label_summary.rows_total else 100.0
                        log.info(
                            "Source %s progress %.1f%% (%d/%d) accepted=%d rejected=%d elapsed=%.2fs eta=%.2fs",
                            scan_id,
                            pct,
                            processed,
                            label_summary.rows_total,
                            accepted,
                            rejected,
                            source_elapsed,
                            eta if eta is not None else 0.0,
                        )
                        emit(
                            "SOURCE_PROGRESS",
                            scan_id=scan_id,
                            processed=processed,
                            total=label_summary.rows_total,
                            progress_pct=pct,
                            accepted=accepted,
                            rejected=rejected,
                            source_elapsed_seconds=source_elapsed,
                            eta_seconds=eta,
                        )
                    continue

                pattern = reader.read_pattern(flat_index=flat_index)
                if target_hw is not None:
                    pattern = _resize_pattern(pattern, target_hw)

                if pattern.ndim != 2:
                    source_reason_counts["pattern_not_2d"] += 1
                    reject_reason_counts["pattern_not_2d"] += 1
                    rejected += 1
                    if processed % progress_interval == 0 or processed == label_summary.rows_total:
                        source_elapsed = float(time.monotonic() - source_t0)
                        eta = _safe_eta_seconds(
                            processed=processed,
                            total=label_summary.rows_total,
                            elapsed=source_elapsed,
                        )
                        pct = (100.0 * processed / label_summary.rows_total) if label_summary.rows_total else 100.0
                        log.info(
                            "Source %s progress %.1f%% (%d/%d) accepted=%d rejected=%d elapsed=%.2fs eta=%.2fs",
                            scan_id,
                            pct,
                            processed,
                            label_summary.rows_total,
                            accepted,
                            rejected,
                            source_elapsed,
                            eta if eta is not None else 0.0,
                        )
                        emit(
                            "SOURCE_PROGRESS",
                            scan_id=scan_id,
                            processed=processed,
                            total=label_summary.rows_total,
                            progress_pct=pct,
                            accepted=accepted,
                            rejected=rejected,
                            source_elapsed_seconds=source_elapsed,
                            eta_seconds=eta,
                        )
                    continue

                sample_id = f"{scan_id}__{row.sample_id}"
                x, y = reader.flat_to_xy(flat_index)

                patterns.append(pattern.astype(np.float32, copy=False))
                labels.append(int(row.label))
                sample_ids.append(sample_id)
                accepted += 1
                phase_counts[row.phase_name] += 1
                accepted_per_phase[row.phase_name] += 1

                records.append(
                    {
                        "sample_id": sample_id,
                        "scan_id": scan_id,
                        "oh5_path": rel_path(oh5_path, repo_root),
                        "labels_csv_path": rel_path(labels_csv_path, repo_root),
                        "source_row_index": row.row_index,
                        "x": x,
                        "y": y,
                        "flat_index": flat_index,
                        "phase_name": row.phase_name,
                        "label": int(row.label),
                        "confidence_index": _safe_float(quality_row.get("confidence_index")),
                        "image_quality": _safe_float(quality_row.get("image_quality")),
                        "fit": _safe_float(quality_row.get("fit")),
                        "valid": bool(quality_row.get("valid")) if quality_row.get("valid") is not None else None,
                        "split": "",
                    }
                )
                if processed % progress_interval == 0 or processed == label_summary.rows_total:
                    source_elapsed = float(time.monotonic() - source_t0)
                    eta = _safe_eta_seconds(
                        processed=processed,
                        total=label_summary.rows_total,
                        elapsed=source_elapsed,
                    )
                    pct = (100.0 * processed / label_summary.rows_total) if label_summary.rows_total else 100.0
                    log.info(
                        "Source %s progress %.1f%% (%d/%d) accepted=%d rejected=%d elapsed=%.2fs eta=%.2fs",
                        scan_id,
                        pct,
                        processed,
                        label_summary.rows_total,
                        accepted,
                        rejected,
                        source_elapsed,
                        eta if eta is not None else 0.0,
                    )
                    emit(
                        "SOURCE_PROGRESS",
                        scan_id=scan_id,
                        processed=processed,
                        total=label_summary.rows_total,
                        progress_pct=pct,
                        accepted=accepted,
                        rejected=rejected,
                        source_elapsed_seconds=source_elapsed,
                        eta_seconds=eta,
                    )

            source_summaries.append(
                {
                    "scan_id": scan_id,
                    "oh5_path": rel_path(oh5_path, repo_root),
                    "labels_csv_path": rel_path(labels_csv_path, repo_root),
                    "rows_total": label_summary.rows_total,
                    "rows_accepted": accepted,
                    "rows_rejected": rejected,
                    "phase_counts": dict(phase_counts),
                    "reject_reason_counts": dict(source_reason_counts),
                    "elapsed_seconds": float(time.monotonic() - source_t0),
                    "grid": {"nx": meta.nx, "ny": meta.ny},
                    "pattern_shape": list(meta.pattern_shape) if meta.pattern_shape else None,
                    "quality_field_map": meta.quality_field_map,
                }
            )
            emit(
                "SOURCE_END",
                scan_id=scan_id,
                status="completed",
                rows_total=label_summary.rows_total,
                rows_accepted=accepted,
                rows_rejected=rejected,
                elapsed_seconds=float(time.monotonic() - source_t0),
            )

    if not patterns:
        raise RuntimeError("No patterns accepted after applying filters")

    # Validate homogeneous pattern shape after optional resizing.
    shape0 = tuple(patterns[0].shape)
    for arr in patterns:
        if tuple(arr.shape) != shape0:
            raise ValueError(
                "Pattern shapes differ after preprocessing. "
                "Set target_pattern_hw in config to enforce a common shape."
            )

    split_assignments = build_split_assignments(labels, split_cfg)
    for rec, split_name in zip(records, split_assignments, strict=True):
        rec["split"] = split_name
    emit("SPLIT_ASSIGNMENT_COMPLETE", total_records=len(records))

    patterns_np = np.stack(patterns, axis=0).astype(np.float32, copy=False)
    labels_np = np.asarray(labels, dtype=np.int64)

    split_npz_paths: dict[str, Path] = {}
    split_counts: dict[str, int] = {}
    split_phase_counts: dict[str, dict[str, int]] = {}

    for split_name in ("train", "val", "test"):
        idx = np.asarray([i for i, s in enumerate(split_assignments) if s == split_name], dtype=np.int64)
        if idx.size == 0:
            split_patterns = np.zeros((0,) + shape0, dtype=np.float32)
            split_labels = np.zeros((0,), dtype=np.int64)
            split_sample_ids: list[str] = []
        else:
            split_patterns = patterns_np[idx]
            split_labels = labels_np[idx]
            split_sample_ids = [sample_ids[int(i)] for i in idx.tolist()]

        path = out_dir / "splits" / f"{split_name}.npz"
        save_split_npz(path, patterns=split_patterns, labels=split_labels, sample_ids=split_sample_ids)
        split_npz_paths[split_name] = path
        split_counts[split_name] = int(idx.size)

        phase_counts = collections.Counter(label_to_phase[int(v)] for v in split_labels.tolist())
        split_phase_counts[split_name] = dict(phase_counts)
        emit(
            "SPLIT_WRITE_COMPLETE",
            split=split_name,
            count=int(idx.size),
            npz_path=rel_path(path, repo_root),
            phase_counts=split_phase_counts[split_name],
        )

    records_csv = out_dir / "records.csv"
    write_records_csv(records_csv, records)
    emit("RECORDS_WRITE_COMPLETE", records_csv=rel_path(records_csv, repo_root), count=len(records))

    sanity_checks = {
        "phase_label_mapping_defined": bool(phase_to_label),
        "phase_label_mapping_unique": len(phase_to_label) == len(set(phase_to_label.values())),
        "source_list_non_empty": len(sources) > 0,
        "pattern_shape_uniform_after_preprocessing": True,
        "all_records_assigned_split": all(bool(rec["split"]) for rec in records),
        "strict_pattern_presence": bool(strict_pattern_presence),
    }

    manifest = {
        "schema_version": "phase_id_xcorr.ml_dataset_manifest.v1",
        "timestamp_utc": _now_iso_utc(),
        "git_commit": _git_commit(repo_root),
        "repo_root": rel_path(repo_root, repo_root),
        "config_path": rel_path(cfg_path, repo_root),
        "output_dir": rel_path(out_dir, repo_root),
        "debug": bool(debug),
        "phase_to_label": phase_to_label,
        "label_to_phase": {str(k): v for k, v in label_to_phase.items()},
        "source_count": len(source_summaries),
        "source_summaries": source_summaries,
        "pattern_shape_hw": list(shape0),
        "num_samples_total": int(len(records)),
        "split_counts": split_counts,
        "split_phase_counts": split_phase_counts,
        "accepted_per_phase": dict(accepted_per_phase),
        "raw_label_rows_total": int(raw_rows_total),
        "rejected_reason_counts": dict(reject_reason_counts),
        "quality_filters": {
            "confidence_index_min": quality_th.confidence_index_min,
            "image_quality_min": quality_th.image_quality_min,
            "fit_max": quality_th.fit_max,
            "valid_required": quality_th.valid_required,
        },
        "split_policy": {
            "train": split_cfg.train,
            "val": split_cfg.val,
            "test": split_cfg.test,
            "seed": split_cfg.seed,
            "stratified": split_cfg.stratified,
        },
        "timing": {
            "total_elapsed_seconds": float(time.monotonic() - run_t0),
        },
        "sanity_checks": sanity_checks,
        "artifacts": {
            "records_csv": rel_path(records_csv, repo_root),
            "train_npz": rel_path(split_npz_paths["train"], repo_root),
            "val_npz": rel_path(split_npz_paths["val"], repo_root),
            "test_npz": rel_path(split_npz_paths["test"], repo_root),
            "event_log_jsonl": rel_path(event_log, repo_root),
        },
    }

    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    emit(
        "RUN_END",
        status="completed",
        total_records=len(records),
        split_counts=split_counts,
        manifest_path=rel_path(manifest_path, repo_root),
        total_elapsed_seconds=float(time.monotonic() - run_t0),
    )

    log.info(
        "Prepared ML dataset | total=%d train=%d val=%d test=%d shape=%s",
        len(records),
        split_counts.get("train", 0),
        split_counts.get("val", 0),
        split_counts.get("test", 0),
        shape0,
    )

    return PrepareDatasetResult(
        out_dir=out_dir,
        manifest_path=manifest_path,
        records_csv=records_csv,
        split_npz=split_npz_paths,
    )
