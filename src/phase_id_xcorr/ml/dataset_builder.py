"""Build ML-ready datasets from `.oh5` scans under config-driven label modes."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterator

import numpy as np

from .config import get_required, load_yaml, resolve_path
from .dataset_io import rel_path, save_split_npz, write_json, write_records_csv
from .labels import load_label_csv
from .oh5_reader import Oh5ScanReader
from .preprocessing_policy import apply_preprocessing, resolve_preprocessing_policy
from .quality import evaluate_quality, quality_policy_from_config
from .split import build_split_assignments, split_config_from_yaml


SOURCE_MODE_CSV = "oh5_csv_labels"
SOURCE_MODE_SINGLE_PHASE = "single_phase_scan_map"
SUPPORTED_SOURCE_MODES = (SOURCE_MODE_CSV, SOURCE_MODE_SINGLE_PHASE)


@dataclass(slots=True)
class PrepareDatasetResult:
    """Paths to generated ML dataset artifacts."""

    out_dir: Path
    manifest_path: Path
    records_csv: Path
    split_npz: dict[str, Path]


@dataclass(slots=True)
class _SourceRow:
    """Normalized per-pixel source row used across all input modes."""

    row_index: int
    sample_id: str
    x: int | None
    y: int | None
    flat_index: int | None
    phase_name: str
    label: int


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


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _format_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


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


def _parse_input_mode(*, cfg: dict[str, Any], sources: list[Any]) -> str:
    """Resolve source-label mode from explicit config or source schema."""

    mode_raw = cfg.get("input_mode")
    if mode_raw is not None:
        mode = str(mode_raw).strip()
        if mode not in SUPPORTED_SOURCE_MODES:
            raise ValueError(
                f"Unsupported input_mode '{mode}'. "
                f"Expected one of: {', '.join(SUPPORTED_SOURCE_MODES)}"
            )
        return mode

    has_csv = [isinstance(src, dict) and str(src.get("labels_csv_path", "")).strip() != "" for src in sources]
    has_scan_phase = [
        isinstance(src, dict)
        and (
            str(src.get("phase_name", "")).strip() != ""
            or str(src.get("phase_label", "")).strip() != ""
        )
        for src in sources
    ]

    if has_csv and all(has_csv):
        return SOURCE_MODE_CSV
    if has_scan_phase and all(has_scan_phase) and not any(has_csv):
        return SOURCE_MODE_SINGLE_PHASE

    raise ValueError(
        "Could not infer input mode from sources. Set 'input_mode' explicitly to "
        f"'{SOURCE_MODE_CSV}' or '{SOURCE_MODE_SINGLE_PHASE}'."
    )


def _normalize_v3_sources(cfg: dict[str, Any], *, base_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    sources = cfg.get("sources")
    if isinstance(sources, list) and sources:
        return [dict(s) if isinstance(s, dict) else s for s in sources]

    file_list = cfg.get("listOfFiles")
    if not isinstance(file_list, list) or not file_list:
        return []

    data_source_folder = cfg.get("data_source_folder", ".")
    source_root = resolve_path(data_source_folder, base_dir=base_dir, repo_root=repo_root)
    allow_filename_phase = bool(cfg.get("allow_filename_phase_fallback", False))

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(file_list):
        if isinstance(row, str):
            entry = {"file": row}
        elif isinstance(row, dict):
            entry = dict(row)
        else:
            raise ValueError("listOfFiles entries must be strings or mappings")

        rel_file = str(entry.get("file") or entry.get("oh5") or entry.get("oh5_path") or "").strip()
        if not rel_file:
            raise ValueError(f"listOfFiles[{idx}] missing file path")

        src: dict[str, Any] = {
            "scan_id": str(entry.get("scan_id") or Path(rel_file).stem),
            "oh5_path": str((source_root / rel_file).resolve()),
        }
        if str(entry.get("labels_csv_path", "")).strip():
            src["labels_csv_path"] = str((source_root / str(entry["labels_csv_path"])).resolve())
        if entry.get("phase_label") is not None:
            src["phase_label"] = entry.get("phase_label")
        if entry.get("phase_name") is not None:
            src["phase_name"] = entry.get("phase_name")

        if allow_filename_phase and ("phase_name" not in src and "phase_label" not in src):
            stem = Path(rel_file).stem
            token = stem.split("__")[-1]
            if token:
                src["phase_name"] = token

        normalized.append(src)
    return normalized


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_source_phase(
    *,
    source: dict[str, Any],
    phase_to_label: dict[str, int],
    label_to_phase: dict[int, str],
    where: str,
) -> tuple[str, int]:
    """Resolve one source mapping into canonical (phase_name, label)."""

    phase_name: str | None = None
    phase_label: int | None = None

    raw_name = source.get("phase_name")
    if raw_name is not None and str(raw_name).strip() != "":
        phase_name = str(raw_name).strip()
        if phase_name not in phase_to_label:
            raise ValueError(
                f"Unknown phase_name '{phase_name}' in {where}. "
                f"Configured phase names: {sorted(phase_to_label)}"
            )
        phase_label = int(phase_to_label[phase_name])

    raw_label = source.get("phase_label")
    if raw_label is not None and str(raw_label).strip() != "":
        parsed_label = int(float(str(raw_label)))
        if parsed_label not in label_to_phase:
            raise ValueError(
                f"Unknown phase_label '{parsed_label}' in {where}. "
                f"Configured labels: {sorted(label_to_phase)}"
            )
        parsed_name = label_to_phase[parsed_label]
        if phase_name is not None and parsed_name != phase_name:
            raise ValueError(
                f"Inconsistent phase mapping in {where}: phase_name='{phase_name}' "
                f"but phase_label={parsed_label} maps to '{parsed_name}'."
            )
        phase_name = parsed_name
        phase_label = parsed_label

    if phase_name is None or phase_label is None:
        raise ValueError(
            f"{where} requires 'phase_name' or 'phase_label' for "
            f"input_mode='{SOURCE_MODE_SINGLE_PHASE}'."
        )

    return phase_name, phase_label


def _iter_single_phase_rows(
    *,
    total_pixels: int,
    phase_name: str,
    phase_label: int,
) -> Iterator[_SourceRow]:
    """Yield synthetic rows for scans where one file corresponds to one phase."""

    for flat_index in range(total_pixels):
        yield _SourceRow(
            row_index=int(flat_index + 1),
            sample_id=f"pix_{flat_index:09d}",
            x=None,
            y=None,
            flat_index=int(flat_index),
            phase_name=phase_name,
            label=int(phase_label),
        )


def _emit_source_progress(
    *,
    emit,
    log: logging.Logger,
    scan_id: str,
    processed: int,
    total: int,
    accepted: int,
    rejected: int,
    source_t0: float,
) -> None:
    """Emit periodic progress and ETA for one source."""

    source_elapsed = float(time.monotonic() - source_t0)
    eta = _safe_eta_seconds(
        processed=processed,
        total=total,
        elapsed=source_elapsed,
    )
    pct = (100.0 * processed / total) if total else 100.0
    log.info(
        "Source %s progress %.1f%% (%d/%d) accepted=%d rejected=%d elapsed=%.2fs eta=%.2fs",
        scan_id,
        pct,
        processed,
        total,
        accepted,
        rejected,
        source_elapsed,
        eta if eta is not None else 0.0,
    )
    emit(
        "SOURCE_PROGRESS",
        scan_id=scan_id,
        processed=processed,
        total=total,
        progress_pct=pct,
        accepted=accepted,
        rejected=rejected,
        source_elapsed_seconds=source_elapsed,
        eta_seconds=eta,
    )


def _write_dataset_html_summary(
    *,
    path: Path,
    manifest: dict[str, Any],
    repo_root: Path,
) -> None:
    raw_total = int(manifest.get("raw_input_rows_total", 0))
    accepted = int(manifest.get("num_samples_total", 0))
    rejected = max(0, raw_total - accepted)
    accept_frac = (accepted / raw_total) if raw_total > 0 else 0.0
    reject_frac = (rejected / raw_total) if raw_total > 0 else 0.0
    split_counts = manifest.get("split_counts") or {}
    split_percentages = manifest.get("split_phase_percentages") or {}
    phase_stats = manifest.get("phase_statistics") or {}

    source_rows = []
    for summary in manifest.get("source_summaries", []):
        if not isinstance(summary, dict):
            continue
        source_rows.append(
            "<tr>"
            f"<td>{summary.get('scan_id', '')}</td>"
            f"<td>{summary.get('phase_name', '')}</td>"
            f"<td>{summary.get('rows_total', 0)}</td>"
            f"<td>{summary.get('rows_accepted', 0)}</td>"
            f"<td>{summary.get('rows_rejected', 0)}</td>"
            f"<td>{float(summary.get('accept_fraction', 0.0)):.3f}</td>"
            f"<td>{summary.get('oh5_path', '')}</td>"
            "</tr>"
        )

    reason_rows = []
    for reason, count in sorted((manifest.get("rejected_reason_counts") or {}).items()):
        reason_rows.append(f"<tr><td>{reason}</td><td>{count}</td></tr>")

    phase_rows = []
    for phase_name, stats in phase_stats.items():
        if not isinstance(stats, dict):
            continue
        ci = stats.get("confidence_index") or {}
        fit = stats.get("fit") or {}
        iq = stats.get("image_quality") or {}
        intensity = stats.get("intensity_distribution") or {}
        ci_mean = "" if ci.get("mean") is None else f"{float(ci['mean']):.4f}"
        ci_median = "" if ci.get("median") is None else f"{float(ci['median']):.4f}"
        ci_std = "" if ci.get("std") is None else f"{float(ci['std']):.4f}"
        fit_mean = "" if fit.get("mean") is None else f"{float(fit['mean']):.4f}"
        fit_median = "" if fit.get("median") is None else f"{float(fit['median']):.4f}"
        fit_std = "" if fit.get("std") is None else f"{float(fit['std']):.4f}"
        iq_mean = "" if iq.get("mean") is None else f"{float(iq['mean']):.4f}"
        iq_median = "" if iq.get("median") is None else f"{float(iq['median']):.4f}"
        iq_std = "" if iq.get("std") is None else f"{float(iq['std']):.4f}"
        phase_rows.append(
            "<tr>"
            f"<td>{phase_name}</td>"
            f"<td>{stats.get('accepted_count', 0)}</td>"
            f"<td>{float(stats.get('accepted_fraction_of_dataset', 0.0)):.3f}</td>"
            f"<td>{stats.get('train_count', 0)}</td>"
            f"<td>{float(stats.get('train_fraction_within_split', 0.0)):.3f}</td>"
            f"<td>{stats.get('val_count', 0)}</td>"
            f"<td>{float(stats.get('val_fraction_within_split', 0.0)):.3f}</td>"
            f"<td>{stats.get('test_count', 0)}</td>"
            f"<td>{float(stats.get('test_fraction_within_split', 0.0)):.3f}</td>"
            f"<td>{ci_mean}</td>"
            f"<td>{ci_median}</td>"
            f"<td>{ci_std}</td>"
            f"<td>{fit_mean}</td>"
            f"<td>{fit_median}</td>"
            f"<td>{fit_std}</td>"
            f"<td>{iq_mean}</td>"
            f"<td>{iq_median}</td>"
            f"<td>{iq_std}</td>"
            f"<td>{intensity.get('mode_intensity_value', '')}</td>"
            f"<td>{intensity.get('mode_pixel_count', '')}</td>"
            "</tr>"
        )

    split_rows = []
    for split_name in ("train", "val", "test"):
        counts = manifest.get("split_phase_counts", {}).get(split_name, {})
        pct_map = split_percentages.get(split_name, {})
        if not isinstance(counts, dict):
            continue
        for phase_name, count in counts.items():
            split_rows.append(
                "<tr>"
                f"<td>{split_name}</td>"
                f"<td>{phase_name}</td>"
                f"<td>{count}</td>"
                f"<td>{float(pct_map.get(phase_name, 0.0)):.3f}</td>"
                "</tr>"
            )

    artifacts = manifest.get("artifacts", {})
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ML Dataset Preparation Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d0d0d0; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .metric {{ border: 1px solid #d0d0d0; padding: 12px; border-radius: 6px; background: #fafafa; }}
    .metric .value {{ font-size: 24px; font-weight: 700; }}
    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>ML Dataset Preparation Summary</h1>
  <p><strong>Config:</strong> <code>{manifest.get('config_path', '')}</code></p>
  <p><strong>Filter:</strong> <code>{((manifest.get('quality_filters') or {}).get('expression')) or 'threshold-only policy'}</code></p>
  <div class="metric-grid">
    <div class="metric"><div>Raw scan pixels</div><div class="value">{raw_total}</div></div>
    <div class="metric"><div>Accepted</div><div class="value">{accepted}</div></div>
    <div class="metric"><div>Rejected</div><div class="value">{rejected}</div></div>
    <div class="metric"><div>Accepted fraction</div><div class="value">{accept_frac:.3f}</div></div>
  </div>
  <p><strong>Rejected fraction:</strong> {reject_frac:.3f}</p>
  <p><strong>Pattern shape:</strong> {manifest.get('pattern_shape_hw')}</p>
  <p><strong>Split counts:</strong> {split_counts}</p>

  <h2>Phase-Wise Accepted Dataset Statistics</h2>
  <table>
    <thead>
      <tr>
        <th>Phase</th><th>Accepted</th><th>Accepted frac</th>
        <th>Train</th><th>Train frac</th><th>Val</th><th>Val frac</th><th>Test</th><th>Test frac</th>
        <th>CI mean</th><th>CI median</th><th>CI std</th>
        <th>Fit mean</th><th>Fit median</th><th>Fit std</th>
        <th>IQ mean</th><th>IQ median</th><th>IQ std</th>
        <th>Mode intensity</th><th>Mode pixel count</th>
      </tr>
    </thead>
    <tbody>
      {''.join(phase_rows) if phase_rows else '<tr><td colspan="20">No phase statistics available</td></tr>'}
    </tbody>
  </table>

  <h2>Split Composition</h2>
  <table>
    <thead>
      <tr><th>Split</th><th>Phase</th><th>Count</th><th>Fraction within split</th></tr>
    </thead>
    <tbody>
      {''.join(split_rows) if split_rows else '<tr><td colspan="4">No split composition available</td></tr>'}
    </tbody>
  </table>

  <h2>Per-Source Summary</h2>
  <table>
    <thead>
      <tr><th>Scan ID</th><th>Phase</th><th>Raw pixels</th><th>Accepted</th><th>Rejected</th><th>Accepted fraction</th><th>OH5 path</th></tr>
    </thead>
    <tbody>
      {''.join(source_rows)}
    </tbody>
  </table>

  <h2>Reject Reasons</h2>
  <table>
    <thead>
      <tr><th>Reason</th><th>Count</th></tr>
    </thead>
    <tbody>
      {''.join(reason_rows) if reason_rows else '<tr><td colspan="2">None</td></tr>'}
    </tbody>
  </table>

  <h2>Artifacts</h2>
  <table>
    <thead>
      <tr><th>Name</th><th>Path</th></tr>
    </thead>
    <tbody>
      <tr><td>Manifest JSON</td><td>{rel_path(path.with_name('manifest.json'), repo_root)}</td></tr>
      <tr><td>Records CSV</td><td>{artifacts.get('records_csv', '')}</td></tr>
      <tr><td>Train split</td><td>{artifacts.get('train_npz', '')}</td></tr>
      <tr><td>Val split</td><td>{artifacts.get('val_npz', '')}</td></tr>
      <tr><td>Test split</td><td>{artifacts.get('test_npz', '')}</td></tr>
      <tr><td>Event log</td><td>{artifacts.get('event_log_jsonl', '')}</td></tr>
      <tr><td>Resolved config</td><td>{artifacts.get('resolved_config_json', '')}</td></tr>
    </tbody>
  </table>
</body>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


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
    quality_policy = quality_policy_from_config(cfg.get("quality_filters"))
    preprocessing_policy = resolve_preprocessing_policy(cfg)

    csv_cfg = cfg.get("label_csv", {})
    if not isinstance(csv_cfg, dict):
        raise ValueError("label_csv must be a mapping")

    sources = _normalize_v3_sources(cfg, base_dir=cfg_dir, repo_root=repo_root)
    if not isinstance(sources, list) or not sources:
        raise ValueError("Config must provide non-empty 'sources' list or v3 listOfFiles")
    input_mode = _parse_input_mode(cfg=cfg, sources=sources)

    emit(
        "RUN_START",
        config_path=rel_path(cfg_path, repo_root),
        output_dir=rel_path(out_dir, repo_root),
        source_count=len(sources),
        input_mode=input_mode,
        strict_pattern_presence=bool(cfg.get("strict_pattern_presence", True)),
        preprocessing_policy=preprocessing_policy.to_dict(),
        preprocessing_fingerprint=preprocessing_policy.fingerprint(),
        quality_expression=quality_policy.expression,
        quality_expression_resolved=quality_policy.resolved_expression,
    )

    strict_pattern_presence = bool(cfg.get("strict_pattern_presence", True))
    source_summaries: list[dict[str, Any]] = []

    patterns: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    records: list[dict[str, Any]] = []

    reject_reason_counts: dict[str, int] = collections.Counter()
    accepted_per_phase: dict[str, int] = collections.Counter()
    phase_intensity_counts: dict[str, np.ndarray] = {}
    phase_intensity_bit_depth: dict[str, int] = {}
    raw_rows_total = 0

    for src_idx, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            raise ValueError("Each source entry must be a mapping")
        where = f"sources[{src_idx - 1}]"

        scan_id = str(source.get("scan_id", f"scan_{src_idx:03d}"))
        oh5_path = resolve_path(
            get_required(source, "oh5_path", where=where),
            base_dir=cfg_dir,
            repo_root=repo_root,
        )
        oh5_path_rel = rel_path(oh5_path, repo_root)

        labels_csv_path: Path | None = None
        labels_csv_path_rel = ""
        source_phase_name: str | None = None
        source_phase_label: int | None = None

        if input_mode == SOURCE_MODE_CSV:
            labels_csv_path = resolve_path(
                get_required(source, "labels_csv_path", where=where),
                base_dir=cfg_dir,
                repo_root=repo_root,
            )
            labels_csv_path_rel = rel_path(labels_csv_path, repo_root)
            log.info(
                "Source %s | mode=%s oh5=%s labels=%s",
                scan_id,
                input_mode,
                oh5_path,
                labels_csv_path,
            )
        else:
            source_phase_name, source_phase_label = _resolve_source_phase(
                source=source,
                phase_to_label=phase_to_label,
                label_to_phase=label_to_phase,
                where=where,
            )
            log.info(
                "Source %s | mode=%s oh5=%s phase=%s(%d)",
                scan_id,
                input_mode,
                oh5_path,
                source_phase_name,
                source_phase_label,
            )

        source_t0 = time.monotonic()
        source_start_payload: dict[str, Any] = {
            "scan_id": scan_id,
            "source_index": src_idx,
            "source_total": len(sources),
            "source_mode": input_mode,
            "oh5_path": oh5_path_rel,
        }
        if labels_csv_path_rel:
            source_start_payload["labels_csv_path"] = labels_csv_path_rel
        if source_phase_name is not None and source_phase_label is not None:
            source_start_payload["phase_name"] = source_phase_name
            source_start_payload["phase_label"] = source_phase_label
        emit("SOURCE_START", **source_start_payload)

        with Oh5ScanReader(oh5_path) as reader:
            meta = reader.meta()
            emit(
                "OH5_OPEN",
                scan_id=scan_id,
                source_mode=input_mode,
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
                        "source_mode": input_mode,
                        "oh5_path": oh5_path_rel,
                        "labels_csv_path": labels_csv_path_rel,
                        "phase_name": source_phase_name,
                        "phase_label": source_phase_label,
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

            if input_mode == SOURCE_MODE_CSV:
                if labels_csv_path is None:
                    raise RuntimeError("Internal error: labels_csv_path missing for CSV mode")
                label_rows, label_summary = load_label_csv(
                    csv_path=labels_csv_path,
                    phase_to_label=phase_to_label,
                    csv_config=csv_cfg,
                )
            else:
                label_rows, label_summary = [], None

            rows_total = 0
            label_rows_iter: Iterator[_SourceRow]
            if input_mode == SOURCE_MODE_CSV:
                if label_summary is None:
                    raise RuntimeError("Internal error: CSV label summary missing")
                rows_total = int(label_summary.rows_total)
                raw_rows_total += rows_total
                emit(
                    "LABELS_LOADED",
                    scan_id=scan_id,
                    source_mode=input_mode,
                    rows_total=label_summary.rows_total,
                    rows_loaded=label_summary.rows_loaded,
                    phase_counts=label_summary.phase_counts,
                )

                def _csv_rows_iter() -> Iterator[_SourceRow]:
                    for row in label_rows:
                        yield _SourceRow(
                            row_index=int(row.row_index),
                            sample_id=str(row.sample_id),
                            x=row.x,
                            y=row.y,
                            flat_index=row.flat_index,
                            phase_name=str(row.phase_name),
                            label=int(row.label),
                        )

                label_rows_iter = _csv_rows_iter()
            else:
                if source_phase_name is None or source_phase_label is None:
                    raise RuntimeError("Internal error: source phase mapping missing")
                rows_total = int(meta.total_pixels)
                raw_rows_total += rows_total
                emit(
                    "LABELS_LOADED",
                    scan_id=scan_id,
                    source_mode=input_mode,
                    rows_total=rows_total,
                    rows_loaded=rows_total,
                    phase_counts={source_phase_name: rows_total},
                )
                label_rows_iter = _iter_single_phase_rows(
                    total_pixels=rows_total,
                    phase_name=source_phase_name,
                    phase_label=source_phase_label,
                )

            accepted = 0
            rejected = 0
            phase_counts: dict[str, int] = collections.Counter()
            source_reason_counts: dict[str, int] = collections.Counter()
            processed = 0
            progress_interval = max(1, rows_total // 10) if rows_total > 0 else 1

            for row in label_rows_iter:
                processed += 1
                flat_index = row.flat_index
                if flat_index is None:
                    if row.x is None or row.y is None:
                        source_reason_counts["missing_coordinates"] += 1
                        reject_reason_counts["missing_coordinates"] += 1
                        rejected += 1
                        if processed % progress_interval == 0 or processed == rows_total:
                            _emit_source_progress(
                                emit=emit,
                                log=log,
                                scan_id=scan_id,
                                processed=processed,
                                total=rows_total,
                                accepted=accepted,
                                rejected=rejected,
                                source_t0=source_t0,
                            )
                        continue
                    flat_index = reader.xy_to_flat(row.x, row.y)

                quality_row = reader.read_quality_row(flat_index=flat_index)
                decision = evaluate_quality(quality_row, quality_policy)
                if not decision.accept:
                    for reason in decision.reasons:
                        source_reason_counts[reason] += 1
                        reject_reason_counts[reason] += 1
                    rejected += 1
                    if processed % progress_interval == 0 or processed == rows_total:
                        _emit_source_progress(
                            emit=emit,
                            log=log,
                            scan_id=scan_id,
                            processed=processed,
                            total=rows_total,
                            accepted=accepted,
                            rejected=rejected,
                            source_t0=source_t0,
                        )
                    continue

                pattern = reader.read_pattern(flat_index=flat_index)
                if reader.pattern_bit_depth is not None:
                    bit_depth = int(reader.pattern_bit_depth)
                    intensity_levels = int((2 ** bit_depth) - 1)
                    phase_intensity_bit_depth[row.phase_name] = max(
                        int(phase_intensity_bit_depth.get(row.phase_name, 0)),
                        bit_depth,
                    )
                    if row.phase_name not in phase_intensity_counts:
                        phase_intensity_counts[row.phase_name] = np.zeros((intensity_levels + 1,), dtype=np.int64)
                    quantized = np.clip(
                        np.rint(pattern * float(intensity_levels)),
                        0,
                        intensity_levels,
                    ).astype(np.int32, copy=False)
                    phase_intensity_counts[row.phase_name] += np.bincount(
                        quantized.ravel(),
                        minlength=intensity_levels + 1,
                    )
                pattern = apply_preprocessing(pattern, preprocessing_policy)

                if pattern.ndim != 2:
                    source_reason_counts["pattern_not_2d"] += 1
                    reject_reason_counts["pattern_not_2d"] += 1
                    rejected += 1
                    if processed % progress_interval == 0 or processed == rows_total:
                        _emit_source_progress(
                            emit=emit,
                            log=log,
                            scan_id=scan_id,
                            processed=processed,
                            total=rows_total,
                            accepted=accepted,
                            rejected=rejected,
                            source_t0=source_t0,
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
                        "source_mode": input_mode,
                        "oh5_path": oh5_path_rel,
                        "labels_csv_path": labels_csv_path_rel,
                        "source_phase_name": source_phase_name,
                        "source_phase_label": source_phase_label,
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
                if processed % progress_interval == 0 or processed == rows_total:
                    _emit_source_progress(
                        emit=emit,
                        log=log,
                        scan_id=scan_id,
                        processed=processed,
                        total=rows_total,
                        accepted=accepted,
                        rejected=rejected,
                        source_t0=source_t0,
                    )

            source_summaries.append(
                {
                    "scan_id": scan_id,
                    "source_mode": input_mode,
                    "oh5_path": oh5_path_rel,
                    "labels_csv_path": labels_csv_path_rel,
                    "phase_name": source_phase_name,
                    "phase_label": source_phase_label,
                    "rows_total": rows_total,
                    "rows_accepted": accepted,
                    "rows_rejected": rejected,
                    "accept_fraction": _format_pct(accepted, rows_total),
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
                source_mode=input_mode,
                rows_total=rows_total,
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

    group_values: list[str] | None = None
    if split_cfg.group_key:
        group_values = [str(rec.get(split_cfg.group_key, "")) for rec in records]
        if any(v == "" for v in group_values):
            raise ValueError(f"split.group_key='{split_cfg.group_key}' references missing record fields")

    split_assignments = build_split_assignments(labels, split_cfg, groups=group_values)
    for rec, split_name in zip(records, split_assignments, strict=True):
        rec["split"] = split_name
    emit("SPLIT_ASSIGNMENT_COMPLETE", total_records=len(records), group_key=split_cfg.group_key)

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

    split_phase_percentages: dict[str, dict[str, float]] = {}
    for split_name, phase_counts in split_phase_counts.items():
        total = int(split_counts.get(split_name, 0))
        split_phase_percentages[split_name] = {
            str(phase_name): _format_pct(int(count), total)
            for phase_name, count in phase_counts.items()
        }

    records_csv = out_dir / "records.csv"
    write_records_csv(records_csv, records)
    emit("RECORDS_WRITE_COMPLETE", records_csv=rel_path(records_csv, repo_root), count=len(records))

    per_phase_metric_values: dict[str, dict[str, list[float]]] = {
        phase_name: {
            "confidence_index": [],
            "image_quality": [],
            "fit": [],
        }
        for phase_name in phase_to_label
    }
    for rec in records:
        phase_name = str(rec.get("phase_name", ""))
        if phase_name not in per_phase_metric_values:
            per_phase_metric_values[phase_name] = {
                "confidence_index": [],
                "image_quality": [],
                "fit": [],
            }
        for metric_name in ("confidence_index", "image_quality", "fit"):
            value = _safe_float(rec.get(metric_name))
            if value is not None:
                per_phase_metric_values[phase_name][metric_name].append(value)

    phase_statistics: dict[str, dict[str, Any]] = {}
    for phase_name in sorted(accepted_per_phase):
        hist = phase_intensity_counts.get(phase_name)
        mode_intensity_value = None
        mode_pixel_count = None
        if hist is not None and hist.size:
            mode_idx = int(np.argmax(hist))
            mode_intensity_value = mode_idx
            mode_pixel_count = int(hist[mode_idx])
        phase_statistics[phase_name] = {
            "accepted_count": int(accepted_per_phase.get(phase_name, 0)),
            "accepted_fraction_of_dataset": _format_pct(int(accepted_per_phase.get(phase_name, 0)), len(records)),
            "train_count": int(split_phase_counts.get("train", {}).get(phase_name, 0)),
            "train_fraction_within_split": float(split_phase_percentages.get("train", {}).get(phase_name, 0.0)),
            "val_count": int(split_phase_counts.get("val", {}).get(phase_name, 0)),
            "val_fraction_within_split": float(split_phase_percentages.get("val", {}).get(phase_name, 0.0)),
            "test_count": int(split_phase_counts.get("test", {}).get(phase_name, 0)),
            "test_fraction_within_split": float(split_phase_percentages.get("test", {}).get(phase_name, 0.0)),
            "confidence_index": _summarize_numeric(per_phase_metric_values.get(phase_name, {}).get("confidence_index", [])),
            "image_quality": _summarize_numeric(per_phase_metric_values.get(phase_name, {}).get("image_quality", [])),
            "fit": _summarize_numeric(per_phase_metric_values.get(phase_name, {}).get("fit", [])),
            "intensity_distribution": {
                "bit_depth": int(phase_intensity_bit_depth.get(phase_name, 0)) or None,
                "mode_intensity_value": mode_intensity_value,
                "mode_pixel_count": mode_pixel_count,
            },
        }

    resolved_config_path = out_dir / "resolved_config.json"
    write_json(resolved_config_path, cfg)

    sanity_checks = {
        "phase_label_mapping_defined": bool(phase_to_label),
        "phase_label_mapping_unique": len(phase_to_label) == len(set(phase_to_label.values())),
        "input_mode_supported": input_mode in SUPPORTED_SOURCE_MODES,
        "source_list_non_empty": len(sources) > 0,
        "pattern_shape_uniform_after_preprocessing": True,
        "all_records_assigned_split": all(bool(rec["split"]) for rec in records),
        "strict_pattern_presence": bool(strict_pattern_presence),
    }
    source_mode_counts = collections.Counter(summary["source_mode"] for summary in source_summaries)

    manifest = {
        "schema_version": "phase_id_xcorr.ml_dataset_manifest.v1",
        "timestamp_utc": _now_iso_utc(),
        "git_commit": _git_commit(repo_root),
        "repo_root": rel_path(repo_root, repo_root),
        "config_path": rel_path(cfg_path, repo_root),
        "output_dir": rel_path(out_dir, repo_root),
        "debug": bool(debug),
        "input_mode": input_mode,
        "phase_to_label": phase_to_label,
        "label_to_phase": {str(k): v for k, v in label_to_phase.items()},
        "source_count": len(source_summaries),
        "source_mode_counts": dict(source_mode_counts),
        "source_summaries": source_summaries,
        "pattern_shape_hw": list(shape0),
        "num_samples_total": int(len(records)),
        "split_counts": split_counts,
        "split_phase_counts": split_phase_counts,
        "split_phase_percentages": split_phase_percentages,
        "accepted_per_phase": dict(accepted_per_phase),
        "phase_statistics": phase_statistics,
        "raw_input_rows_total": int(raw_rows_total),
        "raw_label_rows_total": int(raw_rows_total),
        "rejected_reason_counts": dict(reject_reason_counts),
        "quality_filters": {
            "thresholds": {
                "confidence_index_min": quality_policy.thresholds.confidence_index_min,
                "image_quality_min": quality_policy.thresholds.image_quality_min,
                "fit_max": quality_policy.thresholds.fit_max,
                "valid_required": quality_policy.thresholds.valid_required,
            },
            "expression": quality_policy.expression,
            "resolved_expression": quality_policy.resolved_expression,
            "resolved_aliases": quality_policy.field_aliases,
        },
        "preprocessing_policy": preprocessing_policy.to_dict(),
        "preprocessing_fingerprint": preprocessing_policy.fingerprint(),
        "split_policy": {
            "train": split_cfg.train,
            "val": split_cfg.val,
            "test": split_cfg.test,
            "seed": split_cfg.seed,
            "stratified": split_cfg.stratified,
            "group_key": split_cfg.group_key,
            "max_val_samples": split_cfg.max_val_samples,
            "max_test_samples": split_cfg.max_test_samples,
            "val_samples_per_phase": split_cfg.val_samples_per_phase,
            "test_samples_per_phase": split_cfg.test_samples_per_phase,
        },
        "timing": {
            "total_elapsed_seconds": float(time.monotonic() - run_t0),
        },
        "sanity_checks": sanity_checks,
        "provenance": {
            "code_version": {"git_commit": _git_commit(repo_root)},
            "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
            "input_file_hashes": {
                rel_path(resolve_path(get_required(s, "oh5_path", where="source"), base_dir=cfg_dir, repo_root=repo_root), repo_root): _file_sha256(resolve_path(get_required(s, "oh5_path", where="source"), base_dir=cfg_dir, repo_root=repo_root))
                for s in sources if isinstance(s, dict)
            },
        },
        "artifacts": {
            "resolved_config_json": rel_path(resolved_config_path, repo_root),
            "records_csv": rel_path(records_csv, repo_root),
            "train_npz": rel_path(split_npz_paths["train"], repo_root),
            "val_npz": rel_path(split_npz_paths["val"], repo_root),
            "test_npz": rel_path(split_npz_paths["test"], repo_root),
            "event_log_jsonl": rel_path(event_log, repo_root),
        },
    }

    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    summary_html = out_dir / "summary.html"
    _write_dataset_html_summary(path=summary_html, manifest=manifest, repo_root=repo_root)
    manifest["artifacts"]["summary_html"] = rel_path(summary_html, repo_root)
    write_json(manifest_path, manifest)
    emit(
        "RUN_END",
        status="completed",
        total_records=len(records),
        split_counts=split_counts,
        manifest_path=rel_path(manifest_path, repo_root),
        summary_html=rel_path(summary_html, repo_root),
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
