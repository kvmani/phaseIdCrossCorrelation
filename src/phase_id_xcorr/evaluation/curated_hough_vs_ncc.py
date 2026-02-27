"""Curated workflow comparing image-space NCC and KikuchiPy Hough-space NCC."""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from phase_id_xcorr.features import (
    HoughTransformConfig,
    KikuchiPyHoughExtractor,
    binarize_hough_map,
)
from phase_id_xcorr.preprocessing import load_image_as_float32, prepare_pattern
from phase_id_xcorr.reporting import build_run_manifest
from phase_id_xcorr.similarity import masked_ncc

PHASE_ORDER = ["fe_bcc", "fe3o4_magnetite", "feo_wustite"]


@dataclass(slots=True)
class CandidateScoreRow:
    """Per-candidate scores across metrics."""

    record_id: str
    true_phase: str
    experimental_image: str
    assumed_phase: str
    simulated_image: str
    indexing_status: str
    is_fallback_orientation: bool
    image_ncc: float
    hough_ncc_raw: float
    threshold_scores: dict[str, float]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rel(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _threshold_key(value: float) -> str:
    return f"hough_ncc_bin_t{int(round(float(value) * 1000)):03d}"


def _threshold_label(value: float) -> str:
    return f"{float(value):.3f}"


def _normalize01(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _save_gray_png(array_float01: np.ndarray, out_path: Path) -> None:
    arr = np.clip(np.asarray(array_float01, dtype=np.float32), 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr8, mode="L").save(out_path)


def _image_data_uri(path: Path, max_size: tuple[int, int] = (280, 280)) -> str:
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _rank_with_margin(rows: list[dict[str, Any]], score_key: str) -> tuple[dict[str, Any], float, float | None]:
    ranked = sorted(rows, key=lambda x: float(x["scores"][score_key]), reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    top_score = float(top["scores"][score_key])
    second_score = float(second["scores"][score_key]) if second is not None else None
    margin = top_score - second_score if second_score is not None else top_score
    return top, margin, second_score


def _method_metrics(decisions: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = [d for d in decisions if d["method"] == method]
    total = len(rows)
    correct = int(sum(1 for r in rows if bool(r["is_correct"])))
    acc = float(correct / total) if total else 0.0
    mean_margin = float(np.mean([float(r["margin"]) for r in rows])) if total else 0.0
    mean_top = float(np.mean([float(r["top_score"]) for r in rows])) if total else 0.0
    fallback_winners = int(sum(1 for r in rows if bool(r["top_is_fallback_orientation"])))
    failed_winners = int(sum(1 for r in rows if str(r["top_indexing_status"]) == "failed"))
    return {
        "method": method,
        "count": total,
        "correct": correct,
        "accuracy": acc,
        "mean_margin": mean_margin,
        "mean_top_score": mean_top,
        "fallback_winner_count": fallback_winners,
        "failed_winner_count": failed_winners,
    }


def _best_binary_method(metrics_by_method: dict[str, dict[str, Any]], thresholds: list[float]) -> tuple[str | None, float | None]:
    best_method: str | None = None
    best_threshold: float | None = None
    best_key = None
    for t in thresholds:
        method = _threshold_key(t)
        m = metrics_by_method.get(method)
        if m is None:
            continue
        key = (
            float(m.get("accuracy", 0.0)),
            float(m.get("mean_margin", 0.0)),
            float(m.get("mean_top_score", 0.0)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_method = method
            best_threshold = float(t)
    return best_method, best_threshold


def run_curated_hough_vs_ncc(
    *,
    packet_dir: Path,
    out_dir: Path,
    repo_root: Path,
    binary_thresholds: list[float],
    hough_config: HoughTransformConfig | None = None,
    debug: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run curated workflow comparing image-space NCC and Hough-space NCC."""

    log = logger or logging.getLogger(__name__)
    cfg = hough_config or HoughTransformConfig()

    packet_dir = packet_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = sorted(float(t) for t in binary_thresholds)
    if not thresholds:
        raise ValueError("At least one binary threshold is required")
    if any(t < 0.0 or t > 1.0 for t in thresholds):
        raise ValueError(f"Binary thresholds must be in [0, 1], got {thresholds}")

    exp_json = _load_json(packet_dir / "01_experimental_patterns_template.json")
    sim_json = _load_json(packet_dir / "02_simulated_patterns_template.json")
    proc_json = _load_json(packet_dir / "04_processing_template.json")
    normalization_method = str(proc_json.get("settings", {}).get("normalization_method", "minmax_inside_mask"))

    exp_by_id: dict[str, dict[str, Any]] = {}
    for rec in exp_json.get("records", []):
        exp_by_id[str(rec.get("record_id"))] = rec

    score_rows: list[CandidateScoreRow] = []
    decisions: list[dict[str, Any]] = []
    report_records: list[dict[str, Any]] = []

    hough_extractors: dict[tuple[int, int], KikuchiPyHoughExtractor] = {}

    for rec in sim_json.get("records", []):
        record_id = str(rec.get("record_id"))
        exp_rec = exp_by_id.get(record_id)
        if exp_rec is None:
            log.warning("Skipping record_id=%s (missing in experimental JSON)", record_id)
            continue

        true_phase = str(rec.get("true_phase", exp_rec.get("true_phase", "")))
        exp_rel = str(exp_rec.get("image_file"))
        exp_path = packet_dir / exp_rel
        exp_loaded = load_image_as_float32(exp_path, logger=log)
        exp_prep = prepare_pattern(exp_loaded.array, normalization_method=normalization_method, logger=log)

        shape_key = tuple(int(v) for v in exp_prep.array.shape)
        extractor = hough_extractors.get(shape_key)
        if extractor is None:
            extractor = KikuchiPyHoughExtractor(shape_key, config=cfg, logger=log)
            hough_extractors[shape_key] = extractor

        exp_hough = extractor.transform(exp_prep.array)
        exp_hough_map = _normalize01(exp_hough.hough_map)
        exp_bin_by_threshold = {
            _threshold_key(t): binarize_hough_map(exp_hough_map, t) for t in thresholds
        }

        candidate_payloads: list[dict[str, Any]] = []

        for cand in rec.get("simulated_candidates", []):
            assumed_phase = str(cand.get("assumed_phase"))
            sim_rel = str(cand.get("simulated_image"))
            sim_path = packet_dir / sim_rel
            indexing_status = str(cand.get("indexing_status", ""))
            is_fallback = bool(cand.get("is_fallback_orientation", False))

            sim_loaded = load_image_as_float32(sim_path, logger=log)
            if sim_loaded.array.shape != exp_prep.array.shape:
                raise ValueError(
                    f"Shape mismatch for record {record_id}: exp={exp_prep.array.shape} sim={sim_loaded.array.shape}"
                )
            sim_prep = prepare_pattern(sim_loaded.array, normalization_method=normalization_method, logger=log)

            mask = exp_prep.mask & sim_prep.mask
            image_ncc = masked_ncc(exp_prep.array, sim_prep.array, mask).score

            sim_hough = extractor.transform(sim_prep.array)
            sim_hough_map = _normalize01(sim_hough.hough_map)
            hough_raw_ncc = masked_ncc(
                exp_hough_map,
                sim_hough_map,
                np.ones_like(exp_hough_map, dtype=bool),
            ).score

            threshold_scores: dict[str, float] = {}
            for t in thresholds:
                key = _threshold_key(t)
                sim_bin = binarize_hough_map(sim_hough_map, t)
                threshold_scores[key] = masked_ncc(
                    exp_bin_by_threshold[key],
                    sim_bin,
                    np.ones_like(exp_hough_map, dtype=bool),
                ).score

            score_rows.append(
                CandidateScoreRow(
                    record_id=record_id,
                    true_phase=true_phase,
                    experimental_image=exp_rel,
                    assumed_phase=assumed_phase,
                    simulated_image=sim_rel,
                    indexing_status=indexing_status,
                    is_fallback_orientation=is_fallback,
                    image_ncc=image_ncc,
                    hough_ncc_raw=hough_raw_ncc,
                    threshold_scores=threshold_scores,
                )
            )

            candidate_payloads.append(
                {
                    "assumed_phase": assumed_phase,
                    "simulated_image": sim_rel,
                    "indexing_status": indexing_status,
                    "is_fallback_orientation": is_fallback,
                    "candidate_angles_degrees": cand.get("candidate_angles_degrees", {}),
                    "fallback_reason": cand.get("fallback_reason", ""),
                    "simulation_source": cand.get("simulation_source", ""),
                    "image_meta": {
                        "source_dtype": sim_loaded.source_dtype,
                        "source_bit_depth": int(sim_loaded.source_bit_depth),
                        "source_shape": [int(v) for v in sim_loaded.source_shape],
                        "value_min": float(sim_loaded.value_min),
                        "value_max": float(sim_loaded.value_max),
                    },
                    "scores": {
                        "image_ncc": float(image_ncc),
                        "hough_ncc_raw": float(hough_raw_ncc),
                        **{k: float(v) for k, v in threshold_scores.items()},
                    },
                    "arrays": {
                        "sim_pattern": sim_prep.array,
                        "sim_hough": sim_hough_map,
                    },
                }
            )

        if not candidate_payloads:
            continue

        score_keys = ["image_ncc", "hough_ncc_raw"] + [_threshold_key(t) for t in thresholds]
        per_method_decisions: dict[str, dict[str, Any]] = {}
        for score_key in score_keys:
            top, margin, second_score = _rank_with_margin(candidate_payloads, score_key)
            pred_phase = str(top["assumed_phase"])
            decision = {
                "method": score_key,
                "record_id": record_id,
                "true_phase": true_phase,
                "pred_phase": pred_phase,
                "is_correct": pred_phase == true_phase,
                "top_score": float(top["scores"][score_key]),
                "second_score": float(second_score) if second_score is not None else None,
                "margin": float(margin),
                "top_indexing_status": str(top["indexing_status"]),
                "top_is_fallback_orientation": bool(top["is_fallback_orientation"]),
            }
            decisions.append(decision)
            per_method_decisions[score_key] = decision

        report_records.append(
            {
                "record_id": record_id,
                "true_phase": true_phase,
                "experimental": {
                    "image_file": exp_rel,
                    "orientation_angles_degrees": exp_rec.get("orientation_angles_degrees", {}),
                    "image_meta": {
                        "source_dtype": exp_loaded.source_dtype,
                        "source_bit_depth": int(exp_loaded.source_bit_depth),
                        "source_shape": [int(v) for v in exp_loaded.source_shape],
                        "value_min": float(exp_loaded.value_min),
                        "value_max": float(exp_loaded.value_max),
                    },
                },
                "candidates": candidate_payloads,
                "decisions": per_method_decisions,
                "arrays": {
                    "exp_pattern": exp_prep.array,
                    "exp_hough": exp_hough_map,
                    "exp_bin_by_threshold": exp_bin_by_threshold,
                },
            }
        )

    # Method metrics
    method_keys = ["image_ncc", "hough_ncc_raw"] + [_threshold_key(t) for t in thresholds]
    metrics_by_method = {m: _method_metrics(decisions, m) for m in method_keys}
    best_binary_method, best_binary_threshold = _best_binary_method(metrics_by_method, thresholds)

    cases_total = len(report_records)
    image_accuracy = float(metrics_by_method.get("image_ncc", {}).get("accuracy", 0.0))
    hough_raw_accuracy = float(metrics_by_method.get("hough_ncc_raw", {}).get("accuracy", 0.0))
    best_bin_accuracy = (
        float(metrics_by_method.get(best_binary_method, {}).get("accuracy", 0.0))
        if best_binary_method is not None
        else 0.0
    )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "packet_dir": _rel(packet_dir, repo_root),
        "out_dir": _rel(out_dir, repo_root),
        "normalization_method": normalization_method,
        "cases_total": cases_total,
        "hough_config": {
            "n_theta": int(cfg.n_theta),
            "n_rho": int(cfg.n_rho),
            "n_bands": int(cfg.n_bands),
            "sample_tilt_deg": float(cfg.sample_tilt_deg),
            "detector_tilt_deg": float(cfg.detector_tilt_deg),
            "pc": [float(v) for v in cfg.pc],
            "use_convolved_map": bool(cfg.use_convolved_map),
            "plan_phase_name": str(cfg.plan_phase_name),
            "plan_phase_space_group": int(cfg.plan_phase_space_group),
        },
        "binary_thresholds": thresholds,
        "metrics_by_method": metrics_by_method,
        "best_binary_method": best_binary_method,
        "best_binary_threshold": best_binary_threshold,
        "headline_comparison": {
            "image_ncc_accuracy": image_accuracy,
            "hough_ncc_raw_accuracy": hough_raw_accuracy,
            "best_hough_binary_accuracy": best_bin_accuracy,
            "best_hough_binary_method": best_binary_method,
        },
    }

    # Save case images (best binary threshold shown in report)
    best_bin_key = best_binary_method if best_binary_method is not None else _threshold_key(thresholds[0])
    case_dir = out_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    for rec in report_records:
        rid = str(rec["record_id"])
        rdir = case_dir / rid
        rdir.mkdir(parents=True, exist_ok=True)

        exp_pattern_png = rdir / "exp_pattern.png"
        exp_hough_png = rdir / "exp_hough_map.png"
        exp_hough_bin_png = rdir / f"exp_hough_bin_{best_bin_key}.png"
        _save_gray_png(rec["arrays"]["exp_pattern"], exp_pattern_png)
        _save_gray_png(rec["arrays"]["exp_hough"], exp_hough_png)
        _save_gray_png(rec["arrays"]["exp_bin_by_threshold"][best_bin_key], exp_hough_bin_png)

        rec["artifacts"] = {
            "exp_pattern_png": _rel(exp_pattern_png, repo_root),
            "exp_hough_png": _rel(exp_hough_png, repo_root),
            "exp_hough_bin_png": _rel(exp_hough_bin_png, repo_root),
        }

        for cand in rec["candidates"]:
            phase = str(cand["assumed_phase"])
            safe_phase = phase.replace("/", "_").replace(" ", "_")
            sim_pattern_png = rdir / f"sim_{safe_phase}_pattern.png"
            sim_hough_png = rdir / f"sim_{safe_phase}_hough_map.png"
            sim_hough_bin_png = rdir / f"sim_{safe_phase}_hough_bin_{best_bin_key}.png"
            _save_gray_png(cand["arrays"]["sim_pattern"], sim_pattern_png)
            _save_gray_png(cand["arrays"]["sim_hough"], sim_hough_png)
            _save_gray_png(binarize_hough_map(cand["arrays"]["sim_hough"], thresholds[0] if best_binary_method is None else float(best_binary_threshold)), sim_hough_bin_png)
            cand["artifacts"] = {
                "sim_pattern_png": _rel(sim_pattern_png, repo_root),
                "sim_hough_png": _rel(sim_hough_png, repo_root),
                "sim_hough_bin_png": _rel(sim_hough_bin_png, repo_root),
            }

    # Write scores.csv
    score_cols = [
        "record_id",
        "true_phase",
        "experimental_image",
        "assumed_phase",
        "simulated_image",
        "indexing_status",
        "is_fallback_orientation",
        "image_ncc",
        "hough_ncc_raw",
    ] + [_threshold_key(t) for t in thresholds]

    scores_csv = out_dir / "scores.csv"
    with scores_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=score_cols)
        writer.writeheader()
        for row in score_rows:
            out = {
                "record_id": row.record_id,
                "true_phase": row.true_phase,
                "experimental_image": row.experimental_image,
                "assumed_phase": row.assumed_phase,
                "simulated_image": row.simulated_image,
                "indexing_status": row.indexing_status,
                "is_fallback_orientation": row.is_fallback_orientation,
                "image_ncc": row.image_ncc,
                "hough_ncc_raw": row.hough_ncc_raw,
            }
            for t in thresholds:
                key = _threshold_key(t)
                out[key] = float(row.threshold_scores[key])
            writer.writerow(out)

    # Write decisions.csv
    decisions_csv = out_dir / "decisions.csv"
    with decisions_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "method",
            "record_id",
            "true_phase",
            "pred_phase",
            "is_correct",
            "top_score",
            "second_score",
            "margin",
            "top_indexing_status",
            "top_is_fallback_orientation",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in decisions:
            writer.writerow(d)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write report data JSON (drop in-memory arrays before serialization)
    report_data = {
        "summary": summary,
        "methods": method_keys,
        "binary_threshold_labels": {k: _threshold_label(t) for k, t in [(_threshold_key(t), t) for t in thresholds]},
        "best_binary_method": best_binary_method,
        "best_binary_threshold": best_binary_threshold,
        "records": [],
    }
    for rec in report_records:
        out_rec = {
            "record_id": rec["record_id"],
            "true_phase": rec["true_phase"],
            "experimental": rec["experimental"],
            "artifacts": rec["artifacts"],
            "decisions": rec["decisions"],
            "candidates": [],
        }
        for cand in rec["candidates"]:
            out_cand = {
                "assumed_phase": cand["assumed_phase"],
                "simulated_image": cand["simulated_image"],
                "indexing_status": cand["indexing_status"],
                "is_fallback_orientation": cand["is_fallback_orientation"],
                "candidate_angles_degrees": cand["candidate_angles_degrees"],
                "fallback_reason": cand["fallback_reason"],
                "simulation_source": cand["simulation_source"],
                "image_meta": cand["image_meta"],
                "scores": cand["scores"],
                "artifacts": cand["artifacts"],
            }
            out_rec["candidates"].append(out_cand)
        report_data["records"].append(out_rec)

    report_data_path = out_dir / "report_data.json"
    report_data_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    manifest = build_run_manifest(
        repo_root=repo_root,
        packet_dir=packet_dir,
        out_dir=out_dir,
        debug=debug,
        extra={
            "workflow": "curated_hough_vs_ncc",
            "artifacts": {
                "scores_csv": _rel(scores_csv, repo_root),
                "decisions_csv": _rel(decisions_csv, repo_root),
                "summary_json": _rel(summary_path, repo_root),
                "report_data_json": _rel(report_data_path, repo_root),
                "cases_dir": _rel(case_dir, repo_root),
            },
        },
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info(
        "Curated Hough-vs-NCC completed | cases=%d image_acc=%.4f hough_raw_acc=%.4f best_bin_acc=%.4f",
        cases_total,
        image_accuracy,
        hough_raw_accuracy,
        best_bin_accuracy,
    )

    return {
        "summary_json": summary_path,
        "scores_csv": scores_csv,
        "decisions_csv": decisions_csv,
        "report_data_json": report_data_path,
        "manifest_json": manifest_path,
        "best_binary_method": best_binary_method,
    }


def build_curated_hough_vs_ncc_html(
    *,
    report_data_json: Path,
    out_html: Path,
    repo_root: Path,
) -> Path:
    """Build a single-file HTML report from `report_data.json`."""

    def method_score_breakdown(cands: list[dict[str, Any]], method: str) -> tuple[tuple[str, float] | None, list[tuple[str, float]], list[tuple[str, float]]]:
        pairs: list[tuple[str, float]] = []
        for cand in cands:
            phase = str(cand.get("assumed_phase", ""))
            try:
                score = float(cand.get("scores", {}).get(method, float("nan")))
            except Exception:
                score = float("nan")
            if not np.isfinite(score):
                continue
            pairs.append((phase, score))
        ranked = sorted(pairs, key=lambda x: x[1], reverse=True)
        winner = ranked[0] if ranked else None
        others = ranked[1:] if len(ranked) > 1 else []
        return winner, others, ranked

    def fmt_score_pairs(pairs: list[tuple[str, float]]) -> str:
        if not pairs:
            return "[]"
        return "[" + ", ".join(f"{phase} ({score:.5f})" for phase, score in pairs) + "]"

    payload = _load_json(report_data_json)
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    methods = payload.get("methods", [])
    best_binary_method = payload.get("best_binary_method")
    best_binary_threshold = payload.get("best_binary_threshold")
    threshold_labels = payload.get("binary_threshold_labels", {})

    # Summary cards
    cards = []
    headline = summary.get("headline_comparison", {})
    cards.append(("Cases", str(summary.get("cases_total", 0))))
    cards.append(("Image NCC Acc", f"{float(headline.get('image_ncc_accuracy', 0.0)):.3f}"))
    cards.append(("Hough Raw Acc", f"{float(headline.get('hough_ncc_raw_accuracy', 0.0)):.3f}"))
    cards.append(("Best Hough Bin Acc", f"{float(headline.get('best_hough_binary_accuracy', 0.0)):.3f}"))
    cards.append(("Best Bin Threshold", "n/a" if best_binary_threshold is None else f"{float(best_binary_threshold):.3f}"))
    cards_html = "".join(
        f"<div class='card'><span>{name}</span><strong>{value}</strong></div>" for name, value in cards
    )

    # Method metrics table
    metrics_by_method = summary.get("metrics_by_method", {})
    method_rows = []
    for method in methods:
        m = metrics_by_method.get(method, {})
        display_name = method
        if method.startswith("hough_ncc_bin_t"):
            display_name = f"{method} (thr={threshold_labels.get(method,'n/a')})"
        cls = "best" if method == best_binary_method else ""
        method_rows.append(
            "<tr class='{cls}'><td>{method}</td><td>{acc:.3f}</td><td>{margin:.4f}</td>"
            "<td>{fb}</td><td>{failed}</td></tr>".format(
                cls=cls,
                method=display_name,
                acc=float(m.get("accuracy", 0.0)),
                margin=float(m.get("mean_margin", 0.0)),
                fb=int(m.get("fallback_winner_count", 0)),
                failed=int(m.get("failed_winner_count", 0)),
            )
        )
    metrics_html = "".join(method_rows)

    # Per-record sections
    rec_sections = []
    blend_widget_inits: list[str] = []
    for rec in records:
        rid = str(rec.get("record_id"))
        true_phase = str(rec.get("true_phase"))
        exp = rec.get("experimental", {})
        art = rec.get("artifacts", {})
        decisions = rec.get("decisions", {})
        candidates = rec.get("candidates", [])

        exp_pattern_uri = _image_data_uri(repo_root / art["exp_pattern_png"])
        exp_hough_uri = _image_data_uri(repo_root / art["exp_hough_png"])
        exp_hbin_uri = _image_data_uri(repo_root / art["exp_hough_bin_png"])

        euler = exp.get("orientation_angles_degrees", {})
        euler_text = (
            f"phi1={euler.get('phi1', 'n/a')}, PHI={euler.get('PHI', 'n/a')}, phi2={euler.get('phi2', 'n/a')}"
        )
        img_meta = exp.get("image_meta", {})
        img_meta_text = (
            f"dtype={img_meta.get('source_dtype')} bit_depth={img_meta.get('source_bit_depth')} "
            f"shape={tuple(img_meta.get('source_shape', []))}"
        )

        dec_rows = []
        for method in methods:
            d = decisions.get(method, {})
            name = method
            if method.startswith("hough_ncc_bin_t"):
                name = f"{method} (thr={threshold_labels.get(method,'n/a')})"
            cls = "ok" if bool(d.get("is_correct", False)) else "bad"
            winner, others, _ranked = method_score_breakdown(candidates, method)
            winner_txt = "n/a" if winner is None else f"{winner[0]} ({winner[1]:.5f})"
            others_txt = fmt_score_pairs(others)
            dec_rows.append(
                f"<tr class='{cls}'><td>{name}</td><td>{d.get('pred_phase')}</td>"
                f"<td>{d.get('is_correct')}</td><td>{winner_txt}</td><td>{others_txt}</td>"
                f"<td>{float(d.get('margin',0.0)):.5f}</td><td>{d.get('top_indexing_status')}</td>"
                f"<td>{d.get('top_is_fallback_orientation')}</td></tr>"
            )

        cand_cards = []
        cand_score_rows = []
        sim_pattern_uri_by_phase: dict[str, str] = {}
        for cand in sorted(candidates, key=lambda c: PHASE_ORDER.index(str(c.get("assumed_phase"))) if str(c.get("assumed_phase")) in PHASE_ORDER else 999):
            phase = str(cand.get("assumed_phase"))
            c_art = cand.get("artifacts", {})
            c_pattern = _image_data_uri(repo_root / c_art["sim_pattern_png"])
            c_hough = _image_data_uri(repo_root / c_art["sim_hough_png"])
            c_hbin = _image_data_uri(repo_root / c_art["sim_hough_bin_png"])
            sim_pattern_uri_by_phase[phase] = c_pattern
            c_euler = cand.get("candidate_angles_degrees", {})
            c_euler_text = (
                f"phi1={c_euler.get('phi1','n/a')}, PHI={c_euler.get('PHI','n/a')}, phi2={c_euler.get('phi2','n/a')}"
            )
            c_meta = cand.get("image_meta", {})

            cand_cards.append(
                f"""
                <div class='cand-card {"failed" if str(cand.get("indexing_status")) == "failed" else ""}'>
                  <div class='triplet'>
                    <div><img src='{c_pattern}' alt='sim pattern {phase}'><div class='cap'>SIM Pattern</div></div>
                    <div><img src='{c_hough}' alt='sim hough {phase}'><div class='cap'>SIM Hough</div></div>
                    <div><img src='{c_hbin}' alt='sim hough bin {phase}'><div class='cap'>SIM Hough Bin (best thr)</div></div>
                  </div>
                  <div class='meta'><b>{phase}</b></div>
                  <div class='meta'><code>{cand.get("simulated_image")}</code></div>
                  <div class='meta'>status={cand.get("indexing_status")} fallback={cand.get("is_fallback_orientation")}</div>
                  <div class='meta'>Euler: {c_euler_text}</div>
                  <div class='meta'>dtype={c_meta.get("source_dtype")} bit_depth={c_meta.get("source_bit_depth")} shape={tuple(c_meta.get("source_shape", []))}</div>
                </div>
                """
            )

            score_cells = []
            for method in methods:
                val = cand.get("scores", {}).get(method, 0.0)
                name = method
                if method.startswith("hough_ncc_bin_t"):
                    name = f"{method} ({threshold_labels.get(method,'n/a')})"
                score_cells.append(f"<div><span>{name}</span> <b>{float(val):.5f}</b></div>")
            cand_score_rows.append(
                f"<tr><td>{phase}</td><td>{cand.get('indexing_status')}</td><td>{cand.get('is_fallback_orientation')}</td>"
                f"<td>{''.join(score_cells)}</td></tr>"
            )

        default_method = str(best_binary_method if best_binary_method in methods else "image_ncc")
        _winner, _others, ranked_for_default = method_score_breakdown(candidates, default_method)
        viewer_candidates = []
        for phase, score in ranked_for_default:
            sim_uri = sim_pattern_uri_by_phase.get(phase)
            if sim_uri is None:
                continue
            viewer_candidates.append(
                {
                    "phase": phase,
                    "score": float(score),
                    "src": sim_uri,
                }
            )
        if not viewer_candidates:
            for phase, sim_uri in sim_pattern_uri_by_phase.items():
                viewer_candidates.append({"phase": phase, "score": float("nan"), "src": sim_uri})
        default_phase = str(decisions.get(default_method, {}).get("pred_phase", ""))
        if default_phase not in {c["phase"] for c in viewer_candidates} and viewer_candidates:
            default_phase = str(viewer_candidates[0]["phase"])
        widget_id = f"hblend-{rid}"
        viewer_options = "".join(
            f"<option value='{c['phase']}'>{c['phase']} ({float(c['score']):.5f})</option>" for c in viewer_candidates
        )
        viewer_config = {
            "exp": exp_pattern_uri,
            "candidates": viewer_candidates,
            "default_phase": default_phase,
            "default_alpha": 0.5,
        }
        blend_widget_inits.append(f"initBlendWidget('{widget_id}', {json.dumps(viewer_config)});")

        rec_sections.append(
            f"""
            <section class='record'>
              <h2>Record {rid}</h2>
              <p><b>true_phase={true_phase}</b></p>
              <div class='triplet exp'>
                <div><img src='{exp_pattern_uri}' alt='exp pattern {rid}'><div class='cap'>EXP Pattern</div></div>
                <div><img src='{exp_hough_uri}' alt='exp hough {rid}'><div class='cap'>EXP Hough</div></div>
                <div><img src='{exp_hbin_uri}' alt='exp hough bin {rid}'><div class='cap'>EXP Hough Bin (best thr)</div></div>
              </div>
              <div class='meta'><code>{exp.get("image_file")}</code></div>
              <div class='meta'>Euler: {euler_text}</div>
              <div class='meta'>{img_meta_text}</div>

              <h3>Pattern Match Viewer (Optional)</h3>
              <div class='blend-widget' id='{widget_id}'>
                <div class='blend-controls'>
                  <label>Candidate
                    <select class='bw-phase'>{viewer_options}</select>
                  </label>
                  <label>Mode
                    <select class='bw-mode'>
                      <option value='overlay'>Overlay</option>
                      <option value='split'>Split 50/50</option>
                    </select>
                  </label>
                  <label>Alpha
                    <input class='bw-alpha' type='range' min='0' max='1' step='0.05' value='0.50'>
                    <span class='bw-alpha-value'>0.50</span>
                  </label>
                </div>
                <canvas class='bw-canvas'></canvas>
              </div>

              <h3>Method Decisions</h3>
              <table>
                <thead><tr><th>method</th><th>pred_phase</th><th>correct</th><th>winner_score</th><th>other_scores</th><th>margin</th><th>winner_status</th><th>winner_fallback</th></tr></thead>
                <tbody>{''.join(dec_rows)}</tbody>
              </table>

              <h3>Candidates</h3>
              <div class='cand-grid'>{''.join(cand_cards)}</div>

              <h3>Candidate Scores</h3>
              <table>
                <thead><tr><th>assumed_phase</th><th>indexing_status</th><th>fallback</th><th>scores</th></tr></thead>
                <tbody>{''.join(cand_score_rows)}</tbody>
              </table>
            </section>
            """
        )

    css = """
    body{font-family:Arial,Helvetica,sans-serif;margin:20px;background:#fafafa;color:#111}
    section{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin:0 0 16px 0}
    h1,h2,h3{margin:0 0 10px 0}
    .metrics{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px}
    .card{background:#f2f5f9;border:1px solid #dfe5ee;border-radius:6px;padding:8px}
    .card span{display:block;font-size:12px;color:#444}
    .card strong{font-size:17px}
    table{border-collapse:collapse;width:100%;margin-top:8px}
    th,td{border:1px solid #ddd;padding:6px 8px;font-size:13px;vertical-align:top}
    th{background:#f3f3f3;text-align:left}
    tr.ok{background:#edf9ef}
    tr.bad{background:#fff1f1}
    tr.best{background:#eef5ff}
    .meta{font-size:12px;color:#333;margin-top:4px}
    .triplet{display:grid;grid-template-columns:repeat(3,minmax(140px,1fr));gap:10px;margin-bottom:8px}
    .triplet img{width:100%;height:auto;border:1px solid #ccc;background:#000}
    .triplet .cap{font-size:12px;margin-top:4px}
    .cand-grid{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:10px}
    .cand-card{background:#fcfcfc;border:1px solid #ddd;border-radius:6px;padding:8px}
    .cand-card.failed{border:2px solid #c33838}
    .exp{max-width:980px}
    .blend-widget{border:1px solid #ddd;border-radius:6px;padding:10px;background:#fcfcfc;margin:10px 0}
    .blend-controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
    .blend-controls label{font-size:12px;color:#222;display:flex;gap:6px;align-items:center}
    .blend-controls select,.blend-controls input{font-size:12px}
    .bw-canvas{width:100%;max-width:360px;height:auto;border:1px solid #bbb;background:#000}
    code{background:#f5f5f5;padding:1px 4px;border-radius:4px}
    """

    blend_script = """
    <script>
    function initBlendWidget(widgetId, config){
      const root = document.getElementById(widgetId);
      if(!root || !config || !Array.isArray(config.candidates) || config.candidates.length === 0){ return; }

      const phaseSel = root.querySelector('.bw-phase');
      const modeSel = root.querySelector('.bw-mode');
      const alphaInput = root.querySelector('.bw-alpha');
      const alphaValue = root.querySelector('.bw-alpha-value');
      const canvas = root.querySelector('.bw-canvas');
      const ctx = canvas.getContext('2d');

      const expImg = new Image();
      expImg.src = config.exp;

      const simByPhase = new Map();
      config.candidates.forEach((c) => {
        const img = new Image();
        img.src = c.src;
        simByPhase.set(c.phase, img);
        img.onload = render;
      });
      expImg.onload = render;

      if(config.default_phase){ phaseSel.value = config.default_phase; }
      if(typeof config.default_alpha === 'number'){ alphaInput.value = String(config.default_alpha); }
      alphaValue.textContent = Number(alphaInput.value).toFixed(2);

      phaseSel.addEventListener('change', render);
      modeSel.addEventListener('change', render);
      alphaInput.addEventListener('input', () => {
        alphaValue.textContent = Number(alphaInput.value).toFixed(2);
        render();
      });

      function render(){
        const simImg = simByPhase.get(phaseSel.value);
        if(!simImg || !expImg.complete || !simImg.complete){ return; }
        const w = expImg.naturalWidth || expImg.width;
        const h = expImg.naturalHeight || expImg.height;
        if(!w || !h){ return; }
        canvas.width = w;
        canvas.height = h;
        ctx.clearRect(0, 0, w, h);

        if(modeSel.value === 'split'){
          const mid = Math.floor(w / 2);
          ctx.globalAlpha = 1.0;
          ctx.drawImage(expImg, 0, 0, mid, h, 0, 0, mid, h);
          ctx.drawImage(simImg, mid, 0, w - mid, h, mid, 0, w - mid, h);
          ctx.strokeStyle = 'rgba(255,255,0,0.9)';
          ctx.beginPath();
          ctx.moveTo(mid + 0.5, 0);
          ctx.lineTo(mid + 0.5, h);
          ctx.stroke();
          return;
        }

        ctx.globalAlpha = 1.0;
        ctx.drawImage(expImg, 0, 0, w, h);
        ctx.globalAlpha = Number(alphaInput.value);
        ctx.drawImage(simImg, 0, 0, w, h);
        ctx.globalAlpha = 1.0;
      }

      render();
    }

    document.addEventListener('DOMContentLoaded', function(){
      __BLEND_WIDGET_INITS__
    });
    </script>
    """
    init_js = "\n      ".join(blend_widget_inits) if blend_widget_inits else ""
    blend_script = blend_script.replace("__BLEND_WIDGET_INITS__", init_js)

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Curated NCC vs Hough-NCC Report</title>
  <style>{css}</style>
</head>
<body>
  <section>
    <h1>Curated Phase-ID: Image NCC vs KikuchiPy Hough-NCC</h1>
    <p><b>Packet:</b> <code>{summary.get('packet_dir')}</code></p>
    <p><b>Results:</b> <code>{summary.get('out_dir')}</code></p>
    <div class='metrics'>{cards_html}</div>
  </section>

  <section>
    <h2>Method Comparison</h2>
    <table>
      <thead><tr><th>method</th><th>accuracy</th><th>mean_margin</th><th>fallback_winner_count</th><th>failed_winner_count</th></tr></thead>
      <tbody>{metrics_html}</tbody>
    </table>
  </section>

  {''.join(rec_sections)}
  {blend_script}
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html
