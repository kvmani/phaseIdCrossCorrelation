"""Curated-case NCC workflow for phase identification."""

from __future__ import annotations

import csv
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from phase_id_xcorr.preprocessing import load_image_as_float32, prepare_pattern
from phase_id_xcorr.reporting import build_run_manifest
from phase_id_xcorr.similarity import masked_ncc

PHASE_ORDER = ["fe_bcc", "fe3o4_magnetite", "feo_wustite"]


@dataclass(slots=True)
class CandidateEval:
    record_id: str
    true_phase: str
    experimental_image: str
    assumed_phase: str
    simulated_image: str
    indexing_status: str
    is_fallback_orientation: bool
    ncc: float
    valid_pixels: int
    denom: float
    is_valid: bool
    reason: str
    rank: int = 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _rel(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _image_to_rgb_uint8(array_float01: np.ndarray) -> Image.Image:
    arr8 = np.clip(array_float01, 0.0, 1.0)
    arr8 = (arr8 * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr8, mode="L").convert("RGB")


def _save_case_panel(
    *,
    exp: np.ndarray,
    candidates: list[tuple[str, np.ndarray, float]],
    out_path: Path,
    title: str,
) -> None:
    pad = 12
    tile_h, tile_w = exp.shape
    header_h = 120

    cols = 4
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = tile_h + header_h + 2 * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    x = pad
    y = header_h
    canvas.paste(_image_to_rgb_uint8(exp), (x, y))
    draw.text((x, 16), "EXP", fill=(20, 20, 20))

    for idx, (phase, arr, score) in enumerate(candidates, start=1):
        x = pad + idx * (tile_w + pad)
        canvas.paste(_image_to_rgb_uint8(arr), (x, y))
        draw.text((x, 16), f"SIM {phase}", fill=(20, 20, 20))
        draw.text((x, 36), f"NCC={score:.5f}", fill=(20, 20, 20))

    draw.text((pad, 72), title, fill=(10, 10, 10))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _perturbations(exp: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    variants = [exp]

    # Mild additive noise variants
    for sigma in (0.005, 0.01):
        noisy = exp.copy()
        noise = rng.normal(loc=0.0, scale=sigma, size=exp.shape).astype(np.float32)
        noisy[mask] = np.clip(noisy[mask] + noise[mask], 0.0, 1.0)
        variants.append(noisy)

    # Gain jitter variants
    for gain in (0.95, 1.05):
        g = exp.copy()
        g[mask] = np.clip(g[mask] * gain, 0.0, 1.0)
        variants.append(g)

    return variants


def _winner_index(scores: list[float]) -> int:
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def run_curated_ncc(
    *,
    packet_dir: Path,
    out_dir: Path,
    repo_root: Path,
    debug: bool,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run curated NCC workflow and write report artifacts."""

    log = logger or logging.getLogger(__name__)

    packet_dir = packet_dir.resolve()
    out_dir = out_dir.resolve()
    cases_dir = out_dir / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    exp_json = _load_json(packet_dir / "01_experimental_patterns_template.json")
    sim_json = _load_json(packet_dir / "02_simulated_patterns_template.json")
    proc_json = _load_json(packet_dir / "04_processing_template.json")

    normalization_method = str(proc_json.get("settings", {}).get("normalization_method", "minmax_inside_mask"))

    exp_by_id: dict[str, dict[str, Any]] = {}
    for rec in exp_json.get("records", []):
        exp_by_id[str(rec.get("record_id"))] = rec

    rng = np.random.default_rng(42 if debug else 20260221)

    score_rows: list[CandidateEval] = []
    decision_rows: list[dict[str, Any]] = []

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
        exp_prep = prepare_pattern(
            exp_loaded.array,
            normalization_method=normalization_method,
            logger=log,
        )

        candidate_maps: dict[str, np.ndarray] = {}
        record_scores: list[CandidateEval] = []

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

            sim_prep = prepare_pattern(
                sim_loaded.array,
                normalization_method=normalization_method,
                logger=log,
            )

            mask = exp_prep.mask & sim_prep.mask
            ncc = masked_ncc(exp_prep.array, sim_prep.array, mask)

            candidate_maps[assumed_phase] = sim_prep.array
            record_scores.append(
                CandidateEval(
                    record_id=record_id,
                    true_phase=true_phase,
                    experimental_image=exp_rel,
                    assumed_phase=assumed_phase,
                    simulated_image=sim_rel,
                    indexing_status=indexing_status,
                    is_fallback_orientation=is_fallback,
                    ncc=ncc.score,
                    valid_pixels=ncc.valid_pixels,
                    denom=ncc.denom,
                    is_valid=ncc.is_valid,
                    reason=ncc.reason,
                )
            )

        if not record_scores:
            continue

        record_scores.sort(key=lambda r: r.ncc, reverse=True)
        for rank, item in enumerate(record_scores, start=1):
            item.rank = rank
            score_rows.append(item)

        top = record_scores[0]
        second = record_scores[1] if len(record_scores) > 1 else None
        margin = float(top.ncc - second.ncc) if second is not None else float(top.ncc)

        # Stability probes on experimental pattern against same candidate maps
        perturb_scores: list[list[float]] = []
        candidates_in_rank_order = [r.assumed_phase for r in record_scores]
        for perturbed in _perturbations(exp_prep.array, exp_prep.mask, rng):
            scores = []
            for phase in candidates_in_rank_order:
                sim_arr = candidate_maps[phase]
                res = masked_ncc(perturbed, sim_arr, exp_prep.mask)
                scores.append(res.score)
            perturb_scores.append(scores)

        baseline_winner = _winner_index(perturb_scores[0])
        flip_count = 0
        for scores in perturb_scores[1:]:
            if _winner_index(scores) != baseline_winner:
                flip_count += 1
        flip_rate = float(flip_count / max(1, len(perturb_scores) - 1))

        case_panel_candidates = []
        for phase in PHASE_ORDER:
            if phase in candidate_maps:
                phase_score = next((r.ncc for r in record_scores if r.assumed_phase == phase), float("nan"))
                case_panel_candidates.append((phase, candidate_maps[phase], phase_score))

        panel_path = cases_dir / f"{record_id}_panel.png"
        _save_case_panel(
            exp=exp_prep.array,
            candidates=case_panel_candidates,
            out_path=panel_path,
            title=(
                f"record={record_id} true={true_phase} pred={top.assumed_phase} "
                f"margin={margin:.5f} flip_rate={flip_rate:.3f}"
            ),
        )

        decision_rows.append(
            {
                "record_id": record_id,
                "true_phase": true_phase,
                "pred_phase": top.assumed_phase,
                "is_correct": top.assumed_phase == true_phase,
                "top_ncc": top.ncc,
                "second_ncc": second.ncc if second else None,
                "margin": margin,
                "top_indexing_status": top.indexing_status,
                "top_is_fallback_orientation": top.is_fallback_orientation,
                "flip_rate": flip_rate,
                "panel_path": _rel(panel_path, repo_root),
            }
        )

    # Write scores.csv
    scores_csv = out_dir / "scores.csv"
    with scores_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "record_id",
                "true_phase",
                "experimental_image",
                "assumed_phase",
                "simulated_image",
                "indexing_status",
                "is_fallback_orientation",
                "ncc",
                "valid_pixels",
                "denom",
                "is_valid",
                "reason",
                "rank",
            ],
        )
        writer.writeheader()
        for row in score_rows:
            writer.writerow(asdict(row))

    # Write decisions.csv
    decisions_csv = out_dir / "decisions.csv"
    with decisions_csv.open("w", encoding="utf-8", newline="") as f:
        if decision_rows:
            writer = csv.DictWriter(f, fieldnames=list(decision_rows[0].keys()))
            writer.writeheader()
            writer.writerows(decision_rows)
        else:
            f.write("record_id,true_phase,pred_phase,is_correct,top_ncc,second_ncc,margin,top_indexing_status,top_is_fallback_orientation,flip_rate,panel_path\n")

    total = len(decision_rows)
    correct = int(sum(1 for d in decision_rows if d["is_correct"]))
    accuracy = float(correct / total) if total else 0.0
    mean_margin = float(np.mean([d["margin"] for d in decision_rows])) if total else 0.0
    mean_top = float(np.mean([d["top_ncc"] for d in decision_rows])) if total else 0.0
    mean_flip_rate = float(np.mean([d["flip_rate"] for d in decision_rows])) if total else 0.0

    per_phase: dict[str, dict[str, float]] = {}
    for phase in PHASE_ORDER:
        rows = [d for d in decision_rows if d["true_phase"] == phase]
        n = len(rows)
        c = int(sum(1 for d in rows if d["is_correct"]))
        per_phase[phase] = {
            "count": n,
            "correct": c,
            "accuracy": float(c / n) if n else 0.0,
        }

    error_cases = [d for d in decision_rows if not d["is_correct"]]

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "packet_dir": _rel(packet_dir, repo_root),
        "out_dir": _rel(out_dir, repo_root),
        "normalization_method": normalization_method,
        "cases_total": total,
        "cases_correct": correct,
        "top1_accuracy": accuracy,
        "mean_top_ncc": mean_top,
        "mean_margin": mean_margin,
        "mean_flip_rate": mean_flip_rate,
        "per_phase": per_phase,
        "error_case_ids": [d["record_id"] for d in error_cases],
    }

    # Write summary JSON
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write error report markdown
    err_md = out_dir / "error_cases.md"
    lines = ["# Curated NCC Error Cases", ""]
    if not error_cases:
        lines.append("No misclassified cases in current curated set.")
    else:
        lines.append("| record_id | true_phase | pred_phase | top_ncc | margin | flip_rate |")
        lines.append("|---|---|---|---:|---:|---:|")
        for d in error_cases:
            lines.append(
                "| {record_id} | {true_phase} | {pred_phase} | {top_ncc:.5f} | {margin:.5f} | {flip_rate:.3f} |".format(
                    **d
                )
            )
    err_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = build_run_manifest(
        repo_root=repo_root,
        packet_dir=packet_dir,
        out_dir=out_dir,
        debug=debug,
        extra={
            "workflow": "curated_ncc",
            "artifacts": {
                "scores_csv": _rel(scores_csv, repo_root),
                "decisions_csv": _rel(decisions_csv, repo_root),
                "summary_json": _rel(summary_path, repo_root),
                "error_cases_md": _rel(err_md, repo_root),
                "cases_dir": _rel(cases_dir, repo_root),
            },
        },
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info(
        "Curated NCC completed | cases=%d correct=%d accuracy=%.4f mean_margin=%.5f",
        total,
        correct,
        accuracy,
        mean_margin,
    )
    log.info("Artifacts written under %s", out_dir)

    return {
        "summary": summary,
        "scores_csv": scores_csv,
        "decisions_csv": decisions_csv,
        "summary_json": summary_path,
        "error_cases_md": err_md,
        "manifest": manifest_path,
    }
