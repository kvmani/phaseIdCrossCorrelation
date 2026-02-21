#!/usr/bin/env python3
"""Build a single-file HTML inspection report for curated NCC analysis."""

from __future__ import annotations

import argparse
import base64
import csv
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase_id_xcorr.preprocessing import load_image_as_float32, prepare_pattern
from phase_id_xcorr.similarity import masked_ncc

PHASE_ORDER = ["fe_bcc", "fe3o4_magnetite", "feo_wustite"]


def _rel(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _img_data_uri(path: Path, max_size: tuple[int, int] = (280, 280)) -> str:
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _array_data_uri(array_float01: np.ndarray, max_size: tuple[int, int] = (280, 280)) -> str:
    arr = np.clip(array_float01, 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    im = Image.fromarray(arr8, mode="L")
    im.thumbnail(max_size, Image.Resampling.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _pairwise_candidate_similarity(candidate_prepped: dict[str, Any]) -> dict[str, float]:
    phases = sorted(candidate_prepped.keys())
    out: dict[str, float] = {}
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            p1, p2 = phases[i], phases[j]
            a = candidate_prepped[p1]
            b = candidate_prepped[p2]
            mask = a.mask & b.mask
            r = masked_ncc(a.array, b.array, mask)
            out[f"{p1} vs {p2}"] = r.score
    return out


def _fmt_float(value: Any, digits: int = 5) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def build_report(packet_dir: Path, results_dir: Path, out_html: Path, repo_root: Path) -> None:
    exp_json = _load_json(packet_dir / "01_experimental_patterns_template.json")
    sim_json = _load_json(packet_dir / "02_simulated_patterns_template.json")
    summary_json = _load_json(results_dir / "summary.json")
    scores_rows = _load_csv(results_dir / "scores.csv")
    decisions_rows = _load_csv(results_dir / "decisions.csv")
    normalization_method = str(summary_json.get("normalization_method", "minmax_inside_mask"))

    exp_by_id = {str(r.get("record_id")): r for r in exp_json.get("records", [])}
    score_by_key = {(r["record_id"], r["assumed_phase"]): r for r in scores_rows}
    decision_by_id = {r["record_id"]: r for r in decisions_rows}

    sections: list[str] = []

    # Summary
    sections.append(
        f"""
        <section class='summary'>
          <h1>Curated NCC Inspection Report</h1>
          <p><b>Packet:</b> <code>{_rel(packet_dir, repo_root)}</code></p>
          <p><b>Results:</b> <code>{_rel(results_dir, repo_root)}</code></p>
          <div class='metrics'>
            <div class='card'><span>Cases</span><strong>{summary_json.get('cases_total', 0)}</strong></div>
            <div class='card'><span>Correct</span><strong>{summary_json.get('cases_correct', 0)}</strong></div>
            <div class='card'><span>Top-1 Accuracy</span><strong>{_fmt_float(summary_json.get('top1_accuracy', 0), 3)}</strong></div>
            <div class='card'><span>Mean Top NCC</span><strong>{_fmt_float(summary_json.get('mean_top_ncc', 0), 4)}</strong></div>
            <div class='card'><span>Mean Margin</span><strong>{_fmt_float(summary_json.get('mean_margin', 0), 4)}</strong></div>
            <div class='card'><span>Mean Flip Rate</span><strong>{_fmt_float(summary_json.get('mean_flip_rate', 0), 4)}</strong></div>
          </div>
        </section>
        """
    )

    # Decisions table
    rows_html = []
    for r in decisions_rows:
        cls = "ok" if str(r.get("is_correct", "")).lower() == "true" else "bad"
        rows_html.append(
            f"<tr class='{cls}'><td>{r.get('record_id')}</td><td>{r.get('true_phase')}</td><td>{r.get('pred_phase')}</td>"
            f"<td>{r.get('is_correct')}</td><td>{_fmt_float(r.get('top_ncc'), 5)}</td><td>{_fmt_float(r.get('margin'), 5)}</td>"
            f"<td>{r.get('top_indexing_status')}</td><td>{r.get('top_is_fallback_orientation')}</td><td>{_fmt_float(r.get('flip_rate'), 3)}</td></tr>"
        )

    sections.append(
        """
        <section>
          <h2>Decision Overview</h2>
          <table>
            <thead>
              <tr><th>record</th><th>true</th><th>pred</th><th>correct</th><th>top_ncc</th><th>margin</th><th>winner_status</th><th>winner_fallback</th><th>flip_rate</th></tr>
            </thead>
            <tbody>
        """
        + "\n".join(rows_html)
        + """
            </tbody>
          </table>
        </section>
        """
    )

    # Per-record sections
    for rec in sim_json.get("records", []):
        rid = str(rec.get("record_id"))
        exp_rec = exp_by_id.get(rid, {})
        decision = decision_by_id.get(rid, {})

        exp_rel = str(exp_rec.get("image_file", rec.get("experimental_image", "")))
        exp_path = packet_dir / exp_rel
        exp_loaded = load_image_as_float32(exp_path)
        exp_prep = prepare_pattern(exp_loaded.array, normalization_method=normalization_method)
        exp_img_uri = _array_data_uri(exp_prep.array)

        euler = exp_rec.get("orientation_angles_degrees", {})
        euler_text = f"phi1={euler.get('phi1','n/a')}, PHI={euler.get('PHI','n/a')}, phi2={euler.get('phi2','n/a')}"

        candidate_cards = []
        candidate_rows = []
        candidate_prepped = {}
        for phase in PHASE_ORDER:
            cand = next((c for c in rec.get("simulated_candidates", []) if c.get("assumed_phase") == phase), None)
            if cand is None:
                continue
            sim_rel = str(cand.get("simulated_image"))
            sim_path = packet_dir / sim_rel
            sim_loaded = load_image_as_float32(sim_path)
            sim_prep = prepare_pattern(sim_loaded.array, normalization_method=normalization_method)
            sim_uri = _array_data_uri(sim_prep.array)
            candidate_prepped[phase] = sim_prep

            score = score_by_key.get((rid, phase), {})
            ncc = score.get("ncc", "n/a")
            rank = score.get("rank", "n/a")

            c_euler = cand.get("candidate_angles_degrees", {})
            c_euler_text = f"phi1={c_euler.get('phi1','n/a')}, PHI={c_euler.get('PHI','n/a')}, phi2={c_euler.get('phi2','n/a')}"

            bad_flag = " status-failed" if str(cand.get("indexing_status")) == "failed" else ""
            candidate_cards.append(
                f"""
                <div class='img-card{bad_flag}'>
                  <img src='{sim_uri}' alt='sim {phase}'>
                  <div class='cap'><b>SIM {phase}</b></div>
                  <div class='meta'><code>{sim_rel}</code></div>
                  <div class='meta'>Euler: {c_euler_text}</div>
                  <div class='meta'>dtype={sim_loaded.source_dtype} bit_depth={sim_loaded.source_bit_depth} shape={sim_loaded.source_shape}</div>
                  <div class='meta'>value_range=[{_fmt_float(sim_loaded.value_min,4)}, {_fmt_float(sim_loaded.value_max,4)}]</div>
                  <div class='meta'>status={cand.get('indexing_status')} fallback={cand.get('is_fallback_orientation')}</div>
                  <div class='meta'><b>NCC={_fmt_float(ncc,5)} rank={rank}</b></div>
                </div>
                """
            )

            candidate_rows.append(
                f"<tr><td>{phase}</td><td>{_fmt_float(ncc,5)}</td><td>{rank}</td><td>{cand.get('indexing_status')}</td><td>{cand.get('is_fallback_orientation')}</td></tr>"
            )

        pairwise = _pairwise_candidate_similarity(candidate_prepped) if len(candidate_prepped) >= 2 else {}
        pairwise_html = "".join(
            f"<li>{k}: {_fmt_float(v,5)}</li>" for k, v in sorted(pairwise.items())
        )

        sections.append(
            f"""
            <section class='record'>
              <h2>Record {rid}</h2>
              <p><b>true={exp_rec.get('true_phase','n/a')}</b> | <b>pred={decision.get('pred_phase','n/a')}</b> |
                 correct={decision.get('is_correct','n/a')} | top_ncc={_fmt_float(decision.get('top_ncc'),5)} |
                 margin={_fmt_float(decision.get('margin'),5)} | flip_rate={_fmt_float(decision.get('flip_rate'),3)}</p>

              <div class='exp-block'>
                <div class='img-card exp'>
                  <img src='{exp_img_uri}' alt='exp {rid}'>
                  <div class='cap'><b>EXP</b></div>
                  <div class='meta'><code>{exp_rel}</code></div>
                  <div class='meta'>true_phase={exp_rec.get('true_phase','n/a')}</div>
                  <div class='meta'>Euler: {euler_text}</div>
                  <div class='meta'>display=masked_normalized_for_scoring</div>
                  <div class='meta'>dtype={exp_loaded.source_dtype} bit_depth={exp_loaded.source_bit_depth} shape={exp_loaded.source_shape}</div>
                  <div class='meta'>value_range=[{_fmt_float(exp_loaded.value_min,4)}, {_fmt_float(exp_loaded.value_max,4)}]</div>
                </div>
              </div>

              <div class='grid'>
                {''.join(candidate_cards)}
              </div>

              <h3>NCC Table</h3>
              <table>
                <thead><tr><th>assumed_phase</th><th>ncc</th><th>rank</th><th>indexing_status</th><th>fallback</th></tr></thead>
                <tbody>
                  {''.join(candidate_rows)}
                </tbody>
              </table>

              <h3>Candidate Separability (sim-vs-sim NCC)</h3>
              <ul>{pairwise_html if pairwise_html else '<li>n/a</li>'}</ul>
            </section>
            """
        )

    css = """
    body{font-family:Arial,Helvetica,sans-serif;margin:20px;color:#111;background:#fafafa}
    h1,h2,h3{margin:0 0 10px 0}
    section{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin-bottom:16px}
    .metrics{display:grid;grid-template-columns:repeat(6,minmax(100px,1fr));gap:10px}
    .card{background:#f2f5f9;border:1px solid #dfe5ee;border-radius:6px;padding:8px}
    .card span{display:block;font-size:12px;color:#444}
    .card strong{font-size:16px}
    table{border-collapse:collapse;width:100%;margin-top:8px}
    th,td{border:1px solid #ddd;padding:6px 8px;font-size:13px}
    th{background:#f3f3f3;text-align:left}
    tr.ok{background:#edf9ef}
    tr.bad{background:#fff1f1}
    .grid{display:grid;grid-template-columns:repeat(3,minmax(230px,1fr));gap:10px}
    .img-card{background:#fcfcfc;border:1px solid #ddd;border-radius:6px;padding:8px}
    .img-card.status-failed{border:2px solid #cc3a3a}
    .img-card img{width:100%;height:auto;border:1px solid #ccc;background:#000}
    .img-card .cap{margin-top:6px}
    .img-card .meta{font-size:12px;color:#333;margin-top:3px;word-break:break-word}
    .exp-block{display:flex;gap:10px;margin-bottom:10px}
    .exp-block .img-card{max-width:360px}
    code{background:#f5f5f5;padding:1px 4px;border-radius:4px}
    """

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Curated NCC Inspection Report</title>
  <style>{css}</style>
</head>
<body>
{''.join(sections)}
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated NCC inspection HTML report")
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=Path("data/test/student_data_packet_phaseid"),
        help="Input packet directory (repo-relative by default).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("reports/curated_ncc"),
        help="Directory containing curated NCC artifacts.",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("reports/curated_ncc/inspection_report.html"),
        help="Output HTML path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet_dir = args.packet_dir if args.packet_dir.is_absolute() else (ROOT / args.packet_dir)
    results_dir = args.results_dir if args.results_dir.is_absolute() else (ROOT / args.results_dir)
    out_html = args.out_html if args.out_html.is_absolute() else (ROOT / args.out_html)

    build_report(packet_dir=packet_dir, results_dir=results_dir, out_html=out_html, repo_root=ROOT)
    print(f"Wrote inspection HTML: {Path(os.path.relpath(out_html, ROOT)).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
