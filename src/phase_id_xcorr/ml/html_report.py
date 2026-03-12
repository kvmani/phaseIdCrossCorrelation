"""HTML report generation for ML benchmark suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset_io import read_json, rel_path


def _sparkline(values: list[float]) -> str:
    return ",".join(f"{v:.4f}" for v in values)


def generate_suite_html_report(*, summary_json_path: Path, output_html: Path, repo_root: Path) -> Path:
    summary = read_json(summary_json_path)
    rows = summary.get("rows", []) if isinstance(summary.get("rows"), list) else []

    table_rows: list[str] = []
    detail_blocks: list[str] = []

    for row in rows:
        name = str(row.get("name", ""))
        status = str(row.get("status", ""))
        report_rel = str(row.get("report_path", ""))
        report_path = (repo_root / report_rel).resolve() if report_rel else None
        history_points = ""
        confusion = ""
        report_link = report_rel
        if report_path is not None and report_path.exists():
            report = read_json(report_path)
            history = report.get("history", []) if isinstance(report.get("history"), list) else []
            val_curve = [float(h.get("val_macro_f1", 0.0)) for h in history if isinstance(h, dict)]
            history_points = _sparkline(val_curve)
            cm = report.get("test_metrics", {}).get("confusion_matrix", [])
            confusion = "<br/>".join(" ".join(str(v) for v in r) for r in cm) if isinstance(cm, list) else ""
            report_link = rel_path(report_path, repo_root)

        table_rows.append(
            f"<tr><td>{name}</td><td>{status}</td><td>{row.get('model_name','')}</td>"
            f"<td>{row.get('best_val_macro_f1','')}</td><td>{row.get('test_accuracy','')}</td>"
            f"<td><a href='../{report_link}'>report.json</a></td></tr>"
        )
        detail_blocks.append(
            f"<details><summary>{name}: curves/matrix</summary>"
            f"<p><b>Val macro-F1 curve</b>: {history_points}</p>"
            f"<p><b>Confusion matrix</b><br/><code>{confusion}</code></p>"
            f"</details>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>ML Benchmark Suite Report</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;margin:20px}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:6px}} summary{{cursor:pointer;font-weight:bold}}</style>
</head><body>
<h1>ML Benchmark Suite Report</h1>
<p>Runs: {summary.get('runs_total')} | Completed: {summary.get('runs_completed')} | Failed: {summary.get('runs_failed')}</p>
<table>
<tr><th>Run</th><th>Status</th><th>Model</th><th>Best Val Macro-F1</th><th>Test Accuracy</th><th>Artifacts</th></tr>
{''.join(table_rows)}
</table>
<h2>Detailed analytics</h2>
{''.join(detail_blocks)}
</body></html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html
