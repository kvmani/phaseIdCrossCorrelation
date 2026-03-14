"""HTML report generation for ML benchmark suites."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_io import read_json, rel_path


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return html.escape(str(value))


def _sparkline(values: list[float]) -> str:
    return ", ".join(f"{float(v):.4f}" for v in values)


def _confusion_table_html(cm: list[list[Any]], class_names: list[str]) -> str:
    if not cm:
        return "<p>No confusion matrix available.</p>"
    header = "".join(f"<th>{html.escape(name)}</th>" for name in class_names)
    rows: list[str] = []
    for idx, row in enumerate(cm):
        label = class_names[idx] if idx < len(class_names) else str(idx)
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        rows.append(f"<tr><th>{html.escape(label)}</th>{cells}</tr>")
    return (
        "<table><thead><tr><th>True \\ Pred</th>"
        f"{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _per_class_table_html(per_class: dict[str, Any]) -> str:
    rows: list[str] = []
    for class_name, stats in per_class.items():
        if not isinstance(stats, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(class_name))}</td>"
            f"<td>{_fmt(stats.get('precision'))}</td>"
            f"<td>{_fmt(stats.get('recall'))}</td>"
            f"<td>{_fmt(stats.get('f1'))}</td>"
            f"<td>{html.escape(str(stats.get('support', '')))}</td>"
            f"<td>{html.escape(str(stats.get('pred_support', '')))}</td>"
            "</tr>"
        )
    if not rows:
        return "<p>No per-class metrics available.</p>"
    return (
        "<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th><th>Pred support</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _dataset_overview_html(dataset_manifest: dict[str, Any]) -> str:
    split_counts = dataset_manifest.get("split_counts") or {}
    phase_stats = dataset_manifest.get("phase_statistics") or {}

    phase_rows: list[str] = []
    for phase_name, stats in phase_stats.items():
        if not isinstance(stats, dict):
            continue
        ci = stats.get("confidence_index") or {}
        fit = stats.get("fit") or {}
        iq = stats.get("image_quality") or {}
        intensity = stats.get("intensity_distribution") or {}
        phase_rows.append(
            "<tr>"
            f"<td>{html.escape(str(phase_name))}</td>"
            f"<td>{html.escape(str(stats.get('accepted_count', '')))}</td>"
            f"<td>{_fmt(stats.get('accepted_fraction_of_dataset'), 3)}</td>"
            f"<td>{html.escape(str(stats.get('train_count', '')))}</td>"
            f"<td>{html.escape(str(stats.get('val_count', '')))}</td>"
            f"<td>{html.escape(str(stats.get('test_count', '')))}</td>"
            f"<td>{_fmt(ci.get('mean'))}</td>"
            f"<td>{_fmt(fit.get('mean'))}</td>"
            f"<td>{_fmt(iq.get('mean'))}</td>"
            f"<td>{html.escape(str(intensity.get('mode_intensity_value', '')))}</td>"
            "</tr>"
        )

    phase_body = "".join(phase_rows) if phase_rows else '<tr><td colspan="10">No dataset phase statistics available.</td></tr>'

    return (
        "<section>"
        "<h2>Dataset Overview</h2>"
        f"<p>Accepted samples: <b>{dataset_manifest.get('num_samples_total', 0)}</b> | "
        f"Raw scan pixels: <b>{dataset_manifest.get('raw_input_rows_total', 0)}</b> | "
        f"Splits: train={split_counts.get('train', 0)}, val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}</p>"
        "<table><thead><tr><th>Phase</th><th>Accepted</th><th>Accepted frac</th><th>Train</th><th>Val</th><th>Test</th><th>Mean CI</th><th>Mean Fit</th><th>Mean IQ</th><th>Mode intensity</th></tr></thead>"
        f"<tbody>{phase_body}</tbody></table>"
        "</section>"
    )


def generate_suite_html_report(*, summary_json_path: Path, output_html: Path, repo_root: Path) -> Path:
    summary = read_json(summary_json_path)
    rows = summary.get("rows", []) if isinstance(summary.get("rows"), list) else []

    table_rows: list[str] = []
    detail_blocks: list[str] = []
    dataset_manifest: dict[str, Any] | None = None

    completed_rows = [row for row in rows if str(row.get("status", "")) == "completed"]
    best_name = str(summary.get("best_run") or "")
    best_row = next((row for row in completed_rows if str(row.get("name", "")) == best_name), None)

    for row in rows:
        name = str(row.get("name", ""))
        status = str(row.get("status", ""))
        report_rel = str(row.get("report_path", ""))
        report_path = (repo_root / report_rel).resolve() if report_rel else None

        history_points = ""
        confusion_html = "<p>No confusion matrix available.</p>"
        per_class_html = "<p>No per-class metrics available.</p>"
        extra_meta = ""
        report_link = report_rel
        resolved_cfg = html.escape(str(row.get("resolved_train_config", "")))

        if report_path is not None and report_path.exists():
            report = read_json(report_path)
            report_link = rel_path(report_path, repo_root)
            history = report.get("history", []) if isinstance(report.get("history"), list) else []
            val_curve = [float(h.get("val_macro_f1", 0.0)) for h in history if isinstance(h, dict)]
            history_points = _sparkline(val_curve)

            test_metrics = report.get("test_metrics", {}) if isinstance(report.get("test_metrics"), dict) else {}
            class_names = list((test_metrics.get("per_class") or {}).keys())
            confusion = test_metrics.get("confusion_matrix", [])
            confusion_html = _confusion_table_html(confusion, class_names)
            per_class_html = _per_class_table_html(test_metrics.get("per_class", {}))

            dataset_manifest_rel = str(report.get("dataset_manifest_path", ""))
            if dataset_manifest is None and dataset_manifest_rel:
                dataset_manifest = read_json((repo_root / dataset_manifest_rel).resolve())

            epoch_seconds = [float(h.get("epoch_seconds", 0.0)) for h in history if isinstance(h, dict)]
            extra_meta = (
                f"<p><b>Best epoch:</b> {html.escape(str(report.get('best_epoch', '')))} | "
                f"<b>Runtime:</b> {_fmt(report.get('runtime_seconds'))} s | "
                f"<b>Mean epoch:</b> {_fmt(float(np.mean(epoch_seconds)) if epoch_seconds else None)} s | "
                f"<b>Test macro-F1:</b> {_fmt(test_metrics.get('macro_f1'))}</p>"
            )

        table_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(str(row.get('model_name', '')))}</td>"
            f"<td>{_fmt(row.get('best_val_macro_f1'))}</td>"
            f"<td>{_fmt(row.get('test_accuracy'))}</td>"
            f"<td>{_fmt(row.get('test_macro_f1'))}</td>"
            f"<td>{_fmt(row.get('runtime_seconds'))}</td>"
            f"<td><a href='../{html.escape(report_link)}'>report.json</a><br/><code>{resolved_cfg}</code></td>"
            "</tr>"
        )

        detail_blocks.append(
            "<details>"
            f"<summary>{html.escape(name)}: training evolution, metrics, and confusion matrix</summary>"
            f"{extra_meta}"
            f"<p><b>Validation macro-F1 by epoch:</b> {html.escape(history_points) if history_points else 'No epoch history available.'}</p>"
            "<h4>Test Confusion Matrix</h4>"
            f"{confusion_html}"
            "<h4>Per-Class Metrics</h4>"
            f"{per_class_html}"
            "</details>"
        )

    dataset_html = _dataset_overview_html(dataset_manifest) if dataset_manifest is not None else ""
    best_html = ""
    if best_row is not None:
        best_html = (
            "<section>"
            "<h2>Best Performing Model</h2>"
            f"<p><b>{html.escape(str(best_row.get('name', '')))}</b> | "
            f"model={html.escape(str(best_row.get('model_name', '')))} | "
            f"best val macro-F1={_fmt(best_row.get('best_val_macro_f1'))} | "
            f"test accuracy={_fmt(best_row.get('test_accuracy'))} | "
            f"test macro-F1={_fmt(best_row.get('test_macro_f1'))}</p>"
            "</section>"
        )

    html_text = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>ML Benchmark Suite Report</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:20px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}
th,td{{border:1px solid #ccc;padding:6px;vertical-align:top}}
th{{background:#f4f4f4}}
summary{{cursor:pointer;font-weight:bold}}
code{{background:#f5f5f5;padding:2px 4px;border-radius:4px}}
section{{margin:0 0 28px}}
</style>
</head><body>
<h1>ML Benchmark Suite Report</h1>
<p>Runs: {summary.get('runs_total')} | Completed: {summary.get('runs_completed')} | Failed: {summary.get('runs_failed')}</p>
{dataset_html}
{best_html}
<section>
<h2>Model Comparison</h2>
<table>
<thead><tr><th>Run</th><th>Status</th><th>Model</th><th>Best Val Macro-F1</th><th>Test Accuracy</th><th>Test Macro-F1</th><th>Runtime (s)</th><th>Artifacts</th></tr></thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
</section>
<section>
<h2>Detailed Analytics</h2>
{''.join(detail_blocks)}
</section>
</body></html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    return output_html
