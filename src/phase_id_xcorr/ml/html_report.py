"""HTML report generation for ML benchmark suites."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_io import read_json, rel_path


def _html_rel_path(target: str, *, html_path: Path, repo_root: Path) -> str:
    if not target:
        return ""
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    try:
        return Path(os.path.relpath(candidate, html_path.parent.resolve())).as_posix()
    except ValueError:
        return candidate.as_posix()


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
        report_href = _html_rel_path(report_link, html_path=output_html, repo_root=repo_root)

        table_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(str(row.get('model_name', '')))}</td>"
            f"<td>{_fmt(row.get('best_val_macro_f1'))}</td>"
            f"<td>{_fmt(row.get('test_accuracy'))}</td>"
            f"<td>{_fmt(row.get('test_macro_f1'))}</td>"
            f"<td>{_fmt(row.get('runtime_seconds'))}</td>"
            f"<td><a href='{html.escape(report_href)}'>report.json</a><br/><code>{resolved_cfg}</code></td>"
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


def _phase_fraction_bars_html(phase_fractions: dict[str, Any]) -> str:
    if not isinstance(phase_fractions, dict) or not phase_fractions:
        return "<p>No phase-fraction data available.</p>"
    parts: list[str] = []
    for phase_name, value in sorted(phase_fractions.items()):
        frac = float(value)
        pct = 100.0 * frac
        parts.append(
            "<div class='phase-bar-row'>"
            f"<div class='phase-bar-label'>{html.escape(str(phase_name))}</div>"
            f"<div class='phase-bar-track'><div class='phase-bar-fill' style='width:{pct:.2f}%'></div></div>"
            f"<div class='phase-bar-value'>{pct:.2f}%</div>"
            "</div>"
        )
    return "".join(parts)


def generate_full_scan_suite_html_report(*, summary_json_path: Path, output_html: Path, repo_root: Path) -> Path:
    """Generate a comparative HTML report for suite-level full-scan `.oh5` exports."""

    summary = read_json(summary_json_path)
    rows = summary.get("rows", []) if isinstance(summary.get("rows"), list) else []
    completed_rows = [row for row in rows if str(row.get("status", "")) == "completed"]
    failed_rows = [row for row in rows if str(row.get("status", "")) != "completed"]

    dataset_manifest: dict[str, Any] | None = None
    for row in completed_rows:
        run_dir_rel = str(row.get("run_dir", "")).strip()
        if not run_dir_rel:
            continue
        report_path = (repo_root / run_dir_rel / "report.json").resolve()
        if report_path.exists():
            report = read_json(report_path)
            row["_train_report"] = report
            dataset_manifest_rel = str(report.get("dataset_manifest_path", "")).strip()
            if dataset_manifest is None and dataset_manifest_rel:
                candidate = (repo_root / dataset_manifest_rel).resolve()
                if candidate.exists():
                    dataset_manifest = read_json(candidate)

    dataset_html = _dataset_overview_html(dataset_manifest) if dataset_manifest is not None else ""

    best_by_test = max(
        completed_rows,
        key=lambda row: float((row.get("_train_report", {}) or {}).get("test_metrics", {}).get("macro_f1", -1.0)),
        default=None,
    )
    best_by_conf = max(
        completed_rows,
        key=lambda row: float(row.get("mean_confidence") or -1.0),
        default=None,
    )

    spotlight_lines: list[str] = []
    if best_by_test is not None:
        report = best_by_test.get("_train_report", {}) or {}
        spotlight_lines.append(
            f"<li><b>Best held-out classifier:</b> {html.escape(str(best_by_test.get('run_name','')))} "
            f"(test macro-F1 {_fmt(report.get('test_metrics', {}).get('macro_f1'))}, "
            f"test accuracy {_fmt(report.get('test_metrics', {}).get('accuracy'))}).</li>"
        )
    if best_by_conf is not None:
        spotlight_lines.append(
            f"<li><b>Most confident full-scan map:</b> {html.escape(str(best_by_conf.get('run_name','')))} "
            f"(mean confidence {_fmt(best_by_conf.get('mean_confidence'))}, "
            f"dominant phase {html.escape(str(best_by_conf.get('dominant_phase','')))}).</li>"
        )
    dominant_phases = {str(row.get("dominant_phase", "")) for row in completed_rows if str(row.get("dominant_phase", "")).strip()}
    if dominant_phases:
        spotlight_lines.append(
            f"<li><b>Dominant-phase agreement across models:</b> {html.escape(', '.join(sorted(dominant_phases)))}.</li>"
        )
    if failed_rows:
        spotlight_lines.append(f"<li><b>Failed exports:</b> {len(failed_rows)} model run(s).</li>")
    spotlight_html = "<ul>" + "".join(spotlight_lines) + "</ul>" if spotlight_lines else "<p>No completed runs available.</p>"

    comparison_rows: list[str] = []
    for row in completed_rows:
        report = row.get("_train_report", {}) or {}
        test_metrics = report.get("test_metrics", {}) if isinstance(report.get("test_metrics"), dict) else {}
        comparison_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('run_name', '')))}</td>"
            f"<td>{html.escape(str(row.get('model_name', '')))}</td>"
            f"<td>{_fmt(report.get('best_val_macro_f1'))}</td>"
            f"<td>{_fmt(test_metrics.get('accuracy'))}</td>"
            f"<td>{_fmt(test_metrics.get('macro_f1'))}</td>"
            f"<td>{_fmt(row.get('mean_confidence'))}</td>"
            f"<td>{html.escape(str(row.get('dominant_phase', '')))}</td>"
            f"<td><a href='{html.escape(_html_rel_path(str(row.get('artifacts', {}).get('summary_json', '')), html_path=output_html, repo_root=repo_root))}'>summary.json</a></td>"
            "</tr>"
        )

    shared_legend_rel = ""
    shared_ipf_map_rel = ""
    shared_ipf_ref_rel = ""
    if completed_rows:
        first_artifacts = completed_rows[0].get("artifacts", {}) if isinstance(completed_rows[0].get("artifacts"), dict) else {}
        shared_legend_rel = str(first_artifacts.get("predicted_phase_legend_png", "") or "")
    for row in completed_rows:
        artifacts = row.get("artifacts", {}) if isinstance(row.get("artifacts"), dict) else {}
        if not shared_ipf_map_rel:
            candidate = str(artifacts.get("ipf_colored_ebsd_map_png", "") or "")
            if candidate:
                shared_ipf_map_rel = candidate
        if not shared_ipf_ref_rel:
            candidate = str(artifacts.get("ipf_reference_png", "") or "")
            if candidate:
                shared_ipf_ref_rel = candidate

    map_cards: list[str] = []
    for row in completed_rows:
        artifacts = row.get("artifacts", {}) if isinstance(row.get("artifacts"), dict) else {}
        report = row.get("_train_report", {}) or {}
        test_metrics = report.get("test_metrics", {}) if isinstance(report.get("test_metrics"), dict) else {}
        map_rel = str(artifacts.get("predicted_phase_map_png", "") or "")
        map_href = _html_rel_path(map_rel, html_path=output_html, repo_root=repo_root)
        summary_href = _html_rel_path(str(artifacts.get("summary_html", "") or ""), html_path=output_html, repo_root=repo_root)
        manifest_href = _html_rel_path(str(artifacts.get("manifest_json", "") or ""), html_path=output_html, repo_root=repo_root)
        pixel_csv_href = _html_rel_path(str(artifacts.get("pixel_predictions_csv", "") or ""), html_path=output_html, repo_root=repo_root)
        phase_bars_html = _phase_fraction_bars_html(row.get("phase_fractions", {}) if isinstance(row.get("phase_fractions"), dict) else {})
        map_cards.append(
            "<article class='map-card'>"
            f"<h3>{html.escape(str(row.get('run_name', '')))}</h3>"
            f"<p class='subtle'>{html.escape(str(row.get('model_name', '')))}</p>"
            f"<img src='{html.escape(map_href)}' alt='Predicted phase map for {html.escape(str(row.get('run_name', '')))}'>"
            "<div class='metric-grid'>"
            f"<div><span class='metric-label'>Best val macro-F1</span><span class='metric-value'>{_fmt(report.get('best_val_macro_f1'))}</span></div>"
            f"<div><span class='metric-label'>Test macro-F1</span><span class='metric-value'>{_fmt(test_metrics.get('macro_f1'))}</span></div>"
            f"<div><span class='metric-label'>Mean scan confidence</span><span class='metric-value'>{_fmt(row.get('mean_confidence'))}</span></div>"
            f"<div><span class='metric-label'>Dominant phase</span><span class='metric-value'>{html.escape(str(row.get('dominant_phase','')))}</span></div>"
            "</div>"
            "<div class='phase-bars'>"
            f"{phase_bars_html}"
            "</div>"
            "<p class='artifact-links'>"
            f"<a href='{html.escape(summary_href)}'>run summary</a> | "
            f"<a href='{html.escape(manifest_href)}'>manifest</a> | "
            f"<a href='{html.escape(pixel_csv_href)}'>pixel CSV</a>"
            "</p>"
            "</article>"
        )

    failed_html = ""
    if failed_rows:
        failed_items = "".join(
            f"<li><b>{html.escape(str(row.get('run_name', '')))}</b>: {html.escape(str(row.get('error', 'unknown error')))}</li>"
            for row in failed_rows
        )
        failed_html = f"<section><h2>Failed Runs</h2><ul>{failed_items}</ul></section>"

    shared_visuals_html = ""
    if shared_legend_rel or shared_ipf_map_rel or shared_ipf_ref_rel:
        blocks: list[str] = []
        if shared_legend_rel:
            blocks.append(
                "<figure>"
                f"<img src='{html.escape(_html_rel_path(shared_legend_rel, html_path=output_html, repo_root=repo_root))}' alt='Predicted phase legend'>"
                "<figcaption>Shared predicted-phase legend.</figcaption>"
                "</figure>"
            )
        if shared_ipf_map_rel:
            blocks.append(
                "<figure>"
                f"<img src='{html.escape(_html_rel_path(shared_ipf_map_rel, html_path=output_html, repo_root=repo_root))}' alt='IPF-colored EBSD map'>"
                "<figcaption>Shared IPF-colored EBSD map. This geometry is scan-derived and does not depend on the classifier.</figcaption>"
                "</figure>"
            )
        if shared_ipf_ref_rel:
            blocks.append(
                "<figure>"
                f"<img src='{html.escape(_html_rel_path(shared_ipf_ref_rel, html_path=output_html, repo_root=repo_root))}' alt='IPF reference'>"
                "<figcaption>Shared IPF reference grouped by predicted phase.</figcaption>"
                "</figure>"
            )
        shared_visuals_html = (
            "<section><h2>Shared Scan Visuals</h2>"
            "<div class='shared-visuals'>"
            + "".join(blocks)
            + "</div></section>"
        )

    html_text = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Full-Scan Suite Comparison</title>
<style>
:root{{--ink:#1e2430;--muted:#5f6b7a;--line:#d7dde5;--soft:#f5f7fa;--panel:#ffffff;--accent:#2044c7;}}
body{{font-family:Arial,Helvetica,sans-serif;margin:24px;color:var(--ink);background:#fbfcfe;}}
h1,h2,h3{{margin:0 0 10px 0;}}
section{{margin:0 0 28px 0;}}
.hero{{background:linear-gradient(135deg,#eef3ff 0%,#f7fafc 55%,#eef8f3 100%);border:1px solid var(--line);border-radius:18px;padding:22px;}}
.subtle{{color:var(--muted);margin-top:0;}}
table{{border-collapse:collapse;width:100%;margin:12px 0 24px;background:var(--panel);}}
th,td{{border:1px solid var(--line);padding:8px;vertical-align:top;text-align:left;}}
th{{background:var(--soft);}}
.shared-visuals{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;}}
.shared-visuals figure,.map-card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:14px;}}
figure img,.map-card img{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff;}}
.maps-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px;}}
.metric-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:12px 0;}}
.metric-grid div{{background:var(--soft);border-radius:10px;padding:10px;}}
.metric-label{{display:block;color:var(--muted);font-size:12px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.04em;}}
.metric-value{{display:block;font-size:18px;font-weight:700;}}
.phase-bars{{margin-top:10px;}}
.phase-bar-row{{display:grid;grid-template-columns:58px 1fr 64px;gap:8px;align-items:center;margin:6px 0;}}
.phase-bar-label,.phase-bar-value{{font-size:13px;}}
.phase-bar-track{{height:12px;background:#edf1f5;border-radius:999px;overflow:hidden;border:1px solid #d8dee8;}}
.phase-bar-fill{{height:100%;background:linear-gradient(90deg,#2044c7,#52a3ff);}}
.artifact-links{{margin:10px 0 0 0;color:var(--muted);}}
.artifact-links a{{color:var(--accent);text-decoration:none;}}
.artifact-links a:hover{{text-decoration:underline;}}
code{{background:#f3f5f8;padding:2px 5px;border-radius:5px;}}
</style>
</head><body>
<section class='hero'>
<h1>Full-Scan Model Comparison</h1>
<p class='subtle'>Suite root: <code>{html.escape(str(summary.get('suite_root', '')))}</code><br>
Scan: <code>{html.escape(str(summary.get('oh5_path', '')))}</code><br>
Runs completed: {int(summary.get('runs_completed', 0))} / {int(summary.get('runs_total', 0))}</p>
{spotlight_html}
</section>
{dataset_html}
<section>
<h2>Comparative Metrics</h2>
<table>
<thead><tr><th>Run</th><th>Model</th><th>Best Val Macro-F1</th><th>Test Accuracy</th><th>Test Macro-F1</th><th>Mean Scan Confidence</th><th>Dominant Phase</th><th>Artifacts</th></tr></thead>
<tbody>{''.join(comparison_rows) if comparison_rows else '<tr><td colspan="8">No completed runs available.</td></tr>'}</tbody>
</table>
</section>
{shared_visuals_html}
<section>
<h2>Predicted Phase Maps</h2>
<div class='maps-grid'>
{''.join(map_cards) if map_cards else '<p>No completed run exports were available.</p>'}
</div>
</section>
{failed_html}
</body></html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    return output_html
