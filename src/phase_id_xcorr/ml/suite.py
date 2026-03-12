"""Benchmark-suite orchestration for ML classifier runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any

from .config import apply_overrides, dump_yaml, load_yaml, resolve_path
from .dataset_io import rel_path, write_json
from .html_report import generate_suite_html_report
from .training import train_classifier


@dataclass(slots=True)
class SuiteResult:
    """Top-level benchmark suite artifacts."""

    output_root: Path
    summary_json: Path
    summary_md: Path
    manifest_json: Path


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=False) + "\n")


def _safe_eta_seconds(*, completed: int, total: int, elapsed: float) -> float | None:
    if completed <= 0 or total <= completed or elapsed <= 0:
        return None
    rate = completed / elapsed
    if rate <= 0:
        return None
    return float((total - completed) / rate)


def run_benchmark_suite(
    *,
    suite_config_path: Path,
    repo_root: Path,
    debug: bool,
    logger: logging.Logger | None = None,
    strict: bool = False,
) -> SuiteResult:
    """Execute configured multi-model training/eval suite."""

    log = logger or logging.getLogger(__name__)

    suite_path = suite_config_path.resolve()
    suite_cfg = load_yaml(suite_path)
    suite_dir = suite_path.parent

    output_root = resolve_path(
        suite_cfg.get("output_root", "reports/ml/benchmarks/default_suite"),
        base_dir=suite_dir,
        repo_root=repo_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    event_log = output_root / "events.jsonl"
    event_log.write_text("", encoding="utf-8")
    run_t0 = time.monotonic()

    def emit(event: str, **fields: Any) -> None:
        payload = {
            "timestamp_utc": _now_iso_utc(),
            "elapsed_seconds": float(time.monotonic() - run_t0),
            "event": event,
        }
        payload.update(fields)
        _append_event(event_log, payload)

    base_train_config = resolve_path(
        suite_cfg.get("base_train_config", ""),
        base_dir=suite_dir,
        repo_root=repo_root,
    )
    if not base_train_config.exists():
        raise FileNotFoundError(f"base_train_config not found: {base_train_config}")

    base_train_payload = load_yaml(base_train_config)

    experiments = suite_cfg.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("suite config must contain non-empty experiments list")
    emit(
        "RUN_START",
        suite_config_path=rel_path(suite_path, repo_root),
        output_root=rel_path(output_root, repo_root),
        experiment_count=len(experiments),
        strict=bool(strict),
        debug=bool(debug),
    )

    run_rows: list[dict[str, Any]] = []

    for idx, exp in enumerate(experiments, start=1):
        if not isinstance(exp, dict):
            raise ValueError("each experiment must be a mapping")

        name = str(exp.get("name", f"exp_{idx:03d}"))
        overrides = exp.get("overrides", [])
        if not isinstance(overrides, list):
            raise ValueError(f"experiment '{name}': overrides must be list")

        train_cfg_path = exp.get("train_config")
        if train_cfg_path:
            payload = load_yaml(resolve_path(train_cfg_path, base_dir=suite_dir, repo_root=repo_root))
        else:
            payload = dict(base_train_payload)

        payload = apply_overrides(payload, [str(x) for x in overrides])

        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved_cfg = run_dir / "resolved_train_config.yml"
        dump_yaml(payload, resolved_cfg)

        suite_elapsed = float(time.monotonic() - run_t0)
        eta = _safe_eta_seconds(completed=idx - 1, total=len(experiments), elapsed=suite_elapsed)
        pct = float(100.0 * (idx - 1) / max(1, len(experiments)))
        log.info(
            "Suite progress %.1f%% (%d/%d) | next=%s elapsed=%.2fs eta=%.2fs",
            pct,
            idx - 1,
            len(experiments),
            name,
            suite_elapsed,
            eta if eta is not None else 0.0,
        )
        emit(
            "EXPERIMENT_START",
            name=name,
            index=idx,
            total=len(experiments),
            progress_pct=pct,
            eta_seconds=eta,
            resolved_train_config=rel_path(resolved_cfg, repo_root),
        )

        status = "completed"
        report_path: Path | None = None
        message = ""
        exp_t0 = time.monotonic()
        try:
            result = train_classifier(
                config_path=resolved_cfg,
                repo_root=repo_root,
                debug=debug,
                logger=log,
            )
            report_path = result.report_path
        except Exception as exc:
            status = "failed"
            message = str(exc)
            if strict:
                raise

        row: dict[str, Any] = {
            "name": name,
            "status": status,
            "resolved_train_config": rel_path(resolved_cfg, repo_root),
            "report_path": rel_path(report_path, repo_root) if report_path else None,
            "error": message or None,
        }

        if report_path and report_path.exists():
            report = load_yaml(report_path) if report_path.suffix in {".yml", ".yaml"} else None
            if report is None:
                import json

                report = json.loads(report_path.read_text(encoding="utf-8"))

            row["model_name"] = report.get("model", {}).get("model_name")
            row["best_val_macro_f1"] = report.get("best_val_macro_f1")
            row["test_accuracy"] = report.get("test_metrics", {}).get("accuracy")
            row["test_macro_f1"] = report.get("test_metrics", {}).get("macro_f1")
            row["runtime_seconds"] = report.get("runtime_seconds")

        run_rows.append(row)
        done_pct = float(100.0 * idx / max(1, len(experiments)))
        done_eta = _safe_eta_seconds(completed=idx, total=len(experiments), elapsed=float(time.monotonic() - run_t0))
        emit(
            "EXPERIMENT_END",
            name=name,
            index=idx,
            total=len(experiments),
            progress_pct=done_pct,
            status=status,
            runtime_seconds=float(time.monotonic() - exp_t0),
            eta_seconds=done_eta,
            report_path=row["report_path"],
            error=row["error"],
        )
        log.info(
            "Suite run %d/%d done | %s status=%s elapsed=%.2fs eta=%.2fs",
            idx,
            len(experiments),
            name,
            status,
            float(time.monotonic() - run_t0),
            done_eta if done_eta is not None else 0.0,
        )

    completed = [r for r in run_rows if r["status"] == "completed"]
    ranked = sorted(
        completed,
        key=lambda r: float(r.get("best_val_macro_f1") or -1.0),
        reverse=True,
    )

    summary = {
        "schema_version": "phase_id_xcorr.ml_benchmark_suite.v1",
        "timestamp_utc": _now_iso_utc(),
        "suite_config_path": rel_path(suite_path, repo_root),
        "output_root": rel_path(output_root, repo_root),
        "debug": bool(debug),
        "strict": bool(strict),
        "runs_total": len(run_rows),
        "runs_completed": len(completed),
        "runs_failed": len(run_rows) - len(completed),
        "best_run": ranked[0]["name"] if ranked else None,
        "timing": {
            "total_elapsed_seconds": float(time.monotonic() - run_t0),
        },
        "rows": run_rows,
    }

    summary_json = output_root / "suite_summary.json"
    write_json(summary_json, summary)

    summary_md = output_root / "suite_summary.md"
    lines = [
        "# ML Benchmark Suite Summary",
        "",
        f"- runs total: {summary['runs_total']}",
        f"- runs completed: {summary['runs_completed']}",
        f"- runs failed: {summary['runs_failed']}",
        f"- best run: {summary['best_run']}",
        "",
        "| name | status | model | best_val_macro_f1 | test_accuracy | test_macro_f1 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            "| {name} | {status} | {model_name} | {best_val_macro_f1} | {test_accuracy} | {test_macro_f1} |".format(
                name=row.get("name"),
                status=row.get("status"),
                model_name=row.get("model_name", ""),
                best_val_macro_f1="" if row.get("best_val_macro_f1") is None else f"{float(row['best_val_macro_f1']):.5f}",
                test_accuracy="" if row.get("test_accuracy") is None else f"{float(row['test_accuracy']):.5f}",
                test_macro_f1="" if row.get("test_macro_f1") is None else f"{float(row['test_macro_f1']):.5f}",
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    html_report = generate_suite_html_report(
        summary_json_path=summary_json,
        output_html=output_root / "suite_report.html",
        repo_root=repo_root,
    )

    manifest_json = output_root / "manifest.json"
    write_json(
        manifest_json,
        {
            "schema_version": "phase_id_xcorr.ml_benchmark_manifest.v1",
            "timestamp_utc": _now_iso_utc(),
            "workflow": "ml_benchmark_suite",
            "suite_config_path": rel_path(suite_path, repo_root),
            "output_root": rel_path(output_root, repo_root),
            "debug": bool(debug),
            "strict": bool(strict),
            "timing": {
                "total_elapsed_seconds": float(time.monotonic() - run_t0),
            },
            "sanity_checks": {
                "base_train_config_exists": bool(base_train_config.exists()),
                "experiment_list_non_empty": len(experiments) > 0,
                "suite_summary_written": bool(summary_json.exists()),
            },
            "artifacts": {
                "suite_summary_json": rel_path(summary_json, repo_root),
                "suite_summary_md": rel_path(summary_md, repo_root),
                "suite_report_html": rel_path(html_report, repo_root),
                "event_log_jsonl": rel_path(event_log, repo_root),
            },
        },
    )
    emit(
        "RUN_END",
        status="completed",
        runs_total=len(run_rows),
        runs_completed=len(completed),
        runs_failed=len(run_rows) - len(completed),
        best_run=ranked[0]["name"] if ranked else None,
        summary_json=rel_path(summary_json, repo_root),
        manifest_json=rel_path(manifest_json, repo_root),
    )

    return SuiteResult(
        output_root=output_root,
        summary_json=summary_json,
        summary_md=summary_md,
        manifest_json=manifest_json,
    )
