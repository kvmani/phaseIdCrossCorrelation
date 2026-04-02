"""End-to-end ML full-cycle orchestration: dataset prep -> benchmark suite -> PPTX/report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .config import apply_overrides, load_yaml, resolve_path
from .dataset_builder import prepare_ml_dataset
from .dataset_io import read_json, rel_path, write_json
from .suite import run_benchmark_suite


@dataclass(slots=True)
class FullCycleResult:
    output_root: Path
    manifest_json: Path
    summary_json: Path
    summary_html: Path
    pptx_path: Path | None


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=False) + "\n")


def _find_generated_pptx(output_dir: Path, *, started_at: float) -> Path | None:
    candidates = sorted(output_dir.glob("*.pptx"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if path.stat().st_mtime >= started_at:
            return path
    return candidates[0] if candidates else None


def _write_html_summary(*, path: Path, summary: dict[str, Any], links: dict[str, str]) -> None:
    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>ML Full Cycle Summary</title>
<style>body{{font-family:Arial,sans-serif;margin:20px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ccc;padding:8px}}</style>
</head><body>
<h1>ML Full-Cycle Summary</h1>
<p>Status: <b>{summary.get('status')}</b></p>
<table>
<tr><th>Stage</th><th>Artifact</th></tr>
<tr><td>Dataset prep manifest</td><td><a href="../{links.get('dataset_manifest','')}">{links.get('dataset_manifest','')}</a></td></tr>
<tr><td>Dataset prep HTML</td><td><a href="../{links.get('dataset_summary_html','')}">{links.get('dataset_summary_html','')}</a></td></tr>
<tr><td>Benchmark suite summary</td><td><a href="../{links.get('suite_summary','')}">{links.get('suite_summary','')}</a></td></tr>
<tr><td>Benchmark suite HTML</td><td><a href="../{links.get('suite_report_html','')}">{links.get('suite_report_html','')}</a></td></tr>
<tr><td>PPTX</td><td>{('<a href="../'+links.get('pptx','')+'">'+links.get('pptx','')+'</a>') if links.get('pptx') else 'not generated'}</td></tr>
</table>
<p>Use suite HTML for concise metrics comparison and drill-down links per experiment.</p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def run_full_cycle(
    *,
    workflow_config_path: Path,
    repo_root: Path,
    debug: bool,
    logger: logging.Logger | None = None,
) -> FullCycleResult:
    """Run full ML workflow from raw `.oh5` configs to suite report and optional PPTX."""

    log = logger or logging.getLogger(__name__)
    cfg_path = workflow_config_path.resolve()
    cfg = load_yaml(cfg_path)
    cfg_dir = cfg_path.parent

    out_root = resolve_path(
        cfg.get("output_root", "reports/ml/full_cycle/default"),
        base_dir=cfg_dir,
        repo_root=repo_root,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    event_log = out_root / "events.jsonl"
    event_log.write_text("", encoding="utf-8")
    t0 = time.monotonic()

    def emit(event: str, **fields: Any) -> None:
        payload = {
            "timestamp_utc": _now_iso_utc(),
            "elapsed_seconds": float(time.monotonic() - t0),
            "event": event,
        }
        payload.update(fields)
        _append_event(event_log, payload)

    emit("RUN_START", workflow_config_path=rel_path(cfg_path, repo_root), output_root=rel_path(out_root, repo_root))

    ds_cfg_path = resolve_path(
        cfg.get("dataset_prepare_config", "configs/ml/dataset_prepare.v3_al_ni_cu.example.yml"),
        base_dir=cfg_dir,
        repo_root=repo_root,
    )
    if not ds_cfg_path.exists():
        raise FileNotFoundError(f"dataset_prepare_config not found: {ds_cfg_path}")

    ds_overrides = cfg.get("dataset_prepare_overrides") if isinstance(cfg.get("dataset_prepare_overrides"), list) else []
    if ds_overrides:
        ds_cfg = apply_overrides(load_yaml(ds_cfg_path), [str(x) for x in ds_overrides])
        ds_cfg_override_path = out_root / "dataset_prepare.resolved.yml"
        import yaml

        ds_cfg_override_path.write_text(yaml.safe_dump(ds_cfg, sort_keys=False), encoding="utf-8")
        ds_cfg_path = ds_cfg_override_path

    emit("DATASET_PREP_START", config_path=rel_path(ds_cfg_path, repo_root))
    ds_result = prepare_ml_dataset(config_path=ds_cfg_path, repo_root=repo_root, debug=debug, logger=log)
    emit("DATASET_PREP_END", manifest_path=rel_path(ds_result.manifest_path, repo_root))
    ds_manifest = read_json(ds_result.manifest_path)
    dataset_summary_html = str(ds_manifest.get("artifacts", {}).get("summary_html", ""))

    suite_cfg_path = resolve_path(
        cfg.get("suite_config", "configs/ml/benchmark_suite.debug.yml"),
        base_dir=cfg_dir,
        repo_root=repo_root,
    )
    if not suite_cfg_path.exists():
        raise FileNotFoundError(f"suite_config not found: {suite_cfg_path}")

    suite_overrides = cfg.get("suite_overrides") if isinstance(cfg.get("suite_overrides"), list) else []
    if suite_overrides:
        suite_cfg = apply_overrides(load_yaml(suite_cfg_path), [str(x) for x in suite_overrides])
    else:
        suite_cfg = load_yaml(suite_cfg_path)

    base_train_path = resolve_path(
        suite_cfg.get("base_train_config", "configs/ml/train.simple_cnn.debug.yml"),
        base_dir=suite_cfg_path.parent,
        repo_root=repo_root,
    )
    if not base_train_path.exists():
        raise FileNotFoundError(f"suite base_train_config not found: {base_train_path}")

    base_train = load_yaml(base_train_path)
    base_train["dataset_manifest_path"] = rel_path(ds_result.manifest_path, repo_root)
    import yaml

    resolved_base_train_path = out_root / "train_base.resolved.yml"
    resolved_base_train_path.write_text(yaml.safe_dump(base_train, sort_keys=False), encoding="utf-8")

    suite_cfg["base_train_config"] = rel_path(resolved_base_train_path, repo_root)
    suite_cfg.setdefault("output_root", rel_path(out_root / "suite", repo_root))
    resolved_suite_path = out_root / "suite.resolved.yml"
    resolved_suite_path.write_text(yaml.safe_dump(suite_cfg, sort_keys=False), encoding="utf-8")

    emit("SUITE_START", config_path=rel_path(resolved_suite_path, repo_root))
    suite_result = run_benchmark_suite(
        suite_config_path=resolved_suite_path,
        repo_root=repo_root,
        debug=debug,
        logger=log,
        strict=bool(cfg.get("strict", False)),
    )
    emit("SUITE_END", summary_json=rel_path(suite_result.summary_json, repo_root))

    pptx_path: Path | None = None
    if bool(cfg.get("generate_ppt", True)):
        ppt_script = resolve_path(
            cfg.get(
                "ppt_script",
                str(Path.home() / ".codex" / "skills" / "ml-results-presentation" / "scripts" / "generate_lab_meeting_ppt.py"),
            ),
            base_dir=cfg_dir,
            repo_root=repo_root,
        )
        if not ppt_script.exists():
            log.warning("PPT script not found; skipping PPT generation: %s", ppt_script)
            emit("PPT_SKIP", reason="ppt_script_missing", ppt_script=str(ppt_script))
        else:
            ppt_out_dir = resolve_path(
                cfg.get("ppt_output_dir", "reports/ml/presentations"),
                base_dir=cfg_dir,
                repo_root=repo_root,
            )
            ppt_out_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            cmd = [
                sys.executable,
                str(ppt_script),
                "--scan-root",
                str(suite_result.output_root),
                "--output-dir",
                str(ppt_out_dir),
                "--max-results",
                str(max(1, int(cfg.get("ppt_max_results", 10)))),
            ]
            if cfg.get("deck_title"):
                cmd.extend(["--deck-title", str(cfg.get("deck_title"))])
            emit("PPT_START", command=cmd)
            subprocess.run(cmd, check=True)
            pptx_path = _find_generated_pptx(ppt_out_dir, started_at=started)
            emit("PPT_END", pptx_path=rel_path(pptx_path, repo_root) if pptx_path else None)

    summary = {
        "schema_version": "phase_id_xcorr.ml_full_cycle.v1",
        "timestamp_utc": _now_iso_utc(),
        "status": "completed",
        "workflow_config_path": rel_path(cfg_path, repo_root),
        "output_root": rel_path(out_root, repo_root),
        "dataset_manifest_path": rel_path(ds_result.manifest_path, repo_root),
        "suite_summary_json": rel_path(suite_result.summary_json, repo_root),
        "suite_manifest_json": rel_path(suite_result.manifest_json, repo_root),
        "suite_summary_md": rel_path(suite_result.summary_md, repo_root),
        "pptx_path": rel_path(pptx_path, repo_root) if pptx_path else None,
        "timing": {"total_elapsed_seconds": float(time.monotonic() - t0)},
        "artifacts": {
            "events_jsonl": rel_path(event_log, repo_root),
            "dataset_prepare_config_resolved": rel_path(ds_cfg_path, repo_root),
            "suite_config_resolved": rel_path(resolved_suite_path, repo_root),
            "base_train_config_resolved": rel_path(resolved_base_train_path, repo_root),
        },
    }

    summary_json = out_root / "full_cycle_summary.json"
    write_json(summary_json, summary)

    suite_manifest = load_yaml(suite_result.manifest_json) if suite_result.manifest_json.suffix in {".yml", ".yaml"} else None
    suite_html_rel = ""
    if suite_manifest is None:
        import json as _json

        suite_manifest = _json.loads(suite_result.manifest_json.read_text(encoding="utf-8"))
    suite_html_rel = str(suite_manifest.get("artifacts", {}).get("suite_report_html", ""))

    summary_html = out_root / "full_cycle_summary.html"
    _write_html_summary(
        path=summary_html,
        summary=summary,
        links={
            "dataset_manifest": summary["dataset_manifest_path"],
            "dataset_summary_html": dataset_summary_html,
            "suite_summary": summary["suite_summary_json"],
            "suite_report_html": suite_html_rel,
            "pptx": summary["pptx_path"] or "",
        },
    )

    manifest_json = out_root / "manifest.json"
    write_json(
        manifest_json,
        {
            "schema_version": "phase_id_xcorr.ml_full_cycle_manifest.v1",
            "timestamp_utc": _now_iso_utc(),
            "workflow": "ml_full_cycle",
            "workflow_config_path": rel_path(cfg_path, repo_root),
            "output_root": rel_path(out_root, repo_root),
            "debug": bool(debug),
            "sanity_checks": {
                "dataset_manifest_exists": ds_result.manifest_path.exists(),
                "suite_summary_exists": suite_result.summary_json.exists(),
                "full_cycle_summary_exists": summary_json.exists(),
            },
            "artifacts": {
                "full_cycle_summary_json": rel_path(summary_json, repo_root),
                "full_cycle_summary_html": rel_path(summary_html, repo_root),
                "event_log_jsonl": rel_path(event_log, repo_root),
                "pptx_path": rel_path(pptx_path, repo_root) if pptx_path else None,
            },
        },
    )

    emit("RUN_END", summary_json=rel_path(summary_json, repo_root), manifest_json=rel_path(manifest_json, repo_root))

    return FullCycleResult(
        output_root=out_root,
        manifest_json=manifest_json,
        summary_json=summary_json,
        summary_html=summary_html,
        pptx_path=pptx_path,
    )
