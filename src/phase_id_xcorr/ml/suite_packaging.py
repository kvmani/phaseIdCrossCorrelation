"""Lightweight packaging helpers for ML benchmark-suite artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

from .config import resolve_path
from .dataset_io import read_json, rel_path, write_json


ALLOWED_SUFFIXES = {
    ".csv",
    ".html",
    ".jpeg",
    ".jpg",
    ".json",
    ".jsonl",
    ".md",
    ".pdf",
    ".png",
    ".pptx",
    ".svg",
    ".txt",
    ".yml",
    ".yaml",
}

EXCLUDED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    ".onnx",
    ".pt",
    ".pth",
    ".tar",
    ".zip",
}


@dataclass(slots=True)
class SuitePackageResult:
    zip_path: Path
    manifest_path: Path
    included_files: list[str]
    excluded_files: list[dict[str, Any]]


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*")):
        if item.is_file():
            yield item


def _discover_dataset_related_paths(*, suite_root: Path, repo_root: Path) -> list[Path]:
    discovered: list[Path] = []
    seen: set[str] = set()
    for report_path in sorted(suite_root.rglob("report.json")):
        report = read_json(report_path)
        dataset_manifest_rel = str(report.get("dataset_manifest_path", "")).strip()
        if not dataset_manifest_rel:
            continue
        dataset_manifest_path = resolve_path(dataset_manifest_rel, base_dir=repo_root, repo_root=repo_root)
        if not dataset_manifest_path.exists():
            continue
        key = str(dataset_manifest_path.resolve())
        if key not in seen:
            discovered.append(dataset_manifest_path)
            seen.add(key)

        dataset_manifest = read_json(dataset_manifest_path)
        artifacts = dataset_manifest.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        for value in artifacts.values():
            if not isinstance(value, str) or not value.strip():
                continue
            artifact_path = resolve_path(value, base_dir=repo_root, repo_root=repo_root)
            if artifact_path.exists():
                artifact_key = str(artifact_path.resolve())
                if artifact_key not in seen:
                    discovered.append(artifact_path)
                    seen.add(artifact_key)
    return discovered


def _discover_inference_related_paths(*, inference_root: Path, repo_root: Path) -> list[Path]:
    discovered: list[Path] = []
    seen: set[str] = set()
    summary_json = inference_root / "suite_full_scan_summary.json"
    if not summary_json.exists():
        return discovered

    for path in (summary_json, inference_root / "suite_full_scan_summary.md", inference_root / "comparison_report.html", inference_root / "manifest.json", inference_root / "events.jsonl"):
        if path.exists():
            key = str(path.resolve())
            if key not in seen:
                discovered.append(path)
                seen.add(key)

    payload = read_json(summary_json)
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return discovered

    for row in rows:
        if not isinstance(row, dict):
            continue
        artifacts = row.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        for value in artifacts.values():
            if not isinstance(value, str) or not value.strip():
                continue
            artifact_path = resolve_path(value, base_dir=repo_root, repo_root=repo_root)
            if artifact_path.exists():
                key = str(artifact_path.resolve())
                if key not in seen:
                    discovered.append(artifact_path)
                    seen.add(key)
    return discovered


def package_benchmark_suite_artifacts(
    *,
    suite_root: Path,
    repo_root: Path,
    output_zip: Path,
    inference_roots: list[Path] | None = None,
    extra_paths: list[Path] | None = None,
    max_file_size_mb: float = 5.0,
) -> SuitePackageResult:
    """Package lightweight benchmark and inference-review artifacts into a transfer zip."""

    if not suite_root.exists():
        raise FileNotFoundError(f"suite_root not found: {suite_root}")

    max_bytes = int(max(0.1, float(max_file_size_mb)) * 1024 * 1024)
    extra_paths = list(extra_paths or [])
    inference_roots = list(inference_roots or [])

    candidate_roots = [suite_root]
    candidate_roots.extend(_discover_dataset_related_paths(suite_root=suite_root, repo_root=repo_root))
    for inference_root in inference_roots:
        candidate_roots.append(inference_root)
        candidate_roots.extend(_discover_inference_related_paths(inference_root=inference_root, repo_root=repo_root))
    candidate_roots.extend(extra_paths)

    included_paths: list[Path] = []
    excluded_files: list[dict[str, Any]] = []
    seen: set[str] = set()

    for root in candidate_roots:
        for path in _iter_files(root):
            resolved = path.resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)

            suffix = resolved.suffix.lower()
            size_bytes = int(resolved.stat().st_size)
            rel = rel_path(resolved, repo_root)

            if suffix in EXCLUDED_SUFFIXES:
                excluded_files.append({"path": rel, "reason": f"excluded_suffix:{suffix}", "size_bytes": size_bytes})
                continue
            if suffix not in ALLOWED_SUFFIXES:
                excluded_files.append({"path": rel, "reason": f"unsupported_suffix:{suffix or '<none>'}", "size_bytes": size_bytes})
                continue
            if size_bytes > max_bytes:
                excluded_files.append({"path": rel, "reason": f"too_large>{max_file_size_mb:.1f}MB", "size_bytes": size_bytes})
                continue
            included_paths.append(resolved)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_zip.with_name(f"{output_zip.stem}_manifest.json")

    manifest = {
        "schema_version": "phase_id_xcorr.ml_suite_package.v1",
        "suite_root": rel_path(suite_root, repo_root),
        "inference_roots": [rel_path(path, repo_root) for path in inference_roots],
        "output_zip": rel_path(output_zip, repo_root),
        "max_file_size_mb": float(max_file_size_mb),
        "included_files": [rel_path(path, repo_root) for path in included_paths],
        "excluded_files": excluded_files,
        "included_file_count": len(included_paths),
        "excluded_file_count": len(excluded_files),
        "included_total_bytes": int(sum(path.stat().st_size for path in included_paths)),
    }
    write_json(manifest_path, manifest)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in included_paths:
            zf.write(path, arcname=rel_path(path, repo_root))
        zf.writestr("archive_manifest.json", json.dumps(manifest, indent=2))

    return SuitePackageResult(
        zip_path=output_zip,
        manifest_path=manifest_path,
        included_files=manifest["included_files"],
        excluded_files=excluded_files,
    )
