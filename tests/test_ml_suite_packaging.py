from __future__ import annotations

import zipfile
from pathlib import Path

from phase_id_xcorr.ml.dataset_io import write_json
from phase_id_xcorr.ml.suite_packaging import package_benchmark_suite_artifacts


def test_package_benchmark_suite_artifacts_keeps_lightweight_outputs(tmp_path: Path) -> None:
    repo_root = tmp_path
    suite_root = repo_root / "reports" / "ml" / "benchmarks" / "suite_a"
    run_dir = suite_root / "simple_cnn_w16"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = repo_root / "reports" / "ml" / "datasets" / "dataset_a"
    dataset_root.mkdir(parents=True, exist_ok=True)

    dataset_manifest = dataset_root / "manifest.json"
    write_json(
        dataset_manifest,
        {
            "artifacts": {
                "summary_html": "reports/ml/datasets/dataset_a/summary.html",
                "records_csv": "reports/ml/datasets/dataset_a/records.csv",
            }
        },
    )
    (dataset_root / "summary.html").write_text("<html></html>", encoding="utf-8")
    (dataset_root / "records.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    write_json(
        run_dir / "report.json",
        {
            "dataset_manifest_path": "reports/ml/datasets/dataset_a/manifest.json",
            "test_metrics": {"accuracy": 0.9},
        },
    )
    (run_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "resolved_train_config.yml").write_text("epochs: 2\n", encoding="utf-8")
    (run_dir / "best_checkpoint.pt").write_bytes(b"x" * 1024)
    (run_dir / "last_checkpoint.pt").write_bytes(b"x" * 1024)
    (suite_root / "suite_report.html").write_text("<html>suite</html>", encoding="utf-8")
    write_json(suite_root / "suite_summary.json", {"runs_total": 1})

    pptx_path = repo_root / "reports" / "ml" / "presentations" / "suite_a.pptx"
    pptx_path.parent.mkdir(parents=True, exist_ok=True)
    pptx_path.write_bytes(b"ppt")

    result = package_benchmark_suite_artifacts(
        suite_root=suite_root,
        repo_root=repo_root,
        output_zip=suite_root / "suite_a_lightweight_bundle.zip",
        extra_paths=[pptx_path],
        max_file_size_mb=1.0,
    )

    assert result.zip_path.exists()
    assert result.manifest_path.exists()
    assert "reports/ml/benchmarks/suite_a/simple_cnn_w16/best_checkpoint.pt" not in result.included_files
    assert "reports/ml/benchmarks/suite_a/simple_cnn_w16/report.json" in result.included_files
    assert "reports/ml/datasets/dataset_a/summary.html" in result.included_files
    assert "reports/ml/presentations/suite_a.pptx" in result.included_files

    with zipfile.ZipFile(result.zip_path) as zf:
        names = set(zf.namelist())
    assert "reports/ml/benchmarks/suite_a/suite_report.html" in names
    assert "reports/ml/benchmarks/suite_a/simple_cnn_w16/report.json" in names
    assert "reports/ml/presentations/suite_a.pptx" in names
    assert "archive_manifest.json" in names
