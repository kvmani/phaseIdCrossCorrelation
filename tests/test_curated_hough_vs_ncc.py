"""Integration test for curated image-vs-hough comparison workflow."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

if importlib.util.find_spec("kikuchipy") is None:
    pytest.skip("kikuchipy not available", allow_module_level=True)

from phase_id_xcorr.evaluation.curated_hough_vs_ncc import (  # noqa: E402
    build_curated_hough_vs_ncc_html,
    run_curated_hough_vs_ncc,
)
from phase_id_xcorr.features import HoughTransformConfig  # noqa: E402


def test_curated_hough_vs_ncc_workflow(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    packet_dir = repo_root / "data/test/student_data_packet_phaseid"
    out_dir = tmp_path / "curated_hough_vs_ncc"

    outputs = run_curated_hough_vs_ncc(
        packet_dir=packet_dir,
        out_dir=out_dir,
        repo_root=repo_root,
        binary_thresholds=[0.35, 0.55],
        hough_config=HoughTransformConfig(n_theta=90, n_rho=45, n_bands=6, use_convolved_map=True),
        debug=True,
    )

    summary_path = outputs["summary_json"]
    report_data_path = outputs["report_data_json"]
    assert summary_path.exists()
    assert report_data_path.exists()
    assert (out_dir / "scores.csv").exists()
    assert (out_dir / "decisions.csv").exists()
    assert (out_dir / "manifest.json").exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert int(summary["cases_total"]) == 3
    metrics = summary["metrics_by_method"]
    assert "image_ncc" in metrics
    assert "hough_ncc_raw" in metrics
    assert "hough_ncc_bin_t350" in metrics
    assert "hough_ncc_bin_t550" in metrics

    html_path = out_dir / "inspection_report.html"
    build_curated_hough_vs_ncc_html(
        report_data_json=report_data_path,
        out_html=html_path,
        repo_root=repo_root,
    )
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Curated Phase-ID: Image NCC vs KikuchiPy Hough-NCC" in html

