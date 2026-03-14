from __future__ import annotations

import json
from pathlib import Path

from phase_id_xcorr.ml.full_cycle import run_full_cycle


def test_run_full_cycle_writes_summary_and_manifest(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path

    ds_cfg = tmp_path / "dataset.yml"
    ds_cfg.write_text(
        """
sources:
  - scan_id: s1
    oh5_path: fake.oh5
    phase_name: a
phase_to_label:
  a: 0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    base_train = tmp_path / "train.yml"
    base_train.write_text("dataset_manifest_path: ''\n", encoding="utf-8")

    suite_cfg = tmp_path / "suite.yml"
    suite_cfg.write_text(
        f"""
base_train_config: {base_train}
experiments:
  - name: x
    overrides: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    wf_cfg = tmp_path / "full.yml"
    wf_cfg.write_text(
        f"""
output_root: out
dataset_prepare_config: {ds_cfg}
suite_config: {suite_cfg}
generate_ppt: false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    class _DSResult:
        def __init__(self) -> None:
            self.manifest_path = tmp_path / "dataset_manifest.json"
            self.manifest_path.write_text('{"ok": true}', encoding="utf-8")

    class _SuiteResult:
        def __init__(self) -> None:
            self.output_root = tmp_path / "suite_out"
            self.output_root.mkdir(parents=True, exist_ok=True)
            self.summary_json = self.output_root / "suite_summary.json"
            self.summary_json.write_text('{"rows": []}', encoding="utf-8")
            self.summary_md = self.output_root / "suite_summary.md"
            self.summary_md.write_text("# s\n", encoding="utf-8")
            self.manifest_json = self.output_root / "manifest.json"
            self.manifest_json.write_text('{"artifacts": {"suite_report_html": "suite_out/suite_report.html"}}', encoding="utf-8")
            (self.output_root / "suite_report.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr("phase_id_xcorr.ml.full_cycle.prepare_ml_dataset", lambda **kwargs: _DSResult())
    monkeypatch.setattr("phase_id_xcorr.ml.full_cycle.run_benchmark_suite", lambda **kwargs: _SuiteResult())

    result = run_full_cycle(workflow_config_path=wf_cfg, repo_root=repo_root, debug=True)

    assert result.summary_json.exists()
    assert result.manifest_json.exists()
    assert result.summary_html.exists()
    payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
