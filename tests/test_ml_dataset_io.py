from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from phase_id_xcorr.ml.dataset_io import rel_path


def test_rel_path_returns_repo_relative_path_when_possible(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    path = repo_root / "reports" / "out.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")

    assert rel_path(path, repo_root) == "reports/out.json"


def test_rel_path_falls_back_to_absolute_path_when_relpath_raises(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "external" / "scan.oh5"
    outside.parent.mkdir(parents=True)
    outside.write_text("x", encoding="utf-8")

    with patch("phase_id_xcorr.ml.dataset_io.os.path.relpath", side_effect=ValueError("different mount")):
        assert rel_path(outside, root) == outside.resolve().as_posix()
