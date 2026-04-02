"""Orientation export and IPF diagnostic helpers for dataset preparation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from diffpy.structure import Atom, Lattice, Structure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
from orix.quaternion import Orientation
import numpy as np

from .dataset_io import rel_path, write_json, write_records_csv


EXPORT_EULER_UNIT = "degree"
PHASE_SPACE_GROUPS = {
    "al": 225,
    "cu": 225,
    "ni": 225,
    "fe_bcc": 229,
    "fe3o4_magnetite": 227,
    "feo_wustite": 225,
}


def _phase_for_name(phase_name: str) -> Phase:
    normalized = str(phase_name).strip().lower()
    if normalized not in PHASE_SPACE_GROUPS:
        raise ValueError(
            f"No IPF symmetry mapping configured for phase '{phase_name}'. "
            f"Known phases: {sorted(PHASE_SPACE_GROUPS)}"
        )
    a0 = 1.0
    structure = Structure(
        atoms=[Atom(str(phase_name), [0.0, 0.0, 0.0])],
        lattice=Lattice(a0, a0, a0, 90.0, 90.0, 90.0),
    )
    return Phase(name=str(phase_name), space_group=PHASE_SPACE_GROUPS[normalized], structure=structure)


def _orientation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        rows.append(
            {
                "sample_id": rec.get("sample_id"),
                "scan_id": rec.get("scan_id"),
                "source_mode": rec.get("source_mode"),
                "oh5_path": rec.get("oh5_path"),
                "labels_csv_path": rec.get("labels_csv_path"),
                "source_phase_name": rec.get("source_phase_name"),
                "source_phase_label": rec.get("source_phase_label"),
                "source_row_index": rec.get("source_row_index"),
                "x": rec.get("x"),
                "y": rec.get("y"),
                "flat_index": rec.get("flat_index"),
                "phase_name": rec.get("phase_name"),
                "label": rec.get("label"),
                "confidence_index": rec.get("confidence_index"),
                "image_quality": rec.get("image_quality"),
                "fit": rec.get("fit"),
                "valid": rec.get("valid"),
                "split": rec.get("split", ""),
                "euler_phi1": rec.get("euler_phi1"),
                "euler_Phi": rec.get("euler_Phi"),
                "euler_phi2": rec.get("euler_phi2"),
                "euler_source_unit": rec.get("euler_source_unit"),
                "euler_export_unit": rec.get("euler_export_unit"),
                "euler_convention": rec.get("euler_convention"),
            }
        )
    return rows


def export_orientation_records(
    *,
    stage: str,
    records: list[dict[str, Any]],
    out_dir: Path,
    repo_root: Path,
    config_path: str,
    source_unit_by_scan: dict[str, str],
) -> tuple[Path, Path]:
    rows = _orientation_rows(records)
    csv_path = out_dir / f"{stage}_orientations.csv"
    json_path = out_dir / f"{stage}_orientations.json"
    write_records_csv(csv_path, rows)

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        split_name = str(row.get("split") or "unsplit")
        grouped[split_name][str(row.get("phase_name"))].append(row)
    payload = {
        "schema_version": "phase_id_xcorr.orientation_export.v1",
        "stage": stage,
        "config_path": config_path,
        "record_count": len(rows),
        "euler_convention": "Bunge ZXZ",
        "euler_export_unit": EXPORT_EULER_UNIT,
        "source_units_by_scan": dict(sorted(source_unit_by_scan.items())),
        "artifacts": {
            "csv": rel_path(csv_path, repo_root),
            "json": rel_path(json_path, repo_root),
        },
        "records": rows,
        "grouped_records": {
            split_name: {phase_name: phase_rows for phase_name, phase_rows in sorted(phases.items())}
            for split_name, phases in sorted(grouped.items())
        },
    }
    write_json(json_path, payload)
    return csv_path, json_path


def generate_ipf_diagnostics(
    *,
    qualified_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    out_dir: Path,
    repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    ipf_root = out_dir / "ipf"
    plot_entries: list[dict[str, Any]] = []

    def _plot(records: list[dict[str, Any]], *, stage: str, split_name: str | None, phase_name: str) -> None:
        if not records:
            return
        phase = _phase_for_name(phase_name)
        eulers = np.asarray(
            [[float(rec["euler_phi1"]), float(rec["euler_Phi"]), float(rec["euler_phi2"])] for rec in records],
            dtype=np.float64,
        )
        orientations = Orientation.from_euler(eulers, symmetry=phase.point_group, degrees=True)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="ipf", symmetry=phase.point_group)
        ax.scatter(orientations, c="tab:blue", s=8, alpha=0.75)
        title = f"{phase_name} {stage}" if split_name is None else f"{phase_name} {stage} {split_name}"
        ax.set_title(title)
        if split_name is None:
            path = ipf_root / stage / f"{phase_name}_ipf.png"
        else:
            path = ipf_root / stage / split_name / f"{phase_name}_ipf.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        plot_entries.append(
            {
                "stage": stage,
                "split": split_name,
                "phase_name": phase_name,
                "count": len(records),
                "path": rel_path(path, repo_root),
            }
        )

    qualified_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in qualified_records:
        qualified_by_phase[str(rec["phase_name"])].append(rec)
    for phase_name, phase_records in sorted(qualified_by_phase.items()):
        _plot(phase_records, stage="qualified", split_name=None, phase_name=phase_name)

    selected_by_split_phase: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for rec in selected_records:
        selected_by_split_phase[str(rec["split"])][str(rec["phase_name"])].append(rec)
    for split_name, phase_map in sorted(selected_by_split_phase.items()):
        for phase_name, phase_records in sorted(phase_map.items()):
            _plot(phase_records, stage="selected", split_name=split_name, phase_name=phase_name)

    index_path = out_dir / "ipf_index.json"
    index_payload = {
        "schema_version": "phase_id_xcorr.orientation_ipf_index.v1",
        "euler_convention": "Bunge ZXZ",
        "euler_export_unit": EXPORT_EULER_UNIT,
        "plots": plot_entries,
    }
    write_json(index_path, index_payload)
    return index_path, index_payload
