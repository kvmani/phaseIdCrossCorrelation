"""CSV label ingestion for ML dataset preparation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LabelRecord:
    """Single label row referencing one EBSD pixel."""

    row_index: int
    sample_id: str
    x: int | None
    y: int | None
    flat_index: int | None
    phase_name: str
    label: int
    raw: dict[str, str]


@dataclass(slots=True)
class LabelLoadSummary:
    """Summary of parsed CSV labels."""

    csv_path: str
    rows_total: int
    rows_loaded: int
    phase_counts: dict[str, int]


def _normalize_phase_name(text: str) -> str:
    return text.strip()


def _read_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return int(float(text))


def load_label_csv(
    *,
    csv_path: Path,
    phase_to_label: dict[str, int],
    csv_config: dict[str, Any],
) -> tuple[list[LabelRecord], LabelLoadSummary]:
    """Load one label CSV file into normalized per-pixel records."""

    x_col = str(csv_config.get("x_col", "x"))
    y_col = str(csv_config.get("y_col", "y"))
    flat_index_col = str(csv_config.get("flat_index_col", "flat_index"))
    phase_name_col = str(csv_config.get("phase_name_col", "phase_name"))
    phase_label_col = str(csv_config.get("phase_label_col", ""))
    sample_id_col = str(csv_config.get("sample_id_col", "sample_id"))

    rows: list[LabelRecord] = []
    phase_counts: dict[str, int] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        rows_total = 0
        for idx, raw in enumerate(reader, start=2):
            rows_total += 1

            sample_id = str(raw.get(sample_id_col, "")).strip() or f"row_{idx - 1:06d}"
            x = _read_optional_int(raw.get(x_col)) if x_col else None
            y = _read_optional_int(raw.get(y_col)) if y_col else None
            flat_index = _read_optional_int(raw.get(flat_index_col)) if flat_index_col else None

            phase_name = ""
            label: int | None = None

            raw_phase_name = str(raw.get(phase_name_col, "")).strip() if phase_name_col else ""
            raw_phase_label = str(raw.get(phase_label_col, "")).strip() if phase_label_col else ""

            if raw_phase_name:
                phase_name = _normalize_phase_name(raw_phase_name)
                if phase_name not in phase_to_label:
                    raise ValueError(
                        f"Unknown phase_name '{phase_name}' in {csv_path} line {idx}. "
                        f"Configured phase names: {sorted(phase_to_label)}"
                    )
                label = int(phase_to_label[phase_name])
            elif raw_phase_label:
                label = int(float(raw_phase_label))
                inv = {v: k for k, v in phase_to_label.items()}
                if label not in inv:
                    raise ValueError(
                        f"Unknown numeric label '{label}' in {csv_path} line {idx}. "
                        f"Configured labels: {sorted(inv)}"
                    )
                phase_name = inv[label]
            else:
                raise ValueError(
                    f"Missing phase assignment in {csv_path} line {idx}. "
                    f"Provide '{phase_name_col}' or '{phase_label_col}'."
                )

            if flat_index is None and (x is None or y is None):
                raise ValueError(
                    f"Missing location in {csv_path} line {idx}. "
                    f"Provide '{flat_index_col}' or both '{x_col}' and '{y_col}'."
                )

            rows.append(
                LabelRecord(
                    row_index=idx,
                    sample_id=sample_id,
                    x=x,
                    y=y,
                    flat_index=flat_index,
                    phase_name=phase_name,
                    label=label,
                    raw={k: str(v) for k, v in raw.items()},
                )
            )
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1

    summary = LabelLoadSummary(
        csv_path=str(csv_path),
        rows_total=rows_total,
        rows_loaded=len(rows),
        phase_counts=phase_counts,
    )
    return rows, summary
