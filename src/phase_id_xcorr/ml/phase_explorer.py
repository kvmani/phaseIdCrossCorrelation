"""Raw `.oh5` exploratory analytics backend for phase-wise GUI histogram analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from .config import get_required, load_yaml, resolve_path
from .dataset_io import rel_path, write_json
from .labels import load_label_csv
from .oh5_reader import Oh5ScanReader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SOURCE_MODE_CSV = "oh5_csv_labels"
SOURCE_MODE_SINGLE_PHASE = "single_phase_scan_map"
SUPPORTED_SOURCE_MODES = (SOURCE_MODE_CSV, SOURCE_MODE_SINGLE_PHASE)


@dataclass(slots=True)
class PatternRef:
    phase_name: str
    scan_id: str
    oh5_path: Path
    flat_index: int


@dataclass(slots=True)
class PhaseExplorerData:
    phase_name: str
    pattern_refs: list[PatternRef] = field(default_factory=list)
    intensity_values: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    intensity_max_value: float = 1.0
    scalar_fields: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class ExplorerDataset:
    config_path: Path
    output_dir: Path
    phase_names: list[str]
    phases: dict[str, PhaseExplorerData]
    explorer_config: dict[str, Any] = field(default_factory=dict)

    def pattern_count(self, phase_name: str) -> int:
        return len(self.phases[phase_name].pattern_refs)

    def get_pattern(self, phase_name: str, index: int) -> np.ndarray:
        refs = self.phases[phase_name].pattern_refs
        if index < 0 or index >= len(refs):
            raise IndexError(f"pattern index {index} out of range for phase {phase_name}")
        ref = refs[index]
        with Oh5ScanReader(ref.oh5_path) as reader:
            return reader.read_pattern(flat_index=ref.flat_index)


def _parse_phase_map(cfg: dict[str, Any]) -> dict[str, int]:
    phase_to_label: dict[str, int] = {}
    if isinstance(cfg.get("phase_labels"), list):
        for row in cfg["phase_labels"]:
            if not isinstance(row, dict):
                continue
            phase_to_label[str(get_required(row, "name", where="phase_labels[]")).strip()] = int(
                get_required(row, "label", where="phase_labels[]")
            )
    if isinstance(cfg.get("phase_to_label"), dict):
        for phase_name, label in cfg["phase_to_label"].items():
            phase_to_label[str(phase_name).strip()] = int(label)
    return phase_to_label


def _normalize_v3_sources(cfg: dict[str, Any], *, base_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    sources = cfg.get("sources")
    if isinstance(sources, list) and sources:
        return [dict(s) for s in sources if isinstance(s, dict)]

    file_list = cfg.get("listOfFiles")
    if not isinstance(file_list, list) or not file_list:
        return []

    data_source_folder = cfg.get("data_source_folder", ".")
    source_root = resolve_path(data_source_folder, base_dir=base_dir, repo_root=repo_root)
    allow_filename_phase = bool(cfg.get("allow_filename_phase_fallback", False))

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(file_list):
        entry = {"file": row} if isinstance(row, str) else (dict(row) if isinstance(row, dict) else None)
        if entry is None:
            raise ValueError(f"listOfFiles[{idx}] must be string or mapping")
        file_name = str(entry.get("file") or entry.get("oh5") or entry.get("oh5_path") or "").strip()
        if not file_name:
            raise ValueError(f"listOfFiles[{idx}] missing file/oh5_path")
        src = {
            "scan_id": str(entry.get("scan_id") or Path(file_name).stem),
            "oh5_path": str((source_root / file_name).resolve()),
        }
        if str(entry.get("labels_csv_path", "")).strip():
            src["labels_csv_path"] = str((source_root / str(entry["labels_csv_path"])).resolve())
        if entry.get("phase_name") is not None:
            src["phase_name"] = entry["phase_name"]
        if entry.get("phase_label") is not None:
            src["phase_label"] = entry["phase_label"]
        if allow_filename_phase and src.get("phase_name") is None and src.get("phase_label") is None:
            token = Path(file_name).stem.split("__")[-1]
            if token:
                src["phase_name"] = token
        out.append(src)
    return out


def _parse_input_mode(cfg: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    mode_raw = cfg.get("input_mode")
    if mode_raw is not None:
        mode = str(mode_raw).strip()
        if mode not in SUPPORTED_SOURCE_MODES:
            raise ValueError(f"Unsupported input_mode '{mode}'")
        return mode

    has_csv = [str(src.get("labels_csv_path", "")).strip() != "" for src in sources]
    has_phase = [str(src.get("phase_name", "")).strip() != "" or str(src.get("phase_label", "")).strip() != "" for src in sources]
    if has_csv and all(has_csv):
        return SOURCE_MODE_CSV
    if has_phase and all(has_phase) and not any(has_csv):
        return SOURCE_MODE_SINGLE_PHASE
    raise ValueError("Could not infer input mode. Set input_mode explicitly.")


def _resolve_source_phase_name(source: dict[str, Any], *, phase_to_label: dict[str, int]) -> str:
    raw_name = str(source.get("phase_name", "")).strip()
    if raw_name:
        return raw_name
    raw_label = source.get("phase_label")
    if raw_label is not None and str(raw_label).strip() != "":
        label_int = int(float(str(raw_label)))
        label_to_phase = {v: k for k, v in phase_to_label.items()}
        if label_int in label_to_phase:
            return label_to_phase[label_int]
    raise ValueError("single_phase_scan_map requires phase_name or phase_label")


def load_explorer_dataset(
    *,
    config_path: Path,
    repo_root: Path,
    logger: logging.Logger | None = None,
    max_intensity_points_per_phase: int = 2_000_000,
    max_scalar_points_per_phase: int = 1_000_000,
) -> ExplorerDataset:
    """Load exploratory phase-wise data from `.oh5` sources described by dataset YAML."""

    log = logger or logging.getLogger(__name__)
    cfg_path = config_path.resolve()
    cfg = load_yaml(cfg_path)
    cfg_dir = cfg_path.parent

    sources = _normalize_v3_sources(cfg, base_dir=cfg_dir, repo_root=repo_root)
    if not sources:
        raise ValueError("No sources/listOfFiles found in config")

    phase_to_label = _parse_phase_map(cfg)
    input_mode = _parse_input_mode(cfg, sources)
    csv_cfg = cfg.get("label_csv") if isinstance(cfg.get("label_csv"), dict) else {}
    explorer_cfg = cfg.get("explorer") if isinstance(cfg.get("explorer"), dict) else {}
    output_dir = resolve_path(cfg.get("output_dir", "reports/ml/datasets/default"), base_dir=cfg_dir, repo_root=repo_root)

    phases: dict[str, PhaseExplorerData] = {}

    def _get_phase(name: str) -> PhaseExplorerData:
        if name not in phases:
            phases[name] = PhaseExplorerData(phase_name=name)
        return phases[name]

    for src_idx, src in enumerate(sources):
        scan_id = str(src.get("scan_id") or f"scan_{src_idx:03d}")
        oh5_path = resolve_path(get_required(src, "oh5_path", where=f"sources[{src_idx}]"), base_dir=cfg_dir, repo_root=repo_root)

        with Oh5ScanReader(oh5_path) as reader:
            scalar_fields = reader.discover_scalar_fields()
            log.info("Source %s: %s scalar fields discovered", scan_id, len(scalar_fields))

            rows: list[tuple[str, int]] = []
            if input_mode == SOURCE_MODE_SINGLE_PHASE:
                phase_name = _resolve_source_phase_name(src, phase_to_label=phase_to_label)
                rows = [(phase_name, i) for i in range(reader.total_pixels)]
                phase = _get_phase(phase_name)
                for field_name in scalar_fields:
                    values = reader.read_scalar_field_array(field_name)
                    if values is None or values.size == 0:
                        continue
                    arr = phase.scalar_fields.get(field_name)
                    phase.scalar_fields[field_name] = values.copy() if arr is None else np.concatenate((arr, values))
                    if phase.scalar_fields[field_name].size > max_scalar_points_per_phase:
                        rng = np.random.default_rng(42)
                        idx = rng.choice(
                            phase.scalar_fields[field_name].size,
                            size=max_scalar_points_per_phase,
                            replace=False,
                        )
                        phase.scalar_fields[field_name] = phase.scalar_fields[field_name][idx]
            else:
                labels_csv_path = resolve_path(
                    get_required(src, "labels_csv_path", where=f"sources[{src_idx}]"),
                    base_dir=cfg_dir,
                    repo_root=repo_root,
                )
                label_rows, _ = load_label_csv(csv_path=labels_csv_path, phase_to_label=phase_to_label, csv_config=csv_cfg)
                for row in label_rows:
                    flat_index = row.flat_index
                    if flat_index is None:
                        if row.x is None or row.y is None:
                            continue
                        flat_index = reader.xy_to_flat(row.x, row.y)
                    rows.append((row.phase_name, int(flat_index)))

            for phase_name, flat_index in rows:
                phase = _get_phase(phase_name)
                intensity_max_value = float((2 ** int(reader.pattern_bit_depth)) - 1) if reader.pattern_bit_depth else 1.0
                phase.intensity_max_value = max(float(phase.intensity_max_value), intensity_max_value)
                phase.pattern_refs.append(
                    PatternRef(
                        phase_name=phase_name,
                        scan_id=scan_id,
                        oh5_path=oh5_path,
                        flat_index=int(flat_index),
                    )
                )

                pattern = reader.read_pattern(flat_index=flat_index)
                flat = (pattern.reshape(-1).astype(np.float32, copy=False) * intensity_max_value).astype(np.float32, copy=False)
                if phase.intensity_values.size == 0:
                    phase.intensity_values = flat.copy()
                else:
                    phase.intensity_values = np.concatenate((phase.intensity_values, flat))
                if phase.intensity_values.size > max_intensity_points_per_phase:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(phase.intensity_values.size, size=max_intensity_points_per_phase, replace=False)
                    phase.intensity_values = phase.intensity_values[idx]

                if input_mode != SOURCE_MODE_SINGLE_PHASE:
                    scalar_row = reader.read_scalar_row_all(flat_index=flat_index, field_names=scalar_fields)
                    for field_name, value in scalar_row.items():
                        if value is None:
                            continue
                        arr = phase.scalar_fields.get(field_name)
                        v = np.asarray([float(value)], dtype=np.float32)
                        phase.scalar_fields[field_name] = v if arr is None else np.concatenate((arr, v))
                        if phase.scalar_fields[field_name].size > max_scalar_points_per_phase:
                            rng = np.random.default_rng(42)
                            idx = rng.choice(
                                phase.scalar_fields[field_name].size,
                                size=max_scalar_points_per_phase,
                                replace=False,
                            )
                            phase.scalar_fields[field_name] = phase.scalar_fields[field_name][idx]

    if not phases:
        raise RuntimeError("No phase data loaded from provided configuration")

    return ExplorerDataset(
        config_path=cfg_path,
        output_dir=output_dir,
        phase_names=sorted(phases.keys()),
        phases=phases,
        explorer_config=explorer_cfg,
    )


def _sanitize_axis_limits(x_min: float, x_max: float) -> tuple[float, float]:
    if x_max <= x_min:
        return float(x_min), float(x_min + 1e-6)
    return float(x_min), float(x_max)


def _default_attr_limits(dataset: ExplorerDataset, attribute: str) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for phase_name in dataset.phase_names:
        field_values = dataset.phases[phase_name].scalar_fields.get(attribute)
        if field_values is not None and field_values.size > 0:
            values.append(field_values)
    if not values:
        raise ValueError(f"No scalar field data found for attribute '{attribute}'")
    merged = np.concatenate(values)
    return _sanitize_axis_limits(float(np.min(merged)), float(np.max(merged)))


def _plot_histogram_png(
    *,
    counts: np.ndarray,
    edges: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    color: str,
    output_path: Path,
    dpi: int,
    figure_size: tuple[float, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    ax.bar(centers, counts, width=widths, align="center", color=color, edgecolor=color, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_phase_explorer_artifacts(
    *,
    dataset: ExplorerDataset,
    repo_root: Path,
    logger: logging.Logger | None = None,
) -> Path:
    """Export publication-quality histogram PNGs and a machine-readable JSON manifest."""

    log = logger or logging.getLogger(__name__)
    explorer_cfg = dataset.explorer_config
    intensity_cfg = explorer_cfg.get("intensity_plot") if isinstance(explorer_cfg.get("intensity_plot"), dict) else {}
    attr_cfg = explorer_cfg.get("attribute_plot") if isinstance(explorer_cfg.get("attribute_plot"), dict) else {}
    export_cfg = explorer_cfg.get("export") if isinstance(explorer_cfg.get("export"), dict) else {}

    dpi = int(export_cfg.get("dpi", 300))
    figure_size_cfg = export_cfg.get("figure_size_inches", [6.5, 4.5])
    if not isinstance(figure_size_cfg, (list, tuple)) or len(figure_size_cfg) != 2:
        figure_size_cfg = [6.5, 4.5]
    figure_size = (float(figure_size_cfg[0]), float(figure_size_cfg[1]))
    intensity_bins = int(intensity_cfg.get("bins", 256))
    attr_bins = int(attr_cfg.get("bins", 48))
    field_ranges = attr_cfg.get("field_ranges") if isinstance(attr_cfg.get("field_ranges"), dict) else {}
    exported_attributes = export_cfg.get("attributes", ["CI", "IQ", "Fit"])
    if not isinstance(exported_attributes, list):
        exported_attributes = ["CI", "IQ", "Fit"]
    output_dir = dataset.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )

    intensity_x_min = float(intensity_cfg.get("x_min", 0.0) if intensity_cfg.get("x_min") is not None else 0.0)
    intensity_max_default = max((phase.intensity_max_value for phase in dataset.phases.values()), default=1.0)
    intensity_x_max = float(
        intensity_cfg.get("x_max", intensity_max_default) if intensity_cfg.get("x_max") is not None else intensity_max_default
    )
    intensity_x_limits = _sanitize_axis_limits(intensity_x_min, intensity_x_max)

    intensity_phase_histograms: dict[str, dict[str, Any]] = {}
    intensity_y_max = 0.0
    for phase_name in dataset.phase_names:
        counts, edges = histogram(
            dataset.phases[phase_name].intensity_values,
            bins=intensity_bins,
            x_min=intensity_x_limits[0],
            x_max=intensity_x_limits[1],
        )
        intensity_phase_histograms[phase_name] = {"counts": counts, "edges": edges}
        intensity_y_max = max(intensity_y_max, float(np.max(counts)) if counts.size else 0.0)
    intensity_y_limits = (0.0, max(1.0, intensity_y_max * 1.05))

    exports: list[dict[str, Any]] = []
    for phase_name in dataset.phase_names:
        output_path = output_dir / f"{phase_name}_intensity_distribution.png"
        hist_payload = intensity_phase_histograms[phase_name]
        _plot_histogram_png(
            counts=hist_payload["counts"],
            edges=hist_payload["edges"],
            title=f"{phase_name} intensity distribution",
            x_label="Intensity",
            y_label="Pixel count",
            x_limits=intensity_x_limits,
            y_limits=intensity_y_limits,
            color="#1f77b4",
            output_path=output_path,
            dpi=dpi,
            figure_size=figure_size,
        )
        exports.append(
            {
                "phase": phase_name,
                "plot_type": "intensity_distribution",
                "attribute": None,
                "path": rel_path(output_path, repo_root),
                "x_limits": list(intensity_x_limits),
                "y_limits": list(intensity_y_limits),
                "bins": intensity_bins,
            }
        )

    for attribute in [str(x).strip() for x in exported_attributes if str(x).strip()]:
        if attribute in field_ranges and isinstance(field_ranges[attribute], (list, tuple)) and len(field_ranges[attribute]) == 2:
            attr_x_limits = _sanitize_axis_limits(float(field_ranges[attribute][0]), float(field_ranges[attribute][1]))
        else:
            attr_x_limits = _default_attr_limits(dataset, attribute)

        non_empty_values = [
            dataset.phases[phase_name].scalar_fields.get(attribute)
            for phase_name in dataset.phase_names
            if dataset.phases[phase_name].scalar_fields.get(attribute) is not None
            and dataset.phases[phase_name].scalar_fields.get(attribute).size > 0
        ]
        if not non_empty_values:
            log.warning("Skipping attribute export for %s because no values were loaded for any phase", attribute)
            continue

        probe_has_counts = False
        for values in non_empty_values:
            probe_counts, _ = histogram(values, bins=attr_bins, x_min=attr_x_limits[0], x_max=attr_x_limits[1])
            if np.any(probe_counts):
                probe_has_counts = True
                break
        if not probe_has_counts:
            merged_values = np.concatenate(non_empty_values)
            attr_x_limits = _sanitize_axis_limits(float(np.min(merged_values)), float(np.max(merged_values)))

        attribute_histograms: dict[str, dict[str, Any]] = {}
        attr_y_max = 0.0
        for phase_name in dataset.phase_names:
            values = dataset.phases[phase_name].scalar_fields.get(attribute)
            if values is None or values.size == 0:
                log.warning("Skipping export for phase=%s attribute=%s because no values were loaded", phase_name, attribute)
                continue
            counts, edges = histogram(values, bins=attr_bins, x_min=attr_x_limits[0], x_max=attr_x_limits[1])
            attribute_histograms[phase_name] = {"counts": counts, "edges": edges}
            attr_y_max = max(attr_y_max, float(np.max(counts)) if counts.size else 0.0)

        attr_y_limits = (0.0, max(1.0, attr_y_max * 1.05))
        for phase_name in dataset.phase_names:
            hist_payload = attribute_histograms.get(phase_name)
            if hist_payload is None:
                continue
            output_path = output_dir / f"{phase_name}_{attribute}.png"
            _plot_histogram_png(
                counts=hist_payload["counts"],
                edges=hist_payload["edges"],
                title=f"{phase_name} {attribute} distribution",
                x_label=attribute,
                y_label="Pixel count",
                x_limits=attr_x_limits,
                y_limits=attr_y_limits,
                color="#2ca02c",
                output_path=output_path,
                dpi=dpi,
                figure_size=figure_size,
            )
            exports.append(
                {
                    "phase": phase_name,
                    "plot_type": "attribute_distribution",
                    "attribute": attribute,
                    "path": rel_path(output_path, repo_root),
                    "x_limits": list(attr_x_limits),
                    "y_limits": list(attr_y_limits),
                    "bins": attr_bins,
                }
            )

    manifest = {
        "config_path": rel_path(dataset.config_path, repo_root),
        "output_dir": rel_path(output_dir, repo_root),
        "phase_names": list(dataset.phase_names),
        "exports": exports,
    }
    manifest_path = output_dir / "phase_explorer_exports.json"
    write_json(manifest_path, manifest)
    log.info("Exported %s phase explorer figures to %s", len(exports), output_dir)
    return manifest_path


def histogram(values: np.ndarray, *, bins: int, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, bin_edges)."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    counts, edges = np.histogram(values, bins=bins, range=(x_min, x_max))
    return counts.astype(np.float64), edges.astype(np.float64)


def cdf_from_counts(cumulative_counts: np.ndarray) -> np.ndarray:
    if cumulative_counts.size == 0:
        return np.zeros((0,), dtype=np.float64)
    total = float(cumulative_counts[-1])
    if total <= 0:
        return np.zeros_like(cumulative_counts, dtype=np.float64)
    return cumulative_counts.astype(np.float64) / total


def build_intensity_mask(pattern: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    """Build boolean mask for union of x-ranges over intensity values."""

    if pattern.ndim != 2:
        raise ValueError("pattern must be 2D")
    if not ranges:
        return np.zeros_like(pattern, dtype=bool)
    mask = np.zeros_like(pattern, dtype=bool)
    for lo, hi in ranges:
        lo2, hi2 = (float(lo), float(hi)) if lo <= hi else (float(hi), float(lo))
        mask |= (pattern >= lo2) & (pattern <= hi2)
    return mask
