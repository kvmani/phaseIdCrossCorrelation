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


def _parse_optional_float(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    return float(value)


def _resolve_y_limits(
    cfg: dict[str, Any],
    *,
    computed_max: float,
) -> tuple[float, float]:
    y_min = _parse_optional_float(cfg, "y_min")
    y_max = _parse_optional_float(cfg, "y_max")
    if y_min is None:
        y_min = 0.0
    if y_max is None:
        y_max = max(1.0, computed_max * 1.05)
    return _sanitize_axis_limits(float(y_min), float(y_max))


def _resolve_plot_text(
    cfg: dict[str, Any],
    *,
    default_title: str,
    default_x_label: str,
    default_y_label: str,
    phase_name: str,
    attribute: str | None = None,
) -> tuple[str, str, str]:
    title_template = str(cfg.get("title_template", "")).strip()
    x_label_template = str(cfg.get("x_label_template", "")).strip()
    y_label_template = str(cfg.get("y_label_template", "")).strip()
    title = str(cfg.get("title", "")).strip()
    x_label = str(cfg.get("x_label", "")).strip()
    y_label = str(cfg.get("y_label", "")).strip()
    context = {"phase": phase_name, "attribute": attribute or ""}
    if title_template:
        title = title_template.format(**context)
    if x_label_template:
        x_label = x_label_template.format(**context)
    if y_label_template:
        y_label = y_label_template.format(**context)
    return (
        title or default_title,
        x_label or default_x_label,
        y_label or default_y_label,
    )


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
    figure_facecolor: str,
    axes_facecolor: str,
    title_fontsize: float,
    label_fontsize: float,
    x_tick_labelsize: float,
    y_tick_labelsize: float,
    tick_width: float,
    tick_length: float,
    minor_tick_width: float,
    minor_tick_length: float,
    tick_direction: str,
    x_tick_rotation: float,
    y_tick_rotation: float,
    spine_linewidth: float,
    grid_linewidth: float,
    grid_alpha: float,
    title_pad: float,
    label_pad: float,
    bar_linewidth: float,
    edge_color: str,
    show_minor_ticks: bool,
    savefig_pad_inches: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True, facecolor=figure_facecolor)
    ax.set_facecolor(axes_facecolor)
    ax.bar(
        centers,
        counts,
        width=widths,
        align="center",
        color=color,
        edgecolor=edge_color,
        linewidth=bar_linewidth,
    )
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel(x_label, fontsize=label_fontsize, labelpad=label_pad)
    ax.set_ylabel(y_label, fontsize=label_fontsize, labelpad=label_pad)
    ax.tick_params(
        axis="x",
        which="major",
        labelsize=x_tick_labelsize,
        width=tick_width,
        length=tick_length,
        direction=tick_direction,
        rotation=x_tick_rotation,
    )
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=y_tick_labelsize,
        width=tick_width,
        length=tick_length,
        direction=tick_direction,
        rotation=y_tick_rotation,
    )
    if show_minor_ticks:
        ax.minorticks_on()
        ax.tick_params(
            axis="both",
            which="minor",
            width=minor_tick_width,
            length=minor_tick_length,
            direction=tick_direction,
        )
    else:
        ax.minorticks_off()
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(True, alpha=grid_alpha, linewidth=grid_linewidth)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=savefig_pad_inches, facecolor=figure_facecolor)
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
    figure_size_cfg = export_cfg.get("figure_size_inches", [8.5, 5.5])
    if not isinstance(figure_size_cfg, (list, tuple)) or len(figure_size_cfg) != 2:
        figure_size_cfg = [8.5, 5.5]
    figure_size = (float(figure_size_cfg[0]), float(figure_size_cfg[1]))
    font_family = str(export_cfg.get("font_family", "Arial")).strip() or "Arial"
    base_fontsize = float(export_cfg.get("font_size", 18))
    title_fontsize = float(export_cfg.get("title_font_size", 20))
    label_fontsize = float(export_cfg.get("label_font_size", 18))
    tick_labelsize = float(export_cfg.get("tick_label_size", 16))
    x_tick_labelsize = float(export_cfg.get("x_tick_label_size", tick_labelsize))
    y_tick_labelsize = float(export_cfg.get("y_tick_label_size", tick_labelsize))
    tick_width = float(export_cfg.get("tick_width", max(0.8, float(export_cfg.get("spine_line_width", 1.2)))))
    tick_length = float(export_cfg.get("tick_length", 6.0))
    minor_tick_width = float(export_cfg.get("minor_tick_width", max(0.6, tick_width * 0.8)))
    minor_tick_length = float(export_cfg.get("minor_tick_length", max(3.0, tick_length * 0.6)))
    tick_direction = str(export_cfg.get("tick_direction", "out")).strip() or "out"
    x_tick_rotation = float(export_cfg.get("x_tick_rotation", 0.0))
    y_tick_rotation = float(export_cfg.get("y_tick_rotation", 0.0))
    spine_linewidth = float(export_cfg.get("spine_line_width", 1.2))
    grid_linewidth = float(export_cfg.get("grid_line_width", 0.8))
    grid_alpha = float(export_cfg.get("grid_alpha", 0.25))
    title_pad = float(export_cfg.get("title_pad", 10.0))
    label_pad = float(export_cfg.get("label_pad", 8.0))
    figure_facecolor = str(export_cfg.get("figure_facecolor", "white")).strip() or "white"
    axes_facecolor = str(export_cfg.get("axes_facecolor", "white")).strip() or "white"
    show_minor_ticks = bool(export_cfg.get("show_minor_ticks", False))
    savefig_pad_inches = float(export_cfg.get("savefig_pad_inches", 0.1))
    intensity_bins = int(intensity_cfg.get("bins", 256))
    attr_bins = int(attr_cfg.get("bins", 48))
    field_ranges = attr_cfg.get("field_ranges") if isinstance(attr_cfg.get("field_ranges"), dict) else {}
    field_y_ranges = attr_cfg.get("field_y_ranges") if isinstance(attr_cfg.get("field_y_ranges"), dict) else {}
    exported_attributes = export_cfg.get("attributes", ["CI", "IQ", "Fit"])
    if not isinstance(exported_attributes, list):
        exported_attributes = ["CI", "IQ", "Fit"]
    output_dir = dataset.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": base_fontsize,
            "axes.titlesize": title_fontsize,
            "axes.labelsize": label_fontsize,
            "xtick.labelsize": x_tick_labelsize,
            "ytick.labelsize": y_tick_labelsize,
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
    intensity_color = str(intensity_cfg.get("color", "#1f77b4")).strip() or "#1f77b4"
    intensity_edge_color = str(intensity_cfg.get("edge_color", intensity_color)).strip() or intensity_color
    intensity_bar_linewidth = float(intensity_cfg.get("bar_line_width", 0.8))

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
    intensity_y_limits = _resolve_y_limits(intensity_cfg, computed_max=intensity_y_max)

    exports: list[dict[str, Any]] = []
    for phase_name in dataset.phase_names:
        output_path = output_dir / f"{phase_name}_intensity_distribution.png"
        hist_payload = intensity_phase_histograms[phase_name]
        title, x_label, y_label = _resolve_plot_text(
            intensity_cfg,
            default_title=f"{phase_name} intensity distribution",
            default_x_label="Intensity",
            default_y_label="Pixel count",
            phase_name=phase_name,
        )
        _plot_histogram_png(
            counts=hist_payload["counts"],
            edges=hist_payload["edges"],
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_limits=intensity_x_limits,
            y_limits=intensity_y_limits,
            color=intensity_color,
            output_path=output_path,
            dpi=dpi,
            figure_size=figure_size,
            figure_facecolor=figure_facecolor,
            axes_facecolor=axes_facecolor,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            x_tick_labelsize=x_tick_labelsize,
            y_tick_labelsize=y_tick_labelsize,
            tick_width=tick_width,
            tick_length=tick_length,
            minor_tick_width=minor_tick_width,
            minor_tick_length=minor_tick_length,
            tick_direction=tick_direction,
            x_tick_rotation=x_tick_rotation,
            y_tick_rotation=y_tick_rotation,
            spine_linewidth=spine_linewidth,
            grid_linewidth=grid_linewidth,
            grid_alpha=grid_alpha,
            title_pad=title_pad,
            label_pad=label_pad,
            bar_linewidth=intensity_bar_linewidth,
            edge_color=intensity_edge_color,
            show_minor_ticks=show_minor_ticks,
            savefig_pad_inches=savefig_pad_inches,
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
                "title": title,
                "x_label": x_label,
                "y_label": y_label,
                "color": intensity_color,
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

        attr_y_limits = (
            _sanitize_axis_limits(float(field_y_ranges[attribute][0]), float(field_y_ranges[attribute][1]))
            if attribute in field_y_ranges
            and isinstance(field_y_ranges[attribute], (list, tuple))
            and len(field_y_ranges[attribute]) == 2
            else _resolve_y_limits(attr_cfg, computed_max=attr_y_max)
        )
        attribute_color = str(attr_cfg.get("color", "#2ca02c")).strip() or "#2ca02c"
        attribute_edge_color = str(attr_cfg.get("edge_color", attribute_color)).strip() or attribute_color
        attribute_bar_linewidth = float(attr_cfg.get("bar_line_width", 0.8))
        for phase_name in dataset.phase_names:
            hist_payload = attribute_histograms.get(phase_name)
            if hist_payload is None:
                continue
            output_path = output_dir / f"{phase_name}_{attribute}.png"
            title, x_label, y_label = _resolve_plot_text(
                attr_cfg,
                default_title=f"{phase_name} {attribute} distribution",
                default_x_label=attribute,
                default_y_label="Pixel count",
                phase_name=phase_name,
                attribute=attribute,
            )
            _plot_histogram_png(
                counts=hist_payload["counts"],
                edges=hist_payload["edges"],
                title=title,
                x_label=x_label,
                y_label=y_label,
                x_limits=attr_x_limits,
                y_limits=attr_y_limits,
                color=attribute_color,
                output_path=output_path,
                dpi=dpi,
                figure_size=figure_size,
                figure_facecolor=figure_facecolor,
                axes_facecolor=axes_facecolor,
                title_fontsize=title_fontsize,
                label_fontsize=label_fontsize,
                x_tick_labelsize=x_tick_labelsize,
                y_tick_labelsize=y_tick_labelsize,
                tick_width=tick_width,
                tick_length=tick_length,
                minor_tick_width=minor_tick_width,
                minor_tick_length=minor_tick_length,
                tick_direction=tick_direction,
                x_tick_rotation=x_tick_rotation,
                y_tick_rotation=y_tick_rotation,
                spine_linewidth=spine_linewidth,
                grid_linewidth=grid_linewidth,
                grid_alpha=grid_alpha,
                title_pad=title_pad,
                label_pad=label_pad,
                bar_linewidth=attribute_bar_linewidth,
                edge_color=attribute_edge_color,
                show_minor_ticks=show_minor_ticks,
                savefig_pad_inches=savefig_pad_inches,
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
                    "title": title,
                    "x_label": x_label,
                    "y_label": y_label,
                    "color": attribute_color,
                }
            )

    manifest = {
        "config_path": rel_path(dataset.config_path, repo_root),
        "output_dir": rel_path(output_dir, repo_root),
        "phase_names": list(dataset.phase_names),
        "export_style": {
            "dpi": dpi,
            "figure_size_inches": list(figure_size),
            "font_family": font_family,
            "font_size": base_fontsize,
            "title_font_size": title_fontsize,
            "label_font_size": label_fontsize,
            "x_tick_label_size": x_tick_labelsize,
            "y_tick_label_size": y_tick_labelsize,
            "tick_width": tick_width,
            "tick_length": tick_length,
            "minor_tick_width": minor_tick_width,
            "minor_tick_length": minor_tick_length,
            "tick_direction": tick_direction,
            "x_tick_rotation": x_tick_rotation,
            "y_tick_rotation": y_tick_rotation,
            "spine_line_width": spine_linewidth,
            "grid_line_width": grid_linewidth,
            "grid_alpha": grid_alpha,
            "title_pad": title_pad,
            "label_pad": label_pad,
            "figure_facecolor": figure_facecolor,
            "axes_facecolor": axes_facecolor,
            "show_minor_ticks": show_minor_ticks,
            "savefig_pad_inches": savefig_pad_inches,
        },
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
