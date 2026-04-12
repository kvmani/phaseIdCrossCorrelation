"""Crop/export helpers for EBSD `.oh5` scans plus review-session loading."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from diffpy.structure import Atom, Lattice, Structure
import h5py
import numpy as np
from orix.crystal_map import Phase
from orix.plot import IPFColorKeyTSL
from orix.quaternion import Orientation
from orix.vector import Vector3d

from phase_id_xcorr.ml.dataset_io import rel_path, write_json
from phase_id_xcorr.ml.oh5_reader import EULER_ALIASES, Oh5ScanReader, QUALITY_ALIASES
from phase_id_xcorr.reporting import build_run_manifest


def _normalize_key(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _read_scalar(dataset: h5py.Dataset) -> Any:
    arr = np.asarray(dataset[()])
    if arr.size == 0:
        return None
    return np.ravel(arr)[0]


def _decode_scalar(value: Any) -> str | float | int | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if np.isscalar(value):
        if isinstance(value, (np.bytes_,)):
            return bytes(value).decode("utf-8", errors="replace")
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
    return str(value)


def _normalize_gray(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    values = arr[finite]
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        out = np.zeros_like(arr, dtype=np.float32)
        out[finite] = 0.5
        return out
    out = np.zeros_like(arr, dtype=np.float32)
    out[finite] = (arr[finite] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


@dataclass(slots=True, frozen=True)
class CropSpec:
    """Rectangular crop on a scan grid."""

    row: int
    column: int
    width: int
    height: int

    @property
    def top(self) -> int:
        return int(self.row)

    @property
    def left(self) -> int:
        return int(self.column)

    @property
    def bottom(self) -> int:
        return int(self.row + self.height - 1)

    @property
    def right(self) -> int:
        return int(self.column + self.width - 1)

    def validate_for(self, *, nx: int, ny: int) -> "CropSpec":
        if self.row < 0 or self.column < 0:
            raise ValueError("row and column must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.column + self.width > nx:
            raise ValueError(f"Crop column+width exceeds scan width {nx}")
        if self.row + self.height > ny:
            raise ValueError(f"Crop row+height exceeds scan height {ny}")
        return self

    def source_flat_indices(self, *, nx: int) -> np.ndarray:
        rows = np.arange(self.row, self.row + self.height, dtype=np.int64)
        cols = np.arange(self.column, self.column + self.width, dtype=np.int64)
        grid = rows[:, None] * int(nx) + cols[None, :]
        return np.asarray(grid.reshape(-1), dtype=np.int64)


@dataclass(slots=True, frozen=True)
class PhaseInfo:
    phase_id: int
    material_name: str
    space_group_number: int | None


@dataclass(slots=True)
class ScanVisualData:
    path: Path
    scan_name: str
    nx: int
    ny: int
    total_pixels: int
    header_total_pixels: int
    iq_field_name: str
    iq_map: np.ndarray
    ipf_map: np.ndarray | None
    scalar_field_names: list[str]
    phase_info_by_id: dict[int, PhaseInfo]
    euler_present: bool


@dataclass(slots=True)
class CropExportResult:
    source_path: Path
    output_path: Path
    manifest_path: Path
    crop_spec: CropSpec
    scan_name: str
    pattern_key: str
    original_nx: int
    original_ny: int
    cropped_nx: int
    cropped_ny: int
    iq_field_name: str
    x_position_rebased: bool
    y_position_rebased: bool
    verification_report: "CropVerificationReport"


@dataclass(slots=True)
class CropReviewSession:
    source: ScanVisualData
    cropped: ScanVisualData
    export: CropExportResult
    verification_report: "CropVerificationReport"


@dataclass(slots=True)
class PixelInspectionRecord:
    path: Path
    x: int
    y: int
    flat_index: int
    quality_row: dict[str, float | bool | None]
    scalar_values: dict[str, float | bool | None]
    euler_row_deg: dict[str, float] | None
    phase_id: int | None
    phase_name: str | None
    pattern: np.ndarray


@dataclass(slots=True)
class CropFieldComparison:
    path: str
    kind: str
    status: str
    source_summary: str
    cropped_summary: str
    note: str


@dataclass(slots=True)
class CropVerificationReport:
    scan_name: str
    source_dataset_count: int
    cropped_dataset_count: int
    source_group_count: int
    cropped_group_count: int
    changed_fields: list[CropFieldComparison]
    unchanged_fields: list[CropFieldComparison]


def crop_to_source_coords(crop_spec: CropSpec, *, local_x: int, local_y: int) -> tuple[int, int]:
    if local_x < 0 or local_y < 0 or local_x >= crop_spec.width or local_y >= crop_spec.height:
        raise ValueError(
            f"Local cropped coordinate ({local_x}, {local_y}) out of range for crop {crop_spec.width}x{crop_spec.height}"
        )
    return int(crop_spec.column + local_x), int(crop_spec.row + local_y)


def _find_required_iq_field(reader: Oh5ScanReader) -> str:
    key = reader.quality_field_map.get("image_quality")
    if key is None:
        raise KeyError("IQ / Image Quality field not found in .oh5 file")
    return str(key)


def _phase_info_from_header(header_group: h5py.Group) -> dict[int, PhaseInfo]:
    out: dict[int, PhaseInfo] = {}
    if "Phase" not in header_group:
        return out
    phase_group = header_group["Phase"]
    if not isinstance(phase_group, h5py.Group):
        return out
    for key in phase_group.keys():
        try:
            phase_id = int(str(key))
        except ValueError:
            continue
        entry = phase_group[key]
        if not isinstance(entry, h5py.Group):
            continue
        material_name = str(phase_id)
        space_group_number: int | None = None
        if "MaterialName" in entry:
            decoded = _decode_scalar(_read_scalar(entry["MaterialName"]))
            if decoded is not None:
                material_name = str(decoded)
        if "SpaceGroupNumber" in entry:
            decoded = _decode_scalar(_read_scalar(entry["SpaceGroupNumber"]))
            if isinstance(decoded, int):
                space_group_number = int(decoded)
        out[phase_id] = PhaseInfo(
            phase_id=phase_id,
            material_name=material_name,
            space_group_number=space_group_number,
        )
    return out


def _phase_ids_for_reader(reader: Oh5ScanReader) -> np.ndarray | None:
    if "Phase" in reader.data_group:
        raw = reader.read_scalar_field_array("Phase")
        if raw is not None:
            return np.asarray(np.rint(raw), dtype=np.int32)
    phase_info = _phase_info_from_header(reader.header_group)
    if len(phase_info) == 1:
        phase_id = next(iter(phase_info.keys()))
        return np.full((reader.header_total_pixels,), int(phase_id), dtype=np.int32)
    return None


def _build_phase(space_group_number: int, material_name: str) -> Phase:
    structure = Structure(
        atoms=[Atom(str(material_name), [0.0, 0.0, 0.0])],
        lattice=Lattice(1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
    )
    return Phase(
        name=str(material_name),
        space_group=int(space_group_number),
        structure=structure,
    )


def _render_scan_ipf_map(reader: Oh5ScanReader) -> np.ndarray | None:
    if not reader.euler_present:
        return None
    phase_info_by_id = _phase_info_from_header(reader.header_group)
    phase_ids = _phase_ids_for_reader(reader)
    if phase_ids is None:
        return None
    eulers = np.full((reader.header_total_pixels, 3), np.nan, dtype=np.float32)
    for idx in range(reader.header_total_pixels):
        try:
            row = reader.read_euler_row(flat_index=idx, degrees=True)
        except Exception:
            continue
        eulers[idx, 0] = float(row["phi1"])
        eulers[idx, 1] = float(row["Phi"])
        eulers[idx, 2] = float(row["phi2"])
    image = np.full((reader.ny, reader.nx, 3), 0.1, dtype=np.float32)
    for phase_id, info in sorted(phase_info_by_id.items()):
        if info.space_group_number is None:
            continue
        mask = phase_ids[: reader.header_total_pixels] == int(phase_id)
        if not np.any(mask):
            continue
        phase_eulers = np.asarray(eulers[: reader.header_total_pixels][mask], dtype=np.float64)
        finite_mask = np.all(np.isfinite(phase_eulers), axis=1)
        if not np.any(finite_mask):
            continue
        phase = _build_phase(info.space_group_number, info.material_name)
        orientations = Orientation.from_euler(
            phase_eulers[finite_mask],
            symmetry=phase.point_group,
            degrees=True,
        )
        key = IPFColorKeyTSL(phase.point_group, direction=Vector3d.zvector())
        colors = np.asarray(key.orientation2color(orientations), dtype=np.float32)
        flat_indices = np.flatnonzero(mask)[finite_mask]
        ys = flat_indices // reader.nx
        xs = flat_indices % reader.nx
        image[ys, xs] = colors
    return np.clip(image, 0.0, 1.0)


def load_scan_visual_data(path: Path) -> ScanVisualData:
    resolved = path.expanduser().resolve()
    with Oh5ScanReader(resolved) as reader:
        if not reader.pattern_present:
            raise ValueError(f"Pattern dataset missing in .oh5 file: {resolved}")
        iq_field_name = _find_required_iq_field(reader)
        iq_flat = reader.read_scalar_field_array(iq_field_name)
        if iq_flat is None or iq_flat.size < reader.header_total_pixels:
            raise ValueError(f"IQ field '{iq_field_name}' could not be read as a scan grid from {resolved}")
        iq_map = np.asarray(iq_flat[: reader.header_total_pixels], dtype=np.float32).reshape(reader.ny, reader.nx)
        ipf_map = _render_scan_ipf_map(reader)
        phase_info_by_id = _phase_info_from_header(reader.header_group)
        if reader.scan_group is None:
            raise RuntimeError(f"Scan group missing while loading {resolved}")
        return ScanVisualData(
            path=resolved,
            scan_name=str(reader.scan_group),
            nx=int(reader.nx),
            ny=int(reader.ny),
            total_pixels=int(reader.total_pixels),
            header_total_pixels=int(reader.header_total_pixels),
            iq_field_name=iq_field_name,
            iq_map=_normalize_gray(iq_map),
            ipf_map=ipf_map,
            scalar_field_names=reader.discover_scalar_fields(),
            phase_info_by_id=phase_info_by_id,
            euler_present=bool(reader.euler_present),
        )


def _snapshot_dataset(dataset: h5py.Dataset) -> tuple[dict[str, Any], dict[str, Any]]:
    attrs = {str(key): dataset.attrs[key] for key in dataset.attrs.keys()}
    create_kwargs: dict[str, Any] = {}
    if dataset.compression is not None:
        create_kwargs["compression"] = dataset.compression
    if dataset.compression_opts is not None:
        create_kwargs["compression_opts"] = dataset.compression_opts
    if dataset.shuffle:
        create_kwargs["shuffle"] = True
    if dataset.fletcher32:
        create_kwargs["fletcher32"] = True
    if dataset.scaleoffset is not None:
        create_kwargs["scaleoffset"] = dataset.scaleoffset
    return attrs, create_kwargs


def _replace_dataset(parent: h5py.Group, name: str, data: np.ndarray, *, attrs: dict[str, Any], create_kwargs: dict[str, Any]) -> None:
    if name in parent:
        del parent[name]
    ds = parent.create_dataset(name, data=data, **create_kwargs)
    for key, value in attrs.items():
        ds.attrs[key] = value


def _replace_scalar_dataset(parent: h5py.Group, name: str, value: int) -> None:
    ds = parent[name]
    attrs, create_kwargs = _snapshot_dataset(ds)
    shape_value = np.asarray([value], dtype=ds.dtype)
    _replace_dataset(parent, name, shape_value, attrs=attrs, create_kwargs=create_kwargs)


def _copy_attributes(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key in src.keys():
        dst[key] = src[key]


def _dataset_create_kwargs(dataset: h5py.Dataset, *, shape: tuple[int, ...] | None = None) -> dict[str, Any]:
    attrs, create_kwargs = _snapshot_dataset(dataset)
    kwargs = dict(create_kwargs)
    target_shape = shape if shape is not None else tuple(int(v) for v in dataset.shape)
    if dataset.chunks is not None:
        clipped_chunks = tuple(
            max(1, min(int(chunk), int(size)))
            for chunk, size in zip(dataset.chunks, target_shape)
        )
        kwargs["chunks"] = clipped_chunks
    return {"attrs": attrs, "create_kwargs": kwargs}


def _copy_dataset_verbatim(src_group: h5py.Group, dst_group: h5py.Group, name: str) -> None:
    src_group.copy(name, dst_group, name=name)


def _dataset_summary(dataset: h5py.Dataset) -> str:
    shape = tuple(int(v) for v in dataset.shape)
    if len(shape) == 0 or int(np.prod(shape, dtype=np.int64)) == 1:
        try:
            value = _decode_scalar(_read_scalar(dataset))
        except Exception:
            value = None
        return f"scalar={value!r} dtype={dataset.dtype}"
    return f"shape={shape} dtype={dataset.dtype}"


def _group_paths(handle: h5py.File) -> set[str]:
    out: set[str] = set()
    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Group):
            out.add(name)
    handle.visititems(visit)
    return out


def _dataset_paths(handle: h5py.File) -> set[str]:
    out: set[str] = set()
    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            out.add(name)
    handle.visititems(visit)
    return out


def _expected_dataset_from_source(
    source_ds: h5py.Dataset,
    *,
    scan_group_name: str,
    source_nx: int,
    source_ny: int,
    crop_spec: CropSpec,
    flat_indices: np.ndarray,
) -> tuple[np.ndarray, str]:
    full_name = source_ds.name.lstrip("/")
    if full_name == f"{scan_group_name}/EBSD/Header/nColumns":
        return np.asarray([crop_spec.width], dtype=source_ds.dtype), "grid width updated for crop"
    if full_name == f"{scan_group_name}/EBSD/Header/nRows":
        return np.asarray([crop_spec.height], dtype=source_ds.dtype), "grid height updated for crop"
    cropped = _crop_dataset_array(
        source_ds,
        source_nx=source_nx,
        source_ny=source_ny,
        flat_indices=flat_indices,
        crop_spec=crop_spec,
    )
    if cropped is not None:
        note = "scan-shaped dataset cropped to selected rectangle"
        if source_ds.name.endswith("/X Position"):
            finite = np.isfinite(cropped)
            if np.any(finite):
                cropped = np.asarray(cropped, dtype=np.float32)
                cropped[finite] = cropped[finite] - float(np.min(cropped[finite]))
                cropped = cropped.astype(source_ds.dtype, copy=False)
            note = "scan-shaped X Position cropped and rebased to local origin"
        elif source_ds.name.endswith("/Y Position"):
            finite = np.isfinite(cropped)
            if np.any(finite):
                cropped = np.asarray(cropped, dtype=np.float32)
                cropped[finite] = cropped[finite] - float(np.min(cropped[finite]))
                cropped = cropped.astype(source_ds.dtype, copy=False)
            note = "scan-shaped Y Position cropped and rebased to local origin"
        return np.asarray(cropped), note
    return np.asarray(source_ds[()]), "metadata preserved unchanged"


def _arrays_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if expected.shape != actual.shape:
        return False
    if expected.dtype.fields is not None or actual.dtype.fields is not None:
        return np.array_equal(expected, actual)
    if expected.dtype.kind in {"f", "c"} or actual.dtype.kind in {"f", "c"}:
        return np.array_equal(np.asarray(expected), np.asarray(actual), equal_nan=True)
    return np.array_equal(np.asarray(expected), np.asarray(actual))


def build_crop_verification_report(
    *,
    source_path: Path,
    cropped_path: Path,
    scan_group_name: str,
    source_nx: int,
    source_ny: int,
    crop_spec: CropSpec,
) -> CropVerificationReport:
    flat_indices = crop_spec.source_flat_indices(nx=source_nx)
    changed_fields: list[CropFieldComparison] = []
    unchanged_fields: list[CropFieldComparison] = []
    with h5py.File(source_path, "r") as src_handle, h5py.File(cropped_path, "r") as crop_handle:
        source_groups = _group_paths(src_handle)
        cropped_groups = _group_paths(crop_handle)
        source_datasets = _dataset_paths(src_handle)
        cropped_datasets = _dataset_paths(crop_handle)
        if source_groups != cropped_groups:
            missing = sorted(source_groups - cropped_groups)
            extra = sorted(cropped_groups - source_groups)
            raise AssertionError(f"Group path mismatch after crop. Missing={missing} Extra={extra}")
        if source_datasets != cropped_datasets:
            missing = sorted(source_datasets - cropped_datasets)
            extra = sorted(cropped_datasets - source_datasets)
            raise AssertionError(f"Dataset path mismatch after crop. Missing={missing} Extra={extra}")
        for path in sorted(source_datasets):
            source_ds = src_handle[path]
            cropped_ds = crop_handle[path]
            expected, note = _expected_dataset_from_source(
                source_ds,
                scan_group_name=scan_group_name,
                source_nx=source_nx,
                source_ny=source_ny,
                crop_spec=crop_spec,
                flat_indices=flat_indices,
            )
            actual = np.asarray(cropped_ds[()])
            if not _arrays_equal(expected, actual):
                raise AssertionError(
                    f"Cropped dataset '{path}' did not match expected crop-adjusted value. "
                    f"Expected {_dataset_summary(source_ds)} -> {expected.shape}, got {_dataset_summary(cropped_ds)}."
                )
            original_raw = np.asarray(source_ds[()])
            changed = not _arrays_equal(original_raw, actual)
            item = CropFieldComparison(
                path=path,
                kind="dataset",
                status="changed" if changed else "unchanged",
                source_summary=_dataset_summary(source_ds),
                cropped_summary=_dataset_summary(cropped_ds),
                note=note if changed else "identical to source",
            )
            if changed:
                changed_fields.append(item)
            else:
                unchanged_fields.append(item)
    return CropVerificationReport(
        scan_name=scan_group_name,
        source_dataset_count=len(source_datasets),
        cropped_dataset_count=len(cropped_datasets),
        source_group_count=len(source_groups),
        cropped_group_count=len(cropped_groups),
        changed_fields=changed_fields,
        unchanged_fields=unchanged_fields,
    )


def _copy_group_recursive(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    *,
    scan_group_name: str,
    source_nx: int,
    source_ny: int,
    crop_spec: CropSpec,
    flat_indices: np.ndarray,
) -> None:
    for name in src_group.keys():
        obj = src_group[name]
        full_name = obj.name.lstrip("/")
        if isinstance(obj, h5py.Group):
            new_group = dst_group.create_group(name)
            _copy_attributes(obj.attrs, new_group.attrs)
            _copy_group_recursive(
                obj,
                new_group,
                scan_group_name=scan_group_name,
                source_nx=source_nx,
                source_ny=source_ny,
                crop_spec=crop_spec,
                flat_indices=flat_indices,
            )
            continue
        if not isinstance(obj, h5py.Dataset):
            continue
        if full_name == f"{scan_group_name}/EBSD/Header/nColumns":
            meta = _dataset_create_kwargs(obj, shape=(1,))
            data = np.asarray([crop_spec.width], dtype=obj.dtype)
            ds = dst_group.create_dataset(name, data=data, **meta["create_kwargs"])
            _copy_attributes(obj.attrs, ds.attrs)
            continue
        if full_name == f"{scan_group_name}/EBSD/Header/nRows":
            meta = _dataset_create_kwargs(obj, shape=(1,))
            data = np.asarray([crop_spec.height], dtype=obj.dtype)
            ds = dst_group.create_dataset(name, data=data, **meta["create_kwargs"])
            _copy_attributes(obj.attrs, ds.attrs)
            continue
        cropped = _crop_dataset_array(
            obj,
            source_nx=source_nx,
            source_ny=source_ny,
            flat_indices=flat_indices,
            crop_spec=crop_spec,
        )
        if cropped is None:
            _copy_dataset_verbatim(src_group, dst_group, name)
            continue
        if name == "X Position":
            finite = np.isfinite(cropped)
            if np.any(finite):
                cropped = np.asarray(cropped, dtype=np.float32)
                cropped[finite] = cropped[finite] - float(np.min(cropped[finite]))
                cropped = cropped.astype(obj.dtype, copy=False)
        elif name == "Y Position":
            finite = np.isfinite(cropped)
            if np.any(finite):
                cropped = np.asarray(cropped, dtype=np.float32)
                cropped[finite] = cropped[finite] - float(np.min(cropped[finite]))
                cropped = cropped.astype(obj.dtype, copy=False)
        meta = _dataset_create_kwargs(obj, shape=tuple(int(v) for v in cropped.shape))
        ds = dst_group.create_dataset(name, data=cropped, **meta["create_kwargs"])
        _copy_attributes(obj.attrs, ds.attrs)


def _crop_dataset_array(ds: h5py.Dataset, *, source_nx: int, source_ny: int, flat_indices: np.ndarray, crop_spec: CropSpec) -> np.ndarray | None:
    shape = tuple(int(v) for v in ds.shape)
    if not shape:
        return None
    if len(shape) >= 2 and tuple(shape[:2]) == (int(source_ny), int(source_nx)):
        slices = (
            slice(crop_spec.row, crop_spec.row + crop_spec.height),
            slice(crop_spec.column, crop_spec.column + crop_spec.width),
        ) + tuple(slice(None) for _ in shape[2:])
        return np.asarray(ds[slices])
    if int(shape[0]) >= int(source_nx * source_ny):
        return np.asarray(ds[flat_indices])
    return None


def _rebase_position_dataset(data_group: h5py.Group, field_name: str) -> bool:
    if field_name not in data_group:
        return False
    ds = data_group[field_name]
    if not isinstance(ds, h5py.Dataset):
        return False
    attrs, create_kwargs = _snapshot_dataset(ds)
    arr = np.asarray(ds[()])
    if arr.size == 0:
        return False
    finite = np.isfinite(arr)
    if not np.any(finite):
        return False
    rebased = np.asarray(arr, dtype=np.float32)
    rebased[finite] = rebased[finite] - float(np.min(rebased[finite]))
    rebased = rebased.astype(ds.dtype, copy=False)
    _replace_dataset(data_group, field_name, rebased, attrs=attrs, create_kwargs=create_kwargs)
    return True


def export_cropped_oh5(
    *,
    source_path: Path,
    crop_spec: CropSpec,
    output_path: Path,
    repo_root: Path,
    logger: logging.Logger | None = None,
) -> CropExportResult:
    log = logger or logging.getLogger("oh5_crop")
    resolved_source = source_path.expanduser().resolve()
    resolved_output = output_path.expanduser().resolve()
    if not resolved_source.exists():
        raise FileNotFoundError(f".oh5 file not found: {resolved_source}")
    if resolved_output.exists():
        raise FileExistsError(f"Output .oh5 already exists: {resolved_output}")
    if resolved_output.suffix.lower() != ".oh5":
        raise ValueError(f"Output path must end with .oh5, got {resolved_output}")

    with Oh5ScanReader(resolved_source) as reader:
        if not reader.pattern_present or reader.pattern_key is None:
            raise ValueError(f"Pattern dataset missing in .oh5 file: {resolved_source}")
        validated = crop_spec.validate_for(nx=reader.nx, ny=reader.ny)
        iq_field_name = _find_required_iq_field(reader)
        flat_indices = validated.source_flat_indices(nx=reader.nx)
        if np.max(flat_indices) >= int(reader.total_pixels):
            raise ValueError(
                "Requested crop extends into scan cells that do not have pattern payload in this .oh5 file."
            )
        if reader.scan_group is None:
            raise RuntimeError(f"Could not resolve scan group for {resolved_source}")
        scan_name = str(reader.scan_group)
        original_nx = int(reader.nx)
        original_ny = int(reader.ny)
        pattern_key = str(reader.pattern_key)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    x_rebased = False
    y_rebased = False
    flat_indices = validated.source_flat_indices(nx=original_nx)

    with h5py.File(resolved_source, "r") as src_handle, h5py.File(resolved_output, "w") as dst_handle:
        _copy_attributes(src_handle.attrs, dst_handle.attrs)
        for name in src_handle.keys():
            obj = src_handle[name]
            if isinstance(obj, h5py.Group):
                new_group = dst_handle.create_group(name)
                _copy_attributes(obj.attrs, new_group.attrs)
                _copy_group_recursive(
                    obj,
                    new_group,
                    scan_group_name=scan_name,
                    source_nx=original_nx,
                    source_ny=original_ny,
                    crop_spec=validated,
                    flat_indices=flat_indices,
                )
            elif isinstance(obj, h5py.Dataset):
                _copy_dataset_verbatim(src_handle, dst_handle, name)
        x_rebased = True if f"{scan_name}/EBSD/Data/X Position" in dst_handle else False
        y_rebased = True if f"{scan_name}/EBSD/Data/Y Position" in dst_handle else False
    verification_report = build_crop_verification_report(
        source_path=resolved_source,
        cropped_path=resolved_output,
        scan_group_name=scan_name,
        source_nx=original_nx,
        source_ny=original_ny,
        crop_spec=validated,
    )

    manifest_path = resolved_output.with_name(f"{resolved_output.stem}_manifest.json")
    summary_payload = {
        "schema_version": "phase_id_xcorr.oh5_crop_export.v1",
        "workflow": "oh5_crop_export",
        "source_path": rel_path(resolved_source, repo_root),
        "output_path": rel_path(resolved_output, repo_root),
        "scan_name": scan_name,
        "iq_field_name": iq_field_name,
        "pattern_key": pattern_key,
        "crop": {
            "row": validated.row,
            "column": validated.column,
            "width": validated.width,
            "height": validated.height,
            "left": validated.left,
            "top": validated.top,
            "right": validated.right,
            "bottom": validated.bottom,
        },
        "original_grid": {"nx": original_nx, "ny": original_ny},
        "cropped_grid": {"nx": validated.width, "ny": validated.height},
        "comparison": {
            "source_path": rel_path(resolved_source, repo_root),
            "cropped_path": rel_path(resolved_output, repo_root),
            "crop_origin_row": validated.row,
            "crop_origin_column": validated.column,
            "crop_width": validated.width,
            "crop_height": validated.height,
            "original_nx": original_nx,
            "original_ny": original_ny,
            "cropped_nx": validated.width,
            "cropped_ny": validated.height,
            "scan_name": scan_name,
            "pattern_key": pattern_key,
        },
        "position_policy": {
            "x_position_rebased_to_zero": bool(x_rebased),
            "y_position_rebased_to_zero": bool(y_rebased),
        },
        "verification": {
            "source_dataset_count": verification_report.source_dataset_count,
            "cropped_dataset_count": verification_report.cropped_dataset_count,
            "source_group_count": verification_report.source_group_count,
            "cropped_group_count": verification_report.cropped_group_count,
            "changed_fields": [
                {
                    "path": item.path,
                    "kind": item.kind,
                    "source_summary": item.source_summary,
                    "cropped_summary": item.cropped_summary,
                    "note": item.note,
                }
                for item in verification_report.changed_fields
            ],
            "unchanged_fields": [
                {
                    "path": item.path,
                    "kind": item.kind,
                    "source_summary": item.source_summary,
                    "cropped_summary": item.cropped_summary,
                    "note": item.note,
                }
                for item in verification_report.unchanged_fields
            ],
        },
    }
    write_json(manifest_path, summary_payload)
    manifest_run_payload = build_run_manifest(
        repo_root=repo_root,
        packet_dir=resolved_source.parent,
        out_dir=resolved_output.parent,
        debug=False,
        extra={
            "workflow": "oh5_crop_export",
            "source_path": rel_path(resolved_source, repo_root),
            "output_path": rel_path(resolved_output, repo_root),
            "summary_payload": rel_path(manifest_path, repo_root),
        },
    )
    write_json(manifest_path.with_name(f"{manifest_path.stem}_run_manifest.json"), manifest_run_payload)
    log.info(
        "Exported cropped .oh5 file to %s from %s with crop row=%d col=%d width=%d height=%d",
        resolved_output,
        resolved_source,
        validated.row,
        validated.column,
        validated.width,
        validated.height,
    )
    return CropExportResult(
        source_path=resolved_source,
        output_path=resolved_output,
        manifest_path=manifest_path,
        crop_spec=validated,
        scan_name=scan_name,
        pattern_key=pattern_key,
        original_nx=original_nx,
        original_ny=original_ny,
        cropped_nx=validated.width,
        cropped_ny=validated.height,
        iq_field_name=iq_field_name,
        x_position_rebased=bool(x_rebased),
        y_position_rebased=bool(y_rebased),
        verification_report=verification_report,
    )


def inspect_scan_pixel(path: Path, *, x: int, y: int) -> PixelInspectionRecord:
    resolved = path.expanduser().resolve()
    with Oh5ScanReader(resolved) as reader:
        flat_index = reader.xy_to_flat(int(x), int(y))
        quality_row = reader.read_quality_row(flat_index=flat_index)
        scalar_values = reader.read_scalar_row_all(flat_index=flat_index)
        euler_row_deg = reader.read_euler_row(flat_index=flat_index, degrees=True) if reader.euler_present else None
        phase_info_by_id = _phase_info_from_header(reader.header_group)
        phase_id: int | None = None
        phase_name: str | None = None
        if "Phase" in scalar_values and scalar_values["Phase"] is not None:
            phase_id = int(float(scalar_values["Phase"]))
            if phase_id in phase_info_by_id:
                phase_name = phase_info_by_id[phase_id].material_name
        pattern = reader.read_pattern(flat_index=flat_index)
        return PixelInspectionRecord(
            path=resolved,
            x=int(x),
            y=int(y),
            flat_index=int(flat_index),
            quality_row=quality_row,
            scalar_values=scalar_values,
            euler_row_deg=euler_row_deg,
            phase_id=phase_id,
            phase_name=phase_name,
            pattern=pattern,
        )


def compare_cropped_pixel(
    *,
    source_path: Path,
    cropped_path: Path,
    crop_spec: CropSpec,
    local_x: int,
    local_y: int,
) -> tuple[PixelInspectionRecord, PixelInspectionRecord]:
    source_x, source_y = crop_to_source_coords(crop_spec, local_x=local_x, local_y=local_y)
    source_record = inspect_scan_pixel(source_path, x=source_x, y=source_y)
    cropped_record = inspect_scan_pixel(cropped_path, x=int(local_x), y=int(local_y))
    return source_record, cropped_record


def load_review_session(export_result: CropExportResult) -> CropReviewSession:
    return CropReviewSession(
        source=load_scan_visual_data(export_result.source_path),
        cropped=load_scan_visual_data(export_result.output_path),
        export=export_result,
        verification_report=export_result.verification_report,
    )
