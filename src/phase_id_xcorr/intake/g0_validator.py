"""G0 data-intake validator for student data packets.

This module validates the four packet JSON files, checks referenced files,
validates phase/candidate constraints, and optionally inspects .oh5 scan
consistency when scan files are present.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency
    h5py = None

ALLOWED_PHASES = {"fe_bcc", "fe3o4_magnetite", "feo_wustite"}
ALLOWED_INDEXING_STATUS = {"ok", "failed"}
EXPECTED_SCAN_KEYS = {"assume_fe_bcc", "assume_fe3o4_magnetite", "assume_feo_wustite"}
EXPECTED_ANGLE_KEYS = {"phi1", "PHI", "phi2"}


@dataclass(slots=True)
class Finding:
    """Single validation finding."""

    severity: str
    code: str
    message: str
    location: str


@dataclass(slots=True)
class ValidationResult:
    """Validation output for a packet."""

    packet_dir: str
    timestamp_utc: str
    findings: list[Finding]
    counts: dict[str, int]
    gate_status: str
    checked_files: dict[str, str]


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_repo_relative(path: Path, repo_root: Path | None) -> str:
    root = (repo_root or Path.cwd()).resolve()
    resolved = path.resolve()
    rel = os.path.relpath(resolved, root)
    return Path(rel).as_posix()


def _scan_group_name(h5f: Any) -> str:
    for key in h5f.keys():
        if key not in {"Manufacturer", "Version"}:
            return key
    raise ValueError("No scan group found (excluding Manufacturer/Version)")


def _read_scalar(dataset: Any) -> float:
    value = dataset[()]
    if hasattr(value, "shape") and value.shape == ():
        return float(value)
    try:
        return float(value.ravel()[0])
    except Exception:  # pragma: no cover - defensive
        return float(value[0])


def _read_oh5_grid_shape(path: Path) -> tuple[int, int, int, int]:
    if h5py is None:
        raise RuntimeError("h5py is required to validate .oh5 files")

    with h5py.File(path, "r") as h5f:
        scan = _scan_group_name(h5f)
        header = h5f[f"{scan}/EBSD/Header"]
        data = h5f[f"{scan}/EBSD/Data"]

        nx = int(_read_scalar(header["nColumns"]))
        ny = int(_read_scalar(header["nRows"]))

        if "Pattern Height" in header and "Pattern Width" in header:
            h = int(_read_scalar(header["Pattern Height"]))
            w = int(_read_scalar(header["Pattern Width"]))
        else:
            pattern = data["Pattern"]
            if pattern.ndim < 3:
                raise ValueError(f"Pattern dataset has unexpected shape: {pattern.shape}")
            h = int(pattern.shape[-2])
            w = int(pattern.shape[-1])

        return nx, ny, h, w


def _load_json(path: Path, findings: list[Finding], label: str) -> dict[str, Any] | None:
    if not path.exists():
        findings.append(
            Finding(
                severity="error",
                code="missing_json",
                message=f"Required JSON file not found: {path.name}",
                location=label,
            )
        )
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        findings.append(
            Finding(
                severity="error",
                code="invalid_json",
                message=f"Invalid JSON in {path.name}: {exc}",
                location=label,
            )
        )
        return None


def _check_required_keys(
    obj: dict[str, Any],
    required: set[str],
    findings: list[Finding],
    location: str,
) -> None:
    missing = sorted(required - set(obj.keys()))
    for key in missing:
        findings.append(
            Finding(
                severity="error",
                code="missing_key",
                message=f"Missing required key '{key}'",
                location=location,
            )
        )


def _check_relative_file(path_str: str, base_dir: Path, findings: list[Finding], location: str) -> None:
    file_path = base_dir / path_str
    if not file_path.exists():
        findings.append(
            Finding(
                severity="error",
                code="missing_file",
                message=f"Referenced file does not exist: {path_str}",
                location=location,
            )
        )


def _warn_placeholder_text(findings: list[Finding], text: str, location: str) -> None:
    if "replace with real entry" in text.lower():
        findings.append(
            Finding(
                severity="warning",
                code="placeholder_text",
                message="Placeholder text still present; replace with real notes.",
                location=location,
            )
        )


def validate_data_packet(
    packet_dir: Path,
    logger: logging.Logger | None = None,
    repo_root: Path | None = None,
) -> ValidationResult:
    """Validate packet files and return gate status/results.

    Parameters
    ----------
    packet_dir:
        Directory containing student packet files.
    logger:
        Optional logger.
    """

    log = logger or logging.getLogger(__name__)
    findings: list[Finding] = []

    packet_dir = packet_dir.resolve()
    display_packet_dir = _to_repo_relative(packet_dir, repo_root=repo_root)
    checked_files = {
        "experimental": "01_experimental_patterns_template.json",
        "simulated": "02_simulated_patterns_template.json",
        "scan": "03_scan_files_template.json",
        "processing": "04_processing_template.json",
    }

    exp_json = _load_json(packet_dir / checked_files["experimental"], findings, "experimental_json")
    sim_json = _load_json(packet_dir / checked_files["simulated"], findings, "simulated_json")
    scan_json = _load_json(packet_dir / checked_files["scan"], findings, "scan_json")
    proc_json = _load_json(packet_dir / checked_files["processing"], findings, "processing_json")

    experimental_by_id: dict[str, dict[str, Any]] = {}

    if exp_json is not None:
        _check_required_keys(exp_json, {"records"}, findings, "experimental_json")
        records = exp_json.get("records", [])
        if not isinstance(records, list) or len(records) == 0:
            findings.append(
                Finding(
                    severity="error",
                    code="empty_records",
                    message="Experimental records must be a non-empty list.",
                    location="experimental_json.records",
                )
            )
        else:
            for idx, record in enumerate(records):
                loc = f"experimental_json.records[{idx}]"
                if not isinstance(record, dict):
                    findings.append(
                        Finding(
                            severity="error",
                            code="invalid_record",
                            message="Record must be an object.",
                            location=loc,
                        )
                    )
                    continue

                _check_required_keys(
                    record,
                    {"record_id", "image_file", "true_phase", "orientation_angles_degrees", "image_info", "label_source"},
                    findings,
                    loc,
                )

                record_id = str(record.get("record_id", ""))
                if record_id:
                    if record_id in experimental_by_id:
                        findings.append(
                            Finding(
                                severity="error",
                                code="duplicate_record_id",
                                message=f"Duplicate experimental record_id '{record_id}'.",
                                location=loc,
                            )
                        )
                    experimental_by_id[record_id] = record

                true_phase = str(record.get("true_phase", ""))
                if true_phase not in ALLOWED_PHASES:
                    findings.append(
                        Finding(
                            severity="error",
                            code="invalid_phase",
                            message=f"Invalid true_phase '{true_phase}'.",
                            location=f"{loc}.true_phase",
                        )
                    )

                angles = record.get("orientation_angles_degrees", {})
                if not isinstance(angles, dict) or set(angles.keys()) != EXPECTED_ANGLE_KEYS:
                    findings.append(
                        Finding(
                            severity="error",
                            code="invalid_angles",
                            message="orientation_angles_degrees must contain keys phi1, PHI, phi2.",
                            location=f"{loc}.orientation_angles_degrees",
                        )
                    )

                image_file = str(record.get("image_file", ""))
                if image_file:
                    _check_relative_file(image_file, packet_dir, findings, f"{loc}.image_file")

                notes = str(record.get("notes", ""))
                if notes:
                    _warn_placeholder_text(findings, notes, f"{loc}.notes")

    if sim_json is not None:
        _check_required_keys(sim_json, {"required_assumed_phases", "records"}, findings, "simulated_json")

        required_assumed = sim_json.get("required_assumed_phases", [])
        required_set = set(required_assumed) if isinstance(required_assumed, list) else set()
        if required_set != EXPECTED_SCAN_KEYS:
            # JSON has phase names without assume_ prefix; keep this check tolerant and rely per-record checks.
            if set(required_assumed) != ALLOWED_PHASES:
                findings.append(
                    Finding(
                        severity="warning",
                        code="unexpected_required_assumed_phases",
                        message="required_assumed_phases differs from expected set.",
                        location="simulated_json.required_assumed_phases",
                    )
                )

        records = sim_json.get("records", [])
        if not isinstance(records, list) or len(records) == 0:
            findings.append(
                Finding(
                    severity="error",
                    code="empty_records",
                    message="Simulated records must be a non-empty list.",
                    location="simulated_json.records",
                )
            )
        else:
            for idx, record in enumerate(records):
                loc = f"simulated_json.records[{idx}]"
                if not isinstance(record, dict):
                    findings.append(
                        Finding(
                            severity="error",
                            code="invalid_record",
                            message="Record must be an object.",
                            location=loc,
                        )
                    )
                    continue

                _check_required_keys(
                    record,
                    {"record_id", "experimental_image", "true_phase", "simulated_candidates"},
                    findings,
                    loc,
                )

                record_id = str(record.get("record_id", ""))
                exp_image = str(record.get("experimental_image", ""))
                true_phase = str(record.get("true_phase", ""))

                if true_phase not in ALLOWED_PHASES:
                    findings.append(
                        Finding(
                            severity="error",
                            code="invalid_phase",
                            message=f"Invalid true_phase '{true_phase}'.",
                            location=f"{loc}.true_phase",
                        )
                    )

                if exp_image:
                    _check_relative_file(exp_image, packet_dir, findings, f"{loc}.experimental_image")

                if record_id in experimental_by_id:
                    exp_record = experimental_by_id[record_id]
                    if exp_record.get("image_file") != exp_image:
                        findings.append(
                            Finding(
                                severity="error",
                                code="cross_file_mismatch",
                                message="experimental_image does not match image_file for same record_id.",
                                location=loc,
                            )
                        )
                    if exp_record.get("true_phase") != true_phase:
                        findings.append(
                            Finding(
                                severity="error",
                                code="cross_file_mismatch",
                                message="true_phase mismatch between experimental and simulated files.",
                                location=loc,
                            )
                        )
                elif record_id:
                    findings.append(
                        Finding(
                            severity="error",
                            code="missing_experimental_record",
                            message=f"record_id '{record_id}' not found in experimental records.",
                            location=loc,
                        )
                    )

                candidates = record.get("simulated_candidates", [])
                if not isinstance(candidates, list):
                    findings.append(
                        Finding(
                            severity="error",
                            code="invalid_candidates",
                            message="simulated_candidates must be a list.",
                            location=f"{loc}.simulated_candidates",
                        )
                    )
                    continue

                if len(candidates) != 3:
                    findings.append(
                        Finding(
                            severity="error",
                            code="candidate_count_mismatch",
                            message=f"Expected exactly 3 simulated candidates, found {len(candidates)}.",
                            location=f"{loc}.simulated_candidates",
                        )
                    )

                seen_assumed: set[str] = set()
                for c_idx, cand in enumerate(candidates):
                    c_loc = f"{loc}.simulated_candidates[{c_idx}]"
                    if not isinstance(cand, dict):
                        findings.append(
                            Finding(
                                severity="error",
                                code="invalid_candidate",
                                message="Candidate must be an object.",
                                location=c_loc,
                            )
                        )
                        continue

                    _check_required_keys(
                        cand,
                        {
                            "assumed_phase",
                            "simulated_image",
                            "candidate_angles_degrees",
                            "indexing_status",
                            "is_fallback_orientation",
                        },
                        findings,
                        c_loc,
                    )

                    assumed_phase = str(cand.get("assumed_phase", ""))
                    if assumed_phase not in ALLOWED_PHASES:
                        findings.append(
                            Finding(
                                severity="error",
                                code="invalid_assumed_phase",
                                message=f"Invalid assumed_phase '{assumed_phase}'.",
                                location=f"{c_loc}.assumed_phase",
                            )
                        )
                    if assumed_phase in seen_assumed:
                        findings.append(
                            Finding(
                                severity="error",
                                code="duplicate_assumed_phase",
                                message=f"Duplicate assumed_phase '{assumed_phase}' in same record.",
                                location=f"{c_loc}.assumed_phase",
                            )
                        )
                    seen_assumed.add(assumed_phase)

                    sim_path = str(cand.get("simulated_image", ""))
                    if sim_path:
                        _check_relative_file(sim_path, packet_dir, findings, f"{c_loc}.simulated_image")

                    angles = cand.get("candidate_angles_degrees", {})
                    if not isinstance(angles, dict) or set(angles.keys()) != EXPECTED_ANGLE_KEYS:
                        findings.append(
                            Finding(
                                severity="error",
                                code="invalid_angles",
                                message="candidate_angles_degrees must contain keys phi1, PHI, phi2.",
                                location=f"{c_loc}.candidate_angles_degrees",
                            )
                        )

                    status = str(cand.get("indexing_status", ""))
                    if status not in ALLOWED_INDEXING_STATUS:
                        findings.append(
                            Finding(
                                severity="error",
                                code="invalid_indexing_status",
                                message=f"Invalid indexing_status '{status}'.",
                                location=f"{c_loc}.indexing_status",
                            )
                        )

                    fallback = bool(cand.get("is_fallback_orientation", False))
                    reason = str(cand.get("fallback_reason", ""))

                    if status == "failed":
                        if not fallback:
                            findings.append(
                                Finding(
                                    severity="error",
                                    code="fallback_flag_mismatch",
                                    message="Failed indexing must set is_fallback_orientation=true.",
                                    location=c_loc,
                                )
                            )
                        if reason.strip() == "":
                            findings.append(
                                Finding(
                                    severity="error",
                                    code="missing_fallback_reason",
                                    message="Failed indexing must include fallback_reason.",
                                    location=f"{c_loc}.fallback_reason",
                                )
                            )
                    elif fallback:
                        findings.append(
                            Finding(
                                severity="warning",
                                code="unexpected_fallback_flag",
                                message="is_fallback_orientation=true while indexing_status is 'ok'.",
                                location=c_loc,
                            )
                        )

                if seen_assumed and seen_assumed != ALLOWED_PHASES:
                    findings.append(
                        Finding(
                            severity="error",
                            code="assumed_phase_set_mismatch",
                            message=(
                                "Each record must contain one candidate for each phase: "
                                "fe_bcc, fe3o4_magnetite, feo_wustite."
                            ),
                            location=f"{loc}.simulated_candidates",
                        )
                    )

                notes = str(record.get("notes", ""))
                if notes:
                    _warn_placeholder_text(findings, notes, f"{loc}.notes")

    if scan_json is not None:
        _check_required_keys(scan_json, {"scan_records"}, findings, "scan_json")
        scan_records = scan_json.get("scan_records", [])
        if not isinstance(scan_records, list) or len(scan_records) == 0:
            findings.append(
                Finding(
                    severity="error",
                    code="empty_scan_records",
                    message="scan_records must be a non-empty list.",
                    location="scan_json.scan_records",
                )
            )
        else:
            for s_idx, scan in enumerate(scan_records):
                loc = f"scan_json.scan_records[{s_idx}]"
                if not isinstance(scan, dict):
                    findings.append(
                        Finding(
                            severity="error",
                            code="invalid_scan_record",
                            message="Scan record must be an object.",
                            location=loc,
                        )
                    )
                    continue

                _check_required_keys(scan, {"scan_id", "scan_files", "grid_info"}, findings, loc)
                scan_files = scan.get("scan_files", {})
                grid_info = scan.get("grid_info", {})

                if not isinstance(scan_files, dict):
                    findings.append(
                        Finding(
                            severity="error",
                            code="invalid_scan_files",
                            message="scan_files must be an object.",
                            location=f"{loc}.scan_files",
                        )
                    )
                    continue

                missing_scan_keys = EXPECTED_SCAN_KEYS - set(scan_files.keys())
                for key in sorted(missing_scan_keys):
                    findings.append(
                        Finding(
                            severity="error",
                            code="missing_scan_file_key",
                            message=f"Missing scan_files key '{key}'.",
                            location=f"{loc}.scan_files",
                        )
                    )

                triad_paths: dict[str, Path] = {}
                for key in EXPECTED_SCAN_KEYS:
                    rel = scan_files.get(key)
                    if isinstance(rel, str) and rel:
                        _check_relative_file(rel, packet_dir, findings, f"{loc}.scan_files.{key}")
                        triad_paths[key] = packet_dir / rel

                # Basic point checks
                points = scan.get("manual_check_points", [])
                if isinstance(points, list) and len(points) < 10:
                    findings.append(
                        Finding(
                            severity="warning",
                            code="few_manual_points",
                            message=(
                                f"manual_check_points has {len(points)} points; "
                                "recommend at least 10 for baseline validation."
                            ),
                            location=f"{loc}.manual_check_points",
                        )
                    )

                # Only attempt OH5 consistency checks when all triad files exist
                missing_triads = [p for p in triad_paths.values() if not p.exists()]
                if len(triad_paths) == 3 and not missing_triads:
                    shapes: dict[str, tuple[int, int, int, int]] = {}
                    for key, path in triad_paths.items():
                        try:
                            shapes[key] = _read_oh5_grid_shape(path)
                        except Exception as exc:
                            findings.append(
                                Finding(
                                    severity="error",
                                    code="oh5_read_error",
                                    message=f"Failed reading {key}: {exc}",
                                    location=f"{loc}.scan_files.{key}",
                                )
                            )

                    if len(shapes) == 3:
                        uniq = {v for v in shapes.values()}
                        if len(uniq) != 1:
                            findings.append(
                                Finding(
                                    severity="error",
                                    code="oh5_shape_mismatch",
                                    message=f"Triad .oh5 shapes mismatch: {shapes}",
                                    location=f"{loc}.scan_files",
                                )
                            )

                        if isinstance(grid_info, dict):
                            required_grid = {"nx", "ny", "pattern_height", "pattern_width"}
                            _check_required_keys(grid_info, required_grid, findings, f"{loc}.grid_info")
                            try:
                                declared = (
                                    int(grid_info.get("nx")),
                                    int(grid_info.get("ny")),
                                    int(grid_info.get("pattern_height")),
                                    int(grid_info.get("pattern_width")),
                                )
                                common = next(iter(uniq)) if uniq else None
                                if common is not None and declared != common:
                                    findings.append(
                                        Finding(
                                            severity="error",
                                            code="declared_grid_mismatch",
                                            message=(
                                                f"Declared grid_info {declared} does not match "
                                                f".oh5 observed {common}."
                                            ),
                                            location=f"{loc}.grid_info",
                                        )
                                    )
                            except Exception:
                                findings.append(
                                    Finding(
                                        severity="error",
                                        code="invalid_grid_info",
                                        message="grid_info nx/ny/pattern_height/pattern_width must be integers.",
                                        location=f"{loc}.grid_info",
                                    )
                                )

                # Validate manual point phase labels and bounds when possible
                if isinstance(points, list):
                    nx = grid_info.get("nx") if isinstance(grid_info, dict) else None
                    ny = grid_info.get("ny") if isinstance(grid_info, dict) else None
                    for p_idx, point in enumerate(points):
                        p_loc = f"{loc}.manual_check_points[{p_idx}]"
                        if not isinstance(point, dict):
                            findings.append(
                                Finding(
                                    severity="error",
                                    code="invalid_manual_point",
                                    message="Manual check point must be an object.",
                                    location=p_loc,
                                )
                            )
                            continue
                        _check_required_keys(point, {"x", "y", "expected_phase"}, findings, p_loc)

                        expected_phase = str(point.get("expected_phase", ""))
                        if expected_phase not in ALLOWED_PHASES:
                            findings.append(
                                Finding(
                                    severity="error",
                                    code="invalid_phase",
                                    message=f"Invalid expected_phase '{expected_phase}'.",
                                    location=f"{p_loc}.expected_phase",
                                )
                            )

                        try:
                            x_val = int(point.get("x"))
                            y_val = int(point.get("y"))
                            if isinstance(nx, int) and x_val >= nx:
                                findings.append(
                                    Finding(
                                        severity="error",
                                        code="point_out_of_bounds",
                                        message=f"x={x_val} is outside nx={nx}.",
                                        location=f"{p_loc}.x",
                                    )
                                )
                            if isinstance(ny, int) and y_val >= ny:
                                findings.append(
                                    Finding(
                                        severity="error",
                                        code="point_out_of_bounds",
                                        message=f"y={y_val} is outside ny={ny}.",
                                        location=f"{p_loc}.y",
                                    )
                                )
                        except Exception:
                            findings.append(
                                Finding(
                                    severity="error",
                                    code="invalid_manual_point",
                                    message="x and y must be integers.",
                                    location=p_loc,
                                )
                            )

    if proc_json is not None:
        _check_required_keys(proc_json, {"settings"}, findings, "processing_json")
        settings = proc_json.get("settings", {})
        if not isinstance(settings, dict):
            findings.append(
                Finding(
                    severity="error",
                    code="invalid_settings",
                    message="settings must be an object.",
                    location="processing_json.settings",
                )
            )
        else:
            required_settings = {
                "dtype_target",
                "normalization_method",
                "mask_method",
                "mask_parameters",
                "resize_policy",
                "intensity_clip_policy",
                "ncc_variant",
                "euler_convention",
                "angle_units",
                "exp_sim_alignment_policy",
            }
            _check_required_keys(settings, required_settings, findings, "processing_json.settings")

    # Cross-file completeness: every experimental record should have simulated record
    if experimental_by_id and sim_json is not None and isinstance(sim_json.get("records"), list):
        simulated_ids = {str(r.get("record_id", "")) for r in sim_json["records"] if isinstance(r, dict)}
        for record_id in sorted(experimental_by_id.keys() - simulated_ids):
            findings.append(
                Finding(
                    severity="error",
                    code="missing_simulated_record",
                    message=f"Experimental record_id '{record_id}' missing in simulated records.",
                    location="simulated_json.records",
                )
            )

    error_count = sum(1 for f in findings if f.severity == "error")
    warning_count = sum(1 for f in findings if f.severity == "warning")
    info_count = sum(1 for f in findings if f.severity == "info")

    gate_status = "GO" if error_count == 0 else "HOLD"
    if error_count > 0:
        log.warning("G0 validation completed with %d errors and %d warnings", error_count, warning_count)
    else:
        log.info("G0 validation completed with no errors (%d warnings)", warning_count)

    return ValidationResult(
        packet_dir=display_packet_dir,
        timestamp_utc=_now_iso_utc(),
        findings=findings,
        counts={
            "errors": error_count,
            "warnings": warning_count,
            "info": info_count,
            "total": len(findings),
        },
        gate_status=gate_status,
        checked_files=checked_files,
    )


def _render_markdown_report(result: ValidationResult) -> str:
    lines: list[str] = []
    lines.append("# G0 Data Intake Validation Report")
    lines.append("")
    lines.append(f"- Packet directory: `{result.packet_dir}`")
    lines.append(f"- Timestamp (UTC): `{result.timestamp_utc}`")
    lines.append(f"- Gate status: **{result.gate_status}**")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Errors: **{result.counts['errors']}**")
    lines.append(f"- Warnings: **{result.counts['warnings']}**")
    lines.append(f"- Total findings: **{result.counts['total']}**")
    lines.append("")

    lines.append("## Checked Files")
    lines.append("")
    for key, name in result.checked_files.items():
        lines.append(f"- `{key}`: `{name}`")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    if not result.findings:
        lines.append("No findings. Data intake gate is ready to proceed.")
    else:
        lines.append("| # | Severity | Code | Location | Message |")
        lines.append("|---|---|---|---|---|")
        for idx, finding in enumerate(result.findings, start=1):
            msg = finding.message.replace("|", "\\|")
            loc = finding.location.replace("|", "\\|")
            lines.append(f"| {idx} | {finding.severity} | {finding.code} | `{loc}` | {msg} |")

    lines.append("")
    lines.append("## Gate Decision Guidance")
    lines.append("")
    if result.gate_status == "GO":
        lines.append("Proceed to G1 implementation.")
    else:
        lines.append("Hold progression. Resolve all `error` findings, then re-run G0 validation.")

    return "\n".join(lines) + "\n"


def write_g0_reports(result: ValidationResult, out_dir: Path) -> tuple[Path, Path]:
    """Write markdown and JSON reports for the G0 validator."""

    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / "data_intake_validation.md"
    json_path = out_dir / "data_intake_manifest.json"

    md_path.write_text(_render_markdown_report(result), encoding="utf-8")

    payload = {
        "packet_dir": result.packet_dir,
        "timestamp_utc": result.timestamp_utc,
        "gate_status": result.gate_status,
        "counts": result.counts,
        "checked_files": result.checked_files,
        "findings": [asdict(f) for f in result.findings],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return md_path, json_path
