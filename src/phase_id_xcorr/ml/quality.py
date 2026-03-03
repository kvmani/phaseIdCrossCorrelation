"""Quality-filtering policies for ML dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class QualityThresholds:
    """Configurable thresholds for sample acceptance."""

    confidence_index_min: float | None = None
    image_quality_min: float | None = None
    fit_max: float | None = None
    valid_required: bool = False


@dataclass(slots=True)
class QualityDecision:
    """Outcome of quality-policy evaluation."""

    accept: bool
    reasons: list[str]


def thresholds_from_config(payload: dict[str, Any] | None) -> QualityThresholds:
    """Build threshold dataclass from YAML mapping."""

    cfg = payload or {}
    return QualityThresholds(
        confidence_index_min=float(cfg["confidence_index_min"]) if cfg.get("confidence_index_min") is not None else None,
        image_quality_min=float(cfg["image_quality_min"]) if cfg.get("image_quality_min") is not None else None,
        fit_max=float(cfg["fit_max"]) if cfg.get("fit_max") is not None else None,
        valid_required=bool(cfg.get("valid_required", False)),
    )


def evaluate_quality(values: dict[str, float | bool | None], th: QualityThresholds) -> QualityDecision:
    """Evaluate one sample against thresholds."""

    reasons: list[str] = []

    ci = values.get("confidence_index")
    iq = values.get("image_quality")
    fit = values.get("fit")
    valid = values.get("valid")

    if th.confidence_index_min is not None:
        if ci is None:
            reasons.append("missing_confidence_index")
        elif float(ci) < th.confidence_index_min:
            reasons.append("ci_below_min")

    if th.image_quality_min is not None:
        if iq is None:
            reasons.append("missing_image_quality")
        elif float(iq) < th.image_quality_min:
            reasons.append("iq_below_min")

    if th.fit_max is not None:
        if fit is None:
            reasons.append("missing_fit")
        elif float(fit) > th.fit_max:
            reasons.append("fit_above_max")

    if th.valid_required:
        if valid is None:
            reasons.append("missing_valid")
        elif not bool(valid):
            reasons.append("valid_flag_false")

    return QualityDecision(accept=len(reasons) == 0, reasons=reasons)
