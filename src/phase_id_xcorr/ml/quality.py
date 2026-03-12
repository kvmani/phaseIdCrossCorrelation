"""Quality-filtering policies for ML dataset preparation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


ALIASES: dict[str, str] = {
    "CI": "confidence_index",
    "Confidence Index": "confidence_index",
    "IQ": "image_quality",
    "Image Quality": "image_quality",
    "Fit": "fit",
    "Valid": "valid",
}


@dataclass(slots=True)
class QualityThresholds:
    """Configurable thresholds for sample acceptance."""

    confidence_index_min: float | None = None
    image_quality_min: float | None = None
    fit_max: float | None = None
    valid_required: bool = False


@dataclass(slots=True)
class QualityPolicy:
    """Resolved quality policy, including optional expression."""

    thresholds: QualityThresholds
    expression: str | None
    resolved_expression: str | None
    field_aliases: dict[str, str]


@dataclass(slots=True)
class QualityDecision:
    """Outcome of quality-policy evaluation."""

    accept: bool
    reasons: list[str]


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)
_ALLOWED_UNARY = (ast.UAdd, ast.USub, ast.Not)
_ALLOWED_CMPOPS = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)
_ALLOWED_BOOLOPS = (ast.And, ast.Or)
_ALLOWED_CONST = (int, float, bool)


def _normalize_alias_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _build_alias_map(extra_aliases: dict[str, str] | None = None) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for alias, canonical in ALIASES.items():
        alias_map[_normalize_alias_key(alias)] = canonical
    for canonical in ("confidence_index", "image_quality", "fit", "valid"):
        alias_map[_normalize_alias_key(canonical)] = canonical
    for alias, canonical in (extra_aliases or {}).items():
        alias_map[_normalize_alias_key(str(alias))] = str(canonical)
    return alias_map


def _resolve_expression(expression: str, alias_map: dict[str, str]) -> str:
    expr = expression.replace("&&", " and ").replace("||", " or ")
    expr = expr.replace("!", " not ") if "!=" not in expr else expr

    class _AliasTransformer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            canonical = alias_map.get(_normalize_alias_key(node.id))
            if canonical is None:
                raise ValueError(f"Unknown quality expression field '{node.id}'")
            return ast.copy_location(ast.Name(id=canonical, ctx=node.ctx), node)

    tree = ast.parse(expr, mode="eval")
    tree = _AliasTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _validate_safe_expr(expression: str) -> ast.Expression:
    tree = ast.parse(expression, mode="eval")

    def _walk(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            _walk(node.body)
            return
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, _ALLOWED_BOOLOPS):
                raise ValueError("Disallowed boolean operator in quality expression")
            for v in node.values:
                _walk(v)
            return
        if isinstance(node, ast.Compare):
            _walk(node.left)
            for op in node.ops:
                if not isinstance(op, _ALLOWED_CMPOPS):
                    raise ValueError("Disallowed comparison operator in quality expression")
            for c in node.comparators:
                _walk(c)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise ValueError("Disallowed arithmetic operator in quality expression")
            _walk(node.left)
            _walk(node.right)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY):
                raise ValueError("Disallowed unary operator in quality expression")
            _walk(node.operand)
            return
        if isinstance(node, ast.Name):
            if node.id not in {"confidence_index", "image_quality", "fit", "valid"}:
                raise ValueError(f"Unknown field '{node.id}' in quality expression")
            return
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, _ALLOWED_CONST):
                raise ValueError("Disallowed constant type in quality expression")
            return
        raise ValueError(f"Unsupported syntax in quality expression: {type(node).__name__}")

    _walk(tree)
    return tree


def thresholds_from_config(payload: dict[str, Any] | None) -> QualityThresholds:
    cfg = payload or {}
    return QualityThresholds(
        confidence_index_min=float(cfg["confidence_index_min"]) if cfg.get("confidence_index_min") is not None else None,
        image_quality_min=float(cfg["image_quality_min"]) if cfg.get("image_quality_min") is not None else None,
        fit_max=float(cfg["fit_max"]) if cfg.get("fit_max") is not None else None,
        valid_required=bool(cfg.get("valid_required", False)),
    )


def quality_policy_from_config(payload: dict[str, Any] | None) -> QualityPolicy:
    cfg = payload or {}
    th = thresholds_from_config(cfg)
    expression = str(cfg.get("expression", "")).strip() or None
    alias_map = _build_alias_map(cfg.get("aliases") if isinstance(cfg.get("aliases"), dict) else None)
    resolved_expression = _resolve_expression(expression, alias_map) if expression else None
    if resolved_expression:
        _validate_safe_expr(resolved_expression)
    return QualityPolicy(
        thresholds=th,
        expression=expression,
        resolved_expression=resolved_expression,
        field_aliases=alias_map,
    )


def _eval_expression(tree: ast.Expression, values: dict[str, float | bool | None]) -> bool:
    safe_values = {k: (False if v is None and k == "valid" else v) for k, v in values.items()}
    if any(v is None for v in safe_values.values()):
        return False
    result = eval(compile(tree, "<quality_expr>", "eval"), {"__builtins__": {}}, safe_values)
    return bool(result)


def evaluate_quality(values: dict[str, float | bool | None], policy: QualityPolicy | QualityThresholds) -> QualityDecision:
    if isinstance(policy, QualityThresholds):
        policy = QualityPolicy(
            thresholds=policy,
            expression=None,
            resolved_expression=None,
            field_aliases=_build_alias_map(),
        )

    th = policy.thresholds
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

    if policy.resolved_expression:
        tree = _validate_safe_expr(policy.resolved_expression)
        if not _eval_expression(tree, values):
            reasons.append("quality_expression_false")

    return QualityDecision(accept=len(reasons) == 0, reasons=reasons)
