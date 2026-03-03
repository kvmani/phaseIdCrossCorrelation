"""Classification metrics for ML workflows."""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Build integer confusion matrix."""

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(np.int64), y_pred.astype(np.int64), strict=False):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else 0.0


def metrics_from_confusion(cm: np.ndarray, class_names: list[str] | None = None) -> dict[str, Any]:
    """Compute scalar and per-class metrics from confusion matrix."""

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got {cm.shape}")

    n_classes = int(cm.shape[0])
    support = cm.sum(axis=1).astype(np.int64)
    pred_support = cm.sum(axis=0).astype(np.int64)
    total = int(cm.sum())
    correct = int(np.trace(cm))

    per_class: dict[str, dict[str, float | int]] = {}
    precision_vals: list[float] = []
    recall_vals: list[float] = []
    f1_vals: list[float] = []

    for idx in range(n_classes):
        tp = float(cm[idx, idx])
        fp = float(pred_support[idx] - cm[idx, idx])
        fn = float(support[idx] - cm[idx, idx])

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)

        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)

        key = class_names[idx] if class_names and idx < len(class_names) else str(idx)
        per_class[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[idx]),
            "pred_support": int(pred_support[idx]),
        }

    out = {
        "accuracy": _safe_div(float(correct), float(total)),
        "macro_precision": float(np.mean(precision_vals)) if precision_vals else 0.0,
        "macro_recall": float(np.mean(recall_vals)) if recall_vals else 0.0,
        "macro_f1": float(np.mean(f1_vals)) if f1_vals else 0.0,
        "support_total": total,
        "correct_total": correct,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    return out


def classification_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute metrics directly from prediction arrays."""

    cm = confusion_matrix(num_classes=num_classes, y_true=y_true, y_pred=y_pred)
    return metrics_from_confusion(cm, class_names=class_names)
