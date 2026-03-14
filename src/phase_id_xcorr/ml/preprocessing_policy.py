"""Dataset-stage preprocessing policy for ML pattern preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
from PIL import Image

from phase_id_xcorr.preprocessing import build_max_inscribed_circle_mask


@dataclass(slots=True)
class PreprocessingPolicy:
    """Resolved deterministic preprocessing policy."""

    resize_hw: tuple[int, int] | None = None
    apply_circular_mask: bool = False
    normalize_mode: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "resize_hw": list(self.resize_hw) if self.resize_hw else None,
            "apply_circular_mask": bool(self.apply_circular_mask),
            "normalize_mode": self.normalize_mode,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def resolve_preprocessing_policy(cfg: dict[str, Any]) -> PreprocessingPolicy:
    prep_cfg = cfg.get("preprocessing") if isinstance(cfg.get("preprocessing"), dict) else {}

    resize_raw = prep_cfg.get("resize_hw")
    if resize_raw is None and cfg.get("target_pattern_hw") is not None:
        resize_raw = cfg.get("target_pattern_hw")

    resize_hw: tuple[int, int] | None = None
    if resize_raw is not None:
        if not isinstance(resize_raw, (list, tuple)) or len(resize_raw) != 2:
            raise ValueError("preprocessing.resize_hw (or target_pattern_hw) must be [height, width]")
        resize_hw = (int(resize_raw[0]), int(resize_raw[1]))
        if resize_hw[0] <= 0 or resize_hw[1] <= 0:
            raise ValueError("preprocessing.resize_hw values must be positive")

    normalize_mode = str(prep_cfg.get("normalize_mode", "none")).strip().lower()
    if normalize_mode not in {"none", "per_pattern_minmax"}:
        raise ValueError("preprocessing.normalize_mode must be one of: none, per_pattern_minmax")

    return PreprocessingPolicy(
        resize_hw=resize_hw,
        apply_circular_mask=bool(prep_cfg.get("apply_circular_mask", False)),
        normalize_mode=normalize_mode,
    )


def _resize_pattern(pattern: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if tuple(pattern.shape) == (h, w):
        return pattern.astype(np.float32, copy=False)

    arr8 = np.clip(pattern, 0.0, 1.0)
    arr8 = (arr8 * 255.0).round().astype(np.uint8)
    im = Image.fromarray(arr8, mode="L")
    rs = im.resize((w, h), resample=Image.BILINEAR)
    out = np.asarray(rs, dtype=np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def _normalize_pattern(pattern: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return pattern.astype(np.float32, copy=False)
    if mode == "per_pattern_minmax":
        arr = pattern.astype(np.float32, copy=False)
        lo = float(np.min(arr)) if arr.size else 0.0
        hi = float(np.max(arr)) if arr.size else 0.0
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported normalize mode '{mode}'")


def apply_preprocessing(pattern: np.ndarray, policy: PreprocessingPolicy) -> np.ndarray:
    out = pattern.astype(np.float32, copy=False)
    if policy.resize_hw is not None:
        out = _resize_pattern(out, policy.resize_hw)
    if policy.apply_circular_mask:
        mask = build_max_inscribed_circle_mask(out.shape[0], out.shape[1])
        out = out.copy()
        out[~mask] = 0.0
    out = _normalize_pattern(out, policy.normalize_mode)
    return out
