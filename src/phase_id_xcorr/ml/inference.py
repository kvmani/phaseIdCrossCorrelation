"""Inference utilities for trained ML phase-classifier models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image
import torch

from .config import resolve_path
from .dataset_io import read_json
from .models import build_model
from .preprocessing_policy import PreprocessingPolicy, apply_preprocessing
from .training import _resolve_device


SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(slots=True)
class LoadedModel:
    run_dir: Path
    report_path: Path
    checkpoint_path: Path
    dataset_manifest_path: Path
    class_names: list[str]
    preprocessing_policy: PreprocessingPolicy
    input_mean: float
    input_std: float
    device: torch.device
    model: torch.nn.Module
    model_family: str
    model_name: str


@dataclass(slots=True)
class InferenceResult:
    image_path: Path
    predicted_phase: str
    predicted_index: int
    probabilities: dict[str, float]
    confidence: float
    original_image: np.ndarray
    preprocessed_image: np.ndarray


@dataclass(slots=True)
class PatternInferenceResult:
    """Prediction result for one in-memory grayscale pattern."""

    predicted_phase: str
    predicted_index: int
    probabilities: dict[str, float]
    confidence: float
    preprocessed_image: np.ndarray


def _image_to_float01(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im)

    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            arr = arr[..., :3].astype(np.float32)
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        else:
            arr = arr[..., 0]

    if np.issubdtype(arr.dtype, np.integer):
        max_v = int(np.iinfo(arr.dtype).max)
        return np.clip(arr.astype(np.float32) / float(max(1, max_v)), 0.0, 1.0)

    arr = arr.astype(np.float32)
    lo = float(np.min(arr)) if arr.size else 0.0
    hi = float(np.max(arr)) if arr.size else 0.0
    if 0.0 <= lo <= hi <= 1.0:
        return np.clip(arr, 0.0, 1.0)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _class_names_from_manifest(manifest: dict[str, Any]) -> list[str]:
    phase_to_label = manifest.get("phase_to_label", {})
    if not isinstance(phase_to_label, dict) or not phase_to_label:
        raise ValueError("dataset manifest missing phase_to_label")
    pairs = sorted(((int(label), str(phase)) for phase, label in phase_to_label.items()), key=lambda kv: kv[0])
    return [phase for _, phase in pairs]


def _first_float(value: Any, default: float) -> float:
    if isinstance(value, (list, tuple)):
        if not value:
            return float(default)
        return float(value[0])
    if value is None:
        return float(default)
    return float(value)


def _model_cfg_from_report(report: dict[str, Any]) -> dict[str, Any]:
    model_meta = report.get("model", {}) if isinstance(report.get("model"), dict) else {}
    family = str(model_meta.get("family", "")).strip().lower()
    model_name = str(model_meta.get("model_name", "")).strip()
    in_chans = int(model_meta.get("in_chans", 1))
    if family == "simple_cnn":
        match = re.fullmatch(r"simple_cnn_w(\d+)", model_name)
        if match is None:
            raise ValueError(f"Could not infer simple_cnn width from model_name='{model_name}'")
        return {
            "family": "simple_cnn",
            "width": int(match.group(1)),
            "in_chans": in_chans,
        }
    if family == "timm":
        if not model_name:
            raise ValueError("report.model.model_name missing for timm run")
        return {
            "family": "timm",
            "model_name": model_name,
            "pretrained": False,
            "in_chans": in_chans,
        }
    raise ValueError(f"Unsupported model family '{family}' in report")


def _preprocessing_policy_from_report(report: dict[str, Any]) -> PreprocessingPolicy:
    input_meta = report.get("input", {}) if isinstance(report.get("input"), dict) else {}
    prep_meta = (
        input_meta.get("dataset_preprocessing_policy")
        if isinstance(input_meta.get("dataset_preprocessing_policy"), dict)
        else {}
    )
    resize_hw_raw = prep_meta.get("resize_hw")
    resize_hw = None
    if isinstance(resize_hw_raw, (list, tuple)) and len(resize_hw_raw) == 2:
        resize_hw = (int(resize_hw_raw[0]), int(resize_hw_raw[1]))
    return PreprocessingPolicy(
        resize_hw=resize_hw,
        apply_circular_mask=bool(prep_meta.get("apply_circular_mask", False)),
        normalize_mode=str(prep_meta.get("normalize_mode", "none")).strip().lower() or "none",
    )


def list_model_runs(root: Path) -> list[Path]:
    """Return available model run directories under a suite root or single run root."""

    root = root.resolve()
    if (root / "report.json").exists():
        return [root]
    out: list[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "report.json").exists():
            out.append(path)
    return out


def load_trained_model(
    *,
    run_dir: Path,
    repo_root: Path,
    checkpoint_name: str = "best_checkpoint.pt",
    device: str = "auto",
) -> LoadedModel:
    """Load one trained model bundle from a run directory."""

    run_dir = run_dir.resolve()
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"report.json not found in run_dir: {run_dir}")
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_name} not found in run_dir: {run_dir}")

    report = read_json(report_path)
    dataset_manifest_rel = str(report.get("dataset_manifest_path", "")).strip()
    if not dataset_manifest_rel:
        raise ValueError("report.json missing dataset_manifest_path")
    dataset_manifest_path = resolve_path(dataset_manifest_rel, base_dir=run_dir, repo_root=repo_root)
    dataset_manifest = read_json(dataset_manifest_path)
    class_names = _class_names_from_manifest(dataset_manifest)

    model_cfg = _model_cfg_from_report(report)
    build = build_model(model_cfg, num_classes=len(class_names), in_chans=int(model_cfg.get("in_chans", 1)))
    torch_device = _resolve_device(device)
    model = build.model.to(torch_device)

    ckpt = torch.load(checkpoint_path, map_location=torch_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    input_meta = report.get("input", {}) if isinstance(report.get("input"), dict) else {}
    normalize_meta = input_meta.get("normalize", {}) if isinstance(input_meta.get("normalize"), dict) else {}
    mean = _first_float(normalize_meta.get("mean"), 0.0)
    std = _first_float(normalize_meta.get("std"), 1.0) or 1.0

    return LoadedModel(
        run_dir=run_dir,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        dataset_manifest_path=dataset_manifest_path,
        class_names=class_names,
        preprocessing_policy=_preprocessing_policy_from_report(report),
        input_mean=mean,
        input_std=std,
        device=torch_device,
        model=model,
        model_family=build.family,
        model_name=build.model_name,
    )


def predict_image(*, loaded: LoadedModel, image_path: Path) -> InferenceResult:
    """Predict phase identity for one unknown image file."""

    image_path = image_path.resolve()
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image type '{image_path.suffix}'. Supported: {sorted(SUPPORTED_IMAGE_SUFFIXES)}")

    original = _image_to_float01(image_path)
    pattern_result = predict_pattern_array(loaded=loaded, pattern=original)
    return InferenceResult(
        image_path=image_path,
        predicted_phase=pattern_result.predicted_phase,
        predicted_index=pattern_result.predicted_index,
        probabilities=pattern_result.probabilities,
        confidence=pattern_result.confidence,
        original_image=original,
        preprocessed_image=pattern_result.preprocessed_image,
    )


def predict_pattern_array(*, loaded: LoadedModel, pattern: np.ndarray) -> PatternInferenceResult:
    """Predict phase identity for one in-memory grayscale pattern."""

    processed = apply_preprocessing(np.asarray(pattern, dtype=np.float32), loaded.preprocessing_policy)
    tensor = torch.from_numpy(processed.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - loaded.input_mean) / loaded.input_std
    tensor = tensor.to(loaded.device)

    with torch.no_grad():
        logits = loaded.model(tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    predicted_index = int(np.argmax(probs))
    class_names = loaded.class_names
    probabilities = {class_names[idx]: float(probs[idx]) for idx in range(len(class_names))}
    return PatternInferenceResult(
        predicted_phase=class_names[predicted_index],
        predicted_index=predicted_index,
        probabilities=probabilities,
        confidence=float(probs[predicted_index]),
        preprocessed_image=processed,
    )
