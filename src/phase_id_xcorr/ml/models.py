"""Model factory for ML phase classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

try:
    import timm
except Exception:  # pragma: no cover
    timm = None


class SimpleCnnClassifier(nn.Module):
    """Small fallback CNN for debug/smoke runs."""

    def __init__(self, num_classes: int, in_chans: int = 1, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(width * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)


@dataclass(slots=True)
class ModelBuildResult:
    """Constructed model and resolved metadata."""

    model: nn.Module
    family: str
    model_name: str
    pretrained: bool


def build_model(model_cfg: dict[str, Any], *, num_classes: int, in_chans: int = 1) -> ModelBuildResult:
    """Build classifier model from config."""

    family = str(model_cfg.get("family", "timm")).strip().lower()

    if family == "simple_cnn":
        width = int(model_cfg.get("width", 32))
        model = SimpleCnnClassifier(num_classes=num_classes, in_chans=in_chans, width=width)
        return ModelBuildResult(
            model=model,
            family="simple_cnn",
            model_name=f"simple_cnn_w{width}",
            pretrained=False,
        )

    if family != "timm":
        raise ValueError(f"Unsupported model family '{family}'. Expected 'timm' or 'simple_cnn'")

    if timm is None:
        raise RuntimeError("timm is required for model family='timm'")

    model_name = str(model_cfg.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("model.model_name is required for timm models")

    pretrained = bool(model_cfg.get("pretrained", True))
    drop_rate = float(model_cfg.get("drop_rate", 0.0))
    drop_path_rate = float(model_cfg.get("drop_path_rate", 0.0))

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    return ModelBuildResult(
        model=model,
        family="timm",
        model_name=model_name,
        pretrained=pretrained,
    )
