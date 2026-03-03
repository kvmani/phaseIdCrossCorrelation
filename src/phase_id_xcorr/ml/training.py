"""Training workflow for phase-classification models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from phase_id_xcorr.preprocessing import build_max_inscribed_circle_mask

from .config import load_yaml, resolve_path
from .dataset_io import load_split_npz, read_json, rel_path, write_json
from .metrics import classification_metrics
from .models import build_model


@dataclass(slots=True)
class TrainResult:
    """Paths to generated training artifacts."""

    out_dir: Path
    report_path: Path
    best_checkpoint: Path | None
    last_checkpoint: Path


class ArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple tensor dataset over preprocessed numpy arrays."""

    def __init__(
        self,
        patterns: np.ndarray,
        labels: np.ndarray,
        *,
        mean: float,
        std: float,
    ):
        self.patterns = patterns.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        arr = self.patterns[idx]
        x = torch.from_numpy(arr).unsqueeze(0)
        x = (x - self.mean) / self.std
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


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


def _prepare_patterns(
    patterns: np.ndarray,
    *,
    resize_hw: tuple[int, int] | None,
    apply_circular_mask: bool,
) -> np.ndarray:
    out = patterns.astype(np.float32, copy=False)

    if resize_hw is not None:
        out = np.stack([_resize_pattern(p, resize_hw) for p in out], axis=0).astype(np.float32, copy=False)

    if apply_circular_mask and out.size:
        mask = build_max_inscribed_circle_mask(out.shape[1], out.shape[2])
        out = out.copy()
        out[:, ~mask] = 0.0

    return out


def _limit_samples(patterns: np.ndarray, labels: np.ndarray, max_samples: int | None) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0:
        return patterns, labels
    n = min(int(max_samples), int(labels.shape[0]))
    return patterns[:n], labels[:n]


def _resolve_device(name: str) -> torch.device:
    text = name.strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if text == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device value '{name}' (expected auto/cpu/cuda)")


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    use_amp = bool(amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device="cuda", enabled=(use_amp and is_train))
    else:  # pragma: no cover - compatibility fallback
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and is_train))

    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)
        else:  # pragma: no cover - compatibility fallback
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

        with autocast_ctx:
            logits = model(x)
            loss = criterion(logits, y)

        if is_train and optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        losses.append(float(loss.detach().cpu().item()))
        preds.append(torch.argmax(logits.detach(), dim=1).cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.int64)
    true_arr = np.concatenate(targets, axis=0) if targets else np.zeros((0,), dtype=np.int64)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return mean_loss, true_arr, pred_arr


def _phase_names_from_manifest(manifest: dict[str, Any]) -> list[str]:
    phase_to_label = manifest.get("phase_to_label", {})
    if not isinstance(phase_to_label, dict):
        raise ValueError("dataset manifest missing phase_to_label mapping")

    pairs = sorted(((int(v), str(k)) for k, v in phase_to_label.items()), key=lambda kv: kv[0])
    if not pairs:
        raise ValueError("dataset manifest phase_to_label is empty")
    return [name for _, name in pairs]


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=False) + "\n")


def _safe_eta_seconds(*, completed: int, total: int, elapsed: float) -> float | None:
    if completed <= 0 or total <= completed or elapsed <= 0:
        return None
    rate = completed / elapsed
    if rate <= 0:
        return None
    return float((total - completed) / rate)


def train_classifier(
    *,
    config_path: Path,
    repo_root: Path,
    debug: bool,
    logger: logging.Logger | None = None,
) -> TrainResult:
    """Train a classifier from prepared dataset artifacts."""

    log = logger or logging.getLogger(__name__)
    cfg_path = config_path.resolve()
    cfg_dir = cfg_path.parent
    cfg = load_yaml(cfg_path)

    dataset_manifest_path = resolve_path(
        cfg.get("dataset_manifest_path", ""),
        base_dir=cfg_dir,
        repo_root=repo_root,
    )
    if not dataset_manifest_path.exists():
        raise FileNotFoundError(f"dataset_manifest_path not found: {dataset_manifest_path}")

    dataset_manifest = read_json(dataset_manifest_path)
    out_dir = resolve_path(
        cfg.get("output_dir", "reports/ml/runs/default"),
        base_dir=cfg_dir,
        repo_root=repo_root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    event_log = out_dir / "events.jsonl"
    event_log.write_text("", encoding="utf-8")
    run_t0 = time.monotonic()

    def emit(event: str, **fields: Any) -> None:
        payload = {
            "timestamp_utc": _now_iso_utc(),
            "elapsed_seconds": float(time.monotonic() - run_t0),
            "event": event,
        }
        payload.update(fields)
        _append_event(event_log, payload)

    emit(
        "RUN_START",
        config_path=rel_path(cfg_path, repo_root),
        dataset_manifest_path=rel_path(dataset_manifest_path, repo_root),
        output_dir=rel_path(out_dir, repo_root),
        debug=bool(debug),
    )

    class_names = _phase_names_from_manifest(dataset_manifest)
    num_classes = len(class_names)

    artifacts = dataset_manifest.get("artifacts", {})
    train_npz = resolve_path(artifacts.get("train_npz", ""), base_dir=repo_root, repo_root=repo_root)
    val_npz = resolve_path(artifacts.get("val_npz", ""), base_dir=repo_root, repo_root=repo_root)
    test_npz = resolve_path(artifacts.get("test_npz", ""), base_dir=repo_root, repo_root=repo_root)

    train_data = load_split_npz(train_npz)
    val_data = load_split_npz(val_npz)
    test_data = load_split_npz(test_npz)
    emit(
        "SPLITS_LOADED",
        train_count=int(train_data["labels"].shape[0]),
        val_count=int(val_data["labels"].shape[0]),
        test_count=int(test_data["labels"].shape[0]),
    )

    input_cfg = cfg.get("input", {}) if isinstance(cfg.get("input"), dict) else {}
    resize_hw_raw = input_cfg.get("resize_hw")
    resize_hw: tuple[int, int] | None = None
    if isinstance(resize_hw_raw, (list, tuple)) and len(resize_hw_raw) == 2:
        resize_hw = (int(resize_hw_raw[0]), int(resize_hw_raw[1]))

    apply_circular_mask = bool(input_cfg.get("apply_circular_mask", False))
    mean = float((input_cfg.get("normalize", {}) or {}).get("mean", [0.5])[0])
    std = float((input_cfg.get("normalize", {}) or {}).get("std", [0.25])[0])

    train_patterns = _prepare_patterns(train_data["patterns"], resize_hw=resize_hw, apply_circular_mask=apply_circular_mask)
    val_patterns = _prepare_patterns(val_data["patterns"], resize_hw=resize_hw, apply_circular_mask=apply_circular_mask)
    test_patterns = _prepare_patterns(test_data["patterns"], resize_hw=resize_hw, apply_circular_mask=apply_circular_mask)
    emit(
        "PATTERN_PREPROCESS_COMPLETE",
        resize_hw=list(resize_hw) if resize_hw else None,
        apply_circular_mask=apply_circular_mask,
        train_shape=list(train_patterns.shape),
        val_shape=list(val_patterns.shape),
        test_shape=list(test_patterns.shape),
    )

    max_train_samples = cfg.get("max_train_samples")
    max_val_samples = cfg.get("max_val_samples")
    max_test_samples = cfg.get("max_test_samples")

    train_patterns, train_labels = _limit_samples(train_patterns, train_data["labels"], int(max_train_samples) if max_train_samples is not None else None)
    val_patterns, val_labels = _limit_samples(val_patterns, val_data["labels"], int(max_val_samples) if max_val_samples is not None else None)
    test_patterns, test_labels = _limit_samples(test_patterns, test_data["labels"], int(max_test_samples) if max_test_samples is not None else None)
    emit(
        "LIMIT_SAMPLES_COMPLETE",
        train_count=int(train_labels.shape[0]),
        val_count=int(val_labels.shape[0]),
        test_count=int(test_labels.shape[0]),
    )

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = ArrayDataset(train_patterns, train_labels, mean=mean, std=std)
    val_ds = ArrayDataset(val_patterns, val_labels, mean=mean, std=std)
    test_ds = ArrayDataset(test_patterns, test_labels, mean=mean, std=std)

    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    emit(
        "DATALOADERS_READY",
        batch_size=batch_size,
        num_workers=num_workers,
        train_batches=int(len(train_loader)),
        val_batches=int(len(val_loader)),
        test_batches=int(len(test_loader)),
    )

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    in_chans = int(model_cfg.get("in_chans", 1))
    model_build = build_model(model_cfg, num_classes=num_classes, in_chans=in_chans)

    device = _resolve_device(str(cfg.get("device", "auto")))
    model = model_build.model.to(device)
    emit(
        "MODEL_READY",
        family=model_build.family,
        model_name=model_build.model_name,
        pretrained=bool(model_build.pretrained),
        in_chans=int(in_chans),
        num_classes=int(num_classes),
        device=str(device),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(cfg.get("epochs", 10))
    amp = bool(cfg.get("amp", True))

    history: list[dict[str, Any]] = []
    history_jsonl = out_dir / "epoch_history.jsonl"
    history_jsonl.write_text("", encoding="utf-8")

    best_metric = -float("inf")
    best_epoch = -1
    best_checkpoint = out_dir / "best_checkpoint.pt"
    last_checkpoint = out_dir / "last_checkpoint.pt"

    start_time = time.time()
    emit("TRAIN_LOOP_START", epochs=epochs, learning_rate=float(cfg.get("learning_rate", 3e-4)))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        emit("EPOCH_START", epoch=epoch, epochs=epochs)
        train_loss, train_true, train_pred = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp=amp,
        )
        val_loss, val_true, val_pred = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            amp=amp,
        )

        train_metrics = classification_metrics(
            y_true=train_true,
            y_pred=train_pred,
            num_classes=num_classes,
            class_names=class_names,
        )
        val_metrics = classification_metrics(
            y_true=val_true,
            y_pred=val_pred,
            num_classes=num_classes,
            class_names=class_names,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": float(time.time() - t0),
            "progress_pct": float(100.0 * epoch / max(1, epochs)),
            "eta_seconds": _safe_eta_seconds(
                completed=epoch,
                total=epochs,
                elapsed=float(time.time() - start_time),
            ),
        }
        history.append(row)

        with history_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        metric = float(val_metrics["macro_f1"])
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": metric,
                    "model": {
                        "family": model_build.family,
                        "model_name": model_build.model_name,
                        "pretrained": model_build.pretrained,
                    },
                },
                best_checkpoint,
            )
            emit(
                "BEST_CHECKPOINT_UPDATED",
                epoch=epoch,
                best_val_macro_f1=float(best_metric),
                checkpoint_path=rel_path(best_checkpoint, repo_root),
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": metric,
            },
            last_checkpoint,
        )
        emit("LAST_CHECKPOINT_SAVED", epoch=epoch, checkpoint_path=rel_path(last_checkpoint, repo_root))

        log.info(
            (
                "Epoch %d/%d (%.1f%%) | train_loss=%.5f val_loss=%.5f val_macro_f1=%.4f "
                "elapsed=%.2fs eta=%.2fs"
            ),
            epoch,
            epochs,
            row["progress_pct"],
            train_loss,
            val_loss,
            val_metrics["macro_f1"],
            float(time.time() - start_time),
            row["eta_seconds"] if row["eta_seconds"] is not None else 0.0,
        )
        emit(
            "EPOCH_END",
            epoch=epoch,
            epochs=epochs,
            progress_pct=row["progress_pct"],
            train_loss=train_loss,
            val_loss=val_loss,
            val_macro_f1=val_metrics["macro_f1"],
            epoch_seconds=row["epoch_seconds"],
            eta_seconds=row["eta_seconds"],
        )

    if best_checkpoint.exists():
        ckpt = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_true, test_pred = _run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        amp=amp,
    )
    test_metrics = classification_metrics(
        y_true=test_true,
        y_pred=test_pred,
        num_classes=num_classes,
        class_names=class_names,
    )

    runtime_seconds = float(time.time() - start_time)
    emit(
        "TEST_EVAL_COMPLETE",
        test_loss=float(test_loss),
        test_accuracy=float(test_metrics["accuracy"]),
        test_macro_f1=float(test_metrics["macro_f1"]),
        runtime_seconds=runtime_seconds,
    )

    report = {
        "schema_version": "phase_id_xcorr.ml_training_report.v1",
        "timestamp_utc": _now_iso_utc(),
        "git_commit": _git_commit(repo_root),
        "config_path": rel_path(cfg_path, repo_root),
        "dataset_manifest_path": rel_path(dataset_manifest_path, repo_root),
        "output_dir": rel_path(out_dir, repo_root),
        "status": "completed",
        "debug": bool(debug),
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "device": str(device),
        "model": {
            "family": model_build.family,
            "model_name": model_build.model_name,
            "pretrained": model_build.pretrained,
            "in_chans": in_chans,
            "num_classes": num_classes,
        },
        "input": {
            "resize_hw": list(resize_hw) if resize_hw else None,
            "apply_circular_mask": apply_circular_mask,
            "normalize": {"mean": mean, "std": std},
        },
        "dataset_counts": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "history": history,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "artifacts": {
            "epoch_history_jsonl": rel_path(history_jsonl, repo_root),
            "report_json": rel_path(out_dir / "report.json", repo_root),
            "manifest_json": rel_path(out_dir / "manifest.json", repo_root),
            "event_log_jsonl": rel_path(event_log, repo_root),
            "best_checkpoint": rel_path(best_checkpoint, repo_root) if best_checkpoint.exists() else None,
            "last_checkpoint": rel_path(last_checkpoint, repo_root),
        },
    }

    report_path = out_dir / "report.json"
    write_json(report_path, report)
    emit("REPORT_WRITE_COMPLETE", report_path=rel_path(report_path, repo_root))

    manifest = {
        "schema_version": "phase_id_xcorr.ml_train_manifest.v1",
        "timestamp_utc": _now_iso_utc(),
        "git_commit": _git_commit(repo_root),
        "workflow": "ml_train_classifier",
        "config_path": rel_path(cfg_path, repo_root),
        "dataset_manifest_path": rel_path(dataset_manifest_path, repo_root),
        "output_dir": rel_path(out_dir, repo_root),
        "debug": bool(debug),
        "device": str(device),
        "model": {
            "family": model_build.family,
            "model_name": model_build.model_name,
            "pretrained": model_build.pretrained,
        },
        "timing": {
            "total_runtime_seconds": runtime_seconds,
            "epochs": epochs,
            "mean_epoch_seconds": float(np.mean([float(row["epoch_seconds"]) for row in history])) if history else 0.0,
        },
        "sanity_checks": {
            "dataset_manifest_exists": bool(dataset_manifest_path.exists()),
            "phase_class_count_match": num_classes == len(class_names),
            "non_empty_train_split": len(train_ds) > 0,
            "non_empty_val_split": len(val_ds) > 0,
            "non_empty_test_split": len(test_ds) > 0,
            "history_written": bool(history),
        },
        "artifacts": report["artifacts"],
    }
    write_json(out_dir / "manifest.json", manifest)
    emit("MANIFEST_WRITE_COMPLETE", manifest_path=rel_path(out_dir / "manifest.json", repo_root))

    summary_md = out_dir / "report.md"
    lines = [
        "# ML Training Summary",
        "",
        f"- model: `{model_build.model_name}` ({model_build.family}, pretrained={model_build.pretrained})",
        f"- classes: {', '.join(class_names)}",
        f"- dataset: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        f"- best epoch: {best_epoch}",
        f"- best val macro-F1: {best_metric:.5f}",
        f"- test accuracy: {test_metrics['accuracy']:.5f}",
        f"- test macro-F1: {test_metrics['macro_f1']:.5f}",
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("SUMMARY_WRITE_COMPLETE", summary_path=rel_path(summary_md, repo_root))

    log.info(
        "Training completed | best_epoch=%d best_val_macro_f1=%.5f test_macro_f1=%.5f",
        best_epoch,
        best_metric,
        test_metrics["macro_f1"],
    )
    emit(
        "RUN_END",
        status="completed",
        best_epoch=int(best_epoch),
        best_val_macro_f1=float(best_metric),
        test_macro_f1=float(test_metrics["macro_f1"]),
        total_runtime_seconds=runtime_seconds,
    )

    return TrainResult(
        out_dir=out_dir,
        report_path=report_path,
        best_checkpoint=best_checkpoint if best_checkpoint.exists() else None,
        last_checkpoint=last_checkpoint,
    )
