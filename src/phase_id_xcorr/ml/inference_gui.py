"""PySide6 GUI for phase-classifier inference on unknown images and full `.oh5` scans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .inference import LoadedModel, list_model_runs, load_trained_model, predict_image, predict_pattern_array
from .oh5_inference import FullScanInferenceResult, export_full_scan_artifacts, run_oh5_full_scan_inference
from .orientation_diagnostics import render_ipf_colored_scan_map, render_ipf_reference_panel


INFERENCE_MODE_IMAGE = "image"
INFERENCE_MODE_FULL_SCAN = "full_scan"
_DEFAULT_PHASE_COLORS: tuple[tuple[int, int, int], ...] = (
    (220, 68, 55),
    (55, 126, 34),
    (56, 99, 214),
    (213, 160, 33),
    (118, 84, 172),
    (33, 163, 163),
)


def _gray_array_to_pixmap(array: np.ndarray, *, target_size: QtCore.QSize) -> QtGui.QPixmap:
    arr = np.clip(array, 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    h, w = arr8.shape
    qimg = QtGui.QImage(arr8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    pix = QtGui.QPixmap.fromImage(qimg.copy())
    return pix.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


def _gray_array_to_native_pixmap(array: np.ndarray) -> QtGui.QPixmap:
    arr = np.clip(array, 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    h, w = arr8.shape
    qimg = QtGui.QImage(arr8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qimg.copy())


def _rgb_array_to_pixmap(array: np.ndarray, *, target_size: QtCore.QSize) -> QtGui.QPixmap:
    arr = np.clip(array, 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    h, w, _ = arr8.shape
    qimg = QtGui.QImage(arr8.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    pix = QtGui.QPixmap.fromImage(qimg.copy())
    return pix.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)


def _phase_color_map(class_names: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    explicit = {
        "Al": np.asarray([0.92, 0.30, 0.25], dtype=np.float32),
        "Ni": np.asarray([0.16, 0.64, 0.34], dtype=np.float32),
        "Cu": np.asarray([0.20, 0.44, 0.88], dtype=np.float32),
    }
    for idx, phase in enumerate(class_names):
        if phase in explicit:
            out[phase] = explicit[phase]
            continue
        rgb = _DEFAULT_PHASE_COLORS[idx % len(_DEFAULT_PHASE_COLORS)]
        out[phase] = np.asarray(rgb, dtype=np.float32) / 255.0
    return out


def _contrast_stretch_gray(array: np.ndarray, *, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    values = arr[finite]
    lo = float(np.percentile(values, lower_pct))
    hi = float(np.percentile(values, upper_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    out = (arr - lo) / (hi - lo)
    out = np.where(finite, out, 0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _histogram_equalize_gray(array: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    values = arr[finite]
    hist, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    if np.sum(hist) <= 0:
        return arr.astype(np.float32, copy=False)
    cdf = np.cumsum(hist).astype(np.float32)
    nonzero = cdf > 0
    if not np.any(nonzero):
        return arr.astype(np.float32, copy=False)
    cdf_min = float(cdf[nonzero][0])
    denom = float(cdf[-1] - cdf_min)
    if denom <= 0.0:
        return arr.astype(np.float32, copy=False)
    lut = np.clip((cdf - cdf_min) / denom, 0.0, 1.0)
    bin_idx = np.clip(np.searchsorted(edges[1:], values, side="right"), 0, len(lut) - 1)
    out = np.zeros_like(arr, dtype=np.float32)
    out[finite] = lut[bin_idx]
    return out


def _prepare_display_gray(
    array: np.ndarray,
    *,
    histogram_normalization: bool,
    contrast_stretch: bool,
) -> np.ndarray:
    out = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    if histogram_normalization:
        out = _histogram_equalize_gray(out)
    if contrast_stretch:
        out = _contrast_stretch_gray(out)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _overlay_selected_pixel(image: np.ndarray, *, x: int | None, y: int | None) -> np.ndarray:
    if x is None or y is None:
        return image
    if y < 0 or x < 0 or y >= image.shape[0] or x >= image.shape[1]:
        return image
    marked = np.array(image, copy=True)
    y0 = max(0, y - 2)
    y1 = min(marked.shape[0], y + 3)
    x0 = max(0, x - 2)
    x1 = min(marked.shape[1], x + 3)
    marked[y0:y1, x0:x1] = 1.0
    inner_y0 = max(0, y - 1)
    inner_y1 = min(marked.shape[0], y + 2)
    inner_x0 = max(0, x - 1)
    inner_x1 = min(marked.shape[1], x + 2)
    marked[inner_y0:inner_y1, inner_x0:inner_x1] = 0.0
    marked[y, x] = 1.0
    return marked


def _render_full_scan_phase_map(
    result: FullScanInferenceResult,
    *,
    use_confidence_shading: bool,
) -> np.ndarray:
    image = np.full((result.ny, result.nx, 3), 0.12, dtype=np.float32)
    palette = _phase_color_map(result.class_names)
    neutral = np.asarray([0.55, 0.55, 0.55], dtype=np.float32)

    for flat_index in range(result.header_total_pixels):
        y = flat_index // result.nx
        x = flat_index % result.nx
        class_idx = int(result.predicted_indices[flat_index])
        if class_idx < 0:
            continue
        phase_name = result.class_names[class_idx]
        base = palette[phase_name]
        if use_confidence_shading:
            confidence = float(result.confidences[flat_index])
            strength = float(np.clip(confidence, 0.0, 1.0))
            image[y, x] = neutral * (1.0 - strength) + base * strength
        else:
            image[y, x] = base
    return image


class OverlayImageWidget(QtWidgets.QWidget):
    imageClicked = QtCore.Signal(int, int)

    def __init__(self, placeholder: str) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.image_label = ClickableImageLabel(placeholder)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.image_label.setMinimumSize(540, 420)
        self.image_label.imageClicked.connect(self.imageClicked.emit)
        layout.addWidget(self.image_label)
        self._overlay_visible = False
        self._overlay_progress = 0

    def setText(self, text: str) -> None:  # noqa: N802
        self.image_label.setText(text)

    def text(self) -> str:
        return self.image_label.text()

    def setPixmap(self, pixmap: QtGui.QPixmap) -> None:  # noqa: N802
        self.image_label.setPixmap(pixmap)

    def pixmap(self) -> QtGui.QPixmap | None:
        return self.image_label.pixmap()

    def set_source_image_size(self, width: int | None, height: int | None) -> None:
        self.image_label.set_source_image_size(width, height)

    def set_overlay_progress(self, value: int) -> None:
        self._overlay_progress = max(0, min(100, int(value)))
        self._overlay_visible = True
        self.update()

    def clear_overlay(self) -> None:
        self._overlay_visible = False
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self._overlay_visible:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        center = self.rect().center()
        radius = max(48, min(self.width(), self.height()) // 9)
        ring_rect = QtCore.QRectF(
            float(center.x() - radius),
            float(center.y() - radius - 16),
            float(radius * 2),
            float(radius * 2),
        )

        painter.setBrush(QtGui.QColor(15, 15, 15, 130))
        painter.setPen(QtCore.Qt.NoPen)
        backdrop_rect = ring_rect.adjusted(-28, -20, 28, 72)
        painter.drawRoundedRect(backdrop_rect, 18.0, 18.0)

        base_pen = QtGui.QPen(QtGui.QColor(245, 245, 245, 120))
        base_pen.setWidth(10)
        painter.setPen(base_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(ring_rect)

        progress_pen = QtGui.QPen(QtGui.QColor(250, 250, 250))
        progress_pen.setWidth(10)
        progress_pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(progress_pen)
        span_angle = int(round(-5760.0 * (self._overlay_progress / 100.0)))
        painter.drawArc(ring_rect, 90 * 16, span_angle)

        percent_font = QtGui.QFont(self.font())
        percent_font.setPointSize(max(15, percent_font.pointSize() + 4))
        percent_font.setBold(True)
        painter.setFont(percent_font)
        painter.setPen(QtGui.QColor(255, 255, 255))
        percent_rect = QtCore.QRectF(
            backdrop_rect.left(),
            ring_rect.bottom() + 12.0,
            backdrop_rect.width(),
            34.0,
        )
        painter.drawText(percent_rect, QtCore.Qt.AlignCenter, f"{self._overlay_progress}%")
        painter.end()


class ClickableImageLabel(QtWidgets.QLabel):
    imageClicked = QtCore.Signal(int, int)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._source_size: tuple[int, int] | None = None

    def set_source_image_size(self, width: int | None, height: int | None) -> None:
        if width is None or height is None or width <= 0 or height <= 0:
            self._source_size = None
            return
        self._source_size = (int(width), int(height))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton and self._source_size is not None:
            mapped = self._map_to_source(event.position())
            if mapped is not None:
                self.imageClicked.emit(mapped[0], mapped[1])
                event.accept()
                return
        super().mousePressEvent(event)

    def _map_to_source(self, pos: QtCore.QPointF) -> tuple[int, int] | None:
        if self._source_size is None:
            return None
        rect = self.contentsRect()
        src_w, src_h = self._source_size
        scale = min(rect.width() / max(1, src_w), rect.height() / max(1, src_h))
        if scale <= 0.0:
            return None
        display_w = src_w * scale
        display_h = src_h * scale
        offset_x = rect.x() + (rect.width() - display_w) / 2.0
        offset_y = rect.y() + (rect.height() - display_h) / 2.0
        rel_x = float(pos.x()) - offset_x
        rel_y = float(pos.y()) - offset_y
        if rel_x < 0.0 or rel_y < 0.0 or rel_x >= display_w or rel_y >= display_h:
            return None
        src_x = min(src_w - 1, max(0, int(rel_x / scale)))
        src_y = min(src_h - 1, max(0, int(rel_y / scale)))
        return src_x, src_y


class _SyncedImageView(QtWidgets.QGraphicsView):
    hoverInfoChanged = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene(self)
        self._pixmap_item = self._scene.addPixmap(QtGui.QPixmap())
        self.setScene(self._scene)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setMinimumSize(320, 240)
        self._linked_views: list[_SyncedImageView] = []
        self._fit_mode = True
        self._sync_guard = False
        self._array_shape: tuple[int, int] | None = None
        self.horizontalScrollBar().valueChanged.connect(self._notify_linked_views)
        self.verticalScrollBar().valueChanged.connect(self._notify_linked_views)
        self.setMouseTracking(True)

    def link_views(self, peers: list["_SyncedImageView"]) -> None:
        self._linked_views = [peer for peer in peers if peer is not self]

    def has_image(self) -> bool:
        return not self._pixmap_item.pixmap().isNull()

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        if pixmap.isNull():
            self.resetTransform()
            return
        if self._fit_mode:
            self.fit_to_scene(notify=False)
        else:
            self._notify_linked_views()

    def clear_pixmap(self) -> None:
        self._pixmap_item.setPixmap(QtGui.QPixmap())
        self._scene.setSceneRect(QtCore.QRectF())
        self.resetTransform()
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)
        self._fit_mode = True
        self._array_shape = None
        self.hoverInfoChanged.emit(None)

    def set_array_shape(self, shape_hw: tuple[int, int] | None) -> None:
        self._array_shape = shape_hw

    def fit_to_scene(self, *, notify: bool = True) -> None:
        if not self.has_image():
            return
        self._fit_mode = True
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        if notify:
            self._notify_linked_views()

    def zoom_by(self, factor: float, *, notify: bool = True) -> None:
        if not self.has_image():
            return
        self._fit_mode = False
        self.scale(float(factor), float(factor))
        if notify:
            self._notify_linked_views()

    def reset_zoom(self) -> None:
        self.fit_to_scene(notify=True)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit_mode and self.has_image():
            self.fit_to_scene(notify=False)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        if not self.has_image():
            super().wheelEvent(event)
            return
        delta_y = event.angleDelta().y()
        if delta_y == 0:
            super().wheelEvent(event)
            return
        factor = 1.15 if delta_y > 0 else 1.0 / 1.15
        self.zoom_by(factor, notify=True)
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        self._notify_linked_views()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        super().mouseMoveEvent(event)
        self._emit_hover_info(event.position())

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        super().leaveEvent(event)
        self.hoverInfoChanged.emit(None)

    def _notify_linked_views(self, *_args: object) -> None:
        if self._sync_guard or not self._linked_views or not self.has_image():
            return
        self._sync_guard = True
        try:
            transform = QtGui.QTransform(self.transform())
            h_value = int(self.horizontalScrollBar().value())
            v_value = int(self.verticalScrollBar().value())
            fit_mode = bool(self._fit_mode)
            for peer in self._linked_views:
                peer._apply_view_state(transform, h_value, v_value, fit_mode)
        finally:
            self._sync_guard = False

    def _emit_hover_info(self, pos: QtCore.QPointF) -> None:
        mapped = self._map_viewport_to_source(pos)
        self.hoverInfoChanged.emit(mapped)

    def _map_viewport_to_source(self, pos: QtCore.QPointF) -> tuple[int, int] | None:
        if not self.has_image() or self._array_shape is None:
            return None
        scene_pos = self.mapToScene(QtCore.QPoint(int(round(pos.x())), int(round(pos.y()))))
        x = int(np.floor(scene_pos.x()))
        y = int(np.floor(scene_pos.y()))
        h, w = self._array_shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return x, y

    def _apply_view_state(
        self,
        transform: QtGui.QTransform,
        h_value: int,
        v_value: int,
        fit_mode: bool,
    ) -> None:
        if not self.has_image():
            return
        self._sync_guard = True
        try:
            self._fit_mode = fit_mode
            if fit_mode:
                self.fit_to_scene(notify=False)
            else:
                self.setTransform(QtGui.QTransform(transform))
                self.horizontalScrollBar().setValue(
                    max(self.horizontalScrollBar().minimum(), min(self.horizontalScrollBar().maximum(), int(h_value)))
                )
                self.verticalScrollBar().setValue(
                    max(self.verticalScrollBar().minimum(), min(self.verticalScrollBar().maximum(), int(v_value)))
                )
        finally:
            self._sync_guard = False


class _PatternPane(QtWidgets.QWidget):
    def __init__(self, title: str, placeholder: str) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.group = QtWidgets.QGroupBox(title)
        group_layout = QtWidgets.QVBoxLayout(self.group)
        self._array: np.ndarray | None = None

        self.stack = QtWidgets.QStackedWidget()
        self.placeholder = QtWidgets.QLabel(placeholder)
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.view = _SyncedImageView()
        self.hover_label = QtWidgets.QLabel("Hover: -")
        self.hover_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.view.hoverInfoChanged.connect(self._update_hover_label)
        self.stack.addWidget(self.placeholder)
        self.stack.addWidget(self.view)

        group_layout.addWidget(self.stack)
        group_layout.addWidget(self.hover_label)
        layout.addWidget(self.group)

    def set_array(self, array: np.ndarray) -> None:
        self._array = np.asarray(array, dtype=np.float32)
        pixmap = _gray_array_to_native_pixmap(array)
        self.view.set_array_shape(tuple(self._array.shape))
        self.view.set_pixmap(pixmap)
        self.stack.setCurrentWidget(self.view)
        self.hover_label.setText("Hover: move cursor over image")

    def clear(self, message: str | None = None) -> None:
        self._array = None
        self.view.clear_pixmap()
        if message:
            self.placeholder.setText(message)
        self.stack.setCurrentWidget(self.placeholder)
        self.hover_label.setText("Hover: -")

    def _update_hover_label(self, mapped: object) -> None:
        if mapped is None or self._array is None:
            self.hover_label.setText("Hover: -")
            return
        x, y = mapped
        value = float(self._array[int(y), int(x)])
        self.hover_label.setText(f"Hover: x={int(x)} y={int(y)} value={value:.4f}")


class _PatternCompareWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        self.fit_button = QtWidgets.QPushButton("Fit")
        self.zoom_in_button = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_button = QtWidgets.QPushButton("Zoom -")
        self.reset_button = QtWidgets.QPushButton("Reset")
        controls_layout.addWidget(self.fit_button)
        controls_layout.addWidget(self.zoom_in_button)
        controls_layout.addWidget(self.zoom_out_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addStretch(1)
        layout.addLayout(controls_layout)

        panes_layout = QtWidgets.QHBoxLayout()
        panes_layout.setContentsMargins(0, 0, 0, 0)
        panes_layout.setSpacing(8)
        self.raw_pane = _PatternPane("Original pattern", "No pixel selected")
        self.processed_pane = _PatternPane("Processed pattern", "No pixel selected")
        self.raw_pane.view.link_views([self.processed_pane.view])
        self.processed_pane.view.link_views([self.raw_pane.view])
        panes_layout.addWidget(self.raw_pane, stretch=1)
        panes_layout.addWidget(self.processed_pane, stretch=1)
        layout.addLayout(panes_layout, stretch=1)

        self.fit_button.clicked.connect(self.fit_views)
        self.zoom_in_button.clicked.connect(lambda: self._zoom_views(1.15))
        self.zoom_out_button.clicked.connect(lambda: self._zoom_views(1.0 / 1.15))
        self.reset_button.clicked.connect(self.reset_views)

    def set_patterns(self, raw_pattern: np.ndarray, processed_pattern: np.ndarray) -> None:
        self.raw_pane.set_array(raw_pattern)
        self.processed_pane.set_array(processed_pattern)
        self.fit_views()

    def set_processed_pattern(self, processed_pattern: np.ndarray) -> None:
        self.processed_pane.set_array(processed_pattern)

    def clear(self, message: str = "No pixel selected") -> None:
        self.raw_pane.clear(message)
        self.processed_pane.clear(message)

    def fit_views(self) -> None:
        self.raw_pane.view.fit_to_scene(notify=True)

    def reset_views(self) -> None:
        self.fit_views()

    def _zoom_views(self, factor: float) -> None:
        self.raw_pane.view.zoom_by(float(factor), notify=True)


class DropImageLabel(QtWidgets.QLabel):
    imageDropped = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__("Drop unknown image here\nor use Browse Image")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setMinimumHeight(140)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        local = urls[0].toLocalFile()
        if local:
            self.imageDropped.emit(local)
            event.acceptProposedAction()


@dataclass(slots=True)
class GuiState:
    suite_root: Path | None = None
    run_dirs: list[Path] | None = None
    loaded_model: LoadedModel | None = None
    image_path: Path | None = None
    oh5_path: Path | None = None
    inference_mode: str = INFERENCE_MODE_IMAGE
    full_scan_result: FullScanInferenceResult | None = None
    full_scan_ipf_image: np.ndarray | None = None
    full_scan_ipf_map_image: np.ndarray | None = None
    selected_pixel_x: int | None = None
    selected_pixel_y: int | None = None
    selected_pattern: np.ndarray | None = None
    selected_processed_pattern: np.ndarray | None = None
    selected_pixel_details: dict[str, Any] | None = None


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "-"
    total = max(0, int(round(float(seconds))))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


class FullScanWorker(QtCore.QObject):
    progress = QtCore.Signal(object)
    log_message = QtCore.Signal(str, str)
    finished = QtCore.Signal(object, object, object)
    failed = QtCore.Signal(str)

    def __init__(self, *, loaded: LoadedModel, oh5_path: Path):
        super().__init__()
        self.loaded = loaded
        self.oh5_path = oh5_path

    def run(self) -> None:
        try:
            result = run_oh5_full_scan_inference(
                loaded=self.loaded,
                oh5_path=self.oh5_path,
                scan_name=self.oh5_path.stem,
                progress_callback=self._emit_progress,
                log_callback=self._emit_log,
            )
            ipf_image: np.ndarray | None = None
            if result.euler_rows_deg is not None:
                self._emit_log("info", "Rendering IPF reference view from scan Euler angles.")
                eulers_by_phase: dict[str, np.ndarray] = {}
                palette = _phase_color_map(result.class_names)
                for class_idx, phase_name in enumerate(result.class_names):
                    class_mask = result.predicted_indices == class_idx
                    if not np.any(class_mask):
                        eulers_by_phase[phase_name] = np.empty((0, 3), dtype=np.float64)
                        continue
                    eulers = np.asarray(result.euler_rows_deg[class_mask], dtype=np.float64)
                    finite_mask = np.all(np.isfinite(eulers), axis=1)
                    eulers_by_phase[phase_name] = eulers[finite_mask]
                try:
                    ipf_image = render_ipf_reference_panel(
                        eulers_deg_by_phase=eulers_by_phase,
                        phase_names=list(result.class_names),
                        phase_colors={k: tuple(float(v) for v in rgb) for k, rgb in palette.items()},
                        title=f"{result.scan_name} orientation reference",
                    )
                    self._emit_log("info", "IPF reference rendering complete.")
                except Exception as exc:
                    self._emit_log("warning", f"IPF reference rendering skipped: {exc}")
            ipf_map_image: np.ndarray | None = None
            if result.euler_rows_deg is not None:
                self._emit_log("info", "Rendering IPF-colored EBSD map from scan Euler angles.")
                try:
                    ipf_map_image = render_ipf_colored_scan_map(
                        eulers_deg=result.euler_rows_deg,
                        predicted_indices=result.predicted_indices,
                        class_names=result.class_names,
                        nx=result.nx,
                        ny=result.ny,
                    )
                    self._emit_log("info", "IPF-colored EBSD map rendering complete.")
                except Exception as exc:
                    self._emit_log("warning", f"IPF-colored EBSD map rendering skipped: {exc}")
            self.finished.emit(result, ipf_image, ipf_map_image)
        except Exception as exc:  # pragma: no cover - Qt worker delivery
            self.failed.emit(str(exc))

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        self.progress.emit(payload)

    def _emit_log(self, level: str, message: str) -> None:
        self.log_message.emit(level, message)


class InferenceMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *, repo_root: Path, initial_root: Path | None, logger: logging.Logger):
        super().__init__()
        self.repo_root = repo_root
        self.log = logger
        self.state = GuiState(suite_root=initial_root, run_dirs=[])
        self._full_scan_thread: QtCore.QThread | None = None
        self._full_scan_worker: FullScanWorker | None = None
        self._full_scan_preview_ready = False
        self.setWindowTitle("ML Phase ID Inference")
        self.resize(1320, 820)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QGridLayout()
        layout.addLayout(top)

        self.root_edit = QtWidgets.QLineEdit(str(initial_root) if initial_root else "")
        btn_root = QtWidgets.QPushButton("Browse Suite/Run")
        btn_root.clicked.connect(self._browse_root)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.currentIndexChanged.connect(self._load_selected_model)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Single image", userData=INFERENCE_MODE_IMAGE)
        self.mode_combo.addItem("Full .oh5 scan", userData=INFERENCE_MODE_FULL_SCAN)
        self.mode_combo.currentIndexChanged.connect(self._update_mode_ui)
        self.known_phase_combo = QtWidgets.QComboBox()
        self.known_phase_combo.addItem("(unknown)")
        self.known_phase_combo.currentIndexChanged.connect(self._update_known_phase_status)
        self.status_label = QtWidgets.QLabel("Select a suite root or run directory.")
        self.status_label.setWordWrap(True)
        self.scan_progress = QtWidgets.QProgressBar()
        self.scan_progress.setRange(0, 100)
        self.scan_progress.setValue(0)
        self.scan_eta_label = QtWidgets.QLabel("ETA: -")

        top.addWidget(QtWidgets.QLabel("Suite root / run dir"), 0, 0)
        top.addWidget(self.root_edit, 0, 1)
        top.addWidget(btn_root, 0, 2)
        top.addWidget(QtWidgets.QLabel("Model"), 1, 0)
        top.addWidget(self.model_combo, 1, 1, 1, 2)
        top.addWidget(QtWidgets.QLabel("Inference mode"), 2, 0)
        top.addWidget(self.mode_combo, 2, 1, 1, 2)
        top.addWidget(QtWidgets.QLabel("Known phase"), 3, 0)
        top.addWidget(self.known_phase_combo, 3, 1, 1, 2)
        top.addWidget(self.status_label, 4, 0, 1, 2)
        top.addWidget(self.scan_eta_label, 4, 2)
        top.addWidget(self.scan_progress, 5, 0, 1, 3)

        mid = QtWidgets.QHBoxLayout()
        layout.addLayout(mid, stretch=1)

        left = QtWidgets.QVBoxLayout()
        mid.addLayout(left, stretch=1)
        self.input_stack = QtWidgets.QStackedWidget()
        left.addWidget(self.input_stack)

        image_page = QtWidgets.QWidget()
        image_layout = QtWidgets.QVBoxLayout(image_page)
        self.drop_label = DropImageLabel()
        self.drop_label.imageDropped.connect(self._set_image_path)
        image_layout.addWidget(self.drop_label)
        btn_img = QtWidgets.QPushButton("Browse Image")
        btn_img.clicked.connect(self._browse_image)
        image_layout.addWidget(btn_img)

        self.original_preview = QtWidgets.QLabel("Original image")
        self.original_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.original_preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.original_preview.setMinimumSize(320, 320)
        image_layout.addWidget(self.original_preview, stretch=1)

        self.preprocessed_preview = QtWidgets.QLabel("Preprocessed image")
        self.preprocessed_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preprocessed_preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.preprocessed_preview.setMinimumSize(320, 320)
        image_layout.addWidget(self.preprocessed_preview, stretch=1)
        self.input_stack.addWidget(image_page)

        oh5_page = QtWidgets.QWidget()
        oh5_layout = QtWidgets.QVBoxLayout(oh5_page)
        oh5_controls = QtWidgets.QGridLayout()
        oh5_layout.addLayout(oh5_controls)
        self.oh5_edit = QtWidgets.QLineEdit()
        btn_oh5 = QtWidgets.QPushButton("Browse .oh5")
        btn_oh5.clicked.connect(self._browse_oh5)
        self.btn_oh5 = btn_oh5
        self.confidence_shading_checkbox = QtWidgets.QCheckBox("Use confidence shading")
        self.confidence_shading_checkbox.setChecked(True)
        self.confidence_shading_checkbox.toggled.connect(self._refresh_full_scan_preview)
        btn_full_scan = QtWidgets.QPushButton("Start inference")
        btn_full_scan.clicked.connect(self._run_inference)
        self.btn_full_scan = btn_full_scan
        btn_export = QtWidgets.QPushButton("Export Results")
        btn_export.clicked.connect(self._export_full_scan_results)
        btn_export.setEnabled(False)
        self.btn_export = btn_export
        oh5_controls.addWidget(QtWidgets.QLabel(".oh5 file"), 0, 0)
        oh5_controls.addWidget(self.oh5_edit, 0, 1)
        oh5_controls.addWidget(btn_oh5, 0, 2)
        oh5_controls.addWidget(self.confidence_shading_checkbox, 1, 0, 1, 2)
        oh5_controls.addWidget(btn_full_scan, 1, 2)
        oh5_controls.addWidget(btn_export, 2, 2)

        inspector_group = QtWidgets.QGroupBox("Clicked Pixel Inspector")
        inspector_layout = QtWidgets.QVBoxLayout(inspector_group)

        self.selected_pixel_status = QtWidgets.QLabel(
            "Click a pixel in the predicted phase map to inspect its Kikuchi pattern."
        )
        self.selected_pixel_status.setWordWrap(True)
        inspector_layout.addWidget(self.selected_pixel_status)

        selected_info = QtWidgets.QFormLayout()
        self.selected_pixel_value = QtWidgets.QLabel("-")
        self.selected_phase_value = QtWidgets.QLabel("-")
        self.selected_confidence_value = QtWidgets.QLabel("-")
        self.selected_euler_value = QtWidgets.QLabel("-")
        self.selected_euler_value.setWordWrap(True)
        self.selected_ci_value = QtWidgets.QLabel("-")
        self.selected_iq_value = QtWidgets.QLabel("-")
        self.selected_fit_value = QtWidgets.QLabel("-")
        self.selected_valid_value = QtWidgets.QLabel("-")
        selected_info.addRow("Pixel", self.selected_pixel_value)
        selected_info.addRow("Phase", self.selected_phase_value)
        selected_info.addRow("Confidence", self.selected_confidence_value)
        selected_info.addRow("Euler (deg)", self.selected_euler_value)
        selected_info.addRow("CI", self.selected_ci_value)
        selected_info.addRow("IQ", self.selected_iq_value)
        selected_info.addRow("Fit", self.selected_fit_value)
        selected_info.addRow("Valid", self.selected_valid_value)
        inspector_layout.addLayout(selected_info)

        display_controls_group = QtWidgets.QGroupBox("Pattern Display")
        display_controls = QtWidgets.QVBoxLayout(display_controls_group)
        self.histogram_normalization_checkbox = QtWidgets.QCheckBox("Histogram normalization")
        self.histogram_normalization_checkbox.toggled.connect(self._refresh_selected_pattern_preview)
        self.contrast_stretch_checkbox = QtWidgets.QCheckBox("Contrast stretch")
        self.contrast_stretch_checkbox.toggled.connect(self._refresh_selected_pattern_preview)
        btn_reset_pattern_display = QtWidgets.QPushButton("Reset display controls")
        btn_reset_pattern_display.clicked.connect(self._reset_pattern_display_controls)
        display_controls.addWidget(self.histogram_normalization_checkbox)
        display_controls.addWidget(self.contrast_stretch_checkbox)
        display_controls.addWidget(btn_reset_pattern_display)
        inspector_layout.addWidget(display_controls_group, stretch=0)

        self.pattern_compare = _PatternCompareWidget()
        inspector_layout.addWidget(self.pattern_compare, stretch=1)

        self.oh5_help = QtWidgets.QPlainTextEdit()
        self.oh5_help.setReadOnly(True)
        self.oh5_help.setPlainText(
            "Full-scan mode runs inference on every available pattern in the selected .oh5 file.\n"
            "Click the predicted phase map to inspect the corresponding experimental Kikuchi pattern.\n"
            "The inspector keeps original and processed views synchronized for zoom and pan.\n"
            "Processed display is grayscale-only and can optionally apply histogram normalization and contrast stretch."
        )
        inspector_layout.addWidget(self.oh5_help, stretch=0)
        oh5_layout.addWidget(inspector_group, stretch=1)
        self.input_stack.addWidget(oh5_page)

        right = QtWidgets.QVBoxLayout()
        mid.addLayout(right, stretch=1)
        self.prediction_label = QtWidgets.QLabel("Prediction: -")
        font = self.prediction_label.font()
        font.setPointSize(18)
        font.setBold(True)
        self.prediction_label.setFont(font)
        right.addWidget(self.prediction_label)

        self.preview_tabs = QtWidgets.QTabWidget()
        self.predicted_map_tab = QtWidgets.QWidget()
        self.predicted_map_layout = QtWidgets.QVBoxLayout(self.predicted_map_tab)
        self.predicted_map_layout.setContentsMargins(0, 0, 0, 0)
        self.predicted_map_layout.setSpacing(8)
        self.result_preview = OverlayImageWidget("No result")
        self.result_preview.imageClicked.connect(self._handle_phase_map_click)
        self.predicted_map_layout.addWidget(self.result_preview, stretch=1)

        self.map_legend_widget = QtWidgets.QWidget()
        self.map_legend_layout = QtWidgets.QHBoxLayout(self.map_legend_widget)
        self.map_legend_layout.setContentsMargins(12, 2, 12, 6)
        self.map_legend_layout.setSpacing(18)
        self.predicted_map_layout.addWidget(self.map_legend_widget, stretch=0)

        self.ipf_preview = QtWidgets.QLabel("Load a scan to render the IPF orientation reference.")
        self.ipf_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.ipf_preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ipf_preview.setMinimumSize(540, 420)
        self.ipf_map_preview = QtWidgets.QLabel("Load a scan to render the IPF-colored EBSD map.")
        self.ipf_map_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.ipf_map_preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ipf_map_preview.setMinimumSize(540, 420)
        self.preview_tabs.addTab(self.predicted_map_tab, "Predicted phase map")
        self.preview_tabs.addTab(self.ipf_preview, "IPF reference")
        self.preview_tabs.addTab(self.ipf_map_preview, "IPF-colored EBSD map")
        right.addWidget(self.preview_tabs, stretch=2)

        self.prob_table = QtWidgets.QTableWidget(0, 2)
        self.prob_table.setHorizontalHeaderLabels(["Phase", "Probability"])
        self.prob_table.horizontalHeader().setStretchLastSection(True)
        self.prob_table.verticalHeader().setVisible(False)
        right.addWidget(self.prob_table, stretch=1)

        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setReadOnly(True)
        right.addWidget(self.notes, stretch=1)

        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.document().setMaximumBlockCount(2000)
        self.log_output.setPlaceholderText("Backend progress and errors will appear here.")
        right.addWidget(self.log_output, stretch=1)

        self._update_mode_ui()
        if initial_root is not None:
            self._refresh_run_dirs()

    def _browse_root(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Select suite root or model run directory")
        if not selected:
            return
        self.root_edit.setText(selected)
        self.state.suite_root = Path(selected)
        self._refresh_run_dirs()

    def _browse_image(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select unknown image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if selected:
            self._set_image_path(selected)

    def _browse_oh5(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select .oh5 scan",
            "",
            "OH5 files (*.oh5)",
        )
        if selected:
            self._set_oh5_path(selected)

    def _refresh_run_dirs(self) -> None:
        root_text = self.root_edit.text().strip()
        if not root_text:
            return
        root = Path(root_text).expanduser().resolve()
        self.state.suite_root = root
        run_dirs = list_model_runs(root)
        self.state.run_dirs = run_dirs
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for run_dir in run_dirs:
            self.model_combo.addItem(run_dir.name, userData=str(run_dir))
        self.model_combo.blockSignals(False)
        if run_dirs:
            self._load_selected_model()
        else:
            self.status_label.setText(f"No run directories with report.json found under {root}")

    def _load_selected_model(self) -> None:
        run_dir_text = self.model_combo.currentData()
        if not run_dir_text:
            return
        try:
            loaded = load_trained_model(run_dir=Path(run_dir_text), repo_root=self.repo_root)
        except Exception as exc:
            self.state.loaded_model = None
            self.status_label.setText(f"Failed to load model: {exc}")
            return
        self.state.loaded_model = loaded
        self.known_phase_combo.blockSignals(True)
        self.known_phase_combo.clear()
        self.known_phase_combo.addItem("(unknown)")
        for phase in loaded.class_names:
            self.known_phase_combo.addItem(phase)
        self.known_phase_combo.blockSignals(False)
        self.status_label.setText(f"Loaded {loaded.model_name} from {loaded.run_dir}")
        if self.state.inference_mode == INFERENCE_MODE_IMAGE:
            self._run_inference()
        else:
            self._refresh_full_scan_ready_state()

    def _update_mode_ui(self) -> None:
        mode = str(self.mode_combo.currentData())
        self.state.inference_mode = mode
        self.input_stack.setCurrentIndex(0 if mode == INFERENCE_MODE_IMAGE else 1)
        if mode == INFERENCE_MODE_IMAGE:
            if self.state.image_path is not None:
                self._run_inference()
            else:
                self.result_preview.setText("Load an image to predict.")
                self.result_preview.set_source_image_size(None, None)
            self.preview_tabs.setTabText(0, "Image preview")
            self.preview_tabs.setTabText(1, "IPF reference")
            self.preview_tabs.setTabText(2, "IPF-colored EBSD map")
            self.ipf_preview.setText("IPF reference is only available in full .oh5 scan mode.")
            self.ipf_map_preview.setText("IPF-colored EBSD map is only available in full .oh5 scan mode.")
        else:
            self.preview_tabs.setTabText(0, "Predicted phase map")
            self.preview_tabs.setTabText(1, "IPF reference")
            self.preview_tabs.setTabText(2, "IPF-colored EBSD map")
            if self.state.full_scan_result is not None:
                self._refresh_full_scan_preview()
            elif self.state.oh5_path is not None:
                self._load_full_scan_preview()
            else:
                self.result_preview.setText("Select a .oh5 file to preview the IPF-colored EBSD map.")
                self.result_preview.set_source_image_size(None, None)
                self.ipf_preview.setText("Select a .oh5 file to render the scan orientation preview.")
                self.ipf_map_preview.setText("Select a .oh5 file to render the IPF-colored EBSD map.")
            self._refresh_full_scan_ready_state()

    def _set_image_path(self, image_path: str) -> None:
        self.state.image_path = Path(image_path).expanduser().resolve()
        self.drop_label.setText(str(self.state.image_path))
        if self.state.inference_mode == INFERENCE_MODE_IMAGE:
            self._run_inference()

    def _set_oh5_path(self, oh5_path: str) -> None:
        self.state.oh5_path = Path(oh5_path).expanduser().resolve()
        self.oh5_edit.setText(str(self.state.oh5_path))
        self.state.full_scan_result = None
        self.state.full_scan_ipf_image = None
        self.state.full_scan_ipf_map_image = None
        self._full_scan_preview_ready = False
        self._clear_selected_pixel()
        self.result_preview.clear_overlay()
        self.scan_progress.setValue(0)
        self.scan_eta_label.setText("ETA: -")
        self.notes.setPlainText(
            f"Selected .oh5: {self.state.oh5_path}\n"
            "Loading scan preview now. Review the IPF-colored EBSD map, then click Start inference."
        )
        self.ipf_preview.setText("Loading scan orientation preview...")
        self.ipf_map_preview.setText("Loading IPF-colored EBSD map preview...")
        self._append_log("info", f"Selected .oh5 scan: {self.state.oh5_path}")
        if self.state.inference_mode == INFERENCE_MODE_FULL_SCAN:
            self._load_full_scan_preview()
        self._refresh_full_scan_ready_state()

    def _load_full_scan_preview(self) -> None:
        oh5_path = Path(self.oh5_edit.text().strip()).expanduser() if self.oh5_edit.text().strip() else self.state.oh5_path
        if oh5_path is None:
            return
        resolved = oh5_path.resolve()
        self.state.oh5_path = resolved
        self._full_scan_preview_ready = False
        self.result_preview.clear_overlay()
        self.result_preview.setText("Loading IPF-colored EBSD map preview...")
        self.result_preview.set_source_image_size(None, None)
        self.ipf_preview.setText("Loading scan orientation preview...")
        self.ipf_map_preview.setText("Loading IPF-colored EBSD map preview...")
        self.prediction_label.setText("Prediction: preview loaded, inference not started")
        self.status_label.setText(f"Loading scan preview for {resolved.name}...")
        try:
            from .oh5_reader import Oh5ScanReader

            with Oh5ScanReader(resolved) as reader:
                meta = reader.meta()
                if not reader.euler_present:
                    raise ValueError("No Euler angle fields were available in the selected .oh5 file")
                eulers_deg = np.stack(
                    [
                        np.asarray(
                            [
                                float(euler_row["phi1"]),
                                float(euler_row["Phi"]),
                                float(euler_row["phi2"]),
                            ],
                            dtype=np.float64,
                        )
                        for idx in range(meta.total_pixels)
                        for euler_row in [reader.read_euler_row(flat_index=idx, degrees=True)]
                    ],
                    axis=0,
                )
                preview_phase = self.state.loaded_model.class_names[0] if self.state.loaded_model and self.state.loaded_model.class_names else "Cu"
                class_names = [preview_phase]
                preview_indices = np.zeros((meta.total_pixels,), dtype=np.int64)
                ipf_map_preview = render_ipf_colored_scan_map(
                    eulers_deg=eulers_deg,
                    predicted_indices=preview_indices,
                    class_names=class_names,
                    nx=meta.nx,
                    ny=meta.ny,
                )
                ipf_preview = render_ipf_reference_panel(
                    eulers_deg_by_phase={preview_phase: eulers_deg},
                    phase_names=class_names,
                    phase_colors={preview_phase: (0.20, 0.44, 0.88)},
                    title=f"{resolved.stem} scan orientation preview",
                )
        except Exception as exc:
            self.result_preview.setText(f"Preview loading failed: {exc}")
            self.ipf_preview.setText(f"Preview loading failed: {exc}")
            self.ipf_map_preview.setText(f"Preview loading failed: {exc}")
            self.notes.setPlainText(f"Selected .oh5: {resolved}\nPreview loading failed: {exc}")
            self.status_label.setText(f"Preview loading failed: {exc}")
            self._append_log("error", f"Failed to load .oh5 preview for {resolved}: {exc}")
            self._refresh_full_scan_ready_state()
            return

        self.state.full_scan_ipf_image = ipf_preview
        self.state.full_scan_ipf_map_image = ipf_map_preview
        self.result_preview.setPixmap(_rgb_array_to_pixmap(ipf_map_preview, target_size=self.result_preview.size()))
        self.result_preview.set_source_image_size(meta.nx, meta.ny)
        self.ipf_preview.setPixmap(_rgb_array_to_pixmap(ipf_preview, target_size=self.ipf_preview.size()))
        self.ipf_map_preview.setPixmap(_rgb_array_to_pixmap(ipf_map_preview, target_size=self.ipf_map_preview.size()))
        self.notes.setPlainText(
            "\n".join(
                [
                    f"Selected .oh5: {resolved}",
                    f"Grid: {meta.nx} x {meta.ny}",
                    f"Total pixels: {meta.total_pixels}",
                    f"Euler convention: {meta.euler_convention or 'unavailable'}",
                    f"Euler source unit: {meta.euler_unit or 'unavailable'}",
                    "",
                    "Preview ready. Confirm the scan looks correct, then click Start inference.",
                ]
            )
        )
        self.status_label.setText(f"Preview ready for {resolved.name}. Click Start inference to begin processing.")
        self._append_log("info", f"Loaded IPF preview for {resolved}")
        self._full_scan_preview_ready = True
        self._refresh_full_scan_ready_state()

    def _run_inference(self) -> None:
        if self.state.loaded_model is None:
            return
        if self.state.inference_mode == INFERENCE_MODE_IMAGE:
            self._run_image_prediction()
            return
        self._run_full_scan_prediction()

    def _run_image_prediction(self) -> None:
        if self.state.loaded_model is None or self.state.image_path is None:
            return
        try:
            result = predict_image(loaded=self.state.loaded_model, image_path=self.state.image_path)
        except Exception as exc:
            self.status_label.setText(f"Prediction failed: {exc}")
            return

        self.original_preview.setPixmap(
            _gray_array_to_pixmap(result.original_image, target_size=self.original_preview.size())
        )
        self.preprocessed_preview.setPixmap(
            _gray_array_to_pixmap(result.preprocessed_image, target_size=self.preprocessed_preview.size())
        )
        self.result_preview.setPixmap(
            _gray_array_to_pixmap(result.preprocessed_image, target_size=self.result_preview.size())
        )
        self.result_preview.set_source_image_size(result.preprocessed_image.shape[1], result.preprocessed_image.shape[0])
        self.prediction_label.setText(f"Prediction: {result.predicted_phase} ({result.confidence:.4f})")

        items = sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        self.prob_table.clear()
        self.prob_table.setColumnCount(2)
        self.prob_table.setHorizontalHeaderLabels(["Phase", "Probability"])
        self.prob_table.setRowCount(len(items))
        for row_idx, (phase, prob) in enumerate(items):
            self.prob_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(phase))
            self.prob_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(f"{prob:.6f}"))
        self.notes.setPlainText(
            "\n".join(
                [
                    f"Run: {self.state.loaded_model.run_dir}",
                    f"Model: {self.state.loaded_model.model_name}",
                    f"Checkpoint: {self.state.loaded_model.checkpoint_path.name}",
                    f"Image: {result.image_path}",
                    f"Predicted phase: {result.predicted_phase}",
                    f"Confidence: {result.confidence:.6f}",
                ]
            )
        )
        self.scan_progress.setValue(0)
        self.scan_eta_label.setText("ETA: -")
        self._update_known_phase_status()

    def _run_full_scan_prediction(self) -> None:
        if self.state.loaded_model is None:
            return
        oh5_path = Path(self.oh5_edit.text().strip()).expanduser() if self.oh5_edit.text().strip() else self.state.oh5_path
        if oh5_path is None:
            return
        if not self._full_scan_preview_ready:
            self.status_label.setText("Load and review the scan preview first, then click Start inference.")
            return
        if self._full_scan_thread is not None:
            self._append_log("warning", "Full-scan inference is already running.")
            return
        resolved = oh5_path.resolve()
        self.state.oh5_path = resolved
        self.state.full_scan_result = None
        self.state.full_scan_ipf_image = None
        self.state.full_scan_ipf_map_image = None
        self._clear_selected_pixel()
        if self.state.full_scan_ipf_map_image is not None:
            self.result_preview.setPixmap(
                _rgb_array_to_pixmap(self.state.full_scan_ipf_map_image, target_size=self.result_preview.size())
            )
            self.result_preview.set_source_image_size(None, None)
        else:
            self.result_preview.setText("Running full-scan inference...")
            self.result_preview.set_source_image_size(None, None)
        self.notes.setPlainText(
            f"Running full-scan inference for {resolved}...\n"
            "The centered circular indicator shows job completion over the scan preview. "
            "Selected-pixel notes will update after the predicted phase map is available."
        )
        self.status_label.setText(f"Running full-scan inference for {resolved.name}...")
        self.scan_progress.setValue(0)
        self.scan_eta_label.setText("ETA: estimating...")
        self.result_preview.set_overlay_progress(0)
        self._append_log("info", f"Starting full-scan inference for {resolved}")
        self._set_full_scan_busy(True)

        thread = QtCore.QThread(self)
        worker = FullScanWorker(loaded=self.state.loaded_model, oh5_path=resolved)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_full_scan_progress)
        worker.log_message.connect(self._append_log)
        worker.finished.connect(self._on_full_scan_finished)
        worker.failed.connect(self._on_full_scan_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._full_scan_thread = thread
        self._full_scan_worker = worker
        thread.start()

    def _refresh_full_scan_preview(self) -> None:
        if self.state.inference_mode != INFERENCE_MODE_FULL_SCAN:
            return
        result = self.state.full_scan_result
        if result is None:
            return
        rendered = _render_full_scan_phase_map(
            result,
            use_confidence_shading=bool(self.confidence_shading_checkbox.isChecked()),
        )
        rendered = _overlay_selected_pixel(
            rendered,
            x=self.state.selected_pixel_x,
            y=self.state.selected_pixel_y,
        )
        self.result_preview.setPixmap(_rgb_array_to_pixmap(rendered, target_size=self.result_preview.size()))
        self.result_preview.set_source_image_size(result.nx, result.ny)
        if self.state.full_scan_ipf_image is not None:
            self.ipf_preview.setPixmap(
                _rgb_array_to_pixmap(self.state.full_scan_ipf_image, target_size=self.ipf_preview.size())
            )
        if self.state.full_scan_ipf_map_image is not None:
            self.ipf_map_preview.setPixmap(
                _rgb_array_to_pixmap(self.state.full_scan_ipf_map_image, target_size=self.ipf_map_preview.size())
            )
        self._refresh_phase_map_legend(result.class_names)

    def _update_known_phase_status(self) -> None:
        if self.state.loaded_model is None:
            return
        known = self.known_phase_combo.currentText().strip()
        if known == "(unknown)":
            return
        if self.state.inference_mode == INFERENCE_MODE_IMAGE:
            predicted = self.prediction_label.text()
            match = known in predicted
            self.status_label.setText(
                f"Known phase: {known} | Prediction status: {'correct' if match else 'mismatch'}"
            )
            return
        result = self.state.full_scan_result
        if result is None:
            return
        dominant_phase = max(result.phase_counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if result.phase_counts else ""
        match = dominant_phase == known
        self.status_label.setText(
            f"Known phase: {known} | Dominant predicted phase: {dominant_phase or 'n/a'} | "
            f"Status: {'match' if match else 'mismatch'}"
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.state.inference_mode == INFERENCE_MODE_IMAGE and self.result_preview.pixmap() is not None and self.state.image_path is not None:
            self._run_image_prediction()
        elif self.state.inference_mode == INFERENCE_MODE_FULL_SCAN and self.state.full_scan_result is not None:
            self._refresh_full_scan_preview()

    def _set_full_scan_busy(self, busy: bool) -> None:
        self.btn_full_scan.setEnabled((not busy) and self._full_scan_preview_ready)
        self.btn_export.setEnabled((not busy) and self.state.full_scan_result is not None)
        self.btn_oh5.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.mode_combo.setEnabled(not busy)
        self.root_edit.setEnabled(not busy)
        self.confidence_shading_checkbox.setEnabled(not busy)
        self.histogram_normalization_checkbox.setEnabled(not busy)
        self.contrast_stretch_checkbox.setEnabled(not busy)

    def _refresh_full_scan_ready_state(self) -> None:
        if self.state.inference_mode != INFERENCE_MODE_FULL_SCAN:
            return
        can_start = (
            self.state.loaded_model is not None
            and self.state.oh5_path is not None
            and self._full_scan_preview_ready
            and self._full_scan_thread is None
        )
        self.btn_full_scan.setEnabled(can_start)

    def _append_log(self, level: str, message: str) -> None:
        level_name = str(level).upper()
        level_value = getattr(logging, level_name, logging.INFO)
        self.log.log(level_value, message)
        timestamp = QtCore.QDateTime.currentDateTime().toString("HH:mm:ss")
        line = f"[{timestamp}] {level_name}: {message}"
        self.log_output.appendPlainText(line)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def _on_full_scan_progress(self, payload: dict[str, Any]) -> None:
        fraction = float(payload.get("fraction", 0.0))
        value = int(round(np.clip(fraction, 0.0, 1.0) * 100.0))
        self.scan_progress.setValue(value)
        self.result_preview.set_overlay_progress(value)
        processed = int(payload.get("processed", 0))
        total = int(payload.get("total", 0))
        stage = str(payload.get("stage", "infer"))
        eta_seconds = payload.get("eta_seconds")
        elapsed_seconds = float(payload.get("elapsed_seconds", 0.0))
        self.scan_eta_label.setText(
            f"ETA: {_format_duration(None if eta_seconds is None else float(eta_seconds))} | "
            f"elapsed: {_format_duration(elapsed_seconds)}"
        )
        self.status_label.setText(f"{stage}: {processed}/{total} pixels processed")

    def _on_full_scan_finished(
        self,
        result: FullScanInferenceResult,
        ipf_image: np.ndarray | None,
        ipf_map_image: np.ndarray | None,
    ) -> None:
        self._full_scan_thread = None
        self._full_scan_worker = None
        self.state.full_scan_result = result
        self.state.full_scan_ipf_image = ipf_image
        self.state.full_scan_ipf_map_image = ipf_map_image
        self._set_full_scan_busy(False)
        self._clear_selected_pixel()
        dominant_phase = max(result.phase_counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if result.phase_counts else "-"
        self.prediction_label.setText(
            f"Full scan: {dominant_phase} dominant | mean confidence {result.mean_confidence:.4f}"
        )
        items = sorted(result.phase_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.prob_table.clear()
        self.prob_table.setColumnCount(4)
        self.prob_table.setHorizontalHeaderLabels(["Phase", "Pixels", "Fraction", "Mean score"])
        self.prob_table.setRowCount(len(items))
        for row_idx, (phase, count) in enumerate(items):
            class_idx = result.class_names.index(phase)
            class_mask = result.predicted_indices == class_idx
            mean_score = float(np.nanmean(result.confidences[class_mask])) if np.any(class_mask) else 0.0
            self.prob_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(phase))
            self.prob_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(str(int(count))))
            self.prob_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(f"{result.phase_fractions[phase]:.4f}"))
            self.prob_table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(f"{mean_score:.4f}"))

        self._refresh_full_scan_notes()
        self.status_label.setText(f"Full-scan inference complete for {result.oh5_path.name}")
        self.scan_progress.setValue(100)
        self.result_preview.set_overlay_progress(100)
        self.scan_eta_label.setText("ETA: 00:00 | elapsed: complete")
        if ipf_image is None:
            self.ipf_preview.setText("No Euler angle fields were available in the scan, so no IPF reference could be rendered.")
        if ipf_map_image is None:
            self.ipf_map_preview.setText(
                "No Euler angle fields or no compatible phase symmetry mapping were available, "
                "so no IPF-colored EBSD map could be rendered."
            )
        self._refresh_full_scan_preview()
        self.result_preview.clear_overlay()
        self._update_known_phase_status()
        self._refresh_full_scan_ready_state()

    def _on_full_scan_failed(self, message: str) -> None:
        self._full_scan_thread = None
        self._full_scan_worker = None
        self._set_full_scan_busy(False)
        self.result_preview.clear_overlay()
        self.status_label.setText(f"Full-scan inference failed: {message}")
        self.scan_eta_label.setText("ETA: -")
        self._append_log("error", message)
        self._refresh_full_scan_ready_state()

    def _refresh_phase_map_legend(self, class_names: list[str]) -> None:
        while self.map_legend_layout.count():
            item = self.map_legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not class_names:
            return

        palette = _phase_color_map(class_names)
        self.map_legend_layout.addStretch(1)
        for phase_name in class_names:
            entry = QtWidgets.QWidget()
            entry_layout = QtWidgets.QVBoxLayout(entry)
            entry_layout.setContentsMargins(0, 0, 0, 0)
            entry_layout.setSpacing(4)

            bar = QtWidgets.QFrame()
            bar.setFixedSize(80, 16)
            rgb = tuple(int(round(float(v) * 255.0)) for v in palette[phase_name])
            bar.setStyleSheet(
                "QFrame {"
                f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});"
                "border: 1px solid rgb(210, 210, 210);"
                "border-radius: 2px;"
                "}"
            )
            label = QtWidgets.QLabel(phase_name)
            label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

            entry_layout.addWidget(bar, alignment=QtCore.Qt.AlignHCenter)
            entry_layout.addWidget(label, alignment=QtCore.Qt.AlignHCenter)
            self.map_legend_layout.addWidget(entry, stretch=0)
        self.map_legend_layout.addStretch(1)

    def _export_full_scan_results(self) -> None:
        result = self.state.full_scan_result
        loaded = self.state.loaded_model
        if self.state.inference_mode != INFERENCE_MODE_FULL_SCAN or result is None or loaded is None:
            self.status_label.setText("Run full-scan inference before exporting results.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_dir = loaded.run_dir / "gui_full_scan_exports" / f"{result.oh5_path.stem}_{timestamp}"
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select export directory",
            str(default_dir.parent),
        )
        export_dir = default_dir if not selected else Path(selected).expanduser().resolve()
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = export_full_scan_artifacts(
                repo_root=self.repo_root,
                loaded=loaded,
                result=result,
                output_dir=export_dir,
                predicted_map_image=_render_full_scan_phase_map(
                    result,
                    use_confidence_shading=bool(self.confidence_shading_checkbox.isChecked()),
                ),
                ipf_reference_image=self.state.full_scan_ipf_image,
                ipf_colored_map_image=self.state.full_scan_ipf_map_image,
                use_confidence_shading=bool(self.confidence_shading_checkbox.isChecked()),
            )
        except Exception as exc:
            self._append_log("error", f"Full-scan export failed: {exc}")
            self.status_label.setText(f"Export failed: {exc}")
            return

        self._append_log("info", f"Exported full-scan artifacts to {manifest_path.parent}")
        self.status_label.setText(f"Exported full-scan artifacts to {manifest_path.parent}")

    def _reset_pattern_display_controls(self) -> None:
        self.histogram_normalization_checkbox.blockSignals(True)
        self.contrast_stretch_checkbox.blockSignals(True)
        self.histogram_normalization_checkbox.setChecked(False)
        self.contrast_stretch_checkbox.setChecked(False)
        self.histogram_normalization_checkbox.blockSignals(False)
        self.contrast_stretch_checkbox.blockSignals(False)
        self._refresh_selected_pattern_preview()

    def _clear_selected_pixel(self) -> None:
        self.state.selected_pixel_x = None
        self.state.selected_pixel_y = None
        self.state.selected_pattern = None
        self.state.selected_processed_pattern = None
        self.state.selected_pixel_details = None
        self.selected_pixel_status.setText("Click a pixel in the predicted phase map to inspect its Kikuchi pattern.")
        self.selected_pixel_value.setText("-")
        self.selected_phase_value.setText("-")
        self.selected_confidence_value.setText("-")
        self.selected_euler_value.setText("-")
        self.selected_ci_value.setText("-")
        self.selected_iq_value.setText("-")
        self.selected_fit_value.setText("-")
        self.selected_valid_value.setText("-")
        self.pattern_compare.clear("No pixel selected")
        self._refresh_full_scan_notes()

    def _handle_phase_map_click(self, x: int, y: int) -> None:
        result = self.state.full_scan_result
        if self.state.inference_mode != INFERENCE_MODE_FULL_SCAN or result is None:
            return
        flat_index = y * result.nx + x
        if flat_index < 0 or flat_index >= result.header_total_pixels:
            return
        if flat_index >= result.total_pixels or int(result.predicted_indices[flat_index]) < 0:
            self.state.selected_pixel_x = x
            self.state.selected_pixel_y = y
            self.state.selected_pattern = None
            self.state.selected_processed_pattern = None
            self.state.selected_pixel_details = {
                "x": int(x),
                "y": int(y),
                "phase": "unavailable",
                "confidence": None,
                "euler": None,
                "ci": None,
                "iq": None,
                "fit": None,
                "valid": None,
            }
            self.selected_pixel_status.setText("Selected grid cell has no pattern payload in the .oh5 file.")
            self.selected_pixel_value.setText(f"({x}, {y})")
            self.selected_phase_value.setText("unavailable")
            self.selected_confidence_value.setText("-")
            self.selected_euler_value.setText("-")
            self.selected_ci_value.setText("-")
            self.selected_iq_value.setText("-")
            self.selected_fit_value.setText("-")
            self.selected_valid_value.setText("-")
            self.pattern_compare.clear("No pattern available")
            self._refresh_full_scan_preview()
            self._refresh_full_scan_notes()
            return

        from .oh5_reader import Oh5ScanReader

        try:
            with Oh5ScanReader(result.oh5_path) as reader:
                pattern = reader.read_pattern(flat_index=flat_index)
                quality_row = reader.read_quality_row(flat_index=flat_index)
                euler_row = reader.read_euler_row(flat_index=flat_index, degrees=True) if reader.euler_present else None
            prediction = predict_pattern_array(loaded=self.state.loaded_model, pattern=pattern)
        except Exception as exc:
            self._append_log("error", f"Failed to load Kikuchi pattern for pixel ({x}, {y}): {exc}")
            self.selected_pixel_status.setText(f"Failed to load clicked pixel pattern: {exc}")
            return

        class_idx = int(result.predicted_indices[flat_index])
        phase = result.class_names[class_idx]
        confidence = float(result.confidences[flat_index])
        ci = quality_row.get("confidence_index")
        iq = quality_row.get("image_quality")
        fit = quality_row.get("fit")
        valid = quality_row.get("valid")
        euler_text = (
            f"({float(euler_row['phi1']):.3f}, {float(euler_row['Phi']):.3f}, {float(euler_row['phi2']):.3f})"
            if euler_row is not None
            else "unavailable"
        )

        self.state.selected_pixel_x = int(x)
        self.state.selected_pixel_y = int(y)
        self.state.selected_pattern = np.asarray(pattern, dtype=np.float32)
        self.state.selected_processed_pattern = np.asarray(prediction.preprocessed_image, dtype=np.float32)
        self.state.selected_pixel_details = {
            "x": int(x),
            "y": int(y),
            "phase": phase,
            "confidence": confidence,
            "euler": None if euler_row is None else {k: float(v) for k, v in euler_row.items()},
            "ci": None if ci is None else float(ci),
            "iq": None if iq is None else float(iq),
            "fit": None if fit is None else float(fit),
            "valid": None if valid is None else bool(valid),
        }
        self.selected_pixel_status.setText("Showing experimental Kikuchi pattern for the selected map pixel.")
        self.selected_pixel_value.setText(f"({x}, {y})")
        self.selected_phase_value.setText(phase)
        self.selected_confidence_value.setText(f"{confidence:.4f}")
        self.selected_euler_value.setText(euler_text)
        self.selected_ci_value.setText("-" if ci is None else f"{float(ci):.4f}")
        self.selected_iq_value.setText("-" if iq is None else f"{float(iq):.4f}")
        self.selected_fit_value.setText("-" if fit is None else f"{float(fit):.4f}")
        self.selected_valid_value.setText("-" if valid is None else ("True" if bool(valid) else "False"))
        display = _prepare_display_gray(
            self.state.selected_processed_pattern,
            histogram_normalization=bool(self.histogram_normalization_checkbox.isChecked()),
            contrast_stretch=bool(self.contrast_stretch_checkbox.isChecked()),
        )
        self.pattern_compare.set_patterns(self.state.selected_pattern, display)
        self._refresh_selected_pattern_preview()
        self._refresh_full_scan_preview()
        self._refresh_full_scan_notes()

    def _refresh_selected_pattern_preview(self) -> None:
        raw_pattern = self.state.selected_pattern
        processed_pattern = self.state.selected_processed_pattern
        if raw_pattern is None or processed_pattern is None:
            return
        display = _prepare_display_gray(
            processed_pattern,
            histogram_normalization=bool(self.histogram_normalization_checkbox.isChecked()),
            contrast_stretch=bool(self.contrast_stretch_checkbox.isChecked()),
        )
        if not self.pattern_compare.raw_pane.view.has_image():
            self.pattern_compare.set_patterns(raw_pattern, display)
            return
        self.pattern_compare.set_processed_pattern(display)

    def _refresh_full_scan_notes(self) -> None:
        result = self.state.full_scan_result
        if result is None or self.state.loaded_model is None:
            return

        palette = _phase_color_map(result.class_names)
        legend_lines = []
        for phase in result.class_names:
            rgb = tuple(int(round(float(v) * 255.0)) for v in palette[phase])
            legend_lines.append(f"{phase}: rgb{rgb}")

        lines = [
            f"Run: {self.state.loaded_model.run_dir}",
            f"Model: {self.state.loaded_model.model_name}",
            f"Checkpoint: {self.state.loaded_model.checkpoint_path.name}",
            f".oh5: {result.oh5_path}",
            f"Scan: {result.scan_name}",
            f"Grid: {result.nx} x {result.ny}",
            f"Inferred pixels: {result.total_pixels}",
            f"Header grid cells: {result.header_total_pixels}",
            f"Mean confidence: {result.mean_confidence:.6f}",
            f"Euler convention: {result.euler_convention or 'unavailable'}",
            f"Euler source unit: {result.euler_source_unit or 'unavailable'}",
            f"IPF reference: {'available' if self.state.full_scan_ipf_image is not None else 'unavailable'}",
            f"IPF-colored EBSD map: {'available' if self.state.full_scan_ipf_map_image is not None else 'unavailable'}",
            f"Confidence shading: {'on' if self.confidence_shading_checkbox.isChecked() else 'off'}",
        ]
        details = self.state.selected_pixel_details
        if details is not None:
            euler_details = details.get("euler")
            euler_line = (
                "Euler (deg): "
                f"({float(euler_details['phi1']):.3f}, {float(euler_details['Phi']):.3f}, {float(euler_details['phi2']):.3f})"
                if isinstance(euler_details, dict)
                else "Euler (deg): unavailable"
            )
            lines.extend(
                [
                    "",
                    "Selected pixel:",
                    f"Pixel: ({details['x']}, {details['y']})",
                    f"Phase: {details['phase']}",
                    (
                        f"Confidence: {float(details['confidence']):.4f}"
                        if details.get('confidence') is not None
                        else "Confidence: unavailable"
                    ),
                    euler_line,
                    (
                        f"CI: {float(details['ci']):.4f}"
                        if details.get("ci") is not None
                        else "CI: unavailable"
                    ),
                    (
                        f"IQ: {float(details['iq']):.4f}"
                        if details.get("iq") is not None
                        else "IQ: unavailable"
                    ),
                    (
                        f"Fit: {float(details['fit']):.4f}"
                        if details.get("fit") is not None
                        else "Fit: unavailable"
                    ),
                    (
                        f"Valid: {bool(details['valid'])}"
                        if details.get("valid") is not None
                        else "Valid: unavailable"
                    ),
                    (
                        "Pattern display: histogram normalization="
                        f"{'on' if self.histogram_normalization_checkbox.isChecked() else 'off'}, "
                        f"contrast stretch={'on' if self.contrast_stretch_checkbox.isChecked() else 'off'}"
                    ),
                ]
            )
        lines.extend(
            [
                "",
                "Legend:",
                *legend_lines,
                "",
                "IPF note:",
                "Reference IPF panels are built from scan Euler angles grouped by predicted phase.",
                "IPF-colored EBSD map is generated per pixel from Euler angles using IPF color keys.",
            ]
        )
        self.notes.setPlainText("\n".join(lines))


def run_inference_gui(*, repo_root: Path, suite_root: Path | None, debug: bool = False) -> int:
    log = logging.getLogger("ml_inference_gui")

    def _handle_unhandled_exception(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None:
        log.critical("Unhandled exception in GUI", exc_info=(exc_type, exc_value, exc_tb))
        traceback.print_exception(exc_type, exc_value, exc_tb)

    def _qt_message_handler(mode: QtCore.QtMsgType, context: QtCore.QMessageLogContext, message: str) -> None:
        level_map = {
            QtCore.QtDebugMsg: logging.DEBUG,
            QtCore.QtInfoMsg: logging.INFO,
            QtCore.QtWarningMsg: logging.WARNING,
            QtCore.QtCriticalMsg: logging.ERROR,
            QtCore.QtFatalMsg: logging.CRITICAL,
        }
        level_value = level_map.get(mode, logging.INFO)
        source = context.category if context.category else "qt"
        log.log(level_value, "Qt message [%s]: %s", source, message)

    sys.excepthook = _handle_unhandled_exception
    QtCore.qInstallMessageHandler(_qt_message_handler)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = InferenceMainWindow(repo_root=repo_root, initial_root=suite_root, logger=log)
    window.show()
    return int(app.exec())
