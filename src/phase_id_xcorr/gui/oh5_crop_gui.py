"""Dedicated `.oh5` crop and review desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from phase_id_xcorr.io.oh5_crop import (
    CropExportResult,
    CropFieldComparison,
    CropReviewSession,
    CropSpec,
    CropVerificationReport,
    compare_cropped_pixel,
    export_cropped_oh5,
    load_review_session,
    load_scan_visual_data,
)
from phase_id_xcorr.ml.inference_gui import ClickableImageLabel, _PatternCompareWidget


def _gray_array_to_pixmap(array: np.ndarray) -> QtGui.QPixmap:
    arr = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    h, w = arr8.shape
    qimg = QtGui.QImage(arr8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qimg.copy())


def _rgb_array_to_pixmap(array: np.ndarray) -> QtGui.QPixmap:
    arr = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    h, w, _ = arr8.shape
    qimg = QtGui.QImage(arr8.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


class CropPlotWidget(QtWidgets.QWidget):
    cropChanged = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget(background="w")
        self.plot.setAspectLocked(lock=True, ratio=1.0)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.getViewBox().invertY(True)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)
        layout.addWidget(self.plot)

        self._roi = pg.RectROI(
            [0, 0],
            [1, 1],
            pen=pg.mkPen((0, 220, 0), width=2),
            movable=True,
            rotatable=False,
            resizable=True,
            removable=False,
            maxBounds=QtCore.QRectF(0, 0, 1, 1),
            translateSnap=True,
            scaleSnap=True,
        )
        self._roi.addScaleHandle([1, 1], [0, 0])
        self._roi.addScaleHandle([0, 1], [1, 0])
        self._roi.addScaleHandle([1, 0], [0, 1])
        self._roi.addScaleHandle([0, 0], [1, 1])
        self.plot.addItem(self._roi)
        self._roi.sigRegionChanged.connect(self._emit_crop_change)

        self._guard = False
        self._nx = 1
        self._ny = 1

    def set_image(self, array: np.ndarray) -> None:
        image = np.asarray(array, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError("Crop plot expects a 2D grayscale array")
        self._ny, self._nx = image.shape
        self.image_item.setImage(image, autoLevels=True)
        self._roi.maxBounds = QtCore.QRectF(0, 0, float(self._nx), float(self._ny))
        self.plot.setLimits(xMin=0, xMax=self._nx, yMin=0, yMax=self._ny)
        self.plot.setRange(xRange=(0, self._nx), yRange=(0, self._ny), padding=0.0)
        self.set_crop_spec(CropSpec(row=0, column=0, width=self._nx, height=self._ny))

    def set_crop_spec(self, spec: CropSpec) -> None:
        spec = spec.validate_for(nx=self._nx, ny=self._ny)
        self._guard = True
        try:
            self._roi.setPos((float(spec.column), float(spec.row)))
            self._roi.setSize((float(spec.width), float(spec.height)))
        finally:
            self._guard = False
        self.cropChanged.emit(spec)

    def current_crop_spec(self) -> CropSpec:
        pos = self._roi.pos()
        size = self._roi.size()
        left = int(round(float(pos.x())))
        top = int(round(float(pos.y())))
        width = max(1, int(round(float(size.x()))))
        height = max(1, int(round(float(size.y()))))
        return CropSpec(row=top, column=left, width=width, height=height).validate_for(nx=self._nx, ny=self._ny)

    def _emit_crop_change(self) -> None:
        if self._guard:
            return
        spec = self.current_crop_spec()
        self._guard = True
        try:
            self._roi.setPos((float(spec.column), float(spec.row)))
            self._roi.setSize((float(spec.width), float(spec.height)))
        finally:
            self._guard = False
        self.cropChanged.emit(spec)


class OverlayMapLabel(ClickableImageLabel):
    def __init__(self, *, clickable: bool = False, placeholder: str = "No image loaded") -> None:
        super().__init__(placeholder)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setMinimumSize(320, 260)
        self.setWordWrap(True)
        self._base_pixmap: QtGui.QPixmap | None = None
        self._selection_xy: tuple[int, int] | None = None
        self._crop_spec: CropSpec | None = None
        self._placeholder = placeholder
        self._clickable = clickable

    def set_placeholder(self, text: str) -> None:
        self._base_pixmap = None
        self._placeholder = text
        self.set_source_image_size(None, None)
        self.setPixmap(QtGui.QPixmap())
        self.setText(text)

    def set_gray_image(self, array: np.ndarray) -> None:
        pixmap = _gray_array_to_pixmap(array)
        self._set_base_pixmap(pixmap, width=array.shape[1], height=array.shape[0])

    def set_rgb_image(self, array: np.ndarray) -> None:
        pixmap = _rgb_array_to_pixmap(array)
        self._set_base_pixmap(pixmap, width=array.shape[1], height=array.shape[0])

    def set_selection(self, xy: tuple[int, int] | None) -> None:
        self._selection_xy = None if xy is None else (int(xy[0]), int(xy[1]))
        self._refresh_pixmap()

    def set_crop_rect(self, crop_spec: CropSpec | None) -> None:
        self._crop_spec = crop_spec
        self._refresh_pixmap()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._clickable:
            super().mousePressEvent(event)
            return
        event.ignore()

    def _set_base_pixmap(self, pixmap: QtGui.QPixmap, *, width: int, height: int) -> None:
        self._base_pixmap = pixmap
        self.set_source_image_size(width, height)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._base_pixmap is None:
            self.setPixmap(QtGui.QPixmap())
            self.setText(self._placeholder)
            return
        pixmap = QtGui.QPixmap(self._base_pixmap)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if self._crop_spec is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
            painter.drawRect(
                QtCore.QRectF(
                    float(self._crop_spec.column),
                    float(self._crop_spec.row),
                    float(self._crop_spec.width),
                    float(self._crop_spec.height),
                )
            )
        if self._selection_xy is not None:
            x, y = self._selection_xy
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 60, 60), 2))
            painter.drawRect(QtCore.QRectF(float(x) - 1.5, float(y) - 1.5, 3.0, 3.0))
            painter.drawLine(QtCore.QPointF(float(x) - 6.0, float(y)), QtCore.QPointF(float(x) + 6.0, float(y)))
            painter.drawLine(QtCore.QPointF(float(x), float(y) - 6.0), QtCore.QPointF(float(x), float(y) + 6.0))
        painter.end()
        self.setPixmap(pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.setText("")


class PixelInfoGroup(QtWidgets.QGroupBox):
    FIELD_ORDER: tuple[tuple[str, str], ...] = (
        ("coord", "Coord"),
        ("flat_index", "Flat index"),
        ("phase_name", "Phase"),
        ("phase_id", "Phase id"),
        ("image_quality", "IQ"),
        ("confidence_index", "CI"),
        ("fit", "Fit"),
        ("valid", "Valid"),
        ("x_position", "X Position"),
        ("y_position", "Y Position"),
        ("phi1", "phi1 (deg)"),
        ("Phi", "Phi (deg)"),
        ("phi2", "phi2 (deg)"),
    )

    def __init__(self, title: str) -> None:
        super().__init__(title)
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(8, 8, 8, 8)
        self.labels: dict[str, QtWidgets.QLabel] = {}
        for key, label in self.FIELD_ORDER:
            value_label = QtWidgets.QLabel("-")
            value_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            value_label.setWordWrap(True)
            form.addRow(label, value_label)
            self.labels[key] = value_label

    def clear(self) -> None:
        for label in self.labels.values():
            label.setText("-")

    def set_values(self, values: dict[str, str]) -> None:
        for key, label in self.labels.items():
            label.setText(values.get(key, "-"))


@dataclass(slots=True)
class ReviewSelection:
    local_x: int
    local_y: int
    source_x: int
    source_y: int


class Oh5CropMainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        repo_root: Path,
        logger: logging.Logger,
        initial_input_path: Path | None = None,
        initial_output_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.log = logger
        self.initial_output_dir = None if initial_output_dir is None else initial_output_dir.expanduser().resolve()
        self.source_visual = None
        self.review_session: CropReviewSession | None = None
        self.review_selection: ReviewSelection | None = None
        self._output_path_user_edited = False
        self._spin_guard = False

        self.setWindowTitle("OH5 Crop + Review")
        self.resize(1700, 1050)

        self._build_toolbar()
        self._build_central()
        self.statusBar().showMessage("Load a pattern-bearing .oh5 scan to begin cropping.")

        if initial_input_path is not None:
            self.open_source_oh5(initial_input_path)

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)

        self.open_action = QtGui.QAction("Open OH5", self)
        self.open_action.triggered.connect(self._choose_source_file)
        toolbar.addAction(self.open_action)

        toolbar.addSeparator()
        self.crop_mode_action = QtGui.QAction("Crop Mode", self)
        self.crop_mode_action.triggered.connect(self.show_crop_mode)
        toolbar.addAction(self.crop_mode_action)

        self.review_mode_action = QtGui.QAction("Review Mode", self)
        self.review_mode_action.triggered.connect(self.show_review_mode)
        self.review_mode_action.setEnabled(False)
        toolbar.addAction(self.review_mode_action)

        toolbar.addSeparator()
        self.export_action = QtGui.QAction("Export Crop", self)
        self.export_action.triggered.connect(self._export_crop)
        toolbar.addAction(self.export_action)

        self.back_to_crop_action = QtGui.QAction("Back To Crop", self)
        self.back_to_crop_action.triggered.connect(self.show_crop_mode)
        self.back_to_crop_action.setEnabled(False)
        toolbar.addAction(self.back_to_crop_action)

    def _build_central(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.mode_stack = QtWidgets.QStackedWidget()
        self.crop_page = self._build_crop_page()
        self.review_page = self._build_review_page()
        self.mode_stack.addWidget(self.crop_page)
        self.mode_stack.addWidget(self.review_page)

        root.addWidget(self.mode_stack, stretch=1)
        progress_row = QtWidgets.QHBoxLayout()
        self.progress_label = QtWidgets.QLabel("Idle")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_row.addWidget(self.progress_label, stretch=0)
        progress_row.addWidget(self.progress_bar, stretch=1)
        root.addLayout(progress_row, stretch=0)
        self.log_output = QtWidgets.QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(1500)
        self.log_output.setMaximumHeight(160)
        root.addWidget(self.log_output, stretch=0)
        self.setCentralWidget(central)

    def _build_crop_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.crop_source_size_label = QtWidgets.QLabel("Original scan size: -")
        left_layout.addWidget(self.crop_source_size_label)
        self.crop_plot = CropPlotWidget()
        self.crop_plot.cropChanged.connect(self._sync_crop_spinboxes_from_roi)
        left_layout.addWidget(self.crop_plot, stretch=1)
        self.crop_footer = QtWidgets.QLabel("No scan loaded")
        left_layout.addWidget(self.crop_footer)
        splitter.addWidget(left)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(8, 4, 8, 4)
        controls_layout.setSpacing(10)

        self.source_summary = QtWidgets.QLabel("Load one pattern-bearing .oh5 file with IQ and Pattern datasets.")
        self.source_summary.setWordWrap(True)
        controls_layout.addWidget(self.source_summary)

        rect_group = QtWidgets.QGroupBox("Rectangle")
        rect_form = QtWidgets.QFormLayout(rect_group)
        self.row_spin = QtWidgets.QSpinBox()
        self.col_spin = QtWidgets.QSpinBox()
        self.width_spin = QtWidgets.QSpinBox()
        self.height_spin = QtWidgets.QSpinBox()
        for spin in (self.row_spin, self.col_spin, self.width_spin, self.height_spin):
            spin.setRange(0, 1)
            spin.valueChanged.connect(self._apply_spinbox_crop)
        self.width_spin.setMinimum(1)
        self.height_spin.setMinimum(1)
        self.right_label = QtWidgets.QLabel("0")
        self.bottom_label = QtWidgets.QLabel("0")
        rect_form.addRow("Row", self.row_spin)
        rect_form.addRow("Column", self.col_spin)
        rect_form.addRow("Width", self.width_spin)
        rect_form.addRow("Height", self.height_spin)
        rect_form.addRow("Right", self.right_label)
        rect_form.addRow("Bottom", self.bottom_label)
        controls_layout.addWidget(rect_group)

        export_group = QtWidgets.QGroupBox("Export")
        export_layout = QtWidgets.QVBoxLayout(export_group)
        self.include_patterns_checkbox = QtWidgets.QCheckBox("Include Patterns")
        self.include_patterns_checkbox.setChecked(True)
        self.include_patterns_checkbox.setEnabled(False)
        self.include_patterns_checkbox.setToolTip("V1 always writes cropped pattern payloads.")
        export_layout.addWidget(self.include_patterns_checkbox)
        self.output_path_edit = QtWidgets.QLineEdit()
        self.output_path_edit.textEdited.connect(self._mark_output_path_user_edited)
        browse_row = QtWidgets.QHBoxLayout()
        browse_row.addWidget(self.output_path_edit, stretch=1)
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self._choose_output_path)
        browse_row.addWidget(browse_button)
        export_layout.addLayout(browse_row)
        self.export_button = QtWidgets.QPushButton("Export Crop")
        self.export_button.clicked.connect(self._export_crop)
        export_layout.addWidget(self.export_button)
        controls_layout.addWidget(export_group)
        controls_layout.addStretch(1)
        splitter.addWidget(controls)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)
        return page

    def _build_review_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.review_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.review_tabs, stretch=1)

        iq_tab = QtWidgets.QWidget()
        iq_layout = QtWidgets.QHBoxLayout(iq_tab)
        self.original_iq_label = OverlayMapLabel(clickable=False, placeholder="Original IQ map unavailable")
        self.cropped_iq_label = OverlayMapLabel(clickable=True, placeholder="Cropped IQ map unavailable")
        self.cropped_iq_label.imageClicked.connect(self._handle_review_click)
        self.original_iq_size_label = QtWidgets.QLabel("Original scan size: -")
        self.cropped_iq_size_label = QtWidgets.QLabel("Cropped scan size: -")
        iq_layout.addWidget(self._wrap_map_group("Original IQ Map", self.original_iq_size_label, self.original_iq_label), stretch=1)
        iq_layout.addWidget(self._wrap_map_group("Cropped IQ Map", self.cropped_iq_size_label, self.cropped_iq_label), stretch=1)
        self.review_tabs.addTab(iq_tab, "IQ Maps")

        ipf_tab = QtWidgets.QWidget()
        ipf_layout = QtWidgets.QHBoxLayout(ipf_tab)
        self.original_ipf_label = OverlayMapLabel(clickable=False, placeholder="IPF unavailable")
        self.cropped_ipf_label = OverlayMapLabel(clickable=True, placeholder="IPF unavailable")
        self.cropped_ipf_label.imageClicked.connect(self._handle_review_click)
        self.original_ipf_size_label = QtWidgets.QLabel("Original scan size: -")
        self.cropped_ipf_size_label = QtWidgets.QLabel("Cropped scan size: -")
        ipf_layout.addWidget(self._wrap_map_group("Original IPF Map", self.original_ipf_size_label, self.original_ipf_label), stretch=1)
        ipf_layout.addWidget(self._wrap_map_group("Cropped IPF Map", self.cropped_ipf_size_label, self.cropped_ipf_label), stretch=1)
        self.review_tabs.addTab(ipf_tab, "IPF Maps")

        pattern_tab = QtWidgets.QWidget()
        pattern_layout = QtWidgets.QVBoxLayout(pattern_tab)
        pattern_top = QtWidgets.QHBoxLayout()
        self.original_info_group = PixelInfoGroup("Original Pixel Data")
        self.cropped_info_group = PixelInfoGroup("Cropped Pixel Data")
        self.mapping_group = PixelInfoGroup("Selection Mapping")
        pattern_top.addWidget(self.original_info_group, stretch=1)
        pattern_top.addWidget(self.cropped_info_group, stretch=1)
        pattern_top.addWidget(self.mapping_group, stretch=1)
        pattern_layout.addLayout(pattern_top, stretch=0)
        self.pattern_compare = _PatternCompareWidget()
        self.pattern_compare.raw_pane.group.setTitle("Original Kikuchi Pattern")
        self.pattern_compare.processed_pane.group.setTitle("Cropped Kikuchi Pattern")
        pattern_layout.addWidget(self.pattern_compare, stretch=1)
        self.review_tabs.addTab(pattern_tab, "Patterns + Pixel Data")

        audit_tab = QtWidgets.QWidget()
        audit_layout = QtWidgets.QVBoxLayout(audit_tab)
        self.audit_summary_label = QtWidgets.QLabel("Metadata audit unavailable")
        self.audit_summary_label.setWordWrap(True)
        audit_layout.addWidget(self.audit_summary_label)
        self.audit_tabs = QtWidgets.QTabWidget()
        self.changed_fields_text = QtWidgets.QPlainTextEdit()
        self.changed_fields_text.setReadOnly(True)
        self.unchanged_fields_text = QtWidgets.QPlainTextEdit()
        self.unchanged_fields_text.setReadOnly(True)
        self.audit_tabs.addTab(self.changed_fields_text, "Changed Fields")
        self.audit_tabs.addTab(self.unchanged_fields_text, "Unchanged Fields")
        audit_layout.addWidget(self.audit_tabs, stretch=1)
        self.review_tabs.addTab(audit_tab, "Metadata Audit")

        return page

    def _wrap_map_group(self, title: str, size_label: QtWidgets.QLabel, widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox(title)
        group_layout = QtWidgets.QVBoxLayout(group)
        size_label.setStyleSheet("color: rgb(80, 80, 80);")
        group_layout.addWidget(size_label)
        group_layout.addWidget(widget)
        return group

    def _append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())
        self.log.info(message)

    def _set_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, int(value))))
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)
        QtWidgets.QApplication.processEvents()

    def show_crop_mode(self) -> None:
        self.mode_stack.setCurrentWidget(self.crop_page)
        self.back_to_crop_action.setEnabled(False)
        self._set_progress(100, "Crop mode ready")

    def show_review_mode(self) -> None:
        if self.review_session is None:
            return
        self.mode_stack.setCurrentWidget(self.review_page)
        self.back_to_crop_action.setEnabled(True)
        self._set_progress(100, "Review mode ready")

    def open_source_oh5(self, path: Path) -> None:
        self._set_progress(5, f"Loading source scan {Path(path).name} ...")
        self._append_log(f"Opening source .oh5: {Path(path).resolve()}")
        try:
            visual = load_scan_visual_data(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open OH5 Failed", str(exc))
            self._append_log(f"Open failed: {exc}")
            self._set_progress(0, f"Open failed: {exc}")
            return
        self._set_progress(40, "Source scan loaded; preparing crop view ...")
        self.source_visual = visual
        self.crop_plot.set_image(visual.iq_map)
        self.row_spin.setRange(0, max(0, visual.ny - 1))
        self.col_spin.setRange(0, max(0, visual.nx - 1))
        self.width_spin.setRange(1, visual.nx)
        self.height_spin.setRange(1, visual.ny)
        self._sync_crop_spinboxes_from_roi(CropSpec(row=0, column=0, width=visual.nx, height=visual.ny))
        self.crop_source_size_label.setText(f"Original scan size: {visual.ny} rows x {visual.nx} columns")
        self.source_summary.setText(
            f"Scan: {visual.scan_name}\nPath: {visual.path}\nGrid: {visual.ny} rows x {visual.nx} columns\n"
            f"IQ field: {visual.iq_field_name} | Euler: {'present' if visual.euler_present else 'missing'}"
        )
        self.review_session = None
        self.review_mode_action.setEnabled(False)
        self._output_path_user_edited = False
        self._refresh_default_output_path()
        self._append_log(
            f"Loaded source scan '{visual.scan_name}' with original size {visual.ny}x{visual.nx}; IQ field={visual.iq_field_name}"
        )
        self._set_progress(100, "Source scan ready for crop selection")
        self.show_crop_mode()

    def open_review_from_export(self, export_result: CropExportResult) -> None:
        self._set_progress(70, f"Reloading source and cropped scans for review: {export_result.output_path.name}")
        self._append_log(f"Opening review session for exported crop: {export_result.output_path}")
        session = load_review_session(export_result)
        self._set_progress(85, "Review data loaded; rendering comparison panes ...")
        self.review_session = session
        self.review_mode_action.setEnabled(True)
        self.original_iq_size_label.setText(f"Original scan size: {session.source.ny} rows x {session.source.nx} columns")
        self.cropped_iq_size_label.setText(f"Cropped scan size: {session.cropped.ny} rows x {session.cropped.nx} columns")
        self.original_ipf_size_label.setText(f"Original scan size: {session.source.ny} rows x {session.source.nx} columns")
        self.cropped_ipf_size_label.setText(f"Cropped scan size: {session.cropped.ny} rows x {session.cropped.nx} columns")

        self.original_iq_label.set_gray_image(session.source.iq_map)
        self.original_iq_label.set_crop_rect(session.export.crop_spec)
        self.cropped_iq_label.set_gray_image(session.cropped.iq_map)
        self.cropped_iq_label.set_crop_rect(None)

        if session.source.ipf_map is None or session.cropped.ipf_map is None:
            message = "IPF unavailable: Euler fields or phase metadata were not present in a renderable form."
            self.original_ipf_label.set_placeholder(message)
            self.cropped_ipf_label.set_placeholder(message)
        else:
            self.original_ipf_label.set_rgb_image(session.source.ipf_map)
            self.original_ipf_label.set_crop_rect(session.export.crop_spec)
            self.cropped_ipf_label.set_rgb_image(session.cropped.ipf_map)
            self.cropped_ipf_label.set_crop_rect(None)

        self.original_info_group.clear()
        self.cropped_info_group.clear()
        self.mapping_group.clear()
        self._populate_audit_tab(session.verification_report)
        self.pattern_compare.clear("Click a pixel in the cropped IQ/IPF map to compare source and cropped data.")
        self.review_selection = None
        self._append_log(
            f"Loaded review session for {session.export.output_path.name}; original size={session.source.ny}x{session.source.nx}, cropped size={session.cropped.ny}x{session.cropped.nx}"
        )
        self._set_progress(95, "Comparison panes ready; selecting first cropped pixel ...")
        self._handle_review_click(0, 0)
        self.show_review_mode()

    def _mark_output_path_user_edited(self) -> None:
        self._output_path_user_edited = True

    def _refresh_default_output_path(self) -> None:
        if self.source_visual is None or self._output_path_user_edited:
            return
        spec = self.crop_plot.current_crop_spec()
        directory = self.initial_output_dir or self.source_visual.path.parent
        filename = (
            f"{self.source_visual.path.stem}_crop_r{spec.row}_c{spec.column}_h{spec.height}_w{spec.width}.oh5"
        )
        self.output_path_edit.setText(str((directory / filename).resolve()))

    def _sync_crop_spinboxes_from_roi(self, spec_obj: object) -> None:
        spec = spec_obj if isinstance(spec_obj, CropSpec) else self.crop_plot.current_crop_spec()
        if self.source_visual is None:
            return
        self._spin_guard = True
        try:
            self.row_spin.setValue(spec.row)
            self.col_spin.setValue(spec.column)
            self.width_spin.setValue(spec.width)
            self.height_spin.setValue(spec.height)
            self.right_label.setText(str(spec.right))
            self.bottom_label.setText(str(spec.bottom))
            self.crop_footer.setText(
                f"Crop size = {spec.height} x {spec.width} px | retained pixels = {spec.width * spec.height} | "
                f"source grid = {self.source_visual.ny} x {self.source_visual.nx}"
            )
        finally:
            self._spin_guard = False
        self._refresh_default_output_path()

    def _apply_spinbox_crop(self) -> None:
        if self._spin_guard or self.source_visual is None:
            return
        spec = CropSpec(
            row=int(self.row_spin.value()),
            column=int(self.col_spin.value()),
            width=int(self.width_spin.value()),
            height=int(self.height_spin.value()),
        )
        try:
            spec = spec.validate_for(nx=self.source_visual.nx, ny=self.source_visual.ny)
        except Exception:
            return
        self.crop_plot.set_crop_spec(spec)

    def _choose_source_file(self) -> None:
        start_dir = str(self.source_visual.path.parent if self.source_visual is not None else self.repo_root)
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open OH5 scan",
            start_dir,
            "OH5 scans (*.oh5)",
        )
        if not selected:
            return
        self.open_source_oh5(Path(selected))

    def _choose_output_path(self) -> None:
        start = self.output_path_edit.text().strip() or str(self.repo_root / "cropped_scan.oh5")
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export cropped OH5",
            start,
            "OH5 scans (*.oh5)",
        )
        if not selected:
            return
        self._output_path_user_edited = True
        self.output_path_edit.setText(selected)

    def _export_crop(self) -> None:
        if self.source_visual is None:
            QtWidgets.QMessageBox.warning(self, "No scan loaded", "Load a source .oh5 file before exporting a crop.")
            return
        output_text = self.output_path_edit.text().strip()
        if not output_text:
            QtWidgets.QMessageBox.warning(self, "No output path", "Choose an output .oh5 path first.")
            return
        spec = self.crop_plot.current_crop_spec()
        self._append_log(
            f"Writing cropped .oh5 to {Path(output_text).resolve()} from crop row={spec.row} col={spec.column} width={spec.width} height={spec.height}"
        )
        self._set_progress(45, "Writing cropped .oh5 to disk ...")
        try:
            export_result = export_cropped_oh5(
                source_path=self.source_visual.path,
                crop_spec=spec,
                output_path=Path(output_text),
                repo_root=self.repo_root,
                logger=self.log,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Crop Export Failed", str(exc))
            self._append_log(f"Crop export failed: {exc}")
            self._set_progress(0, f"Export failed: {exc}")
            return
        self._append_log(f"Crop export complete: {export_result.output_path}")
        self._set_progress(60, f"Cropped .oh5 written: {export_result.output_path.name}")
        self.open_review_from_export(export_result)

    def _handle_review_click(self, x: int, y: int) -> None:
        if self.review_session is None:
            return
        try:
            source_record, cropped_record = compare_cropped_pixel(
                source_path=self.review_session.source.path,
                cropped_path=self.review_session.cropped.path,
                crop_spec=self.review_session.export.crop_spec,
                local_x=int(x),
                local_y=int(y),
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Pixel compare failed: {exc}")
            return

        self.review_selection = ReviewSelection(
            local_x=int(x),
            local_y=int(y),
            source_x=int(source_record.x),
            source_y=int(source_record.y),
        )
        self.original_iq_label.set_selection((source_record.x, source_record.y))
        self.cropped_iq_label.set_selection((cropped_record.x, cropped_record.y))
        self.original_ipf_label.set_selection((source_record.x, source_record.y))
        self.cropped_ipf_label.set_selection((cropped_record.x, cropped_record.y))
        self.original_info_group.set_values(self._record_to_display_values(source_record))
        self.cropped_info_group.set_values(self._record_to_display_values(cropped_record))
        self.mapping_group.set_values(
            {
                "coord": f"local=({cropped_record.x}, {cropped_record.y})",
                "flat_index": f"source=({source_record.x}, {source_record.y})",
                "phase_name": f"crop origin row={self.review_session.export.crop_spec.row}",
                "phase_id": f"crop origin col={self.review_session.export.crop_spec.column}",
                "image_quality": self.review_session.export.output_path.name,
                "confidence_index": self.review_session.export.source_path.name,
                "fit": self.review_session.export.pattern_key,
                "valid": self.review_session.export.scan_name,
            }
        )
        self.pattern_compare.set_patterns(source_record.pattern, cropped_record.pattern)
        self._append_log(
            f"Selected cropped pixel ({cropped_record.x}, {cropped_record.y}) mapped to original ({source_record.x}, {source_record.y})"
        )
        self._set_progress(100, f"Selected cropped pixel ({cropped_record.x}, {cropped_record.y})")

    def _populate_audit_tab(self, report: CropVerificationReport) -> None:
        self.audit_summary_label.setText(
            "Verification checks passed. "
            f"Dataset paths: {report.source_dataset_count} source / {report.cropped_dataset_count} cropped. "
            f"Group paths: {report.source_group_count} source / {report.cropped_group_count} cropped. "
            f"Changed fields: {len(report.changed_fields)}. "
            f"Unchanged fields: {len(report.unchanged_fields)}."
        )
        self.changed_fields_text.setPlainText(self._format_comparison_items(report.changed_fields))
        self.unchanged_fields_text.setPlainText(self._format_comparison_items(report.unchanged_fields))

    def _format_comparison_items(self, items: list[CropFieldComparison]) -> str:
        if not items:
            return "None"
        lines: list[str] = []
        for item in items:
            lines.extend(
                [
                    f"Path: {item.path}",
                    f"Status: {item.status}",
                    f"Source: {item.source_summary}",
                    f"Cropped: {item.cropped_summary}",
                    f"Note: {item.note}",
                    "",
                ]
            )
        return "\n".join(lines).rstrip()

    def _record_to_display_values(self, record: object) -> dict[str, str]:
        from phase_id_xcorr.io.oh5_crop import PixelInspectionRecord

        if not isinstance(record, PixelInspectionRecord):
            return {}
        scalars = record.scalar_values
        euler = record.euler_row_deg or {}
        return {
            "coord": f"({record.x}, {record.y})",
            "flat_index": str(record.flat_index),
            "phase_name": str(record.phase_name or "-"),
            "phase_id": "-" if record.phase_id is None else str(record.phase_id),
            "image_quality": self._format_scalar(record.quality_row.get("image_quality")),
            "confidence_index": self._format_scalar(record.quality_row.get("confidence_index")),
            "fit": self._format_scalar(record.quality_row.get("fit")),
            "valid": self._format_scalar(record.quality_row.get("valid")),
            "x_position": self._format_scalar(scalars.get("X Position")),
            "y_position": self._format_scalar(scalars.get("Y Position")),
            "phi1": self._format_scalar(euler.get("phi1")),
            "Phi": self._format_scalar(euler.get("Phi")),
            "phi2": self._format_scalar(euler.get("phi2")),
        }

    @staticmethod
    def _format_scalar(value: object) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)


def run_oh5_crop_gui(
    *,
    repo_root: Path,
    debug: bool = False,
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> int:
    log = logging.getLogger("oh5_crop_gui")
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
    window = Oh5CropMainWindow(
        repo_root=repo_root,
        logger=log,
        initial_input_path=input_path,
        initial_output_dir=output_dir,
    )
    window.show()
    return int(app.exec())
