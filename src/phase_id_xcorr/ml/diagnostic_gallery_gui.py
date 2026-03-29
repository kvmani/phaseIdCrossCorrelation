"""PySide6/PyQtGraph diagnostic gallery GUI for `.oh5` pattern inspection."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .config import load_yaml
from .dataset_io import write_json
from .diagnostic_gallery import (
    DiagnosticGallerySession,
    DiagnosticPatternRecord,
    add_manual_record,
    build_diagnostic_gallery_session_from_config,
    export_diagnostic_gallery_artifacts,
)


def _array_to_pixmap(array: np.ndarray, *, target_size: QtCore.QSize) -> QtGui.QPixmap:
    arr = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    h, w = arr8.shape
    qimg = QtGui.QImage(arr8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    pix = QtGui.QPixmap.fromImage(qimg.copy())
    return pix.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


def _fallback_size(widget: QtWidgets.QWidget, default: QtCore.QSize) -> QtCore.QSize:
    size = widget.size()
    if size.width() <= 1 or size.height() <= 1:
        return default
    return size


class FileDropListWidget(QtWidgets.QListWidget):
    filesDropped = QtCore.Signal(list)

    def __init__(self, placeholder: str):
        super().__init__()
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(True)
        self.setMinimumHeight(120)
        self.setToolTip("Drop `.oh5` files here or use Add")
        self.setStyleSheet("QListWidget { background: white; }")
        self.add_placeholder(placeholder)

    def add_placeholder(self, text: str) -> None:
        if self.count() == 0:
            item = QtWidgets.QListWidgetItem(text)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            item.setForeground(QtGui.QBrush(QtGui.QColor("#666666")))
            item.setData(QtCore.Qt.UserRole, None)
            self.addItem(item)

    def clear_placeholder(self) -> None:
        for idx in range(self.count() - 1, -1, -1):
            item = self.item(idx)
            if item.data(QtCore.Qt.UserRole) is None:
                self.takeItem(idx)

    def add_paths(self, paths: list[str]) -> None:
        self.clear_placeholder()
        for path in paths:
            existing = {self.item(i).data(QtCore.Qt.UserRole) for i in range(self.count())}
            if path in existing:
                continue
            item = QtWidgets.QListWidgetItem(Path(path).name)
            item.setToolTip(path)
            item.setData(QtCore.Qt.UserRole, path)
            self.addItem(item)

    def paths(self) -> list[str]:
        out: list[str] = []
        for idx in range(self.count()):
            item = self.item(idx)
            payload = item.data(QtCore.Qt.UserRole)
            if payload:
                out.append(str(payload))
        return out

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        files = [url.toLocalFile() for url in urls if url.isLocalFile() and url.toLocalFile()]
        if files:
            self.filesDropped.emit(files)
            event.acceptProposedAction()


class PatternTileWidget(QtWidgets.QFrame):
    clicked = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.record_id: str | None = None
        self.record: DiagnosticPatternRecord | None = None
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setLineWidth(2)
        self.setStyleSheet("QFrame { border: 2px solid #9aa0a6; border-radius: 4px; background: white; }")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(180, 180)
        self.image_label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.text_label = QtWidgets.QLabel()
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.text_label.setMinimumHeight(36)
        self.text_label.setStyleSheet("QLabel { font-size: 10px; color: #1f1f1f; }")
        layout.addWidget(self.image_label)
        layout.addWidget(self.text_label)

    def set_record(self, record: DiagnosticPatternRecord) -> None:
        self.record = record
        self.record_id = record.record_id
        pix = _array_to_pixmap(record.raw_pattern, target_size=_fallback_size(self.image_label, QtCore.QSize(180, 180)))
        self.image_label.setPixmap(pix)
        self.text_label.setText(
            f"{record.flat_index} | {record.predicted_phase}\n"
            f"c={record.confidence:.3f} m={record.margin:.3f}"
        )
        self.setToolTip(
            "\n".join(
                [
                    f"Source: {record.source_label}",
                    f"Index: {record.flat_index}",
                    f"Pixel: ({record.x}, {record.y})",
                    f"Predicted: {record.predicted_phase}",
                    f"Confidence: {record.confidence:.6f}",
                    f"Margin: {record.margin:.6f}",
                    f"Filter pass: {record.filter_pass}",
                ]
            )
        )

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.setStyleSheet("QFrame { border: 3px solid #d62728; border-radius: 4px; background: #fff7f7; }")
        else:
            self.setStyleSheet("QFrame { border: 2px solid #9aa0a6; border-radius: 4px; background: white; }")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self.record_id is not None:
            self.clicked.emit(self.record_id)
        super().mousePressEvent(event)


class DiagnosticGalleryWindow(QtWidgets.QMainWindow):
    def __init__(self, *, repo_root: Path, config_path: Path | None, logger: logging.Logger):
        super().__init__()
        self.repo_root = repo_root
        self.config_path = config_path or (repo_root / "diagnostic_gallery.runtime.yml")
        self.log = logger
        self.session: DiagnosticGallerySession | None = None
        self._selected_tile: PatternTileWidget | None = None
        self._tile_widgets: dict[str, PatternTileWidget] = {}
        self._source_combo_key_map: dict[str, str] = {}

        self.setWindowTitle("Diagnostic Pattern Gallery")
        self.resize(1760, 1020)

        self._config = self._default_config()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setFixedWidth(340)
        control_host = QtWidgets.QWidget()
        self.controls_scroll.setWidget(control_host)
        self.control_layout = QtWidgets.QVBoxLayout(control_host)
        self.control_layout.setContentsMargins(4, 4, 4, 4)
        self.control_layout.setSpacing(10)
        root.addWidget(self.controls_scroll, stretch=0)

        self._build_controls()

        self.content_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root.addWidget(self.content_splitter, stretch=1)

        self.gallery_scroll = QtWidgets.QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.gallery_container = QtWidgets.QWidget()
        self.gallery_layout = QtWidgets.QVBoxLayout(self.gallery_container)
        self.gallery_layout.setContentsMargins(4, 4, 4, 4)
        self.gallery_layout.setSpacing(12)
        self.gallery_layout.addStretch(1)
        self.gallery_scroll.setWidget(self.gallery_container)
        self.content_splitter.addWidget(self.gallery_scroll)

        self.detail_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.detail_splitter.setChildrenCollapsible(False)
        self.content_splitter.addWidget(self.detail_splitter)

        self.preview_tabs = QtWidgets.QTabWidget()
        self.raw_preview = QtWidgets.QLabel("Raw preview")
        self.raw_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.raw_preview.setMinimumSize(420, 320)
        self.raw_preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.preprocessed_preview = QtWidgets.QLabel("Preprocessed preview")
        self.preprocessed_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preprocessed_preview.setMinimumSize(420, 320)
        self.preprocessed_preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.preview_tabs.addTab(self.raw_preview, "Raw")
        self.preview_tabs.addTab(self.preprocessed_preview, "Preprocessed")
        self.detail_splitter.addWidget(self.preview_tabs)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(10)
        self.meta_table = QtWidgets.QTableWidget(0, 2)
        self.meta_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.meta_table.horizontalHeader().setStretchLastSection(True)
        self.meta_table.verticalHeader().setVisible(False)
        self.meta_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.meta_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.meta_table.setMinimumHeight(220)
        right_layout.addWidget(self._group_box("Metadata", self.meta_table))

        self.prob_table = QtWidgets.QTableWidget(0, 2)
        self.prob_table.setHorizontalHeaderLabels(["Phase", "Probability"])
        self.prob_table.horizontalHeader().setStretchLastSection(True)
        self.prob_table.verticalHeader().setVisible(False)
        self.prob_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.prob_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        right_layout.addWidget(self._group_box("Probabilities", self.prob_table))

        self.detail_status = QtWidgets.QLabel("Select a tile to inspect raw and preprocessed views.")
        self.detail_status.setWordWrap(True)
        right_layout.addWidget(self.detail_status)
        right_layout.addStretch(1)
        self.detail_splitter.addWidget(right_panel)
        self.detail_splitter.setStretchFactor(0, 4)
        self.detail_splitter.setStretchFactor(1, 2)

        self.content_splitter.setStretchFactor(0, 8)
        self.content_splitter.setStretchFactor(1, 2)

        self.statusBar().showMessage("Load a config or drop files, then build the session.")
        self._sync_config_to_controls()
        if config_path is not None and config_path.exists():
            self._load_config_file(config_path)

    def _default_config(self) -> dict[str, object]:
        return {
            "gallery_title": "Diagnostic Pattern Gallery",
            "output_dir": "reports/ml/diagnostic_gallery",
            "run_dir": "",
            "checkpoint": "best_checkpoint.pt",
            "device": "auto",
            "sampling": {
                "patterns_per_source": 5,
                "seed": 0,
                "strategy": "random",
            },
            "quality_filters": {
                "expression": "CI > 0.5 && Fit < 1.5",
            },
            "prediction_filters": {
                "min_confidence": 0.5,
                "min_margin": 0.15,
            },
            "source_groups": {
                "reference": [],
                "unknown": [],
            },
        }

    def _group_box(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(6, 16, 6, 6)
        layout.addWidget(widget)
        return box

    def _build_controls(self) -> None:
        title_label = QtWidgets.QLabel("<b>Session</b>")
        self.control_layout.addWidget(title_label)

        self.title_edit = QtWidgets.QLineEdit(str(self._config.get("gallery_title", "")))
        self.run_dir_edit = QtWidgets.QLineEdit(str(self._config.get("run_dir", "")))
        self.checkpoint_edit = QtWidgets.QLineEdit(str(self._config.get("checkpoint", "best_checkpoint.pt")))
        self.device_edit = QtWidgets.QLineEdit(str(self._config.get("device", "auto")))
        self.output_dir_edit = QtWidgets.QLineEdit(str(self._config.get("output_dir", "reports/ml/diagnostic_gallery")))

        form = QtWidgets.QFormLayout()
        form.addRow("Title", self.title_edit)
        form.addRow("Run dir", self.run_dir_edit)
        form.addRow("Checkpoint", self.checkpoint_edit)
        form.addRow("Device", self.device_edit)
        form.addRow("Output dir", self.output_dir_edit)
        self.control_layout.addLayout(form)

        source_box = QtWidgets.QGroupBox("Sources")
        source_layout = QtWidgets.QVBoxLayout(source_box)
        self.source_tabs = QtWidgets.QTabWidget()
        self.reference_list = FileDropListWidget("Drop reference .oh5 files")
        self.unknown_list = FileDropListWidget("Drop unknown .oh5 files")
        self.reference_list.filesDropped.connect(lambda files: self._add_files_to_list("reference", files))
        self.unknown_list.filesDropped.connect(lambda files: self._add_files_to_list("unknown", files))
        self.source_tabs.addTab(self._list_page(self.reference_list, "reference"), "Reference")
        self.source_tabs.addTab(self._list_page(self.unknown_list, "unknown"), "Unknown")
        source_layout.addWidget(self.source_tabs)
        self.control_layout.addWidget(source_box)

        sampling_box = QtWidgets.QGroupBox("Sampling / Filters")
        sampling_form = QtWidgets.QFormLayout(sampling_box)
        self.patterns_spin = QtWidgets.QSpinBox()
        self.patterns_spin.setRange(1, 99)
        self.patterns_spin.setValue(int(self._config["sampling"]["patterns_per_source"]))  # type: ignore[index]
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(int(self._config["sampling"]["seed"]))  # type: ignore[index]
        self.strategy_combo = QtWidgets.QComboBox()
        self.strategy_combo.addItems(["random", "top_confidence", "top_margin"])
        self.strategy_combo.setCurrentText(str(self._config["sampling"]["strategy"]))  # type: ignore[index]
        self.quality_edit = QtWidgets.QLineEdit(str(self._config["quality_filters"]["expression"]))  # type: ignore[index]
        self.min_conf_spin = QtWidgets.QDoubleSpinBox()
        self.min_conf_spin.setDecimals(3)
        self.min_conf_spin.setRange(0.0, 1.0)
        self.min_conf_spin.setSingleStep(0.05)
        self.min_conf_spin.setValue(float(self._config["prediction_filters"]["min_confidence"]))  # type: ignore[index]
        self.min_margin_spin = QtWidgets.QDoubleSpinBox()
        self.min_margin_spin.setDecimals(3)
        self.min_margin_spin.setRange(0.0, 1.0)
        self.min_margin_spin.setSingleStep(0.05)
        self.min_margin_spin.setValue(float(self._config["prediction_filters"]["min_margin"]))  # type: ignore[index]
        sampling_form.addRow("Patterns/source", self.patterns_spin)
        sampling_form.addRow("Seed", self.seed_spin)
        sampling_form.addRow("Strategy", self.strategy_combo)
        sampling_form.addRow("Quality expr", self.quality_edit)
        sampling_form.addRow("Min confidence", self.min_conf_spin)
        sampling_form.addRow("Min margin", self.min_margin_spin)
        self.control_layout.addWidget(sampling_box)

        manual_box = QtWidgets.QGroupBox("Manual pattern")
        manual_form = QtWidgets.QFormLayout(manual_box)
        self.source_combo = QtWidgets.QComboBox()
        self.flat_index_spin = QtWidgets.QSpinBox()
        self.flat_index_spin.setRange(0, 1_000_000_000)
        self.manual_add_btn = QtWidgets.QPushButton("Add pattern")
        self.manual_add_btn.clicked.connect(self._add_manual_pattern)
        manual_form.addRow("Source", self.source_combo)
        manual_form.addRow("Index", self.flat_index_spin)
        manual_form.addRow(self.manual_add_btn)
        self.control_layout.addWidget(manual_box)

        button_row = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load Config")
        self.load_btn.clicked.connect(self._load_config_dialog)
        self.build_btn = QtWidgets.QPushButton("Build Session")
        self.build_btn.clicked.connect(self._build_session)
        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.clicked.connect(self._export_session)
        button_row.addWidget(self.load_btn)
        button_row.addWidget(self.build_btn)
        button_row.addWidget(self.export_btn)
        self.control_layout.addLayout(button_row)

        self.control_status = QtWidgets.QLabel("Ready.")
        self.control_status.setWordWrap(True)
        self.control_layout.addWidget(self.control_status)
        self.control_layout.addStretch(1)

    def _list_page(self, widget: FileDropListWidget, group_name: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(widget)

        row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add")
        add_btn.clicked.connect(lambda: self._browse_and_add(group_name))
        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_selected(widget))
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self._clear_list(widget))
        row.addWidget(add_btn)
        row.addWidget(remove_btn)
        row.addWidget(clear_btn)
        layout.addLayout(row)
        return page

    def _sync_config_to_controls(self) -> None:
        source_groups = self._config.get("source_groups")
        if not isinstance(source_groups, dict):
            source_groups = {"reference": [], "unknown": []}
            if isinstance(self._config.get("reference_sources"), list):
                source_groups["reference"] = list(self._config.get("reference_sources", []))  # type: ignore[assignment]
            if isinstance(self._config.get("unknown_sources"), list):
                source_groups["unknown"] = list(self._config.get("unknown_sources", []))  # type: ignore[assignment]
            self._config["source_groups"] = source_groups
        self.reference_list.clear()
        self.unknown_list.clear()
        self.reference_list.add_placeholder("Drop reference .oh5 files")
        self.unknown_list.add_placeholder("Drop unknown .oh5 files")
        for row in source_groups.get("reference", []):
            if isinstance(row, dict) and row.get("file"):
                self.reference_list.add_paths([str(row["file"])])
            elif isinstance(row, str) and row.strip():
                self.reference_list.add_paths([row.strip()])
        for row in source_groups.get("unknown", []):
            if isinstance(row, dict) and row.get("file"):
                self.unknown_list.add_paths([str(row["file"])])
            elif isinstance(row, str) and row.strip():
                self.unknown_list.add_paths([row.strip()])
        self._refresh_source_combo()

    def _load_config_file(self, path: Path) -> None:
        cfg = load_yaml(path)
        self._config = cfg
        self.config_path = path.resolve()
        self.title_edit.setText(str(cfg.get("gallery_title", self.title_edit.text())))
        run_dir_value = cfg.get("run_dir")
        if run_dir_value in (None, "") and isinstance(cfg.get("model"), dict):
            run_dir_value = cfg["model"].get("run_dir", "")
        self.run_dir_edit.setText(str(run_dir_value or ""))
        self.checkpoint_edit.setText(str(cfg.get("checkpoint", "best_checkpoint.pt")))
        self.device_edit.setText(str(cfg.get("device", "auto")))
        self.output_dir_edit.setText(str(cfg.get("output_dir", "reports/ml/diagnostic_gallery")))

        sampling = cfg.get("sampling") if isinstance(cfg.get("sampling"), dict) else {}
        prediction_filters = cfg.get("prediction_filters") if isinstance(cfg.get("prediction_filters"), dict) else {}
        quality_filters = cfg.get("quality_filters") if isinstance(cfg.get("quality_filters"), dict) else {}

        self.patterns_spin.setValue(int(sampling.get("patterns_per_source", sampling.get("samples_per_scan", 5))))
        self.seed_spin.setValue(int(sampling.get("seed", 0)))
        self.strategy_combo.setCurrentText(str(sampling.get("strategy", "random")))
        self.quality_edit.setText(str(quality_filters.get("expression", "CI > 0.5 && Fit < 1.5")))
        self.min_conf_spin.setValue(float(prediction_filters.get("min_confidence", 0.5)))
        self.min_margin_spin.setValue(float(prediction_filters.get("min_margin", 0.15)))
        self._sync_config_to_controls()
        self.control_status.setText(f"Loaded config {path}")

    def _load_config_dialog(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select diagnostic gallery config", "", "YAML (*.yml *.yaml)")
        if selected:
            self._load_config_file(Path(selected))

    def _add_files_to_list(self, group_name: str, files: list[str]) -> None:
        target = self.reference_list if group_name == "reference" else self.unknown_list
        target.add_paths(files)
        self._refresh_source_combo()

    def _browse_and_add(self, group_name: str) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select .oh5 files", "", "All Files (*)")
        if selected:
            self._add_files_to_list(group_name, selected)

    def _remove_selected(self, widget: FileDropListWidget) -> None:
        rows = sorted({item.row() for item in widget.selectedItems()}, reverse=True)
        for row in rows:
            widget.takeItem(row)
        if widget.count() == 0:
            widget.add_placeholder("Drop files here")
        self._refresh_source_combo()

    def _clear_list(self, widget: FileDropListWidget) -> None:
        widget.clear()
        widget.add_placeholder("Drop files here")
        self._refresh_source_combo()

    def _refresh_source_combo(self) -> None:
        self.source_combo.clear()
        self._source_combo_key_map = {}
        for group_name, widget in (("reference", self.reference_list), ("unknown", self.unknown_list)):
            for path_text in widget.paths():
                path = Path(path_text)
                key = f"{group_name}:{path.stem}"
                label = f"{group_name} | {path.stem}"
                self._source_combo_key_map[label] = key
                self.source_combo.addItem(label)
        if self.source_combo.count() > 0:
            self.source_combo.setCurrentIndex(0)

    def _config_dict_from_controls(self) -> dict[str, object]:
        def _rows(widget: FileDropListWidget, group_name: str) -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []
            for file_text in widget.paths():
                path = Path(file_text)
                row: dict[str, object] = {
                    "file": path.as_posix(),
                    "scan_id": path.stem,
                }
                if group_name == "reference":
                    row["phase_name"] = path.stem
                rows.append(row)
            return rows

        return {
            "gallery_title": self.title_edit.text().strip() or "Diagnostic Pattern Gallery",
            "output_dir": self.output_dir_edit.text().strip() or "reports/ml/diagnostic_gallery",
            "run_dir": self.run_dir_edit.text().strip(),
            "checkpoint": self.checkpoint_edit.text().strip() or "best_checkpoint.pt",
            "device": self.device_edit.text().strip() or "auto",
            "sampling": {
                "patterns_per_source": int(self.patterns_spin.value()),
                "seed": int(self.seed_spin.value()),
                "strategy": self.strategy_combo.currentText().strip(),
            },
            "quality_filters": {
                "expression": self.quality_edit.text().strip(),
            },
            "prediction_filters": {
                "min_confidence": float(self.min_conf_spin.value()),
                "min_margin": float(self.min_margin_spin.value()),
            },
            "source_groups": {
                "reference": _rows(self.reference_list, "reference"),
                "unknown": _rows(self.unknown_list, "unknown"),
            },
        }

    def _build_session(self) -> None:
        cfg = self._config_dict_from_controls()
        self._config = cfg
        try:
            self.control_status.setText("Building session...")
            self.session = build_diagnostic_gallery_session_from_config(
                cfg=cfg,
                config_path=self.config_path,
                repo_root=self.repo_root,
                debug=False,
                logger=self.log,
            )
        except Exception as exc:
            self.session = None
            self.control_status.setText(f"Build failed: {exc}")
            self.statusBar().showMessage(f"Build failed: {exc}")
            return

        self._render_session()
        self.control_status.setText(f"Built session with {self.session.tile_count} displayed tiles.")
        self.statusBar().showMessage(f"Built session from {self.config_path}")

    def _render_session(self) -> None:
        if self.session is None:
            return
        self._tile_widgets = {}
        self._selected_tile = None
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        for source_key in self.session.source_order:
            result = self.session.source_results[source_key]
            section = QtWidgets.QGroupBox(f"{result.spec.display_name} [{result.spec.group_name}]")
            section_layout = QtWidgets.QVBoxLayout(section)
            header = QtWidgets.QLabel(
                f"{result.spec.file_path.name} | selected={len(result.display_records)} | "
                f"eligible={result.eligible_pixels} | candidates={result.candidate_pixels}"
            )
            header.setWordWrap(True)
            section_layout.addWidget(header)
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(10)
            for record in result.display_records:
                tile = PatternTileWidget()
                tile.set_record(record)
                tile.clicked.connect(self._select_record)
                self._tile_widgets[record.record_id] = tile
                row.addWidget(tile)
            row.addStretch(1)
            section_layout.addLayout(row)
            self.gallery_layout.addWidget(section)

        self.gallery_layout.addStretch(1)
        self._refresh_source_combo()
        if self.session.records:
            self._select_record(self.session.records[0].record_id)

    def _find_record(self, record_id: str) -> DiagnosticPatternRecord | None:
        if self.session is None:
            return None
        for record in self.session.records:
            if record.record_id == record_id:
                return record
        return None

    def _select_record(self, record_id: str) -> None:
        if self.session is None:
            return
        record = self._find_record(record_id)
        if record is None:
            return
        if self._selected_tile is not None:
            self._selected_tile.set_selected(False)
        tile = self._tile_widgets.get(record_id)
        if tile is not None:
            tile.set_selected(True)
            self._selected_tile = tile

        self.raw_preview.setPixmap(_array_to_pixmap(record.raw_pattern, target_size=_fallback_size(self.raw_preview, QtCore.QSize(640, 480))))
        self.preprocessed_preview.setPixmap(_array_to_pixmap(record.preprocessed_pattern, target_size=_fallback_size(self.preprocessed_preview, QtCore.QSize(640, 480))))

        meta_rows = [
            ("Source", record.source_label),
            ("Group", record.group_name),
            ("Scan ID", record.scan_id),
            ("File", record.file_path.name),
            ("Pattern index", str(record.flat_index)),
            ("Pixel x", str(record.x)),
            ("Pixel y", str(record.y)),
            ("Selected by", record.selected_by),
            ("Predicted phase", record.predicted_phase),
            ("Confidence", f"{record.confidence:.6f}"),
            ("Margin", f"{record.margin:.6f}"),
            ("CI", "" if record.confidence_index is None else f"{record.confidence_index:.6f}"),
            ("IQ", "" if record.image_quality is None else f"{record.image_quality:.6f}"),
            ("Fit", "" if record.fit is None else f"{record.fit:.6f}"),
            ("Valid", "" if record.valid is None else str(record.valid)),
            ("Filter pass", str(record.filter_pass)),
            ("Filter reasons", "; ".join(record.filter_reasons) if record.filter_reasons else ""),
        ]
        self.meta_table.setRowCount(len(meta_rows))
        for row_idx, (field, value) in enumerate(meta_rows):
            self.meta_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(field))
            self.meta_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(value))

        prob_items = sorted(record.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        self.prob_table.setRowCount(len(prob_items))
        for row_idx, (phase, prob) in enumerate(prob_items):
            self.prob_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(phase))
            self.prob_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(f"{prob:.6f}"))
        self.detail_status.setText(f"Selected {record.record_id}")

    def _add_manual_pattern(self) -> None:
        if self.session is None:
            self.control_status.setText("Build a session before adding manual patterns.")
            return
        label = self.source_combo.currentText().strip()
        source_key = self._source_combo_key_map.get(label)
        if not source_key:
            self.control_status.setText("Select a valid source first.")
            return
        try:
            record = add_manual_record(
                session=self.session,
                source_key=source_key,
                flat_index=int(self.flat_index_spin.value()),
            )
        except Exception as exc:
            self.control_status.setText(f"Manual add failed: {exc}")
            return

        self._render_session()
        self._select_record(record.record_id)
        self.control_status.setText(f"Added manual pattern {record.flat_index} from {record.source_label}")

    def _export_session(self) -> None:
        if self.session is None:
            self.control_status.setText("Build a session before exporting.")
            return
        try:
            manifest_path = export_diagnostic_gallery_artifacts(session=self.session, repo_root=self.repo_root, logger=self.log)
        except Exception as exc:
            self.control_status.setText(f"Export failed: {exc}")
            return

        cfg_path = self.session.output_dir / "session_config.json"
        write_json(cfg_path, self._config_dict_from_controls())
        self.control_status.setText(f"Exported {manifest_path}")


def run_diagnostic_gallery_app(
    *,
    repo_root: Path,
    config_path: Path | None,
    debug: bool = False,
) -> int:
    log = logging.getLogger("ml_diagnostic_gallery_gui")
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DiagnosticGalleryWindow(repo_root=repo_root, config_path=config_path, logger=log)
    window.show()
    return int(app.exec())
