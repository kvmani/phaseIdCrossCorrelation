"""PySide6 GUI for phase-classifier inference on unknown images and full `.oh5` scans."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .inference import LoadedModel, list_model_runs, load_trained_model, predict_image
from .oh5_inference import FullScanInferenceResult, run_oh5_full_scan_inference


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


class InferenceMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *, repo_root: Path, initial_root: Path | None, logger: logging.Logger):
        super().__init__()
        self.repo_root = repo_root
        self.log = logger
        self.state = GuiState(suite_root=initial_root, run_dirs=[])
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

        top.addWidget(QtWidgets.QLabel("Suite root / run dir"), 0, 0)
        top.addWidget(self.root_edit, 0, 1)
        top.addWidget(btn_root, 0, 2)
        top.addWidget(QtWidgets.QLabel("Model"), 1, 0)
        top.addWidget(self.model_combo, 1, 1, 1, 2)
        top.addWidget(QtWidgets.QLabel("Inference mode"), 2, 0)
        top.addWidget(self.mode_combo, 2, 1, 1, 2)
        top.addWidget(QtWidgets.QLabel("Known phase"), 3, 0)
        top.addWidget(self.known_phase_combo, 3, 1, 1, 2)
        top.addWidget(self.status_label, 4, 0, 1, 3)

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
        self.confidence_shading_checkbox = QtWidgets.QCheckBox("Use confidence shading")
        self.confidence_shading_checkbox.setChecked(True)
        self.confidence_shading_checkbox.toggled.connect(self._refresh_full_scan_preview)
        btn_full_scan = QtWidgets.QPushButton("Run Full-Scan Inference")
        btn_full_scan.clicked.connect(self._run_inference)
        oh5_controls.addWidget(QtWidgets.QLabel(".oh5 file"), 0, 0)
        oh5_controls.addWidget(self.oh5_edit, 0, 1)
        oh5_controls.addWidget(btn_oh5, 0, 2)
        oh5_controls.addWidget(self.confidence_shading_checkbox, 1, 0, 1, 2)
        oh5_controls.addWidget(btn_full_scan, 1, 2)

        self.oh5_help = QtWidgets.QPlainTextEdit()
        self.oh5_help.setReadOnly(True)
        self.oh5_help.setPlainText(
            "Full-scan mode runs inference on every available pattern in the selected .oh5 file.\n"
            "The phase map uses model class colors; with confidence shading enabled, low-score pixels are dulled."
        )
        oh5_layout.addWidget(self.oh5_help, stretch=1)
        self.input_stack.addWidget(oh5_page)

        right = QtWidgets.QVBoxLayout()
        mid.addLayout(right, stretch=1)
        self.prediction_label = QtWidgets.QLabel("Prediction: -")
        font = self.prediction_label.font()
        font.setPointSize(18)
        font.setBold(True)
        self.prediction_label.setFont(font)
        right.addWidget(self.prediction_label)

        self.result_preview_title = QtWidgets.QLabel("Result preview")
        right.addWidget(self.result_preview_title)
        self.result_preview = QtWidgets.QLabel("No result")
        self.result_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.result_preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.result_preview.setMinimumSize(540, 420)
        right.addWidget(self.result_preview, stretch=2)

        self.prob_table = QtWidgets.QTableWidget(0, 2)
        self.prob_table.setHorizontalHeaderLabels(["Phase", "Probability"])
        self.prob_table.horizontalHeader().setStretchLastSection(True)
        self.prob_table.verticalHeader().setVisible(False)
        right.addWidget(self.prob_table, stretch=1)

        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setReadOnly(True)
        right.addWidget(self.notes, stretch=1)

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
        self._run_inference()

    def _update_mode_ui(self) -> None:
        mode = str(self.mode_combo.currentData())
        self.state.inference_mode = mode
        self.input_stack.setCurrentIndex(0 if mode == INFERENCE_MODE_IMAGE else 1)
        if mode == INFERENCE_MODE_IMAGE:
            self.result_preview_title.setText("Image inference preview")
            if self.state.image_path is not None:
                self._run_inference()
            else:
                self.result_preview.setText("Load an image to predict.")
        else:
            self.result_preview_title.setText("Predicted phase map")
            if self.state.full_scan_result is not None:
                self._refresh_full_scan_preview()
            elif self.state.oh5_path is not None:
                self._run_inference()
            else:
                self.result_preview.setText("Select a .oh5 file to run full-scan inference.")

    def _set_image_path(self, image_path: str) -> None:
        self.state.image_path = Path(image_path).expanduser().resolve()
        self.drop_label.setText(str(self.state.image_path))
        if self.state.inference_mode == INFERENCE_MODE_IMAGE:
            self._run_inference()

    def _set_oh5_path(self, oh5_path: str) -> None:
        self.state.oh5_path = Path(oh5_path).expanduser().resolve()
        self.oh5_edit.setText(str(self.state.oh5_path))
        self.state.full_scan_result = None
        if self.state.inference_mode == INFERENCE_MODE_FULL_SCAN:
            self._run_inference()

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
        self._update_known_phase_status()

    def _run_full_scan_prediction(self) -> None:
        if self.state.loaded_model is None:
            return
        oh5_path = Path(self.oh5_edit.text().strip()).expanduser() if self.oh5_edit.text().strip() else self.state.oh5_path
        if oh5_path is None:
            return
        try:
            result = run_oh5_full_scan_inference(
                loaded=self.state.loaded_model,
                oh5_path=oh5_path,
                scan_name=oh5_path.stem,
            )
        except Exception as exc:
            self.status_label.setText(f"Full-scan inference failed: {exc}")
            return

        self.state.oh5_path = oh5_path.resolve()
        self.state.full_scan_result = result
        dominant_phase = max(result.phase_counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if result.phase_counts else "-"
        self.prediction_label.setText(
            f"Full scan: {dominant_phase} dominant | mean confidence {result.mean_confidence:.4f}"
        )
        self._refresh_full_scan_preview()

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

        palette = _phase_color_map(result.class_names)
        legend_lines = []
        for phase in result.class_names:
            rgb = tuple(int(round(float(v) * 255.0)) for v in palette[phase])
            legend_lines.append(f"{phase}: rgb{rgb}")
        self.notes.setPlainText(
            "\n".join(
                [
                    f"Run: {self.state.loaded_model.run_dir}",
                    f"Model: {self.state.loaded_model.model_name}",
                    f"Checkpoint: {self.state.loaded_model.checkpoint_path.name}",
                    f".oh5: {result.oh5_path}",
                    f"Scan: {result.scan_name}",
                    f"Grid: {result.nx} x {result.ny}",
                    f"Inferred pixels: {result.total_pixels}",
                    f"Header grid cells: {result.header_total_pixels}",
                    f"Mean confidence: {result.mean_confidence:.6f}",
                    f"Confidence shading: {'on' if self.confidence_shading_checkbox.isChecked() else 'off'}",
                    "",
                    "Legend:",
                    *legend_lines,
                ]
            )
        )
        self.status_label.setText(f"Full-scan inference complete for {result.oh5_path.name}")
        self._update_known_phase_status()

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
        self.result_preview.setPixmap(_rgb_array_to_pixmap(rendered, target_size=self.result_preview.size()))

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


def run_inference_gui(*, repo_root: Path, suite_root: Path | None, debug: bool = False) -> int:
    log = logging.getLogger("ml_inference_gui")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = InferenceMainWindow(repo_root=repo_root, initial_root=suite_root, logger=log)
    window.show()
    return int(app.exec())
