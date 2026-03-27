"""PySide6 GUI for phase-classifier inference on unknown images."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .inference import LoadedModel, list_model_runs, load_trained_model, predict_image


def _array_to_pixmap(array: np.ndarray, *, target_size: QtCore.QSize) -> QtGui.QPixmap:
    arr = np.clip(array, 0.0, 1.0)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    h, w = arr8.shape
    qimg = QtGui.QImage(arr8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    pix = QtGui.QPixmap.fromImage(qimg.copy())
    return pix.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


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


class InferenceMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *, repo_root: Path, initial_root: Path | None, logger: logging.Logger):
        super().__init__()
        self.repo_root = repo_root
        self.log = logger
        self.state = GuiState(suite_root=initial_root, run_dirs=[])
        self.setWindowTitle("ML Phase ID Inference")
        self.resize(1180, 760)

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
        top.addWidget(QtWidgets.QLabel("Known phase"), 2, 0)
        top.addWidget(self.known_phase_combo, 2, 1, 1, 2)
        top.addWidget(self.status_label, 3, 0, 1, 3)

        mid = QtWidgets.QHBoxLayout()
        layout.addLayout(mid, stretch=1)

        left = QtWidgets.QVBoxLayout()
        mid.addLayout(left, stretch=1)
        self.drop_label = DropImageLabel()
        self.drop_label.imageDropped.connect(self._set_image_path)
        left.addWidget(self.drop_label)
        btn_img = QtWidgets.QPushButton("Browse Image")
        btn_img.clicked.connect(self._browse_image)
        left.addWidget(btn_img)

        self.original_preview = QtWidgets.QLabel("Original image")
        self.original_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.original_preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.original_preview.setMinimumSize(320, 320)
        left.addWidget(self.original_preview, stretch=1)

        self.preprocessed_preview = QtWidgets.QLabel("Preprocessed image")
        self.preprocessed_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preprocessed_preview.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.preprocessed_preview.setMinimumSize(320, 320)
        left.addWidget(self.preprocessed_preview, stretch=1)

        right = QtWidgets.QVBoxLayout()
        mid.addLayout(right, stretch=1)
        self.prediction_label = QtWidgets.QLabel("Prediction: -")
        font = self.prediction_label.font()
        font.setPointSize(18)
        font.setBold(True)
        self.prediction_label.setFont(font)
        right.addWidget(self.prediction_label)

        self.prob_table = QtWidgets.QTableWidget(0, 2)
        self.prob_table.setHorizontalHeaderLabels(["Phase", "Probability"])
        self.prob_table.horizontalHeader().setStretchLastSection(True)
        self.prob_table.verticalHeader().setVisible(False)
        right.addWidget(self.prob_table, stretch=1)

        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setReadOnly(True)
        right.addWidget(self.notes, stretch=1)

        if initial_root is not None:
            self._refresh_run_dirs()

    def _browse_root(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Select suite root or model run directory")
        if not selected:
            return
        self.root_edit.setText(selected)
        self.state.suite_root = Path(selected)
        self._refresh_run_dirs()

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
        if self.state.image_path is not None:
            self._run_prediction()

    def _browse_image(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select unknown image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if selected:
            self._set_image_path(selected)

    def _set_image_path(self, image_path: str) -> None:
        self.state.image_path = Path(image_path).expanduser().resolve()
        self.drop_label.setText(str(self.state.image_path))
        self._run_prediction()

    def _run_prediction(self) -> None:
        if self.state.loaded_model is None or self.state.image_path is None:
            return
        try:
            result = predict_image(loaded=self.state.loaded_model, image_path=self.state.image_path)
        except Exception as exc:
            self.status_label.setText(f"Prediction failed: {exc}")
            return

        self.original_preview.setPixmap(_array_to_pixmap(result.original_image, target_size=self.original_preview.size()))
        self.preprocessed_preview.setPixmap(_array_to_pixmap(result.preprocessed_image, target_size=self.preprocessed_preview.size()))
        self.prediction_label.setText(f"Prediction: {result.predicted_phase} ({result.confidence:.4f})")

        items = sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True)
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

    def _update_known_phase_status(self) -> None:
        if self.state.loaded_model is None:
            return
        known = self.known_phase_combo.currentText().strip()
        if known == "(unknown)":
            return
        predicted = self.prediction_label.text()
        match = known in predicted
        self.status_label.setText(
            f"Known phase: {known} | Prediction status: {'correct' if match else 'mismatch'}"
        )


def run_inference_gui(*, repo_root: Path, suite_root: Path | None, debug: bool = False) -> int:
    log = logging.getLogger("ml_inference_gui")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = InferenceMainWindow(repo_root=repo_root, initial_root=suite_root, logger=log)
    window.show()
    return int(app.exec())
