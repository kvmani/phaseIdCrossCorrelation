"""PySide6/PyQtGraph desktop GUI for phase-wise `.oh5` exploratory analytics."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .phase_explorer import ExplorerDataset, build_intensity_mask, cdf_from_counts, histogram, load_explorer_dataset


@dataclass(slots=True)
class PlotSettings:
    bins: int
    x_min: float
    x_max: float
    y_min: float | None = None
    y_max: float | None = None
    show_cdf: bool = True


class PlotSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, settings: PlotSettings, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._settings = settings
        form = QtWidgets.QFormLayout(self)

        self.bins = QtWidgets.QSpinBox()
        self.bins.setRange(4, 8192)
        self.bins.setValue(settings.bins)
        self.xmin = QtWidgets.QDoubleSpinBox()
        self.xmin.setRange(-1e9, 1e9)
        self.xmin.setDecimals(6)
        self.xmin.setValue(settings.x_min)
        self.xmax = QtWidgets.QDoubleSpinBox()
        self.xmax.setRange(-1e9, 1e9)
        self.xmax.setDecimals(6)
        self.xmax.setValue(settings.x_max)
        self.ymin = QtWidgets.QDoubleSpinBox()
        self.ymin.setRange(-1e12, 1e12)
        self.ymin.setDecimals(6)
        self.ymin.setValue(settings.y_min if settings.y_min is not None else 0.0)
        self.ymin_en = QtWidgets.QCheckBox("Enable")
        self.ymin_en.setChecked(settings.y_min is not None)
        self.ymax = QtWidgets.QDoubleSpinBox()
        self.ymax.setRange(-1e12, 1e12)
        self.ymax.setDecimals(6)
        self.ymax.setValue(settings.y_max if settings.y_max is not None else 0.0)
        self.ymax_en = QtWidgets.QCheckBox("Enable")
        self.ymax_en.setChecked(settings.y_max is not None)
        self.show_cdf = QtWidgets.QCheckBox("Show CDF")
        self.show_cdf.setChecked(settings.show_cdf)

        form.addRow("Bins", self.bins)
        form.addRow("X min", self.xmin)
        form.addRow("X max", self.xmax)

        yrow_min = QtWidgets.QHBoxLayout()
        yrow_min.addWidget(self.ymin)
        yrow_min.addWidget(self.ymin_en)
        form.addRow("Y min", yrow_min)

        yrow_max = QtWidgets.QHBoxLayout()
        yrow_max.addWidget(self.ymax)
        yrow_max.addWidget(self.ymax_en)
        form.addRow("Y max", yrow_max)
        form.addRow(self.show_cdf)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def result_settings(self) -> PlotSettings:
        return PlotSettings(
            bins=int(self.bins.value()),
            x_min=float(self.xmin.value()),
            x_max=float(self.xmax.value()),
            y_min=float(self.ymin.value()) if self.ymin_en.isChecked() else None,
            y_max=float(self.ymax.value()) if self.ymax_en.isChecked() else None,
            show_cdf=bool(self.show_cdf.isChecked()),
        )


class PhaseColumnWidget(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        phase_name: str,
        data: ExplorerDataset,
        intensity_settings: PlotSettings,
        attr_settings: PlotSettings,
        on_intensity_settings: Callable[[], None],
        on_attr_settings: Callable[[], None],
        logger: logging.Logger,
    ):
        super().__init__()
        self.phase_name = phase_name
        self.data = data
        self.intensity_settings = intensity_settings
        self.attr_settings = attr_settings
        self.on_intensity_settings = on_intensity_settings
        self.on_attr_settings = on_attr_settings
        self.log = logger

        self.selection_regions: list[pg.LinearRegionItem] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(f"<b>{phase_name}</b>")
        header.addWidget(title)
        header.addStretch(1)
        self.pattern_spin = QtWidgets.QSpinBox()
        self.pattern_spin.setRange(0, max(0, self.data.pattern_count(phase_name) - 1))
        self.pattern_spin.valueChanged.connect(self._refresh_pattern)
        header.addWidget(QtWidgets.QLabel("Pattern ID"))
        header.addWidget(self.pattern_spin)
        layout.addLayout(header)

        # Intensity histogram panel
        irow = QtWidgets.QHBoxLayout()
        irow.addWidget(QtWidgets.QLabel("Intensity cumulative histogram"))
        irow.addStretch(1)
        self.btn_add_range = QtWidgets.QPushButton("+ range")
        self.btn_add_range.clicked.connect(self._add_region)
        irow.addWidget(self.btn_add_range)
        self.btn_clear_ranges = QtWidgets.QPushButton("Clear")
        self.btn_clear_ranges.clicked.connect(self._clear_regions)
        irow.addWidget(self.btn_clear_ranges)
        self.btn_gear_i = QtWidgets.QToolButton()
        self.btn_gear_i.setText("⚙")
        self.btn_gear_i.clicked.connect(self.on_intensity_settings)
        irow.addWidget(self.btn_gear_i)
        layout.addLayout(irow)

        self.intensity_plot = pg.PlotWidget(background="w")
        self.intensity_plot.showGrid(x=True, y=True, alpha=0.2)
        self.intensity_curve = self.intensity_plot.plot(pen=pg.mkPen("#1f77b4", width=2), name="cumulative")
        self.cdf_curve = self.intensity_plot.plot(pen=pg.mkPen("#d62728", width=2, style=QtCore.Qt.DashLine), name="cdf")
        layout.addWidget(self.intensity_plot, stretch=2)

        # Scalar field histogram panel
        arow = QtWidgets.QHBoxLayout()
        arow.addWidget(QtWidgets.QLabel("Attribute histogram"))
        self.attr_combo = QtWidgets.QComboBox()
        self.attr_combo.addItems(sorted(self.data.phases[self.phase_name].scalar_fields.keys()))
        self.attr_combo.currentTextChanged.connect(self.refresh_attribute_plot)
        arow.addWidget(self.attr_combo)
        arow.addStretch(1)
        self.btn_gear_a = QtWidgets.QToolButton()
        self.btn_gear_a.setText("⚙")
        self.btn_gear_a.clicked.connect(self.on_attr_settings)
        arow.addWidget(self.btn_gear_a)
        layout.addLayout(arow)

        self.attr_plot = pg.PlotWidget(background="w")
        self.attr_plot.showGrid(x=True, y=True, alpha=0.2)
        self.attr_curve = self.attr_plot.plot(pen=pg.mkPen("#2ca02c", width=2), name="attribute")
        layout.addWidget(self.attr_plot, stretch=1)

        # Pattern display panel
        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view, stretch=2)

        self._add_region()
        self.refresh_intensity_plot()
        self.refresh_attribute_plot()
        self._refresh_pattern()

    def _add_region(self) -> None:
        region = pg.LinearRegionItem(values=(0.4, 0.6), movable=True, brush=(100, 100, 255, 40))
        region.sigRegionChanged.connect(self._refresh_pattern)
        self.intensity_plot.addItem(region)
        self.selection_regions.append(region)
        self._refresh_pattern()

    def _clear_regions(self) -> None:
        for region in self.selection_regions:
            self.intensity_plot.removeItem(region)
        self.selection_regions = []
        self._refresh_pattern()

    def selected_ranges(self) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for r in self.selection_regions:
            lo, hi = r.getRegion()
            out.append((float(lo), float(hi)))
        return out

    def refresh_intensity_plot(self) -> None:
        values = self.data.phases[self.phase_name].intensity_values
        counts, cumulative, edges = histogram(
            values,
            bins=self.intensity_settings.bins,
            x_min=self.intensity_settings.x_min,
            x_max=self.intensity_settings.x_max,
        )
        x = edges[:-1]
        self.intensity_curve.setData(x=x, y=cumulative)
        if self.intensity_settings.show_cdf:
            self.cdf_curve.setData(x=x, y=cdf_from_counts(cumulative))
            self.cdf_curve.show()
        else:
            self.cdf_curve.hide()

        y_min = self.intensity_settings.y_min
        y_max = self.intensity_settings.y_max
        if y_min is not None or y_max is not None:
            vb = self.intensity_plot.getViewBox()
            cur_y = vb.viewRange()[1]
            vb.setYRange(y_min if y_min is not None else cur_y[0], y_max if y_max is not None else cur_y[1], padding=0)
        self.intensity_plot.setXRange(self.intensity_settings.x_min, self.intensity_settings.x_max, padding=0)

    def refresh_attribute_plot(self) -> None:
        field_name = self.attr_combo.currentText().strip()
        if not field_name:
            self.attr_curve.setData([], [])
            return
        values = self.data.phases[self.phase_name].scalar_fields.get(field_name)
        if values is None or values.size == 0:
            self.attr_curve.setData([], [])
            return

        x_min = self.attr_settings.x_min if self.attr_settings.x_min is not None else float(np.min(values))
        x_max = self.attr_settings.x_max if self.attr_settings.x_max is not None else float(np.max(values))
        if x_max <= x_min:
            x_max = x_min + 1e-6
        counts, cumulative, edges = histogram(values, bins=self.attr_settings.bins, x_min=x_min, x_max=x_max)
        self.attr_curve.setData(x=edges[:-1], y=cumulative)
        self.attr_plot.setXRange(x_min, x_max, padding=0)

    def _refresh_pattern(self) -> None:
        try:
            pattern = self.data.get_pattern(self.phase_name, int(self.pattern_spin.value()))
            ranges = self.selected_ranges()
            mask = build_intensity_mask(pattern, ranges)

            rgb = np.stack([pattern, pattern, pattern], axis=2)
            rgb[..., 0] = np.where(mask, 1.0, rgb[..., 0])
            rgb[..., 1] = np.where(mask, 0.1, rgb[..., 1])
            rgb[..., 2] = np.where(mask, 0.1, rgb[..., 2])
            self.image_view.setImage((rgb * 255).astype(np.uint8), autoLevels=True)
        except Exception as exc:
            self.log.exception("Pattern refresh failed for phase=%s: %s", self.phase_name, exc)


class PhaseExplorerMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *, dataset: ExplorerDataset, logger: logging.Logger):
        super().__init__()
        self.dataset = dataset
        self.log = logger
        self.setWindowTitle("Phase OH5 Explorer")
        self.resize(1800, 980)

        self.intensity_settings = PlotSettings(bins=256, x_min=0.0, x_max=1.0, show_cdf=True)
        self.attr_settings = PlotSettings(bins=128, x_min=0.0, x_max=1.0, show_cdf=False)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QGridLayout(top)
        top_layout.setContentsMargins(2, 2, 2, 2)

        self.phase_widgets: list[PhaseColumnWidget] = []
        for idx, phase_name in enumerate(self.dataset.phase_names):
            widget = PhaseColumnWidget(
                phase_name=phase_name,
                data=self.dataset,
                intensity_settings=self.intensity_settings,
                attr_settings=self.attr_settings,
                on_intensity_settings=self._edit_intensity_settings,
                on_attr_settings=self._edit_attr_settings,
                logger=self.log,
            )
            row = idx // 3
            col = idx % 3
            top_layout.addWidget(widget, row, col)
            self.phase_widgets.append(widget)

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(2000)

        splitter.addWidget(top)
        splitter.addWidget(self.log_text)
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)
        self.setCentralWidget(central)

    def append_log(self, text: str) -> None:
        self.log_text.appendPlainText(text)

    def _edit_intensity_settings(self) -> None:
        dlg = PlotSettingsDialog(self, self.intensity_settings, "Intensity plot settings (synced)")
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.intensity_settings = dlg.result_settings()
            for w in self.phase_widgets:
                w.intensity_settings = self.intensity_settings
                w.refresh_intensity_plot()
                w._refresh_pattern()

    def _edit_attr_settings(self) -> None:
        dlg = PlotSettingsDialog(self, self.attr_settings, "Attribute plot settings (synced)")
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.attr_settings = dlg.result_settings()
            for w in self.phase_widgets:
                w.attr_settings = self.attr_settings
                w.refresh_attribute_plot()


class QtLogHandler(logging.Handler):
    def __init__(self, sink: Callable[[str], None]):
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.sink(msg)


def run_phase_explorer_app(*, config_path: Path, repo_root: Path, debug: bool = False) -> int:
    log = logging.getLogger("ml_phase_explorer")
    log.setLevel(logging.DEBUG if debug else logging.INFO)

    dataset = load_explorer_dataset(config_path=config_path, repo_root=repo_root, logger=log)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True)

    win = PhaseExplorerMainWindow(dataset=dataset, logger=log)

    qt_handler = QtLogHandler(win.append_log)
    qt_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    log.addHandler(qt_handler)

    win.show()
    log.info("Loaded config=%s | phases=%s", dataset.config_path, ",".join(dataset.phase_names))
    return app.exec()
