"""Graphical user interface for the SVM Analyst simulator.

This module builds a PySide6 application that allows users to select modulation
modes, configure system parameters, and visualize the resulting PWM signals and
harmonics in real time using pyqtgraph for interactive plots.

Plot features:
- Zoom with mouse wheel, rubber-band rectangle selection.
- Pan with right-click drag.
- Crosshair cursor with live time/voltage readout.
- Clickable legend to hide/show individual traces.
- Per-phase line style panel: color picker, line width, dash pattern, marker.
- Pause / resume / step oscilloscope scrolling.
- Right-click context menu on any plot for extra options (ViewBox built-in).

Accessibility notes:
- All interactive widgets have accessible names and descriptive tooltips.
- Keyboard navigation is supported through standard Qt focus handling.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Force pyqtgraph to use PySide6 before any Qt import.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from . import __version__
from .core import SimulatorConfig, run_simulation
from .io import (
    export_fft_csv,
    export_report_pdf,
    export_waveform_csv,
    load_config,
    save_config,
)
from .modulations import ModulationMode, PulseAlignment
from .sweep import sweep_thd
from .visualization import svm_hexagon_vertices, svm_reference_vector


_DASH_STYLES: dict[str, Qt.PenStyle] = {
    "Solid": Qt.PenStyle.SolidLine,
    "Dashed": Qt.PenStyle.DashLine,
    "Dotted": Qt.PenStyle.DotLine,
    "DashDot": Qt.PenStyle.DashDotLine,
}

_MARKER_SYMBOLS: dict[str, Optional[str]] = {
    "None": None,
    "Circle": "o",
    "Cross": "x",
    "Square": "s",
    "Triangle": "t",
}

_PHASE_DEFAULTS = {
    "A": {"color": "#1f77b4", "width": 1.5, "dash": "Solid", "symbol": "None"},
    "B": {"color": "#ff7f0e", "width": 1.5, "dash": "Solid", "symbol": "None"},
    "C": {"color": "#2ca02c", "width": 1.5, "dash": "Solid", "symbol": "None"},
}

# Minimum and maximum width-to-height ratios enforced when the window is not maximized.
# Prevents the user from resizing to a portrait-like or ultrawide extreme.
_MIN_ASPECT_RATIO: float = 4.0 / 3.0  # ~1.333 – narrower than 4:3 looks broken
_MAX_ASPECT_RATIO: float = 8.0 / 3.0  # ~2.667 – wider than ultrawide monitor


class _DutyPercentAxisItem(pg.AxisItem):
    """Y-axis for the duty cycle plot that formats tick labels as XX.XX %."""

    def tickStrings(self, values, scale, spacing):  # noqa: N802
        return [f"{v:.2f}" for v in values]


class PlotCanvas(QtWidgets.QWidget):
    """pyqtgraph-based oscilloscope + FFT widget.

    Features
    --------
    - Zoom: mouse wheel on any plot, or drag a rubber-band rectangle.
    - Pan: right-click drag.
    - Crosshair: moves with the mouse over the waveform, shows t / V values.
    - Legend: click a legend label to hide/show the corresponding trace.
    - styles: per-phase color, line width, dash pattern and marker.
    - Right-click context menu (ViewBox built-in) to reset zoom.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """Initialise the waveform and FFT plots with default phase styles."""
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setAccessibleName("Oscilloscope and FFT display")
        self.setAccessibleDescription(
            "Dual-panel plot: upper panel shows the three-phase PWM waveforms with "
            "crosshair cursor; lower panel shows the FFT spectrum. Use the style panel "
            "on the right to change line colors, widths, and dash patterns per phase."
        )

        # Per-phase style state (mutable, driven by PlotStylePanel).
        self._styles: dict[str, dict] = {
            phase: dict(cfg) for phase, cfg in _PHASE_DEFAULTS.items()
        }

        self._wave_curves: dict[str, pg.PlotDataItem] = {}
        self._fft_curve: Optional[pg.PlotDataItem] = None
        self._duty_curves: dict[str, Optional[pg.PlotDataItem]] = {
            "A": None,
            "B": None,
            "C": None,
        }
        self._duty_fft_curve: Optional[pg.PlotDataItem] = None
        self._duty_deadtime_lines: list[pg.InfiniteLine] = []
        self._switch_markers: dict[str, Optional[pg.ScatterPlotItem]] = {
            "A": None,
            "B": None,
            "C": None,
        }
        self._cmv_curve: Optional[pg.PlotDataItem] = None
        self._dc_bus_curve: Optional[pg.PlotDataItem] = None
        self._ref_wave_curves: dict[str, Optional[pg.PlotDataItem]] = {
            "A": None,
            "B": None,
            "C": None,
        }
        self._ref_fft_curve: Optional[pg.PlotDataItem] = None
        self._ref_duty_curves: dict[str, Optional[pg.PlotDataItem]] = {
            "A": None,
            "B": None,
            "C": None,
        }
        # Whether the user has manually zoomed/panned the waveform view.
        self._wave_user_zoomed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)

        # --- Waveform plot ---
        self._wave_plot = pg.PlotWidget(title="Waveforms")
        self._wave_plot.setLabel("left", "Voltage", units="V")
        self._wave_plot.setLabel("bottom", "Time", units="s")
        self._wave_plot.showGrid(x=True, y=True, alpha=0.3)
        self._wave_legend = self._wave_plot.addLegend(offset=(10, 10))
        # Detect when the user manually moves the view so we stop forcing ranges.
        self._wave_plot.plotItem.vb.sigRangeChangedManually.connect(
            self._on_wave_manual_range
        )

        # Crosshair items
        _ch_pen = pg.mkPen(color="#888888", width=1, style=Qt.PenStyle.DashLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=_ch_pen)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=_ch_pen)
        self._crosshair_label = pg.TextItem("", anchor=(0.0, 1.0), color="#333333")
        self._wave_plot.addItem(self._vline, ignoreBounds=True)
        self._wave_plot.addItem(self._hline, ignoreBounds=True)
        self._wave_plot.addItem(self._crosshair_label, ignoreBounds=True)
        self._wave_plot.scene().sigMouseMoved.connect(self._on_mouse_move)

        # --- FFT plot ---
        self._fft_plot = pg.PlotWidget(title="FFT (PWM signal)")
        self._fft_plot.setLabel("left", "Magnitude")
        self._fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self._fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self._fft_plot.addLegend(offset=(10, 10))

        # --- Duty Cycle Envelope plot ---
        self._duty_plot = pg.PlotWidget(
            title="Duty Cycle Envelope",
            axisItems={"left": _DutyPercentAxisItem("left")},
        )
        self._duty_plot.setLabel("left", "Duty Cycle (%)")
        self._duty_plot.setLabel("bottom", "Time", units="s")
        self._duty_plot.showGrid(x=True, y=True, alpha=0.3)
        self._duty_plot.addLegend(offset=(10, 10))
        self._duty_plot.setAccessibleName("Duty cycle envelope plot")
        self._duty_plot.setAccessibleDescription(
            "Shows the per-PWM-period duty cycle (0-100 %) for each phase over time. "
            "The shape of the curve reveals the modulating reference waveform "
            "(sinusoid, SVM envelope, or DPWM clamped segments)."
        )

        # Checkbox row so the user can show/hide individual phase curves.
        self._duty_check: dict[str, QtWidgets.QCheckBox] = {}
        _duty_filter_row = QtWidgets.QWidget()
        _duty_filter_layout = QtWidgets.QHBoxLayout(_duty_filter_row)
        _duty_filter_layout.setContentsMargins(4, 2, 4, 0)
        _duty_filter_layout.addWidget(QtWidgets.QLabel("Show:"))
        for _phase in ("A", "B", "C"):
            _cb = QtWidgets.QCheckBox(f"Phase {_phase}")
            _cb.setChecked(True)
            _cb.setAccessibleName(f"Duty cycle Phase {_phase} visible")
            _cb.setToolTip(f"Show or hide the duty cycle curve for Phase {_phase}")
            _cb.stateChanged.connect(
                lambda state, p=_phase: self.set_duty_phase_visible(
                    p, state == Qt.CheckState.Checked.value
                )
            )
            self._duty_check[_phase] = _cb
            _duty_filter_layout.addWidget(_cb)
        _duty_filter_layout.addStretch(1)

        _duty_container = QtWidgets.QWidget()
        _duty_ctr_layout = QtWidgets.QVBoxLayout(_duty_container)
        _duty_ctr_layout.setContentsMargins(0, 0, 0, 0)
        _duty_ctr_layout.setSpacing(0)
        _duty_ctr_layout.addWidget(_duty_filter_row)
        _duty_ctr_layout.addWidget(self._duty_plot, stretch=2)

        # Dead-time limit reference lines on the duty cycle envelope plot.
        for _pen in (
            pg.mkPen(color="#e04060", style=Qt.PenStyle.DashLine, width=1),
            pg.mkPen(color="#e04060", style=Qt.PenStyle.DashLine, width=1),
        ):
            _line = pg.InfiniteLine(angle=0, movable=False, pen=_pen)
            _line.setVisible(False)
            self._duty_plot.addItem(_line)
            self._duty_deadtime_lines.append(_line)

        # Duty Cycle FFT sub-panel (toggle-able).
        self._duty_fft_plot = pg.PlotWidget(
            title="Duty Cycle FFT (Phase A)",
            axisItems={"left": pg.AxisItem("left")},
        )
        self._duty_fft_plot.setLabel("left", "Magnitude")
        self._duty_fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self._duty_fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self._duty_fft_plot.setBackground("k")

        self._duty_fft_check = QtWidgets.QCheckBox("Show Duty Cycle FFT")
        self._duty_fft_check.setChecked(True)
        self._duty_fft_check.setAccessibleName("Show duty cycle FFT panel")
        self._duty_fft_check.stateChanged.connect(
            lambda s: self._duty_fft_plot.setVisible(s == Qt.CheckState.Checked.value)
        )

        _duty_fft_row = QtWidgets.QWidget()
        _duty_fft_row_layout = QtWidgets.QHBoxLayout(_duty_fft_row)
        _duty_fft_row_layout.setContentsMargins(4, 2, 4, 0)
        _duty_fft_row_layout.addWidget(self._duty_fft_check)
        _duty_fft_row_layout.addStretch(1)

        _duty_ctr_layout.addWidget(_duty_fft_row)
        _duty_ctr_layout.addWidget(self._duty_fft_plot, stretch=1)

        # --- Common Mode Voltage panel -------------------------------------------
        self._cmv_check = QtWidgets.QCheckBox("Show CMV")
        self._cmv_check.setChecked(True)
        self._cmv_check.setAccessibleName("Show common mode voltage panel")
        self._cmv_check.setToolTip("Show or hide the common-mode voltage panel")
        self._cmv_check.stateChanged.connect(
            lambda s: self._cmv_plot.setVisible(s == Qt.CheckState.Checked.value)
        )

        self._cmv_plot = pg.PlotWidget(title="Common Mode Voltage  (Va+Vb+Vc)/3")
        self._cmv_plot.setLabel("left", "Voltage", units="V")
        self._cmv_plot.setLabel("bottom", "Time", units="s")
        self._cmv_plot.showGrid(x=True, y=True, alpha=0.3)
        self._cmv_plot.setAccessibleName("Common mode voltage plot")
        self._cmv_plot.setAccessibleDescription(
            "Shows the common-mode voltage CMV = (Va+Vb+Vc)/3. "
            "For ideal balanced SVM, CMV hovers at Vdc/2. "
            "Deviations equal the zero-sequence voltage injected by the modulator "
            "(triplen harmonics). Peak-to-peak CMV drives bearing currents and "
            "sets common-mode filter requirements."
        )

        _cmv_row = QtWidgets.QWidget()
        _cmv_row_layout = QtWidgets.QHBoxLayout(_cmv_row)
        _cmv_row_layout.setContentsMargins(4, 2, 4, 0)
        _cmv_row_layout.addWidget(self._cmv_check)
        _cmv_row_layout.addStretch(1)

        _cmv_container = QtWidgets.QWidget()
        _cmv_ctr_layout = QtWidgets.QVBoxLayout(_cmv_container)
        _cmv_ctr_layout.setContentsMargins(0, 0, 0, 0)
        _cmv_ctr_layout.setSpacing(0)
        _cmv_ctr_layout.addWidget(_cmv_row)
        _cmv_ctr_layout.addWidget(self._cmv_plot, stretch=1)

        # --- DC bus current ripple panel -----------------------------------------
        self._dc_bus_check = QtWidgets.QCheckBox("Show DC Bus Ripple")
        self._dc_bus_check.setChecked(True)
        self._dc_bus_check.setAccessibleName("Show DC bus current ripple panel")
        self._dc_bus_check.setToolTip(
            "Show or hide the normalised DC bus current ripple panel"
        )
        self._dc_bus_check.stateChanged.connect(
            lambda s: self._dc_bus_plot.setVisible(s == Qt.CheckState.Checked.value)
        )

        self._dc_bus_plot = pg.PlotWidget(
            title="DC Bus Current Ripple  (normalised, I_peak = 1)"
        )
        self._dc_bus_plot.setLabel("left", "I_dc (norm.)")
        self._dc_bus_plot.setLabel("bottom", "Time", units="s")
        self._dc_bus_plot.showGrid(x=True, y=True, alpha=0.3)
        self._dc_bus_plot.setAccessibleName("DC bus current ripple plot")
        self._dc_bus_plot.setAccessibleDescription(
            "Shows the normalised DC bus current: Da·Ia + Db·Ib + Dc·Ic per unit of peak "
            "phase current. Values are in [A / A_peak]. Peak-to-peak amplitude determines "
            "DC capacitor ripple-current rating and sizing."
        )

        _dc_bus_row = QtWidgets.QWidget()
        _dc_bus_row_layout = QtWidgets.QHBoxLayout(_dc_bus_row)
        _dc_bus_row_layout.setContentsMargins(4, 2, 4, 0)
        _dc_bus_row_layout.addWidget(self._dc_bus_check)
        _dc_bus_row_layout.addStretch(1)

        _dc_bus_container = QtWidgets.QWidget()
        _dc_bus_ctr_layout = QtWidgets.QVBoxLayout(_dc_bus_container)
        _dc_bus_ctr_layout.setContentsMargins(0, 0, 0, 0)
        _dc_bus_ctr_layout.setSpacing(0)
        _dc_bus_ctr_layout.addWidget(_dc_bus_row)
        _dc_bus_ctr_layout.addWidget(self._dc_bus_plot, stretch=1)

        splitter.addWidget(self._wave_plot)
        splitter.addWidget(_duty_container)
        splitter.addWidget(self._fft_plot)
        splitter.addWidget(_cmv_container)
        splitter.addWidget(_dc_bus_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setStretchFactor(3, 2)
        splitter.setStretchFactor(4, 2)
        layout.addWidget(splitter)

        # Expose a fake .figure attribute so io.export_plot_png gracefully
        # finds None and the caller knows to use grab_pixmap() instead.
        self.figure = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_pen(self, phase: str) -> pg.QPen:
        s = self._styles[phase]
        return pg.mkPen(
            color=s["color"],
            width=s["width"],
            style=_DASH_STYLES.get(s["dash"], Qt.PenStyle.SolidLine),
        )

    def _make_symbol(self, phase: str) -> Optional[str]:
        return _MARKER_SYMBOLS.get(self._styles[phase]["symbol"])

    def _on_wave_manual_range(self) -> None:
        self._wave_user_zoomed = True

    def _on_mouse_move(self, scene_pos: QtCore.QPointF) -> None:
        if self._wave_plot.sceneBoundingRect().contains(scene_pos):
            vb = self._wave_plot.plotItem.vb
            view_pos = vb.mapSceneToView(scene_pos)
            self._vline.setPos(view_pos.x())
            self._hline.setPos(view_pos.y())
            self._crosshair_label.setPos(view_pos.x(), view_pos.y())
            self._crosshair_label.setText(
                f"t = {view_pos.x():.4e} s   V = {view_pos.y():.3f} V"
            )

    def _update_curve_style(self, phase: str) -> None:
        """Apply current style state to an already-created curve."""
        curve = self._wave_curves.get(phase)
        if curve is None:
            return
        sym = self._make_symbol(phase)
        curve.setPen(self._make_pen(phase))
        curve.setSymbol(sym)
        curve.setSymbolSize(6 if sym is not None else 0)
        curve.setSymbolBrush(pg.mkBrush(self._styles[phase]["color"]))
        curve.setSymbolPen(pg.mkPen(self._styles[phase]["color"]))

    # ------------------------------------------------------------------
    # Public API (kept compatible with old matplotlib PlotCanvas)
    # ------------------------------------------------------------------

    def reset_zoom(self) -> None:
        """Re-enable auto-range on the waveform plot."""
        self._wave_user_zoomed = False
        self._wave_plot.enableAutoRange()

    def update_style(
        self,
        phase: str,
        color: Optional[str] = None,
        width: Optional[float] = None,
        dash: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> None:
        """Update visual style for one phase and redraw."""
        if color is not None:
            self._styles[phase]["color"] = color
        if width is not None:
            self._styles[phase]["width"] = width
        if dash is not None:
            self._styles[phase]["dash"] = dash
        if symbol is not None:
            self._styles[phase]["symbol"] = symbol
        self._update_curve_style(phase)

    def set_phase_visible(self, phase: str, visible: bool) -> None:
        """Show or hide one phase curve."""
        curve = self._wave_curves.get(phase)
        if curve is not None:
            curve.setVisible(visible)

    def update_waveform(
        self,
        time: np.ndarray,
        phases: dict[str, np.ndarray],
        switch_times: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """Update the waveform plot (oscilloscope trace)."""

        if not self._wave_curves:
            # First call – create curves and legend entries.
            for phase in ("A", "B", "C"):
                sym = self._make_symbol(phase)
                curve = self._wave_plot.plot(
                    time,
                    phases[phase],
                    name=f"Phase {phase}",
                    pen=self._make_pen(phase),
                    symbol=sym,
                    symbolSize=6 if sym is not None else 0,
                    symbolBrush=pg.mkBrush(self._styles[phase]["color"]),
                    symbolPen=pg.mkPen(self._styles[phase]["color"]),
                )
                self._wave_curves[phase] = curve
        else:
            for phase in ("A", "B", "C"):
                self._wave_curves[phase].setData(time, phases[phase])

        # Switching edge markers
        for phase in ("A", "B", "C"):
            if self._switch_markers[phase] is not None:
                self._wave_plot.removeItem(self._switch_markers[phase])
                self._switch_markers[phase] = None

        if switch_times is not None:
            for phase in ("A", "B", "C"):
                times = switch_times.get(phase)
                if times is not None and len(times) > 0:
                    sc = pg.ScatterPlotItem(
                        x=times,
                        y=np.zeros_like(times),
                        symbol="|",
                        size=12,
                        pen=pg.mkPen(self._styles[phase]["color"]),
                        brush=pg.mkBrush(self._styles[phase]["color"]),
                    )
                    self._wave_plot.addItem(sc)
                    self._switch_markers[phase] = sc

        if not self._wave_user_zoomed:
            self._wave_plot.setXRange(time[0], time[-1], padding=0)
            self._wave_plot.setYRange(-1.15, 1.15, padding=0)

    def update_fft(self, freqs: np.ndarray, magnitude: np.ndarray) -> None:
        """Update the FFT plot."""
        if self._fft_curve is None:
            self._fft_curve = self._fft_plot.plot(
                freqs,
                magnitude,
                name="FFT",
                pen=pg.mkPen(color="#1f77b4", width=1.5),
            )
        else:
            self._fft_curve.setData(freqs, magnitude)
        self._fft_plot.setXRange(0, float(freqs.max()), padding=0)
        self._fft_plot.setYRange(0, max(1e-3, float(magnitude.max()) * 1.1), padding=0)

    def update_duty_cycle(
        self,
        time: np.ndarray,
        duty: dict[str, np.ndarray],
        dead_time_duty_limit: float = 0.0,
    ) -> None:
        """Update the duty cycle envelope plot.

        Parameters
        ----------
        time:
            Time axis (one point per PWM period, at period mid-points).
        duty:
            Mapping from phase label (``"A"``, ``"B"``, ``"C"``) to duty cycle
            arrays in [0, 1].  Values are converted to percent (× 100) before
            plotting so the Y axis reads in % with two decimal places.
        dead_time_duty_limit:
            Fraction of the PWM period consumed by dead time
            (``dead_time_us × 1e-6 × pwm_frequency_hz``).  When non-zero,
            horizontal reference lines are drawn at the effective D_min and
            D_max boundaries.
        """
        # Build period-edge x-array (N+1 edges from N period mid-points) so that
        # stepMode=True draws a proper zero-order hold staircase: each duty cycle
        # value is held constant from the start to the end of its PWM period.
        if time.size > 1:
            dt = time[1] - time[0]
        elif time.size == 1:
            dt = 1.0
        else:
            dt = 1.0
        step_edges = np.empty(time.size + 1)
        step_edges[:-1] = time - dt / 2.0
        step_edges[-1] = time[-1] + dt / 2.0
        for phase in ("A", "B", "C"):
            duty_pct = duty[phase] * 100.0
            if self._duty_curves[phase] is None:
                self._duty_curves[phase] = self._duty_plot.plot(
                    step_edges,
                    duty_pct,
                    name=f"Phase {phase}",
                    pen=self._make_pen(phase),
                    stepMode=True,
                )
            else:
                self._duty_curves[phase].setData(step_edges, duty_pct)
        if time.size > 0:
            self._duty_plot.setYRange(0.0, 100.0, padding=0.05)

        # Dead-time limit lines: D_min = limit%, D_max = (1-limit)%.
        if len(self._duty_deadtime_lines) == 2 and dead_time_duty_limit > 0.0:
            self._duty_deadtime_lines[0].setValue(dead_time_duty_limit * 100.0)
            self._duty_deadtime_lines[1].setValue((1.0 - dead_time_duty_limit) * 100.0)
            self._duty_deadtime_lines[0].setVisible(True)
            self._duty_deadtime_lines[1].setVisible(True)
        elif len(self._duty_deadtime_lines) == 2:
            self._duty_deadtime_lines[0].setVisible(False)
            self._duty_deadtime_lines[1].setVisible(False)

    def update_duty_fft(
        self,
        freqs: np.ndarray,
        magnitude: np.ndarray,
    ) -> None:
        """Update the duty cycle FFT sub-panel.

        Parameters
        ----------
        freqs:
            Frequency axis in Hz.
        magnitude:
            Spectral magnitude of the duty cycle FFT.
        """
        if self._duty_fft_curve is None:
            self._duty_fft_curve = self._duty_fft_plot.plot(
                freqs,
                magnitude,
                name="Duty Cycle FFT",
                pen=pg.mkPen(color="#00bfff", width=1),
            )
        else:
            self._duty_fft_curve.setData(freqs, magnitude)

    def set_duty_phase_visible(self, phase: str, visible: bool) -> None:
        """Show or hide one phase curve in the duty cycle plot."""
        curve = self._duty_curves.get(phase)
        if curve is not None:
            curve.setVisible(visible)

    def update_cmv(self, time: np.ndarray, cmv: np.ndarray) -> None:
        """Update the Common Mode Voltage plot.

        Parameters
        ----------
        time:
            Time axis — same length as ``cmv`` (raw PWM sample rate).
        cmv:
            Common-mode voltage array (Va+Vb+Vc)/3, in volts.
        """
        if self._cmv_curve is None:
            self._cmv_curve = self._cmv_plot.plot(
                time,
                cmv,
                name="CMV",
                pen=pg.mkPen(color="#9467bd", width=1.5),
            )
        else:
            self._cmv_curve.setData(time, cmv)

    def update_dc_bus_ripple(self, time: np.ndarray, current_norm: np.ndarray) -> None:
        """Update the DC bus normalised current ripple plot.

        Parameters
        ----------
        time:
            Time axis — same length as ``current_norm`` (one point per PWM period).
        current_norm:
            Normalised DC bus current Da·Ia + Db·Ib + Dc·Ic [A/A_peak].
        """
        if self._dc_bus_curve is None:
            self._dc_bus_curve = self._dc_bus_plot.plot(
                time,
                current_norm,
                name="DC bus I (norm.)",
                pen=pg.mkPen(color="#d62728", width=1.5),
            )
        else:
            self._dc_bus_curve.setData(time, current_norm)

    def set_reference_static(
        self,
        fft_freqs: np.ndarray,
        fft_mag: np.ndarray,
        duty_time: np.ndarray,
        duty: dict[str, np.ndarray],
    ) -> None:
        """Overlay static reference curves on the FFT and duty cycle plots.

        Called once each time the user saves a reference snapshot.
        """
        _ref_pen = pg.mkPen(color="#888888", width=1.0, style=Qt.PenStyle.DashLine)
        # FFT overlay
        if self._ref_fft_curve is None:
            self._ref_fft_curve = self._fft_plot.plot(
                fft_freqs, fft_mag, name="Ref FFT", pen=_ref_pen
            )
        else:
            self._ref_fft_curve.setData(fft_freqs, fft_mag)
        # Duty cycle overlay (N+1 edges for ZOH staircase, same as live curves)
        if duty_time.size > 1:
            _dt = duty_time[1] - duty_time[0]
        elif duty_time.size == 1:
            _dt = 1.0
        else:
            _dt = 1.0
        _edges = np.empty(duty_time.size + 1)
        _edges[:-1] = duty_time - _dt / 2.0
        _edges[-1] = duty_time[-1] + _dt / 2.0 if duty_time.size > 0 else _dt / 2.0
        for phase in ("A", "B", "C"):
            duty_pct = duty[phase] * 100.0
            if self._ref_duty_curves[phase] is None:
                self._ref_duty_curves[phase] = self._duty_plot.plot(
                    _edges,
                    duty_pct,
                    name=f"Ref {phase}",
                    pen=_ref_pen,
                    stepMode=True,
                )
            else:
                self._ref_duty_curves[phase].setData(_edges, duty_pct)

    def update_reference_waveform(
        self, time: np.ndarray, phases: dict[str, np.ndarray]
    ) -> None:
        """Overlay reference waveform curves (updated each scroll tick).

        Parameters
        ----------
        time:
            Time axis for the current scroll window.
        phases:
            Mapping ``"A" / "B" / "C"`` → voltage array for the reference.
        """
        _ref_pen = pg.mkPen(color="#888888", width=1.0, style=Qt.PenStyle.DashLine)
        for phase in ("A", "B", "C"):
            if self._ref_wave_curves[phase] is None:
                self._ref_wave_curves[phase] = self._wave_plot.plot(
                    time,
                    phases[phase],
                    name=f"Ref {phase}",
                    pen=_ref_pen,
                )
            else:
                self._ref_wave_curves[phase].setData(time, phases[phase])

    def clear_reference(self) -> None:
        """Remove all reference overlay curves from every plot."""
        for phase in ("A", "B", "C"):
            if self._ref_wave_curves[phase] is not None:
                self._wave_plot.removeItem(self._ref_wave_curves[phase])
                self._ref_wave_curves[phase] = None
            if self._ref_duty_curves[phase] is not None:
                self._duty_plot.removeItem(self._ref_duty_curves[phase])
                self._ref_duty_curves[phase] = None
        if self._ref_fft_curve is not None:
            self._fft_plot.removeItem(self._ref_fft_curve)
            self._ref_fft_curve = None

    def grab_pixmap(self) -> QtGui.QPixmap:
        """Return a QPixmap screenshot of this widget for export."""
        return self.grab()


class PlotStylePanel(QtWidgets.QGroupBox):
    """Side panel that lets users customise per-phase plot appearance.

    Controls per phase
    ------------------
    - Visibility checkbox
    - Color picker button
    - Line width spinbox (0.5 – 5 px)
    - Dash pattern combobox
    - Marker shape combobox
    - Reset button to restore defaults
    """

    def __init__(self, canvas: PlotCanvas, parent: Optional[QtWidgets.QWidget] = None):
        """Initialise the style panel linked to *canvas*."""
        super().__init__("Plot style", parent)
        self.setAccessibleName("Plot style panel")
        self.setAccessibleDescription(
            "Controls the visual appearance of each phase curve: visibility toggle, "
            "color, line width, dash pattern, and marker shape."
        )
        self._canvas = canvas
        self._controls: dict[str, dict] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the grid layout with per-phase style controls."""
        grid = QtWidgets.QGridLayout(self)

        headers = ["Phase", "Visible", "Color", "Width (px)", "Dash", "Marker"]
        for col, text in enumerate(headers):
            lbl = QtWidgets.QLabel(f"<b>{text}</b>")
            grid.addWidget(lbl, 0, col)

        for row, phase in enumerate(("A", "B", "C"), start=1):
            style = self._canvas._styles[phase]

            # Visibility
            vis_cb = QtWidgets.QCheckBox()
            vis_cb.setChecked(True)
            vis_cb.setAccessibleName(f"Phase {phase} visible")
            vis_cb.stateChanged.connect(
                lambda state, p=phase: self._canvas.set_phase_visible(
                    p, state == Qt.CheckState.Checked.value
                )
            )

            # Color
            color_btn = QtWidgets.QPushButton()
            color_btn.setFixedSize(28, 22)
            color_btn.setToolTip(f"Pick color for Phase {phase}")
            color_btn.setAccessibleName(f"Phase {phase} color picker")
            color_btn.setStyleSheet(
                f"background-color: {style['color']}; border: 1px solid #888;"
            )
            color_btn.clicked.connect(lambda _=False, p=phase: self._pick_color(p))

            # Width
            width_spin = QtWidgets.QDoubleSpinBox()
            width_spin.setRange(0.5, 5.0)
            width_spin.setSingleStep(0.5)
            width_spin.setDecimals(1)
            width_spin.setValue(style["width"])
            width_spin.setFixedWidth(68)
            width_spin.setAccessibleName(f"Phase {phase} line width")
            width_spin.valueChanged.connect(
                lambda v, p=phase: self._canvas.update_style(p, width=v)
            )

            # Dash
            dash_cb = QtWidgets.QComboBox()
            dash_cb.addItems(list(_DASH_STYLES.keys()))
            dash_cb.setCurrentText(style["dash"])
            dash_cb.setAccessibleName(f"Phase {phase} dash pattern")
            dash_cb.currentTextChanged.connect(
                lambda s, p=phase: self._canvas.update_style(p, dash=s)
            )

            # Marker
            marker_cb = QtWidgets.QComboBox()
            marker_cb.addItems(list(_MARKER_SYMBOLS.keys()))
            marker_cb.setCurrentText(style["symbol"])
            marker_cb.setAccessibleName(f"Phase {phase} marker")
            marker_cb.currentTextChanged.connect(
                lambda s, p=phase: self._canvas.update_style(p, symbol=s)
            )

            grid.addWidget(QtWidgets.QLabel(f"Phase {phase}"), row, 0)
            grid.addWidget(vis_cb, row, 1)
            grid.addWidget(color_btn, row, 2)
            grid.addWidget(width_spin, row, 3)
            grid.addWidget(dash_cb, row, 4)
            grid.addWidget(marker_cb, row, 5)

            self._controls[phase] = {
                "vis": vis_cb,
                "color_btn": color_btn,
                "width": width_spin,
                "dash": dash_cb,
                "marker": marker_cb,
            }

        reset_btn = QtWidgets.QPushButton("Reset styles")
        reset_btn.setToolTip("Restore default colors, widths and dash patterns")
        reset_btn.setAccessibleName("Reset all plot styles")
        reset_btn.setAccessibleDescription(
            "Restore all per-phase line colors, widths, dash patterns, and markers to their defaults."
        )
        reset_btn.clicked.connect(self._reset_styles)
        grid.addWidget(reset_btn, len(("A", "B", "C")) + 1, 0, 1, 6)

    def _pick_color(self, phase: str) -> None:
        """Open a color dialog and apply the chosen color to *phase*."""
        current = self._canvas._styles[phase]["color"]
        color = QColorDialog.getColor(
            QtGui.QColor(current), self, f"Color – Phase {phase}"
        )
        if color.isValid():
            hex_color = color.name()
            self._controls[phase]["color_btn"].setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid #888;"
            )
            self._canvas.update_style(phase, color=hex_color)

    def _reset_styles(self) -> None:
        """Restore all phases to their default colors, widths, dash patterns, and markers."""
        for phase, defaults in _PHASE_DEFAULTS.items():
            ctl = self._controls[phase]
            ctl["color_btn"].setStyleSheet(
                f"background-color: {defaults['color']}; border: 1px solid #888;"
            )
            ctl["width"].setValue(defaults["width"])
            ctl["dash"].setCurrentText(defaults["dash"])
            ctl["marker"].setCurrentText(defaults["symbol"])
            self._canvas._styles[phase] = dict(defaults)
            self._canvas._update_curve_style(phase)


class SimulationWorker(QtCore.QObject):
    """Worker that runs the PWM simulation in a background thread.

    Emits :attr:`finished` with the :class:`~svm_shaper.core.SimulationResult`
    once the computation completes, so the GUI thread can update plots without
    blocking the event loop.
    """

    #: Emitted when the simulation finishes; carries the SimulationResult.
    finished = Signal(object)

    def __init__(self, config: SimulatorConfig):
        """Initialise the worker with the given simulation configuration."""
        super().__init__()
        self._config = config

    def run(self) -> None:
        """Execute the simulation and emit the result signal."""
        result = run_simulation(self._config)
        self.finished.emit(result)


class SweepDialog(QtWidgets.QDialog):
    """Dialog that sweeps one simulation parameter and plots the resulting THD.

    The user selects the variable to sweep (speed or PWM frequency), enters a
    range and step count, then presses *Run* to compute and display the
    THD-vs-parameter curve via pyqtgraph.
    """

    def __init__(
        self, base_config: SimulatorConfig, parent: Optional[QtWidgets.QWidget] = None
    ):
        """Initialise the sweep dialog with *base_config* as the reference configuration."""
        super().__init__(parent)
        self.setWindowTitle("Sweep THD")
        self.setAccessibleName("Sweep THD dialog")
        self.setAccessibleDescription(
            "Sweeps a scalar simulation parameter over a range and plots the resulting "
            "Total Harmonic Distortion curve."
        )
        self._base_config = base_config

        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the sweep-dialog layout with form inputs and a plot widget."""
        layout = QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self._variable_choice = QtWidgets.QComboBox()
        self._variable_choice.setAccessibleName("Sweep variable")
        self._variable_choice.setToolTip("Select the parameter to sweep")
        self._variable_choice.addItems(["speed_rpm", "pwm_frequency_hz"])
        self._min_edit = QLineEdit("0")
        self._min_edit.setAccessibleName("Sweep minimum value")
        self._max_edit = QLineEdit("1000")
        self._max_edit.setAccessibleName("Sweep maximum value")
        self._steps_edit = QLineEdit("20")
        self._steps_edit.setAccessibleName("Sweep steps count")

        form.addRow("Variable:", self._variable_choice)
        form.addRow("Min:", self._min_edit)
        form.addRow("Max:", self._max_edit)
        form.addRow("Steps:", self._steps_edit)

        self._run_button = QPushButton("Run")
        self._run_button.setAccessibleName("Run sweep")
        self._run_button.setToolTip("Run the parameter sweep and update the THD plot")
        self._run_button.clicked.connect(self._on_run)

        layout.addLayout(form)
        layout.addWidget(self._run_button)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("left", "THD (%)")
        self._plot_widget.setLabel("bottom", "Variable value")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setMinimumSize(500, 300)
        self._plot_widget.setAccessibleName("Sweep THD result plot")
        layout.addWidget(self._plot_widget)

    def _on_run(self) -> None:
        """Validate inputs, run the sweep, and update the plot."""
        try:
            variable = self._variable_choice.currentText()
            start = float(self._min_edit.text())
            stop = float(self._max_edit.text())
            steps = int(self._steps_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Invalid input", "Please enter valid numeric values."
            )
            return

        xs, thd = sweep_thd(self._base_config, variable, start, stop, steps)
        self._plot_widget.clear()
        self._plot_widget.setTitle(f"THD vs {variable}")
        self._plot_widget.setLabel("bottom", variable)
        self._plot_widget.plot(
            xs,
            thd,
            symbol="o",
            symbolSize=7,
            pen=pg.mkPen(color="#1f77b4", width=2),
            symbolBrush=pg.mkBrush("#1f77b4"),
        )


class SvmHexagonDialog(QtWidgets.QDialog):
    """Dialog showing the SVM hexagon and a rotating reference vector.

    The hexagon displays the six active space vectors of a three-phase inverter
    in the alpha-beta (Clarke) plane. The current active sector is highlighted
    in orange and the rotating reference vector is drawn in red. The animation
    updates every 100 ms using a QTimer.

    Uses matplotlib for geometry drawing (Polygon, arrow).
    """

    def __init__(
        self, config: SimulatorConfig, parent: Optional[QtWidgets.QWidget] = None
    ):
        """Initialise the dialog with the current simulation configuration."""
        super().__init__(parent)
        self.setWindowTitle("SVM Hexagon – SVM Analyst")
        self.setAccessibleName("SVM hexagon dialog")
        self.setAccessibleDescription(
            "Animated space-vector diagram showing the SVM hexagon, active sector "
            "highlight, and rotating reference vector."
        )
        self._config = config
        self._time = 0.0

        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._ax = self._figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

        self._refresh()

    def _refresh(self) -> None:
        """Redraw the hexagon animation frame for the current ``_time``."""
        self._ax.clear()

        # Draw hexagon vertices
        verts = svm_hexagon_vertices(vdc=1.0)
        poly = np.vstack((verts, verts[0]))
        self._ax.plot(poly[:, 0], poly[:, 1], "-o", label="Active vectors")

        # Compute reference vector based on time
        electrical_freq = (
            self._config.speed_rpm / 60.0
        ) * self._config.motor_pole_pairs
        theta = 2.0 * np.pi * electrical_freq * self._time

        # Determine the active SVM sector (1-6).
        sector = int(np.floor((theta % (2.0 * np.pi)) / (np.pi / 3.0))) + 1
        sector = ((sector - 1) % 6) + 1

        # Highlight the current active sector for clarity.
        # Sectors are numbered 1-6 around the hexagon.
        sector_angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6.0
        sectors = []
        for i in range(6):
            a = sector_angles[i]
            b = sector_angles[i + 1] if i + 1 < len(sector_angles) else 2 * np.pi
            # Create the triangular region from origin to the two vertices
            sectors.append(
                np.array([[0.0, 0.0], [np.cos(a), np.sin(a)], [np.cos(b), np.sin(b)]])
            )

        active_sector = sector - 1
        if 0 <= active_sector < len(sectors):
            poly_patch = Polygon(
                sectors[active_sector],
                closed=True,
                facecolor="orange",
                alpha=0.2,
                edgecolor="none",
            )
            self._ax.add_patch(poly_patch)

        ref = svm_reference_vector(theta, vdc=1.0)
        self._ax.arrow(
            0,
            0,
            ref[0],
            ref[1],
            head_width=0.05,
            length_includes_head=True,
            color="red",
        )
        self._ax.set_title(f"SVM hexagon (normalized) - Sector {sector}")
        self._ax.set_xlabel("Alpha")
        self._ax.set_ylabel("Beta")
        self._ax.axis("equal")
        self._ax.grid(True)
        self._canvas.draw()

        self._time += 0.01


class DqPhasorDialog(QtWidgets.QDialog):
    """Dialog showing dq-frame analysis: Clarke/Park trajectories, angle sawtooth, and metrics.

    Uses pyqtgraph PlotWidgets for real-time performance.

    Panels:
    • Top-left:     Clarke αβ space-vector trajectory + SVM hexagon
    • Top-right:    Park dq phasors (Vs voltage arrow, Is current arrow)
    • Bottom-left:  Electrical angle θ_e sawtooth (0–360 °elec)
    • Bottom-right: Mechanical angle θ_mech sawtooth (0–360 °mech)
    • Footer:       Voltage metrics (Vα, Vβ, Vd, Vq, |Vαβ|, |Vdq| — RMS, peak, mean)

    The dialog refreshes automatically whenever the main window produces a new
    simulation result via :meth:`refresh`.
    """

    def __init__(
        self,
        result,
        config: "SimulatorConfig",
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """Open the phasor dialog with the latest simulation *result*."""
        super().__init__(parent)
        self.setWindowTitle("dq-frame Analysis – SVM Analyst")
        self.setAccessibleName("dq phasor diagram dialog")
        self.setAccessibleDescription(
            "Shows Clarke αβ trajectory, Park dq phasors, electrical and mechanical "
            "angle sawtooth waveforms, and scalar metrics for α, β, d, q voltages."
        )
        self.setMinimumSize(1100, 700)

        # ── αβ Clarke trajectory ─────────────────────────────────────────────────
        self._pw_ab = pg.PlotWidget(title="Clarke αβ — space-vector trajectory")
        self._pw_ab.setLabel("bottom", "α (V)")
        self._pw_ab.setLabel("left", "β (V)")
        self._pw_ab.setAspectLocked(True)
        self._pw_ab.showGrid(x=True, y=True, alpha=0.25)
        self._pw_ab.addLegend(offset=(5, 5))
        self._curve_ab = self._pw_ab.plot(
            [], [],
            pen=pg.mkPen("#4ea6dc", width=1),
            name="Voltage trajectory",
        )
        self._curve_hex = self._pw_ab.plot(
            [], [],
            pen=pg.mkPen("#ff8c00", width=1.5, style=Qt.PenStyle.DashLine),
            name="SVM hexagon",
        )

        # ── Park dq phasors ──────────────────────────────────────────────────────
        self._pw_dq = pg.PlotWidget(title="Park dq — fundamental phasors")
        self._pw_dq.setLabel("bottom", "d (V)")
        self._pw_dq.setLabel("left", "q (V)")
        self._pw_dq.setAspectLocked(True)
        self._pw_dq.showGrid(x=True, y=True, alpha=0.25)
        self._pw_dq.addLine(x=0, pen=pg.mkPen("#666", width=1))
        self._pw_dq.addLine(y=0, pen=pg.mkPen("#666", width=1))
        # Voltage phasor: line + tip marker
        self._line_vs = self._pw_dq.plot(
            [], [], pen=pg.mkPen("#4ea6dc", width=3), name="Vs"
        )
        self._tip_vs = self._pw_dq.plot(
            [], [], symbol="t", symbolSize=14,
            symbolBrush="#4ea6dc", symbolPen=None, pen=None,
        )
        # Current phasor: line + tip marker
        self._line_is = self._pw_dq.plot(
            [], [], pen=pg.mkPen("#e05252", width=3), name="Is"
        )
        self._tip_is = self._pw_dq.plot(
            [], [], symbol="t", symbolSize=14,
            symbolBrush="#e05252", symbolPen=None, pen=None,
        )
        self._txt_vs = pg.TextItem("", color="#4ea6dc", anchor=(0, 1))
        self._txt_is = pg.TextItem("", color="#e05252", anchor=(0, 0))
        self._pw_dq.addItem(self._txt_vs)
        self._pw_dq.addItem(self._txt_is)

        # ── Electrical angle sawtooth ────────────────────────────────────────────
        self._pw_te = pg.PlotWidget(title="Electrical angle θ_e")
        self._pw_te.setLabel("bottom", "Time (s)")
        self._pw_te.setLabel("left", "θ_e (°elec)")
        self._pw_te.setYRange(0, 360, padding=0.05)
        self._pw_te.showGrid(x=True, y=True, alpha=0.25)
        self._curve_te = self._pw_te.plot(
            [], [], pen=pg.mkPen("#7fc97f", width=1)
        )

        # ── Mechanical angle sawtooth ────────────────────────────────────────────
        self._pw_tm = pg.PlotWidget(title="Mechanical angle θ_mech")
        self._pw_tm.setLabel("bottom", "Time (s)")
        self._pw_tm.setLabel("left", "θ_mech (°mech)")
        self._pw_tm.showGrid(x=True, y=True, alpha=0.25)
        self._curve_tm = self._pw_tm.plot(
            [], [], pen=pg.mkPen("#beaed4", width=1)
        )

        # ── Metrics footer ───────────────────────────────────────────────────────
        self._metrics_box = QtWidgets.QGroupBox("Voltage Metrics")
        metrics_layout = QtWidgets.QGridLayout(self._metrics_box)
        metrics_layout.setSpacing(6)
        # (display_name, result_field, unit)
        _metric_defs = [
            ("Vα RMS",      "dq_valpha_rms",         "V"),
            ("Vα peak",     "dq_valpha_peak",        "V"),
            ("Vβ RMS",      "dq_vbeta_rms",          "V"),
            ("Vβ peak",     "dq_vbeta_peak",         "V"),
            ("Vd mean",     "dq_vd",                 "V"),
            ("Vd RMS",      "dq_vd_rms",             "V"),
            ("Vq mean",     "dq_vq",                 "V"),
            ("Vq RMS",      "dq_vq_rms",             "V"),
            ("|Vαβ| mean",  "dq_vab_magnitude_mean", "V"),
            ("|Vαβ| RMS",   "dq_vab_magnitude_rms",  "V"),
            ("|Vdq| mean",  "dq_vdq_magnitude_mean", "V"),
            ("|Vdq| RMS",   "dq_vdq_magnitude_rms",  "V"),
        ]
        self._metric_value_labels: dict[str, QtWidgets.QLabel] = {}
        for idx, (name, field, unit) in enumerate(_metric_defs):
            row, col = divmod(idx, 4)
            lbl_name = QtWidgets.QLabel(f"{name}:")
            lbl_val = QtWidgets.QLabel("—")
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)
            metrics_layout.addWidget(lbl_name, row, col * 2)
            metrics_layout.addWidget(lbl_val, row, col * 2 + 1)
            self._metric_value_labels[field] = (lbl_val, unit)

        # ── Main layout: 2×2 grid of plots + metrics footer ─────────────────────
        plots_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(plots_widget)
        grid.setSpacing(4)
        grid.addWidget(self._pw_ab, 0, 0)
        grid.addWidget(self._pw_dq, 0, 1)
        grid.addWidget(self._pw_te, 1, 0)
        grid.addWidget(self._pw_tm, 1, 1)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(plots_widget, stretch=5)
        main_layout.addWidget(self._metrics_box, stretch=1)

        self._refresh(result, config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self, result, config: "SimulatorConfig") -> None:
        """Redraw all panels with *result* from the latest simulation."""
        self._refresh(result, config)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh(self, result, config: "SimulatorConfig") -> None:
        """Update all pyqtgraph curves and metric labels in-place (no full redraw)."""
        vdc = config.battery_voltage

        # ── Clarke αβ trajectory ─────────────────────────────────────────────────
        va = result.dq_valpha
        vb = result.dq_vbeta
        if va.size > 0:
            step = max(1, va.size // 5000)
            self._curve_ab.setData(va[::step], vb[::step])
        else:
            self._curve_ab.setData([], [])
        verts = svm_hexagon_vertices(vdc=vdc)
        poly = np.vstack((verts, verts[0]))
        self._curve_hex.setData(poly[:, 0], poly[:, 1])

        # ── Park dq phasors ──────────────────────────────────────────────────────
        vd, vq    = result.dq_vd, result.dq_vq
        v_mag     = result.dq_vs_magnitude
        v_ang     = result.dq_vs_angle_deg
        id_f, iq_f = result.dq_id, result.dq_iq
        i_ang     = result.dq_is_angle_deg
        ang_diff  = i_ang - v_ang

        self._line_vs.setData([0, vd], [0, vq])
        self._tip_vs.setData([vd], [vq])
        self._line_is.setData([0, id_f], [0, iq_f])
        self._tip_is.setData([id_f], [iq_f])

        if v_mag > 1e-6:
            self._txt_vs.setText(f"Vs={v_mag:.1f} V  ∠{v_ang:.1f}°")
            self._txt_vs.setPos(vd, vq)
            self._txt_is.setText(f"Is (norm.)  ∠{i_ang:.1f}°  φ={ang_diff:.1f}°")
            self._txt_is.setPos(id_f, iq_f)
        else:
            self._txt_vs.setText("No result yet")
            self._txt_vs.setPos(0.0, 0.0)
            self._txt_is.setText("")

        _lim = max(v_mag * 1.4, 1.0)
        self._pw_dq.setXRange(-_lim, _lim, padding=0)
        self._pw_dq.setYRange(-_lim, _lim, padding=0)

        # ── Angle sawtooth waveforms ─────────────────────────────────────────────
        time_arr  = result.time
        te_arr    = result.theta_e_deg
        tm_arr    = result.theta_mech_deg
        if time_arr.size > 0:
            step = max(1, time_arr.size // 10000)
            self._curve_te.setData(time_arr[::step], te_arr[::step])
            self._curve_tm.setData(time_arr[::step], tm_arr[::step])
            mech_max = 360.0 / config.motor_pole_pairs
            self._pw_tm.setYRange(0, mech_max, padding=0.05)
        else:
            self._curve_te.setData([], [])
            self._curve_tm.setData([], [])

        # ── Metrics labels ───────────────────────────────────────────────────────
        for attr, (lbl_widget, unit) in self._metric_value_labels.items():
            val = getattr(result, attr, 0.0)
            lbl_widget.setText(f"{val:.2f} {unit}")



class SvmShaperApp(QtWidgets.QMainWindow):
    """Main window for the SVM Analyst application.

    This QMainWindow hosts the parameter control panel, the pyqtgraph
    oscilloscope/FFT canvas, the per-phase style panel, and the explanation
    text box.  A background :class:`SimulationWorker` thread is used so that
    the GUI remains responsive while the simulation runs.
    """

    def __init__(self):
        """Initialise the main window, build the UI, and start the simulation loop."""
        super().__init__()
        self.setWindowTitle("SVM Analyst - PWM Modulation Simulator")
        self.setAccessibleName("SVM Analyst main window")

        # Use the company logo as the application icon when available.
        logo_path = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "Logo_rectangle_blindsystems (300x200) (1).png"
        )
        if logo_path.exists():
            try:
                self.setWindowIcon(QtGui.QIcon(str(logo_path)))
            except Exception:
                # Fall back to default icon if Qt cannot load the file.
                self.setWindowIcon(QtGui.QIcon())
        else:
            self.setWindowIcon(QtGui.QIcon())

        self._config = SimulatorConfig()
        self._timer: Optional[QtCore.QTimer] = None
        self._sim_result = None
        self._scroll_index = 0
        self._scroll_step = 1

        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker: Optional[SimulationWorker] = None

        self._constraining_size: bool = False
        self._ref_result = None
        self._ref_display_signals: dict = {}
        self._dq_dialog: Optional[DqPhasorDialog] = None

        self._build_ui()
        self._apply_config_to_ui(self._config)
        self._start_simulation_loop()
        self._start_simulation_worker()

    def _build_ui(self) -> None:
        """Build the main window layout: control panel, plots, style panel, info box."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.setMinimumSize(1280, 720)

        main_layout = QtWidgets.QVBoxLayout(central)

        self._menu_bar = self.menuBar()
        self._build_menu()

        self._control_panel = self._create_control_panel()
        self._plot_canvas = PlotCanvas(parent=central)
        self._plot_style_panel = PlotStylePanel(self._plot_canvas, parent=central)

        self._info_box = QPlainTextEdit(readOnly=True)
        self._info_box.setMinimumHeight(120)
        self._info_box.setMaximumHeight(180)
        self._info_box.setAccessibleName("Explanation text")
        self._info_box.setAccessibleDescription(
            "Displays a textual summary of the current modulation settings, THD, and harmonics. "
            "Use the Copy button to place the full text on the clipboard for screen readers."
        )
        self._info_box.setToolTip(
            "Explanation of the current modulation and parameter settings"
        )
        self._info_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self._copy_explanation_button = QtWidgets.QPushButton("Copy explanation")
        self._copy_explanation_button.setAccessibleName("Copy explanation text")
        self._copy_explanation_button.setToolTip(
            "Copy the full explanation text to the clipboard for screen readers."
        )
        self._copy_explanation_button.clicked.connect(
            self._copy_explanation_to_clipboard
        )

        # Plot area: canvas on the left, style panel on the right.
        plot_row = QtWidgets.QHBoxLayout()
        plot_row.addWidget(self._plot_canvas, stretch=1)
        plot_row.addWidget(self._plot_style_panel, stretch=0)

        main_layout.addWidget(self._control_panel, stretch=0)
        main_layout.addLayout(plot_row, stretch=1)

        info_row = QtWidgets.QHBoxLayout()
        info_row.addWidget(self._info_box, stretch=1)
        info_row.addWidget(self._copy_explanation_button, stretch=0)
        main_layout.addLayout(info_row)

        self._setup_focus_and_tab_order()

    def _setup_focus_and_tab_order(self) -> None:
        """Establish an explicit Tab-key order and record it for testing.

        The sequence starts at the first field in the System Parameters panel
        so that screen readers land in the right place as soon as the window
        opens.  Every interactive widget is added in the logical reading order
        (left panel top-to-bottom, then modulation list, display options,
        oscilloscope controls, and finally the Copy explanation button).
        """
        self._tab_sequence = [
            # -- System parameters (left panel, top → bottom) --
            self._author_name_edit,
            self._project_name_edit,
            self._pole_pairs_spin,
            self._pwm_freq_spin,
            self._battery_voltage_spin,
            self._amplitude_spin,
            self._speed_spin,
            self._filter_cutoff_spin,
            self._injection_spin,
            self._alignment_choice,
            self._dead_time_spin,
            self._diode_vf_spin,
            self._current_phase_spin,
            # -- Modulation selection --
            self._modulation_list,
            # -- Display options --
            self._voltage_choice,
            self._filter_checkbox,
            self._edges_checkbox,
            self._run_button,
            # -- Oscilloscope controls --
            self._pause_button,
            self._step_button,
            self._reset_zoom_button,
            self._export_csv_button,
            self._export_png_button,
            self._save_ref_button,
            self._clear_ref_button,
            # -- Explanation / copy --
            self._copy_explanation_button,
        ]

        for first, second in zip(self._tab_sequence, self._tab_sequence[1:]):
            QtWidgets.QWidget.setTabOrder(first, second)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Set initial keyboard focus to the first System Parameters input."""
        super().showEvent(event)
        self._author_name_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Constrain window aspect ratio when manually resized.

        Prevents the window from becoming a portrait-like slit or an
        extreme ultrawide strip that would ruin the graph layout.
        Skipped whenever the window is maximized or full-screen.
        """
        super().resizeEvent(event)
        if self._constraining_size or self.isMaximized() or self.isFullScreen():
            return
        w = self.width()
        h = self.height()
        if h <= 0:
            return
        ratio = w / h
        if ratio > _MAX_ASPECT_RATIO:
            self._constraining_size = True
            self.resize(w, max(int(w / _MAX_ASPECT_RATIO), self.minimumHeight()))
            self._constraining_size = False
        elif ratio < _MIN_ASPECT_RATIO:
            self._constraining_size = True
            self.resize(max(int(h * _MIN_ASPECT_RATIO), self.minimumWidth()), h)
            self._constraining_size = False

    def _build_menu(self) -> None:
        """Build the application menu bar with File, View, Tools, and Help menus."""
        file_menu = self._menu_bar.addMenu("&File")

        config_menu = file_menu.addMenu("&Config")
        save_config_action = QtGui.QAction("Save configuration...", self)
        save_config_action.triggered.connect(self._save_configuration)
        config_menu.addAction(save_config_action)

        load_config_action = QtGui.QAction("Load configuration...", self)
        load_config_action.triggered.connect(self._load_configuration)
        config_menu.addAction(load_config_action)

        export_menu = file_menu.addMenu("&Export")
        export_wave_csv = QtGui.QAction("Waveform CSV...", self)
        export_wave_csv.triggered.connect(self._export_waveform_csv)
        export_menu.addAction(export_wave_csv)

        export_fft_csv = QtGui.QAction("FFT CSV...", self)
        export_fft_csv.triggered.connect(self._export_fft_csv)
        export_menu.addAction(export_fft_csv)

        export_png = QtGui.QAction("Plot PNG...", self)
        export_png.triggered.connect(self._export_plot_png)
        export_menu.addAction(export_png)

        export_report = QtGui.QAction("Report PDF...", self)
        export_report.triggered.connect(self._export_report_pdf)
        export_menu.addAction(export_report)

        exit_action = QtGui.QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)

        view_menu = self._menu_bar.addMenu("&View")
        svm_hex = QtGui.QAction("SVM Hexagon", self)
        svm_hex.triggered.connect(self._show_svm_hexagon)
        view_menu.addAction(svm_hex)

        dq_phasor_action = QtGui.QAction("dq-frame Phasor Diagram", self)
        dq_phasor_action.setAccessibleName("Open dq phasor diagram")
        dq_phasor_action.setToolTip(
            "Show the voltage space-vector trajectory (αβ) and fundamental dq phasors"
        )
        dq_phasor_action.triggered.connect(self._show_dq_phasor)
        view_menu.addAction(dq_phasor_action)

        tools_menu = self._menu_bar.addMenu("&Tools")
        sweep_action = QtGui.QAction("Sweep THD...", self)
        sweep_action.triggered.connect(self._open_sweep_dialog)
        tools_menu.addAction(sweep_action)

        help_menu = self._menu_bar.addMenu("&Help")
        about_action = QtGui.QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_control_panel(self) -> QtWidgets.QWidget:
        """Create the horizontal control strip with four groups: system params, modulation, display, oscilloscope."""
        cp_widget = QtWidgets.QWidget()
        cp_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        layout = QtWidgets.QHBoxLayout(cp_widget)

        def _lock_numeric_entry(widget: QtWidgets.QAbstractSpinBox) -> None:
            # Keep arrow-button interaction but prevent free text entry.
            line_edit = widget.lineEdit()
            if line_edit is not None:
                line_edit.setReadOnly(True)

        # Left side: parameter controls
        param_group = QtWidgets.QGroupBox("System parameters")
        param_group.setAccessibleName("System parameters group")
        param_group.setAccessibleDescription(
            "Motor and inverter parameters: pole pairs, PWM frequency, bus voltage, "
            "speed, dead time, diode forward voltage, and current phase."
        )
        param_layout = QtWidgets.QFormLayout(param_group)

        self._pole_pairs_spin = QtWidgets.QSpinBox()
        self._pole_pairs_spin.setRange(1, 20)
        self._pole_pairs_spin.setValue(self._config.motor_pole_pairs)
        self._pole_pairs_spin.setToolTip("Number of pole pairs in the PMSM")
        self._pole_pairs_spin.setAccessibleName("Motor pole pairs")
        _lock_numeric_entry(self._pole_pairs_spin)

        self._pwm_freq_spin = QtWidgets.QDoubleSpinBox()
        self._pwm_freq_spin.setRange(100.0, 200000.0)
        self._pwm_freq_spin.setSingleStep(100.0)
        self._pwm_freq_spin.setValue(self._config.pwm_frequency_hz)
        self._pwm_freq_spin.setToolTip("PWM carrier frequency in Hz")
        self._pwm_freq_spin.setAccessibleName("PWM frequency")
        _lock_numeric_entry(self._pwm_freq_spin)

        self._battery_voltage_spin = QtWidgets.QDoubleSpinBox()
        self._battery_voltage_spin.setRange(1.0, 1000.0)
        self._battery_voltage_spin.setSingleStep(10.0)
        self._battery_voltage_spin.setValue(self._config.battery_voltage)
        self._battery_voltage_spin.setToolTip("DC bus voltage (battery) in volts")
        self._battery_voltage_spin.setAccessibleName("Battery voltage")
        _lock_numeric_entry(self._battery_voltage_spin)

        self._amplitude_spin = QtWidgets.QDoubleSpinBox()
        self._amplitude_spin.setRange(0.0, 100.0)
        self._amplitude_spin.setSingleStep(1.0)
        self._amplitude_spin.setValue(self._config.amplitude_percent)
        self._amplitude_spin.setToolTip(
            "Modulation amplitude as a percentage of full scale (0-100%)."
        )
        self._amplitude_spin.setAccessibleName("Modulation amplitude percent")
        _lock_numeric_entry(self._amplitude_spin)

        self._speed_spin = QtWidgets.QDoubleSpinBox()
        self._speed_spin.setRange(0.0, 20000.0)
        self._speed_spin.setSingleStep(10.0)
        self._speed_spin.setValue(self._config.speed_rpm)
        self._speed_spin.setToolTip("Motor speed in RPM")
        self._speed_spin.setAccessibleName("Speed in RPM")
        _lock_numeric_entry(self._speed_spin)

        self._author_name_edit = QLineEdit()
        self._author_name_edit.setAccessibleName("Report author")
        self._author_name_edit.setToolTip(
            "Author name to include on the generated PDF report"
        )

        self._project_name_edit = QLineEdit()
        self._project_name_edit.setAccessibleName("Project name")
        self._project_name_edit.setToolTip(
            "Project name to include on the generated PDF report"
        )

        self._filter_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self._filter_cutoff_spin.setRange(0.0, 200000.0)
        self._filter_cutoff_spin.setSingleStep(10.0)
        self._filter_cutoff_spin.setValue(self._config.filter_cutoff_hz)
        self._filter_cutoff_spin.setToolTip(
            "Cutoff frequency for the low-pass filter (Hz). Set to 0 for automatic (3× electrical frequency)."
        )
        self._filter_cutoff_spin.setAccessibleName("Filter cutoff frequency")
        _lock_numeric_entry(self._filter_cutoff_spin)

        self._injection_spin = QtWidgets.QDoubleSpinBox()
        self._injection_spin.setRange(0.0, 100.0)
        self._injection_spin.setSingleStep(1.0)
        self._injection_spin.setValue(self._config.injection_percent)
        self._injection_spin.setToolTip(
            "Third harmonic injection factor (percent of 1/6). Only used for custom THIPWM mode."
        )
        self._injection_spin.setAccessibleName("Third harmonic injection percent")
        self._injection_spin.setEnabled(False)
        _lock_numeric_entry(self._injection_spin)

        self._alignment_choice = QtWidgets.QComboBox()
        self._alignment_choice.setAccessibleName("PWM alignment mode")
        self._alignment_choice.setToolTip(
            "PWM pulse alignment similar to MCU timers: left, right, or center aligned."
        )
        for mode in PulseAlignment:
            self._alignment_choice.addItem(mode.value, mode)

        self._dead_time_spin = QtWidgets.QDoubleSpinBox()
        self._dead_time_spin.setRange(0.0, 1000.0)
        self._dead_time_spin.setSingleStep(0.1)
        self._dead_time_spin.setValue(self._config.dead_time_us)
        self._dead_time_spin.setToolTip(
            "Dead time inserted at each PWM edge (microseconds)."
        )
        self._dead_time_spin.setAccessibleName("PWM dead time microseconds")
        _lock_numeric_entry(self._dead_time_spin)

        self._diode_vf_spin = QtWidgets.QDoubleSpinBox()
        self._diode_vf_spin.setRange(0.0, 10.0)
        self._diode_vf_spin.setSingleStep(0.01)
        self._diode_vf_spin.setDecimals(3)
        self._diode_vf_spin.setValue(self._config.diode_forward_voltage_v)
        self._diode_vf_spin.setToolTip(
            "Body diode forward voltage used during dead-time conduction (volts)."
        )
        self._diode_vf_spin.setAccessibleName("Body diode forward voltage")
        _lock_numeric_entry(self._diode_vf_spin)

        self._current_phase_spin = QtWidgets.QDoubleSpinBox()
        self._current_phase_spin.setRange(-45.0, 45.0)
        self._current_phase_spin.setSingleStep(1.0)
        self._current_phase_spin.setDecimals(1)
        self._current_phase_spin.setValue(self._config.current_phase_deg)
        self._current_phase_spin.setToolTip(
            "Synthetic phase-current angle (degrees) used for dead-time diode polarity."
        )
        self._current_phase_spin.setAccessibleName("Current phase angle")
        _lock_numeric_entry(self._current_phase_spin)

        param_layout.addRow("Author:", self._author_name_edit)
        param_layout.addRow("Project:", self._project_name_edit)
        param_layout.addRow("Pole pairs:", self._pole_pairs_spin)
        param_layout.addRow("PWM frequency (Hz):", self._pwm_freq_spin)
        param_layout.addRow("Battery voltage (V):", self._battery_voltage_spin)
        param_layout.addRow("Amplitude (%):", self._amplitude_spin)
        param_layout.addRow("Speed (RPM):", self._speed_spin)
        param_layout.addRow("LPF cutoff (Hz):", self._filter_cutoff_spin)
        param_layout.addRow("Injection (%):", self._injection_spin)
        param_layout.addRow("PWM alignment:", self._alignment_choice)
        param_layout.addRow("Dead time (us):", self._dead_time_spin)
        param_layout.addRow("Diode Vf (V):", self._diode_vf_spin)
        param_layout.addRow("Current phase (deg):", self._current_phase_spin)

        self._pwm_freq_spin.valueChanged.connect(self._update_dynamic_constraints)
        self._update_dynamic_constraints()

        # Modulation selection
        modulation_group = QtWidgets.QGroupBox("Modulation selection")
        modulation_group.setAccessibleName("Modulation selection group")
        modulation_group.setAccessibleDescription(
            "List of available PWM and SVM modulation modes. "
            "Select a mode to configure and simulate."
        )
        modulation_layout = QtWidgets.QVBoxLayout(modulation_group)
        self._modulation_list = QtWidgets.QListWidget()
        self._modulation_list.setAccessibleName("Modulation list")
        self._modulation_list.setAccessibleDescription(
            "Scrollable list of modulation techniques: SPWM, THIPWM, SVM, DPWM, and custom variants. "
            "Use arrow keys to move between entries; press Enter to select."
        )
        self._modulation_list.setToolTip("Select the modulation technique to simulate")

        for mode in ModulationMode:
            item = QtWidgets.QListWidgetItem(mode.value)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, mode)
            self._modulation_list.addItem(item)

        self._modulation_list.currentItemChanged.connect(self._on_modulation_changed)

        modulation_layout.addWidget(self._modulation_list)

        # Display options
        display_group = QtWidgets.QGroupBox("Display options")
        display_group.setAccessibleName("Display options group")
        display_group.setAccessibleDescription(
            "Controls what is shown in the plot: line or phase voltages, "
            "LPF-filtered curve overlay, and switching-edge markers."
        )
        display_layout = QtWidgets.QVBoxLayout(display_group)

        self._voltage_choice = QtWidgets.QComboBox()
        self._voltage_choice.setAccessibleName("Voltage view")
        self._voltage_choice.setAccessibleDescription(
            "Line voltages show inverter terminal to DC-minus (0 to Vdc). "
            "Phase voltages show terminal-to-terminal across delta winding."
        )
        self._voltage_choice.addItems(["Line voltages", "Phase voltages"])
        self._voltage_choice.setToolTip(
            "Line voltages: inverter terminal to DC− (0…Vdc). "
            "Phase voltages: across delta winding, terminal-to-terminal (−Vdc…+Vdc)."
        )

        self._filter_checkbox = QtWidgets.QCheckBox("Filtered (fundamental)")
        self._filter_checkbox.setAccessibleName("Filtered waveform checkbox")
        self._filter_checkbox.setToolTip("Enable filtered (low-pass) waveform display")

        self._edges_checkbox = QtWidgets.QCheckBox("Show switching edges")
        self._edges_checkbox.setAccessibleName("Show switching edges checkbox")
        self._edges_checkbox.setToolTip(
            "Overlay markers at PWM switching edges for each phase"
        )

        self._run_button = QtWidgets.QPushButton("Update")
        self._run_button.setAccessibleName("Update simulation")
        self._run_button.setToolTip("Re-run the simulation with the current parameters")
        self._run_button.clicked.connect(self._on_update_clicked)

        display_layout.addWidget(self._voltage_choice)
        display_layout.addWidget(self._filter_checkbox)
        display_layout.addWidget(self._edges_checkbox)
        display_layout.addWidget(self._run_button)

        # Oscilloscope controls
        osc_group = QtWidgets.QGroupBox("Oscilloscope")
        osc_group.setAccessibleName("Oscilloscope controls group")
        osc_group.setAccessibleDescription(
            "Controls for the live waveform display: pause/resume, single-step, zoom reset, and CSV/PNG export."
        )
        osc_layout = QtWidgets.QVBoxLayout(osc_group)

        self._pause_button = QtWidgets.QPushButton("Pause")
        self._pause_button.setAccessibleName("Pause oscilloscope")
        self._pause_button.setAccessibleDescription(
            "Pause or resume the automatic waveform scrolling."
        )
        self._pause_button.setToolTip(
            "Pause or resume the oscilloscope waveform scrolling"
        )
        self._pause_button.clicked.connect(self._toggle_pause)

        self._step_button = QtWidgets.QPushButton("Step")
        self._step_button.setAccessibleName("Step oscilloscope")
        self._step_button.setAccessibleDescription(
            "Advance the waveform display by one scroll step when the oscilloscope is paused."
        )
        self._step_button.setToolTip("Advance the oscilloscope one frame when paused")
        self._step_button.clicked.connect(self._step_once)

        self._reset_zoom_button = QtWidgets.QPushButton("Reset zoom")
        self._reset_zoom_button.setAccessibleName("Reset plot zoom")
        self._reset_zoom_button.setAccessibleDescription(
            "Return the waveform and FFT views to auto-range after a manual zoom."
        )
        self._reset_zoom_button.setToolTip(
            "Reset waveform view to full auto-range after manual zoom"
        )
        self._reset_zoom_button.clicked.connect(self._reset_zoom)

        self._export_csv_button = QtWidgets.QPushButton("Export CSV")
        self._export_csv_button.setAccessibleName("Export waveform CSV")
        self._export_csv_button.setAccessibleDescription(
            "Save the current three-phase waveform to a comma-separated values file."
        )
        self._export_csv_button.setToolTip(
            "Export the current waveform data to a CSV file"
        )
        self._export_csv_button.clicked.connect(self._export_waveform_csv)

        self._export_png_button = QtWidgets.QPushButton("Export PNG")
        self._export_png_button.setAccessibleName("Export plot PNG")
        self._export_png_button.setAccessibleDescription(
            "Save a screenshot of the current waveform and FFT plots to a PNG image file."
        )
        self._export_png_button.setToolTip("Export the current plots as a PNG image")
        self._export_png_button.clicked.connect(self._export_plot_png)

        self._save_ref_button = QtWidgets.QPushButton("Save Reference")
        self._save_ref_button.setAccessibleName("Save reference snapshot")
        self._save_ref_button.setAccessibleDescription(
            "Freeze the current simulation result as a reference overlay for comparison."
        )
        self._save_ref_button.setToolTip(
            "Snapshot the current result as a grey dashed overlay for visual comparison"
        )
        self._save_ref_button.clicked.connect(self._on_save_reference)

        self._clear_ref_button = QtWidgets.QPushButton("Clear Reference")
        self._clear_ref_button.setAccessibleName("Clear reference snapshot")
        self._clear_ref_button.setAccessibleDescription(
            "Remove the grey dashed reference overlay from all plots."
        )
        self._clear_ref_button.setToolTip("Remove the comparison reference overlay")
        self._clear_ref_button.setEnabled(False)
        self._clear_ref_button.clicked.connect(self._on_clear_reference)

        osc_layout.addWidget(self._pause_button)
        osc_layout.addWidget(self._step_button)
        osc_layout.addWidget(self._reset_zoom_button)
        osc_layout.addWidget(self._export_csv_button)
        osc_layout.addWidget(self._export_png_button)
        osc_layout.addWidget(self._save_ref_button)
        osc_layout.addWidget(self._clear_ref_button)

        for grp in (param_group, modulation_group, display_group, osc_group):
            grp.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Maximum,
            )

        layout.addWidget(param_group, stretch=0)
        layout.addWidget(modulation_group, stretch=0)
        layout.addWidget(display_group, stretch=0)
        layout.addWidget(osc_group, stretch=0)
        layout.addStretch(1)

        return cp_widget

    def _apply_config_to_ui(self, config: SimulatorConfig) -> None:
        """Populate all widget values from the given configuration object."""
        self._pole_pairs_spin.setValue(config.motor_pole_pairs)
        self._pwm_freq_spin.setValue(config.pwm_frequency_hz)
        self._speed_spin.setValue(config.speed_rpm)
        self._battery_voltage_spin.setValue(config.battery_voltage)
        self._author_name_edit.setText(config.author_name)
        self._project_name_edit.setText(config.project_name)
        self._filter_cutoff_spin.setValue(config.filter_cutoff_hz)
        self._injection_spin.setValue(config.injection_percent)
        for idx in range(self._alignment_choice.count()):
            if self._alignment_choice.itemData(idx) == config.alignment:
                self._alignment_choice.setCurrentIndex(idx)
                break
        self._dead_time_spin.setValue(config.dead_time_us)
        self._diode_vf_spin.setValue(config.diode_forward_voltage_v)
        self._current_phase_spin.setValue(config.current_phase_deg)
        self._update_dynamic_constraints()
        self._filter_checkbox.setChecked(config.show_filtered)
        self._edges_checkbox.setChecked(config.show_switching_edges)
        self._amplitude_spin.setValue(config.amplitude_percent)
        # index 0 = Line voltages, index 1 = Phase voltages
        self._voltage_choice.setCurrentIndex(1 if config.show_phase_voltages else 0)

        # Select the current modulation in the list
        for idx in range(self._modulation_list.count()):
            item = self._modulation_list.item(idx)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == config.modulation:
                self._modulation_list.setCurrentRow(idx)
                break

    def _read_ui_to_config(self) -> SimulatorConfig:
        """Read all widget values and return a populated SimulatorConfig."""
        modulation_item = self._modulation_list.currentItem()
        modulation = (
            modulation_item.data(QtCore.Qt.ItemDataRole.UserRole)
            if modulation_item is not None
            else ModulationMode.SVM
        )

        return SimulatorConfig(
            motor_pole_pairs=self._pole_pairs_spin.value(),
            pwm_frequency_hz=self._pwm_freq_spin.value(),
            speed_rpm=self._speed_spin.value(),
            battery_voltage=self._battery_voltage_spin.value(),
            amplitude_percent=self._amplitude_spin.value(),
            modulation=modulation,
            show_phase_voltages=self._voltage_choice.currentIndex() == 1,
            show_filtered=self._filter_checkbox.isChecked(),
            show_switching_edges=self._edges_checkbox.isChecked(),
            filter_cutoff_hz=self._filter_cutoff_spin.value(),
            injection_percent=self._injection_spin.value(),
            alignment=self._alignment_choice.currentData(),
            dead_time_us=self._dead_time_spin.value(),
            diode_forward_voltage_v=self._diode_vf_spin.value(),
            current_phase_deg=self._current_phase_spin.value(),
            author_name=self._author_name_edit.text(),
            project_name=self._project_name_edit.text(),
            num_cycles=10,
            display_cycles=3,
        )

    def _update_dynamic_constraints(self) -> None:
        """Update parameter bounds that depend on other parameters."""

        pwm_hz = max(1.0, self._pwm_freq_spin.value())
        pwm_period_us = 1e6 / pwm_hz
        # Keep dead time below half-period so pulses keep a meaningful ON interval.
        dead_time_max_us = 0.49 * pwm_period_us
        self._dead_time_spin.setRange(0.0, dead_time_max_us)
        if self._dead_time_spin.value() > dead_time_max_us:
            self._dead_time_spin.setValue(dead_time_max_us)

    def _on_update_clicked(self) -> None:
        """Handle the Update button click: re-read UI and trigger a new simulation."""
        self._config = self._read_ui_to_config()
        self._update_simulation()

    def _on_modulation_changed(self, current: QtWidgets.QListWidgetItem | None) -> None:
        """Enable/disable controls based on modulation selection."""

        if current is None:
            return

        mode = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self._injection_spin.setEnabled(mode == ModulationMode.CUSTOM_THIPWM)

    def _start_simulation_loop(self) -> None:
        """Create and start the QTimer that drives oscilloscope scrolling."""
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._scroll_plot)
        self._timer.start()

    def _start_simulation_worker(self) -> None:
        """Start or restart the simulation worker thread."""

        # If a worker is already running, stop it cleanly.
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(100)

        self._worker_thread = QtCore.QThread()
        self._worker = SimulationWorker(self._config)
        self._worker.moveToThread(self._worker_thread)
        self._worker.finished.connect(self._on_simulation_finished)
        self._worker_thread.started.connect(self._worker.run)
        self._worker_thread.start()

    def _update_simulation(self) -> None:
        """Request simulation update with current configuration."""

        self._config = self._read_ui_to_config()
        self._start_simulation_worker()

    def _on_simulation_finished(self, result) -> None:
        """Callback when the background simulation finishes."""

        self._sim_result = result
        self._scroll_index = 0

        total_samples = self._sim_result.time.size
        self._window_samples = int(
            total_samples * self._config.display_cycles / self._config.num_cycles
        )
        self._window_samples = max(3, self._window_samples)

        # Line voltages = inverter terminal to DC− (0…Vdc) = phase_a/b/c.
        # Phase voltages = across delta winding, terminal-to-terminal = phase_voltage_ab/bc/ca.
        if self._config.show_phase_voltages:
            self._display_signals = {
                "A": self._sim_result.phase_voltage_ab,
                "B": self._sim_result.phase_voltage_bc,
                "C": self._sim_result.phase_voltage_ca,
            }
        else:
            self._display_signals = {
                "A": (
                    self._sim_result.filtered_phase_a
                    if self._config.show_filtered
                    else self._sim_result.phase_a
                ),
                "B": (
                    self._sim_result.filtered_phase_b
                    if self._config.show_filtered
                    else self._sim_result.phase_b
                ),
                "C": (
                    self._sim_result.filtered_phase_c
                    if self._config.show_filtered
                    else self._sim_result.phase_c
                ),
            }

        # Update the static plots (FFT, duty cycle, and info) as they do not scroll.
        self._plot_canvas.update_fft(
            self._sim_result.fft_freqs, self._sim_result.fft_magnitude
        )
        self._plot_canvas.update_duty_cycle(
            self._sim_result.duty_cycle_time,
            {
                "A": self._sim_result.duty_cycle_a,
                "B": self._sim_result.duty_cycle_b,
                "C": self._sim_result.duty_cycle_c,
            },
            dead_time_duty_limit=self._sim_result.dead_time_duty_limit,
        )
        self._plot_canvas.update_duty_fft(
            self._sim_result.duty_cycle_fft_freqs,
            self._sim_result.duty_cycle_fft_magnitude,
        )
        self._plot_canvas.update_cmv(
            self._sim_result.time,
            self._sim_result.cmv,
        )
        self._plot_canvas.update_dc_bus_ripple(
            self._sim_result.duty_cycle_time,
            self._sim_result.dc_bus_current_norm,
        )
        # Refresh reference static overlays if a reference snapshot is active.
        if self._ref_result is not None:
            self._plot_canvas.set_reference_static(
                self._ref_result.fft_freqs,
                self._ref_result.fft_magnitude,
                self._ref_result.duty_cycle_time,
                {
                    "A": self._ref_result.duty_cycle_a,
                    "B": self._ref_result.duty_cycle_b,
                    "C": self._ref_result.duty_cycle_c,
                },
            )
        self._update_info_text()

        # Refresh the dq phasor dialog if it is currently open.
        if self._dq_dialog is not None and self._dq_dialog.isVisible():
            self._dq_dialog.refresh(self._sim_result, self._config)

        # Set an appropriate scroll step to simulate an oscilloscope sweep.
        self._scroll_step = max(1, int(self._window_samples / 20))
        self._scroll_plot()

    def _scroll_plot(self) -> None:
        """Scroll the waveform display to simulate an oscilloscope trace."""

        if self._sim_result is None:
            return

        total_samples = self._sim_result.time.size
        if total_samples <= self._window_samples:
            start = 0
        else:
            start = self._scroll_index
        end = start + self._window_samples
        if end > total_samples:
            end = total_samples
            start = end - self._window_samples

        window_time = self._sim_result.time[start:end]
        window_phases = {
            "A": self._display_signals["A"][start:end],
            "B": self._display_signals["B"][start:end],
            "C": self._display_signals["C"][start:end],
        }

        # Provide switching edge markers for an interactive PWM switching view
        switch_times = {
            phase: self._compute_switch_times(window_time, window_phases[phase])
            for phase in ("A", "B", "C")
        }

        self._plot_canvas.update_waveform(
            window_time,
            window_phases,
            switch_times if self._config.show_switching_edges else None,
        )

        # Reference waveform overlay — same scroll window, clamped to ref length.
        if self._ref_result is not None and self._ref_display_signals:
            ref_total = self._ref_result.time.size
            ref_window = min(self._window_samples, ref_total)
            ref_start = min(start, max(0, ref_total - ref_window))
            ref_end = ref_start + ref_window
            self._plot_canvas.update_reference_waveform(
                self._ref_result.time[ref_start:ref_end],
                {
                    ph: self._ref_display_signals[ph][ref_start:ref_end]
                    for ph in ("A", "B", "C")
                },
            )

        self._scroll_index += self._scroll_step
        if self._scroll_index + self._window_samples >= total_samples:
            self._scroll_index = 0

    def _update_info_text(self) -> None:
        sim = self._sim_result
        if sim is None:
            return

        # Format top harmonics for screen-reader friendly output
        top_harmonics_lines = [
            f"  {freq:.0f} Hz: {mag:.2f}" for freq, mag in sim.top_harmonics
        ]

        # Only show the injection factor when the user has selected the custom THIPWM mode.
        injection_line = ""
        if self._config.modulation == ModulationMode.CUSTOM_THIPWM:
            injection_line = f"Injection: {self._config.injection_percent:.1f}%\n"

        # Keep metrics concise: one line voltage (A) and one phase voltage (AB).
        line_signal = (
            sim.filtered_phase_a if self._config.show_filtered else sim.phase_a
        )
        line_label = (
            "Filtered line voltage A"
            if self._config.show_filtered
            else "Line voltage A"
        )
        line_mean = float(np.mean(line_signal))
        line_rms = float(np.sqrt(np.mean(line_signal**2)))
        line_min = float(np.min(line_signal))
        line_max = float(np.max(line_signal))

        phase_signal = (
            sim.filtered_phase_a - sim.filtered_phase_b
            if self._config.show_filtered
            else sim.phase_voltage_ab
        )
        phase_label = (
            "Filtered phase voltage AB"
            if self._config.show_filtered
            else "Phase voltage AB"
        )

        phase_mean = float(np.mean(phase_signal))
        phase_rms = float(np.sqrt(np.mean(phase_signal**2)))
        phase_min = float(np.min(phase_signal))
        phase_max = float(np.max(phase_signal))

        info_text = (
            f"Project: {self._config.project_name or 'N/A'}\n"
            f"Author: {self._config.author_name or 'N/A'}\n"
            f"Modulation: {self._config.modulation.value}\n"
            + injection_line
            + f"Pole pairs: {self._config.motor_pole_pairs}\n"
            f"Battery voltage: {self._config.battery_voltage:.1f} V\n"
            f"PWM frequency: {self._config.pwm_frequency_hz:.0f} Hz\n"
            f"Requested speed: {self._config.speed_rpm:.2f} RPM\n"
            f"Real speed (quantized by PWM pulses): {sim.actual_speed_rpm:.2f} RPM\n"
            f"Speed deviation: {sim.speed_deviation_rpm:+.2f} RPM ({sim.speed_deviation_percent:+.3f}%)\n"
            f"Electrical frequency (real): {(sim.actual_speed_rpm / 60.0) * self._config.motor_pole_pairs:.3f} Hz\n"
            f"Average phase PWM pulses per electrical cycle: {sim.pulses_per_electrical_cycle}\n"
            f"Electrical degrees per PWM pulse: {sim.degrees_per_pwm_pulse:.2f}°\n"
            f"LPF cutoff: {self._config.filter_cutoff_hz or (3.0 * (sim.actual_speed_rpm / 60.0) * self._config.motor_pole_pairs):.1f} Hz\n"
            f"PWM alignment: {self._config.alignment.value}\n"
            f"Dead time: {self._config.dead_time_us:.2f} us\n"
            f"Diode Vf: {self._config.diode_forward_voltage_v:.3f} V\n"
            f"Current phase: {self._config.current_phase_deg:.1f}°\n"
            f"THD line voltage A: {sim.thd_line_percent:.2f}%\n"
            f"THD phase voltage AB: {sim.thd_phase_percent:.2f}%\n\n"
            "THD basis: both THD values are computed on filtered analysis waveforms.\n\n"
            "THD note: line A includes common-mode (triplen) content, while phase AB cancels it.\n"
            "Filtering note: filtered waveforms are fundamental envelopes, so they usually do not hit 0 V or Vbatt rails.\n\n"
            f"{line_label}: mean {line_mean:.2f} V, RMS {line_rms:.2f} V, min {line_min:.2f} V, max {line_max:.2f} V\n"
            f"{phase_label}: mean {phase_mean:.2f} V, RMS {phase_rms:.2f} V, min {phase_min:.2f} V, max {phase_max:.2f} V\n\n"
            "─── Duty Cycle Metrics ───\n"
            f"Line duty A: min {sim.duty_cycle_a_min * 100:.2f}%,"
            f" max {sim.duty_cycle_a_max * 100:.2f}%,"
            f" avg {sim.duty_cycle_a_mean * 100:.2f}%,"
            f" RMS {sim.duty_cycle_a_rms * 100:.2f}%\n"
            f"Line duty B: min {sim.duty_cycle_b_min * 100:.2f}%,"
            f" max {sim.duty_cycle_b_max * 100:.2f}%,"
            f" avg {sim.duty_cycle_b_mean * 100:.2f}%,"
            f" RMS {sim.duty_cycle_b_rms * 100:.2f}%\n"
            f"Line duty C: min {sim.duty_cycle_c_min * 100:.2f}%,"
            f" max {sim.duty_cycle_c_max * 100:.2f}%,"
            f" avg {sim.duty_cycle_c_mean * 100:.2f}%,"
            f" RMS {sim.duty_cycle_c_rms * 100:.2f}%\n"
            f"Phase duty AB (A-B): min {sim.duty_cycle_ab_min * 100:.2f}%,"
            f" max {sim.duty_cycle_ab_max * 100:.2f}%,"
            f" avg {sim.duty_cycle_ab_mean * 100:.2f}%,"
            f" RMS {sim.duty_cycle_ab_rms * 100:.2f}%\n"
            f"Phase duty BC (B-C): min {sim.duty_cycle_bc_min * 100:.2f}%,"
            f" max {sim.duty_cycle_bc_max * 100:.2f}%,"
            f" avg {sim.duty_cycle_bc_mean * 100:.2f}%,"
            f" RMS {sim.duty_cycle_bc_rms * 100:.2f}%\n"
            f"Phase duty CA (C-A): min {sim.duty_cycle_ca_min * 100:.2f}%,"
            f" max {sim.duty_cycle_ca_max * 100:.2f}%,"
            f" avg {sim.duty_cycle_ca_mean * 100:.2f}%,"
            f" RMS {sim.duty_cycle_ca_rms * 100:.2f}%\n"
            f"Dead-time duty loss: {sim.dead_time_duty_limit * 100:.3f}%"
            f" → D_max = {(1.0 - sim.dead_time_duty_limit) * 100:.3f}%,"
            f" D_min = {sim.dead_time_duty_limit * 100:.3f}%\n\n"
            "─── Common Mode Voltage ───\n"
            f"CMV mean: {sim.cmv_mean:.3f} V,  RMS: {sim.cmv_rms:.3f} V\n"
            f"CMV min: {sim.cmv_min:.3f} V,  max: {sim.cmv_max:.3f} V\n"
            f"CMV peak-to-peak: {sim.cmv_pp:.3f} V\n\n"
            "─── DC Bus Current Ripple (normalised) ───\n"
            f"DC bus min: {sim.dc_bus_current_norm_min:.4f},  max: {sim.dc_bus_current_norm_max:.4f}\n"
            f"DC bus RMS: {sim.dc_bus_current_norm_rms:.4f}\n"
            f"DC bus peak-to-peak: {sim.dc_bus_current_norm_pp:.4f}\n\n"
            f"Top harmonics (freq -> magnitude):\n"
            + "\n".join(top_harmonics_lines)
            + "\n\n"
            f"Show switching edges: {'Yes' if self._config.show_switching_edges else 'No'}\n\n"
            + sim.description_text
        )

        if self._ref_result is not None:
            _r = self._ref_result
            info_text += (
                "\n─── Comparison vs Reference ───\n"
                f"Ref THD line: {_r.thd_line_percent:.2f}%"
                f"  →  ΔTHD line: {sim.thd_line_percent - _r.thd_line_percent:+.2f}%\n"
                f"Ref THD phase: {_r.thd_phase_percent:.2f}%"
                f"  →  ΔTHD phase: {sim.thd_phase_percent - _r.thd_phase_percent:+.2f}%\n"
                f"Ref CMV pp: {_r.cmv_pp:.3f} V"
                f"  →  ΔCMV pp: {sim.cmv_pp - _r.cmv_pp:+.3f} V\n"
                f"Ref DC bus pp: {_r.dc_bus_current_norm_pp:.4f}"
                f"  →  ΔDC bus pp: {sim.dc_bus_current_norm_pp - _r.dc_bus_current_norm_pp:+.4f}\n"
            )

        self._info_box.setPlainText(info_text)

    def _copy_explanation_to_clipboard(self) -> None:
        """Copy the info box text to the system clipboard."""

        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(self._info_box.toPlainText())

    def _on_save_reference(self) -> None:
        """Freeze the current simulation result as a comparison reference overlay."""

        if self._sim_result is None:
            return

        self._ref_result = self._sim_result
        if self._config.show_phase_voltages:
            self._ref_display_signals = {
                "A": self._ref_result.phase_voltage_ab,
                "B": self._ref_result.phase_voltage_bc,
                "C": self._ref_result.phase_voltage_ca,
            }
        else:
            self._ref_display_signals = {
                "A": (
                    self._ref_result.filtered_phase_a
                    if self._config.show_filtered
                    else self._ref_result.phase_a
                ),
                "B": (
                    self._ref_result.filtered_phase_b
                    if self._config.show_filtered
                    else self._ref_result.phase_b
                ),
                "C": (
                    self._ref_result.filtered_phase_c
                    if self._config.show_filtered
                    else self._ref_result.phase_c
                ),
            }
        self._plot_canvas.set_reference_static(
            self._ref_result.fft_freqs,
            self._ref_result.fft_magnitude,
            self._ref_result.duty_cycle_time,
            {
                "A": self._ref_result.duty_cycle_a,
                "B": self._ref_result.duty_cycle_b,
                "C": self._ref_result.duty_cycle_c,
            },
        )
        self._clear_ref_button.setEnabled(True)
        self._update_info_text()

    def _on_clear_reference(self) -> None:
        """Remove the comparison reference overlay from all plots."""

        self._ref_result = None
        self._ref_display_signals = {}
        self._plot_canvas.clear_reference()
        self._clear_ref_button.setEnabled(False)
        self._update_info_text()

    def _compute_switch_times(self, time: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Compute the times at which a digital PWM signal switches state."""

        # Detect changes in the binary +/-1 signal and return the corresponding time stamps.
        transitions = np.where(np.diff(signal) != 0)[0]
        # +1 because diff shifts indices by 1
        return time[transitions + 1]

    def _reset_zoom(self) -> None:
        """Re-enable auto-range on the waveform plot after manual zoom."""
        self._plot_canvas.reset_zoom()

    def _toggle_pause(self) -> None:
        """Toggle pause/resume of the oscilloscope scrolling."""

        if self._timer is None:
            return

        if self._timer.isActive():
            self._timer.stop()
            self._pause_button.setText("Resume")
            self._pause_button.setToolTip("Resume the oscilloscope scrolling")
        else:
            self._timer.start()
            self._pause_button.setText("Pause")
            self._pause_button.setToolTip("Pause the oscilloscope scrolling")

    def _step_once(self) -> None:
        """Advance the oscilloscope display by one step (useful when paused)."""

        if self._timer is not None and self._timer.isActive():
            # Ensure we are paused first
            self._timer.stop()
            self._pause_button.setText("Resume")
        self._scroll_plot()

    def _export_waveform_csv(self) -> None:
        """Export the current waveform data to a CSV file."""

        if self._sim_result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export waveform CSV",
            "svm-analyst_waveform.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return

        if self._config.show_phase_voltages:
            labels = ["Phase voltage AB", "Phase voltage BC", "Phase voltage CA"]
            simulated = self._sim_result
            # Create a temporary SimulationResult with phase voltages in phase positions
            simulated = type(simulated)(
                time=simulated.time,
                phase_a=simulated.phase_voltage_ab,
                phase_b=simulated.phase_voltage_bc,
                phase_c=simulated.phase_voltage_ca,
                phase_voltage_ab=simulated.phase_voltage_ab,
                phase_voltage_bc=simulated.phase_voltage_bc,
                phase_voltage_ca=simulated.phase_voltage_ca,
                filtered_phase_a=simulated.phase_voltage_ab,
                filtered_phase_b=simulated.phase_voltage_bc,
                filtered_phase_c=simulated.phase_voltage_ca,
                fft_freqs=simulated.fft_freqs,
                fft_magnitude=simulated.fft_magnitude,
                thd_line_percent=simulated.thd_line_percent,
                thd_phase_percent=simulated.thd_phase_percent,
                top_harmonics=simulated.top_harmonics,
                pulses_per_electrical_cycle=simulated.pulses_per_electrical_cycle,
                degrees_per_pwm_pulse=simulated.degrees_per_pwm_pulse,
                actual_speed_rpm=simulated.actual_speed_rpm,
                speed_deviation_rpm=simulated.speed_deviation_rpm,
                speed_deviation_percent=simulated.speed_deviation_percent,
                filtered_mean=simulated.filtered_mean,
                filtered_rms=simulated.filtered_rms,
                filtered_min=simulated.filtered_min,
                filtered_max=simulated.filtered_max,
                raw_mean=simulated.raw_mean,
                raw_rms=simulated.raw_rms,
                raw_min=simulated.raw_min,
                raw_max=simulated.raw_max,
                description_text=simulated.description_text,
            )
        else:
            labels = ["Line A", "Line B", "Line C"]
            simulated = self._sim_result

        export_waveform_csv(path, simulated, labels)

    def _export_fft_csv(self) -> None:
        """Export the current FFT data to CSV."""

        if self._sim_result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export FFT CSV",
            "svm-analyst_fft.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return

        export_fft_csv(path, self._sim_result)

    def _export_plot_png(self) -> None:
        """Export the full plot canvas as a PNG image."""

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export plot image",
            "svm-analyst_plot.png",
            "PNG files (*.png)",
        )
        if not path:
            return

        pixmap = self._plot_canvas.grab_pixmap()
        if not pixmap.save(path, "PNG"):
            QtWidgets.QMessageBox.warning(
                self, "Export failed", f"Could not save image to:\n{path}"
            )

    def _export_report_pdf(self) -> None:
        """Export a multi-page PDF report including plots and explanation."""

        if self._sim_result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export report",
            "svm-analyst_report.pdf",
            "PDF files (*.pdf)",
        )
        if not path:
            return

        info_text = self._info_box.toPlainText()

        # Grab the pyqtgraph widget as a temporary PNG and pass its path to
        # the PDF exporter (which uses matplotlib imread to embed the image).
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            pixmap = self._plot_canvas.grab_pixmap()
            pixmap.save(tmp_path, "PNG")
            export_report_pdf(
                path,
                self._config,
                self._sim_result,
                info_text,
                show_phase_voltages=self._config.show_phase_voltages,
                plot_image_path=tmp_path,
                app_name="SVM Analyst",
                app_version=__version__,
                company_name="BLIND SYSTEMS",
                include_hexagon=True,
                include_harmonics_table=True,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _save_configuration(self) -> None:
        """Save the current simulation configuration to disk."""

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save configuration",
            "svm-analyst_config.json",
            "JSON files (*.json)",
        )
        if not path:
            return

        try:
            save_config(path, self._config)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Save failed",
                f"Could not save configuration:\n{exc}",
            )

    def _load_configuration(self) -> None:
        """Load a simulation configuration from disk."""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load configuration",
            "",
            "JSON files (*.json)",
        )
        if not path:
            return

        try:
            config = load_config(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Load failed",
                f"Could not load configuration:\n{exc}",
            )
            return

        self._config = config
        self._apply_config_to_ui(config)
        self._update_simulation()

    def _open_sweep_dialog(self) -> None:
        """Open a dialog to sweep a parameter and visualize THD."""

        dialog = SweepDialog(self._config, parent=self)
        dialog.exec()

    def _show_svm_hexagon(self) -> None:
        """Open a dialog showing the SVM hexagon and reference vector."""

        dialog = SvmHexagonDialog(self._config, parent=self)
        dialog.exec()

    def _show_dq_phasor(self) -> None:
        """Open (or bring to front) the dq-frame phasor diagram dialog."""
        if self._sim_result is None:
            QtWidgets.QMessageBox.information(
                self,
                "No simulation result",
                "Run the simulation first before opening the phasor diagram.",
            )
            return
        if self._dq_dialog is None or not self._dq_dialog.isVisible():
            self._dq_dialog = DqPhasorDialog(
                self._sim_result, self._config, parent=self
            )
            self._dq_dialog.show()
        else:
            self._dq_dialog.raise_()
            self._dq_dialog.activateWindow()

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "About SVM Analyst",
            "SVM Analyst\n"
            "Educational PWM/SVM modulation visualizer.\n"
            "Developed by Amine KHETTAT.",
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Ensure the background worker thread stops cleanly on exit."""

        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(100)

        super().closeEvent(event)


#: Name used for the Windows named mutex that prevents duplicate instances.
_SINGLE_INSTANCE_MUTEX_NAME = "Global\\SvmAnalyst_SingleInstanceMutex"

#: Fallback lock-file path used on non-Windows platforms.
_SINGLE_INSTANCE_LOCK_FILE = Path(tempfile.gettempdir()) / "svm_analyst.lock"


# ---------------------------------------------------------------------------
# Auto-update helpers (Qt workers + orchestrator)
# ---------------------------------------------------------------------------


class _UpdateCheckWorker(QtCore.QThread):
    """Background thread: queries GitHub for a newer release.

    Signals
    -------
    update_found(str, str)
        Emitted when a newer version exists; carries (tag_name, download_url).
    no_update()
        Emitted when the running version is already up to date or the network
        is unavailable.
    """

    update_found = Signal(str, str)
    no_update = Signal()

    def run(self) -> None:
        from .updater import is_update_available

        result = is_update_available()
        if result is not None:
            self.update_found.emit(result[0], result[1])
        else:
            self.no_update.emit()


class _DownloadWorker(QtCore.QThread):
    """Background thread: streams the new EXE to a temporary file.

    Parameters
    ----------
    url:
        Direct download URL for the new ``svm-analyst.exe``.

    Signals
    -------
    progress(int, int)
        Emitted periodically as ``(bytes_received, total_bytes)``.
        *total_bytes* is ``0`` when the server omits ``Content-Length``.
    finished(str)
        Emitted with the local path of the fully downloaded file.
    error(str)
        Emitted when the download fails; carries an error message.
    """

    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, url: str, parent=None) -> None:
        super().__init__(parent)
        self._url = url
        self._dest: Optional[str] = None

    def run(self) -> None:
        from .updater import download_update

        try:
            fd, dest = tempfile.mkstemp(suffix=".exe", prefix="svm_update_")
            os.close(fd)
            self._dest = dest
            download_update(self._url, dest, progress_callback=self.progress.emit)
            self.finished.emit(dest)
        except Exception as exc:
            if self._dest and os.path.exists(self._dest):
                try:
                    os.unlink(self._dest)
                except OSError:
                    pass
            self.error.emit(str(exc))


def _run_update_check(parent: QtWidgets.QWidget) -> None:
    """Silently check for updates then, if one is found, prompt the user.

    Runs the network request in a background :class:`_UpdateCheckWorker` so
    startup is never blocked.  If the user confirms the update, a progress
    dialog is shown while the EXE downloads, then ``apply_update`` is called
    and the application exits so the replacement script can take over.
    """

    worker = _UpdateCheckWorker(parent)

    def _on_update_found(tag: str, url: str) -> None:
        reply = QtWidgets.QMessageBox.question(
            parent,
            "Update available – SVM Analyst",
            f"A new version is available: <b>{tag}</b><br><br>"
            f"You are currently running <b>v{__version__}</b>.<br><br>"
            "Would you like to download and install the update now?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        _start_download(url)

    def _start_download(url: str) -> None:
        progress_dlg = QtWidgets.QProgressDialog(
            "Downloading update…",
            "Cancel",
            0,
            100,
            parent,
        )
        progress_dlg.setWindowTitle("SVM Analyst – Updating")
        progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dlg.setMinimumDuration(0)
        progress_dlg.setValue(0)

        dl_worker = _DownloadWorker(url, parent)

        def _on_progress(received: int, total: int) -> None:
            if total > 0:
                progress_dlg.setValue(int(received * 100 / total))
            else:
                # Unknown total – pulse the bar
                progress_dlg.setMaximum(0)

        def _on_finished(dest_path: str) -> None:
            progress_dlg.close()
            if not getattr(__import__("sys"), "frozen", False):
                QtWidgets.QMessageBox.information(
                    parent,
                    "Update downloaded",
                    f"Update downloaded to:\n{dest_path}\n\n"
                    "(Running from source – automatic replacement skipped.)",
                )
                return

            from .updater import apply_update

            try:
                apply_update(dest_path)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    parent,
                    "Update failed",
                    f"Could not apply the update:\n{exc}",
                )
                return

            QtWidgets.QMessageBox.information(
                parent,
                "Restarting",
                "The update will be applied after the application closes.\n"
                "SVM Analyst will restart automatically.",
            )
            QtWidgets.QApplication.instance().quit()

        def _on_error(msg: str) -> None:
            progress_dlg.close()
            QtWidgets.QMessageBox.critical(
                parent,
                "Download failed",
                f"Could not download the update:\n{msg}",
            )

        def _on_cancelled() -> None:
            dl_worker.terminate()

        dl_worker.progress.connect(_on_progress)
        dl_worker.finished.connect(_on_finished)
        dl_worker.error.connect(_on_error)
        progress_dlg.canceled.connect(_on_cancelled)
        dl_worker.start()

    worker.update_found.connect(_on_update_found)
    worker.start()


def _acquire_single_instance_lock():
    """Acquire a process-wide lock so only one instance can run.

    On Windows a named kernel mutex is created.  On other platforms a lock
    file is used instead.  Returns the lock handle/file-object on success,
    or ``None`` if another instance is already holding the lock.
    """
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.CreateMutexW(None, True, _SINGLE_INSTANCE_MUTEX_NAME)
        # ERROR_ALREADY_EXISTS == 183
        if handle and kernel32.GetLastError() != 183:
            return handle
        if handle:
            kernel32.CloseHandle(handle)
        return None
    else:
        import fcntl  # available on Linux / macOS

        try:
            lock_fh = open(_SINGLE_INSTANCE_LOCK_FILE, "w")  # noqa: WPS515
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_fh
        except OSError:
            return None


def _release_single_instance_lock(handle) -> None:
    """Release the lock previously acquired by :func:`_acquire_single_instance_lock`."""
    if handle is None:
        return
    if sys.platform == "win32":
        ctypes.windll.kernel32.ReleaseMutex(handle)  # type: ignore[attr-defined]
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    else:
        import fcntl

        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        except OSError:
            pass


def main(argv=None) -> int:
    """Launch the SVM Analyst application."""

    lock = _acquire_single_instance_lock()
    if lock is None:
        # Another instance is running — show a minimal Qt message box and exit.
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            argv or sys.argv
        )
        QtWidgets.QMessageBox.warning(
            None,
            "SVM Analyst – already running",
            "An instance of SVM Analyst is already open.\n"
            "Please use the existing window.",
        )
        return 1

    try:
        # Configure pyqtgraph appearance before creating any widget.
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOption("antialias", True)

        app = QtWidgets.QApplication(argv or sys.argv)
        app.setApplicationName("SVM Analyst")
        window = SvmShaperApp()
        window.showMaximized()
        # Trigger update check 3 s after launch so startup responsiveness is
        # not affected.  The worker runs in a background thread.
        QtCore.QTimer.singleShot(3000, lambda: _run_update_check(window))
        return app.exec()
    finally:
        _release_single_instance_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())
