"""Graphical user interface for the SVM Analyst simulator.

This module builds a PyQt6 application that allows users to select modulation
modes, configure system parameters, and visualize the resulting PWM signals and
harmonics in real time.

Accessibility notes:
- All interactive widgets have accessible names and descriptive tooltips.
- Keyboard navigation is supported through standard Qt focus handling.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
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
    export_plot_png,
    export_report_pdf,
    export_waveform_csv,
    load_config,
    save_config,
)
from .modulations import ModulationMode
from .sweep import sweep_thd
from .visualization import svm_hexagon_vertices, svm_reference_vector


class PlotCanvas(FigureCanvas):
    """Matplotlib canvas used for waveform and FFT plots."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        fig = Figure(figsize=(10, 6), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)

        self._wave_ax = fig.add_subplot(211)
        self._fft_ax = fig.add_subplot(212)

        self._wave_line_a = None
        self._wave_line_b = None
        self._wave_line_c = None
        self._fft_line = None

        # Optional switching edge markers (for interactive PWM switching display)
        self._switch_markers: dict[str, Optional[any]] = {
            "A": None,
            "B": None,
            "C": None,
        }

        self._init_plot()

    def _init_plot(self) -> None:
        self._wave_ax.set_title("Waveform")
        self._wave_ax.set_ylabel("Normalized voltage")
        self._wave_ax.set_xlabel("Time (s)")

        self._fft_ax.set_title("FFT (PWM signal)")
        self._fft_ax.set_ylabel("Magnitude")
        self._fft_ax.set_xlabel("Frequency (Hz)")

    def update_waveform(
        self,
        time: np.ndarray,
        phases: dict[str, np.ndarray],
        switch_times: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """Update the waveform plot with the latest data.

        If `switch_times` is provided, display markers at the PWM switching edges.
        """

        if self._wave_line_a is None:
            (self._wave_line_a,) = self._wave_ax.plot(
                time, phases["A"], label="Phase A"
            )
            (self._wave_line_b,) = self._wave_ax.plot(
                time, phases["B"], label="Phase B"
            )
            (self._wave_line_c,) = self._wave_ax.plot(
                time, phases["C"], label="Phase C"
            )
            self._wave_ax.legend(loc="upper right")
        else:
            self._wave_line_a.set_data(time, phases["A"])
            self._wave_line_b.set_data(time, phases["B"])
            self._wave_line_c.set_data(time, phases["C"])

        # Update switching edge markers when provided
        if switch_times is not None:
            # Remove old markers if they exist
            for phase in ("A", "B", "C"):
                if self._switch_markers.get(phase) is not None:
                    try:
                        self._switch_markers[phase].remove()
                    except Exception:
                        pass
                    self._switch_markers[phase] = None

            # Add new markers for each phase
            colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
            for phase, times in switch_times.items():
                if times is None or len(times) == 0:
                    continue
                self._switch_markers[phase] = self._wave_ax.scatter(
                    times,
                    np.zeros_like(times),
                    marker="|",
                    color=colors.get(phase, "black"),
                    s=80,
                    label=f"{phase} switches",
                    zorder=5,
                )

        self._wave_ax.set_xlim(time[0], time[-1])
        self._wave_ax.set_ylim(-1.15, 1.15)
        self._wave_ax.figure.canvas.draw_idle()

    def update_fft(self, freqs: np.ndarray, magnitude: np.ndarray) -> None:
        """Update the FFT plot."""

        if self._fft_line is None:
            (self._fft_line,) = self._fft_ax.plot(freqs, magnitude, label="FFT")
            self._fft_ax.set_xlim(0, freqs.max())
        else:
            self._fft_line.set_data(freqs, magnitude)
            self._fft_ax.set_xlim(0, freqs.max())

        self._fft_ax.set_ylim(0, max(1e-3, magnitude.max() * 1.1))
        self._fft_ax.figure.canvas.draw_idle()


class SimulationWorker(QtCore.QObject):
    """Worker for running simulation in a background thread."""

    finished = pyqtSignal(object)

    def __init__(self, config: SimulatorConfig):
        super().__init__()
        self._config = config

    def run(self) -> None:
        result = run_simulation(self._config)
        self.finished.emit(result)


class SweepDialog(QtWidgets.QDialog):
    """Dialog to sweep a parameter and visualize THD results."""

    def __init__(
        self, base_config: SimulatorConfig, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Sweep THD")
        self._base_config = base_config

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self._variable_choice = QtWidgets.QComboBox()
        self._variable_choice.addItems(["speed_rpm", "pwm_frequency_hz"])
        self._min_edit = QLineEdit("0")
        self._max_edit = QLineEdit("1000")
        self._steps_edit = QLineEdit("20")

        form.addRow("Variable:", self._variable_choice)
        form.addRow("Min:", self._min_edit)
        form.addRow("Max:", self._max_edit)
        form.addRow("Steps:", self._steps_edit)

        self._run_button = QPushButton("Run")
        self._run_button.clicked.connect(self._on_run)

        layout.addLayout(form)
        layout.addWidget(self._run_button)

        self._figure = Figure(figsize=(5, 3), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas)

        self._ax = self._figure.add_subplot(111)

    def _on_run(self) -> None:
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
        self._ax.clear()
        self._ax.plot(xs, thd, marker="o")
        self._ax.set_title(f"THD vs {variable}")
        self._ax.set_xlabel(variable)
        self._ax.set_ylabel("THD (%)")
        self._ax.grid(True)
        self._canvas.draw()


class SvmHexagonDialog(QtWidgets.QDialog):
    """Dialog showing the SVM hexagon and a rotating reference vector."""

    def __init__(
        self, config: SimulatorConfig, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("SVM Hexagon")
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


class SvmShaperApp(QtWidgets.QMainWindow):
    """Main window for the SVM Analyst application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SVM Analyst - PWM Modulation Simulator")

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

        self._build_ui()
        self._apply_config_to_ui(self._config)
        self._start_simulation_loop()
        self._start_simulation_worker()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)

        self._menu_bar = self.menuBar()
        self._build_menu()

        self._control_panel = self._create_control_panel()
        self._plot_canvas = PlotCanvas(parent=central)
        self._info_box = QPlainTextEdit(readOnly=True)
        self._info_box.setMinimumHeight(120)
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

        main_layout.addLayout(self._control_panel)
        main_layout.addWidget(self._plot_canvas, stretch=1)

        info_row = QtWidgets.QHBoxLayout()
        info_row.addWidget(self._info_box, stretch=1)
        info_row.addWidget(self._copy_explanation_button, stretch=0)
        main_layout.addLayout(info_row)

    def _build_menu(self) -> None:
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

        tools_menu = self._menu_bar.addMenu("&Tools")
        sweep_action = QtGui.QAction("Sweep THD...", self)
        sweep_action.triggered.connect(self._open_sweep_dialog)
        tools_menu.addAction(sweep_action)

        help_menu = self._menu_bar.addMenu("&Help")
        about_action = QtGui.QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_control_panel(self) -> QtWidgets.QHBoxLayout:
        layout = QtWidgets.QHBoxLayout()

        # Left side: parameter controls
        param_group = QtWidgets.QGroupBox("System parameters")
        param_layout = QtWidgets.QFormLayout(param_group)

        self._pole_pairs_spin = QtWidgets.QSpinBox()
        self._pole_pairs_spin.setRange(1, 20)
        self._pole_pairs_spin.setValue(self._config.motor_pole_pairs)
        self._pole_pairs_spin.setToolTip("Number of pole pairs in the PMSM")
        self._pole_pairs_spin.setAccessibleName("Motor pole pairs")

        self._pwm_freq_spin = QtWidgets.QDoubleSpinBox()
        self._pwm_freq_spin.setRange(100.0, 200000.0)
        self._pwm_freq_spin.setSingleStep(100.0)
        self._pwm_freq_spin.setValue(self._config.pwm_frequency_hz)
        self._pwm_freq_spin.setToolTip("PWM carrier frequency in Hz")
        self._pwm_freq_spin.setAccessibleName("PWM frequency")

        self._battery_voltage_spin = QtWidgets.QDoubleSpinBox()
        self._battery_voltage_spin.setRange(1.0, 1000.0)
        self._battery_voltage_spin.setSingleStep(10.0)
        self._battery_voltage_spin.setValue(self._config.battery_voltage)
        self._battery_voltage_spin.setToolTip("DC bus voltage (battery) in volts")
        self._battery_voltage_spin.setAccessibleName("Battery voltage")

        self._amplitude_spin = QtWidgets.QDoubleSpinBox()
        self._amplitude_spin.setRange(0.0, 100.0)
        self._amplitude_spin.setSingleStep(1.0)
        self._amplitude_spin.setValue(self._config.amplitude_percent)
        self._amplitude_spin.setToolTip(
            "Modulation amplitude as a percentage of full scale (0-100%)."
        )
        self._amplitude_spin.setAccessibleName("Modulation amplitude percent")

        self._speed_spin = QtWidgets.QDoubleSpinBox()
        self._speed_spin.setRange(0.0, 20000.0)
        self._speed_spin.setSingleStep(10.0)
        self._speed_spin.setValue(self._config.speed_rpm)
        self._speed_spin.setToolTip("Motor speed in RPM")
        self._speed_spin.setAccessibleName("Speed in RPM")

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

        self._injection_spin = QtWidgets.QDoubleSpinBox()
        self._injection_spin.setRange(0.0, 100.0)
        self._injection_spin.setSingleStep(1.0)
        self._injection_spin.setValue(self._config.injection_percent)
        self._injection_spin.setToolTip(
            "Third harmonic injection factor (percent of 1/6). Only used for custom THIPWM mode."
        )
        self._injection_spin.setAccessibleName("Third harmonic injection percent")
        self._injection_spin.setEnabled(False)

        param_layout.addRow("Author:", self._author_name_edit)
        param_layout.addRow("Project:", self._project_name_edit)
        param_layout.addRow("Pole pairs:", self._pole_pairs_spin)
        param_layout.addRow("PWM frequency (Hz):", self._pwm_freq_spin)
        param_layout.addRow("Battery voltage (V):", self._battery_voltage_spin)
        param_layout.addRow("Amplitude (%):", self._amplitude_spin)
        param_layout.addRow("Speed (RPM):", self._speed_spin)
        param_layout.addRow("LPF cutoff (Hz):", self._filter_cutoff_spin)
        param_layout.addRow("Injection (%):", self._injection_spin)

        # Modulation selection
        modulation_group = QtWidgets.QGroupBox("Modulation selection")
        modulation_layout = QtWidgets.QVBoxLayout(modulation_group)
        self._modulation_list = QtWidgets.QListWidget()
        self._modulation_list.setAccessibleName("Modulation list")
        self._modulation_list.setToolTip("Select the modulation technique to simulate")

        for mode in ModulationMode:
            item = QtWidgets.QListWidgetItem(mode.value)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, mode)
            self._modulation_list.addItem(item)

        self._modulation_list.currentItemChanged.connect(self._on_modulation_changed)

        modulation_layout.addWidget(self._modulation_list)

        # Display options
        display_group = QtWidgets.QGroupBox("Display options")
        display_layout = QtWidgets.QVBoxLayout(display_group)

        self._voltage_choice = QtWidgets.QComboBox()
        self._voltage_choice.setAccessibleName("Voltage view")
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
        osc_layout = QtWidgets.QVBoxLayout(osc_group)

        self._pause_button = QtWidgets.QPushButton("Pause")
        self._pause_button.setAccessibleName("Pause oscilloscope")
        self._pause_button.setToolTip(
            "Pause or resume the oscilloscope waveform scrolling"
        )
        self._pause_button.clicked.connect(self._toggle_pause)

        self._step_button = QtWidgets.QPushButton("Step")
        self._step_button.setAccessibleName("Step oscilloscope")
        self._step_button.setToolTip("Advance the oscilloscope one frame when paused")
        self._step_button.clicked.connect(self._step_once)

        self._export_csv_button = QtWidgets.QPushButton("Export CSV")
        self._export_csv_button.setAccessibleName("Export waveform CSV")
        self._export_csv_button.setToolTip(
            "Export the current waveform data to a CSV file"
        )
        self._export_csv_button.clicked.connect(self._export_waveform_csv)

        self._export_png_button = QtWidgets.QPushButton("Export PNG")
        self._export_png_button.setAccessibleName("Export plot PNG")
        self._export_png_button.setToolTip("Export the current plots as a PNG image")
        self._export_png_button.clicked.connect(self._export_plot_png)

        osc_layout.addWidget(self._pause_button)
        osc_layout.addWidget(self._step_button)
        osc_layout.addWidget(self._export_csv_button)
        osc_layout.addWidget(self._export_png_button)

        layout.addWidget(param_group, stretch=0)
        layout.addWidget(modulation_group, stretch=0)
        layout.addWidget(display_group, stretch=0)
        layout.addWidget(osc_group, stretch=0)

        return layout

    def _apply_config_to_ui(self, config: SimulatorConfig) -> None:
        self._pole_pairs_spin.setValue(config.motor_pole_pairs)
        self._pwm_freq_spin.setValue(config.pwm_frequency_hz)
        self._speed_spin.setValue(config.speed_rpm)
        self._battery_voltage_spin.setValue(config.battery_voltage)
        self._author_name_edit.setText(config.author_name)
        self._project_name_edit.setText(config.project_name)
        self._filter_cutoff_spin.setValue(config.filter_cutoff_hz)
        self._injection_spin.setValue(config.injection_percent)
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
            author_name=self._author_name_edit.text(),
            project_name=self._project_name_edit.text(),
            num_cycles=10,
            display_cycles=3,
        )

    def _on_update_clicked(self) -> None:
        self._config = self._read_ui_to_config()
        self._update_simulation()

    def _on_modulation_changed(self, current: QtWidgets.QListWidgetItem | None) -> None:
        """Enable/disable controls based on modulation selection."""

        if current is None:
            return

        mode = current.data(QtCore.Qt.ItemDataRole.UserRole)
        self._injection_spin.setEnabled(mode == ModulationMode.CUSTOM_THIPWM)

    def _start_simulation_loop(self) -> None:
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

        # Update the static plots (FFT and info) as they do not scroll.
        self._plot_canvas.update_fft(
            self._sim_result.fft_freqs, self._sim_result.fft_magnitude
        )
        self._update_info_text()

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
            f"THD line voltage A: {sim.thd_line_percent:.2f}%\n"
            f"THD phase voltage AB: {sim.thd_phase_percent:.2f}%\n\n"
            "THD basis: both THD values are computed on filtered analysis waveforms.\n\n"
            "THD note: line A includes common-mode (triplen) content, while phase AB cancels it.\n"
            "Filtering note: filtered waveforms are fundamental envelopes, so they usually do not hit 0 V or Vbatt rails.\n\n"
            f"{line_label}: mean {line_mean:.2f} V, RMS {line_rms:.2f} V, min {line_min:.2f} V, max {line_max:.2f} V\n"
            f"{phase_label}: mean {phase_mean:.2f} V, RMS {phase_rms:.2f} V, min {phase_min:.2f} V, max {phase_max:.2f} V\n\n"
            f"Top harmonics (freq -> magnitude):\n"
            + "\n".join(top_harmonics_lines)
            + "\n\n"
            f"Show switching edges: {'Yes' if self._config.show_switching_edges else 'No'}\n\n"
            + sim.description_text
        )

        self._info_box.setPlainText(info_text)

    def _copy_explanation_to_clipboard(self) -> None:
        """Copy the info box text to the system clipboard."""

        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(self._info_box.toPlainText())

    def _compute_switch_times(self, time: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Compute the times at which a digital PWM signal switches state."""

        # Detect changes in the binary +/-1 signal and return the corresponding time stamps.
        transitions = np.where(np.diff(signal) != 0)[0]
        # +1 because diff shifts indices by 1
        return time[transitions + 1]

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
            "svm_shaper_waveform.csv",
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
            "svm_shaper_fft.csv",
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
            "svm_shaper_plot.png",
            "PNG files (*.png)",
        )
        if not path:
            return

        export_plot_png(path, self._plot_canvas.figure)

    def _export_report_pdf(self) -> None:
        """Export a multi-page PDF report including plots and explanation."""

        if self._sim_result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export report",
            "svm_shaper_report.pdf",
            "PDF files (*.pdf)",
        )
        if not path:
            return

        info_text = self._info_box.toPlainText()
        export_report_pdf(
            path,
            self._config,
            self._sim_result,
            info_text,
            show_phase_voltages=self._config.show_phase_voltages,
            plot_figure=self._plot_canvas.figure,
            app_name="SVM Analyst",
            app_version=__version__,
            company_name="BLIND SYSTEMS",
            include_hexagon=True,
            include_harmonics_table=True,
        )

    def _save_configuration(self) -> None:
        """Save the current simulation configuration to disk."""

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save configuration",
            "svm_shaper_config.json",
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


def main(argv=None) -> int:
    """Launch the SVM Analyst application."""

    app = QtWidgets.QApplication(argv or sys.argv)
    app.setApplicationName("SVM Analyst")
    window = SvmShaperApp()
    window.resize(1200, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
