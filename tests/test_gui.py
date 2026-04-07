"""GUI unit tests: verify instantiation, basic state, and non-regression behavior.

Atomic features covered:
- SvmShaperApp window title and initial state
- SvmShaperApp._read_ui_to_config returns a valid SimulatorConfig
- SvmShaperApp._apply_config_to_ui round-trip
- PlotCanvas instantiation and update_waveform / update_fft
- PlotStylePanel instantiation and _reset_styles
- SweepDialog._on_run with valid / invalid inputs
- SvmHexagonDialog instantiation and _refresh
- SimulationWorker.run emits finished signal
- SvmShaperApp.closeEvent cleans up worker thread
- Single-instance lock: acquire succeeds first time, fails second time, releases correctly
- Window layout / auto-resize: PlotCanvas Expanding policy, info_box max height, minimum window size,
  control panel Maximum vertical policy, aspect ratio enforcement
"""

import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.usefixtures("qapp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_window(qapp):
    from svm_shaper.gui import SvmShaperApp

    win = SvmShaperApp()
    if win._timer is not None:
        win._timer.stop()
    if win._worker_thread is not None and win._worker_thread.isRunning():
        win._worker_thread.quit()
        win._worker_thread.wait(300)
    return win


# ---------------------------------------------------------------------------
# SvmShaperApp — window
# ---------------------------------------------------------------------------


class TestSvmShaperAppInit:
    @pytest.fixture(scope="class")
    def win(self, qapp):
        w = _make_window(qapp)
        yield w
        w.close()

    def test_window_title_contains_svm_analyst(self, win):
        assert "SVM Analyst" in win.windowTitle()

    def test_initial_config_is_simulator_config(self, win):
        from svm_shaper.core import SimulatorConfig

        assert isinstance(win._config, SimulatorConfig)

    def test_read_ui_to_config_returns_valid(self, win):
        from svm_shaper.core import SimulatorConfig

        cfg = win._read_ui_to_config()
        assert isinstance(cfg, SimulatorConfig)

    def test_apply_and_read_config_roundtrip(self, win):
        from svm_shaper.core import SimulatorConfig
        from svm_shaper.modulations import ModulationMode

        original = SimulatorConfig(
            motor_pole_pairs=3,
            speed_rpm=750.0,
            pwm_frequency_hz=8000.0,
            modulation=ModulationMode.SVM,
        )
        win._apply_config_to_ui(original)
        recovered = win._read_ui_to_config()
        assert recovered.motor_pole_pairs == 3
        assert recovered.speed_rpm == pytest.approx(750.0)
        assert recovered.pwm_frequency_hz == pytest.approx(8000.0)
        assert recovered.modulation == ModulationMode.SVM

    def test_pause_toggle(self, win):
        """Pause should stop the timer; Resume should restart it."""
        win._timer.start()  # ensure running
        win._toggle_pause()
        assert not win._timer.isActive()
        assert "Resume" in win._pause_button.text()
        win._toggle_pause()
        assert win._timer.isActive()
        assert "Pause" in win._pause_button.text()
        win._timer.stop()  # leave stopped for other tests


# ---------------------------------------------------------------------------
# PlotCanvas
# ---------------------------------------------------------------------------


class TestPlotCanvas:
    @pytest.fixture(scope="class")
    def canvas(self, qapp):
        from svm_shaper.gui import PlotCanvas

        c = PlotCanvas()
        yield c
        c.close()

    def test_instantiation(self, canvas):
        assert canvas is not None

    def test_update_waveform(self, canvas):
        t = np.linspace(0, 0.01, 200)
        phases = {
            "A": np.sin(2 * np.pi * 50 * t),
            "B": np.sin(2 * np.pi * 50 * t - 2 * np.pi / 3),
            "C": np.sin(2 * np.pi * 50 * t + 2 * np.pi / 3),
        }
        canvas.update_waveform(t, phases, switch_times=None)

    def test_update_fft(self, canvas):
        freqs = np.linspace(0, 5000, 200)
        mag = np.random.default_rng(1).random(200)
        canvas.update_fft(freqs, mag)

    def test_reset_zoom(self, canvas):
        canvas.reset_zoom()  # should not raise

    def test_update_style(self, canvas):
        canvas.update_style("A", color="#ff0000", width=2.0, dash="dash", symbol="x")

    def test_update_duty_cycle_creates_curves_in_percent(self, canvas):
        """update_duty_cycle must plot duty values scaled to percent (0-100)."""
        n = 100
        t = np.linspace(0, 1e-3, n)
        duty = {"A": np.ones(n) * 0.6, "B": np.ones(n) * 0.5, "C": np.ones(n) * 0.4}
        canvas.update_duty_cycle(t, duty)  # first call: creates curves
        # Verify the plotted Y data is in percent range
        data_a = canvas._duty_curves["A"].getData()[1]
        assert float(np.mean(data_a)) == pytest.approx(60.0, abs=0.01)

    def test_update_duty_cycle_second_call_updates_data(self, canvas):
        n = 80
        t = np.linspace(0, 1e-3, n)
        duty = {"A": np.ones(n) * 0.75, "B": np.ones(n) * 0.5, "C": np.ones(n) * 0.25}
        canvas.update_duty_cycle(t, duty)  # second call: updates existing curves
        data_a = canvas._duty_curves["A"].getData()[1]
        assert float(np.mean(data_a)) == pytest.approx(75.0, abs=0.01)

    def test_set_duty_phase_visible_hides_curve(self, canvas):
        """set_duty_phase_visible(False) must hide; True must show the curve."""
        # Ensure curves exist
        n = 50
        t = np.linspace(0, 1e-3, n)
        canvas.update_duty_cycle(
            t, {"A": np.ones(n) * 0.5, "B": np.ones(n) * 0.5, "C": np.ones(n) * 0.5}
        )
        canvas.set_duty_phase_visible("B", False)
        assert not canvas._duty_curves["B"].isVisible()
        canvas.set_duty_phase_visible("B", True)
        assert canvas._duty_curves["B"].isVisible()

    def test_duty_check_checkboxes_exist_for_all_phases(self, canvas):
        """PlotCanvas must expose _duty_check dict with checkboxes for A, B, C."""
        assert set(canvas._duty_check.keys()) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# PlotStylePanel
# ---------------------------------------------------------------------------


class TestPlotStylePanel:
    @pytest.fixture(scope="class")
    def panel(self, qapp):
        from svm_shaper.gui import PlotCanvas, PlotStylePanel

        canvas = PlotCanvas()
        psp = PlotStylePanel(canvas)
        yield psp, canvas
        psp.close()
        canvas.close()

    def test_instantiation(self, panel):
        psp, _ = panel
        assert psp is not None

    def test_reset_styles_does_not_raise(self, panel):
        psp, _ = panel
        psp._reset_styles()


# ---------------------------------------------------------------------------
# SweepDialog
# ---------------------------------------------------------------------------


class TestSweepDialog:
    @pytest.fixture(scope="class")
    def dialog(self, qapp):
        from svm_shaper.core import SimulatorConfig
        from svm_shaper.gui import SweepDialog
        from svm_shaper.modulations import ModulationMode

        # Use a very fast config so sweeps complete quickly in tests
        fast_cfg = SimulatorConfig(
            modulation=ModulationMode.SINUSOIDAL,
            num_cycles=1,
            oversample=10,
            pwm_frequency_hz=2000.0,
            speed_rpm=400.0,
            motor_pole_pairs=1,
        )
        dlg = SweepDialog(fast_cfg)
        yield dlg
        dlg.close()

    def test_instantiation(self, dialog):
        assert dialog is not None

    def test_window_title(self, dialog):
        assert "Sweep" in dialog.windowTitle()

    def test_run_with_invalid_range_does_not_crash(self, dialog, qapp, monkeypatch):
        """Non-numeric input in min/max fields should not raise an exception;
        _on_run should show a warning and return early (no sweep triggered)."""
        from PySide6 import QtWidgets

        # Suppress the QMessageBox so it doesn't block the test event loop
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kw: None)
        dialog._min_edit.setText("abc")
        dialog._max_edit.setText("100")
        dialog._steps_edit.setText("5")
        dialog._on_run()  # should handle error gracefully

    def test_run_with_valid_range(self, dialog, qapp):
        dialog._min_edit.setText("100")
        dialog._max_edit.setText("500")
        dialog._steps_edit.setText("3")
        dialog._on_run()  # should complete without exception


# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------


class TestSingleInstanceLock:
    """Verify that the single-instance guard correctly prevents duplicate launches."""

    def test_first_acquire_succeeds(self):
        from svm_shaper.gui import (
            _acquire_single_instance_lock,
            _release_single_instance_lock,
        )

        lock = _acquire_single_instance_lock()
        try:
            assert lock is not None
        finally:
            _release_single_instance_lock(lock)

    def test_second_acquire_fails_while_held(self):
        from svm_shaper.gui import (
            _acquire_single_instance_lock,
            _release_single_instance_lock,
        )

        lock1 = _acquire_single_instance_lock()
        try:
            lock2 = _acquire_single_instance_lock()
            assert lock2 is None
        finally:
            _release_single_instance_lock(lock1)

    def test_acquire_succeeds_after_release(self):
        from svm_shaper.gui import (
            _acquire_single_instance_lock,
            _release_single_instance_lock,
        )

        lock1 = _acquire_single_instance_lock()
        _release_single_instance_lock(lock1)

        lock2 = _acquire_single_instance_lock()
        try:
            assert lock2 is not None
        finally:
            _release_single_instance_lock(lock2)

    def test_release_none_is_safe(self):
        from svm_shaper.gui import _release_single_instance_lock

        _release_single_instance_lock(None)  # must not raise


# ---------------------------------------------------------------------------
# SvmHexagonDialog
# ---------------------------------------------------------------------------


class TestSvmHexagonDialog:
    @pytest.fixture(scope="class")
    def dialog(self, qapp):
        from svm_shaper.core import SimulatorConfig
        from svm_shaper.gui import SvmHexagonDialog

        dlg = SvmHexagonDialog(SimulatorConfig())
        dlg._timer.stop()
        yield dlg
        dlg.close()

    def test_instantiation(self, dialog):
        assert dialog is not None

    def test_window_title_contains_svm(self, dialog):
        assert "SVM" in dialog.windowTitle()

    def test_refresh_does_not_raise(self, dialog):
        dialog._refresh()


# ---------------------------------------------------------------------------
# Window layout / auto-resize
# ---------------------------------------------------------------------------


class TestWindowLayout:
    """Verify layout properties that guarantee graphs fill the window when maximized."""

    @pytest.fixture(scope="class")
    def win(self, qapp):
        w = _make_window(qapp)
        yield w
        w.close()

    def test_plot_canvas_horizontal_size_policy_is_expanding(self, win):
        from PySide6.QtWidgets import QSizePolicy

        assert (
            win._plot_canvas.sizePolicy().horizontalPolicy()
            == QSizePolicy.Policy.Expanding
        )

    def test_plot_canvas_vertical_size_policy_is_expanding(self, win):
        from PySide6.QtWidgets import QSizePolicy

        assert (
            win._plot_canvas.sizePolicy().verticalPolicy()
            == QSizePolicy.Policy.Expanding
        )

    def test_info_box_has_max_height(self, win):
        assert win._info_box.maximumHeight() == 180

    def test_info_box_min_height_respected(self, win):
        assert win._info_box.minimumHeight() == 120

    def test_minimum_window_size(self, win):
        from PySide6.QtCore import QSize

        assert win.minimumSize() == QSize(1280, 720)

    def test_control_panel_is_qwidget(self, win):
        from PySide6.QtWidgets import QWidget

        assert isinstance(win._control_panel, QWidget)

    def test_control_panel_vertical_policy_is_maximum(self, win):
        from PySide6.QtWidgets import QSizePolicy

        assert (
            win._control_panel.sizePolicy().verticalPolicy()
            == QSizePolicy.Policy.Maximum
        )

    def test_aspect_ratio_too_wide_is_corrected(self, win, qapp):
        from svm_shaper.gui import _MAX_ASPECT_RATIO

        win.show()
        qapp.processEvents()
        win.resize(4000, 400)  # extreme ultrawide — should be constrained
        qapp.processEvents()
        h = win.height()
        assert h > 0
        assert win.width() / h <= _MAX_ASPECT_RATIO + 0.05

    def test_aspect_ratio_too_tall_is_corrected(self, win, qapp):
        from svm_shaper.gui import _MIN_ASPECT_RATIO

        win.show()
        qapp.processEvents()
        win.resize(800, 2000)  # nearly portrait — should be corrected
        qapp.processEvents()
        h = win.height()
        assert h > 0
        assert win.width() / h >= _MIN_ASPECT_RATIO - 0.05


# ---------------------------------------------------------------------------
# SimulationWorker
# ---------------------------------------------------------------------------


class TestSimulationWorker:
    def test_run_emits_finished(self, qapp, qtbot):
        from svm_shaper.core import SimulatorConfig
        from svm_shaper.gui import SimulationWorker

        worker = SimulationWorker(SimulatorConfig(num_cycles=1, oversample=10))
        with qtbot.waitSignal(worker.finished, timeout=10_000):
            worker.run()
