"""Accessibility unit tests — verify that every interactive widget in the
SVM Analyst GUI has both an accessible name and an accessible description.

Atomic features covered:
- All top-level groups have setAccessibleName + setAccessibleDescription
- All QPushButton controls have accessible name and description
- All QDoubleSpinBox / QSpinBox controls have accessible name
- All QComboBox controls have accessible name and description
- QPlainTextEdit info box has accessible name and description
- PlotCanvas has accessible name and description
- PlotStylePanel has accessible name and description
- SweepDialog has accessible name and description
- SweepDialog widgets (_variable_choice, _min_edit, _max_edit, _steps_edit,
  _run_button, _plot_widget) have accessible names
- SvmHexagonDialog has accessible name and description
"""

import os

import pytest

# Keep matplotlib and Qt from opening display windows in CI
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.usefixtures("qapp")


@pytest.fixture(scope="module")
def app_window(qapp):
    """Instantiate the main window once for all tests in this module."""
    from svm_shaper.gui import SvmShaperApp

    win = SvmShaperApp()
    # Stop the background worker and timer so tests run quickly
    if win._timer is not None:
        win._timer.stop()
    if win._worker_thread is not None and win._worker_thread.isRunning():
        win._worker_thread.quit()
        win._worker_thread.wait(200)
    yield win
    win.close()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class TestMainWindowAccessibility:
    def test_main_window_has_accessible_name(self, app_window):
        assert app_window.accessibleName()

    def test_info_box_has_accessible_name(self, app_window):
        assert app_window._info_box.accessibleName()

    def test_info_box_has_accessible_description(self, app_window):
        assert app_window._info_box.accessibleDescription()


# ---------------------------------------------------------------------------
# PlotCanvas
# ---------------------------------------------------------------------------


class TestPlotCanvasAccessibility:
    def test_canvas_has_accessible_name(self, app_window):
        assert app_window._plot_canvas.accessibleName()

    def test_canvas_has_accessible_description(self, app_window):
        assert app_window._plot_canvas.accessibleDescription()


# ---------------------------------------------------------------------------
# PlotStylePanel
# ---------------------------------------------------------------------------


class TestPlotStylePanelAccessibility:
    def test_panel_has_accessible_name(self, app_window):
        assert app_window._plot_style_panel.accessibleName()

    def test_panel_has_accessible_description(self, app_window):
        assert app_window._plot_style_panel.accessibleDescription()


# ---------------------------------------------------------------------------
# System-parameter group and its spin boxes
# ---------------------------------------------------------------------------


class TestSystemParamAccessibility:
    def test_pwm_freq_spin_has_name(self, app_window):
        assert app_window._pwm_freq_spin.accessibleName()

    def test_battery_voltage_has_name(self, app_window):
        assert app_window._battery_voltage_spin.accessibleName()

    def test_speed_has_name(self, app_window):
        assert app_window._speed_spin.accessibleName()

    def test_amplitude_has_name(self, app_window):
        assert app_window._amplitude_spin.accessibleName()

    def test_dead_time_has_name(self, app_window):
        assert app_window._dead_time_spin.accessibleName()

    def test_diode_vf_has_name(self, app_window):
        assert app_window._diode_vf_spin.accessibleName()

    def test_filter_cutoff_has_name(self, app_window):
        assert app_window._filter_cutoff_spin.accessibleName()

    def test_injection_has_name(self, app_window):
        assert app_window._injection_spin.accessibleName()

    def test_current_phase_has_name(self, app_window):
        assert app_window._current_phase_spin.accessibleName()


# ---------------------------------------------------------------------------
# Modulation / Display groups
# ---------------------------------------------------------------------------


class TestModulationGroupAccessibility:
    def test_modulation_list_has_name(self, app_window):
        assert app_window._modulation_list.accessibleName()

    def test_modulation_list_has_description(self, app_window):
        assert app_window._modulation_list.accessibleDescription()

    def test_voltage_choice_has_name(self, app_window):
        assert app_window._voltage_choice.accessibleName()

    def test_voltage_choice_has_description(self, app_window):
        assert app_window._voltage_choice.accessibleDescription()


# ---------------------------------------------------------------------------
# Oscilloscope controls
# ---------------------------------------------------------------------------


class TestOscilloscopeControlsAccessibility:
    def test_pause_button_has_name(self, app_window):
        assert app_window._pause_button.accessibleName()

    def test_pause_button_has_description(self, app_window):
        assert app_window._pause_button.accessibleDescription()

    def test_step_button_has_name(self, app_window):
        assert app_window._step_button.accessibleName()

    def test_step_button_has_description(self, app_window):
        assert app_window._step_button.accessibleDescription()

    def test_reset_zoom_has_name(self, app_window):
        assert app_window._reset_zoom_button.accessibleName()

    def test_reset_zoom_has_description(self, app_window):
        assert app_window._reset_zoom_button.accessibleDescription()

    def test_export_csv_has_name(self, app_window):
        assert app_window._export_csv_button.accessibleName()

    def test_export_csv_has_description(self, app_window):
        assert app_window._export_csv_button.accessibleDescription()

    def test_export_png_has_name(self, app_window):
        assert app_window._export_png_button.accessibleName()

    def test_export_png_has_description(self, app_window):
        assert app_window._export_png_button.accessibleDescription()


# ---------------------------------------------------------------------------
# SweepDialog
# ---------------------------------------------------------------------------


class TestSweepDialogAccessibility:
    @pytest.fixture(autouse=True)
    def dialog(self, qapp, app_window):
        from svm_shaper.gui import SweepDialog

        dlg = SweepDialog(app_window._config, parent=None)
        yield dlg
        dlg.close()

    def test_dialog_has_accessible_name(self, dialog):
        assert dialog.accessibleName()

    def test_dialog_has_accessible_description(self, dialog):
        assert dialog.accessibleDescription()

    def test_variable_choice_has_name(self, dialog):
        assert dialog._variable_choice.accessibleName()

    def test_min_edit_has_name(self, dialog):
        assert dialog._min_edit.accessibleName()

    def test_max_edit_has_name(self, dialog):
        assert dialog._max_edit.accessibleName()

    def test_steps_edit_has_name(self, dialog):
        assert dialog._steps_edit.accessibleName()

    def test_run_button_has_name(self, dialog):
        assert dialog._run_button.accessibleName()

    def test_plot_widget_has_name(self, dialog):
        assert dialog._plot_widget.accessibleName()


# ---------------------------------------------------------------------------
# SvmHexagonDialog
# ---------------------------------------------------------------------------


class TestSvmHexagonDialogAccessibility:
    @pytest.fixture(autouse=True)
    def dialog(self, qapp, app_window):
        from svm_shaper.gui import SvmHexagonDialog

        dlg = SvmHexagonDialog(app_window._config, parent=None)
        dlg._timer.stop()
        yield dlg
        dlg.close()

    def test_dialog_has_accessible_name(self, dialog):
        assert dialog.accessibleName()

    def test_dialog_has_accessible_description(self, dialog):
        assert dialog.accessibleDescription()
