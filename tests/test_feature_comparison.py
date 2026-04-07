"""Feature tests — Side-by-side Comparison Mode.

Atomic features covered:
- PlotCanvas.set_reference_static creates FFT reference curve
- PlotCanvas.set_reference_static creates duty cycle reference curves (A/B/C)
- PlotCanvas.set_reference_static reference duty curves use N+1 edges (stepMode)
- PlotCanvas.update_reference_waveform creates wave reference curves (A/B/C)
- PlotCanvas.update_reference_waveform updates existing curves on second call
- PlotCanvas.clear_reference removes all reference curves (wave, duty, FFT)
- After clear_reference, _ref_wave_curves / _ref_duty_curves / _ref_fft_curve are None
- SvmShaperApp._ref_result is None initially
- SvmShaperApp._on_save_reference stores current sim result in _ref_result
- SvmShaperApp._on_save_reference calls set_reference_static (FFT + duty overlays)
- SvmShaperApp._on_save_reference enables the clear-ref button
- SvmShaperApp._on_clear_reference resets _ref_result to None
- SvmShaperApp._on_clear_reference disables the clear-ref button
- Info text contains comparison section only when reference is active
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


def _make_canvas(qapp):
    from svm_shaper.gui import PlotCanvas

    return PlotCanvas()


# ---------------------------------------------------------------------------
# PlotCanvas reference API
# ---------------------------------------------------------------------------


class TestPlotCanvasReference:
    @pytest.fixture(scope="class")
    def canvas(self, qapp):
        c = _make_canvas(qapp)
        yield c
        c.destroy()

    def test_ref_fft_curve_none_initially(self, canvas):
        assert canvas._ref_fft_curve is None

    def test_ref_wave_curves_none_initially(self, canvas):
        assert all(v is None for v in canvas._ref_wave_curves.values())

    def test_ref_duty_curves_none_initially(self, canvas):
        assert all(v is None for v in canvas._ref_duty_curves.values())

    def test_set_reference_static_creates_fft_curve(self, canvas):
        freqs = np.array([0.0, 50.0, 100.0])
        mag = np.array([0.0, 1.0, 0.2])
        t = np.linspace(0, 1e-3, 10)
        duty = {"A": np.ones(10) * 0.5, "B": np.ones(10) * 0.5, "C": np.ones(10) * 0.5}
        canvas.set_reference_static(freqs, mag, t, duty)
        assert canvas._ref_fft_curve is not None

    def test_set_reference_static_creates_duty_curves(self, canvas):
        for phase in ("A", "B", "C"):
            assert canvas._ref_duty_curves[phase] is not None

    def test_ref_duty_curves_use_n_plus_1_edges(self, canvas):
        """Duty reference must use stepMode with N+1 x-edges for N periods."""
        for phase in ("A", "B", "C"):
            x_data, y_data = canvas._ref_duty_curves[phase].getData()
            assert len(x_data) == len(y_data) + 1

    def test_update_reference_waveform_creates_curves(self, canvas):
        t = np.linspace(0, 1e-3, 30)
        phases = {"A": np.ones(30), "B": np.ones(30) * 0.5, "C": np.zeros(30)}
        canvas.update_reference_waveform(t, phases)
        for phase in ("A", "B", "C"):
            assert canvas._ref_wave_curves[phase] is not None

    def test_update_reference_waveform_updates_data(self, canvas):
        t = np.linspace(0, 1e-3, 30)
        phases = {"A": np.ones(30) * 99.0, "B": np.ones(30), "C": np.zeros(30)}
        canvas.update_reference_waveform(t, phases)
        x_data, y_data = canvas._ref_wave_curves["A"].getData()
        assert abs(float(y_data[0]) - 99.0) < 1e-6

    def test_clear_reference_removes_all_curves(self, canvas):
        canvas.clear_reference()
        assert all(v is None for v in canvas._ref_wave_curves.values())
        assert all(v is None for v in canvas._ref_duty_curves.values())
        assert canvas._ref_fft_curve is None

    def test_set_reference_static_after_clear_recreates_curves(self, canvas):
        freqs = np.array([0.0, 50.0])
        mag = np.array([0.0, 1.0])
        t = np.linspace(0, 1e-3, 5)
        duty = {"A": np.ones(5) * 0.5, "B": np.ones(5) * 0.5, "C": np.ones(5) * 0.5}
        canvas.set_reference_static(freqs, mag, t, duty)
        assert canvas._ref_fft_curve is not None


# ---------------------------------------------------------------------------
# SvmShaperApp — comparison mode integration
# ---------------------------------------------------------------------------


class TestSvmShaperAppComparison:
    @pytest.fixture(scope="class")
    def win(self, qapp):
        w = _make_window(qapp)
        yield w
        w.close()

    def test_ref_result_is_none_initially(self, win):
        assert win._ref_result is None

    def test_clear_ref_button_disabled_initially(self, win):
        assert not win._clear_ref_button.isEnabled()

    def test_save_reference_does_nothing_when_no_sim_result(self, win):
        win._sim_result = None
        win._on_save_reference()
        assert win._ref_result is None

    def test_save_reference_stores_sim_result(self, win, qapp):
        from svm_shaper.core import run_simulation, SimulatorConfig

        win._sim_result = run_simulation(SimulatorConfig(num_cycles=2))
        win._on_save_reference()
        assert win._ref_result is not None

    def test_clear_ref_button_enabled_after_save(self, win):
        assert win._clear_ref_button.isEnabled()

    def test_ref_display_signals_populated_after_save(self, win):
        assert "A" in win._ref_display_signals
        assert "B" in win._ref_display_signals
        assert "C" in win._ref_display_signals

    def test_clear_reference_resets_ref_result(self, win):
        win._on_clear_reference()
        assert win._ref_result is None

    def test_clear_ref_button_disabled_after_clear(self, win):
        assert not win._clear_ref_button.isEnabled()

    def test_ref_display_signals_empty_after_clear(self, win):
        assert win._ref_display_signals == {}

    def test_info_text_no_comparison_when_no_reference(self, win, qapp):
        from svm_shaper.core import run_simulation, SimulatorConfig

        win._sim_result = run_simulation(SimulatorConfig(num_cycles=2))
        win._ref_result = None
        win._update_info_text()
        text = win._info_box.toPlainText()
        assert "Comparison vs Reference" not in text

    def test_info_text_has_comparison_when_reference_active(self, win, qapp):
        from svm_shaper.core import run_simulation, SimulatorConfig

        win._sim_result = run_simulation(SimulatorConfig(num_cycles=2))
        win._ref_result = run_simulation(SimulatorConfig(num_cycles=2))
        win._update_info_text()
        text = win._info_box.toPlainText()
        assert "Comparison vs Reference" in text
        # Cleanup
        win._ref_result = None
        win._ref_display_signals = {}
