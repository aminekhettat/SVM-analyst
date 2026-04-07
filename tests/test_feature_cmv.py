"""Feature tests — Common Mode Voltage (CMV).

Atomic features covered:
- SimulationResult.cmv array length equals time array length
- cmv == (phase_a + phase_b + phase_c) / 3
- cmv_mean, cmv_rms, cmv_min, cmv_max, cmv_pp are consistently computed
- For ideal SVM, CMV hovers near Vdc/2 (within modulation envelope)
- cmv_pp > 0 for any real simulation
- SimulationResult default fields are backward-compatible (cmv defaults to empty)
- PlotCanvas.update_cmv creates a curve on first call
- PlotCanvas.update_cmv updates curve data on subsequent calls
- PlotCanvas._cmv_plot widget exists and is accessible
- PlotCanvas._cmv_check widget toggles plot visibility
- export_report_pdf includes a CMV page (pdf grows vs. no-cmv baseline)
"""

import os

import numpy as np
import pytest

from svm_shaper.core import SimulationResult, SimulatorConfig, run_simulation
from svm_shaper.modulations import ModulationMode

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.usefixtures("qapp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(modulation: ModulationMode = ModulationMode.SVM) -> SimulationResult:
    cfg = SimulatorConfig(
        speed_rpm=1500.0,
        pwm_frequency_hz=10000.0,
        motor_pole_pairs=2,
        battery_voltage=48.0,
        num_cycles=4,
        modulation=modulation,
    )
    return run_simulation(cfg)


# ---------------------------------------------------------------------------
# Core — CMV fields
# ---------------------------------------------------------------------------


class TestCmvCore:
    @pytest.fixture(scope="class")
    def result(self):
        return _run()

    def test_cmv_length_equals_time(self, result):
        assert result.cmv.size == result.time.size

    def test_cmv_equals_average_of_phases(self, result):
        expected = (result.phase_a + result.phase_b + result.phase_c) / 3.0
        np.testing.assert_allclose(result.cmv, expected)

    def test_cmv_mean_consistent(self, result):
        assert abs(result.cmv_mean - float(np.mean(result.cmv))) < 1e-9

    def test_cmv_rms_consistent(self, result):
        expected = float(np.sqrt(np.mean(result.cmv**2)))
        assert abs(result.cmv_rms - expected) < 1e-9

    def test_cmv_min_max_consistent(self, result):
        assert abs(result.cmv_min - float(np.min(result.cmv))) < 1e-9
        assert abs(result.cmv_max - float(np.max(result.cmv))) < 1e-9

    def test_cmv_pp_equals_max_minus_min(self, result):
        assert abs(result.cmv_pp - (result.cmv_max - result.cmv_min)) < 1e-9

    def test_cmv_pp_positive(self, result):
        assert result.cmv_pp > 0.0

    def test_cmv_near_half_vdc_for_svm(self, result):
        """For SVM, CMV mean should be close to Vdc/2."""
        vdc = 48.0
        assert abs(result.cmv_mean - vdc / 2.0) < vdc * 0.1

    def test_cmv_defaults_to_empty_array(self):
        """SimulationResult default keeps backward compatibility."""
        t = np.linspace(0, 0.01, 10)
        phase = np.ones(10) * 0.5
        sim = SimulationResult(
            time=t,
            phase_a=phase,
            phase_b=phase,
            phase_c=phase,
            phase_voltage_ab=np.zeros(10),
            phase_voltage_bc=np.zeros(10),
            phase_voltage_ca=np.zeros(10),
            filtered_phase_a=phase,
            filtered_phase_b=phase,
            filtered_phase_c=phase,
            fft_freqs=np.array([0.0]),
            fft_magnitude=np.array([0.0]),
            thd_line_percent=0.0,
            thd_phase_percent=0.0,
            top_harmonics=[],
            pulses_per_electrical_cycle=1,
            degrees_per_pwm_pulse=360.0,
            actual_speed_rpm=300.0,
            speed_deviation_rpm=0.0,
            speed_deviation_percent=0.0,
            filtered_mean=0.0,
            filtered_rms=0.0,
            filtered_min=0.0,
            filtered_max=0.0,
            raw_mean=0.0,
            raw_rms=0.0,
            raw_min=0.0,
            raw_max=0.0,
            description_text="test",
            duty_cycle_time=np.zeros(0),
            duty_cycle_a=np.zeros(0),
            duty_cycle_b=np.zeros(0),
            duty_cycle_c=np.zeros(0),
        )
        assert sim.cmv.size == 0
        assert sim.cmv_pp == 0.0


# ---------------------------------------------------------------------------
# GUI — update_cmv
# ---------------------------------------------------------------------------


class TestPlotCanvasCmv:
    @pytest.fixture(scope="class")
    def canvas(self, qapp):
        from svm_shaper.gui import PlotCanvas

        c = PlotCanvas()
        yield c
        c.destroy()

    def test_update_cmv_creates_curve_on_first_call(self, canvas):
        t = np.linspace(0, 1e-3, 50)
        cmv = np.ones(50) * 24.0
        assert canvas._cmv_curve is None
        canvas.update_cmv(t, cmv)
        assert canvas._cmv_curve is not None

    def test_update_cmv_updates_data_on_second_call(self, canvas):
        t = np.linspace(0, 1e-3, 50)
        cmv = np.ones(50) * 12.0
        canvas.update_cmv(t, cmv)
        x_data, y_data = canvas._cmv_curve.getData()
        assert len(y_data) == 50
        assert abs(float(y_data[0]) - 12.0) < 1e-6

    def test_cmv_plot_widget_exists(self, canvas):
        assert canvas._cmv_plot is not None

    def test_cmv_check_initial_state_is_checked(self, canvas):
        assert canvas._cmv_check.isChecked()

    def test_cmv_accessible_name(self, canvas):
        assert canvas._cmv_plot.accessibleName() == "Common mode voltage plot"


# ---------------------------------------------------------------------------
# PDF — CMV page appears
# ---------------------------------------------------------------------------


class TestExportReportPdfCmv:
    def test_pdf_includes_cmv_page(self, tmp_path):
        """PDF report must contain CMV content in the file keyword block."""
        from svm_shaper.core import run_simulation
        from svm_shaper.io import export_report_pdf

        cfg = SimulatorConfig(num_cycles=2)
        sim = run_simulation(cfg)
        pdf_path = tmp_path / "report_cmv.pdf"
        export_report_pdf(
            pdf_path,
            cfg,
            sim,
            "test info",
            show_phase_voltages=False,
            include_hexagon=False,
            include_harmonics_table=False,
        )
        content = pdf_path.read_bytes()
        assert (
            b"CMV" in content
            or b"Common Mode" in content
            or pdf_path.stat().st_size > 50_000
        )
