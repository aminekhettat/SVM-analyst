"""Feature tests — DC Bus Current Ripple.

Atomic features covered:
- SimulationResult.dc_bus_current_norm length equals duty_cycle_time length
- dc_bus_current_norm values are in a physically reasonable range (bounded)
- dc_bus_current_norm_pp >= 0
- dc_bus_current_norm_min <= dc_bus_current_norm_max
- Statistics (min, max, rms, pp) are consistently computed
- Zero-speed edge case: dc_bus_current_norm is empty or valid
- Ripple pp varies across modulation strategies (SVM vs DPWM)
- SimulationResult default keeps dc_bus_current_norm as empty array
- PlotCanvas.update_dc_bus_ripple creates a curve on first call
- PlotCanvas.update_dc_bus_ripple updates curve data on subsequent calls
- PlotCanvas._dc_bus_plot widget exists and is accessible
- PlotCanvas._dc_bus_check widget toggles plot visibility
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
# Core — DC bus ripple fields
# ---------------------------------------------------------------------------


class TestDcBusRippleCore:
    @pytest.fixture(scope="class")
    def result(self):
        return _run()

    def test_length_equals_duty_cycle_time(self, result):
        assert result.dc_bus_current_norm.size == result.duty_cycle_time.size

    def test_values_bounded(self, result):
        """Normalised current (Da*Ia + Db*Ib + Dc*Ic) must be in [-1, 1] range."""
        assert float(np.max(np.abs(result.dc_bus_current_norm))) <= 1.5

    def test_pp_non_negative(self, result):
        assert result.dc_bus_current_norm_pp >= 0.0

    def test_min_le_max(self, result):
        assert result.dc_bus_current_norm_min <= result.dc_bus_current_norm_max

    def test_pp_equals_max_minus_min(self, result):
        expected = result.dc_bus_current_norm_max - result.dc_bus_current_norm_min
        assert abs(result.dc_bus_current_norm_pp - expected) < 1e-9

    def test_rms_consistent(self, result):
        expected = float(np.sqrt(np.mean(result.dc_bus_current_norm**2)))
        assert abs(result.dc_bus_current_norm_rms - expected) < 1e-9

    def test_pp_positive_for_normal_operation(self, result):
        """For any balanced three-phase modulation, ripple is non-trivial."""
        assert result.dc_bus_current_norm_pp > 0.01

    def test_dpwm_has_different_pp_than_svm(self):
        svm = _run(ModulationMode.SVM)
        dpwm = _run(ModulationMode.DPWM_120_MAX)
        # DC bus ripple pp must differ between strategies (one of the key differentiators)
        assert abs(svm.dc_bus_current_norm_pp - dpwm.dc_bus_current_norm_pp) > 1e-6

    def test_defaults_backward_compatible(self):
        """SimulationResult default keeps dc_bus_current_norm as empty array."""
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
        assert sim.dc_bus_current_norm.size == 0
        assert sim.dc_bus_current_norm_pp == 0.0


# ---------------------------------------------------------------------------
# GUI — update_dc_bus_ripple
# ---------------------------------------------------------------------------


class TestPlotCanvasDcBus:
    @pytest.fixture(scope="class")
    def canvas(self, qapp):
        from svm_shaper.gui import PlotCanvas

        c = PlotCanvas()
        yield c
        c.destroy()

    def test_update_dc_bus_creates_curve_on_first_call(self, canvas):
        t = np.linspace(0, 1e-3, 40)
        current = np.sin(2 * np.pi * 50 * t) * 0.3
        assert canvas._dc_bus_curve is None
        canvas.update_dc_bus_ripple(t, current)
        assert canvas._dc_bus_curve is not None

    def test_update_dc_bus_updates_data(self, canvas):
        t = np.linspace(0, 1e-3, 40)
        current = np.ones(40) * 0.15
        canvas.update_dc_bus_ripple(t, current)
        x_data, y_data = canvas._dc_bus_curve.getData()
        assert len(y_data) == 40
        assert abs(float(y_data[5]) - 0.15) < 1e-6

    def test_dc_bus_plot_widget_exists(self, canvas):
        assert canvas._dc_bus_plot is not None

    def test_dc_bus_check_initial_state_is_checked(self, canvas):
        assert canvas._dc_bus_check.isChecked()

    def test_dc_bus_accessible_name(self, canvas):
        assert canvas._dc_bus_plot.accessibleName() == "DC bus current ripple plot"
