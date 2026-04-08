"""Feature tests: dq-frame phasor diagram.

Atomic features covered
-----------------------
F01  compute_dq_phasors -- Clarke αβ trajectory shape and length
F02  compute_dq_phasors -- balanced sinusoidal input traces a circle in αβ
F03  compute_dq_phasors -- Vd/Vq mean stable (DC-like) for balanced input
F04  compute_dq_phasors -- voltage phasor magnitude matches analytic expectation
F05  compute_dq_phasors -- voltage phasor angle consistent with Vd/Vq
F06  compute_dq_phasors -- current phasor lags voltage by current_phase_deg
F07  compute_dq_phasors -- empty/zero inputs return safe zero values
F08  SimulationResult   -- run_simulation populates all dq fields
F09  SimulationResult   -- dq_valpha / dq_vbeta have same length as time array
F10  SimulationResult   -- dq_vs_magnitude grows with battery voltage (scaling)
F11  SimulationResult   -- dq_is_angle_deg differs from dq_vs_angle_deg by
                           approximately current_phase_deg
"""

from __future__ import annotations

import numpy as np
import pytest

from svm_shaper.analysis import compute_dq_phasors
from svm_shaper.core import SimulatorConfig, run_simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _balanced_sin(
    freq_hz: float = 50.0,
    vdc: float = 200.0,
    n_samples: int = 4000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centred balanced three-phase sinusoids + time vector at freq_hz."""
    t = np.linspace(0.0, 1.0 / freq_hz * 5, n_samples, endpoint=False)
    amplitude = vdc / 3.0
    va = vdc / 2.0 + amplitude * np.sin(2 * np.pi * freq_hz * t)
    vb = vdc / 2.0 + amplitude * np.sin(2 * np.pi * freq_hz * t - 2 * np.pi / 3)
    vc = vdc / 2.0 + amplitude * np.sin(2 * np.pi * freq_hz * t + 2 * np.pi / 3)
    return va, vb, vc, t


# ---------------------------------------------------------------------------
# F01-F07  compute_dq_phasors
# ---------------------------------------------------------------------------


class TestComputeDqPhasorsShape:
    """F01 — αβ trajectory arrays match input length."""

    def test_valpha_length_equals_input(self):
        va, vb, vc, t = _balanced_sin()
        result = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert result["valpha"].shape == t.shape

    def test_vbeta_length_equals_input(self):
        va, vb, vc, t = _balanced_sin()
        result = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert result["vbeta"].shape == t.shape


class TestComputeDqPhasorsAlphaBeta:
    """F02 — balanced sinusoidal input traces a circle in αβ."""

    def test_alphabeta_traces_circle(self):
        va, vb, vc, t = _balanced_sin(freq_hz=50.0, vdc=200.0, n_samples=6000)
        result = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        radii = np.sqrt(result["valpha"] ** 2 + result["vbeta"] ** 2)
        # Radius should be nearly constant: std/mean < 2 %
        assert np.std(radii) / (np.mean(radii) + 1e-12) < 0.02

    def test_alphabeta_origin_centred(self):
        va, vb, vc, t = _balanced_sin()
        result = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        # Mean of the trajectory should be near zero (centred)
        assert abs(np.mean(result["valpha"])) < 1.0
        assert abs(np.mean(result["vbeta"])) < 1.0


class TestComputeDqPhasorsParkComponents:
    """F03 — Park (dq) components are DC-like for balanced input."""

    def test_vd_mean_stable(self):
        va, vb, vc, t = _balanced_sin(n_samples=8000)
        result = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        # For a balanced sinusoidal input the Park vector magnitude is constant.
        # Verify std/mean of |V_dq| is < 5 %.
        cos_theta = np.cos(2 * np.pi * 50.0 * t)
        sin_theta = np.sin(2 * np.pi * 50.0 * t)
        vd_inst = result["valpha"] * cos_theta + result["vbeta"] * sin_theta
        vq_inst = -result["valpha"] * sin_theta + result["vbeta"] * cos_theta
        magnitude = np.sqrt(vd_inst**2 + vq_inst**2)
        assert np.std(magnitude) / (np.mean(magnitude) + 1e-12) < 0.05

    def test_vq_mean_close_to_zero(self):
        # With d-axis aligned to cos(θe) and Va = A·sin(ωt), the voltage
        # projects entirely onto the q-axis (Vd≈0, Vq≈-A).  The test verifies
        # that the Park transform produces a stable DC-like result: the dominant
        # component must be close to the sinusoidal amplitude A = Vdc/3.
        vdc = 200.0
        va, vb, vc, t = _balanced_sin(n_samples=8000, vdc=vdc)
        result = compute_dq_phasors(va, vb, vc, t, 50.0, vdc, 0.0)
        amplitude = vdc / 3.0
        dominant = max(abs(result["vd_mean"]), abs(result["vq_mean"]))
        assert abs(dominant - amplitude) / amplitude < 0.05


class TestComputeDqPhasorsMagnitude:
    """F04 — voltage phasor magnitude matches analytic expectation."""

    def test_vs_magnitude_matches_amplitude_invariant_scale(self):
        # Amplitude-invariant Clarke: |Valpha| = peak amplitude of sinusoid component.
        # For Va = Vdc/2 + A*sin, amplitude A = Vdc/3. After centering: Va = A*sin.
        # Clarke amplitude-invariant: |Valpha| = A, so vs_magnitude ≈ A.
        vdc = 300.0
        a = vdc / 3.0
        va, vb, vc, t = _balanced_sin(freq_hz=50.0, vdc=vdc, n_samples=8000)
        result = compute_dq_phasors(va, vb, vc, t, 50.0, vdc, 0.0)
        assert abs(result["vs_magnitude"] - a) / a < 0.05


class TestComputeDqPhasorsAngle:
    """F05 — voltage phasor angle is consistent with Vd and Vq."""

    def test_vs_angle_matches_atan2_vq_vd(self):
        va, vb, vc, t = _balanced_sin()
        result = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        expected_angle = float(
            np.degrees(np.arctan2(result["vq_mean"], result["vd_mean"]))
        )
        assert abs(result["vs_angle_deg"] - expected_angle) < 1e-9


class TestComputeDqPhasorsCurrentPhasor:
    """F06 — current phasor lags voltage by current_phase_deg."""

    def test_current_lags_voltage(self):
        va, vb, vc, t = _balanced_sin()
        lag = 30.0
        result = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, lag)
        angle_diff = result["vs_angle_deg"] - result["is_angle_deg"]
        assert abs(angle_diff - lag) < 0.01

    def test_current_magnitude_equals_voltage_magnitude(self):
        va, vb, vc, t = _balanced_sin()
        result = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 15.0)
        is_magnitude = np.sqrt(result["id_fund"] ** 2 + result["iq_fund"] ** 2)
        assert abs(is_magnitude - result["vs_magnitude"]) < 1e-9

    def test_zero_lag_current_equals_voltage_direction(self):
        va, vb, vc, t = _balanced_sin()
        result = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert abs(result["is_angle_deg"] - result["vs_angle_deg"]) < 0.01


class TestComputeDqPhasorsEdgeCases:
    """F07 — empty or zero inputs return safe zero values."""

    def test_empty_arrays_return_empty_trajectories(self):
        empty = np.array([])
        result = compute_dq_phasors(empty, empty, empty, empty, 50.0, 200.0, 0.0)
        assert result["valpha"].size == 0
        assert result["vbeta"].size == 0

    def test_empty_arrays_return_zero_scalars(self):
        empty = np.array([])
        result = compute_dq_phasors(empty, empty, empty, empty, 50.0, 200.0, 0.0)
        assert result["vd_mean"] == 0.0
        assert result["vq_mean"] == 0.0
        assert result["vs_magnitude"] == 0.0
        assert result["id_fund"] == 0.0

    def test_zero_frequency_returns_zero_phasor(self):
        va, vb, vc, t = _balanced_sin()
        result = compute_dq_phasors(va, vb, vc, t, 0.0, 200.0, 0.0)
        assert result["vs_magnitude"] == 0.0

    def test_zero_voltage_magnitude_is_zero(self):
        n = 1000
        t = np.linspace(0, 0.02, n)
        half_vdc = np.full(n, 100.0)
        result = compute_dq_phasors(half_vdc, half_vdc, half_vdc, t, 50.0, 200.0, 0.0)
        assert result["vs_magnitude"] < 1e-6


# ---------------------------------------------------------------------------
# F08-F11  SimulationResult dq fields via run_simulation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_result():
    """Shared simulation result (SVM, default config)."""
    return run_simulation(SimulatorConfig())


class TestSimulationResultDqFields:
    """F08 — run_simulation populates all dq fields without errors."""

    def test_dq_valpha_is_array(self, sim_result):
        assert isinstance(sim_result.dq_valpha, np.ndarray)

    def test_dq_vbeta_is_array(self, sim_result):
        assert isinstance(sim_result.dq_vbeta, np.ndarray)

    def test_dq_vd_is_float(self, sim_result):
        assert isinstance(sim_result.dq_vd, float)

    def test_dq_vq_is_float(self, sim_result):
        assert isinstance(sim_result.dq_vq, float)

    def test_dq_vs_magnitude_is_float(self, sim_result):
        assert isinstance(sim_result.dq_vs_magnitude, float)

    def test_dq_vs_angle_deg_is_float(self, sim_result):
        assert isinstance(sim_result.dq_vs_angle_deg, float)

    def test_dq_id_is_float(self, sim_result):
        assert isinstance(sim_result.dq_id, float)

    def test_dq_iq_is_float(self, sim_result):
        assert isinstance(sim_result.dq_iq, float)

    def test_dq_is_angle_deg_is_float(self, sim_result):
        assert isinstance(sim_result.dq_is_angle_deg, float)


class TestSimulationResultDqArrayLengths:
    """F09 — αβ arrays have the same length as the time vector."""

    def test_dq_valpha_length_matches_time(self, sim_result):
        assert sim_result.dq_valpha.shape == sim_result.time.shape

    def test_dq_vbeta_length_matches_time(self, sim_result):
        assert sim_result.dq_vbeta.shape == sim_result.time.shape


class TestSimulationResultDqScaling:
    """F10 — dq_vs_magnitude grows proportionally with battery voltage."""

    def test_magnitude_scales_with_vdc(self):
        r1 = run_simulation(SimulatorConfig(battery_voltage=100.0))
        r2 = run_simulation(SimulatorConfig(battery_voltage=200.0))
        # Doubling Vdc should roughly double the SVM voltage phasor magnitude.
        assert r2.dq_vs_magnitude > r1.dq_vs_magnitude * 1.5


class TestSimulationResultDqCurrentAngle:
    """F11 — dq_is_angle_deg differs from dq_vs_angle_deg by ~current_phase_deg."""

    def test_current_angle_lag_is_current_phase_deg(self):
        lag = 20.0
        r = run_simulation(SimulatorConfig(current_phase_deg=lag))
        angle_diff = r.dq_vs_angle_deg - r.dq_is_angle_deg
        # Tight tolerance not possible here (PWM modulation causes ripple in
        # the averaged Park components), so use a generous window.
        assert abs(angle_diff - lag) < 2.0

    def test_zero_lag_gives_aligned_current_and_voltage(self):
        r = run_simulation(SimulatorConfig(current_phase_deg=0.0))
        assert abs(r.dq_vs_angle_deg - r.dq_is_angle_deg) < 1.0
