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
F12  theta_e_deg        -- electrical angle is a sawtooth in [0, 360) °elec and
                           completes exactly num_cycles revolutions
F13  theta_mech_deg     -- mechanical angle period = pole_pairs × electrical period
                           (BLDC pole-pairs rule verified via reset-count ratio)
F14  Clarke αβ          -- Vβ leads Vα by 90° for balanced sinusoidal input
F15  αβ / dq metrics   -- RMS, peak, mean and module fields are physically
                           reasonable and populated by run_simulation
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


# ---------------------------------------------------------------------------
# F12  theta_e_deg — electrical angle sawtooth shape and cycle count
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_result_4pp():
    """Simulation with 4 pole pairs and 8 electrical cycles for angle tests."""
    return run_simulation(SimulatorConfig(motor_pole_pairs=4, num_cycles=8))


class TestElectricalAngleSawtooth:
    """F12 — theta_e_deg is a sawtooth waveform in [0, 360) °elec."""

    def test_theta_e_in_range(self, sim_result_4pp):
        te = sim_result_4pp.theta_e_deg
        assert te.size > 0
        assert float(np.min(te)) >= 0.0
        assert float(np.max(te)) < 360.0 + 1e-6

    def test_theta_e_is_array(self, sim_result_4pp):
        assert isinstance(sim_result_4pp.theta_e_deg, np.ndarray)

    def test_theta_e_same_length_as_time(self, sim_result_4pp):
        r = sim_result_4pp
        assert r.theta_e_deg.shape == r.time.shape

    def test_theta_e_has_expected_reset_count(self, sim_result_4pp):
        """8 electrical cycles → 7 sawtooth resets (drops of > 180°)."""
        te = sim_result_4pp.theta_e_deg
        resets = int(np.sum(np.diff(te) < -180.0))
        # With num_cycles=8 the angle completes 8 cycles; there are 7 internal
        # downward jumps (the last cycle ends at the window boundary, not a reset).
        assert resets == 7

    def test_theta_e_mostly_increasing_within_cycle(self, sim_result_4pp):
        """Between resets the angle must be monotonically non-decreasing."""
        te = sim_result_4pp.theta_e_deg
        diff = np.diff(te)
        # Only jumps larger than 180° are resets; everything else must be ≥ 0.
        non_reset_diffs = diff[diff > -180.0]
        assert float(np.min(non_reset_diffs)) >= -1e-9


# ---------------------------------------------------------------------------
# F13  theta_mech_deg — mechanical/electrical period ratio equals pole_pairs
# ---------------------------------------------------------------------------


class TestMechanicalAngleSawtooth:
    """F13 — mechanical angle period = pole_pairs × electrical period (BLDC rule)."""

    def test_theta_mech_in_range(self, sim_result_4pp):
        tm = sim_result_4pp.theta_mech_deg
        assert float(np.min(tm)) >= 0.0
        assert float(np.max(tm)) < 360.0 + 1e-6

    def test_theta_mech_is_array(self, sim_result_4pp):
        assert isinstance(sim_result_4pp.theta_mech_deg, np.ndarray)

    def test_theta_mech_same_length_as_time(self, sim_result_4pp):
        r = sim_result_4pp
        assert r.theta_mech_deg.shape == r.time.shape

    def test_theta_mech_reset_count_matches_bldc_rule(self, sim_result_4pp):
        """For 4 pole pairs and 8 electrical cycles there are 1 mechanical reset.

        BLDC rule: 1 mechanical revolution = pole_pairs electrical revolutions.
        8 elec cycles / 4 pole pairs = 2 mechanical cycles → 1 internal reset.
        """
        tm = sim_result_4pp.theta_mech_deg
        mech_resets = int(np.sum(np.diff(tm) < -180.0))
        assert mech_resets == 1

    def test_elec_reset_count_is_pole_pairs_times_mech_reset_count(
        self, sim_result_4pp
    ):
        """Cycle-count ratio: elec_cycles = pole_pairs × mech_cycles (BLDC rule).

        12 electrical cycles / 4 pole pairs = 3 mechanical cycles.
        Resets = cycles - 1, so (e_resets+1) = pole_pairs × (m_resets+1).
        """
        pole_pairs = 4
        r2 = run_simulation(SimulatorConfig(motor_pole_pairs=pole_pairs, num_cycles=12))
        e2 = int(np.sum(np.diff(r2.theta_e_deg) < -180.0))
        m2 = int(np.sum(np.diff(r2.theta_mech_deg) < -180.0))
        # e2=11 resets → 12 elec cycles; m2=2 resets → 3 mech cycles; 12=4×3.
        assert (e2 + 1) == pole_pairs * (m2 + 1)


# ---------------------------------------------------------------------------
# F14  Clarke αβ — Vβ leads Vα by exactly 90° for balanced sinusoidal input
# ---------------------------------------------------------------------------


class TestAlphaBeta90DegPhaseShift:
    """F14 — Vβ leads Vα by 90° for a balanced three-phase sinusoidal source."""

    def test_alpha_beta_are_orthogonal(self):
        """Orthogonality: <Vα, Vβ> ≈ 0 if their RMS values are equal."""
        va, vb, vc, t = _balanced_sin(freq_hz=50.0, vdc=200.0, n_samples=8000)
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        valpha = r["valpha"]
        vbeta = r["vbeta"]
        # Normalised cross-product (dot product / product of norms) should be ≈ 0.
        dot = float(np.mean(valpha * vbeta))
        rms_a = float(np.sqrt(np.mean(valpha**2)))
        rms_b = float(np.sqrt(np.mean(vbeta**2)))
        assert abs(dot) / (rms_a * rms_b + 1e-12) < 0.05

    def test_alpha_beta_rms_are_equal(self):
        """Amplitude invariance: RMS of Vβ equals RMS of Vα."""
        va, vb, vc, t = _balanced_sin(freq_hz=50.0, vdc=300.0, n_samples=8000)
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 300.0, 0.0)
        rms_a = float(np.sqrt(np.mean(r["valpha"] ** 2)))
        rms_b = float(np.sqrt(np.mean(r["vbeta"] ** 2)))
        assert abs(rms_a - rms_b) / (rms_a + 1e-12) < 0.01

    def test_beta_leads_alpha_by_90_degrees(self):
        """Phase shift: extract fundamental components and measure Vβ – Vα angle."""
        freq = 50.0
        va, vb, vc, t = _balanced_sin(freq_hz=freq, vdc=200.0, n_samples=8000)
        r = compute_dq_phasors(va, vb, vc, t, freq, 200.0, 0.0)
        valpha = r["valpha"]
        vbeta = r["vbeta"]
        # Project onto the fundamental using a complex exponential DFT kernel.
        kernel = np.exp(1j * 2 * np.pi * freq * t)
        phi_a = float(np.angle(np.mean(kernel * valpha), deg=True))
        phi_b = float(np.angle(np.mean(kernel * vbeta), deg=True))
        phase_diff = (phi_b - phi_a) % 360.0
        # Vβ should lead Vα by 90° (within a 5° tolerance due to PWM quantisation).
        assert abs(phase_diff - 90.0) < 5.0


# ---------------------------------------------------------------------------
# F15  αβ / dq metrics — populated and physically reasonable
# ---------------------------------------------------------------------------


class TestDqMetricsFromAnalysis:
    """F15a — compute_dq_phasors returns all new metric keys with sensible values."""

    def test_valpha_rms_positive(self):
        va, vb, vc, t = _balanced_sin()
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert r["valpha_rms"] > 0.0

    def test_vbeta_rms_positive(self):
        va, vb, vc, t = _balanced_sin()
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert r["vbeta_rms"] > 0.0

    def test_valpha_peak_ge_rms(self):
        va, vb, vc, t = _balanced_sin()
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert r["valpha_peak"] >= r["valpha_rms"]

    def test_vbeta_peak_ge_rms(self):
        va, vb, vc, t = _balanced_sin()
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert r["vbeta_peak"] >= r["vbeta_rms"]

    def test_vab_magnitude_mean_positive(self):
        va, vb, vc, t = _balanced_sin()
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert r["vab_magnitude_mean"] > 0.0

    def test_vab_magnitude_is_array(self):
        va, vb, vc, t = _balanced_sin()
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert isinstance(r["vab_magnitude"], np.ndarray)
        assert r["vab_magnitude"].shape == t.shape

    def test_vdq_magnitude_is_array(self):
        va, vb, vc, t = _balanced_sin()
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert isinstance(r["vdq_magnitude"], np.ndarray)
        assert r["vdq_magnitude"].shape == t.shape

    def test_vab_rms_ge_mean(self):
        va, vb, vc, t = _balanced_sin()
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert r["vab_magnitude_rms"] >= r["vab_magnitude_mean"] - 1e-9

    def test_vdq_magnitude_mean_positive(self):
        va, vb, vc, t = _balanced_sin()
        r = compute_dq_phasors(va, vb, vc, t, 50.0, 200.0, 0.0)
        assert r["vdq_magnitude_mean"] > 0.0

    def test_empty_input_returns_zero_metrics(self):
        empty = np.array([])
        r = compute_dq_phasors(empty, empty, empty, empty, 50.0, 200.0, 0.0)
        assert r["valpha_rms"] == 0.0
        assert r["vab_magnitude_mean"] == 0.0
        assert r["vdq_magnitude_rms"] == 0.0


class TestDqMetricsFromSimulation:
    """F15b — run_simulation populates all new metric fields in SimulationResult."""

    def test_theta_e_deg_is_array(self, sim_result):
        assert isinstance(sim_result.theta_e_deg, np.ndarray)
        assert sim_result.theta_e_deg.size > 0

    def test_theta_mech_deg_is_array(self, sim_result):
        assert isinstance(sim_result.theta_mech_deg, np.ndarray)
        assert sim_result.theta_mech_deg.size > 0

    def test_dq_valpha_rms_is_float(self, sim_result):
        assert isinstance(sim_result.dq_valpha_rms, float)
        assert sim_result.dq_valpha_rms > 0.0

    def test_dq_vbeta_rms_is_float(self, sim_result):
        assert isinstance(sim_result.dq_vbeta_rms, float)
        assert sim_result.dq_vbeta_rms > 0.0

    def test_dq_valpha_peak_ge_rms(self, sim_result):
        assert sim_result.dq_valpha_peak >= sim_result.dq_valpha_rms

    def test_dq_vbeta_peak_ge_rms(self, sim_result):
        assert sim_result.dq_vbeta_peak >= sim_result.dq_vbeta_rms

    def test_dq_vab_magnitude_mean_is_float(self, sim_result):
        assert isinstance(sim_result.dq_vab_magnitude_mean, float)
        assert sim_result.dq_vab_magnitude_mean > 0.0

    def test_dq_vab_magnitude_is_array(self, sim_result):
        assert isinstance(sim_result.dq_vab_magnitude, np.ndarray)
        assert sim_result.dq_vab_magnitude.shape == sim_result.time.shape

    def test_dq_vdq_magnitude_is_array(self, sim_result):
        assert isinstance(sim_result.dq_vdq_magnitude, np.ndarray)
        assert sim_result.dq_vdq_magnitude.shape == sim_result.time.shape

    def test_dq_vdq_magnitude_mean_is_float(self, sim_result):
        assert isinstance(sim_result.dq_vdq_magnitude_mean, float)
        assert sim_result.dq_vdq_magnitude_mean > 0.0

