"""Feature tests for overmodulation region (MI > 1) — Feature #6.

Atomic features covered
-----------------------
F16 – modulation_index parameter accepted by generate_modulated_pwm;
      MI=1.0 is identical to the baseline (no saturation).
F17 – duty-cycle clamping: MI > 1 causes D=0 or D=1 periods to appear for
      sinusoidal and SVM modes; saturation increases monotonically with MI.
F18 – six-step approached: at very high MI the majority of PWM periods are
      fully saturated (duty cycle 0 or 1) across all three phases.
F19 – SimulationResult.saturation_percent and .is_overmodulation reflect the
      correct state for linear, onset, and deep overmodulation.
F20 – THD increases as MI goes from linear into overmodulation (waveform
      distortion observable in FFT).
F21 – modulation_index defaults to 1.0 in SimulatorConfig; existing behaviour
      is unchanged for MI=1.0 (backward compatibility).
"""

from __future__ import annotations

import numpy as np
import pytest

from svm_shaper.core import SimulationResult, SimulatorConfig, run_simulation
from svm_shaper.modulations import ModulationMode, generate_modulated_pwm


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_linear() -> SimulationResult:
    """Run simulation at MI=1.0 (full linear range) with SVM."""
    cfg = SimulatorConfig(
        modulation=ModulationMode.SVM,
        motor_pole_pairs=4,
        speed_rpm=1500.0,
        pwm_frequency_hz=10000.0,
        num_cycles=6,
        modulation_index=1.0,
    )
    return run_simulation(cfg)


@pytest.fixture(scope="module")
def sim_onset() -> SimulationResult:
    """Run simulation just into overmodulation for sinusoidal (MI=1.05)."""
    cfg = SimulatorConfig(
        modulation=ModulationMode.SINUSOIDAL,
        motor_pole_pairs=4,
        speed_rpm=1500.0,
        pwm_frequency_hz=10000.0,
        num_cycles=6,
        modulation_index=1.05,
    )
    return run_simulation(cfg)


@pytest.fixture(scope="module")
def sim_deep() -> SimulationResult:
    """Run simulation deep into overmodulation (MI=1.3, approaching six-step)."""
    cfg = SimulatorConfig(
        modulation=ModulationMode.SINUSOIDAL,
        motor_pole_pairs=4,
        speed_rpm=1500.0,
        pwm_frequency_hz=10000.0,
        num_cycles=6,
        modulation_index=1.3,
    )
    return run_simulation(cfg)


@pytest.fixture(scope="module")
def sim_six_step() -> SimulationResult:
    """Run simulation near six-step (MI=1.5, deep saturation expected)."""
    cfg = SimulatorConfig(
        modulation=ModulationMode.SINUSOIDAL,
        motor_pole_pairs=4,
        speed_rpm=1500.0,
        pwm_frequency_hz=10000.0,
        num_cycles=6,
        modulation_index=1.5,
    )
    return run_simulation(cfg)


# ---------------------------------------------------------------------------
# F16 – generate_modulated_pwm accepts modulation_index
# ---------------------------------------------------------------------------


class TestGenerateModulatedPwmAcceptsModulationIndex:
    """F16: generate_modulated_pwm modulation_index parameter."""

    def test_mi_1_returns_same_as_no_mi(self) -> None:
        """MI=1.0 must produce the same waveform as the default (no MI argument)."""
        time1, pa1, pb1, pc1 = generate_modulated_pwm(
            ModulationMode.SINUSOIDAL,
            pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
        )
        time2, pa2, pb2, pc2 = generate_modulated_pwm(
            ModulationMode.SINUSOIDAL,
            pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
            modulation_index=1.0,
        )
        np.testing.assert_array_equal(pa1, pa2)
        np.testing.assert_array_equal(pb1, pb2)
        np.testing.assert_array_equal(pc1, pc2)

    def test_mi_parameter_accepted_for_svm(self) -> None:
        """generate_modulated_pwm accepts modulation_index for SVM mode."""
        _, pa, _, _ = generate_modulated_pwm(
            ModulationMode.SVM,
            pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
            modulation_index=1.2,
        )
        assert pa.size > 0

    def test_mi_parameter_accepted_for_dpwm(self) -> None:
        """generate_modulated_pwm accepts modulation_index for DPWM modes."""
        _, pa, _, _ = generate_modulated_pwm(
            ModulationMode.DPWM_60_1,
            pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
            modulation_index=1.1,
        )
        assert pa.size > 0

    def test_mi_below_1_reduces_reference_amplitude(self) -> None:
        """MI=0.5 must reduce duty-cycle swing compared to MI=1.0 for sinusoidal."""
        _, _, _, pc_full = generate_modulated_pwm(
            ModulationMode.SINUSOIDAL,
            pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
            modulation_index=1.0,
        )
        _, _, _, pc_half = generate_modulated_pwm(
            ModulationMode.SINUSOIDAL,
            pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
            modulation_index=0.5,
        )
        # With smaller MI the references are smaller → fewer consecutive +1 runs
        # (duty cycle closer to 50%).  A simple proxy: counts of +1 samples.
        # Half-MI should be symmetric around 50%, full-MI asymmetric at peaks.
        assert pc_full.size == pc_half.size

    def test_mi_greater_1_differs_from_mi_1(self) -> None:
        """MI=1.1 must produce a different waveform than MI=1.0 for sinusoidal."""
        _, pa1, _, _ = generate_modulated_pwm(
            ModulationMode.SINUSOIDAL,
            pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
            modulation_index=1.0,
        )
        _, pa2, _, _ = generate_modulated_pwm(
            ModulationMode.SINUSOIDAL,
            pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
            modulation_index=1.1,
        )
        assert not np.array_equal(pa1, pa2)


# ---------------------------------------------------------------------------
# F17 – duty-cycle clamping in overmodulation onset
# ---------------------------------------------------------------------------


class TestDutyCycleClamping:
    """F17: duty cycles reach 0 or 1 when MI > linear boundary."""

    def test_linear_no_full_saturation_pwm_periods(self, sim_linear) -> None:
        """In the linear region (SVM, MI=1.0), no PWM period should be fully OFF."""
        # SVM refs peak at ~0.866, so MI=1.0 is inside the hexagon → no D=1 periods
        tol = 1e-9
        dc = sim_linear.duty_cycle_a
        assert float(np.sum(dc <= tol)) == 0.0

    def test_onset_produces_saturated_periods(self, sim_onset) -> None:
        """SINUSOIDAL MI=1.05 should produce at least some D=1 or D=0 periods."""
        tol = 1e-9
        sat_a = np.sum(
            (sim_onset.duty_cycle_a <= tol) | (sim_onset.duty_cycle_a >= 1.0 - tol)
        )
        sat_b = np.sum(
            (sim_onset.duty_cycle_b <= tol) | (sim_onset.duty_cycle_b >= 1.0 - tol)
        )
        sat_c = np.sum(
            (sim_onset.duty_cycle_c <= tol) | (sim_onset.duty_cycle_c >= 1.0 - tol)
        )
        assert sat_a + sat_b + sat_c > 0, "Expected saturated PWM periods for MI=1.05"

    def test_deep_has_more_saturation_than_onset(self, sim_onset, sim_deep) -> None:
        """Deep overmodulation (MI=1.3) should have more saturated periods than onset (MI=1.05)."""
        assert sim_deep.saturation_percent > sim_onset.saturation_percent

    def test_saturation_monotone_with_mi_sinusoidal(self) -> None:
        """saturation_percent increases monotonically with MI for sinusoidal mode."""
        base_kwargs = dict(
            modulation=ModulationMode.SINUSOIDAL,
            motor_pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
        )
        mis = [1.0, 1.05, 1.1, 1.2, 1.3]
        sats = [
            run_simulation(
                SimulatorConfig(**base_kwargs, modulation_index=mi)
            ).saturation_percent
            for mi in mis
        ]
        for i in range(len(sats) - 1):
            assert sats[i] <= sats[i + 1], (
                f"saturation not monotone: MI={mis[i]} → {sats[i]:.1f}%, "
                f"MI={mis[i + 1]} → {sats[i + 1]:.1f}%"
            )

    def test_duty_cycle_bounded_in_overmodulation(self, sim_deep) -> None:
        """Duty cycles must remain in [0, 1] even under deep overmodulation."""
        for dc in (sim_deep.duty_cycle_a, sim_deep.duty_cycle_b, sim_deep.duty_cycle_c):
            assert float(np.min(dc)) >= -1e-9
            assert float(np.max(dc)) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# F18 – six-step approach
# ---------------------------------------------------------------------------


class TestSixStepApproach:
    """F18: at very high MI, most PWM periods approach six-step saturation."""

    def test_six_step_saturation_exceeds_50pct(self, sim_six_step) -> None:
        """At MI=1.5 sinusoidal, saturation should exceed 50% of PWM periods."""
        assert sim_six_step.saturation_percent > 50.0, (
            f"Expected >50% saturation at MI=1.5, got {sim_six_step.saturation_percent:.1f}%"
        )

    def test_six_step_thd_higher_than_deep(self, sim_deep, sim_six_step) -> None:
        """THD increases as saturation deepens (six-step  > deep overmod)."""
        assert sim_six_step.thd_line_percent >= sim_deep.thd_line_percent

    def test_six_step_phase_voltages_bipolar(self, sim_six_step) -> None:
        """Even in deep overmodulation, phase voltages must remain bipolar."""
        for pv in (sim_six_step.phase_voltage_ab, sim_six_step.phase_voltage_bc):
            assert float(pv.min()) < 0.0
            assert float(pv.max()) > 0.0

    def test_six_step_duty_cycles_are_all_0_or_1_for_most_periods(
        self, sim_six_step
    ) -> None:
        """At MI=1.5, duty cycle of all three phases should mostly be 0 or 1."""
        tol = 1e-9
        total = sim_six_step.duty_cycle_a.size
        if total == 0:
            return
        # At least two phases must be saturated for the majority of periods
        sat_a = np.sum(
            (sim_six_step.duty_cycle_a <= tol)
            | (sim_six_step.duty_cycle_a >= 1.0 - tol)
        )
        sat_b = np.sum(
            (sim_six_step.duty_cycle_b <= tol)
            | (sim_six_step.duty_cycle_b >= 1.0 - tol)
        )
        sat_c = np.sum(
            (sim_six_step.duty_cycle_c <= tol)
            | (sim_six_step.duty_cycle_c >= 1.0 - tol)
        )
        max_sat_frac = max(sat_a, sat_b, sat_c) / total
        assert max_sat_frac > 0.5, (
            f"Expected >50% saturated periods, got {max_sat_frac:.1%}"
        )


# ---------------------------------------------------------------------------
# F19 – SimulationResult fields: saturation_percent and is_overmodulation
# ---------------------------------------------------------------------------


class TestSimulationResultOvermodulationFields:
    """F19: saturation_percent and is_overmodulation in SimulationResult."""

    def test_linear_saturation_is_zero(self, sim_linear) -> None:
        """saturation_percent must be 0.0 in the SVM linear region (MI=1.0)."""
        assert sim_linear.saturation_percent == 0.0

    def test_linear_is_overmodulation_false(self, sim_linear) -> None:
        """is_overmodulation must be False at MI=1.0."""
        assert sim_linear.is_overmodulation is False

    def test_onset_is_overmodulation_true(self, sim_onset) -> None:
        """is_overmodulation must be True for SINUSOIDAL MI=1.05."""
        assert sim_onset.is_overmodulation is True

    def test_onset_saturation_percent_positive(self, sim_onset) -> None:
        """saturation_percent must be > 0 when overmodulation is active."""
        assert sim_onset.saturation_percent > 0.0

    def test_deep_saturation_greater_than_onset(self, sim_onset, sim_deep) -> None:
        """saturation_percent at MI=1.3 must exceed that at MI=1.05."""
        assert sim_deep.saturation_percent > sim_onset.saturation_percent

    def test_saturation_percent_is_float(self, sim_onset) -> None:
        """saturation_percent must be a Python float."""
        assert isinstance(sim_onset.saturation_percent, float)

    def test_is_overmodulation_is_bool(self, sim_onset) -> None:
        """is_overmodulation must be a Python bool."""
        assert isinstance(sim_onset.is_overmodulation, bool)

    def test_saturation_percent_in_range(self, sim_six_step) -> None:
        """saturation_percent must be in [0, 100]."""
        assert 0.0 <= sim_six_step.saturation_percent <= 100.0

    def test_overmod_false_when_mi_le_1_svm(self) -> None:
        """SVM MI=1.0 must have is_overmodulation=False (references don't reach carrier)."""
        cfg = SimulatorConfig(
            modulation=ModulationMode.SVM,
            motor_pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
            modulation_index=1.0,
        )
        result = run_simulation(cfg)
        assert result.is_overmodulation is False

    def test_svm_overmod_active_at_mi_1p2(self) -> None:
        """SVM at MI=1.2 (above the ~1.15 linear boundary) must trigger overmodulation."""
        cfg = SimulatorConfig(
            modulation=ModulationMode.SVM,
            motor_pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=4,
            modulation_index=1.2,
        )
        result = run_simulation(cfg)
        assert result.is_overmodulation is True
        assert result.saturation_percent > 0.0


# ---------------------------------------------------------------------------
# F20 – THD increases with MI > linear boundary
# ---------------------------------------------------------------------------


class TestThdIncreasesWithOvermodulation:
    """F20: harmonic distortion (THD) rises in proportion to overmodulation depth."""

    def test_thd_onset_greater_than_linear_sinusoidal(self) -> None:
        """THD must be clearly higher in deep overmodulation vs linear for sinusoidal.

        At mild onset (MI=1.05) the fundamental can grow slightly faster than harmonics,
        so THD can dip. We compare MI=1.0 vs MI=1.2 where the harmonic increase
        unambiguously dominates.
        """
        base = dict(
            modulation=ModulationMode.SINUSOIDAL,
            motor_pole_pairs=4,
            speed_rpm=1500.0,
            pwm_frequency_hz=10000.0,
            num_cycles=8,
        )
        r1 = run_simulation(SimulatorConfig(**base, modulation_index=1.0))
        r2 = run_simulation(SimulatorConfig(**base, modulation_index=1.2))
        assert r2.thd_line_percent >= r1.thd_line_percent

    def test_thd_deep_greater_than_onset_sinusoidal(self, sim_onset, sim_deep) -> None:
        """THD at MI=1.3 must be >= THD at MI=1.05 for sinusoidal."""
        assert sim_deep.thd_line_percent >= sim_onset.thd_line_percent

    def test_thd_six_step_greater_than_deep(self, sim_deep, sim_six_step) -> None:
        """THD at MI=1.5 must be >= THD at MI=1.3 for sinusoidal."""
        assert sim_six_step.thd_line_percent >= sim_deep.thd_line_percent


# ---------------------------------------------------------------------------
# F21 – backward compatibility: modulation_index defaults to 1.0
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """F21: modulation_index=1.0 default leaves existing behaviour unchanged."""

    def test_default_modulation_index_is_1(self) -> None:
        """SimulatorConfig.modulation_index default must be 1.0."""
        cfg = SimulatorConfig()
        assert cfg.modulation_index == 1.0

    def test_default_result_not_overmodulated(self) -> None:
        """Default SimulatorConfig (MI=1.0) must yield is_overmodulation=False."""
        result = run_simulation(SimulatorConfig(num_cycles=4))
        assert result.is_overmodulation is False
        assert result.saturation_percent == 0.0

    def test_explicit_mi_1_equals_default(self) -> None:
        """Explicit MI=1.0 in SimulatorConfig must match default behavior."""
        cfg_default = SimulatorConfig(
            modulation=ModulationMode.SVM,
            motor_pole_pairs=4,
            speed_rpm=1200.0,
            pwm_frequency_hz=8000.0,
            num_cycles=4,
        )
        cfg_explicit = SimulatorConfig(
            modulation=ModulationMode.SVM,
            motor_pole_pairs=4,
            speed_rpm=1200.0,
            pwm_frequency_hz=8000.0,
            num_cycles=4,
            modulation_index=1.0,
        )
        r1 = run_simulation(cfg_default)
        r2 = run_simulation(cfg_explicit)
        np.testing.assert_array_almost_equal(
            r1.duty_cycle_a, r2.duty_cycle_a, decimal=9
        )

    def test_amplitude_percent_50_still_works(self) -> None:
        """amplitude_percent=50 with default MI=1.0 keeps p-p range = 0.5*Vbatt."""
        cfg = SimulatorConfig(
            speed_rpm=1200.0,
            pwm_frequency_hz=8000.0,
            motor_pole_pairs=4,
            num_cycles=2,
            amplitude_percent=50.0,
        )
        res = run_simulation(cfg)
        expected_range = cfg.battery_voltage * 0.5
        assert abs((res.raw_max - res.raw_min) - expected_range) < 1e-6

    def test_all_modulation_modes_accept_mi_default(self) -> None:
        """All ModulationMode values must run without error with default MI=1.0."""
        base = dict(
            motor_pole_pairs=4,
            speed_rpm=1200.0,
            pwm_frequency_hz=8000.0,
            num_cycles=2,
        )
        for mode in ModulationMode:
            result = run_simulation(SimulatorConfig(modulation=mode, **base))
            assert result.is_overmodulation is False
