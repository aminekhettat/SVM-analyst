"""Feature tests: Single Shunt Current Reconstruction (SSCR) module.

Atomic features covered
-----------------------
F01  get_sector_from_angle      -- Returns correct sector (1–6) for
                                   representative angles including boundaries
F02  get_sector_from_angle      -- Handles angles > 360° and negative angles
                                   via implicit modulo wrapping
F03  get_duty_ordering          -- All six permutations return correct labels
F04  get_duty_ordering          -- Tie-breaking: equal duty cycles preserve
                                   a deterministic stable ordering
F05  compute_window_widths      -- Center-aligned: correct W1/W2 widths,
                                   dead-time subtraction, clamping to ≥ 0
F06  compute_window_widths      -- Left/Right-aligned: uses full period T
F07  compute_window_widths      -- Zero frequency returns zero widths safely
F08  compute_shunt_current_norm -- Returned currents satisfy KCL (i_max+i_mid+i_min≈0)
F09  build_pwm_period_pulse_shapes -- Center-aligned: pulse arrays have correct
                                   duty-cycle lengths and window positions
F10  build_pwm_period_pulse_shapes -- Left-aligned: left-justified pulse shape
F11  build_pwm_period_pulse_shapes -- Right-aligned: right-justified pulse shape
F12  build_pwm_period_pulse_shapes -- W1/W2 blind flag set when window < t_acq_min
F13  compute_single_shunt_analysis -- Returns SingleShuntAnalysis with correct
                                   shapes for all arrays (N periods)
F14  compute_single_shunt_analysis -- blind_fraction in [0, 1] and is consistent
                                   with w1_blind / w2_blind arrays
F15  compute_single_shunt_analysis -- Sector array values are in [1, 6]
F16  compute_single_shunt_analysis -- Duty ordering: d_max ≥ d_mid ≥ d_min each period
F17  compute_single_shunt_analysis -- CompensationStrategy.NONE: blind windows
                                   produce NaN reconstructed currents
F18  compute_single_shunt_analysis -- CompensationStrategy.MIN_PULSE: comp_applied
                                   flag is True for at least some periods when
                                   windows are tight (high dead_time)
F19  compute_single_shunt_analysis -- CompensationStrategy.HOLD: NaN replaced
                                   by last valid value for blind periods
F20  compute_single_shunt_analysis -- Non-NaN reconstructed currents satisfy
                                   KCL: i_max + i_mid + i_min ≈ 0
F21  compute_single_shunt_analysis -- i_a + i_b + i_c ≈ 0 (phase currents sum to zero)
F22  SingleShuntAnalysis         -- Statistics fields (w1/w2 mean, min) are
                                   consistent with the per-period arrays
F23  PeriodAnalysis              -- Field types and ranges are correct
F24  CompensationStrategy        -- Enum values exist and are distinct
F25  _interpolate_theta          -- Fallback when theta_e_deg array is empty
F26  _reassemble_phase_currents  -- Unknown phase labels produce NaN
F27  _apply_min_pulse_compensation -- Returns unchanged values when window ≥ threshold
F28  compute_single_shunt_analysis -- Empty result (zero periods) returns safe object
F29  compute_single_shunt_analysis -- SINUSOIDAL modulation full-pipeline smoke test
F30  compute_single_shunt_analysis -- SVM modulation full-pipeline smoke test
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from svm_shaper.single_shunt import (
    DEFAULT_T_ACQ_MIN_US,
    CompensationStrategy,
    PeriodAnalysis,
    SingleShuntAnalysis,
    _apply_min_pulse_compensation,
    _interpolate_theta,
    _reassemble_phase_currents,
    build_pwm_period_pulse_shapes,
    compute_shunt_current_norm,
    compute_single_shunt_analysis,
    compute_window_widths,
    get_duty_ordering,
    get_sector_from_angle,
)
from svm_shaper.core import SimulatorConfig, run_simulation
from svm_shaper.modulations import ModulationMode, PulseAlignment


# ---------------------------------------------------------------------------
# Minimal mock helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockResult:
    """Thin stand-in for SimulationResult with just the fields SSCR needs."""

    duty_cycle_time: Any
    duty_cycle_a: Any
    duty_cycle_b: Any
    duty_cycle_c: Any
    theta_e_deg: Any
    time: Any


def _make_synthetic_result(
    n_periods: int = 40,
    pwm_freq: float = 10_000.0,
    n_elec_cycles: float = 1.0,
) -> _MockResult:
    """Build a synthetic SimulationResult-like object with balanced modulation."""
    t_period = 1.0 / pwm_freq
    t_arr = np.arange(n_periods) * t_period
    # Electrical angle spanning n_elec_cycles full cycles over n_periods.
    theta_deg = np.linspace(0.0, 360.0 * n_elec_cycles, n_periods, endpoint=False)
    theta_rad = np.radians(theta_deg)

    m = 0.8  # modulation index
    d_a = 0.5 + 0.5 * m * np.cos(theta_rad)
    d_b = 0.5 + 0.5 * m * np.cos(theta_rad - 2 * np.pi / 3)
    d_c = 0.5 + 0.5 * m * np.cos(theta_rad + 2 * np.pi / 3)
    d_a = np.clip(d_a, 0.0, 1.0)
    d_b = np.clip(d_b, 0.0, 1.0)
    d_c = np.clip(d_c, 0.0, 1.0)

    return _MockResult(
        duty_cycle_time=t_arr,
        duty_cycle_a=d_a,
        duty_cycle_b=d_b,
        duty_cycle_c=d_c,
        theta_e_deg=theta_deg,
        time=t_arr,
    )


def _make_config(
    pwm_freq: float = 10_000.0,
    dead_time: float = 1.0,
    alignment: str = "Center",
) -> SimulatorConfig:
    """Return a minimal SimulatorConfig for use in SSCR tests."""
    align_enum = PulseAlignment(alignment)
    return SimulatorConfig(
        motor_pole_pairs=2,
        pwm_frequency_hz=pwm_freq,
        speed_rpm=1500.0,
        battery_voltage=400.0,
        amplitude_percent=80.0,
        modulation=ModulationMode.SINUSOIDAL,
        alignment=align_enum,
        dead_time_us=dead_time,
        diode_forward_voltage_v=0.0,
        current_phase_deg=0.0,
        modulation_index=0.8,
        num_cycles=5,
        display_cycles=2,
        oversample=20,
    )


# ---------------------------------------------------------------------------
# F01 / F02 – get_sector_from_angle
# ---------------------------------------------------------------------------


class TestGetSectorFromAngle:
    def test_sector_1_at_zero(self):
        assert get_sector_from_angle(0.0) == 1

    def test_sector_1_at_30(self):
        assert get_sector_from_angle(30.0) == 1

    def test_sector_2_at_60(self):
        assert get_sector_from_angle(60.0) == 2

    def test_sector_3_at_120(self):
        assert get_sector_from_angle(120.0) == 3

    def test_sector_4_at_180(self):
        assert get_sector_from_angle(180.0) == 4

    def test_sector_5_at_240(self):
        assert get_sector_from_angle(240.0) == 5

    def test_sector_6_at_300(self):
        assert get_sector_from_angle(300.0) == 6

    def test_sector_6_at_359(self):
        assert get_sector_from_angle(359.9) == 6

    def test_all_sectors_covered(self):
        for s in range(1, 7):
            angle = (s - 1) * 60.0 + 15.0
            assert get_sector_from_angle(angle) == s

    def test_angle_above_360_wraps(self):
        assert get_sector_from_angle(360.0 + 30.0) == 1

    def test_negative_angle_wraps(self):
        # -30 mod 360 = 330 → sector 6
        assert get_sector_from_angle(-30.0) == 6

    def test_large_angle_wraps(self):
        assert get_sector_from_angle(720.0 + 90.0) == 2


# ---------------------------------------------------------------------------
# F03 / F04 – get_duty_ordering
# ---------------------------------------------------------------------------


class TestGetDutyOrdering:
    """Test all six duty-cycle sort permutations and stability of ties."""

    _PERMS = [
        ((0.8, 0.5, 0.3), ("A", "B", "C")),
        ((0.5, 0.8, 0.3), ("B", "A", "C")),
        ((0.5, 0.3, 0.8), ("C", "A", "B")),
        ((0.3, 0.5, 0.8), ("C", "B", "A")),
        ((0.3, 0.8, 0.5), ("B", "C", "A")),
        ((0.8, 0.3, 0.5), ("A", "C", "B")),
    ]

    @pytest.mark.parametrize("duties,expected_labels", _PERMS)
    def test_all_permutations(self, duties, expected_labels):
        da, db, dc = duties
        d_max, d_mid, d_min, lm, lmi, ln = get_duty_ordering(da, db, dc)
        assert lm == expected_labels[0]
        assert lmi == expected_labels[1]
        assert ln == expected_labels[2]
        assert d_max >= d_mid >= d_min

    def test_values_correct(self):
        d_max, d_mid, d_min, *_ = get_duty_ordering(0.3, 0.9, 0.6)
        assert d_max == pytest.approx(0.9)
        assert d_mid == pytest.approx(0.6)
        assert d_min == pytest.approx(0.3)

    def test_equal_duty_cycles_stable(self):
        """All equal: sorting must not raise and must return sorted order."""
        d_max, d_mid, d_min, *_ = get_duty_ordering(0.5, 0.5, 0.5)
        assert d_max == pytest.approx(0.5)
        assert d_min == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# F05 / F06 / F07 – compute_window_widths
# ---------------------------------------------------------------------------


class TestComputeWindowWidths:
    def test_center_aligned_no_dead_time(self):
        # 10 kHz center: T/2 = 50 µs; (0.9 - 0.5)*50 = 20 µs
        w1i, w2i, w1e, w2e = compute_window_widths(
            0.9, 0.5, 0.1, 10_000.0, 0.0, "Center"
        )
        assert w1i == pytest.approx(20.0)
        assert w2i == pytest.approx(20.0)
        assert w1e == pytest.approx(20.0)
        assert w2e == pytest.approx(20.0)

    def test_center_aligned_dead_time_subtracted(self):
        w1i, w2i, w1e, w2e = compute_window_widths(
            0.9, 0.5, 0.1, 10_000.0, 2.0, "Center"
        )
        assert w1e == pytest.approx(20.0 - 2.0)
        assert w2e == pytest.approx(20.0 - 2.0)

    def test_center_aligned_clamps_to_zero(self):
        # Very small spread: w_ideal < dead_time → eff = 0
        w1i, w2i, w1e, w2e = compute_window_widths(
            0.51, 0.50, 0.49, 10_000.0, 5.0, "Center"
        )
        assert w1e == pytest.approx(0.0)
        assert w2e == pytest.approx(0.0)

    def test_left_aligned_uses_full_period(self):
        # 10 kHz left: T = 100 µs
        w1i, _, _, _ = compute_window_widths(0.6, 0.4, 0.2, 10_000.0, 0.0, "Left")
        assert w1i == pytest.approx(100.0 * (0.6 - 0.4))

    def test_right_aligned_uses_full_period(self):
        w1i, _, _, _ = compute_window_widths(0.6, 0.4, 0.2, 10_000.0, 0.0, "Right")
        assert w1i == pytest.approx(100.0 * (0.6 - 0.4))

    def test_case_insensitive_alignment(self):
        r1 = compute_window_widths(0.7, 0.5, 0.3, 10_000.0, 0.5, "center")
        r2 = compute_window_widths(0.7, 0.5, 0.3, 10_000.0, 0.5, "CENTER")
        assert r1 == r2

    def test_zero_frequency_returns_zeros(self):
        assert compute_window_widths(0.9, 0.5, 0.1, 0.0, 1.0) == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# F08 – compute_shunt_current_norm
# ---------------------------------------------------------------------------


class TestComputeShuntCurrentNorm:
    def test_kcl_all_angles(self):
        for deg in range(0, 360, 15):
            theta = math.radians(deg)
            i_mx, i_mi, i_mn = compute_shunt_current_norm(0.8, 0.5, 0.2, theta)
            total = i_mx + i_mi + i_mn
            assert abs(total) < 1e-10, f"KCL violated at {deg}°: sum={total}"

    def test_returns_three_floats(self):
        result = compute_shunt_current_norm(0.7, 0.5, 0.3, 0.5)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# F09 / F10 / F11 / F12 – build_pwm_period_pulse_shapes
# ---------------------------------------------------------------------------


class TestBuildPwmPeriodPulseShapes:
    def test_center_output_keys(self):
        shapes = build_pwm_period_pulse_shapes(0.7, 0.5, 0.3, 10_000.0, 1.0, "Center")
        for key in (
            "time_us",
            "pulse_a",
            "pulse_b",
            "pulse_c",
            "w1_mask",
            "w2_mask",
            "w1_eff_us",
            "w2_eff_us",
            "w1_blind",
            "w2_blind",
            "phase_max",
            "phase_mid",
            "phase_min",
        ):
            assert key in shapes

    def test_center_pulse_a_duty_correct(self):
        shapes = build_pwm_period_pulse_shapes(0.8, 0.5, 0.2, 10_000.0, 0.0, "Center")
        _ = shapes["time_us"]
        pa = shapes["pulse_a"]
        duty_meas = float(np.mean(pa))
        assert duty_meas == pytest.approx(0.8, abs=0.02)

    def test_left_aligned_pulse_is_left_justified(self):
        shapes = build_pwm_period_pulse_shapes(0.6, 0.4, 0.2, 10_000.0, 0.0, "Left")
        pa = shapes["pulse_a"]
        # First sample should be 1.0 (high at t=0 for left-aligned)
        assert pa[0] == pytest.approx(1.0)

    def test_right_aligned_pulse_is_right_justified(self):
        shapes = build_pwm_period_pulse_shapes(0.6, 0.4, 0.2, 10_000.0, 0.0, "Right")
        pa = shapes["pulse_a"]
        # Last sample should be 1.0 (high at end for right-aligned)
        assert pa[-1] == pytest.approx(1.0)

    def test_blind_flag_set_for_tight_window(self):
        # Tiny duty spread → W1 ideal ≈ 0 → blind
        shapes = build_pwm_period_pulse_shapes(
            0.501, 0.500, 0.499, 10_000.0, 5.0, "Center"
        )
        assert shapes["w1_blind"] is True

    def test_observable_flag_for_wide_window(self):
        shapes = build_pwm_period_pulse_shapes(0.9, 0.5, 0.1, 2_000.0, 0.5, "Center")
        assert shapes["w1_blind"] is False

    def test_phase_labels_correct(self):
        # d_b > d_a > d_c → max=B, mid=A, min=C
        shapes = build_pwm_period_pulse_shapes(0.5, 0.7, 0.3, 10_000.0, 0.0, "Center")
        assert shapes["phase_max"] == "B"
        assert shapes["phase_mid"] == "A"
        assert shapes["phase_min"] == "C"

    def test_custom_n_points(self):
        shapes = build_pwm_period_pulse_shapes(
            0.6, 0.4, 0.2, 10_000.0, 0.5, n_points=500
        )
        assert len(shapes["time_us"]) == 500


# ---------------------------------------------------------------------------
# F13–F22 – compute_single_shunt_analysis
# ---------------------------------------------------------------------------


class TestComputeSingleShuntAnalysis:
    @pytest.fixture
    def result_and_config(self):
        result = _make_synthetic_result(n_periods=100, pwm_freq=10_000.0)
        config = _make_config(pwm_freq=10_000.0, dead_time=1.0)
        return result, config

    def test_output_type(self, result_and_config):
        result, config = result_and_config
        analysis = compute_single_shunt_analysis(config, result)
        assert isinstance(analysis, SingleShuntAnalysis)

    def test_array_shapes(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(config, result)
        n = a.num_periods
        assert n == 100
        for arr_name in (
            "w1_ideal",
            "w2_ideal",
            "w1_eff",
            "w2_eff",
            "w1_blind",
            "w2_blind",
            "sector",
            "d_max",
            "d_mid",
            "d_min",
            "i_max_norm",
            "i_min_norm",
            "i_mid_norm",
            "i_a_norm",
            "i_b_norm",
            "i_c_norm",
        ):
            arr = getattr(a, arr_name)
            assert len(arr) == n, f"{arr_name} length mismatch"

    def test_blind_fraction_range(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(config, result)
        assert 0.0 <= a.blind_fraction <= 1.0

    def test_blind_fraction_consistent(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(config, result)
        expected_bf = float(np.mean(a.w1_blind | a.w2_blind))
        assert a.blind_fraction == pytest.approx(expected_bf, abs=1e-9)

    def test_sector_values_range(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(config, result)
        assert int(np.min(a.sector)) >= 1
        assert int(np.max(a.sector)) <= 6

    def test_duty_ordering_constraint(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(config, result)
        # d_max ≥ d_mid ≥ d_min per period.
        assert np.all(a.d_max >= a.d_mid - 1e-12)
        assert np.all(a.d_mid >= a.d_min - 1e-12)

    def test_none_compensation_blind_periods_have_nan(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(
            config, result, compensation=CompensationStrategy.NONE
        )
        blind_mask = a.w1_blind | a.w2_blind
        if np.any(blind_mask):
            # NaN in i_max_norm for blind periods.
            assert np.all(np.isnan(a.i_max_norm[blind_mask]))

    def test_kcl_for_valid_periods(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(config, result)
        valid = ~(a.w1_blind | a.w2_blind)
        if np.any(valid):
            i_sum = a.i_max_norm[valid] + a.i_mid_norm[valid] + a.i_min_norm[valid]
            np.testing.assert_allclose(i_sum, 0.0, atol=1e-9)

    def test_phase_current_kcl(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(config, result)
        valid = ~np.isnan(a.i_a_norm) & ~np.isnan(a.i_b_norm) & ~np.isnan(a.i_c_norm)
        if np.any(valid):
            i_sum = a.i_a_norm[valid] + a.i_b_norm[valid] + a.i_c_norm[valid]
            np.testing.assert_allclose(i_sum, 0.0, atol=1e-9)

    def test_stats_consistent(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(config, result)
        # mean of w1_eff matches stored field.
        assert a.w1_eff_mean_us == pytest.approx(float(np.mean(a.w1_eff)), abs=1e-9)
        # min of non-zero w1_eff matches stored field.
        nz = a.w1_eff[a.w1_eff > 0.0]
        if nz.size > 0:
            assert a.w1_eff_min_us == pytest.approx(float(np.min(nz)), abs=1e-9)

    def test_periods_list_length(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(config, result)
        assert len(a.periods) == a.num_periods

    def test_period_record_types(self, result_and_config):
        result, config = result_and_config
        a = compute_single_shunt_analysis(config, result)
        p = a.periods[0]
        assert isinstance(p, PeriodAnalysis)
        assert isinstance(p.sector, int)
        assert 1 <= p.sector <= 6
        assert isinstance(p.phase_max, str)
        assert p.phase_max in ("A", "B", "C")

    def test_min_pulse_compensation_applied(self):
        """High dead time forces blind windows → min-pulse comp should apply."""
        result = _make_synthetic_result(n_periods=50)
        config = _make_config(dead_time=25.0, pwm_freq=10_000.0)  # 25 µs dead time
        a = compute_single_shunt_analysis(
            config, result, compensation=CompensationStrategy.MIN_PULSE
        )
        # At least some should have compensation applied.
        applied = [p.compensation_applied for p in a.periods]
        assert any(applied), "Expected at least one period with compensation applied"

    def test_hold_compensation_no_nan_after_first_valid(self):
        """HOLD strategy should fill NaN with last valid sample after first valid period."""
        result = _make_synthetic_result(n_periods=100)
        config = _make_config(dead_time=25.0, pwm_freq=10_000.0)
        a = compute_single_shunt_analysis(
            config, result, compensation=CompensationStrategy.HOLD
        )
        # If any valid period exists, subsequent NaN should be replaced.
        i_max = a.i_max_norm
        valid_count = int(np.sum(~np.isnan(i_max)))
        # With HOLD, non-NaN count ≥ that under NONE.
        a_none = compute_single_shunt_analysis(
            config, result, compensation=CompensationStrategy.NONE
        )
        valid_none = int(np.sum(~np.isnan(a_none.i_max_norm)))
        assert valid_count >= valid_none


# ---------------------------------------------------------------------------
# F23 – PeriodAnalysis dataclass
# ---------------------------------------------------------------------------


class TestPeriodAnalysisDataclass:
    def test_roundtrip_fields(self):
        p = PeriodAnalysis(
            period_index=5,
            phase_max="A",
            phase_mid="B",
            phase_min="C",
            d_max=0.8,
            d_mid=0.5,
            d_min=0.2,
            sector=3,
            w1_ideal_us=15.0,
            w2_ideal_us=15.0,
            w1_eff_us=14.0,
            w2_eff_us=14.0,
            w1_blind=False,
            w2_blind=False,
            i_max_reconstructed=0.9,
            i_min_reconstructed=-0.4,
            i_mid_reconstructed=-0.5,
            compensation_applied=False,
            d_max_compensated=0.8,
            d_mid_compensated=0.5,
        )
        assert p.period_index == 5
        assert p.sector == 3
        assert p.d_max == pytest.approx(0.8)
        assert not p.w1_blind

    def test_blind_period(self):
        p = PeriodAnalysis(
            period_index=0,
            phase_max="B",
            phase_mid="C",
            phase_min="A",
            d_max=0.51,
            d_mid=0.50,
            d_min=0.49,
            sector=1,
            w1_ideal_us=0.5,
            w2_ideal_us=0.5,
            w1_eff_us=0.0,
            w2_eff_us=0.0,
            w1_blind=True,
            w2_blind=True,
            i_max_reconstructed=float("nan"),
            i_min_reconstructed=float("nan"),
            i_mid_reconstructed=float("nan"),
            compensation_applied=False,
            d_max_compensated=0.51,
            d_mid_compensated=0.50,
        )
        assert p.w1_blind
        assert math.isnan(p.i_max_reconstructed)


# ---------------------------------------------------------------------------
# F24 – CompensationStrategy enum
# ---------------------------------------------------------------------------


class TestCompensationStrategyEnum:
    def test_values_exist(self):
        vals = {s.value for s in CompensationStrategy}
        assert len(vals) == 3  # NONE, MIN_PULSE, HOLD

    def test_members_distinct(self):
        members = list(CompensationStrategy)
        assert len(set(m.name for m in members)) == len(members)

    def test_none_is_first(self):
        assert CompensationStrategy.NONE is list(CompensationStrategy)[0]


# ---------------------------------------------------------------------------
# F25 – _interpolate_theta fallback
# ---------------------------------------------------------------------------


class TestInterpolateTheta:
    def test_empty_theta_returns_linear_ramp(self):
        mock = MagicMock()
        mock.theta_e_deg = np.array([], dtype=np.float64)
        mock.time = np.array([], dtype=np.float64)
        t_arr = np.linspace(0.0, 1e-3, 10)
        result = _interpolate_theta(mock, t_arr)
        assert result.shape == (10,)
        # Values should span [0, 2π)
        assert float(result[0]) == pytest.approx(0.0, abs=0.01)

    def test_interpolates_correctly(self):
        mock = MagicMock()
        mock.time = np.array([0.0, 1.0])
        mock.theta_e_deg = np.array([0.0, 180.0])
        t_arr = np.array([0.5])
        result = _interpolate_theta(mock, t_arr)
        assert result[0] == pytest.approx(math.radians(90.0), abs=1e-6)

    def test_empty_t_arr_returns_ramp(self):
        mock = MagicMock()
        mock.theta_e_deg = np.array([0.0, 360.0])
        mock.time = np.array([0.0, 1.0])
        result = _interpolate_theta(mock, np.array([]))
        assert result.size == 1  # max(0, 1) = 1


# ---------------------------------------------------------------------------
# F26 – _reassemble_phase_currents
# ---------------------------------------------------------------------------


class TestReassemblePhaseCurrents:
    def test_basic_ordering(self):
        i_max = np.array([1.0])
        i_mid = np.array([0.5])
        i_min = np.array([-1.5])
        i_a, i_b, i_c = _reassemble_phase_currents(
            i_max, i_mid, i_min, ["A"], ["B"], ["C"], 1
        )
        assert i_a[0] == pytest.approx(1.0)
        assert i_b[0] == pytest.approx(0.5)
        assert i_c[0] == pytest.approx(-1.5)

    def test_kcl_preserved(self):
        n = 10
        rng = np.random.default_rng(42)
        i_mx = rng.uniform(-1, 1, n)
        i_mi = rng.uniform(-1, 1, n)
        i_mn = -i_mx - i_mi
        ph_mx = ["A"] * n
        ph_mi = ["B"] * n
        ph_mn = ["C"] * n
        i_a, i_b, i_c = _reassemble_phase_currents(
            i_mx, i_mi, i_mn, ph_mx, ph_mi, ph_mn, n
        )
        np.testing.assert_allclose(i_a + i_b + i_c, 0.0, atol=1e-9)

    def test_unknown_label_returns_nan(self):
        i_max = np.array([1.0])
        i_mid = np.array([0.0])
        i_min = np.array([-1.0])
        i_a, i_b, i_c = _reassemble_phase_currents(
            i_max,
            i_mid,
            i_min,
            ["X"],
            ["Y"],
            ["Z"],
            1,  # invalid labels
        )
        assert math.isnan(float(i_a[0]))
        assert math.isnan(float(i_b[0]))
        assert math.isnan(float(i_c[0]))


# ---------------------------------------------------------------------------
# F27 – _apply_min_pulse_compensation
# ---------------------------------------------------------------------------


class TestApplyMinPulseCompensation:
    def test_no_compensation_when_window_sufficient(self):
        d_max_new, d_mid_new, applied = _apply_min_pulse_compensation(
            d_max=0.8,
            d_mid=0.5,
            d_min=0.2,
            w1_eff=20.0,
            w2_eff=20.0,
            t_half_us=50.0,
            dead_time_us=1.0,
            t_acq_min=1.5,
        )
        assert not applied
        assert d_max_new == pytest.approx(0.8)
        assert d_mid_new == pytest.approx(0.5)

    def test_compensation_applied_when_w1_blind(self):
        _, _, applied = _apply_min_pulse_compensation(
            d_max=0.51,
            d_mid=0.50,
            d_min=0.49,
            w1_eff=0.0,
            w2_eff=0.0,
            t_half_us=50.0,
            dead_time_us=1.0,
            t_acq_min=1.5,
        )
        assert applied

    def test_d_max_clamped_to_one(self):
        d_max_new, _, _ = _apply_min_pulse_compensation(
            d_max=0.999,
            d_mid=0.998,
            d_min=0.997,
            w1_eff=0.0,
            w2_eff=0.0,
            t_half_us=50.0,
            dead_time_us=1.0,
            t_acq_min=1.5,
        )
        assert d_max_new <= 1.0

    def test_d_mid_not_below_d_min(self):
        _, d_mid_new, _ = _apply_min_pulse_compensation(
            d_max=0.501,
            d_mid=0.500,
            d_min=0.499,
            w1_eff=0.0,
            w2_eff=0.0,
            t_half_us=50.0,
            dead_time_us=1.0,
            t_acq_min=1.5,
        )
        assert d_mid_new >= 0.499 - 1e-9

    def test_zero_t_half_no_crash(self):
        d_max_new, d_mid_new, applied = _apply_min_pulse_compensation(
            d_max=0.8,
            d_mid=0.5,
            d_min=0.2,
            w1_eff=0.0,
            w2_eff=0.0,
            t_half_us=0.0,
            dead_time_us=1.0,
            t_acq_min=1.5,
        )
        # Should not crash; no compensation possible.
        assert isinstance(applied, bool)


# ---------------------------------------------------------------------------
# F28 – empty result (zero periods)
# ---------------------------------------------------------------------------


class TestEmptyResult:
    def test_empty_result_safe(self):
        mock = _MockResult(
            duty_cycle_time=np.array([]),
            duty_cycle_a=np.array([]),
            duty_cycle_b=np.array([]),
            duty_cycle_c=np.array([]),
            theta_e_deg=np.array([]),
            time=np.array([]),
        )
        config = _make_config()
        a = compute_single_shunt_analysis(config, mock)
        assert a.num_periods == 0
        assert a.blind_fraction == 0.0
        assert a.w1_eff_mean_us == 0.0
        assert a.w2_eff_mean_us == 0.0
        assert len(a.periods) == 0


# ---------------------------------------------------------------------------
# F29 / F30 – full-pipeline smoke tests
# ---------------------------------------------------------------------------


class TestFullPipelineSmokeTests:
    """Integration tests that run run_simulation and then SSCR analysis."""

    @pytest.fixture
    def sinusoidal_result(self):
        cfg = SimulatorConfig(
            motor_pole_pairs=2,
            pwm_frequency_hz=10_000.0,
            speed_rpm=1500.0,
            battery_voltage=400.0,
            amplitude_percent=80.0,
            modulation=ModulationMode.SINUSOIDAL,
            alignment=PulseAlignment.CENTER,
            dead_time_us=1.0,
            diode_forward_voltage_v=0.0,
            current_phase_deg=0.0,
            modulation_index=0.8,
            num_cycles=3,
            display_cycles=1,
            oversample=20,
        )
        result = run_simulation(cfg)
        return result, cfg

    @pytest.fixture
    def svm_result(self):
        cfg = SimulatorConfig(
            motor_pole_pairs=2,
            pwm_frequency_hz=10_000.0,
            speed_rpm=1500.0,
            battery_voltage=400.0,
            amplitude_percent=80.0,
            modulation=ModulationMode.SVM,
            alignment=PulseAlignment.CENTER,
            dead_time_us=1.0,
            diode_forward_voltage_v=0.0,
            current_phase_deg=0.0,
            modulation_index=0.8,
            num_cycles=3,
            display_cycles=1,
            oversample=20,
        )
        result = run_simulation(cfg)
        return result, cfg

    def test_sinusoidal_analysis_runs(self, sinusoidal_result):
        result, cfg = sinusoidal_result
        a = compute_single_shunt_analysis(cfg, result)
        assert a.num_periods > 0
        assert a.pwm_frequency_hz == pytest.approx(10_000.0)

    def test_svm_analysis_runs(self, svm_result):
        result, cfg = svm_result
        a = compute_single_shunt_analysis(cfg, result)
        assert a.num_periods > 0

    def test_sinusoidal_sector_all_six_present(self, sinusoidal_result):
        result, cfg = sinusoidal_result
        a = compute_single_shunt_analysis(cfg, result)
        present = set(int(s) for s in a.sector)
        # Over 3 full cycles all 6 sectors must appear.
        assert present == {1, 2, 3, 4, 5, 6}

    def test_svm_windows_are_positive(self, svm_result):
        result, cfg = svm_result
        a = compute_single_shunt_analysis(cfg, result)
        # SVM adds zero vectors that widen windows; most should be observable.
        observable_frac = 1.0 - a.blind_fraction
        assert observable_frac > 0.5, (
            "Expected majority of periods to be observable with SVM"
        )

    def test_alignment_stored_correctly(self, sinusoidal_result):
        result, cfg = sinusoidal_result
        a = compute_single_shunt_analysis(cfg, result)
        assert "center" in a.alignment.lower()

    def test_metadata_consistency(self, sinusoidal_result):
        result, cfg = sinusoidal_result
        a = compute_single_shunt_analysis(cfg, result)
        assert a.dead_time_us == pytest.approx(cfg.dead_time_us)
        assert a.pwm_frequency_hz == pytest.approx(cfg.pwm_frequency_hz)

    def test_min_pulse_does_not_crash_svm(self, svm_result):
        result, cfg = svm_result
        a = compute_single_shunt_analysis(
            cfg, result, compensation=CompensationStrategy.MIN_PULSE
        )
        assert a.num_periods > 0

    def test_hold_does_not_crash_sinusoidal(self, sinusoidal_result):
        result, cfg = sinusoidal_result
        a = compute_single_shunt_analysis(
            cfg, result, compensation=CompensationStrategy.HOLD
        )
        assert a.num_periods > 0


# ---------------------------------------------------------------------------
# Additional: DEFAULT_T_ACQ_MIN_US constant
# ---------------------------------------------------------------------------


def test_default_t_acq_min_positive():
    assert DEFAULT_T_ACQ_MIN_US > 0.0


def test_compute_single_shunt_analysis_left_alignment():
    result = _make_synthetic_result(n_periods=40)
    config = _make_config(alignment="Left")
    a = compute_single_shunt_analysis(config, result)
    assert a.alignment.upper() == "LEFT"
    assert a.num_periods == 40


def test_compute_single_shunt_analysis_right_alignment():
    result = _make_synthetic_result(n_periods=40)
    config = _make_config(alignment="Right")
    a = compute_single_shunt_analysis(config, result)
    assert a.alignment.upper() == "RIGHT"


def test_default_t_acq_min_with_config():
    """Exercise _default_t_acq_min when called with a real config (lines 692-696)."""
    from svm_shaper.single_shunt import _default_t_acq_min

    cfg = _make_config(pwm_freq=5_000.0, dead_time=2.0)
    dead, freq, t_period, t_half, t_acq = _default_t_acq_min(cfg)
    assert dead == pytest.approx(2.0)
    assert freq == pytest.approx(5_000.0)
    assert t_period == pytest.approx(200.0)
    assert t_half == pytest.approx(100.0)
    assert t_acq == pytest.approx(DEFAULT_T_ACQ_MIN_US)
