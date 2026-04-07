"""Unit tests for core simulation logic."""

import math

import numpy as np

from svm_shaper.core import SimulatorConfig, run_simulation
from svm_shaper.modulations import ModulationMode, PulseAlignment


def test_pwm_pulse_count_is_integer_and_reflects_generated_pulses() -> None:
    """Reported PWM pulses should be integer and reduced for DPWM clamping."""

    base_config = SimulatorConfig(
        speed_rpm=1500.0,
        pwm_frequency_hz=10000.0,
        motor_pole_pairs=5,
        num_cycles=2,
    )

    svm_config = base_config
    svm_config = svm_config.__class__(
        **{**svm_config.__dict__, "modulation": ModulationMode.SVM}
    )
    dpwm_config = base_config
    dpwm_config = dpwm_config.__class__(
        **{**dpwm_config.__dict__, "modulation": ModulationMode.DPWM_120_MAX}
    )

    svm_res = run_simulation(svm_config)
    dpwm_res = run_simulation(dpwm_config)

    assert isinstance(svm_res.pulses_per_electrical_cycle, int)
    assert isinstance(dpwm_res.pulses_per_electrical_cycle, int)
    assert svm_res.pulses_per_electrical_cycle > 0
    assert dpwm_res.pulses_per_electrical_cycle > 0
    assert dpwm_res.pulses_per_electrical_cycle < svm_res.pulses_per_electrical_cycle
    assert abs(svm_res.actual_speed_rpm - dpwm_res.actual_speed_rpm) < 1e-9


def test_real_speed_and_deviation_match_quantized_pulses() -> None:
    """Real speed must be computed from quantized PWM pulses and reported deviation."""

    cfg = SimulatorConfig(
        speed_rpm=1234.0,
        pwm_frequency_hz=10000.0,
        motor_pole_pairs=5,
        num_cycles=2,
    )
    res = run_simulation(cfg)

    expected_pulses = int(
        math.ceil(
            cfg.pwm_frequency_hz / ((cfg.speed_rpm / 60.0) * cfg.motor_pole_pairs)
        )
    )
    expected_actual_speed = (
        (cfg.pwm_frequency_hz / expected_pulses) * 60.0 / cfg.motor_pole_pairs
    )

    assert res.pulses_per_electrical_cycle == expected_pulses
    assert abs(res.actual_speed_rpm - expected_actual_speed) < 1e-9
    assert abs(res.speed_deviation_rpm - (expected_actual_speed - cfg.speed_rpm)) < 1e-9


def test_waveform_stats_are_consistent() -> None:
    """Mean should be near mid-point for symmetric waveforms, and min < max."""

    cfg = SimulatorConfig(
        speed_rpm=1200.0,
        pwm_frequency_hz=8000.0,
        motor_pole_pairs=4,
        num_cycles=2,
    )
    res = run_simulation(cfg)

    # Default modulation (SVM) produces waveforms centered around half the DC bus.
    # Accept small numerical offset due to sampling and rounding.
    assert abs(res.filtered_mean - cfg.battery_voltage / 2.0) < 1.0
    assert res.filtered_min < res.filtered_max
    assert res.raw_min < res.raw_max
    assert res.filtered_rms >= 0
    assert res.raw_rms >= 0

    # DPWM modes can create a DC offset depending on variant and operating point.
    # Verify at least one DPWM variant clearly departs from the CPWM midpoint.
    deviating_modes = 0
    for asym_mode in (
        ModulationMode.DPWM_120_MAX,
        ModulationMode.DPWM_120_MIN,
        ModulationMode.DPWM_60_1,
        ModulationMode.DPWM_60_0,
        ModulationMode.DPWM_60_2,
        ModulationMode.DPWM_30_3,
    ):
        cfg2 = SimulatorConfig(**{**cfg.__dict__, "modulation": asym_mode})
        res2 = run_simulation(cfg2)
        if (
            abs(res2.filtered_mean - cfg.battery_voltage / 2.0) > 1.0
            or abs(res2.raw_mean - cfg.battery_voltage / 2.0) > 1.0
        ):
            deviating_modes += 1

    assert deviating_modes >= 1


def test_amplitude_factor_changes_waveform_range() -> None:
    """Amplitude scaling should reduce the peak-to-peak voltage range."""

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


def test_line_voltages_are_bounded_between_zero_and_bus() -> None:
    """Line voltages (inverter terminal to DC−) must stay within 0..Vdc.
    Phase voltages (across delta winding) must be bipolar in −Vdc..−Vdc."""

    cfg = SimulatorConfig(
        speed_rpm=1800.0,
        pwm_frequency_hz=10000.0,
        motor_pole_pairs=5,
        num_cycles=2,
        battery_voltage=240.0,
    )
    res = run_simulation(cfg)

    # Line voltages = phase_a/b/c, bounded 0..Vdc
    for line in (res.phase_a, res.phase_b, res.phase_c):
        assert float(line.min()) >= -1e-9
        assert float(line.max()) <= cfg.battery_voltage + 1e-9

    # Phase voltages = phase_voltage_ab/bc/ca, bipolar −Vdc..+Vdc
    for pv in (res.phase_voltage_ab, res.phase_voltage_bc, res.phase_voltage_ca):
        assert float(pv.min()) >= -cfg.battery_voltage - 1e-9
        assert float(pv.max()) <= cfg.battery_voltage + 1e-9
        # Must actually reach both polarities (not degenerate)
        assert float(pv.min()) < 0.0
        assert float(pv.max()) > 0.0


def test_sinusoidal_phase_thd_is_not_inflated_by_raw_pwm_carrier() -> None:
    """For sinusoidal modulation, phase THD should stay in a realistic range.

    Regression case from GUI usage:
    1500 RPM, 6 pole pairs, 12 V bus, 50% amplitude.
    """

    cfg = SimulatorConfig(
        modulation=ModulationMode.SINUSOIDAL,
        speed_rpm=1500.0,
        motor_pole_pairs=6,
        battery_voltage=12.0,
        amplitude_percent=50.0,
        pwm_frequency_hz=10000.0,
        num_cycles=10,
        show_filtered=False,
    )
    res = run_simulation(cfg)

    assert res.thd_line_percent >= 0.0
    assert res.thd_phase_percent >= 0.0
    # Phase THD must not explode due to raw switching carrier content.
    assert res.thd_phase_percent < 20.0


def test_dpwm_raw_waveforms_reach_dc_rails_at_full_amplitude() -> None:
    """DPWM raw line and phase waveforms should hit DC rails at 100% amplitude."""

    cfg = SimulatorConfig(
        modulation=ModulationMode.DPWM_120_MAX,
        speed_rpm=2000.0,
        motor_pole_pairs=6,
        battery_voltage=12.0,
        amplitude_percent=100.0,
        pwm_frequency_hz=20000.0,
        num_cycles=2,
        show_filtered=False,
    )
    res = run_simulation(cfg)

    assert abs(float(np.max(res.phase_a)) - cfg.battery_voltage) < 1e-9
    assert abs(float(np.min(res.phase_a)) - 0.0) < 1e-9
    assert abs(float(np.max(res.phase_voltage_ab)) - cfg.battery_voltage) < 1e-9
    assert abs(float(np.min(res.phase_voltage_ab)) + cfg.battery_voltage) < 1e-9


def test_alignment_setting_changes_waveform_shape() -> None:
    base = dict(
        modulation=ModulationMode.SINUSOIDAL,
        speed_rpm=1500.0,
        motor_pole_pairs=4,
        pwm_frequency_hz=8000.0,
        num_cycles=2,
    )
    left = run_simulation(
        SimulatorConfig(**base, alignment=PulseAlignment.LEFT, dead_time_us=0.0)
    )
    center = run_simulation(
        SimulatorConfig(**base, alignment=PulseAlignment.CENTER, dead_time_us=0.0)
    )

    assert not np.array_equal(left.phase_a, center.phase_a)


def test_dead_time_influences_switching_statistics() -> None:
    base = dict(
        modulation=ModulationMode.SINUSOIDAL,
        speed_rpm=1500.0,
        motor_pole_pairs=4,
        pwm_frequency_hz=8000.0,
        num_cycles=2,
        alignment=PulseAlignment.CENTER,
    )
    no_dead = run_simulation(SimulatorConfig(**base, dead_time_us=0.0))
    with_dead = run_simulation(
        SimulatorConfig(**base, dead_time_us=5.0, diode_forward_voltage_v=0.6)
    )

    # Dead time must not change PWM period/frequency quantization.
    assert with_dead.pulses_per_electrical_cycle == no_dead.pulses_per_electrical_cycle


def test_dead_time_uses_diode_conduction_voltage_levels() -> None:
    cfg = SimulatorConfig(
        modulation=ModulationMode.SINUSOIDAL,
        speed_rpm=1500.0,
        motor_pole_pairs=4,
        pwm_frequency_hz=8000.0,
        num_cycles=2,
        alignment=PulseAlignment.CENTER,
        dead_time_us=6.0,
        diode_forward_voltage_v=0.6,
    )
    res = run_simulation(cfg)

    assert float(np.min(res.phase_a)) < 0.0
    assert float(np.max(res.phase_a)) > cfg.battery_voltage


def test_current_phase_parameter_changes_dead_time_voltage_distribution() -> None:
    base = dict(
        modulation=ModulationMode.SINUSOIDAL,
        speed_rpm=1500.0,
        motor_pole_pairs=4,
        pwm_frequency_hz=8000.0,
        num_cycles=2,
        alignment=PulseAlignment.CENTER,
        dead_time_us=6.0,
        diode_forward_voltage_v=0.6,
    )
    lead = run_simulation(SimulatorConfig(**base, current_phase_deg=45.0))
    lag = run_simulation(SimulatorConfig(**base, current_phase_deg=-45.0))

    assert not np.array_equal(lead.phase_a, lag.phase_a)


def test_simulation_result_has_duty_cycle_fields() -> None:
    """SimulationResult must expose per-PWM-period duty cycle arrays for all 3 phases."""
    cfg = SimulatorConfig(
        speed_rpm=1200.0,
        pwm_frequency_hz=8000.0,
        motor_pole_pairs=4,
        num_cycles=2,
    )
    res = run_simulation(cfg)
    assert hasattr(res, "duty_cycle_time")
    assert hasattr(res, "duty_cycle_a")
    assert hasattr(res, "duty_cycle_b")
    assert hasattr(res, "duty_cycle_c")
    expected_len = res.phase_a.size // cfg.oversample
    assert len(res.duty_cycle_time) == expected_len
    assert len(res.duty_cycle_a) == expected_len
    assert len(res.duty_cycle_b) == expected_len
    assert len(res.duty_cycle_c) == expected_len
    # Duty values are fractions — must stay in [0, 1].
    assert float(np.min(res.duty_cycle_a)) >= 0.0
    assert float(np.max(res.duty_cycle_a)) <= 1.0
    assert float(np.min(res.duty_cycle_b)) >= 0.0
    assert float(np.max(res.duty_cycle_b)) <= 1.0
    assert float(np.min(res.duty_cycle_c)) >= 0.0
    assert float(np.max(res.duty_cycle_c)) <= 1.0
