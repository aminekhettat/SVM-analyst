"""Unit tests for core simulation logic."""

from svm_shaper.core import SimulatorConfig, run_simulation
from svm_shaper.modulations import ModulationMode


def test_switching_event_count_changes_for_dpwm_vs_svm() -> None:
    """DPWM should reduce the number of switching events compared to continuous SVM."""

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

    assert svm_res.pulses_per_electrical_cycle > 0
    assert dpwm_res.pulses_per_electrical_cycle > 0
    assert dpwm_res.pulses_per_electrical_cycle <= svm_res.pulses_per_electrical_cycle


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

    # DPWM modes intentionally create a DC offset; verify the mean is not
    # centered around half the DC bus.
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
        assert (
            abs(res2.filtered_mean - cfg.battery_voltage / 2.0) > 1.0
            or abs(res2.raw_mean - cfg.battery_voltage / 2.0) > 1.0
        )


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
