"""Core simulation helpers for PWM and SVM modulation.

This module defines the configuration data structures and the main entry point
for generating signals and performing analysis based on user-configurable system
parameters.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

from .analysis import compute_fft, compute_thd, compute_top_harmonics
from .modulations import (
    ModulationMode,
    PulseAlignment,
    generate_modulated_pwm,
    get_modulation_description,
)


@dataclass(frozen=True)  # pylint: disable=too-many-instance-attributes
class SimulatorConfig:
    """User-editable simulation configuration."""

    motor_pole_pairs: int = 5
    pwm_frequency_hz: float = 10000.0
    speed_rpm: float = 2000.0
    battery_voltage: float = 240.0
    # Output amplitude scaling (0–100%). This scales the modulation depth while
    # keeping the signal within the 0..Vbatt range.
    amplitude_percent: float = 100.0
    modulation: ModulationMode = ModulationMode.SVM
    show_phase_voltages: bool = False
    show_filtered: bool = False
    show_switching_edges: bool = False
    # Filter cutoff frequency (Hz). Set to 0 to use the default (3× electrical frequency).
    filter_cutoff_hz: float = 0.0
    # Injection percentage for third harmonic injection (0–100%).
    injection_percent: float = 100.0
    # PWM alignment mode used for carrier comparison.
    alignment: PulseAlignment = PulseAlignment.CENTER
    # Dead time inserted around switching events (microseconds).
    dead_time_us: float = 0.0
    # Body diode forward voltage used during dead-time freewheeling (volts).
    diode_forward_voltage_v: float = 0.6
    # Synthetic phase current angle offset relative to phase voltage (degrees).
    # Used only for dead-time diode conduction polarity estimation.
    current_phase_deg: float = 30.0
    # Author/project metadata for reports
    author_name: str = ""
    project_name: str = ""
    # Number of electrical cycles generated in the simulation (should be >= display_cycles)
    # Keep at 10 so FFT and reported metrics are based on 10 electrical cycles.
    num_cycles: int = 10
    # Number of electrical cycles shown at once in the oscilloscope view
    display_cycles: int = 3
    # Fixed oversampling used internally for waveform generation (high precision)
    oversample: int = 50


@dataclass  # pylint: disable=too-many-instance-attributes
class SimulationResult:
    """Container for computed waveform and analysis results."""

    time: np.ndarray
    phase_a: np.ndarray
    phase_b: np.ndarray
    phase_c: np.ndarray
    # phase_voltage_ab/bc/ca: voltage across a delta winding (terminal-to-terminal),
    # bipolar in the range [-Vbatt, +Vbatt].
    phase_voltage_ab: np.ndarray
    phase_voltage_bc: np.ndarray
    phase_voltage_ca: np.ndarray
    filtered_phase_a: np.ndarray
    filtered_phase_b: np.ndarray
    filtered_phase_c: np.ndarray

    fft_freqs: np.ndarray
    fft_magnitude: np.ndarray
    thd_line_percent: float
    thd_phase_percent: float
    top_harmonics: list[tuple[float, float]]
    pulses_per_electrical_cycle: int
    degrees_per_pwm_pulse: float
    actual_speed_rpm: float
    speed_deviation_rpm: float
    speed_deviation_percent: float
    filtered_mean: float
    filtered_rms: float
    filtered_min: float
    filtered_max: float
    raw_mean: float
    raw_rms: float
    raw_min: float
    raw_max: float
    description_text: str


def _apply_leg_dead_time_with_diode(
    commanded_pwm: np.ndarray,
    current_sign: np.ndarray,
    dead_samples: int,
    battery_voltage: float,
    diode_forward_voltage_v: float,
) -> np.ndarray:
    """Apply non-overlap dead time and map open-leg state through body diodes.

    Commanded PWM is expected in {-1, +1}, where +1 means upper switch ON and
    -1 means lower switch ON in ideal conditions.
    """

    n = commanded_pwm.size
    if n == 0:
        return np.array([], dtype=np.float64)

    if dead_samples <= 0:
        return np.where(commanded_pwm >= 0.0, battery_voltage, 0.0).astype(np.float64)

    # Switch state encoding: +1 upper ON, -1 lower ON, 0 both OFF (dead time).
    switch_state = np.empty(n, dtype=np.int8)
    desired_prev = 1 if commanded_pwm[0] >= 0.0 else -1
    active_state = desired_prev
    pending_state = 0
    pending_apply_index = -1
    switch_state[0] = active_state

    for i in range(1, n):
        desired = 1 if commanded_pwm[i] >= 0.0 else -1

        # Apply pending turn-on at the scheduled sample.
        if pending_apply_index >= 0 and i >= pending_apply_index:
            active_state = pending_state
            pending_apply_index = -1

        # On each commanded transition, turn current device OFF immediately,
        # and delay the opposite device turn-ON by dead_samples.
        if desired != desired_prev:
            active_state = 0
            pending_state = desired
            pending_apply_index = i + dead_samples

        desired_prev = desired
        switch_state[i] = active_state

    vf = max(0.0, diode_forward_voltage_v)
    phase_voltage = np.empty(n, dtype=np.float64)
    for i in range(n):
        state = switch_state[i]
        if state > 0:
            phase_voltage[i] = battery_voltage
        elif state < 0:
            phase_voltage[i] = 0.0
        else:
            # During dead time, output depends on current direction via diode conduction:
            # i > 0  -> lower diode conducts -> -Vf
            # i < 0  -> upper diode conducts -> Vdc + Vf
            phase_voltage[i] = -vf if current_sign[i] >= 0.0 else (battery_voltage + vf)

    return phase_voltage


def run_simulation(
    config: SimulatorConfig,
) -> SimulationResult:  # pylint: disable=too-many-locals,too-many-statements
    """Run the core simulation and return waveforms and analysis.

    The returned values are ready for plotting.
    """

    # Quantize the commanded speed to an integer number of PWM pulses per
    # electrical cycle. Any started pulse must be completed, so we round up.
    requested_electrical_freq = (config.speed_rpm / 60.0) * config.motor_pole_pairs
    if requested_electrical_freq <= 0.0:
        requested_electrical_freq = 1e-9

    requested_pulses_per_electrical = (
        config.pwm_frequency_hz / requested_electrical_freq
    )
    pulses_per_electrical = max(1, int(np.ceil(requested_pulses_per_electrical)))

    actual_electrical_freq = config.pwm_frequency_hz / pulses_per_electrical
    actual_speed_rpm = (actual_electrical_freq * 60.0) / config.motor_pole_pairs
    speed_deviation_rpm = actual_speed_rpm - config.speed_rpm
    speed_deviation_percent = (
        (speed_deviation_rpm / config.speed_rpm) * 100.0
        if abs(config.speed_rpm) > 1e-12
        else 0.0
    )

    # Generate commanded PWM waveforms for the selected modulation using the
    # quantized speed. Dead-time voltage effects are applied below at leg level.
    time, phase_a_cmd, phase_b_cmd, phase_c_cmd = generate_modulated_pwm(
        modulation=config.modulation,
        pole_pairs=config.motor_pole_pairs,
        speed_rpm=actual_speed_rpm,
        pwm_frequency_hz=config.pwm_frequency_hz,
        num_cycles=config.num_cycles,
        oversample=config.oversample,
        injection_percent=config.injection_percent,
        alignment=config.alignment,
        dead_time_s=0.0,
    )

    electrical_freq = actual_electrical_freq
    theta = 2.0 * np.pi * electrical_freq * time
    current_phase_deg = float(np.clip(config.current_phase_deg, -45.0, 45.0))
    current_phase_rad = np.deg2rad(current_phase_deg)
    phase_a_current_sign = np.sin(theta + current_phase_rad)
    phase_b_current_sign = np.sin(theta - 2.0 * np.pi / 3.0 + current_phase_rad)
    phase_c_current_sign = np.sin(theta + 2.0 * np.pi / 3.0 + current_phase_rad)

    dt = 1.0 / (config.pwm_frequency_hz * config.oversample)
    dead_samples = int(np.round(max(0.0, config.dead_time_us) * 1e-6 / dt))

    phase_a = _apply_leg_dead_time_with_diode(
        commanded_pwm=phase_a_cmd,
        current_sign=phase_a_current_sign,
        dead_samples=dead_samples,
        battery_voltage=config.battery_voltage,
        diode_forward_voltage_v=config.diode_forward_voltage_v,
    )
    phase_b = _apply_leg_dead_time_with_diode(
        commanded_pwm=phase_b_cmd,
        current_sign=phase_b_current_sign,
        dead_samples=dead_samples,
        battery_voltage=config.battery_voltage,
        diode_forward_voltage_v=config.diode_forward_voltage_v,
    )
    phase_c = _apply_leg_dead_time_with_diode(
        commanded_pwm=phase_c_cmd,
        current_sign=phase_c_current_sign,
        dead_samples=dead_samples,
        battery_voltage=config.battery_voltage,
        diode_forward_voltage_v=config.diode_forward_voltage_v,
    )

    # Apply user-configurable amplitude scaling around Vdc/2. This preserves the
    # requested PWM period while dead time reduces the effective +Vdc (or 0V)
    # on-time by inserting an open-leg interval.
    amplitude = max(0.0, min(config.amplitude_percent / 100.0, 1.0))
    center = 0.5 * config.battery_voltage
    phase_a = center + (phase_a - center) * amplitude
    phase_b = center + (phase_b - center) * amplitude
    phase_c = center + (phase_c - center) * amplitude

    # Count real generated PWM pulses on phase A and report pulses per electrical
    # cycle. Use transition pairs for robustness against start/end window
    # alignment, then ceil so any started pulse contributes.
    def _count_pwm_pulses(signal: np.ndarray) -> int:
        if signal.size < 2:
            return 0
        threshold = 0.5 * (float(np.min(signal)) + float(np.max(signal)))
        states = signal > threshold
        transitions = int(np.count_nonzero(states[1:] != states[:-1]))
        # Close the analysis window periodically to avoid losing one transition
        # at the start/end boundary when an integer number of electrical cycles
        # is simulated.
        if bool(states[-1]) != bool(states[0]):
            transitions += 1
        return int(np.ceil(transitions / 2.0))

    total_phase_a_pulses = _count_pwm_pulses(phase_a_cmd)
    total_phase_b_pulses = _count_pwm_pulses(phase_b_cmd)
    total_phase_c_pulses = _count_pwm_pulses(phase_c_cmd)
    total_avg_phase_pulses = (
        total_phase_a_pulses + total_phase_b_pulses + total_phase_c_pulses
    ) / 3.0
    pulses_per_electrical = int(
        np.ceil(total_avg_phase_pulses / max(1, config.num_cycles))
    )

    # Phase voltages: terminal-to-terminal voltage across a delta winding.
    phase_voltage_ab = phase_a - phase_b
    phase_voltage_bc = phase_b - phase_c
    phase_voltage_ca = phase_c - phase_a

    # Create filtered signals (and ensure the filtered waveform is centered at 0).
    # The sampling rate is derived from the generated time vector to match the
    # waveform resolution used elsewhere in the simulation.
    sampling_rate = (
        1.0 / (time[1] - time[0]) if time.size > 1 else config.pwm_frequency_hz
    )
    analysis_phase_a = _lowpass(
        phase_a,
        sampling_rate=sampling_rate,
        speed_rpm=actual_speed_rpm,
        pole_pairs=config.motor_pole_pairs,
        cutoff_hz=config.filter_cutoff_hz,
    )
    analysis_phase_b = _lowpass(
        phase_b,
        sampling_rate=sampling_rate,
        speed_rpm=actual_speed_rpm,
        pole_pairs=config.motor_pole_pairs,
        cutoff_hz=config.filter_cutoff_hz,
    )
    analysis_phase_c = _lowpass(
        phase_c,
        sampling_rate=sampling_rate,
        speed_rpm=actual_speed_rpm,
        pole_pairs=config.motor_pole_pairs,
        cutoff_hz=config.filter_cutoff_hz,
    )

    # Keep UI behavior: "show filtered" toggles the displayed line waveforms.
    if config.show_filtered:
        filtered_phase_a = analysis_phase_a
        filtered_phase_b = analysis_phase_b
        filtered_phase_c = analysis_phase_c
    else:
        filtered_phase_a = phase_a
        filtered_phase_b = phase_b
        filtered_phase_c = phase_c

    # Use the filtered analysis signals consistently for THD on both line and phase voltages.
    analysis_phase_voltage_ab = analysis_phase_a - analysis_phase_b

    # FFT on line voltage A (terminal A to DC−, 0..Vbatt), using the filtered
    # analysis waveform so THD reflects motor-relevant harmonics.
    fft_freqs, fft_magnitude = compute_fft(
        signal=analysis_phase_a,
        sampling_rate=config.pwm_frequency_hz * config.oversample,
        num_cycles=config.num_cycles,
        electrical_frequency_hz=electrical_freq,
    )
    thd_line_percent = compute_thd(
        fft_magnitude,
        fundamental_hz=electrical_freq,
        freqs=fft_freqs,
    )

    # THD on phase voltage AB (terminal-to-terminal, -Vbatt..+Vbatt), built
    # from the same filtered analysis basis for consistency with line THD.
    _, fft_magnitude_phase = compute_fft(
        signal=analysis_phase_voltage_ab,
        sampling_rate=config.pwm_frequency_hz * config.oversample,
        num_cycles=config.num_cycles,
        electrical_frequency_hz=electrical_freq,
    )
    thd_phase_percent = compute_thd(
        fft_magnitude_phase,
        fundamental_hz=electrical_freq,
        freqs=fft_freqs,
    )

    top_harmonics = compute_top_harmonics(fft_freqs, fft_magnitude, count=5)

    # Electrical angle per PWM pulse based on measured generated pulses.
    degrees_per_pulse = (
        360.0 / pulses_per_electrical if pulses_per_electrical != 0 else float("inf")
    )

    description_text = get_modulation_description(config.modulation)

    # Compute summary statistics for the filtered waveform (represents the motor input).
    filtered_mean = float(np.mean(filtered_phase_a))
    filtered_rms = float(np.sqrt(np.mean(filtered_phase_a**2)))
    filtered_min = float(np.min(filtered_phase_a))
    filtered_max = float(np.max(filtered_phase_a))

    # Also provide stats for the raw PWM output (phase A)
    raw_mean = float(np.mean(phase_a))
    raw_rms = float(np.sqrt(np.mean(phase_a**2)))
    raw_min = float(np.min(phase_a))
    raw_max = float(np.max(phase_a))

    return SimulationResult(
        time=time,
        phase_a=phase_a,
        phase_b=phase_b,
        phase_c=phase_c,
        phase_voltage_ab=phase_voltage_ab,
        phase_voltage_bc=phase_voltage_bc,
        phase_voltage_ca=phase_voltage_ca,
        filtered_phase_a=filtered_phase_a,
        filtered_phase_b=filtered_phase_b,
        filtered_phase_c=filtered_phase_c,
        fft_freqs=fft_freqs,
        fft_magnitude=fft_magnitude,
        thd_line_percent=thd_line_percent,
        thd_phase_percent=thd_phase_percent,
        top_harmonics=top_harmonics,
        pulses_per_electrical_cycle=pulses_per_electrical,
        degrees_per_pwm_pulse=degrees_per_pulse,
        actual_speed_rpm=actual_speed_rpm,
        speed_deviation_rpm=speed_deviation_rpm,
        speed_deviation_percent=speed_deviation_percent,
        filtered_mean=filtered_mean,
        filtered_rms=filtered_rms,
        filtered_min=filtered_min,
        filtered_max=filtered_max,
        raw_mean=raw_mean,
        raw_rms=raw_rms,
        raw_min=raw_min,
        raw_max=raw_max,
        description_text=description_text,
    )


def _lowpass(
    signal: np.ndarray,
    sampling_rate: float,
    speed_rpm: float,
    pole_pairs: int,
    cutoff_hz: float = 0.0,
) -> np.ndarray:
    """Apply a low-pass filter to the PWM waveform to reveal the fundamental shape.

    The filter is applied in a zero-phase way (filtfilt) to avoid phase
    distortion. The sampling rate must match the signal sampling used during
    PWM generation.
    """

    electrical_freq = (speed_rpm / 60.0) * pole_pairs

    # Default choice: give a moderate margin above the fundamental (3× fundamental).
    # User can override by setting `filter_cutoff_hz` in the UI.
    if cutoff_hz <= 0.0:
        cutoff = max(1.0, 3.0 * electrical_freq)
    else:
        cutoff = max(0.1, cutoff_hz)

    fs = sampling_rate
    b, a = butter(1, cutoff / (fs / 2.0), btype="low", analog=False)
    return filtfilt(b, a, signal)
