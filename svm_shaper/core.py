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
    amplitude_percent: float = 100.0
    modulation: ModulationMode = ModulationMode.SVM
    show_line_voltages: bool = True
    show_filtered: bool = False
    show_switching_edges: bool = False
    # Filter cutoff frequency (Hz). Set to 0 to use the default (3× electrical frequency).
    filter_cutoff_hz: float = 0.0
    # Injection percentage for third harmonic injection (0–100%).
    injection_percent: float = 100.0
    # Author/project metadata for reports
    author_name: str = ""
    project_name: str = ""
    # Number of electrical cycles generated in the simulation (should be >= display_cycles)
    num_cycles: int = 6
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
    line_ab: np.ndarray
    line_bc: np.ndarray
    line_ca: np.ndarray
    filtered_phase_a: np.ndarray
    filtered_phase_b: np.ndarray
    filtered_phase_c: np.ndarray

    fft_freqs: np.ndarray
    fft_magnitude: np.ndarray
    thd_percent: float
    top_harmonics: list[tuple[float, float]]
    pulses_per_electrical_cycle: float
    degrees_per_pwm_pulse: float
    filtered_mean: float
    filtered_rms: float
    filtered_min: float
    filtered_max: float
    raw_mean: float
    raw_rms: float
    raw_min: float
    raw_max: float
    description_text: str


def run_simulation(
    config: SimulatorConfig,
) -> SimulationResult:  # pylint: disable=too-many-locals,too-many-statements
    """Run the core simulation and return waveforms and analysis.

    The returned values are ready for plotting.
    """

    # Generate PWM waveforms for the selected modulation
    time, phase_a, phase_b, phase_c = generate_modulated_pwm(
        modulation=config.modulation,
        pole_pairs=config.motor_pole_pairs,
        speed_rpm=config.speed_rpm,
        pwm_frequency_hz=config.pwm_frequency_hz,
        num_cycles=config.num_cycles,
        oversample=config.oversample,
        injection_percent=config.injection_percent,
    )

    # Convert normalized PWM outputs (-1..+1) to 0..1 (ground..Vbatt) and apply
    # the user-configurable amplitude scaling.
    #
    # In a real inverter, the half-bridge output can never go below ground (0V)
    # or above the DC bus voltage. CPWM modes are centered around Vbatt/2, while
    # DPWM modes can introduce a DC offset due to clamping.
    amplitude = max(0.0, min(config.amplitude_percent / 100.0, 1.0))
    phase_a = 0.5 + (phase_a * amplitude) / 2.0
    phase_b = 0.5 + (phase_b * amplitude) / 2.0
    phase_c = 0.5 + (phase_c * amplitude) / 2.0

    phase_a = np.clip(phase_a, 0.0, 1.0)
    phase_b = np.clip(phase_b, 0.0, 1.0)
    phase_c = np.clip(phase_c, 0.0, 1.0)

    # Compute line voltages (phase-to-phase)
    line_ab = phase_a - phase_b
    line_bc = phase_b - phase_c
    line_ca = phase_c - phase_a

    # Create filtered signals (and ensure the filtered waveform is centered at 0).
    # The sampling rate is derived from the generated time vector to match the
    # waveform resolution used elsewhere in the simulation.
    sampling_rate = (
        1.0 / (time[1] - time[0]) if time.size > 1 else config.pwm_frequency_hz
    )
    if config.show_filtered:
        filtered_phase_a = _lowpass(
            phase_a,
            sampling_rate=sampling_rate,
            speed_rpm=config.speed_rpm,
            pole_pairs=config.motor_pole_pairs,
            cutoff_hz=config.filter_cutoff_hz,
        )
        filtered_phase_b = _lowpass(
            phase_b,
            sampling_rate=sampling_rate,
            speed_rpm=config.speed_rpm,
            pole_pairs=config.motor_pole_pairs,
            cutoff_hz=config.filter_cutoff_hz,
        )
        filtered_phase_c = _lowpass(
            phase_c,
            sampling_rate=sampling_rate,
            speed_rpm=config.speed_rpm,
            pole_pairs=config.motor_pole_pairs,
            cutoff_hz=config.filter_cutoff_hz,
        )
    else:
        filtered_phase_a = phase_a
        filtered_phase_b = phase_b
        filtered_phase_c = phase_c

    # Scale from normalized waveform (0..1) to actual voltage (0..Vbatt).
    phase_a = phase_a * config.battery_voltage
    phase_b = phase_b * config.battery_voltage
    phase_c = phase_c * config.battery_voltage

    line_ab = line_ab * config.battery_voltage
    line_bc = line_bc * config.battery_voltage
    line_ca = line_ca * config.battery_voltage

    filtered_phase_a = filtered_phase_a * config.battery_voltage
    filtered_phase_b = filtered_phase_b * config.battery_voltage
    filtered_phase_c = filtered_phase_c * config.battery_voltage

    # FFT and THD are computed on the *filtered* waveform to match the effective
    # motor output waveform rather than the raw PWM edges.
    fft_freqs, fft_magnitude = compute_fft(
        signal=filtered_phase_a,
        sampling_rate=config.pwm_frequency_hz * config.oversample,
        num_cycles=config.num_cycles,
        electrical_frequency_hz=(config.speed_rpm / 60.0) * config.motor_pole_pairs,
    )
    electrical_freq = (config.speed_rpm / 60.0) * config.motor_pole_pairs
    thd_percent = compute_thd(
        fft_magnitude,
        fundamental_hz=electrical_freq,
        freqs=fft_freqs,
    )
    top_harmonics = compute_top_harmonics(fft_freqs, fft_magnitude, count=5)

    # Calculate switching events per electrical cycle by counting transitions
    # in each phase waveform (PWM edges). DPWM modes reduce switching by clamping
    # phases for part of the cycle, so the count differs from the raw carrier
    # frequency.
    def _count_transitions(x: np.ndarray) -> int:
        return int(np.count_nonzero(np.diff(x) != 0))

    transitions_a = _count_transitions(phase_a)
    transitions_b = _count_transitions(phase_b)
    transitions_c = _count_transitions(phase_c)
    avg_transitions_per_cycle = (
        (transitions_a + transitions_b + transitions_c) / 3.0 / config.num_cycles
    )

    pulses_per_electrical = float(avg_transitions_per_cycle)
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
        line_ab=line_ab,
        line_bc=line_bc,
        line_ca=line_ca,
        filtered_phase_a=filtered_phase_a,
        filtered_phase_b=filtered_phase_b,
        filtered_phase_c=filtered_phase_c,
        fft_freqs=fft_freqs,
        fft_magnitude=fft_magnitude,
        thd_percent=thd_percent,
        top_harmonics=top_harmonics,
        pulses_per_electrical_cycle=pulses_per_electrical,
        degrees_per_pwm_pulse=degrees_per_pulse,
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
