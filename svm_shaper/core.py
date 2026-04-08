"""Core simulation helpers for PWM and SVM modulation.

This module defines the configuration data structures and the main entry point
for generating signals and performing analysis based on user-configurable system
parameters.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt

try:
    from numba import njit as _njit

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

from .analysis import (
    compute_dq_phasors,
    compute_duty_cycle_envelope,
    compute_fft,
    compute_thd,
    compute_top_harmonics,
)
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
    # Modulation index (MI). 1.0 = full linear range for the reference signals used
    # by each modulation type. MI > 1 scales the references above the carrier bounds,
    # inducing duty-cycle clamping and waveform distortion (overmodulation region).
    # For sinusoidal and THIPWM modes the linear boundary is MI=1.0; for SVM/DPWM
    # the references already peak at ~0.866 so overmodulation starts near MI=1.15.
    modulation_index: float = 1.0


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
    # Per-PWM-period duty cycle envelope: one value per switching period.
    # duty_cycle_time contains the time at the midpoint of each period.
    duty_cycle_time: np.ndarray
    duty_cycle_a: np.ndarray
    duty_cycle_b: np.ndarray
    duty_cycle_c: np.ndarray

    # --- Per-leg (line) duty cycle statistics [0, 1] ---
    duty_cycle_a_min: float = 0.0
    duty_cycle_a_max: float = 0.0
    duty_cycle_a_mean: float = 0.0
    duty_cycle_a_rms: float = 0.0
    duty_cycle_b_min: float = 0.0
    duty_cycle_b_max: float = 0.0
    duty_cycle_b_mean: float = 0.0
    duty_cycle_b_rms: float = 0.0
    duty_cycle_c_min: float = 0.0
    duty_cycle_c_max: float = 0.0
    duty_cycle_c_mean: float = 0.0
    duty_cycle_c_rms: float = 0.0

    # --- Phase-to-phase duty cycle: D_A-D_B, D_B-D_C, D_C-D_A (may be negative) ---
    duty_cycle_ab: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    duty_cycle_bc: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    duty_cycle_ca: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    duty_cycle_ab_min: float = 0.0
    duty_cycle_ab_max: float = 0.0
    duty_cycle_ab_mean: float = 0.0
    duty_cycle_ab_rms: float = 0.0
    duty_cycle_bc_min: float = 0.0
    duty_cycle_bc_max: float = 0.0
    duty_cycle_bc_mean: float = 0.0
    duty_cycle_bc_rms: float = 0.0
    duty_cycle_ca_min: float = 0.0
    duty_cycle_ca_max: float = 0.0
    duty_cycle_ca_mean: float = 0.0
    duty_cycle_ca_rms: float = 0.0

    # --- Dead-time impact on duty cycle ---
    # Fraction of PWM period consumed by dead time: dead_time_us * 1e-6 * pwm_frequency_hz.
    # Effective D_max = 1 - dead_time_duty_limit, D_min = dead_time_duty_limit.
    dead_time_duty_limit: float = 0.0

    # --- FFT of per-PWM-period duty cycle (Phase A, sampled at pwm_frequency_hz) ---
    duty_cycle_fft_freqs: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    duty_cycle_fft_magnitude: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # --- Common Mode Voltage: CMV = (Va + Vb + Vc) / 3 ---------------------------
    # Shares the raw PWM time axis.  For ideal balanced SVM, CMV hovers at
    # Vdc/2.  Deviations equal the zero-sequence voltage injected by the
    # modulator (triplen harmonics).  Peak-to-peak CMV is the key EMC metric
    # for bearing-current and CM-filter dimensioning.
    cmv: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    cmv_mean: float = 0.0
    cmv_rms: float = 0.0
    cmv_min: float = 0.0
    cmv_max: float = 0.0
    cmv_pp: float = 0.0

    # --- DC bus normalised current (per PWM period) --------------------------------
    # I_dc_norm(t) = Da·sin(ωt+φ) + Db·sin(ωt−2π/3+φ) + Dc·sin(ωt+2π/3+φ).
    # Shares the ``duty_cycle_time`` axis.  Values are in units of peak phase
    # current [A/A_peak].  ``dc_bus_current_norm_pp`` is proportional to the
    # RMS ripple current seen by the DC-link capacitor.
    dc_bus_current_norm: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    dc_bus_current_norm_min: float = 0.0
    dc_bus_current_norm_max: float = 0.0
    dc_bus_current_norm_rms: float = 0.0
    dc_bus_current_norm_pp: float = 0.0

    # --- dq-frame phasor diagram -------------------------------------------------
    # Clarke (αβ) trajectory — same time axis as ``time``.
    dq_valpha: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    dq_vbeta: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    # Park (dq) fundamental components — mean over the simulation window.
    dq_vd: float = 0.0
    dq_vq: float = 0.0
    # Voltage phasor in dq frame: magnitude (V) and angle (degrees).
    dq_vs_magnitude: float = 0.0
    dq_vs_angle_deg: float = 0.0
    # Current phasor in dq frame (normalised to same magnitude as voltage phasor).
    dq_id: float = 0.0
    dq_iq: float = 0.0
    dq_is_angle_deg: float = 0.0

    # --- Angle sawtooth waveforms -----------------------------------------------
    # Electrical angle θ_e wrapped to [0, 360) degrees electrical.
    theta_e_deg: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    # Mechanical angle θ_mech wrapped to [0, 360) degrees mechanical.
    theta_mech_deg: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # --- Extended αβ metrics ----------------------------------------------------
    dq_valpha_rms: float = 0.0
    dq_valpha_peak: float = 0.0
    dq_vbeta_rms: float = 0.0
    dq_vbeta_peak: float = 0.0
    # dq per-sample RMS
    dq_vd_rms: float = 0.0
    dq_vq_rms: float = 0.0
    # |Vαβ| instantaneous module and statistics
    dq_vab_magnitude: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    dq_vab_magnitude_mean: float = 0.0
    dq_vab_magnitude_rms: float = 0.0
    # |Vdq| instantaneous module and statistics
    dq_vdq_magnitude: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    dq_vdq_magnitude_mean: float = 0.0
    dq_vdq_magnitude_rms: float = 0.0

    # --- Overmodulation metrics -------------------------------------------------
    # Percentage of PWM periods where at least one phase duty cycle is fully
    # saturated (D=0 or D=1). 0.0 in the linear region; rises toward 100.0
    # as the modulation index approaches six-step operation.
    saturation_percent: float = 0.0
    # True when modulation_index > 1.0 and at least one PWM period is saturated.
    is_overmodulation: bool = False


def _state_machine_py(commanded_pwm: np.ndarray, dead_samples: int) -> np.ndarray:
    """Compute per-sample switch states for dead-time insertion.

    Returns an int8 array: +1 upper ON, -1 lower ON, 0 both OFF (dead time).
    Extracted for optional numba JIT compilation.
    """
    n = commanded_pwm.shape[0]
    switch_state = np.empty(n, dtype=np.int8)
    desired_prev = 1 if commanded_pwm[0] >= 0.0 else -1
    active_state = desired_prev
    pending_state = 0
    pending_apply_index = -1
    switch_state[0] = active_state

    for i in range(1, n):
        desired = 1 if commanded_pwm[i] >= 0.0 else -1

        if desired != desired_prev:
            active_state = 0
            pending_state = desired
            pending_apply_index = i + dead_samples

        if pending_apply_index >= 0 and i >= pending_apply_index:
            active_state = pending_state
            pending_apply_index = -1

        desired_prev = desired
        switch_state[i] = active_state

    return switch_state


if _NUMBA_AVAILABLE:
    _compute_switch_states = _njit(_state_machine_py)
else:
    _compute_switch_states = _state_machine_py


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
    switch_state = _compute_switch_states(commanded_pwm, dead_samples)

    vf = max(0.0, diode_forward_voltage_v)
    # During dead time the body diode of the conducting device clamps the output:
    # current >= 0 -> lower diode -> -Vf; current < 0 -> upper diode -> Vdc + Vf.
    dead_time_voltage = np.where(current_sign >= 0.0, -vf, battery_voltage + vf)
    phase_voltage = np.where(
        switch_state > 0,
        battery_voltage,
        np.where(switch_state < 0, 0.0, dead_time_voltage),
    )
    return phase_voltage.astype(np.float64)


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
        modulation_index=config.modulation_index,
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

    # Per-PWM-period duty cycle envelope — reveals the modulating reference waveform.
    duty_cycle_time, duty_cycle_a = compute_duty_cycle_envelope(
        phase_a, time, config.oversample, config.battery_voltage
    )
    _, duty_cycle_b = compute_duty_cycle_envelope(
        phase_b, time, config.oversample, config.battery_voltage
    )
    _, duty_cycle_c = compute_duty_cycle_envelope(
        phase_c, time, config.oversample, config.battery_voltage
    )

    # --- Duty cycle descriptive statistics (per-leg, [0,1]) ---
    def _dc_stats(dc: np.ndarray) -> tuple[float, float, float, float]:
        if dc.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(np.min(dc)),
            float(np.max(dc)),
            float(np.mean(dc)),
            float(np.sqrt(np.mean(dc**2))),
        )

    dc_a_min, dc_a_max, dc_a_mean, dc_a_rms = _dc_stats(duty_cycle_a)
    dc_b_min, dc_b_max, dc_b_mean, dc_b_rms = _dc_stats(duty_cycle_b)
    dc_c_min, dc_c_max, dc_c_mean, dc_c_rms = _dc_stats(duty_cycle_c)

    # --- Overmodulation saturation metric ----------------------------------------
    # A duty cycle period is "saturated" when the switch is either fully ON (D=1)
    # or fully OFF (D=0) for the entire switching period, meaning the reference
    # signal exceeded the carrier bounds at that point.  The saturation percentage
    # is the worst-case (maximum) fraction across the three phases.
    def _sat(dc: np.ndarray) -> float:
        if dc.size == 0:
            return 0.0
        tol = 1e-9
        return float(np.mean((dc <= tol) | (dc >= 1.0 - tol)))

    saturation_percent = float(
        max(_sat(duty_cycle_a), _sat(duty_cycle_b), _sat(duty_cycle_c)) * 100.0
    )
    is_overmodulation = (config.modulation_index > 1.0) and (saturation_percent > 0.0)

    # Phase-to-phase duty cycles: D_AB = D_A - D_B, etc.
    duty_cycle_ab = duty_cycle_a - duty_cycle_b
    duty_cycle_bc = duty_cycle_b - duty_cycle_c
    duty_cycle_ca = duty_cycle_c - duty_cycle_a
    dc_ab_min, dc_ab_max, dc_ab_mean, dc_ab_rms = _dc_stats(duty_cycle_ab)
    dc_bc_min, dc_bc_max, dc_bc_mean, dc_bc_rms = _dc_stats(duty_cycle_bc)
    dc_ca_min, dc_ca_max, dc_ca_mean, dc_ca_rms = _dc_stats(duty_cycle_ca)

    # Dead-time fraction of PWM period = time_dead / T_pwm.
    dead_time_duty_limit = (config.dead_time_us * 1e-6) * config.pwm_frequency_hz

    # FFT of the per-leg duty cycle (Phase A), sampled at pwm_frequency_hz.
    duty_cycle_fft_freqs, duty_cycle_fft_magnitude = compute_fft(
        signal=duty_cycle_a,
        sampling_rate=config.pwm_frequency_hz,
        num_cycles=config.num_cycles,
        electrical_frequency_hz=electrical_freq,
    )

    # --- Common Mode Voltage: CMV = (Va + Vb + Vc) / 3 ---------------------------
    cmv = (phase_a + phase_b + phase_c) / 3.0
    cmv_mean = float(np.mean(cmv))
    cmv_rms = float(np.sqrt(np.mean(cmv**2)))
    cmv_min = float(np.min(cmv))
    cmv_max = float(np.max(cmv))
    cmv_pp = cmv_max - cmv_min

    # --- DC bus normalised current (per PWM period) --------------------------------
    # Synthetic unit-amplitude phase currents aligned to current_phase_deg,
    # sampled at PWM-period midpoints to match the duty cycle time axis.
    if duty_cycle_time.size > 0:
        theta_dc = 2.0 * np.pi * electrical_freq * duty_cycle_time
        i_a_unit = np.sin(theta_dc + current_phase_rad)
        i_b_unit = np.sin(theta_dc - 2.0 * np.pi / 3.0 + current_phase_rad)
        i_c_unit = np.sin(theta_dc + 2.0 * np.pi / 3.0 + current_phase_rad)
        dc_bus_current_norm = (
            duty_cycle_a * i_a_unit + duty_cycle_b * i_b_unit + duty_cycle_c * i_c_unit
        )
        dc_bus_min = float(np.min(dc_bus_current_norm))
        dc_bus_max = float(np.max(dc_bus_current_norm))
        dc_bus_rms = float(np.sqrt(np.mean(dc_bus_current_norm**2)))
        dc_bus_pp = dc_bus_max - dc_bus_min
    else:
        dc_bus_current_norm = np.array([], dtype=np.float64)
        dc_bus_min = dc_bus_max = dc_bus_rms = dc_bus_pp = 0.0

    # --- Electrical and mechanical angle sawtooth waveforms ----------------------
    theta_e_deg = np.degrees(theta % (2.0 * np.pi))
    theta_mech_deg = np.degrees((theta / config.motor_pole_pairs) % (2.0 * np.pi))

    # --- dq-frame phasor diagram --------------------------------------------------
    dq = compute_dq_phasors(
        phase_a=phase_a,
        phase_b=phase_b,
        phase_c=phase_c,
        time=time,
        electrical_freq_hz=electrical_freq,
        battery_voltage=config.battery_voltage,
        current_phase_deg=float(config.current_phase_deg),
    )

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
        duty_cycle_time=duty_cycle_time,
        duty_cycle_a=duty_cycle_a,
        duty_cycle_b=duty_cycle_b,
        duty_cycle_c=duty_cycle_c,
        duty_cycle_a_min=dc_a_min,
        duty_cycle_a_max=dc_a_max,
        duty_cycle_a_mean=dc_a_mean,
        duty_cycle_a_rms=dc_a_rms,
        duty_cycle_b_min=dc_b_min,
        duty_cycle_b_max=dc_b_max,
        duty_cycle_b_mean=dc_b_mean,
        duty_cycle_b_rms=dc_b_rms,
        duty_cycle_c_min=dc_c_min,
        duty_cycle_c_max=dc_c_max,
        duty_cycle_c_mean=dc_c_mean,
        duty_cycle_c_rms=dc_c_rms,
        duty_cycle_ab=duty_cycle_ab,
        duty_cycle_bc=duty_cycle_bc,
        duty_cycle_ca=duty_cycle_ca,
        duty_cycle_ab_min=dc_ab_min,
        duty_cycle_ab_max=dc_ab_max,
        duty_cycle_ab_mean=dc_ab_mean,
        duty_cycle_ab_rms=dc_ab_rms,
        duty_cycle_bc_min=dc_bc_min,
        duty_cycle_bc_max=dc_bc_max,
        duty_cycle_bc_mean=dc_bc_mean,
        duty_cycle_bc_rms=dc_bc_rms,
        duty_cycle_ca_min=dc_ca_min,
        duty_cycle_ca_max=dc_ca_max,
        duty_cycle_ca_mean=dc_ca_mean,
        duty_cycle_ca_rms=dc_ca_rms,
        dead_time_duty_limit=dead_time_duty_limit,
        duty_cycle_fft_freqs=duty_cycle_fft_freqs,
        duty_cycle_fft_magnitude=duty_cycle_fft_magnitude,
        cmv=cmv,
        cmv_mean=cmv_mean,
        cmv_rms=cmv_rms,
        cmv_min=cmv_min,
        cmv_max=cmv_max,
        cmv_pp=cmv_pp,
        dc_bus_current_norm=dc_bus_current_norm,
        dc_bus_current_norm_min=dc_bus_min,
        dc_bus_current_norm_max=dc_bus_max,
        dc_bus_current_norm_rms=dc_bus_rms,
        dc_bus_current_norm_pp=dc_bus_pp,
        dq_valpha=dq["valpha"],
        dq_vbeta=dq["vbeta"],
        dq_vd=dq["vd_mean"],
        dq_vq=dq["vq_mean"],
        dq_vs_magnitude=dq["vs_magnitude"],
        dq_vs_angle_deg=dq["vs_angle_deg"],
        dq_id=dq["id_fund"],
        dq_iq=dq["iq_fund"],
        dq_is_angle_deg=dq["is_angle_deg"],
        theta_e_deg=theta_e_deg,
        theta_mech_deg=theta_mech_deg,
        dq_valpha_rms=dq["valpha_rms"],
        dq_valpha_peak=dq["valpha_peak"],
        dq_vbeta_rms=dq["vbeta_rms"],
        dq_vbeta_peak=dq["vbeta_peak"],
        dq_vd_rms=dq["vd_rms"],
        dq_vq_rms=dq["vq_rms"],
        dq_vab_magnitude=dq["vab_magnitude"],
        dq_vab_magnitude_mean=dq["vab_magnitude_mean"],
        dq_vab_magnitude_rms=dq["vab_magnitude_rms"],
        dq_vdq_magnitude=dq["vdq_magnitude"],
        dq_vdq_magnitude_mean=dq["vdq_magnitude_mean"],
        dq_vdq_magnitude_rms=dq["vdq_magnitude_rms"],
        saturation_percent=saturation_percent,
        is_overmodulation=is_overmodulation,
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
