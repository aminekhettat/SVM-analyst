"""Modulation algorithms for PWM waveforms.

This module contains implementations of the modulation techniques presented in
"A comparison of modulation techniques and motor performance evaluation" (Chalmers 2018).

Each modulation function produces three phase PWM waveforms (A/B/C) normalized
between -1 and +1 (where +1 corresponds to +Vdc/2 and -1 corresponds to -Vdc/2).

All functions are implemented in a deterministic, vectorized manner to support
interactive plotting and unit testing.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np


class ModulationMode(str, Enum):
    """Supported modulation methods."""

    SINUSOIDAL = "Sinusoidal"
    THIPWM_1_6 = "THIPWM 1/6"
    THIPWM_1_4 = "THIPWM 1/4"
    CUSTOM_THIPWM = "Custom THIPWM"
    SVM = "SVM"
    DPWM_120_MAX = "DPWM 120° (MAX)"
    DPWM_120_MIN = "DPWM 120° (MIN)"
    DPWM_60_1 = "DPWM 60° (DPWM1)"
    DPWM_60_0 = "DPWM 60° (DPWM0)"
    DPWM_60_2 = "DPWM 60° (DPWM2)"
    DPWM_30_3 = "DPWM 30° (DPWM3)"


class PulseAlignment(str, Enum):
    """PWM pulse alignment modes used by MCU timers."""

    LEFT = "Left"
    RIGHT = "Right"
    CENTER = "Center"


def get_modulation_description(modulation: ModulationMode) -> str:
    """Return a short, user-facing description of the selected modulation."""

    descriptions = {
        ModulationMode.THIPWM_1_6: (
            "Third harmonic injection PWM (1/6) uses a 3rd harmonic common-mode signal "
            "to increase DC link utilization by ~15% while keeping the output fundamental "
            "sinusoidal."
        ),
        ModulationMode.THIPWM_1_4: (
            "Third harmonic injection PWM (1/4) improves harmonic symmetry by placing "
            "the active vector interval more evenly in each carrier half-cycle."
        ),
        ModulationMode.SINUSOIDAL: (
            "Basic sinusoidal PWM comparison (no third-harmonic injection)."
        ),
        ModulationMode.THIPWM_1_6: (
            "Third harmonic injection PWM (1/6) uses a 3rd harmonic common-mode signal "
            "to increase DC link utilization by ~15% while keeping the output fundamental "
            "sinusoidal."
        ),
        ModulationMode.THIPWM_1_4: (
            "Third harmonic injection PWM (1/4) improves harmonic symmetry by placing "
            "the active vector interval more evenly in each carrier half-cycle."
        ),
        ModulationMode.CUSTOM_THIPWM: (
            "Third harmonic injection PWM with a user-adjustable injection factor (0–100%)."
        ),
        ModulationMode.SVM: (
            "Space vector modulation (SVM) uses the six active inverter states and two "
            "zero states to synthesize the reference voltage vector with minimal switching "
            "losses and maximal linear range."
        ),
        ModulationMode.DPWM_120_MAX: (
            "120° discontinuous PWM (DPWM) with positive clamping (MAX) locks one leg "
            "to +Vdc for 120° to reduce switching losses at the expense of harmonics."
        ),
        ModulationMode.DPWM_120_MIN: (
            "120° discontinuous PWM (DPWM) with negative clamping (MIN) locks one leg "
            "to -Vdc for 120° to reduce switching losses at the expense of harmonics."
        ),
        ModulationMode.DPWM_60_1: (
            "60° discontinuous PWM (DPWM1) alternates clamping between legs every 60° "
            "to improve conduction sharing while still reducing switching events."
        ),
        ModulationMode.DPWM_60_0: (
            "60° discontinuous PWM (DPWM0) is a variant of 60° DPWM with a different "
            "zero vector selection strategy to influence harmonic content."
        ),
        ModulationMode.DPWM_60_2: (
            "60° discontinuous PWM (DPWM2) alternates clamping patterns to balance "
            "conduction and harmonics across the three phases."
        ),
        ModulationMode.DPWM_30_3: (
            "30° discontinuous PWM (DPWM3) clamps each leg for only 30° per cycle, "
            "further reducing switching losses while increasing harmonic distortion."
        ),
    }
    return descriptions.get(modulation, "Unknown modulation mode")


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize a waveform so that its maximum absolute value is 1."""

    peak = np.max(np.abs(signal))
    if peak <= 0:
        return signal
    return signal / peak


def _carrier_waveform(
    time: np.ndarray,
    pwm_frequency_hz: float,
    alignment: PulseAlignment,
) -> np.ndarray:
    """Carrier waveform from -1 to +1 for the selected PWM alignment."""

    period = 1.0 / pwm_frequency_hz
    phase = (time % period) / period

    if alignment == PulseAlignment.LEFT:
        # Edge-aligned up-count equivalent.
        return 2.0 * phase - 1.0

    if alignment == PulseAlignment.RIGHT:
        # Edge-aligned down-count equivalent.
        return 1.0 - 2.0 * phase

    # Center-aligned up-down timer equivalent.
    tri = np.abs(2.0 * phase - 1.0)
    return 2.0 * tri - 1.0


def _apply_dead_time(signal: np.ndarray, dead_samples: int) -> np.ndarray:
    """Delay each switching event by a fixed number of samples."""

    if dead_samples <= 0 or signal.size < 2:
        return signal

    out = np.empty_like(signal)
    current_state = signal[0]
    desired_prev = signal[0]
    pending_state = None
    pending_apply_index = -1
    out[0] = current_state

    for i in range(1, signal.size):
        desired = signal[i]

        if pending_state is not None and i >= pending_apply_index:
            current_state = pending_state
            pending_state = None

        if desired != desired_prev:
            pending_state = desired
            pending_apply_index = i + dead_samples

        desired_prev = desired
        out[i] = current_state

    return out


def _pwm_compare(ref: np.ndarray, carrier: np.ndarray) -> np.ndarray:
    """Compare reference and carrier waveforms to create PWM outputs."""

    return np.where(ref >= carrier, 1.0, -1.0)


def _phase_reference(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the three-phase sine references for a given electrical angle array."""

    va = np.sin(theta)
    vb = np.sin(theta - 2.0 * np.pi / 3.0)
    vc = np.sin(theta + 2.0 * np.pi / 3.0)
    return va, vb, vc


def _thipwm_reference(
    theta: np.ndarray, x: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the three-phase THIPWM reference signals using a common third harmonic."""

    # Thesis Eq. (4.1)-(4.3)/(4.6)-(4.8): 1.15 scaling with third-harmonic
    # common-mode term. We then apply a common normalization across all phases to
    # keep the carrier-comparison references in [-1, 1].
    sin3 = np.sin(3.0 * theta)
    va = 1.15 * (np.sin(theta) + x * sin3)
    vb = 1.15 * (np.sin(theta - 2.0 * np.pi / 3.0) + x * sin3)
    vc = 1.15 * (np.sin(theta + 2.0 * np.pi / 3.0) + x * sin3)

    peak = float(np.max(np.abs(np.vstack((va, vb, vc)))))
    if peak > 1.0:
        va = va / peak
        vb = vb / peak
        vc = vc / peak
    return va, vb, vc


def _svm_reference(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a reference that is equivalent to SVM by using third-harmonic injection.

    The reference defined here is equivalent to the standard SVM implementation and
    is useful for comparing FFT/THD, since the resulting PWM looks identical to a
    carrier-comparison-based SVM.
    """

    # Thesis Eq. (4.9): apply common-mode component from instantaneous max/min.
    va, vb, vc = _phase_reference(theta)
    vmax = np.maximum(np.maximum(va, vb), vc)
    vmin = np.minimum(np.minimum(va, vb), vc)
    ucm = 0.5 * (vmax + vmin)

    va = va - ucm
    vb = vb - ucm
    vc = vc - ucm
    return va, vb, vc


def _custom_thipwm_reference(
    theta: np.ndarray, injection_percent: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a custom third-harmonic injection reference.

    The injection parameter is specified as a percentage of the standard 1/6
    injection. For example, 100% corresponds to x=1/6, while 0% yields a pure
    sinusoid.
    """

    x = (injection_percent / 100.0) * (1.0 / 6.0)
    return _thipwm_reference(theta, x=x)


def _dpwm_clamp(mask: np.ndarray, phase: np.ndarray, clamp_value: float) -> np.ndarray:
    """Apply a clamping mask to a phase waveform.

    Parameters
    ----------
    mask:
        Boolean array indicating where to apply the clamp.
    phase:
        Original phase waveform values in [-1, 1].
    clamp_value:
        The value (-1 or +1) to clamp to.
    """

    out = phase.copy()
    out[mask] = clamp_value
    return out


def generate_modulated_pwm(
    modulation: ModulationMode,
    pole_pairs: int,
    speed_rpm: float,
    pwm_frequency_hz: float,
    num_cycles: int = 3,
    oversample: int = 50,
    injection_percent: float = 100.0,
    alignment: PulseAlignment = PulseAlignment.CENTER,
    dead_time_s: float = 0.0,
    modulation_index: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate three-phase PWM waveforms for a given modulation mode.

    This function supports a basic sinusoidal PWM reference, third-harmonic
    injection variations (including adjustable injection factor), space-vector
    modulation, and a set of discontinuous PWM modes.

    Parameters
    ----------
    modulation:
        The modulation technique to simulate.
    pole_pairs:
        Number of motor pole pairs. Used to calculate electrical frequency.
    speed_rpm:
        Rotor speed in RPM.
    pwm_frequency_hz:
        PWM carrier frequency in Hz.
    num_cycles:
        Number of electrical cycles to generate.
    oversample:
        Samples per PWM period (higher values give better resolution).
    injection_percent:
        Third harmonic injection factor expressed as a percentage of the standard
        1/6 injection. Only used for CUSTOM_THIPWM mode.
    alignment:
        PWM pulse alignment mode (left, right, center).
    dead_time_s:
        Dead-time delay applied to each phase switching event in seconds.

    Returns
    -------
    time:
        Time vector (s).
    phase_a, phase_b, phase_c:
        Normalized PWM waveforms for each phase (range [-1, +1]).
    """
    """Generate three-phase PWM waveforms for a given modulation mode.

    Parameters
    ----------
    modulation:
        The modulation technique to simulate.
    pole_pairs:
        Number of motor pole pairs. Used to calculate electrical frequency.
    speed_rpm:
        Rotor speed in RPM.
    pwm_frequency_hz:
        PWM carrier frequency in Hz.
    num_cycles:
        Number of electrical cycles to generate.
    oversample:
        Samples per PWM period (must be >= 3).

    Returns
    -------
    time:
        Time vector (s).
    phase_a, phase_b, phase_c:
        Normalized PWM waveforms for each phase (range [-1, +1]).
    """

    assert oversample >= 3, "oversample must be >= 3"

    electrical_freq = (speed_rpm / 60.0) * pole_pairs
    if electrical_freq <= 0:
        electrical_freq = 1e-9

    # Total duration to cover the requested number of electrical cycles
    total_time = num_cycles / electrical_freq
    dt = 1.0 / (pwm_frequency_hz * oversample)
    time = np.arange(0.0, total_time, dt)

    theta = 2.0 * np.pi * electrical_freq * time

    if modulation == ModulationMode.SINUSOIDAL:
        va_ref, vb_ref, vc_ref = _phase_reference(theta)
    elif modulation == ModulationMode.THIPWM_1_6:
        va_ref, vb_ref, vc_ref = _thipwm_reference(theta, x=1.0 / 6.0)
    elif modulation == ModulationMode.THIPWM_1_4:
        va_ref, vb_ref, vc_ref = _thipwm_reference(theta, x=1.0 / 4.0)
    elif modulation == ModulationMode.CUSTOM_THIPWM:
        va_ref, vb_ref, vc_ref = _custom_thipwm_reference(
            theta, injection_percent=injection_percent
        )
    elif modulation == ModulationMode.SVM:
        va_ref, vb_ref, vc_ref = _svm_reference(theta)
    else:
        # For DPWM methods we start from an SVM-equivalent reference and then
        # modify the zero vector placement by clamping one phase for a portion of
        # the electrical cycle.
        va_ref, vb_ref, vc_ref = _svm_reference(theta)

    # Scale reference signals by the modulation index. MI=1.0 keeps the signals
    # within the carrier bounds (linear region). MI>1.0 causes reference
    # excursions beyond ±1, resulting in duty-cycle clamping (overmodulation).
    # For sinusoidal/THIPWM modes the linear boundary is MI=1.0; for SVM the
    # references peak at ~0.866 so the linear boundary is near MI=1.15.
    if modulation_index != 1.0:
        va_ref = va_ref * modulation_index
        vb_ref = vb_ref * modulation_index
        vc_ref = vc_ref * modulation_index

    carrier = _carrier_waveform(time, pwm_frequency_hz, alignment)

    if modulation in (
        ModulationMode.DPWM_120_MAX,
        ModulationMode.DPWM_120_MIN,
        ModulationMode.DPWM_60_1,
        ModulationMode.DPWM_60_0,
        ModulationMode.DPWM_60_2,
        ModulationMode.DPWM_30_3,
    ):
        # Unified voltage modulation (thesis Eq. 4.15-4.26): compute shifted gate
        # times via a single Toffset degree of freedom.
        tas = 0.5 * (va_ref + 1.0)
        tbs = 0.5 * (vb_ref + 1.0)
        tcs = 0.5 * (vc_ref + 1.0)

        tmax = np.maximum(np.maximum(tas, tbs), tcs)
        tmin = np.minimum(np.minimum(tas, tbs), tcs)
        ts = 1.0

        cond_60 = (tmin + tmax) >= ts
        cond_60_shift_m30 = np.sin(theta - np.pi / 6.0) >= 0.0
        cond_30 = np.sin(6.0 * theta) >= 0.0

        if modulation == ModulationMode.DPWM_120_MAX:
            toffset = ts - tmax
        elif modulation == ModulationMode.DPWM_120_MIN:
            toffset = -tmin
        elif modulation == ModulationMode.DPWM_60_1:
            # Thesis Eq. (4.21)-(4.22)
            toffset = np.where(cond_60, ts - tmax, -tmin)
        elif modulation == ModulationMode.DPWM_60_0:
            # Thesis Eq. (4.23)-(4.24)
            toffset = np.where(cond_60, -tmin, ts - tmax)
        elif modulation == ModulationMode.DPWM_60_2:
            # +/-30° shifted variant: apply a 30° retarded decision boundary.
            toffset = np.where(cond_60_shift_m30, ts - tmax, -tmin)
        else:
            # DPWM3 (30°): alternate DPWM1/DPWM0 every 30° electrical interval.
            toffset_1 = np.where(cond_60, ts - tmax, -tmin)
            toffset_0 = np.where(cond_60, -tmin, ts - tmax)
            toffset = np.where(cond_30, toffset_1, toffset_0)

        tga = np.clip(tas + toffset, 0.0, 1.0)
        tgb = np.clip(tbs + toffset, 0.0, 1.0)
        tgc = np.clip(tcs + toffset, 0.0, 1.0)

        va_ref_mod = 2.0 * tga - 1.0
        vb_ref_mod = 2.0 * tgb - 1.0
        vc_ref_mod = 2.0 * tgc - 1.0

        phase_a = _pwm_compare(va_ref_mod, carrier)
        phase_b = _pwm_compare(vb_ref_mod, carrier)
        phase_c = _pwm_compare(vc_ref_mod, carrier)
    else:
        phase_a = _pwm_compare(va_ref, carrier)
        phase_b = _pwm_compare(vb_ref, carrier)
        phase_c = _pwm_compare(vc_ref, carrier)

    if dead_time_s > 0.0:
        dt = 1.0 / (pwm_frequency_hz * oversample)
        dead_samples = int(np.round(dead_time_s / dt))
        phase_a = _apply_dead_time(phase_a, dead_samples)
        phase_b = _apply_dead_time(phase_b, dead_samples)
        phase_c = _apply_dead_time(phase_c, dead_samples)

    return time, phase_a, phase_b, phase_c
