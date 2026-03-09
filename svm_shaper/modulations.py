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

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Tuple

import numpy as np


class ModulationMode(str, Enum):
    """Supported modulation methods."""

    THIPWM_1_6 = "THIPWM 1/6"
    THIPWM_1_4 = "THIPWM 1/4"
    SVM = "SVM"
    DPWM_120_MAX = "DPWM 120° (MAX)"
    DPWM_120_MIN = "DPWM 120° (MIN)"
    DPWM_60_1 = "DPWM 60° (DPWM1)"
    DPWM_60_0 = "DPWM 60° (DPWM0)"
    DPWM_60_2 = "DPWM 60° (DPWM2)"
    DPWM_30_3 = "DPWM 30° (DPWM3)"


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


def _triangle_carrier(time: np.ndarray, pwm_frequency_hz: float) -> np.ndarray:
    """Triangular carrier waveform from -1 to +1 at the PWM frequency."""

    period = 1.0 / pwm_frequency_hz
    # Generate a triangle wave in [0,1] then map to [-1,1]
    phase = (time % period) / period
    tri = np.abs(2.0 * phase - 1.0)
    return 2.0 * tri - 1.0


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

    va = np.sin(theta) + x * np.sin(3.0 * theta)
    vb = np.sin(theta - 2.0 * np.pi / 3.0) + x * np.sin(3.0 * theta)
    vc = np.sin(theta + 2.0 * np.pi / 3.0) + x * np.sin(3.0 * theta)

    va = _normalize(va)
    vb = _normalize(vb)
    vc = _normalize(vc)
    return va, vb, vc


def _svm_reference(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a reference that is equivalent to SVM by using third-harmonic injection.

    The reference defined here is equivalent to the standard SVM implementation and
    is useful for comparing FFT/THD, since the resulting PWM looks identical to a
    carrier-comparison-based SVM.
    """

    # For SVM the equivalent common-mode injection is (Vmax + Vmin)/2. In a
    # normalized sine reference that is equivalent to injecting 1/6 of the third harmonic.
    return _thipwm_reference(theta, x=1.0 / 6.0)


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
    oversample: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    if modulation == ModulationMode.THIPWM_1_6:
        va_ref, vb_ref, vc_ref = _thipwm_reference(theta, x=1.0 / 6.0)
    elif modulation == ModulationMode.THIPWM_1_4:
        va_ref, vb_ref, vc_ref = _thipwm_reference(theta, x=1.0 / 4.0)
    elif modulation == ModulationMode.SVM:
        va_ref, vb_ref, vc_ref = _svm_reference(theta)
    else:
        # For DPWM methods we start from an SVM-equivalent reference and then
        # modify the zero vector placement by clamping one phase for a portion of
        # the electrical cycle.
        va_ref, vb_ref, vc_ref = _svm_reference(theta)

    carrier = _triangle_carrier(time, pwm_frequency_hz)
    phase_a = np.where(va_ref >= carrier, 1.0, -1.0)
    phase_b = np.where(vb_ref >= carrier, 1.0, -1.0)
    phase_c = np.where(vc_ref >= carrier, 1.0, -1.0)

    # Apply discontinuous PWM adjustments if requested
    if modulation in (ModulationMode.DPWM_120_MAX, ModulationMode.DPWM_120_MIN):
        # One phase is clamped for 120 degrees (2 sectors). In this implementation,
        # we clamp the phase that is highest in the reference at each time.
        clamp_value = +1.0 if modulation == ModulationMode.DPWM_120_MAX else -1.0
        sector = np.floor((theta % (2 * np.pi)) / (np.pi / 3.0)).astype(int)
        # Choose which phase to clamp by sector (each pair of sectors clamps a different phase)
        clamp_phase = (sector // 2) % 3
        for i in range(time.size):
            if clamp_phase[i] == 0:
                phase_a[i] = clamp_value
            elif clamp_phase[i] == 1:
                phase_b[i] = clamp_value
            else:
                phase_c[i] = clamp_value

    elif modulation in (
        ModulationMode.DPWM_60_1,
        ModulationMode.DPWM_60_0,
        ModulationMode.DPWM_60_2,
    ):
        # 60° DPWM: clamp one phase for 60° (one sector) and rotate the clamped phase.
        # The variants differ in which phase is clamped first.
        sector = np.floor((theta % (2 * np.pi)) / (np.pi / 3.0)).astype(int)
        if modulation == ModulationMode.DPWM_60_1:
            clamp_map = [0, 1, 2, 0, 1, 2]
        elif modulation == ModulationMode.DPWM_60_0:
            clamp_map = [1, 2, 0, 1, 2, 0]
        else:
            clamp_map = [2, 0, 1, 2, 0, 1]
        clamp_phase = [clamp_map[s % 6] for s in sector]
        clamp_value = -1.0  # clamp to -Vdc to reduce switching losses
        for i in range(time.size):
            if clamp_phase[i] == 0:
                phase_a[i] = clamp_value
            elif clamp_phase[i] == 1:
                phase_b[i] = clamp_value
            else:
                phase_c[i] = clamp_value

    elif modulation == ModulationMode.DPWM_30_3:
        # 30° DPWM: clamp each phase for 30° increments.
        # For simplicity, we clamp the phase that has the largest instantaneous
        # reference magnitude to reduce switching.
        ref_mag = np.vstack((np.abs(va_ref), np.abs(vb_ref), np.abs(vc_ref)))
        clamp_phase = np.argmax(ref_mag, axis=0)
        for i in range(time.size):
            if clamp_phase[i] == 0:
                phase_a[i] = -1.0
            elif clamp_phase[i] == 1:
                phase_b[i] = -1.0
            else:
                phase_c[i] = -1.0

    return time, phase_a, phase_b, phase_c
