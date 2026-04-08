"""Analysis utilities for waveform processing.

This module provides FFT, THD and other helper functions used by the GUI and
unit tests.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from __future__ import annotations

import numpy as np


def compute_fft(
    signal: np.ndarray,
    sampling_rate: float,
    num_cycles: int,
    electrical_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided magnitude FFT of a time-domain signal.

    Parameters
    ----------
    signal:
        Time-domain signal samples.
    sampling_rate:
        Sampling rate in Hz.
    num_cycles:
        Number of electrical cycles included in the signal.
    electrical_frequency_hz:
        Fundamental electrical frequency in Hz.

    Returns
    -------
    freqs:
        Frequency axis values.
    magnitude:
        Single-sided magnitude spectrum.
    """

    if signal.size == 0:
        return np.array([]), np.array([])

    # Use the exact number of samples to avoid introducing interpolation.
    n = signal.size
    fft_vals = np.fft.rfft(signal, n=n)
    freqs = np.fft.rfftfreq(n, 1.0 / sampling_rate)

    # Scale to the peak amplitude for a single-sided spectrum.
    magnitude = np.abs(fft_vals) * 2.0 / n
    return freqs, magnitude


def compute_thd(
    magnitude: np.ndarray,
    fundamental_hz: float | None = None,
    freqs: np.ndarray | None = None,
) -> float:
    """Compute total harmonic distortion (THD) from an FFT magnitude spectrum.

    If freqs and fundamental_hz are provided, the THD numerator is built from
    integer harmonics only (2*f0, 3*f0, ... up to Nyquist), matching the
    conventional THD definition. Otherwise, falls back to summing all bins
    except DC and the fundamental.

    Returns
    -------
    thd_percent:
        THD in percent.
    """

    if magnitude.size == 0:
        return 0.0

    if freqs is not None and fundamental_hz is not None:
        # Find the index closest to fundamental_hz
        idx_fund = int(np.argmin(np.abs(freqs - fundamental_hz)))
    else:
        # Skip DC component at index 0
        idx_fund = 1 if magnitude.size > 1 else 0

    # Magnitude values are already doubled (single sided) and include the fundamental.
    fundamental = magnitude[idx_fund] if idx_fund < magnitude.size else 0.0
    if fundamental == 0:
        return 0.0

    if freqs is not None and fundamental_hz is not None and fundamental_hz > 0.0:
        nyquist = float(freqs[-1])
        max_order = int(np.floor(nyquist / fundamental_hz))

        # Vectorised harmonic bin selection — avoids the O(max_order × bins)
        # per-order np.argmin loop that hangs at very low fundamental frequencies.
        # For evenly-spaced rfftfreq bins, the closest bin to n·f0 is simply
        # round(n·f0 / freq_resolution).
        freq_resolution = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
        orders = np.arange(2, max_order + 1)
        target_bins = np.rint(orders * fundamental_hz / freq_resolution).astype(int)
        valid = (
            (target_bins >= 1) & (target_bins < freqs.size) & (target_bins != idx_fund)
        )
        harmonic_indices = np.unique(target_bins[valid])
        if harmonic_indices.size > 0:
            harmonic_sum_sq = float(np.sum(magnitude[harmonic_indices] ** 2))
        else:
            harmonic_sum_sq = 0.0
    else:
        # Fallback behavior when harmonic order cannot be inferred.
        harmonic_indices = [i for i in range(len(magnitude)) if i not in (0, idx_fund)]
        harmonic_sum_sq = float(np.sum(magnitude[harmonic_indices] ** 2))

    thd = np.sqrt(harmonic_sum_sq) / fundamental
    return float(thd * 100.0)


def compute_top_harmonics(
    freqs: np.ndarray, magnitude: np.ndarray, count: int = 5
) -> list[tuple[float, float]]:
    """Return the top N harmonics from an FFT magnitude spectrum.

    The returned list is sorted by magnitude descending.

    Returns
    -------
    top_harmonics:
        List of tuples (frequency, magnitude).
    """

    if freqs.size == 0 or magnitude.size == 0:
        return []

    # Exclude DC component
    indices = list(range(1, len(freqs)))
    if not indices:
        return []

    mags = magnitude[indices]
    freqs_sub = freqs[indices]

    order = np.argsort(mags)[::-1]
    top = []
    for idx in order[:count]:
        top.append((float(freqs_sub[idx]), float(mags[idx])))

    return top


def compute_duty_cycle_envelope(
    signal: np.ndarray,
    time: np.ndarray,
    oversample: int,
    battery_voltage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a per-PWM-period duty cycle envelope from a voltage waveform.

    The duty cycle for each switching period is the fraction of samples whose
    voltage exceeds half the DC-bus voltage (``battery_voltage / 2``).  Plotting
    this against time reveals the modulating reference waveform (sinusoid for
    SPWM, SVM envelope, clamped segments for DPWM, etc.).

    Parameters
    ----------
    signal:
        High-resolution phase voltage samples (e.g. ``phase_a`` from
        ``SimulationResult``).  Shape: ``(N,)``.
    time:
        Corresponding time vector in seconds.  Same length as *signal*.
    oversample:
        Number of samples per PWM period used during waveform generation.
        Must be >= 1.
    battery_voltage:
        DC-link voltage used as the HIGH-level reference.  The threshold is
        set at ``battery_voltage / 2``.

    Returns
    -------
    duty_time:
        Time at the mid-point of each PWM period.  Shape: ``(M,)`` where
        ``M = len(signal) // oversample``.
    duty:
        Duty cycle in [0, 1] for each period.  Shape: ``(M,)``.
    """

    if signal.size == 0 or oversample < 1:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    n_periods = signal.size // oversample
    if n_periods == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Reshape into (n_periods, oversample) blocks; discard trailing partial period.
    trimmed = signal[: n_periods * oversample].reshape(n_periods, oversample)
    threshold = battery_voltage * 0.5
    duty = np.mean(trimmed > threshold, axis=1).astype(np.float64)

    # Time coordinate: centre of each PWM period.
    mid_indices = np.arange(n_periods) * oversample + oversample // 2
    mid_indices = np.clip(mid_indices, 0, time.size - 1)
    duty_time = time[mid_indices]

    return duty_time, duty


def compute_dq_phasors(
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    phase_c: np.ndarray,
    time: np.ndarray,
    electrical_freq_hz: float,
    battery_voltage: float,
    current_phase_deg: float,
) -> dict:
    """Compute Clarke (αβ) and Park (dq) frame vectors from three-phase PWM waveforms.

    The three-phase voltages are centred around zero before the transforms so
    that the DC-link mid-point offset (Vdc/2) does not contribute to the
    rotating space vector.

    Parameters
    ----------
    phase_a, phase_b, phase_c:
        Per-sample leg voltages from the simulation (line voltages 0…Vdc).
    time:
        Time vector matching the phase arrays.
    electrical_freq_hz:
        Electrical frequency in Hz (used to build the Park transform angle θe).
    battery_voltage:
        DC-link voltage; used to centre the phase signals around zero.
    current_phase_deg:
        Phase lag of the fundamental current relative to the fundamental voltage
        in degrees (positive = lagging).  Used to derive the current phasor.

    Returns
    -------
    dict with keys:
        ``valpha``, ``vbeta`` — Clarke trajectory arrays (same length as input).
        ``vd_mean``, ``vq_mean`` — average Park components over the simulation.
        ``vs_magnitude`` — voltage phasor magnitude (V).
        ``vs_angle_deg`` — voltage phasor angle in the dq frame (degrees).
        ``id_fund``, ``iq_fund`` — fundamental current phasor components
            (normalised to the same magnitude as the voltage phasor for display).
        ``is_angle_deg`` — current phasor angle in the dq frame (degrees).
    """
    if phase_a.size == 0 or time.size == 0 or electrical_freq_hz <= 0.0:
        empty = np.array([], dtype=np.float64)
        return {
            "valpha": empty,
            "vbeta": empty,
            "vd_mean": 0.0,
            "vq_mean": 0.0,
            "vs_magnitude": 0.0,
            "vs_angle_deg": 0.0,
            "id_fund": 0.0,
            "iq_fund": 0.0,
            "is_angle_deg": 0.0,
            "valpha_rms": 0.0,
            "valpha_peak": 0.0,
            "vbeta_rms": 0.0,
            "vbeta_peak": 0.0,
            "vd_rms": 0.0,
            "vq_rms": 0.0,
            "vab_magnitude": empty,
            "vab_magnitude_mean": 0.0,
            "vab_magnitude_rms": 0.0,
            "vdq_magnitude": empty,
            "vdq_magnitude_mean": 0.0,
            "vdq_magnitude_rms": 0.0,
        }

    # Centre voltages around zero so the space vector origin sits at (0, 0).
    vdc_half = battery_voltage / 2.0
    va = phase_a - vdc_half
    vb = phase_b - vdc_half
    vc = phase_c - vdc_half

    # Amplitude-invariant Clarke transform: abc → αβ.
    valpha = (2.0 / 3.0) * (va - 0.5 * vb - 0.5 * vc)
    vbeta = (2.0 / 3.0) * ((np.sqrt(3.0) / 2.0) * (vb - vc))

    # Park transform: αβ → dq  (d-axis aligned to θe=0 at t=0).
    theta_e = 2.0 * np.pi * electrical_freq_hz * time
    cos_theta = np.cos(theta_e)
    sin_theta = np.sin(theta_e)
    vd = valpha * cos_theta + vbeta * sin_theta
    vq = -valpha * sin_theta + vbeta * cos_theta

    # Average over all samples: for the fundamental the Park components are DC.
    vd_mean = float(np.mean(vd))
    vq_mean = float(np.mean(vq))

    vs_magnitude = float(np.sqrt(vd_mean**2 + vq_mean**2))
    vs_angle_deg = float(np.degrees(np.arctan2(vq_mean, vd_mean)))

    # Current phasor: same magnitude as voltage phasor, lagging by current_phase_deg.
    is_angle_deg = vs_angle_deg - float(current_phase_deg)
    is_angle_rad = np.radians(is_angle_deg)
    id_fund = vs_magnitude * float(np.cos(is_angle_rad))
    iq_fund = vs_magnitude * float(np.sin(is_angle_rad))

    # αβ signal metrics
    valpha_rms = float(np.sqrt(np.mean(valpha**2)))
    valpha_peak = float(np.max(np.abs(valpha)))
    vbeta_rms = float(np.sqrt(np.mean(vbeta**2)))
    vbeta_peak = float(np.max(np.abs(vbeta)))

    # dq RMS (per-sample, over the full simulation window)
    vd_rms = float(np.sqrt(np.mean(vd**2)))
    vq_rms = float(np.sqrt(np.mean(vq**2)))

    # |Vαβ| instantaneous module and its statistics
    vab_magnitude = np.sqrt(valpha**2 + vbeta**2)
    vab_magnitude_mean = float(np.mean(vab_magnitude))
    vab_magnitude_rms = float(np.sqrt(np.mean(vab_magnitude**2)))

    # |Vdq| instantaneous module and its statistics
    vdq_magnitude = np.sqrt(vd**2 + vq**2)
    vdq_magnitude_mean = float(np.mean(vdq_magnitude))
    vdq_magnitude_rms = float(np.sqrt(np.mean(vdq_magnitude**2)))

    return {
        "valpha": valpha,
        "vbeta": vbeta,
        "vd_mean": vd_mean,
        "vq_mean": vq_mean,
        "vs_magnitude": vs_magnitude,
        "vs_angle_deg": vs_angle_deg,
        "id_fund": id_fund,
        "iq_fund": iq_fund,
        "is_angle_deg": is_angle_deg,
        "valpha_rms": valpha_rms,
        "valpha_peak": valpha_peak,
        "vbeta_rms": vbeta_rms,
        "vbeta_peak": vbeta_peak,
        "vd_rms": vd_rms,
        "vq_rms": vq_rms,
        "vab_magnitude": vab_magnitude,
        "vab_magnitude_mean": vab_magnitude_mean,
        "vab_magnitude_rms": vab_magnitude_rms,
        "vdq_magnitude": vdq_magnitude,
        "vdq_magnitude_mean": vdq_magnitude_mean,
        "vdq_magnitude_rms": vdq_magnitude_rms,
    }
