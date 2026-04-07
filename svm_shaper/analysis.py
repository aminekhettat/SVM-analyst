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

        harmonic_indices = []
        for order in range(2, max_order + 1):
            target_hz = order * fundamental_hz
            idx = int(np.argmin(np.abs(freqs - target_hz)))
            if idx not in (0, idx_fund):
                harmonic_indices.append(idx)

        # Keep unique bins in case two harmonic targets map to the same FFT bin.
        if harmonic_indices:
            harmonic_indices = sorted(set(harmonic_indices))
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
