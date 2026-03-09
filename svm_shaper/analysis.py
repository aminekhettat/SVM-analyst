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

    If freqs is provided, determines the fundamental component closest to the
    provided frequency. Otherwise, assumes the first nonzero bin is the
    fundamental.

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

    # Consider harmonics up to Nyquist, excluding DC and fundamental.
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
