"""Unit tests for spectrum analysis utilities.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

import numpy as np

from svm_shaper.analysis import compute_fft, compute_thd


def test_fft_peak_at_fundamental():
    fs = 1000.0
    f0 = 50.0
    t = np.arange(0, 0.2, 1.0 / fs)
    signal = np.sin(2 * np.pi * f0 * t)

    freqs, mag = compute_fft(
        signal, sampling_rate=fs, num_cycles=10, electrical_frequency_hz=f0
    )
    # Find the index of the peak in the magnitude spectrum
    peak_idx = np.argmax(mag)
    assert abs(freqs[peak_idx] - f0) < 1.0


def test_thd_of_pure_sine_is_zero():
    fs = 1000.0
    f0 = 60.0
    # Use an integer number of cycles to minimize spectral leakage in the test.
    t = np.arange(0, 1.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * f0 * t)

    freqs, mag = compute_fft(
        signal, sampling_rate=fs, num_cycles=60, electrical_frequency_hz=f0
    )
    thd = compute_thd(mag, fundamental_hz=f0, freqs=freqs)
    assert thd < 1e-2
