"""Unit tests for spectrum analysis utilities.

Atomic features covered:
- compute_fft: peak at fundamental, empty input
- compute_thd: pure sine yields zero THD, non-harmonic component ignored
- compute_duty_cycle_envelope: constant HIGH, constant LOW, 50% duty, correct length, empty input

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

import numpy as np

from svm_shaper.analysis import compute_duty_cycle_envelope, compute_fft, compute_thd


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


def test_thd_ignores_non_harmonic_component():
    fs = 10000.0
    f0 = 50.0
    t = np.arange(0, 1.0, 1.0 / fs)

    # 130 Hz is not an integer harmonic of 50 Hz and should not contribute to THD.
    signal = np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 130.0 * t)

    freqs, mag = compute_fft(
        signal, sampling_rate=fs, num_cycles=50, electrical_frequency_hz=f0
    )
    thd = compute_thd(mag, fundamental_hz=f0, freqs=freqs)
    assert thd < 1e-2


# --- compute_duty_cycle_envelope ---


def test_duty_cycle_constant_high_returns_ones():
    """A signal always at Vbatt should yield duty = 1.0 for all periods."""
    vbatt = 240.0
    oversample = 50
    n_periods = 10
    signal = np.full(n_periods * oversample, vbatt)
    time = np.linspace(0, 1e-3, signal.size)
    dt, duty = compute_duty_cycle_envelope(signal, time, oversample, vbatt)
    assert len(dt) == n_periods
    assert len(duty) == n_periods
    np.testing.assert_allclose(duty, 1.0)


def test_duty_cycle_constant_low_returns_zeros():
    """A signal always at 0 V should yield duty = 0.0 for all periods."""
    vbatt = 240.0
    oversample = 50
    n_periods = 10
    signal = np.zeros(n_periods * oversample)
    time = np.linspace(0, 1e-3, signal.size)
    dt, duty = compute_duty_cycle_envelope(signal, time, oversample, vbatt)
    np.testing.assert_allclose(duty, 0.0)


def test_duty_cycle_half_duty_cycle():
    """A 50% square wave should yield duty \u2248 0.5 (within 1/oversample)."""
    vbatt = 240.0
    oversample = 50
    n_periods = 8
    half = oversample // 2
    period = np.concatenate([np.full(half, vbatt), np.zeros(oversample - half)])
    signal = np.tile(period, n_periods)
    time = np.linspace(0, 1e-3, signal.size)
    dt, duty = compute_duty_cycle_envelope(signal, time, oversample, vbatt)
    np.testing.assert_allclose(duty, 0.5, atol=1.0 / oversample)


def test_duty_cycle_output_length_matches_period_count():
    """Output arrays should have length = len(signal) // oversample."""
    vbatt = 100.0
    oversample = 20
    for n in (7, 15, 100):
        signal = np.random.default_rng(0).random(n * oversample) * vbatt
        time = np.arange(n * oversample, dtype=float)
        dt, duty = compute_duty_cycle_envelope(signal, time, oversample, vbatt)
        assert len(dt) == n
        assert len(duty) == n


def test_duty_cycle_empty_signal_returns_empty():
    """Empty input should return two empty arrays without raising."""
    dt, duty = compute_duty_cycle_envelope(
        np.array([]), np.array([]), oversample=50, battery_voltage=240.0
    )
    assert dt.size == 0
    assert duty.size == 0


def test_duty_cycle_oversample_less_than_one_returns_empty():
    """oversample < 1 should return two empty arrays."""
    signal = np.ones(100) * 240.0
    time = np.arange(100, dtype=float)
    dt, duty = compute_duty_cycle_envelope(
        signal, time, oversample=0, battery_voltage=240.0
    )
    assert dt.size == 0
    assert duty.size == 0


def test_duty_cycle_time_values_are_within_signal_time_range():
    """Returned time points must lie within the original time vector bounds."""
    vbatt = 240.0
    oversample = 50
    n_periods = 5
    signal = np.ones(n_periods * oversample) * vbatt
    time = np.linspace(0.0, 1.0e-3, signal.size)
    dt, _ = compute_duty_cycle_envelope(signal, time, oversample, vbatt)
    assert float(dt.min()) >= float(time.min())
    assert float(dt.max()) <= float(time.max())
