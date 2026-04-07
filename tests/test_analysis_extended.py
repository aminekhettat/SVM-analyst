"""Extended unit tests for the analysis module.

Atomic features covered:
- compute_fft: empty signal, basic sine components
- compute_thd: empty spectrum, pure sine (THD≈0), known harmonic content,
               mode without fundamental argument, with fundamental argument
- compute_top_harmonics: empty input, signal with distinct harmonics,
                          count clamping, count=0, result sorted by magnitude
"""

import numpy as np

from svm_shaper.analysis import compute_fft, compute_thd, compute_top_harmonics


# Sampling constants used throughout
FS = 10_000.0  # 10 kHz sampling rate
FUND = 50.0  # 50 Hz fundamental


def _make_sine(fs: float, freq: float, n_cycles: int = 5) -> np.ndarray:
    """Helper: return n_cycles of a unit-amplitude sine at *freq*."""
    total = n_cycles / freq
    t = np.arange(0.0, total, 1.0 / fs)
    return np.sin(2.0 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# compute_fft
# ---------------------------------------------------------------------------


class TestComputeFft:
    def test_empty_returns_empty(self):
        freqs, mag = compute_fft(np.array([]), FS, 1, FUND)
        assert freqs.size == 0
        assert mag.size == 0

    def test_fundamental_peak_at_correct_frequency(self):
        signal = _make_sine(FS, FUND, n_cycles=5)
        freqs, mag = compute_fft(signal, FS, 5, FUND)
        peak_idx = int(np.argmax(mag))
        assert abs(freqs[peak_idx] - FUND) < 2.0  # within 2 Hz

    def test_output_is_single_sided(self):
        """rfft produces N/2+1 real bins — the output length must be ≤ N/2+1."""
        signal = _make_sine(FS, FUND, n_cycles=3)
        freqs, mag = compute_fft(signal, FS, 3, FUND)
        assert len(freqs) == len(mag)
        assert len(freqs) <= signal.size // 2 + 1

    def test_dc_component_absent_for_pure_sine(self):
        signal = _make_sine(FS, FUND, n_cycles=5)
        freqs, mag = compute_fft(signal, FS, 5, FUND)
        # DC bin (index 0) should be tiny compared to the fundamental peak
        assert mag[0] < 0.1 * mag.max()


# ---------------------------------------------------------------------------
# compute_thd
# ---------------------------------------------------------------------------


class TestComputeThd:
    def test_empty_spectrum_returns_zero(self):
        assert compute_thd(np.array([])) == 0.0

    def test_pure_sine_thd_near_zero(self):
        signal = _make_sine(FS, FUND, n_cycles=10)
        freqs, mag = compute_fft(signal, FS, 10, FUND)
        thd = compute_thd(mag, fundamental_hz=FUND, freqs=freqs)
        assert thd < 2.0  # < 2% for a pure sine

    def test_added_harmonic_increases_thd(self):
        """Adding a 3rd harmonic at 10% amplitude should produce THD ≈10%."""
        t = np.linspace(0, 5 / FUND, int(FS * 5 / FUND), endpoint=False)
        signal = np.sin(2 * np.pi * FUND * t) + 0.10 * np.sin(2 * np.pi * 3 * FUND * t)
        freqs, mag = compute_fft(signal, FS, 5, FUND)
        thd = compute_thd(mag, fundamental_hz=FUND, freqs=freqs)
        assert 5.0 < thd < 20.0  # rough range around 10%

    def test_without_fundamental_arg_runs(self):
        signal = _make_sine(FS, FUND, n_cycles=5)
        freqs, mag = compute_fft(signal, FS, 5, FUND)
        thd = compute_thd(mag)  # no freqs/fundamental_hz kwargs
        assert thd >= 0.0

    def test_zero_fundamental_returns_zero(self):
        """If the fundamental bin is zero, THD should be 0 (no division)."""
        mag = np.zeros(50)
        thd = compute_thd(mag, fundamental_hz=FUND, freqs=np.linspace(0, 1000, 50))
        assert thd == 0.0


# ---------------------------------------------------------------------------
# compute_top_harmonics
# ---------------------------------------------------------------------------


class TestComputeTopHarmonics:
    def test_empty_arrays_returns_empty(self):
        result = compute_top_harmonics(np.array([]), np.array([]))
        assert result == []

    def test_returns_at_most_count_items(self):
        freqs = np.linspace(0, 1000, 100)
        mag = np.random.default_rng(0).random(100)
        result = compute_top_harmonics(freqs, mag, count=3)
        assert len(result) <= 3

    def test_sorted_by_magnitude_descending(self):
        freqs = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
        mag = np.array([0.0, 1.0, 0.5, 2.0, 0.1])
        result = compute_top_harmonics(freqs, mag, count=3)
        mags = [m for _, m in result]
        assert mags == sorted(mags, reverse=True)

    def test_dc_excluded(self):
        """Bin at index 0 (DC) must not appear in the top harmonics."""
        freqs = np.array([0.0, 50.0, 100.0])
        mag = np.array([999.0, 1.0, 0.5])
        result = compute_top_harmonics(freqs, mag, count=3)
        for freq, _ in result:
            assert freq != 0.0

    def test_count_zero_returns_empty(self):
        freqs = np.array([0.0, 50.0, 100.0])
        mag = np.array([0.0, 1.0, 0.5])
        result = compute_top_harmonics(freqs, mag, count=0)
        assert result == []

    def test_returns_correct_top(self):
        """The strongest harmonic should be identified correctly."""
        signal = _make_sine(FS, FUND, n_cycles=5)
        freqs, mag = compute_fft(signal, FS, 5, FUND)
        top = compute_top_harmonics(freqs, mag, count=1)
        assert len(top) == 1
        freq, _ = top[0]
        assert abs(freq - FUND) < 2.0  # closest bin to 50 Hz
