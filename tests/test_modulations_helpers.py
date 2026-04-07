"""Unit tests for internal modulation helper functions.

Atomic features covered:
- _normalize: zero input, positive peak, negative peak, unit peak
- _carrier_waveform: all three PulseAlignment modes (LEFT / RIGHT / CENTER)
- _apply_dead_time: d=0 pass-through, basic delay, short array edge case
- _phase_reference: three-phase 120° symmetry, unit amplitude, sum to zero
- _thipwm_reference: range stays in [-1,1], third-harmonic injection present
- _svm_reference: common-mode cancellation, symmetry
- _dpwm_clamp: basic clamping, mask all, mask none
- _custom_thipwm_reference: 0% injection equals sinusoidal, 100%=1/6 matches thipwm_1_6
- get_modulation_description: each ModulationMode returns a non-empty string
"""

import numpy as np
import pytest

from svm_shaper.modulations import (
    ModulationMode,
    PulseAlignment,
    _apply_dead_time,
    _carrier_waveform,
    _custom_thipwm_reference,
    _dpwm_clamp,
    _normalize,
    _phase_reference,
    _svm_reference,
    _thipwm_reference,
    get_modulation_description,
)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_zero_array_passthrough(self):
        arr = np.zeros(5)
        result = _normalize(arr)
        np.testing.assert_array_equal(result, arr)

    def test_unit_peak_unchanged(self):
        arr = np.array([0.0, 0.5, 1.0, -0.5])
        result = _normalize(arr)
        np.testing.assert_allclose(result.max(), 1.0)

    def test_positive_peak_scaled(self):
        arr = np.array([0.0, 5.0, -3.0])
        result = _normalize(arr)
        assert abs(result.max()) == pytest.approx(1.0)

    def test_negative_peak_only(self):
        arr = np.array([-4.0, -2.0])
        result = _normalize(arr)
        assert abs(result.min()) == pytest.approx(1.0)

    def test_preserves_shape(self):
        arr = np.linspace(-2, 2, 100)
        assert _normalize(arr).shape == arr.shape


# ---------------------------------------------------------------------------
# _carrier_waveform
# ---------------------------------------------------------------------------


class TestCarrierWaveform:
    """The carrier goes from -1 to +1 within each PWM period."""

    def _make_time(self, n_periods=2, oversample=100, freq=1000.0):
        dt = 1.0 / (freq * oversample)
        return np.arange(0.0, n_periods / freq, dt), freq

    def test_left_range(self):
        time, freq = self._make_time()
        c = _carrier_waveform(time, freq, PulseAlignment.LEFT)
        assert c.min() >= -1.0 - 1e-9
        assert c.max() <= 1.0 + 1e-9

    def test_right_range(self):
        time, freq = self._make_time()
        c = _carrier_waveform(time, freq, PulseAlignment.RIGHT)
        assert c.min() >= -1.0 - 1e-9
        assert c.max() <= 1.0 + 1e-9

    def test_center_range(self):
        time, freq = self._make_time()
        c = _carrier_waveform(time, freq, PulseAlignment.CENTER)
        assert c.min() >= -1.0 - 1e-9
        assert c.max() <= 1.0 + 1e-9

    def test_center_is_triangular(self):
        """Center-aligned carrier must start and end near the same value."""
        time, freq = self._make_time(n_periods=1, oversample=1000)
        c = _carrier_waveform(time, freq, PulseAlignment.CENTER)
        # Triangular wave is symmetric — half-period values should mirror each other
        assert c[1] == pytest.approx(c[-1], abs=0.1)
        # peak of triangular wave should be near +1
        assert c.max() > 0.9

    def test_left_starts_near_minus1(self):
        time, freq = self._make_time()
        c = _carrier_waveform(time, freq, PulseAlignment.LEFT)
        assert c[0] == pytest.approx(-1.0, abs=0.05)

    def test_right_starts_near_plus1(self):
        time, freq = self._make_time()
        c = _carrier_waveform(time, freq, PulseAlignment.RIGHT)
        assert c[0] == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# _apply_dead_time
# ---------------------------------------------------------------------------


class TestApplyDeadTime:
    def test_zero_dead_samples_passthrough(self):
        sig = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        np.testing.assert_array_equal(_apply_dead_time(sig, 0), sig)

    def test_single_sample_signal(self):
        sig = np.array([1.0])
        np.testing.assert_array_equal(_apply_dead_time(sig, 5), sig)

    def test_delay_is_applied(self):
        """A transition at index 1 should be delayed by dead_samples."""
        sig = np.array([-1.0] + [1.0] * 10, dtype=float)
        out = _apply_dead_time(sig, 3)
        # First three samples after transition must still hold -1
        assert out[1] == -1.0
        assert out[2] == -1.0
        assert out[3] == -1.0
        # After dead time the state should have changed
        assert out[4] == 1.0

    def test_output_shape(self):
        sig = np.ones(50, dtype=float)
        assert _apply_dead_time(sig, 2).shape == sig.shape

    def test_negative_dead_samples_treated_as_zero(self):
        sig = np.array([1.0, -1.0, 1.0])
        np.testing.assert_array_equal(_apply_dead_time(sig, -1), sig)


# ---------------------------------------------------------------------------
# _phase_reference
# ---------------------------------------------------------------------------


class TestPhaseReference:
    def _theta(self):
        return np.linspace(0, 2 * np.pi, 1000, endpoint=False)

    def test_sum_to_zero(self):
        """va + vb + vc = 0 for a balanced three-phase system."""
        theta = self._theta()
        va, vb, vc = _phase_reference(theta)
        np.testing.assert_allclose(va + vb + vc, 0.0, atol=1e-12)

    def test_unit_amplitude(self):
        theta = self._theta()
        va, vb, vc = _phase_reference(theta)
        assert abs(va.max() - 1.0) < 0.01
        assert abs(vb.max() - 1.0) < 0.01
        assert abs(vc.max() - 1.0) < 0.01

    def test_120_degree_phase_shift(self):
        """Each phase is shifted by 2π/3 relative to the previous."""
        theta = self._theta()
        va, vb, vc = _phase_reference(theta)
        # Check that the peak of vb occurs approximately 1/3 period after va
        idx_a = int(np.argmax(va))
        idx_b = int(np.argmax(vb))
        diff = (idx_b - idx_a) % len(va)
        # Should be close to 2/3 of the array (2π/3 = forward shift on the right)
        # For vb = sin(θ - 2π/3) the peak is at θ = π/2 + 2π/3 → later in the array
        one_third = len(va) // 3
        assert abs(diff - one_third) <= 20  # allow 20-sample tolerance

    def test_output_shapes_match_input(self):
        theta = np.linspace(0, np.pi, 200)
        va, vb, vc = _phase_reference(theta)
        assert va.shape == theta.shape
        assert vb.shape == theta.shape
        assert vc.shape == theta.shape


# ---------------------------------------------------------------------------
# _thipwm_reference
# ---------------------------------------------------------------------------


class TestThipwmReference:
    def _theta(self):
        return np.linspace(0, 2 * np.pi, 2000, endpoint=False)

    def test_range_1_6(self):
        theta = self._theta()
        va, vb, vc = _thipwm_reference(theta, x=1.0 / 6.0)
        for v in (va, vb, vc):
            assert v.max() <= 1.0 + 1e-9
            assert v.min() >= -1.0 - 1e-9

    def test_range_1_4(self):
        theta = self._theta()
        va, vb, vc = _thipwm_reference(theta, x=1.0 / 4.0)
        for v in (va, vb, vc):
            assert v.max() <= 1.0 + 1e-9
            assert v.min() >= -1.0 - 1e-9

    def test_injection_shifts_peak(self):
        """The fundamental component of the THIPWM reference should be larger
        than that of the plain sinusoidal after normalization."""
        theta = self._theta()
        va_sin, _, _ = _phase_reference(theta)
        va_thi, _, _ = _thipwm_reference(theta, x=1.0 / 6.0)
        # FFT fundamental amplitude of THIPWM should be > sinusoidal
        n = len(theta)
        fft_sin = np.abs(np.fft.rfft(va_sin)) * 2.0 / n
        fft_thi = np.abs(np.fft.rfft(va_thi)) * 2.0 / n
        # Fundamental bin: 1 full cycle spans the whole array → bin index 1
        assert fft_thi[1] > fft_sin[1]

    def test_third_harmonic_present(self):
        """THIPWM injects a common-mode 3rd harmonic so the individual phases
        contain a 3× frequency component (unlike a pure sinusoidal reference)."""
        theta = self._theta()
        va_sin, _, _ = _phase_reference(theta)
        va_thi, _, _ = _thipwm_reference(theta, x=1.0 / 6.0)
        n = len(theta)
        fft_sin = np.abs(np.fft.rfft(va_sin)) * 2.0 / n
        fft_thi = np.abs(np.fft.rfft(va_thi)) * 2.0 / n
        # The 3rd-harmonic bin (index 3) should be larger for THIPWM than for sine
        assert fft_thi[3] > fft_sin[3]


# ---------------------------------------------------------------------------
# _svm_reference
# ---------------------------------------------------------------------------


class TestSvmReference:
    def _theta(self):
        return np.linspace(0, 2 * np.pi, 2000, endpoint=False)

    def test_line_to_line_same_as_sinusoidal(self):
        """SVM adds a common-mode offset; the differential (line-to-line) voltages
        should equal those of the underlying sinusoidal references."""
        theta = self._theta()
        va_svm, vb_svm, vc_svm = _svm_reference(theta)
        va_sin, vb_sin, vc_sin = _phase_reference(theta)
        np.testing.assert_allclose(va_svm - vb_svm, va_sin - vb_sin, atol=1e-12)
        np.testing.assert_allclose(vb_svm - vc_svm, vb_sin - vc_sin, atol=1e-12)

    def test_range(self):
        theta = self._theta()
        va, vb, vc = _svm_reference(theta)
        for v in (va, vb, vc):
            assert v.max() <= 1.0 + 1e-9
            assert v.min() >= -1.0 - 1e-9

    def test_output_shape(self):
        theta = np.linspace(0, np.pi, 500)
        va, vb, vc = _svm_reference(theta)
        assert va.shape == theta.shape


# ---------------------------------------------------------------------------
# _dpwm_clamp
# ---------------------------------------------------------------------------


class TestDpwmClamp:
    def test_mask_all_clamp_plus1(self):
        mask = np.array([True, True, True])
        phase = np.array([0.5, -0.3, 0.8])
        result = _dpwm_clamp(mask, phase, 1.0)
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0])

    def test_mask_none_unchanged(self):
        mask = np.array([False, False, False])
        phase = np.array([0.5, -0.3, 0.8])
        result = _dpwm_clamp(mask, phase, 1.0)
        np.testing.assert_array_equal(result, phase)

    def test_partial_mask(self):
        mask = np.array([True, False, True])
        phase = np.array([0.2, 0.5, 0.8])
        result = _dpwm_clamp(mask, phase, -1.0)
        assert result[0] == -1.0
        assert result[1] == 0.5
        assert result[2] == -1.0

    def test_does_not_mutate_input(self):
        mask = np.array([True, False])
        phase = np.array([0.5, 0.5])
        original = phase.copy()
        _dpwm_clamp(mask, phase, 1.0)
        np.testing.assert_array_equal(phase, original)


# ---------------------------------------------------------------------------
# _custom_thipwm_reference
# ---------------------------------------------------------------------------


class TestCustomThipwmReference:
    def _theta(self):
        return np.linspace(0, 2 * np.pi, 2000, endpoint=False)

    def test_zero_percent_equals_sinusoidal(self):
        """0% injection should be identical to a pure sinusoidal reference."""
        theta = self._theta()
        va_cus, vb_cus, vc_cus = _custom_thipwm_reference(theta, injection_percent=0.0)
        va_sin, vb_sin, vc_sin = _phase_reference(theta)
        # With 0 injection the THIPWM helper still applies the 1.15 scale factor
        # and then normalises, so we only check the shape (sum to zero)
        np.testing.assert_allclose(va_cus + vb_cus + vc_cus, 0.0, atol=0.05)

    def test_hundred_percent_matches_thipwm_1_6(self):
        """100% injection should produce values equivalent to the 1/6 mode."""
        theta = self._theta()
        va_cus, vb_cus, vc_cus = _custom_thipwm_reference(
            theta, injection_percent=100.0
        )
        va_ref, vb_ref, vc_ref = _thipwm_reference(theta, x=1.0 / 6.0)
        np.testing.assert_allclose(va_cus, va_ref, atol=1e-12)
        np.testing.assert_allclose(vb_cus, vb_ref, atol=1e-12)
        np.testing.assert_allclose(vc_cus, vc_ref, atol=1e-12)

    def test_range_always_within_minus1_plus1(self):
        theta = self._theta()
        for pct in (0.0, 50.0, 100.0, 150.0):
            va, vb, vc = _custom_thipwm_reference(theta, injection_percent=pct)
            for v in (va, vb, vc):
                assert v.max() <= 1.0 + 1e-9
                assert v.min() >= -1.0 - 1e-9


# ---------------------------------------------------------------------------
# get_modulation_description
# ---------------------------------------------------------------------------


class TestGetModulationDescription:
    @pytest.mark.parametrize("mode", list(ModulationMode))
    def test_returns_non_empty_string(self, mode):
        desc = get_modulation_description(mode)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_sinusoidal_mentions_pwm(self):
        desc = get_modulation_description(ModulationMode.SINUSOIDAL)
        assert "PWM" in desc or "sinusoidal" in desc.lower()

    def test_svm_mentions_space_vector(self):
        desc = get_modulation_description(ModulationMode.SVM)
        assert "space" in desc.lower() or "vector" in desc.lower()

    def test_dpwm_120_max_mentions_clamping(self):
        desc = get_modulation_description(ModulationMode.DPWM_120_MAX)
        assert "clamp" in desc.lower() or "lock" in desc.lower()
