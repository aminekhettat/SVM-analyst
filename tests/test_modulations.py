"""Unit tests for the modulation generation functions.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

import numpy as np

from svm_shaper.modulations import (
    ModulationMode,
    generate_modulated_pwm,
)


def test_thipwm_peak_normalization():
    t, a, b, c = generate_modulated_pwm(
        modulation=ModulationMode.THIPWM_1_6,
        pole_pairs=1,
        speed_rpm=3000,
        pwm_frequency_hz=1000,
        num_cycles=1,
        oversample=10,
    )
    assert np.all(np.abs(a) <= 1.0 + 1e-9)
    assert np.all(np.abs(b) <= 1.0 + 1e-9)
    assert np.all(np.abs(c) <= 1.0 + 1e-9)


def test_svm_symmetric_waveforms():
    t, a, b, c = generate_modulated_pwm(
        modulation=ModulationMode.SVM,
        pole_pairs=1,
        speed_rpm=1500,
        pwm_frequency_hz=2000,
        num_cycles=1,
        oversample=8,
    )
    # The three phases should be 120° shifted, so their mean should be close to zero
    assert abs(np.mean(a)) < 0.05
    assert abs(np.mean(b)) < 0.05
    assert abs(np.mean(c)) < 0.05


def test_dpwm_clamping_changes_waveform():
    t1, a1, _, _ = generate_modulated_pwm(
        modulation=ModulationMode.SVM,
        pole_pairs=1,
        speed_rpm=1000,
        pwm_frequency_hz=2000,
        num_cycles=1,
        oversample=8,
    )
    t2, a2, _, _ = generate_modulated_pwm(
        modulation=ModulationMode.DPWM_120_MAX,
        pole_pairs=1,
        speed_rpm=1000,
        pwm_frequency_hz=2000,
        num_cycles=1,
        oversample=8,
    )
    assert not np.allclose(a1, a2)
