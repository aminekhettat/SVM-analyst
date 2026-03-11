"""Unit tests for the modulation generation functions.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

import numpy as np

from svm_shaper.modulations import (
    ModulationMode,
    PulseAlignment,
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


def _count_phase_a_pulses(signal: np.ndarray) -> int:
    threshold = 0.5 * (float(np.min(signal)) + float(np.max(signal)))
    states = signal > threshold
    return int(np.count_nonzero((~states[:-1]) & states[1:]))


def test_all_modulations_are_two_level_and_bounded() -> None:
    for mode in ModulationMode:
        _, a, b, c = generate_modulated_pwm(
            modulation=mode,
            pole_pairs=2,
            speed_rpm=1800,
            pwm_frequency_hz=4000,
            num_cycles=1,
            oversample=10,
        )
        for phase in (a, b, c):
            assert phase.size > 0
            assert np.all(phase <= 1.0 + 1e-12)
            assert np.all(phase >= -1.0 - 1e-12)
            # Comparator output should remain two-level.
            assert set(np.unique(phase)).issubset({-1.0, 1.0})


def test_dpwm_modes_increase_clamping_vs_svm() -> None:
    _, a_svm, b_svm, c_svm = generate_modulated_pwm(
        modulation=ModulationMode.SVM,
        pole_pairs=2,
        speed_rpm=1800,
        pwm_frequency_hz=4000,
        num_cycles=1,
        oversample=20,
    )

    svm_pos_frac = max(
        np.mean(a_svm == 1.0), np.mean(b_svm == 1.0), np.mean(c_svm == 1.0)
    )
    svm_neg_frac = max(
        np.mean(a_svm == -1.0), np.mean(b_svm == -1.0), np.mean(c_svm == -1.0)
    )

    dpwm_modes = [
        ModulationMode.DPWM_120_MAX,
        ModulationMode.DPWM_120_MIN,
        ModulationMode.DPWM_60_1,
        ModulationMode.DPWM_60_0,
        ModulationMode.DPWM_60_2,
        ModulationMode.DPWM_30_3,
    ]
    for mode in dpwm_modes:
        _, a, b, c = generate_modulated_pwm(
            modulation=mode,
            pole_pairs=2,
            speed_rpm=1800,
            pwm_frequency_hz=4000,
            num_cycles=1,
            oversample=20,
        )
        pos_frac = max(np.mean(a == 1.0), np.mean(b == 1.0), np.mean(c == 1.0))
        neg_frac = max(np.mean(a == -1.0), np.mean(b == -1.0), np.mean(c == -1.0))
        assert pos_frac >= svm_pos_frac or neg_frac >= svm_neg_frac


def test_dpwm_generated_phase_a_pulses_are_reduced_vs_svm() -> None:
    # User-reported operating region where DPWM clamping should reduce pulses.
    kwargs = dict(
        pole_pairs=6,
        speed_rpm=2000,
        pwm_frequency_hz=20000,
        num_cycles=2,
        oversample=20,
    )

    _, a_svm, _, _ = generate_modulated_pwm(modulation=ModulationMode.SVM, **kwargs)
    pulses_svm = _count_phase_a_pulses(a_svm)

    for mode in (
        ModulationMode.DPWM_120_MAX,
        ModulationMode.DPWM_120_MIN,
        ModulationMode.DPWM_60_1,
        ModulationMode.DPWM_60_0,
        ModulationMode.DPWM_60_2,
        ModulationMode.DPWM_30_3,
    ):
        _, a_dpwm, _, _ = generate_modulated_pwm(modulation=mode, **kwargs)
        pulses_dpwm = _count_phase_a_pulses(a_dpwm)
        assert pulses_dpwm < pulses_svm


def _edge_indices(signal: np.ndarray) -> np.ndarray:
    return np.where(np.diff(signal) != 0)[0] + 1


def test_alignment_modes_produce_distinct_edge_placement() -> None:
    kwargs = dict(
        modulation=ModulationMode.SINUSOIDAL,
        pole_pairs=2,
        speed_rpm=1800,
        pwm_frequency_hz=4000,
        num_cycles=1,
        oversample=20,
    )
    _, a_left, _, _ = generate_modulated_pwm(
        alignment=PulseAlignment.LEFT,
        **kwargs,
    )
    _, a_right, _, _ = generate_modulated_pwm(
        alignment=PulseAlignment.RIGHT,
        **kwargs,
    )
    _, a_center, _, _ = generate_modulated_pwm(
        alignment=PulseAlignment.CENTER,
        **kwargs,
    )

    left_edges = _edge_indices(a_left)
    right_edges = _edge_indices(a_right)
    center_edges = _edge_indices(a_center)

    assert left_edges.size > 0
    assert right_edges.size > 0
    assert center_edges.size > 0
    assert not np.array_equal(left_edges, right_edges)
    assert not np.array_equal(left_edges, center_edges)


def test_dead_time_reduces_switching_event_count() -> None:
    kwargs = dict(
        modulation=ModulationMode.SINUSOIDAL,
        pole_pairs=2,
        speed_rpm=1200,
        pwm_frequency_hz=8000,
        num_cycles=1,
        oversample=20,
        alignment=PulseAlignment.CENTER,
    )
    _, a_no_dead, _, _ = generate_modulated_pwm(dead_time_s=0.0, **kwargs)
    _, a_dead, _, _ = generate_modulated_pwm(dead_time_s=5e-6, **kwargs)

    edges_no_dead = _edge_indices(a_no_dead).size
    edges_dead = _edge_indices(a_dead).size

    assert edges_dead <= edges_no_dead
    assert set(np.unique(a_dead)).issubset({-1.0, 1.0})
