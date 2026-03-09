"""Unit tests for import/export helper functions.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

import os
from pathlib import Path

import numpy as np

from svm_shaper.core import SimulatorConfig, SimulationResult
from svm_shaper.io import (
    export_fft_csv,
    export_waveform_csv,
    load_config,
    save_config,
)


def _make_dummy_sim_result() -> SimulationResult:
    t = np.linspace(0, 0.01, 100)
    phase = np.sin(2 * np.pi * 50 * t)

    return SimulationResult(
        time=t,
        phase_a=phase,
        phase_b=phase * 0.5,
        phase_c=phase * -0.5,
        line_ab=phase - phase * 0.5,
        line_bc=phase * 0.5 - phase * -0.5,
        line_ca=phase * -0.5 - phase,
        filtered_phase_a=phase,
        filtered_phase_b=phase * 0.5,
        filtered_phase_c=phase * -0.5,
        fft_freqs=np.array([0.0, 50.0]),
        fft_magnitude=np.array([0.0, 1.0]),
        thd_percent=0.0,
        top_harmonics=[],
        pulses_per_electrical_cycle=100.0,
        degrees_per_pwm_pulse=3.6,
        description_text="test",
    )


def test_export_waveform_csv(tmp_path: Path):
    sim = _make_dummy_sim_result()
    out_file = tmp_path / "waveform.csv"
    export_waveform_csv(out_file, sim, labels=["A", "B", "C"])
    assert out_file.exists()
    text = out_file.read_text()
    assert "time,A,B,C" in text
    assert "0.000000e+00" in text


def test_export_fft_csv(tmp_path: Path):
    sim = _make_dummy_sim_result()
    out_file = tmp_path / "fft.csv"
    export_fft_csv(out_file, sim)
    assert out_file.exists()
    text = out_file.read_text()
    assert "frequency,magnitude" in text
    assert "5.000000e+01" in text


def test_save_and_load_config(tmp_path: Path):
    config = SimulatorConfig(
        motor_pole_pairs=7,
        pwm_frequency_hz=12345.0,
        speed_rpm=1500.0,
        battery_voltage=48.0,
    )
    cfg_file = tmp_path / "svm_shaper_config.json"
    save_config(cfg_file, config)

    loaded = load_config(cfg_file)
    assert loaded == config
