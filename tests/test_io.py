"""Unit tests for import/export helper functions.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from pathlib import Path

import numpy as np

from svm_shaper.core import SimulationResult, SimulatorConfig
from svm_shaper.io import export_fft_csv, export_waveform_csv, load_config, save_config
from svm_shaper.modulations import ModulationMode, PulseAlignment


def _make_dummy_sim_result() -> SimulationResult:
    t = np.linspace(0, 0.01, 100)
    phase = np.sin(2 * np.pi * 50 * t)

    return SimulationResult(
        time=t,
        phase_a=phase,
        phase_b=phase * 0.5,
        phase_c=phase * -0.5,
        phase_voltage_ab=phase - phase * 0.5,
        phase_voltage_bc=phase * 0.5 - phase * -0.5,
        phase_voltage_ca=phase * -0.5 - phase,
        filtered_phase_a=phase,
        filtered_phase_b=phase * 0.5,
        filtered_phase_c=phase * -0.5,
        fft_freqs=np.array([0.0, 50.0]),
        fft_magnitude=np.array([0.0, 1.0]),
        thd_line_percent=0.0,
        thd_phase_percent=0.0,
        top_harmonics=[],
        pulses_per_electrical_cycle=100,
        degrees_per_pwm_pulse=3.6,
        actual_speed_rpm=600.0,
        speed_deviation_rpm=0.0,
        speed_deviation_percent=0.0,
        filtered_mean=0.0,
        filtered_rms=0.0,
        filtered_min=0.0,
        filtered_max=0.0,
        raw_mean=0.0,
        raw_rms=0.0,
        raw_min=0.0,
        raw_max=0.0,
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
        alignment=PulseAlignment.RIGHT,
        dead_time_us=2.5,
        diode_forward_voltage_v=0.7,
        current_phase_deg=20.0,
    )
    cfg_file = tmp_path / "svm_shaper_config.json"
    save_config(cfg_file, config)

    loaded = load_config(cfg_file)
    assert loaded == config


def test_export_report_pdf(tmp_path: Path):
    sim = _make_dummy_sim_result()
    config = SimulatorConfig()
    out_file = tmp_path / "report.pdf"

    # Ensure no exception is thrown
    import matplotlib.pyplot as plt

    from svm_shaper.io import export_report_pdf

    # Create a simple figure to pass through to the report
    fig, ax = plt.subplots()
    ax.plot(sim.time, sim.phase_a)

    export_report_pdf(
        out_file,
        config,
        sim,
        info_text="Test report",
        show_phase_voltages=True,
        plot_figure=fig,
        app_name="SVM Analyst",
        app_version="0.1",
    )

    assert out_file.exists()


def test_report_includes_injection_line_only_for_custom_thipwm(tmp_path: Path) -> None:
    from svm_shaper.io import export_report_pdf

    # Use a simulation result for the report
    sim = _make_dummy_sim_result()

    # Case 1: custom THIPWM should include Injection in report
    config = SimulatorConfig(modulation=ModulationMode.CUSTOM_THIPWM)
    out_file = tmp_path / "report_injection.pdf"
    export_report_pdf(
        out_file,
        config,
        sim,
        info_text="Test report",
        show_phase_voltages=False,
        plot_figure=None,
        app_name="SVM Analyst",
        app_version="0.1",
    )
    data = out_file.read_bytes()
    assert b"Injection:" in data

    # Case 2: SVM should not include Injection in the report
    config = SimulatorConfig(modulation=ModulationMode.SVM)
    out_file = tmp_path / "report_no_injection.pdf"
    export_report_pdf(
        out_file,
        config,
        sim,
        info_text="Test report",
        show_phase_voltages=False,
        plot_figure=None,
        app_name="SVM Analyst",
        app_version="0.1",
    )
    data = out_file.read_bytes()
    assert b"Injection:" not in data


def test_report_includes_waveform_statistics(tmp_path: Path) -> None:
    from svm_shaper.io import export_report_pdf

    sim = _make_dummy_sim_result()
    config = SimulatorConfig()
    out_file = tmp_path / "report_stats.pdf"

    export_report_pdf(
        out_file,
        config,
        sim,
        info_text="Test report",
        show_phase_voltages=False,
        plot_figure=None,
        app_name="SVM Analyst",
        app_version="0.1",
    )

    data = out_file.read_bytes()
    assert b"mean" in data.lower()
    assert b"rms" in data.lower()
    assert b"min" in data.lower()
    assert b"max" in data.lower()
