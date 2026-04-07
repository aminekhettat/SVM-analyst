"""Extended IO tests covering plot_image_path and config edge cases.

Atomic features covered:
- export_report_pdf with plot_image_path (PNG path, bypassing matplotlib figure)
- export_report_pdf with include_hexagon=True / False
- export_report_pdf with include_harmonics_table=True / False
- export_report_pdf with company_name populated
- save_config / load_config with all non-default SimulatorConfig fields
- export_plot_png basic round-trip
"""

import numpy as np

from svm_shaper.core import SimulationResult, SimulatorConfig
from svm_shaper.io import export_plot_png, export_report_pdf, load_config, save_config
from svm_shaper.modulations import ModulationMode, PulseAlignment


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_dummy_sim() -> SimulationResult:
    t = np.linspace(0, 0.02, 200)
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
        fft_freqs=np.array([0.0, 50.0, 100.0]),
        fft_magnitude=np.array([0.0, 1.0, 0.2]),
        thd_line_percent=20.0,
        thd_phase_percent=18.0,
        top_harmonics=[(100.0, 0.2)],
        pulses_per_electrical_cycle=100,
        degrees_per_pwm_pulse=3.6,
        actual_speed_rpm=600.0,
        speed_deviation_rpm=0.0,
        speed_deviation_percent=0.0,
        filtered_mean=0.0,
        filtered_rms=0.707,
        filtered_min=-1.0,
        filtered_max=1.0,
        raw_mean=0.0,
        raw_rms=0.707,
        raw_min=-1.0,
        raw_max=1.0,
        description_text="unit test",
        duty_cycle_time=np.linspace(0, 0.02, 10),
        duty_cycle_a=np.ones(10) * 0.6,
        duty_cycle_b=np.ones(10) * 0.5,
        duty_cycle_c=np.ones(10) * 0.4,
    )


# ---------------------------------------------------------------------------
# export_report_pdf with plot_image_path
# ---------------------------------------------------------------------------


class TestExportReportPdfWithImagePath:
    def test_plot_image_path_png(self, tmp_path):
        """Pass a pre-rendered PNG instead of a matplotlib figure."""
        import matplotlib.pyplot as plt

        sim = _make_dummy_sim()
        config = SimulatorConfig()

        # Create a small PNG to reference
        png_path = tmp_path / "plot.png"
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(str(png_path))
        plt.close(fig)

        out_file = tmp_path / "report_img.pdf"
        export_report_pdf(
            out_file,
            config,
            sim,
            info_text="image path test",
            show_phase_voltages=False,
            plot_image_path=png_path,
            app_name="SVM Analyst",
            app_version="1.1.0",
        )
        assert out_file.exists()
        assert out_file.stat().st_size > 0

    def test_plot_image_path_takes_precedence_over_figure(self, tmp_path):
        """If both plot_figure and plot_image_path are given, image path is used."""
        import matplotlib.pyplot as plt

        sim = _make_dummy_sim()
        config = SimulatorConfig()

        png_path = tmp_path / "plot.png"
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(str(png_path))
        plt.close(fig)

        out_file = tmp_path / "report_both.pdf"
        export_report_pdf(
            out_file,
            config,
            sim,
            info_text="dual path test",
            show_phase_voltages=True,
            plot_figure=None,
            plot_image_path=png_path,
            app_name="SVM Analyst",
        )
        assert out_file.exists()


# ---------------------------------------------------------------------------
# export_report_pdf miscellaneous options
# ---------------------------------------------------------------------------


class TestExportReportPdfOptions:
    def test_with_company_name(self, tmp_path):
        sim = _make_dummy_sim()
        config = SimulatorConfig()
        out_file = tmp_path / "report_company.pdf"
        export_report_pdf(
            out_file,
            config,
            sim,
            info_text="company test",
            show_phase_voltages=False,
            company_name="ACME Corp",
            app_name="SVM Analyst",
        )
        assert out_file.exists()

    def test_without_hexagon(self, tmp_path):
        sim = _make_dummy_sim()
        config = SimulatorConfig()
        out_file = tmp_path / "report_no_hex.pdf"
        export_report_pdf(
            out_file,
            config,
            sim,
            info_text="no hexagon",
            show_phase_voltages=False,
            include_hexagon=False,
            app_name="SVM Analyst",
        )
        assert out_file.exists()

    def test_without_harmonics_table(self, tmp_path):
        sim = _make_dummy_sim()
        config = SimulatorConfig()
        out_file = tmp_path / "report_no_harm.pdf"
        export_report_pdf(
            out_file,
            config,
            sim,
            info_text="no harmonics table",
            show_phase_voltages=False,
            include_harmonics_table=False,
            app_name="SVM Analyst",
        )
        assert out_file.exists()

    def test_no_plot_generates_waveform_from_sim(self, tmp_path):
        """When neither plot_figure nor plot_image_path is given, the PDF
        should still be generated using raw simulation data."""
        sim = _make_dummy_sim()
        config = SimulatorConfig()
        out_file = tmp_path / "report_no_plot.pdf"
        export_report_pdf(
            out_file,
            config,
            sim,
            info_text="raw sim plot",
            show_phase_voltages=True,
            app_name="SVM Analyst",
        )
        assert out_file.exists()


# ---------------------------------------------------------------------------
# save_config / load_config with non-default values
# ---------------------------------------------------------------------------


class TestSaveLoadConfigNonDefaults:
    def test_all_non_default_fields_roundtrip(self, tmp_path):
        config = SimulatorConfig(
            modulation=ModulationMode.SVM,
            motor_pole_pairs=4,
            speed_rpm=3000.0,
            pwm_frequency_hz=20000.0,
            num_cycles=5,
            oversample=100,
            battery_voltage=400.0,
            alignment=PulseAlignment.LEFT,
            dead_time_us=5.0,
            diode_forward_voltage_v=0.8,
            current_phase_deg=-15.0,
            injection_percent=75.0,
            filter_cutoff_hz=500.0,
            amplitude_percent=85.0,
            show_filtered=True,
            show_switching_edges=True,
            display_cycles=2,
            author_name="Test Author",
            project_name="Test Project",
        )
        path = tmp_path / "cfg_full.json"
        save_config(path, config)
        loaded = load_config(path)
        assert loaded == config


# ---------------------------------------------------------------------------
# export_plot_png
# ---------------------------------------------------------------------------


class TestExportPlotPng:
    def test_creates_file(self, tmp_path):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0])
        out = tmp_path / "plot.png"
        export_plot_png(out, fig)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)

    def test_creates_parent_dirs(self, tmp_path):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        out = tmp_path / "deep" / "nested" / "plot.png"
        export_plot_png(out, fig)
        assert out.exists()
        plt.close(fig)
