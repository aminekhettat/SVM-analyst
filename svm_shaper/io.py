"""Import/export helpers for SVM Shaper.

This module provides helpers to export waveform data, FFT spectra, plots,
 and simulation configurations for offline analysis and reporting.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core import SimulationResult, SimulatorConfig
from .modulations import ModulationMode


def export_waveform_csv(
    path: str | Path, sim: SimulationResult, labels: list[str]
) -> None:
    """Export waveform data to a CSV file.

    The CSV contains a time column and three waveform columns (A/B/C), which
    may represent phase voltages or line voltages depending on the caller.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.vstack((sim.time, sim.phase_a, sim.phase_b, sim.phase_c)).T
    header = "time," + ",".join(labels)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6e")


def export_fft_csv(path: str | Path, sim: SimulationResult) -> None:
    """Export FFT frequency and magnitude to a CSV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.vstack((sim.fft_freqs, sim.fft_magnitude)).T
    header = "frequency,magnitude"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6e")


def export_plot_png(path: str | Path, figure) -> None:
    """Save a Matplotlib Figure to a PNG file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=150)


def save_config(path: str | Path, config: SimulatorConfig) -> None:
    """Save simulation configuration to a JSON file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def load_config(path: str | Path) -> SimulatorConfig:
    """Load a simulation configuration from a JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert modulation string back to ModulationMode enum
    if isinstance(data.get("modulation"), str):
        try:
            data["modulation"] = ModulationMode(data["modulation"])
        except ValueError:
            # Fallback to default if the saved modulation is unrecognized
            data["modulation"] = ModulationMode.SVM

    return SimulatorConfig(**data)
