"""SVM Shaper: educational simulator for PWM modulation techniques.

This package provides simulation functions, analysis utilities, and a PyQt-based
GUI to visualize PWM waveforms and their harmonic content.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from .core import SimulatorConfig, run_simulation
from .gui import SvmShaperApp

__all__ = [
    "SimulatorConfig",
    "run_simulation",
    "SvmShaperApp",
]
