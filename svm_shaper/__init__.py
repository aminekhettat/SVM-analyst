"""SVM Analyst: educational simulator for PWM modulation techniques.

This package provides simulation functions, analysis utilities, and a PySide6-based
GUI to visualize PWM waveforms and their harmonic content.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

__version__ = "1.2.1"

from .core import SimulatorConfig, run_simulation

__all__ = [
    "__version__",
    "SimulatorConfig",
    "run_simulation",
    "SvmShaperApp",
]


def __getattr__(name: str):
    if name == "SvmShaperApp":
        from .gui import SvmShaperApp  # lazy – avoids Qt import at package init

        return SvmShaperApp
    raise AttributeError(f"module 'svm_shaper' has no attribute {name!r}")
