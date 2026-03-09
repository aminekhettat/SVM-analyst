"""Sweep utilities for parameter studies.

This module provides helpers to run parameter sweeps (e.g., speed or PWM
frequency) and compute resulting performance metrics (THD, etc.) for visualization
and comparison.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, Tuple

import numpy as np

from .core import SimulatorConfig, run_simulation


SweepVariable = Literal["speed_rpm", "pwm_frequency_hz"]


def sweep_thd(
    base_config: SimulatorConfig,
    variable: SweepVariable,
    start: float,
    stop: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep a parameter and return the corresponding THD values.

    Parameters
    ----------
    base_config:
        Base simulation configuration used for all sweep points.
    variable:
        The simulation parameter to sweep.
    start:
        Starting value (inclusive).
    stop:
        Ending value (inclusive).
    steps:
        Number of sweep points.

    Returns
    -------
    xs:
        Values of the swept variable.
    thd:
        Computed THD for each sweep point.
    """

    xs = np.linspace(start, stop, steps)
    thd_values = np.zeros_like(xs)

    for i, x in enumerate(xs):
        config = replace(base_config, **{variable: float(x)})
        sim = run_simulation(config)
        thd_values[i] = sim.thd_percent

    return xs, thd_values
