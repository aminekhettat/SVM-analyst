"""Unit tests for parameter sweep utilities.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

import numpy as np

from svm_shaper.core import SimulatorConfig
from svm_shaper.sweep import sweep_thd


def test_sweep_thd_decreases_with_speed():
    config = SimulatorConfig()
    # Sweep speed from low (100) to higher (1000) RPM. THD may change.
    xs, thd = sweep_thd(config, variable="speed_rpm", start=100.0, stop=1000.0, steps=5)
    assert xs.shape[0] == 5
    assert thd.shape == xs.shape
    assert np.all(thd >= 0.0)
