"""Extended unit tests for the sweep module.

Atomic features covered:
- sweep_thd with variable="speed_rpm": output shapes, values are floats >= 0
- sweep_thd with variable="pwm_frequency_hz": output shapes, values finite
- edge case: steps=1 returns single-element arrays
- xs range matches linspace from start to stop
"""

import numpy as np
import pytest

from svm_shaper.core import SimulatorConfig
from svm_shaper.modulations import ModulationMode, PulseAlignment
from svm_shaper.sweep import sweep_thd


@pytest.fixture()
def base_cfg():
    return SimulatorConfig(
        modulation=ModulationMode.SINUSOIDAL,
        motor_pole_pairs=2,
        speed_rpm=300.0,
        pwm_frequency_hz=5000.0,
        num_cycles=2,
        oversample=20,
        battery_voltage=48.0,
        alignment=PulseAlignment.CENTER,
    )


class TestSweepThdSpeedRpm:
    def test_output_length_matches_steps(self, base_cfg):
        xs, thd = sweep_thd(base_cfg, "speed_rpm", 100.0, 500.0, 5)
        assert len(xs) == 5
        assert len(thd) == 5

    def test_xs_range_correct(self, base_cfg):
        xs, _ = sweep_thd(base_cfg, "speed_rpm", 100.0, 500.0, 5)
        assert xs[0] == pytest.approx(100.0)
        assert xs[-1] == pytest.approx(500.0)

    def test_thd_is_non_negative(self, base_cfg):
        _, thd = sweep_thd(base_cfg, "speed_rpm", 100.0, 600.0, 4)
        assert np.all(thd >= 0.0)

    def test_thd_values_are_finite(self, base_cfg):
        _, thd = sweep_thd(base_cfg, "speed_rpm", 200.0, 400.0, 3)
        assert np.all(np.isfinite(thd))


class TestSweepThdPwmFrequency:
    def test_output_length_matches_steps(self, base_cfg):
        xs, thd = sweep_thd(base_cfg, "pwm_frequency_hz", 2000.0, 10000.0, 6)
        assert len(xs) == 6
        assert len(thd) == 6

    def test_xs_range_correct(self, base_cfg):
        xs, _ = sweep_thd(base_cfg, "pwm_frequency_hz", 2000.0, 8000.0, 4)
        assert xs[0] == pytest.approx(2000.0)
        assert xs[-1] == pytest.approx(8000.0)

    def test_thd_is_finite(self, base_cfg):
        _, thd = sweep_thd(base_cfg, "pwm_frequency_hz", 3000.0, 7000.0, 3)
        assert np.all(np.isfinite(thd))

    def test_thd_is_non_negative(self, base_cfg):
        _, thd = sweep_thd(base_cfg, "pwm_frequency_hz", 5000.0, 20000.0, 4)
        assert np.all(thd >= 0.0)


class TestSweepThdEdgeCases:
    def test_single_step(self, base_cfg):
        xs, thd = sweep_thd(base_cfg, "speed_rpm", 300.0, 300.0, 1)
        assert len(xs) == 1
        assert len(thd) == 1
        assert np.isfinite(thd[0])
        assert thd[0] >= 0.0

    def test_two_steps(self, base_cfg):
        xs, thd = sweep_thd(base_cfg, "speed_rpm", 200.0, 400.0, 2)
        assert len(xs) == 2
        assert xs[0] == pytest.approx(200.0)
        assert xs[1] == pytest.approx(400.0)
