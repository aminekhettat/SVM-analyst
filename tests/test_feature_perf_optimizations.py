"""Unit tests for performance optimizations.

Atomic features covered:
- _state_machine_py: no-transition (idle) signal, single transition with dead-time,
  zero dead_samples, multi-transition sequence, initial-state detection
- _compute_switch_states: produces identical output to _state_machine_py
  (validates the numba JIT wrapper does not change behavior)
- _pwm_compare: always uses np.where (numba path removed), ref>=carrier->+1,
  ref<carrier->-1, equality boundary
"""

import numpy as np
import pytest

from svm_shaper.core import _compute_switch_states, _state_machine_py
from svm_shaper.modulations import _pwm_compare


# ---------------------------------------------------------------------------
# _state_machine_py
# ---------------------------------------------------------------------------


class TestStateMachinePy:
    def test_no_transitions_returns_constant_upper_state(self):
        """Constant high commanded signal -> switch stays at +1 throughout."""
        cmd = np.ones(20, dtype=np.float64) * 0.5  # all positive
        result = _state_machine_py(cmd, dead_samples=3)
        np.testing.assert_array_equal(result, np.ones(20, dtype=np.int8))

    def test_no_transitions_returns_constant_lower_state(self):
        """Constant low commanded signal -> switch stays at -1 throughout."""
        cmd = np.ones(20, dtype=np.float64) * -0.5  # all negative
        result = _state_machine_py(cmd, dead_samples=3)
        np.testing.assert_array_equal(result, np.full(20, -1, dtype=np.int8))

    def test_zero_dead_samples_no_dead_time_inserted(self):
        """When dead_samples=0 state machine should follow commanded signal directly."""
        cmd = np.array([0.5, 0.5, -0.5, -0.5, 0.5, 0.5], dtype=np.float64)
        result = _state_machine_py(cmd, dead_samples=0)
        expected = np.array([1, 1, -1, -1, 1, 1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_single_transition_high_to_low_inserts_dead_time(self):
        """Transition from +1 to -1 must insert dead_samples zeros before -1 turns on."""
        dead = 2
        # transitions at index 3: 0..2 high, 3..7 low
        cmd = np.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5], dtype=np.float64)
        result = _state_machine_py(cmd, dead_samples=dead)
        # idx 0-2: upper ON (+1)
        # idx 3: transition detected → active=0, lower ON pending at idx 3+2=5
        # idx 4: still dead time (0)
        # idx 5+: lower ON (-1)
        expected = np.array([1, 1, 1, 0, 0, -1, -1, -1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_single_transition_low_to_high_inserts_dead_time(self):
        """Transition from -1 to +1 must insert dead_samples zeros before +1 turns on."""
        dead = 3
        cmd = np.array(
            [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64
        )
        result = _state_machine_py(cmd, dead_samples=dead)
        # idx 0-2: lower ON (-1)
        # idx 3: transition → dead time, upper ON pending at idx 3+3=6
        # idx 4-5: dead time (0)
        # idx 6+: upper ON (+1)
        expected = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_boundary_command_zero_treated_as_negative(self):
        """commanded_pwm[0] == 0.0 is not >= 0.0... wait, 0.0 >= 0.0 is True -> upper ON."""
        cmd = np.zeros(5, dtype=np.float64)  # all == 0.0 -> desired = 1 (upper ON)
        result = _state_machine_py(cmd, dead_samples=2)
        np.testing.assert_array_equal(result, np.ones(5, dtype=np.int8))

    def test_output_dtype_is_int8(self):
        cmd = np.ones(10, dtype=np.float64)
        result = _state_machine_py(cmd, dead_samples=0)
        assert result.dtype == np.int8

    def test_output_length_matches_input(self):
        for n in (1, 7, 100):
            cmd = np.ones(n, dtype=np.float64)
            result = _state_machine_py(cmd, dead_samples=1)
            assert result.shape[0] == n

    def test_multi_transition_sequence(self):
        """Multiple transitions: each should insert dead time independently."""
        dead = 1
        # Pattern: high (2), low (2), high (2), low (2) = 8 samples
        cmd = np.array([0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5], dtype=np.float64)
        result = _state_machine_py(cmd, dead_samples=dead)
        # idx 0-1: +1
        # idx 2: transition H->L, dead, pending -1 at 2+1=3
        # idx 3: -1 (pending resolves)
        # idx 4: transition L->H, but pending resolves at 4+1=5 → idx 4 = 0
        # idx 5: +1
        # idx 6: transition H->L → dead, pending at 6+1=7
        # idx 7: -1
        expected = np.array([1, 1, 0, -1, 0, 1, 0, -1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# _compute_switch_states (numba JIT or fallback)
# ---------------------------------------------------------------------------


class TestComputeSwitchStates:
    """_compute_switch_states must produce the same output as _state_machine_py
    regardless of whether numba is available (tests the JIT wrapper)."""

    @pytest.mark.parametrize("dead_samples", [0, 1, 3, 5])
    def test_matches_python_reference_constant_high(self, dead_samples):
        cmd = np.ones(30, dtype=np.float64)
        ref = _state_machine_py(cmd, dead_samples)
        jit = _compute_switch_states(cmd, dead_samples)
        np.testing.assert_array_equal(jit, ref)

    @pytest.mark.parametrize("dead_samples", [0, 1, 3])
    def test_matches_python_reference_single_transition(self, dead_samples):
        cmd = np.concatenate([np.ones(10), -np.ones(10)]).astype(np.float64)
        ref = _state_machine_py(cmd, dead_samples)
        jit = _compute_switch_states(cmd, dead_samples)
        np.testing.assert_array_equal(jit, ref)

    @pytest.mark.parametrize("dead_samples", [0, 2, 4])
    def test_matches_python_reference_alternating(self, dead_samples):
        # Alternating +/-0.5 in chunks of 6
        cmd = np.tile(np.repeat([0.5, -0.5], 6), 5).astype(np.float64)
        ref = _state_machine_py(cmd, dead_samples)
        jit = _compute_switch_states(cmd, dead_samples)
        np.testing.assert_array_equal(jit, ref)

    def test_output_dtype_is_int8(self):
        cmd = np.ones(10, dtype=np.float64)
        result = _compute_switch_states(cmd, 2)
        assert result.dtype == np.int8


# ---------------------------------------------------------------------------
# _pwm_compare (always np.where, numba path removed)
# ---------------------------------------------------------------------------


class TestPwmCompare:
    def test_ref_greater_than_carrier_returns_one(self):
        ref = np.array([0.5, 0.8, 1.0])
        carrier = np.array([0.3, 0.5, 0.9])
        result = _pwm_compare(ref, carrier)
        np.testing.assert_array_equal(result, np.ones(3))

    def test_ref_less_than_carrier_returns_minus_one(self):
        ref = np.array([0.1, 0.2, -0.5])
        carrier = np.array([0.5, 0.9, 0.0])
        result = _pwm_compare(ref, carrier)
        np.testing.assert_array_equal(result, np.full(3, -1.0))

    def test_ref_equal_carrier_returns_one(self):
        """ref >= carrier at equality -> +1.0"""
        ref = np.array([0.5, -0.3, 0.0])
        carrier = ref.copy()
        result = _pwm_compare(ref, carrier)
        np.testing.assert_array_equal(result, np.ones(3))

    def test_mixed_comparison(self):
        ref = np.array([1.0, 0.0, -1.0])
        carrier = np.array([0.5, 0.5, 0.5])
        result = _pwm_compare(ref, carrier)
        np.testing.assert_array_equal(result, np.array([1.0, -1.0, -1.0]))

    def test_output_values_are_only_plus_minus_one(self):
        rng = np.random.default_rng(42)
        ref = rng.uniform(-1, 1, 1000)
        carrier = rng.uniform(-1, 1, 1000)
        result = _pwm_compare(ref, carrier)
        assert set(np.unique(result)).issubset({1.0, -1.0})
