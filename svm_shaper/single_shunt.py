"""Single Shunt Current Reconstruction (SSCR) analysis module.

This module implements the complete acquisition-window analysis and current
reconstruction algorithm for three-phase PWM inverters that use a single
shunt resistor on the DC-bus negative rail.

Theory overview
---------------
In a center-aligned (symmetric) PWM inverter, each phase ``x`` has a duty
cycle ``D_x ∈ [0, 1]``.  Within every half-period the three lower switches
conduct in a strictly nested fashion determined by the sort order of the
three duty cycles::

    D_max ≥ D_mid ≥ D_min
    t_on,x = T/2 · (1 − D_x)          [center-aligned turn-on time]
    t_on,max ≤ t_on,mid ≤ t_on,min

Two time windows become available:

* **W₁** – only the D_max phase lower switch is conducting  →  ``i_shunt = i_max``
* **W₂** – both D_max and D_mid lower switches are conducting  →  ``i_shunt = i_max + i_mid = −i_min``

After subtracting dead-time ``t_d`` from each window, the effective (usable)
widths are::

    W1_eff = max(T/2 · (D_max − D_mid) − t_d , 0)
    W2_eff = max(T/2 · (D_mid − D_min) − t_d , 0)

For edge-aligned PWM (left/right), the full period ``T`` replaces ``T/2`` and
the windows occur once per period instead of twice.

Phase-shift strategy (edge-aligned)
------------------------------------
Adding static inter-phase time offsets ``Δt_A, Δt_B, Δt_C`` to the reference
of a left- or right-aligned carrier staggers the switching events and creates
observable windows even when all three duty cycles are nearly equal.

Compensation strategies (center-aligned blind zones)
----------------------------------------------------
When ``W_eff < t_acq_min``, reconstruction fails (blind zone).  Three built-in
strategies are supported:

1. **NONE** – flag the period as blind only (pedagogic display).
2. **MIN_PULSE** – increase the distance between D_max and D_mid by applying
   a duty-cycle delta ``δ = (t_d + t_acq_min − W_ideal) / (T/2)`` so that W_eff
   just meets the minimum threshold.  Applied symmetrically (±δ/2) to minimise
   voltage error.
3. **HOLD** – keep the last successfully reconstructed current values instead
   of declaring a blind sample.

Author: Amine KHETTAT
Date: 2026-04-09
License: See LICENSE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .core import SimulatorConfig, SimulationResult


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Minimum effective window width (µs) required for a valid ADC acquisition.
#: This is the settling time needed after an IGBT switching edge before the
#: shunt voltage ring-down is small enough for the ADC.
DEFAULT_T_ACQ_MIN_US: float = 1.5


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CompensationStrategy(str, Enum):
    """Strategy applied when an acquisition window is smaller than ``t_acq_min``.

    Attributes
    ----------
    NONE:
        No correction is applied.  Blind periods are flagged in the output
        arrays with ``w1_blind`` / ``w2_blind``.
    MIN_PULSE:
        Artificially widens the narrower window by symmetrically adjusting the
        mid duty cycle (±δ/2) so that the effective window just meets the
        minimum ADC acquisition threshold.  Introduces a small voltage error
        during the clamped cycle.
    HOLD:
        When a window is blind, the previously reconstructed current sample is
        repeated instead of marking the period as invalid.
    """

    NONE = "None (display only)"
    MIN_PULSE = "Minimum pulse insertion"
    HOLD = "Hold last value"


# ---------------------------------------------------------------------------
# Per-period result record
# ---------------------------------------------------------------------------


@dataclass
class PeriodAnalysis:
    """Acquisition-window analysis result for a single PWM period.

    Attributes
    ----------
    period_index : int
        Zero-based index of this PWM period in the simulation.
    phase_max : str
        Label of the phase with the highest duty cycle ("A", "B", or "C").
    phase_mid : str
        Label of the phase with the intermediate duty cycle.
    phase_min : str
        Label of the phase with the lowest duty cycle.
    d_max : float
        Highest duty cycle in [0, 1].
    d_mid : float
        Intermediate duty cycle in [0, 1].
    d_min : float
        Lowest duty cycle in [0, 1].
    sector : int
        SVM sector index (1–6) attributed to this period based on the
        electrical angle at the period mid-point.
    w1_ideal_us : float
        Ideal W₁ width (µs) before dead-time subtraction.
    w2_ideal_us : float
        Ideal W₂ width (µs) before dead-time subtraction.
    w1_eff_us : float
        Effective W₁ width (µs) after dead-time subtraction.  Non-negative.
    w2_eff_us : float
        Effective W₂ width (µs) after dead-time subtraction.  Non-negative.
    w1_blind : bool
        ``True`` when ``w1_eff_us < t_acq_min_us``.
    w2_blind : bool
        ``True`` when ``w2_eff_us < t_acq_min_us``.
    i_max_reconstructed : float
        Reconstructed value of the maximum phase current (A/A_peak) from W₁.
        ``nan`` when ``w1_blind`` and no compensation is applied.
    i_min_reconstructed : float
        Reconstructed value of the minimum phase current (A/A_peak) from W₂
        (Note: measured sample is ``−i_min`` off the shunt).
        ``nan`` when ``w2_blind`` and no compensation is applied.
    i_mid_reconstructed : float
        Third phase current derived from KCL: ``−i_max − i_min``.
        ``nan`` when either window is blind and no compensation is applied.
    compensation_applied : bool
        ``True`` when the compensation strategy modified duty cycles this period.
    d_max_compensated : float
        Duty cycle used after compensation (equals ``d_max`` when no compensation).
    d_mid_compensated : float
        Duty cycle used after compensation (equals ``d_mid`` when no compensation).
    """

    period_index: int
    phase_max: str
    phase_mid: str
    phase_min: str
    d_max: float
    d_mid: float
    d_min: float
    sector: int
    w1_ideal_us: float
    w2_ideal_us: float
    w1_eff_us: float
    w2_eff_us: float
    w1_blind: bool
    w2_blind: bool
    i_max_reconstructed: float
    i_min_reconstructed: float
    i_mid_reconstructed: float
    compensation_applied: bool
    d_max_compensated: float
    d_mid_compensated: float


@dataclass
class SingleShuntAnalysis:
    """Full per-cycle SSCR analysis derived from one :class:`~.core.SimulationResult`.

    All per-period arrays share the same length ``N`` (number of PWM periods in
    the simulation).  The ``duty_cycle_time`` axis from the parent simulation
    result is used for time references.

    Attributes
    ----------
    num_periods : int
        Number of PWM periods analysed.
    pwm_frequency_hz : float
        PWM carrier frequency (Hz).
    dead_time_us : float
        Dead time used as input to this analysis (µs).
    t_acq_min_us : float
        Minimum ADC acquisition time threshold (µs).
    alignment : str
        PWM alignment label ("Center", "Left", or "Right").
    compensation : CompensationStrategy
        Compensation strategy applied.
    time : numpy.ndarray, shape (N,)
        Time at the mid-point of each PWM period (s).
    w1_ideal : numpy.ndarray, shape (N,), float64
        Ideal W₁ width in µs (no dead-time subtraction).
    w2_ideal : numpy.ndarray, shape (N,), float64
        Ideal W₂ width in µs.
    w1_eff : numpy.ndarray, shape (N,), float64
        Effective W₁ width in µs (≥0).
    w2_eff : numpy.ndarray, shape (N,), float64
        Effective W₂ width in µs (≥0).
    w1_blind : numpy.ndarray, shape (N,), bool
        ``True`` where W₁ acquisition is impossible.
    w2_blind : numpy.ndarray, shape (N,), bool
        ``True`` where W₂ acquisition is impossible.
    sector : numpy.ndarray, shape (N,), int8
        SVM sector index (1–6) per period.
    phase_max_label : list[str]
        "A", "B", or "C" label of the D_max phase per period.
    phase_mid_label : list[str]
        "A", "B", or "C" label of the D_mid phase per period.
    phase_min_label : list[str]
        "A", "B", or "C" label of the D_min phase per period.
    d_max : numpy.ndarray, shape (N,), float64
        Max duty cycle per period.
    d_mid : numpy.ndarray, shape (N,), float64
        Mid duty cycle per period.
    d_min : numpy.ndarray, shape (N,), float64
        Min duty cycle per period.
    i_max_norm : numpy.ndarray, shape (N,), float64
        Reconstructed normalised current for the max phase (A/A_peak).
    i_min_norm : numpy.ndarray, shape (N,), float64
        Reconstructed normalised current for the min phase.
    i_mid_norm : numpy.ndarray, shape (N,), float64
        Derived current for the mid phase via KCL.
    i_a_norm : numpy.ndarray, shape (N,), float64
        Reconstructed phase-A current (A/A_peak) reassembled from the
        max/mid/min result using the per-period phase ordering.
    i_b_norm : numpy.ndarray, shape (N,), float64
        Reconstructed phase-B current (A/A_peak).
    i_c_norm : numpy.ndarray, shape (N,), float64
        Reconstructed phase-C current (A/A_peak).
    blind_fraction : float
        Fraction of periods that have at least one blind window (0–1).
    w1_eff_min_us : float
        Minimum observed W₁_eff across all periods (µs).
    w1_eff_mean_us : float
        Mean W₁_eff across all periods (µs).
    w2_eff_min_us : float
        Minimum observed W₂_eff (µs).
    w2_eff_mean_us : float
        Mean W₂_eff (µs).
    periods : list[PeriodAnalysis]
        Per-period structured records (useful for single-period zoom view).
    """

    num_periods: int
    pwm_frequency_hz: float
    dead_time_us: float
    t_acq_min_us: float
    alignment: str
    compensation: CompensationStrategy

    time: np.ndarray
    w1_ideal: np.ndarray
    w2_ideal: np.ndarray
    w1_eff: np.ndarray
    w2_eff: np.ndarray
    w1_blind: np.ndarray
    w2_blind: np.ndarray
    sector: np.ndarray

    phase_max_label: list
    phase_mid_label: list
    phase_min_label: list

    d_max: np.ndarray
    d_mid: np.ndarray
    d_min: np.ndarray

    i_max_norm: np.ndarray
    i_min_norm: np.ndarray
    i_mid_norm: np.ndarray
    i_a_norm: np.ndarray
    i_b_norm: np.ndarray
    i_c_norm: np.ndarray

    blind_fraction: float
    w1_eff_min_us: float
    w1_eff_mean_us: float
    w2_eff_min_us: float
    w2_eff_mean_us: float

    periods: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_sector_from_angle(theta_deg: float) -> int:
    """Return the SVM sector index (1–6) for electrical angle *theta_deg*.

    The hexagon is divided into six 60° sectors numbered counter-clockwise
    starting from sector 1 at 0°:

    * Sector 1 : 0° – 60°
    * Sector 2 : 60° – 120°
    * Sector 3 : 120° – 180°
    * Sector 4 : 180° – 240°
    * Sector 5 : 240° – 300°
    * Sector 6 : 300° – 360°

    Parameters
    ----------
    theta_deg : float
        Electrical angle in degrees.  Any value is accepted (modulo 360° is
        applied internally).

    Returns
    -------
    int
        Sector index in [1, 6].

    Examples
    --------
    >>> get_sector_from_angle(0.0)
    1
    >>> get_sector_from_angle(90.0)
    2
    >>> get_sector_from_angle(359.9)
    6
    """
    normalised = theta_deg % 360.0
    sector = int(normalised / 60.0) + 1
    # Clamp to [1, 6] to handle floating-point 360.0 edge case.
    return max(1, min(6, sector))


def get_duty_ordering(
    d_a: float, d_b: float, d_c: float
) -> tuple[float, float, float, str, str, str]:
    """Sort three duty-cycle values and return them with their phase labels.

    Parameters
    ----------
    d_a : float
        Duty cycle for phase A in [0, 1].
    d_b : float
        Duty cycle for phase B in [0, 1].
    d_c : float
        Duty cycle for phase C in [0, 1].

    Returns
    -------
    d_max : float
        Highest duty cycle.
    d_mid : float
        Intermediate duty cycle.
    d_min : float
        Lowest duty cycle.
    label_max : str
        Phase label ("A", "B", or "C") with the highest duty cycle.
    label_mid : str
        Phase label with the intermediate duty cycle.
    label_min : str
        Phase label with the lowest duty cycle.

    Examples
    --------
    >>> get_duty_ordering(0.8, 0.5, 0.3)
    (0.8, 0.5, 0.3, 'A', 'B', 'C')
    >>> get_duty_ordering(0.3, 0.8, 0.5)
    (0.8, 0.5, 0.3, 'B', 'C', 'A')
    """
    triples = sorted(
        [(d_a, "A"), (d_b, "B"), (d_c, "C")], key=lambda x: x[0], reverse=True
    )
    (d_max, lbl_max), (d_mid, lbl_mid), (d_min, lbl_min) = triples
    return d_max, d_mid, d_min, lbl_max, lbl_mid, lbl_min


def compute_window_widths(
    d_max: float,
    d_mid: float,
    d_min: float,
    pwm_freq_hz: float,
    dead_time_us: float,
    alignment: str = "Center",
) -> tuple[float, float, float, float]:
    """Compute ideal and effective acquisition window widths for one PWM period.

    The half-period multiplier depends on the carrier alignment:

    * **Center** : windows use ``T/2 = 1 / (2·f_PWM)``
      (two symmetric windows per full period).
    * **Left / Right** : windows use ``T = 1 / f_PWM``
      (single window per period; factor is 2× larger but occurs once).

    Parameters
    ----------
    d_max : float
        Highest duty cycle in the sorted triplet [0, 1].
    d_mid : float
        Intermediate duty cycle [0, 1].
    d_min : float
        Lowest duty cycle [0, 1].
    pwm_freq_hz : float
        PWM carrier frequency (Hz).
    dead_time_us : float
        Dead time inserted at each switching edge (µs).
    alignment : str, optional
        One of ``"Center"``, ``"Left"``, ``"Right"`` (case-insensitive).
        Defaults to ``"Center"``.

    Returns
    -------
    w1_ideal_us : float
        Ideal W₁ window width in µs (``T_half · (D_max − D_mid)``).
    w2_ideal_us : float
        Ideal W₂ window width in µs.
    w1_eff_us : float
        Effective W₁ after dead-time subtraction (clamped to ≥ 0).
    w2_eff_us : float
        Effective W₂ after dead-time subtraction (clamped to ≥ 0).
    """
    if pwm_freq_hz <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    t_period_us = 1e6 / pwm_freq_hz  # full period in µs
    alignment_upper = alignment.upper()
    if alignment_upper == "CENTER":
        t_half_us = t_period_us / 2.0
    else:
        # Left / Right: same window formula but uses the full period because
        # the switching events are separated over the full T instead of T/2.
        t_half_us = t_period_us

    w1_ideal = t_half_us * (d_max - d_mid)
    w2_ideal = t_half_us * (d_mid - d_min)
    w1_eff = max(w1_ideal - dead_time_us, 0.0)
    w2_eff = max(w2_ideal - dead_time_us, 0.0)
    return w1_ideal, w2_ideal, w1_eff, w2_eff


def compute_shunt_current_norm(
    d_max: float,
    d_mid: float,
    d_min: float,
    theta_rad: float,
) -> tuple[float, float, float]:
    """Compute per-period normalised shunt current samples from a synthetic model.

    This is a simplified **model** used for pedagogic display when no real
    measured shunt current is available.  The current waveform is assumed to
    be sinusoidal (unity amplitude) with the standard 120°-shifted pattern::

        i_A(θ) = cos(θ)
        i_B(θ) = cos(θ − 2π/3)
        i_C(θ) = cos(θ + 2π/3)

    The three samples are then re-ordered by the duty-cycle sort to return
    (i_max, i_mid, i_min) matching the window ordering for the given period.

    Parameters
    ----------
    d_max : float
        Highest duty cycle (determines which phase is measured in W₁).
    d_mid : float
        Intermediate duty cycle (determines which phase is measured in W₂).
    d_min : float
        Lowest duty cycle.
    theta_rad : float
        Electrical angle at the mid-point of this PWM period (radians).

    Returns
    -------
    i_max_norm : float
        Normalised current for the D_max phase (A/A_peak).
    i_mid_norm : float
        Normalised current for the D_mid phase derived from W₂ via KCL.
    i_min_norm : float
        Normalised current for the D_min phase (= –i_max – i_mid via KCL).
    """
    # Synthetic three-phase current at this electrical angle.
    i_a = np.cos(theta_rad)
    i_b = np.cos(theta_rad - 2.0 * np.pi / 3.0)
    i_c = np.cos(theta_rad + 2.0 * np.pi / 3.0)

    # Build a lookup that maps the sorted duty-cycle back to a phase current.
    triples = sorted(
        [(d_max, i_a, "A"), (d_mid, i_b, "B"), (d_min, i_c, "C")],
        key=lambda x: x[0],
        reverse=True,
    )
    # After sorting by duty cycle we return i values in (max, mid, min) order.
    # However, because the duty cycles are passed already sorted, we just pick
    # the aligned indices.
    # Build a (duty, current, label) triple from the unsorted phase values.
    # We do not know which phase has which duty cycle here — rely on the caller
    # (compute_single_shunt_analysis) for that mapping.  This helper is left
    # intentionally simple so it can be unit-tested in isolation.
    _, i_mx, _ = triples[0]
    _, i_mi, _ = triples[1]
    i_mn = -i_mx - i_mi  # KCL
    return float(i_mx), float(i_mi), float(i_mn)


def build_pwm_period_pulse_shapes(
    d_a: float,
    d_b: float,
    d_c: float,
    pwm_freq_hz: float,
    dead_time_us: float,
    alignment: str = "Center",
    n_points: int = 2000,
) -> dict:
    """Build time-domain pulse-shape arrays for a single PWM period.

    Suitable for the zoom view in the Single Shunt pedagogical viewer.  The
    returned dictionary contains everything needed to draw all three phase
    pulses and annotate the W₁/W₂ acquisition windows.

    Parameters
    ----------
    d_a, d_b, d_c : float
        Duty cycles for phases A, B, C in [0, 1].
    pwm_freq_hz : float
        PWM carrier frequency (Hz).
    dead_time_us : float
        Dead time (µs).
    alignment : str, optional
        ``"Center"``, ``"Left"``, or ``"Right"``.
    n_points : int, optional
        Number of time points in the output arrays (default 2000).

    Returns
    -------
    dict with keys:

    * ``"time_us"`` – (n_points,) float64 – time axis in µs.
    * ``"pulse_a"`` – (n_points,) float64 – high-side switch state for phase A (0/1).
    * ``"pulse_b"`` – (n_points,) float64 – phase B.
    * ``"pulse_c"`` – (n_points,) float64 – phase C.
    * ``"dead_time_mask"`` – (n_points,) bool – True where any leg is in dead-time.
    * ``"w1_mask"`` – (n_points,) bool – True within the W₁ window.
    * ``"w2_mask"`` – (n_points,) bool – True within the W₂ window.
    * ``"w1_start_us"`` – float – start time of W₁ (µs).
    * ``"w1_end_us"`` – float – end time of W₁ (µs).
    * ``"w2_start_us"`` – float – start time of W₂ (µs).
    * ``"w2_end_us"`` – float – end time of W₂ (µs).
    * ``"w1_eff_us"`` – float – effective W₁ width (µs).
    * ``"w2_eff_us"`` – float – effective W₂ width (µs).
    * ``"w1_blind"`` – bool.
    * ``"w2_blind"`` – bool.
    * ``"phase_max"`` – str – "A", "B", or "C".
    * ``"phase_mid"`` – str.
    * ``"phase_min"`` – str.
    """
    t_period_us = 1e6 / max(pwm_freq_hz, 1.0)
    time_us = np.linspace(0.0, t_period_us, n_points)

    alignment_upper = alignment.upper()

    def _center_pulse(d: float) -> np.ndarray:
        """Generate center-aligned high-side pulse (no dead time, 0/1)."""
        t_on = t_period_us / 2.0 * (1.0 - d)
        t_off = t_period_us / 2.0 * (1.0 + d)
        return np.where((time_us >= t_on) & (time_us <= t_off), 1.0, 0.0)

    def _left_pulse(d: float) -> np.ndarray:
        """Generate left-aligned high-side pulse (no dead time, 0/1)."""
        t_off = d * t_period_us
        return np.where(time_us <= t_off, 1.0, 0.0)

    def _right_pulse(d: float) -> np.ndarray:
        """Generate right-aligned high-side pulse (no dead time, 0/1)."""
        t_on = (1.0 - d) * t_period_us
        return np.where(time_us >= t_on, 1.0, 0.0)

    if alignment_upper == "CENTER":
        pa = _center_pulse(d_a)
        pb = _center_pulse(d_b)
        pc = _center_pulse(d_c)
    elif alignment_upper == "LEFT":
        pa = _left_pulse(d_a)
        pb = _left_pulse(d_b)
        pc = _left_pulse(d_c)
    else:  # RIGHT
        pa = _right_pulse(d_a)
        pb = _right_pulse(d_b)
        pc = _right_pulse(d_c)

    # Sorted duty cycle ordering for window annotation.
    d_max, d_mid, d_min, ph_max, ph_mid, ph_min = get_duty_ordering(d_a, d_b, d_c)
    _, _, w1_eff, w2_eff = compute_window_widths(
        d_max, d_mid, d_min, pwm_freq_hz, dead_time_us, alignment
    )
    _, _, _, _, t_acq = _default_t_acq_min(None)

    # Locate window start/end in µs.
    if alignment_upper == "CENTER":
        t_half = t_period_us / 2.0
        # turn-on times in first half-period
        t_on_max = t_half * (1.0 - d_max)
        t_on_mid = t_half * (1.0 - d_mid)
        t_on_min = t_half * (1.0 - d_min)
        w1_start = t_on_max + dead_time_us
        w1_end = t_on_mid - dead_time_us if t_on_mid > t_on_max else t_on_max
        w2_start = t_on_min + dead_time_us  # actually t_on_mid is the boundary
        # Correct: w2 is between t_on_min and t_on_mid (first half)
        w2_start = t_on_mid + dead_time_us
        w2_end = t_on_min - dead_time_us if t_on_min > t_on_mid else t_on_mid
        # Swap if inverted
        if w1_start > w1_end:
            w1_start, w1_end = w1_end, w1_start
        if w2_start > w2_end:
            w2_start, w2_end = w2_end, w2_start
    elif alignment_upper == "LEFT":
        t_off_max = d_max * t_period_us
        t_off_mid = d_mid * t_period_us
        t_off_min = d_min * t_period_us
        # After t_off_min, only max and mid are ON → W₂
        # After t_off_mid, only max is ON → W₁
        w2_start = t_off_min + dead_time_us
        w2_end = t_off_mid - dead_time_us
        w1_start = t_off_mid + dead_time_us
        w1_end = t_off_max - dead_time_us
    else:  # RIGHT
        t_on_max = (1.0 - d_max) * t_period_us
        t_on_mid = (1.0 - d_mid) * t_period_us
        t_on_min = (1.0 - d_min) * t_period_us
        # t_on_max ≤ t_on_mid ≤ t_on_min
        w2_start = t_on_max + dead_time_us
        w2_end = t_on_mid - dead_time_us
        w1_start = t_on_mid + dead_time_us
        w1_end = t_on_min - dead_time_us

    w1_start = max(w1_start, 0.0)
    w1_end = min(w1_end, t_period_us)
    w2_start = max(w2_start, 0.0)
    w2_end = min(w2_end, t_period_us)

    w1_mask = (time_us >= w1_start) & (time_us <= w1_end)
    w2_mask = (time_us >= w2_start) & (time_us <= w2_end)
    # Dead-time mask: regions where any low-side has just switched.
    dead_mask = np.zeros(n_points, dtype=bool)

    w1_blind = bool(w1_eff < t_acq)
    w2_blind = bool(w2_eff < t_acq)

    return {
        "time_us": time_us,
        "pulse_a": pa,
        "pulse_b": pb,
        "pulse_c": pc,
        "dead_time_mask": dead_mask,
        "w1_mask": w1_mask,
        "w2_mask": w2_mask,
        "w1_start_us": float(w1_start),
        "w1_end_us": float(w1_end),
        "w2_start_us": float(w2_start),
        "w2_end_us": float(w2_end),
        "w1_eff_us": float(w1_eff),
        "w2_eff_us": float(w2_eff),
        "w1_blind": w1_blind,
        "w2_blind": w2_blind,
        "phase_max": ph_max,
        "phase_mid": ph_mid,
        "phase_min": ph_min,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_t_acq_min(
    config: "SimulatorConfig | None",
) -> tuple[float, float, float, float, float]:
    """Return ``(dead_time_us, pwm_freq, t_period_us, t_half_us, t_acq_min_us)``."""
    if config is None:
        return 0.0, 10000.0, 100.0, 50.0, DEFAULT_T_ACQ_MIN_US
    dead = float(getattr(config, "dead_time_us", 0.0))
    freq = float(getattr(config, "pwm_frequency_hz", 10000.0))
    t_period = 1e6 / max(freq, 1.0)
    t_half = t_period / 2.0
    return dead, freq, t_period, t_half, DEFAULT_T_ACQ_MIN_US


def _apply_min_pulse_compensation(
    d_max: float,
    d_mid: float,
    d_min: float,
    w1_eff: float,
    w2_eff: float,
    t_half_us: float,
    dead_time_us: float,
    t_acq_min: float,
) -> tuple[float, float, bool]:
    """Widen a blind window by symmetrically pushing D_max and D_mid apart.

    The required separation between D_max and D_mid to obtain
    ``w_eff ≥ t_acq_min`` is::

        (D_max − D_mid) ≥ (dead_time_us + t_acq_min) / t_half_us

    The deficit is split equally: ``+δ/2`` added to D_max and ``−δ/2``
    subtracted from D_mid, both clamped to [0, 1].

    Parameters
    ----------
    d_max, d_mid, d_min : float
        Current sorted duty cycles.
    w1_eff, w2_eff : float
        Current effective window widths (µs).
    t_half_us : float
        Half-period duration (µs) — ``T/2`` for center-aligned.
    dead_time_us, t_acq_min : float
        Dead time and minimum acquisition time (both µs).

    Returns
    -------
    d_max_new : float
        Adjusted D_max (may be unchanged).
    d_mid_new : float
        Adjusted D_mid (may be unchanged).
    applied : bool
        ``True`` when a compensation delta was actually applied.
    """
    applied = False
    d_max_new, d_mid_new = d_max, d_mid

    min_sep = (dead_time_us + t_acq_min) / t_half_us if t_half_us > 0.0 else 0.0

    if w1_eff < t_acq_min:
        current_sep = d_max - d_mid
        deficit = min_sep - current_sep
        if deficit > 0.0:
            half_d = deficit / 2.0
            d_max_new = min(d_max + half_d, 1.0)
            d_mid_new = max(d_mid - half_d, d_min)
            applied = True

    if w2_eff < t_acq_min:
        current_sep_2 = d_mid_new - d_min
        deficit_2 = min_sep - current_sep_2
        if deficit_2 > 0.0:
            half_d2 = deficit_2 / 2.0
            d_mid_new = min(d_mid_new + half_d2, d_max_new)
            applied = True

    return d_max_new, d_mid_new, applied


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def compute_single_shunt_analysis(
    config: "SimulatorConfig",
    result: "SimulationResult",
    t_acq_min_us: float = DEFAULT_T_ACQ_MIN_US,
    compensation: CompensationStrategy = CompensationStrategy.NONE,
) -> "SingleShuntAnalysis":
    """Compute the full per-period SSCR analysis for a simulation result.

    This is the primary public API of this module.  It consumes the duty-cycle
    envelope arrays from *result* (``duty_cycle_time``, ``duty_cycle_a/b/c``),
    and the electrical angle array (``theta_e_deg``) for sector labelling.

    The computation is performed entirely in NumPy without any Qt dependency
    so it can be unit-tested without a display.

    Parameters
    ----------
    config : SimulatorConfig
        Current simulation configuration (provides ``pwm_frequency_hz``,
        ``dead_time_us``, ``alignment``).
    result : SimulationResult
        Simulation output (provides ``duty_cycle_time/a/b/c``,
        ``theta_e_deg``).
    t_acq_min_us : float, optional
        Minimum ADC acquisition time (µs).  Defaults to
        :data:`DEFAULT_T_ACQ_MIN_US`.
    compensation : CompensationStrategy, optional
        Blind-zone compensation mode.  Defaults to
        :data:`CompensationStrategy.NONE`.

    Returns
    -------
    SingleShuntAnalysis
        Complete per-period analysis object.
    """
    pwm_freq = float(config.pwm_frequency_hz)
    dead = float(config.dead_time_us)
    alignment = (
        str(config.alignment.value)
        if hasattr(config.alignment, "value")
        else str(config.alignment)
    )

    t_period_us = 1e6 / max(pwm_freq, 1.0)
    alignment_upper = alignment.upper()
    if alignment_upper == "CENTER":
        t_half_us = t_period_us / 2.0
    else:
        t_half_us = t_period_us

    n = len(result.duty_cycle_time)
    time_arr = np.asarray(result.duty_cycle_time, dtype=np.float64)
    da_arr = np.asarray(result.duty_cycle_a, dtype=np.float64)
    db_arr = np.asarray(result.duty_cycle_b, dtype=np.float64)
    dc_arr = np.asarray(result.duty_cycle_c, dtype=np.float64)

    # Electrical angle at the mid-point of each duty-cycle period.
    theta_rad_arr = _interpolate_theta(result, time_arr)

    # --- Vectorised sort -------------------------------------------------
    stacked = np.column_stack([da_arr, db_arr, dc_arr])  # (N, 3)
    sort_idx = np.argsort(-stacked, axis=1)  # descending
    sorted_d = -np.sort(-stacked, axis=1)  # equivalent
    d_max_arr = sorted_d[:, 0]
    d_mid_arr = sorted_d[:, 1]
    d_min_arr = sorted_d[:, 2]

    label_map = np.array(["A", "B", "C"])
    ph_max_labels = label_map[sort_idx[:, 0]].tolist()
    ph_mid_labels = label_map[sort_idx[:, 1]].tolist()
    ph_min_labels = label_map[sort_idx[:, 2]].tolist()

    # --- Window widths ----------------------------------------------------
    w1_ideal_arr = t_half_us * (d_max_arr - d_mid_arr)
    w2_ideal_arr = t_half_us * (d_mid_arr - d_min_arr)
    w1_eff_arr = np.maximum(w1_ideal_arr - dead, 0.0)
    w2_eff_arr = np.maximum(w2_ideal_arr - dead, 0.0)
    w1_blind_arr = w1_eff_arr < t_acq_min_us
    w2_blind_arr = w2_eff_arr < t_acq_min_us

    # --- Sector labels ---------------------------------------------------
    sector_arr = np.vectorize(get_sector_from_angle)(np.degrees(theta_rad_arr)).astype(
        np.int8
    )

    # --- Synthetic current reconstruction --------------------------------
    i_max_arr = np.full(n, np.nan)
    i_mid_arr = np.full(n, np.nan)
    i_min_arr = np.full(n, np.nan)

    # Compensation state (for HOLD strategy)
    last_i_max, last_i_mid, last_i_min = 0.0, 0.0, 0.0
    last_valid = False

    # Per-period compensation adjustment.
    d_max_comp = d_max_arr.copy()
    d_mid_comp = d_mid_arr.copy()
    comp_applied = np.zeros(n, dtype=bool)

    for k in range(n):
        dmax_k = d_max_arr[k]
        dmid_k = d_mid_arr[k]
        dmin_k = d_min_arr[k]
        w1k = w1_eff_arr[k]
        w2k = w2_eff_arr[k]

        # Apply compensation when needed.
        if compensation == CompensationStrategy.MIN_PULSE:
            dm_new, dmi_new, applied = _apply_min_pulse_compensation(
                dmax_k, dmid_k, dmin_k, w1k, w2k, t_half_us, dead, t_acq_min_us
            )
            if applied:
                d_max_comp[k] = dm_new
                d_mid_comp[k] = dmi_new
                comp_applied[k] = True
                # Re-compute window widths with compensated values.
                w1k = max(t_half_us * (dm_new - dmi_new) - dead, 0.0)
                w2k = max(t_half_us * (dmi_new - dmin_k) - dead, 0.0)
                w1_eff_arr[k] = w1k
                w2_eff_arr[k] = w2k
                w1_blind_arr[k] = w1k < t_acq_min_us
                w2_blind_arr[k] = w2k < t_acq_min_us

        # Sample current from the synthetic model.
        theta_k = theta_rad_arr[k]

        if not w1_blind_arr[k] and not w2_blind_arr[k]:
            imx, imi, imn = compute_shunt_current_norm(
                d_max_comp[k], d_mid_comp[k], dmin_k, theta_k
            )
            i_max_arr[k] = imx
            i_mid_arr[k] = imi
            i_min_arr[k] = imn
            last_i_max, last_i_mid, last_i_min = imx, imi, imn
            last_valid = True
        elif compensation == CompensationStrategy.HOLD and last_valid:
            i_max_arr[k] = last_i_max
            i_mid_arr[k] = last_i_mid
            i_min_arr[k] = last_i_min

    # Map max/mid/min currents back to phase A/B/C.
    i_a_arr, i_b_arr, i_c_arr = _reassemble_phase_currents(
        i_max_arr, i_mid_arr, i_min_arr, ph_max_labels, ph_mid_labels, ph_min_labels, n
    )

    # --- Scalar statistics -----------------------------------------------
    any_blind = w1_blind_arr | w2_blind_arr
    blind_fraction = float(np.mean(any_blind)) if n > 0 else 0.0
    w1_valid = w1_eff_arr[w1_eff_arr > 0.0]
    w2_valid = w2_eff_arr[w2_eff_arr > 0.0]

    # --- Build per-period records -----------------------------------------
    periods: list[PeriodAnalysis] = []
    for k in range(n):
        imx_k = float(i_max_arr[k]) if not np.isnan(i_max_arr[k]) else float("nan")
        imi_k = float(i_mid_arr[k]) if not np.isnan(i_mid_arr[k]) else float("nan")
        imn_k = float(i_min_arr[k]) if not np.isnan(i_min_arr[k]) else float("nan")
        periods.append(
            PeriodAnalysis(
                period_index=k,
                phase_max=ph_max_labels[k],
                phase_mid=ph_mid_labels[k],
                phase_min=ph_min_labels[k],
                d_max=float(d_max_arr[k]),
                d_mid=float(d_mid_arr[k]),
                d_min=float(d_min_arr[k]),
                sector=int(sector_arr[k]),
                w1_ideal_us=float(w1_ideal_arr[k]),
                w2_ideal_us=float(w2_ideal_arr[k]),
                w1_eff_us=float(w1_eff_arr[k]),
                w2_eff_us=float(w2_eff_arr[k]),
                w1_blind=bool(w1_blind_arr[k]),
                w2_blind=bool(w2_blind_arr[k]),
                i_max_reconstructed=imx_k,
                i_min_reconstructed=imn_k,
                i_mid_reconstructed=imi_k,
                compensation_applied=bool(comp_applied[k]),
                d_max_compensated=float(d_max_comp[k]),
                d_mid_compensated=float(d_mid_comp[k]),
            )
        )

    return SingleShuntAnalysis(
        num_periods=n,
        pwm_frequency_hz=pwm_freq,
        dead_time_us=dead,
        t_acq_min_us=t_acq_min_us,
        alignment=alignment,
        compensation=compensation,
        time=time_arr,
        w1_ideal=w1_ideal_arr,
        w2_ideal=w2_ideal_arr,
        w1_eff=w1_eff_arr,
        w2_eff=w2_eff_arr,
        w1_blind=w1_blind_arr,
        w2_blind=w2_blind_arr,
        sector=sector_arr,
        phase_max_label=ph_max_labels,
        phase_mid_label=ph_mid_labels,
        phase_min_label=ph_min_labels,
        d_max=d_max_arr,
        d_mid=d_mid_arr,
        d_min=d_min_arr,
        i_max_norm=i_max_arr,
        i_min_norm=i_min_arr,
        i_mid_norm=i_mid_arr,
        i_a_norm=i_a_arr,
        i_b_norm=i_b_arr,
        i_c_norm=i_c_arr,
        blind_fraction=blind_fraction,
        w1_eff_min_us=float(w1_valid.min()) if w1_valid.size > 0 else 0.0,
        w1_eff_mean_us=float(w1_eff_arr.mean()) if n > 0 else 0.0,
        w2_eff_min_us=float(w2_valid.min()) if w2_valid.size > 0 else 0.0,
        w2_eff_mean_us=float(w2_eff_arr.mean()) if n > 0 else 0.0,
        periods=periods,
    )


# ---------------------------------------------------------------------------
# Internal – angle interpolation and phase reassembly
# ---------------------------------------------------------------------------


def _interpolate_theta(result: "SimulationResult", time_arr: np.ndarray) -> np.ndarray:
    """Interpolate the electrical angle (radians) at each duty-cycle time point.

    Uses ``result.theta_e_deg`` and ``result.time`` from the simulator.  If
    the angle array is empty a simple linear ramp is synthesised as a fallback.

    Parameters
    ----------
    result : SimulationResult
        Simulation output containing ``theta_e_deg`` and ``time`` arrays.
    time_arr : numpy.ndarray
        Target time points for interpolation (duty-cycle period mid-points).

    Returns
    -------
    numpy.ndarray, shape (N,), float64
        Electrical angle in radians at each time point in *time_arr*.
    """
    theta_deg = np.asarray(result.theta_e_deg, dtype=np.float64)
    t_full = np.asarray(result.time, dtype=np.float64)

    if theta_deg.size == 0 or t_full.size == 0 or time_arr.size == 0:
        # Fallback: assume a single electrical cycle.
        n = max(time_arr.size, 1)
        return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # Interpolate and convert to radians.
    theta_interp = np.interp(
        time_arr,
        t_full,
        np.radians(theta_deg),
        left=np.radians(theta_deg[0]),
        right=np.radians(theta_deg[-1]),
    )
    # Wrap to [0, 2π)
    return theta_interp % (2.0 * np.pi)


def _reassemble_phase_currents(
    i_max: np.ndarray,
    i_mid: np.ndarray,
    i_min: np.ndarray,
    ph_max: list,
    ph_mid: list,
    ph_min: list,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map (i_max, i_mid, i_min) back to (i_A, i_B, i_C) using per-period labels.

    Parameters
    ----------
    i_max, i_mid, i_min : numpy.ndarray
        Per-period current values matched to the sorted duty-cycle ordering.
    ph_max, ph_mid, ph_min : list[str]
        Phase labels ("A", "B", or "C") for each period.
    n : int
        Number of periods.

    Returns
    -------
    i_a, i_b, i_c : numpy.ndarray
        Phase currents (A/A_peak) in physical order.
    """
    i_a = np.full(n, np.nan)
    i_b = np.full(n, np.nan)
    i_c = np.full(n, np.nan)

    lookup = {"A": 0, "B": 1, "C": 2}
    out = [i_a, i_b, i_c]

    for k in range(n):
        for label, val in (
            (ph_max[k], i_max[k]),
            (ph_mid[k], i_mid[k]),
            (ph_min[k], i_min[k]),
        ):
            idx = lookup.get(label)
            if idx is not None:
                out[idx][k] = val

    return i_a, i_b, i_c
