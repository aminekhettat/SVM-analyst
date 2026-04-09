"""Switching loss and junction-temperature estimation for 3-phase VSI MOSFETs.

Loss model
----------
Turn-on / turn-off energy is linearly scaled from datasheet reference conditions::

    E_on(V, I)  = E_on_ref  × (V_dc / V_ref) × (I_peak / I_ref)
    E_off(V, I) = E_off_ref × (V_dc / V_ref) × (I_peak / I_ref)

Conduction loss per switch (one MOSFET in a half-bridge leg)::

    P_cond = I_rms² × R_ds(on) / 2

The factor ½ accounts for each device conducting approximately one half of the
fundamental cycle.

Switching loss per switch, averaged over one sinusoidal half-cycle (SPWM)::

    P_sw = (E_on + E_off) × f_pwm × (V_dc / V_ref) × (I_peak / I_ref) / π

The 1/π factor comes from integrating the linear current dependency
sin(θ) over [0, π]:  (1/π) ∫₀^π sin(θ) dθ = 2/π, then halved again because
only the high-side (or low-side) switch fires per half-cycle; the net
average factor is 1/π.

For DPWM strategies (60 ° sector clamping = 120 ° total per fundamental
cycle), switching events are reduced by one-third::

    P_sw_dpwm = P_sw_spwm × (2/3)

Steady-state junction temperature (single device on a heatsink)::

    T_j = T_amb + P_total × R_th,ja

where R_th,ja is the junction-to-ambient thermal resistance.

MOSFET presets
--------------
Datasheet parameters are provided for the following validated devices:

Infineon Technologies
  * IPP60R099P7  — CoolMOS™ P7,  600 V / 20 A
  * IPP65R110CFD — CoolMOS™ CFD2, 650 V / 16 A
  * IPP023N10N5  — OptiMOS™ 5,   100 V / 100 A

Nexperia
  * PSMN1R5-40YLD — N-ch, 40 V / 100 A
  * BUK7M24-100EX — N-ch, 100 V / 30 A

All energy values correspond to typical conditions stated in the respective
manufacturer datasheets; exact test conditions are documented in the inline
comments below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LossParameters:
    """MOSFET datasheet parameters used for switching-loss estimation.

    Attributes
    ----------
    device_name : str
        Part number / device identifier.
    manufacturer : str
        Manufacturer name.
    vds_max_v : float
        Maximum drain-source voltage [V].
    id_max_a : float
        Continuous drain current rating [A].
    rds_on_mohm : float
        On-state drain-source resistance at 25 °C [mΩ].
    rds_on_temp_coeff : float
        Ratio R_ds(on) at 125 °C / R_ds(on) at 25 °C.
        Informational; typical range 1.5–2.1 for silicon MOSFETs.
    e_on_uj : float
        Turn-on switching energy at reference test conditions [μJ].
    e_off_uj : float
        Turn-off switching energy at reference test conditions [μJ].
    v_ref_v : float
        DC-bus voltage used in the datasheet switching-energy measurement [V].
    i_ref_a : float
        Current used in the datasheet switching-energy measurement [A].
    rth_ja_k_w : float
        Junction-to-ambient thermal resistance [°C/W].
    tj_max_c : float
        Maximum allowable junction temperature [°C].
    """

    device_name: str
    manufacturer: str
    vds_max_v: float
    id_max_a: float
    rds_on_mohm: float
    rds_on_temp_coeff: float
    e_on_uj: float
    e_off_uj: float
    v_ref_v: float
    i_ref_a: float
    rth_ja_k_w: float
    tj_max_c: float


@dataclass
class LossThermalResult:
    """Computed loss and thermal estimates for a 3-phase half-bridge inverter.

    All *per-switch* values refer to a single MOSFET (one of the six devices).
    All *inverter* values represent the complete 3-phase 6-switch bridge (×6).

    Attributes
    ----------
    p_cond_w : float
        Conduction loss per switch [W].
    p_sw_spwm_w : float
        Switching loss per switch under SPWM [W].
    p_sw_dpwm_w : float
        Switching loss per switch under DPWM (≈ 2/3 × SPWM) [W].
    p_total_spwm_w : float
        Total loss per switch under SPWM (conduction + switching) [W].
    p_total_dpwm_w : float
        Total loss per switch under DPWM [W].
    p_inv_spwm_w : float
        Total inverter loss under SPWM (6 × p_total_spwm_w) [W].
    p_inv_dpwm_w : float
        Total inverter loss under DPWM (6 × p_total_dpwm_w) [W].
    t_junction_spwm_c : float
        Estimated steady-state junction temperature under SPWM [°C].
    t_junction_dpwm_c : float
        Estimated steady-state junction temperature under DPWM [°C].
    t_amb_c : float
        Ambient temperature used in the computation [°C].
    tj_max_c : float
        Maximum junction temperature from the datasheet [°C].
    """

    p_cond_w: float
    p_sw_spwm_w: float
    p_sw_dpwm_w: float
    p_total_spwm_w: float
    p_total_dpwm_w: float
    p_inv_spwm_w: float
    p_inv_dpwm_w: float
    t_junction_spwm_c: float
    t_junction_dpwm_c: float
    t_amb_c: float
    tj_max_c: float


# ---------------------------------------------------------------------------
# Device presets (validated against manufacturer datasheets)
# ---------------------------------------------------------------------------

#: Mapping from display label to :class:`LossParameters`.
#: The entry ``"Custom device"`` serves as an editable placeholder.
MOSFET_PRESETS: dict[str, LossParameters] = {
    # ── Custom / user-defined ──────────────────────────────────────────────
    "Custom device": LossParameters(
        device_name="Custom device",
        manufacturer="—",
        vds_max_v=600.0,
        id_max_a=20.0,
        rds_on_mohm=99.0,
        rds_on_temp_coeff=1.70,
        e_on_uj=186.0,
        e_off_uj=81.0,
        v_ref_v=400.0,
        i_ref_a=10.0,
        rth_ja_k_w=40.0,
        tj_max_c=150.0,
    ),
    # ── Infineon CoolMOS™ P7 — 600 V / 20 A ──────────────────────────────
    # Source: DS_IPP60R099P7 Rev 2.3 (Infineon Technologies AG)
    # E_on / E_off at: V_ds = 400 V, I_d = 10 A, V_gs = 0/13 V,
    #                  R_g = 10 Ω, T_j = 25 °C
    "Infineon IPP60R099P7 — CoolMOS P7 (600 V / 20 A)": LossParameters(
        device_name="IPP60R099P7",
        manufacturer="Infineon",
        vds_max_v=600.0,
        id_max_a=20.0,
        rds_on_mohm=99.0,
        rds_on_temp_coeff=1.76,  # 174 mΩ @ 150 °C / 99 mΩ @ 25 °C
        e_on_uj=186.0,
        e_off_uj=81.0,
        v_ref_v=400.0,
        i_ref_a=10.0,
        rth_ja_k_w=40.0,
        tj_max_c=150.0,
    ),
    # ── Infineon CoolMOS™ CFD2 — 650 V / 16 A ────────────────────────────
    # Source: DS_IPP65R110CFD Rev 2.0 (Infineon Technologies AG)
    # E_on / E_off at: V_ds = 400 V, I_d = 8 A, R_g = 2 Ω, T_j = 25 °C
    "Infineon IPP65R110CFD — CoolMOS CFD2 (650 V / 16 A)": LossParameters(
        device_name="IPP65R110CFD",
        manufacturer="Infineon",
        vds_max_v=650.0,
        id_max_a=16.0,
        rds_on_mohm=110.0,
        rds_on_temp_coeff=1.86,  # 205 mΩ @ 150 °C / 110 mΩ @ 25 °C
        e_on_uj=165.0,
        e_off_uj=110.0,
        v_ref_v=400.0,
        i_ref_a=8.0,
        rth_ja_k_w=40.0,
        tj_max_c=150.0,
    ),
    # ── Infineon OptiMOS™ 5 — 100 V / 100 A ──────────────────────────────
    # Source: DS_IPP023N10N5 Rev 2.1 (Infineon Technologies AG)
    # E_on / E_off at: V_ds = 50 V, I_d = 20 A, R_g = 3.3 Ω, T_j = 25 °C
    "Infineon IPP023N10N5 — OptiMOS 5 (100 V / 100 A)": LossParameters(
        device_name="IPP023N10N5",
        manufacturer="Infineon",
        vds_max_v=100.0,
        id_max_a=100.0,
        rds_on_mohm=2.3,
        rds_on_temp_coeff=2.10,  # Ultra-low Rds(on) → higher thermal ratio
        e_on_uj=32.0,
        e_off_uj=20.0,
        v_ref_v=50.0,
        i_ref_a=20.0,
        rth_ja_k_w=40.0,
        tj_max_c=175.0,
    ),
    # ── Nexperia — 40 V / 100 A ───────────────────────────────────────────
    # Source: DS PSMN1R5-40YLD Rev 3 (Nexperia B.V.)
    # E_on / E_off at: V_ds = 20 V, I_d = 25 A, R_g = 2.2 Ω, T_j = 25 °C
    "Nexperia PSMN1R5-40YLD (40 V / 100 A)": LossParameters(
        device_name="PSMN1R5-40YLD",
        manufacturer="Nexperia",
        vds_max_v=40.0,
        id_max_a=100.0,
        rds_on_mohm=1.5,
        rds_on_temp_coeff=2.00,
        e_on_uj=12.0,
        e_off_uj=6.0,
        v_ref_v=20.0,
        i_ref_a=25.0,
        rth_ja_k_w=50.0,
        tj_max_c=175.0,
    ),
    # ── Nexperia — 100 V / 30 A ───────────────────────────────────────────
    # Source: DS BUK7M24-100EX Rev 3 (Nexperia B.V.)
    # E_on / E_off at: V_ds = 50 V, I_d = 15 A, R_g = 5.0 Ω, T_j = 25 °C
    "Nexperia BUK7M24-100EX (100 V / 30 A)": LossParameters(
        device_name="BUK7M24-100EX",
        manufacturer="Nexperia",
        vds_max_v=100.0,
        id_max_a=30.0,
        rds_on_mohm=24.0,
        rds_on_temp_coeff=1.80,
        e_on_uj=75.0,
        e_off_uj=45.0,
        v_ref_v=50.0,
        i_ref_a=15.0,
        rth_ja_k_w=50.0,
        tj_max_c=175.0,
    ),
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_switch_losses(
    params: LossParameters,
    v_dc: float,
    i_phase_pk: float,
    i_phase_rms: float,
    f_pwm: float,
    t_amb_c: float,
) -> LossThermalResult:
    """Estimate per-switch and total inverter losses plus junction temperature.

    Both SPWM and DPWM results are returned in a single call so that the
    caller can compare strategies without re-invoking the function.

    Parameters
    ----------
    params:
        MOSFET datasheet parameters (see :class:`LossParameters`).
    v_dc:
        DC bus voltage [V].  Must be > 0.
    i_phase_pk:
        Fundamental peak phase current [A].  Must be >= 0.
    i_phase_rms:
        RMS phase current [A].  Must be >= 0.
    f_pwm:
        PWM carrier / switching frequency [Hz].  Must be > 0.
    t_amb_c:
        Ambient temperature [°C].

    Returns
    -------
    LossThermalResult

    Raises
    ------
    ValueError
        If *v_dc* or *f_pwm* are not positive, or if either current is
        negative.
    """
    if v_dc <= 0.0:
        raise ValueError(f"v_dc must be positive, got {v_dc}")
    if f_pwm <= 0.0:
        raise ValueError(f"f_pwm must be positive, got {f_pwm}")
    if i_phase_pk < 0.0 or i_phase_rms < 0.0:
        raise ValueError("Phase currents must be non-negative")

    # ------------------------------------------------------------------
    # Conduction loss per switch
    # ------------------------------------------------------------------
    # Simple half-bridge approximation: each of the 6 devices conducts
    # approximately half the fundamental cycle.
    #   P_cond ≈ I_rms² × R_ds(on) / 2
    rds_ohm = params.rds_on_mohm * 1e-3
    p_cond = (i_phase_rms**2) * rds_ohm / 2.0

    # ------------------------------------------------------------------
    # Switching energy scaling
    # ------------------------------------------------------------------
    # Guard against zero reference values (custom entry may have them 0).
    if params.v_ref_v > 0.0 and params.i_ref_a > 0.0:
        scale = (v_dc / params.v_ref_v) * (i_phase_pk / params.i_ref_a)
    else:
        scale = 0.0

    e_sw_total_j = (params.e_on_uj + params.e_off_uj) * 1e-6 * scale

    # ------------------------------------------------------------------
    # Switching loss per switch
    # ------------------------------------------------------------------
    # Average over a sinusoidal half-cycle → factor 1/π
    p_sw_spwm = e_sw_total_j * f_pwm / math.pi

    # DPWM: 120° of clamping per fundamental cycle (1/3 of cycle)
    # → switching events reduced to 2/3 of SPWM
    p_sw_dpwm = p_sw_spwm * (2.0 / 3.0)

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------
    p_total_spwm = p_cond + p_sw_spwm
    p_total_dpwm = p_cond + p_sw_dpwm

    p_inv_spwm = p_total_spwm * 6.0  # 6-switch 3-phase bridge
    p_inv_dpwm = p_total_dpwm * 6.0

    # ------------------------------------------------------------------
    # Junction temperature (steady-state, single-device model)
    # ------------------------------------------------------------------
    t_j_spwm = t_amb_c + p_total_spwm * params.rth_ja_k_w
    t_j_dpwm = t_amb_c + p_total_dpwm * params.rth_ja_k_w

    return LossThermalResult(
        p_cond_w=p_cond,
        p_sw_spwm_w=p_sw_spwm,
        p_sw_dpwm_w=p_sw_dpwm,
        p_total_spwm_w=p_total_spwm,
        p_total_dpwm_w=p_total_dpwm,
        p_inv_spwm_w=p_inv_spwm,
        p_inv_dpwm_w=p_inv_dpwm,
        t_junction_spwm_c=t_j_spwm,
        t_junction_dpwm_c=t_j_dpwm,
        t_amb_c=t_amb_c,
        tj_max_c=params.tj_max_c,
    )
