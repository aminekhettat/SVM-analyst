"""Tests for Feature #9 – Switching Loss & Junction Temperature Estimation.

Atomic features covered
-----------------------
1. ``LossParameters`` dataclass — construction and field access.
2. ``LossThermalResult`` dataclass — construction and field access.
3. ``MOSFET_PRESETS`` — presence of all expected device keys; values are
   correctly typed ``LossParameters`` instances with physically sensible data.
4. ``compute_switch_losses`` — correctness of the loss model:
   * Conduction loss formula: P_cond = I_rms² × Rds(on) / 2
   * Switching loss SPWM formula: P_sw = (Eon + Eoff) × f_pwm × scale / π
   * DPWM is exactly 2/3 of SPWM switching loss
   * Junction temperature formula: Tj = Tamb + P_total × Rth_ja
   * Returns both SPWM and DPWM results in one call
   * Inverter total = 6 × per-switch
   * Zero-current edge case — zero switching loss, conduction loss
   * Scaling monotonicity — higher Vdc → higher switching loss
   * ValueError for invalid inputs (v_dc ≤ 0, f_pwm ≤ 0, negative currents)
   * Zero reference values produce zero switching loss without error
5. ``LossThermalDialog`` GUI — instantiation, preset handling, field
   validation, accessible names, and Calculate flow.
"""

from __future__ import annotations

import math

import pytest

from svm_shaper.losses import (
    MOSFET_PRESETS,
    LossParameters,
    LossThermalResult,
    compute_switch_losses,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EXPECTED_PRESET_KEYS = [
    "Custom device",
    "Infineon IPP60R099P7 — CoolMOS P7 (600 V / 20 A)",
    "Infineon IPP65R110CFD — CoolMOS CFD2 (650 V / 16 A)",
    "Infineon IPP023N10N5 — OptiMOS 5 (100 V / 100 A)",
    "Nexperia PSMN1R5-40YLD (40 V / 100 A)",
    "Nexperia BUK7M24-100EX (100 V / 30 A)",
]


def _make_params(
    rds_mohm: float = 99.0,
    eon_uj: float = 186.0,
    eoff_uj: float = 81.0,
    v_ref: float = 400.0,
    i_ref: float = 10.0,
    rth_ja: float = 40.0,
    tj_max: float = 150.0,
) -> LossParameters:
    return LossParameters(
        device_name="TestDevice",
        manufacturer="TestMfg",
        vds_max_v=600.0,
        id_max_a=20.0,
        rds_on_mohm=rds_mohm,
        rds_on_temp_coeff=1.75,
        e_on_uj=eon_uj,
        e_off_uj=eoff_uj,
        v_ref_v=v_ref,
        i_ref_a=i_ref,
        rth_ja_k_w=rth_ja,
        tj_max_c=tj_max,
    )


# ===========================================================================
# 1. LossParameters dataclass
# ===========================================================================


class TestLossParametersDataclass:
    def test_construction_and_field_access(self):
        p = _make_params(rds_mohm=99.0, eon_uj=186.0, eoff_uj=81.0)
        assert p.device_name == "TestDevice"
        assert p.manufacturer == "TestMfg"
        assert p.vds_max_v == 600.0
        assert p.id_max_a == 20.0
        assert p.rds_on_mohm == 99.0
        assert p.rds_on_temp_coeff == 1.75
        assert p.e_on_uj == 186.0
        assert p.e_off_uj == 81.0
        assert p.v_ref_v == 400.0
        assert p.i_ref_a == 10.0
        assert p.rth_ja_k_w == 40.0
        assert p.tj_max_c == 150.0

    def test_is_mutable_dataclass(self):
        p = _make_params()
        p.rds_on_mohm = 50.0
        assert p.rds_on_mohm == 50.0

    def test_different_instances_are_independent(self):
        p1 = _make_params(rds_mohm=10.0)
        p2 = _make_params(rds_mohm=20.0)
        assert p1.rds_on_mohm != p2.rds_on_mohm


# ===========================================================================
# 2. LossThermalResult dataclass
# ===========================================================================


class TestLossThermalResultDataclass:
    def test_construction_and_all_fields(self):
        r = LossThermalResult(
            p_cond_w=0.1,
            p_sw_spwm_w=0.5,
            p_sw_dpwm_w=0.333,
            p_total_spwm_w=0.6,
            p_total_dpwm_w=0.433,
            p_inv_spwm_w=3.6,
            p_inv_dpwm_w=2.6,
            t_junction_spwm_c=49.0,
            t_junction_dpwm_c=42.3,
            t_amb_c=25.0,
            tj_max_c=150.0,
        )
        assert r.p_cond_w == 0.1
        assert r.p_sw_spwm_w == 0.5
        assert r.p_sw_dpwm_w == 0.333
        assert r.p_total_spwm_w == 0.6
        assert r.p_total_dpwm_w == 0.433
        assert r.p_inv_spwm_w == 3.6
        assert r.t_amb_c == 25.0  # noqa: PLR2004
        assert r.tj_max_c == 150.0  # noqa: PLR2004


# ===========================================================================
# 3. MOSFET_PRESETS dictionary
# ===========================================================================


class TestMosfetPresets:
    def test_all_expected_keys_present(self):
        for key in _EXPECTED_PRESET_KEYS:
            assert key in MOSFET_PRESETS, f"Missing preset: {key}"

    def test_all_values_are_loss_parameters(self):
        for key, val in MOSFET_PRESETS.items():
            assert isinstance(val, LossParameters), (
                f"Preset '{key}' is not a LossParameters instance"
            )

    def test_infineon_ipp60r099p7_datasheet_values(self):
        p = MOSFET_PRESETS["Infineon IPP60R099P7 — CoolMOS P7 (600 V / 20 A)"]
        assert p.manufacturer == "Infineon"
        assert p.device_name == "IPP60R099P7"
        assert p.vds_max_v == 600.0
        assert p.rds_on_mohm == 99.0
        assert p.e_on_uj == 186.0
        assert p.e_off_uj == 81.0
        assert p.v_ref_v == 400.0
        assert p.i_ref_a == 10.0
        assert p.tj_max_c == 150.0

    def test_infineon_ipp65r110cfd_datasheet_values(self):
        p = MOSFET_PRESETS["Infineon IPP65R110CFD — CoolMOS CFD2 (650 V / 16 A)"]
        assert p.vds_max_v == 650.0
        assert p.rds_on_mohm == 110.0
        assert p.e_on_uj == 165.0
        assert p.e_off_uj == 110.0

    def test_infineon_ipp023n10n5_datasheet_values(self):
        p = MOSFET_PRESETS["Infineon IPP023N10N5 — OptiMOS 5 (100 V / 100 A)"]
        assert p.vds_max_v == 100.0
        assert p.rds_on_mohm == 2.3
        assert p.v_ref_v == 50.0
        assert p.tj_max_c == 175.0

    def test_nexperia_psmn1r5_datasheet_values(self):
        p = MOSFET_PRESETS["Nexperia PSMN1R5-40YLD (40 V / 100 A)"]
        assert p.manufacturer == "Nexperia"
        assert p.vds_max_v == 40.0
        assert p.rds_on_mohm == 1.5
        assert p.v_ref_v == 20.0

    def test_nexperia_buk7m24_datasheet_values(self):
        p = MOSFET_PRESETS["Nexperia BUK7M24-100EX (100 V / 30 A)"]
        assert p.vds_max_v == 100.0
        assert p.rds_on_mohm == 24.0
        assert p.e_on_uj == 75.0
        assert p.e_off_uj == 45.0

    def test_all_presets_have_positive_vds_max(self):
        for key, p in MOSFET_PRESETS.items():
            assert p.vds_max_v > 0, f"{key}: vds_max_v must be positive"

    def test_all_presets_have_positive_rds_on(self):
        for key, p in MOSFET_PRESETS.items():
            assert p.rds_on_mohm > 0, f"{key}: rds_on_mohm must be positive"

    def test_all_presets_have_positive_tj_max(self):
        for key, p in MOSFET_PRESETS.items():
            assert p.tj_max_c > 0, f"{key}: tj_max_c must be positive"

    def test_all_presets_have_positive_rth_ja(self):
        for key, p in MOSFET_PRESETS.items():
            assert p.rth_ja_k_w > 0, f"{key}: rth_ja_k_w must be positive"

    def test_custom_device_exists_as_first_entry(self):
        first_key = next(iter(MOSFET_PRESETS))
        assert first_key == "Custom device"


# ===========================================================================
# 4. compute_switch_losses
# ===========================================================================


class TestComputeSwitchLossesReturnType:
    """Return type and shape."""

    def test_returns_loss_thermal_result(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert isinstance(res, LossThermalResult)

    def test_all_fields_are_floats(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        for field in res.__dataclass_fields__:
            assert isinstance(getattr(res, field), float), (
                f"Field '{field}' is not a float"
            )


class TestConductionLoss:
    """P_cond = I_rms² × Rds_on / 2."""

    def test_conduction_loss_formula(self):
        rds_mohm = 100.0
        i_rms = 10.0
        p = _make_params(rds_mohm=rds_mohm)
        res = compute_switch_losses(p, 400.0, 14.14, i_rms, 10_000.0, 25.0)
        expected = (i_rms ** 2) * (rds_mohm * 1e-3) / 2.0
        assert abs(res.p_cond_w - expected) < 1e-9

    def test_conduction_loss_scales_with_rds(self):
        p_low = _make_params(rds_mohm=10.0)
        p_high = _make_params(rds_mohm=100.0)
        i_rms = 10.0
        res_low = compute_switch_losses(p_low, 400.0, 14.14, i_rms, 10_000.0, 25.0)
        res_high = compute_switch_losses(p_high, 400.0, 14.14, i_rms, 10_000.0, 25.0)
        assert res_high.p_cond_w > res_low.p_cond_w

    def test_zero_irms_gives_zero_conduction(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 0.0, 0.0, 10_000.0, 25.0)
        assert res.p_cond_w == 0.0


class TestSwitchingLoss:
    """P_sw_spwm = (Eon + Eoff) × f_pwm × (V/V_ref) × (I/I_ref) / π."""

    def test_switching_loss_spwm_formula(self):
        eon, eoff = 186.0, 81.0
        v_ref, i_ref = 400.0, 10.0
        v_dc, i_pk = 400.0, 10.0
        f_pwm = 10_000.0
        p = _make_params(eon_uj=eon, eoff_uj=eoff, v_ref=v_ref, i_ref=i_ref)
        res = compute_switch_losses(p, v_dc, i_pk, 7.07, f_pwm, 25.0)
        scale = (v_dc / v_ref) * (i_pk / i_ref)
        e_sw_j = (eon + eoff) * 1e-6 * scale
        expected_spwm = e_sw_j * f_pwm / math.pi
        assert abs(res.p_sw_spwm_w - expected_spwm) < 1e-9

    def test_dpwm_is_two_thirds_of_spwm(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert abs(res.p_sw_dpwm_w - res.p_sw_spwm_w * 2.0 / 3.0) < 1e-9

    def test_zero_ipk_gives_zero_switching(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 0.0, 0.0, 10_000.0, 25.0)
        assert res.p_sw_spwm_w == 0.0
        assert res.p_sw_dpwm_w == 0.0

    def test_switching_loss_scales_with_vdc(self):
        p = _make_params()
        res_low = compute_switch_losses(p, 200.0, 10.0, 7.07, 10_000.0, 25.0)
        res_high = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert res_high.p_sw_spwm_w > res_low.p_sw_spwm_w

    def test_switching_loss_scales_with_ipk(self):
        p = _make_params()
        res_low = compute_switch_losses(p, 400.0, 5.0, 3.54, 10_000.0, 25.0)
        res_high = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert res_high.p_sw_spwm_w > res_low.p_sw_spwm_w

    def test_switching_loss_scales_with_fpwm(self):
        p = _make_params()
        res_low = compute_switch_losses(p, 400.0, 10.0, 7.07, 5_000.0, 25.0)
        res_high = compute_switch_losses(p, 400.0, 10.0, 7.07, 20_000.0, 25.0)
        assert res_high.p_sw_spwm_w > res_low.p_sw_spwm_w

    def test_dpwm_always_lower_than_spwm(self):
        for f in [5_000.0, 10_000.0, 20_000.0]:
            p = _make_params()
            res = compute_switch_losses(p, 400.0, 10.0, 7.07, f, 25.0)
            assert res.p_sw_dpwm_w <= res.p_sw_spwm_w


class TestTotalsAndInverter:
    """p_total = p_cond + p_sw; p_inv = 6 × p_total."""

    def test_spwm_total_equals_cond_plus_sw(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert abs(res.p_total_spwm_w - (res.p_cond_w + res.p_sw_spwm_w)) < 1e-9

    def test_dpwm_total_equals_cond_plus_sw(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert abs(res.p_total_dpwm_w - (res.p_cond_w + res.p_sw_dpwm_w)) < 1e-9

    def test_inverter_spwm_equals_six_times_per_switch(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert abs(res.p_inv_spwm_w - 6.0 * res.p_total_spwm_w) < 1e-9

    def test_inverter_dpwm_equals_six_times_per_switch(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert abs(res.p_inv_dpwm_w - 6.0 * res.p_total_dpwm_w) < 1e-9

    def test_spwm_inverter_higher_than_dpwm_inverter(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert res.p_inv_spwm_w > res.p_inv_dpwm_w


class TestJunctionTemperature:
    """T_j = T_amb + P_total × Rth_ja."""

    def test_spwm_tj_formula(self):
        p = _make_params(rth_ja=40.0, tj_max=150.0)
        t_amb = 25.0
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, t_amb)
        expected = t_amb + res.p_total_spwm_w * 40.0
        assert abs(res.t_junction_spwm_c - expected) < 1e-9

    def test_dpwm_tj_formula(self):
        p = _make_params(rth_ja=40.0)
        t_amb = 25.0
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, t_amb)
        expected = t_amb + res.p_total_dpwm_w * 40.0
        assert abs(res.t_junction_dpwm_c - expected) < 1e-9

    def test_tamb_stored_in_result(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 40.0)
        assert res.t_amb_c == 40.0

    def test_tjmax_propagated_to_result(self):
        p = _make_params(tj_max=175.0)
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert res.tj_max_c == 175.0

    def test_dpwm_tj_lower_than_spwm_tj(self):
        p = _make_params()
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert res.t_junction_dpwm_c <= res.t_junction_spwm_c

    def test_higher_tamb_increases_tj(self):
        p = _make_params()
        res_cold = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        res_hot = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 85.0)
        assert res_hot.t_junction_spwm_c > res_cold.t_junction_spwm_c


class TestZeroReferenceValues:
    """When v_ref_v or i_ref_a is 0, switching loss must be zero (no division)."""

    def test_zero_vref_gives_zero_switching(self):
        p = _make_params(v_ref=0.0)
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert res.p_sw_spwm_w == 0.0

    def test_zero_iref_gives_zero_switching(self):
        p = _make_params(i_ref=0.0)
        res = compute_switch_losses(p, 400.0, 10.0, 7.07, 10_000.0, 25.0)
        assert res.p_sw_spwm_w == 0.0


class TestInputValidation:
    """ValueError is raised for invalid inputs."""

    def test_nonpositive_vdc_raises(self):
        p = _make_params()
        with pytest.raises(ValueError, match="v_dc"):
            compute_switch_losses(p, 0.0, 10.0, 7.07, 10_000.0, 25.0)

    def test_negative_vdc_raises(self):
        p = _make_params()
        with pytest.raises(ValueError):
            compute_switch_losses(p, -100.0, 10.0, 7.07, 10_000.0, 25.0)

    def test_nonpositive_fpwm_raises(self):
        p = _make_params()
        with pytest.raises(ValueError, match="f_pwm"):
            compute_switch_losses(p, 400.0, 10.0, 7.07, 0.0, 25.0)

    def test_negative_fpwm_raises(self):
        p = _make_params()
        with pytest.raises(ValueError):
            compute_switch_losses(p, 400.0, 10.0, 7.07, -1.0, 25.0)

    def test_negative_ipk_raises(self):
        p = _make_params()
        with pytest.raises(ValueError):
            compute_switch_losses(p, 400.0, -1.0, 7.07, 10_000.0, 25.0)

    def test_negative_irms_raises(self):
        p = _make_params()
        with pytest.raises(ValueError):
            compute_switch_losses(p, 400.0, 10.0, -1.0, 10_000.0, 25.0)


class TestAllPresetsSmokeTest:
    """Each datasheet preset can be used in compute_switch_losses without error."""

    @pytest.mark.parametrize("key", _EXPECTED_PRESET_KEYS)
    def test_preset_runs_without_error(self, key: str):
        p = MOSFET_PRESETS[key]
        # Use operating conditions well within the device ratings.
        v_dc = p.v_ref_v or 50.0
        i_pk = min(p.i_ref_a, p.id_max_a) if p.i_ref_a > 0 else 10.0
        i_rms = i_pk / math.sqrt(2.0)
        res = compute_switch_losses(p, v_dc, i_pk, i_rms, 10_000.0, 25.0)
        assert res.p_total_spwm_w >= 0.0
        assert res.p_total_dpwm_w >= 0.0
        assert res.t_junction_spwm_c >= 25.0


# ===========================================================================
# 5. LossThermalDialog GUI tests
# ===========================================================================


@pytest.fixture(scope="module")
def loss_dialog(qapp):  # noqa: ANN001
    """Instantiate LossThermalDialog once for all GUI tests in this module."""
    from unittest.mock import MagicMock

    from svm_shaper.gui import LossThermalDialog

    config = MagicMock()
    config.battery_voltage = 400.0
    config.pwm_frequency_hz = 10_000.0
    config.modulation = "SVM"

    dlg = LossThermalDialog(config)
    dlg.show()
    return dlg


class TestLossThermalDialogInstantiation:
    def test_dialog_is_visible(self, loss_dialog):
        assert loss_dialog.isVisible()

    def test_window_title_contains_svm_analyst(self, loss_dialog):
        assert "SVM Analyst" in loss_dialog.windowTitle()

    def test_accessible_name_set(self, loss_dialog):
        assert loss_dialog.accessibleName() == "Loss and thermal estimator dialog"

    def test_accessible_description_set(self, loss_dialog):
        assert "3-phase" in loss_dialog.accessibleDescription()


class TestLossThermalDialogPresetCombo:
    def test_combo_has_correct_number_of_items(self, loss_dialog):
        assert loss_dialog._combo_preset.count() == len(MOSFET_PRESETS)

    def test_combo_contains_all_preset_keys(self, loss_dialog):
        items = [
            loss_dialog._combo_preset.itemText(i)
            for i in range(loss_dialog._combo_preset.count())
        ]
        for key in _EXPECTED_PRESET_KEYS:
            assert key in items

    def test_combo_accessible_name(self, loss_dialog):
        assert "preset" in loss_dialog._combo_preset.accessibleName().lower()

    def test_selecting_infineon_preset_fills_rds_field(self, loss_dialog, qtbot):
        key = "Infineon IPP60R099P7 — CoolMOS P7 (600 V / 20 A)"
        loss_dialog._combo_preset.setCurrentText(key)
        qtbot.wait(50)
        val = float(loss_dialog._ed_rds.text())
        assert abs(val - 99.0) < 0.01

    def test_selecting_nexperia_preset_fills_eon_field(self, loss_dialog, qtbot):
        key = "Nexperia PSMN1R5-40YLD (40 V / 100 A)"
        loss_dialog._combo_preset.setCurrentText(key)
        qtbot.wait(50)
        val = float(loss_dialog._ed_eon.text())
        assert abs(val - 12.0) < 0.01

    def test_device_info_label_updates_on_preset_change(self, loss_dialog, qtbot):
        key = "Infineon IPP60R099P7 — CoolMOS P7 (600 V / 20 A)"
        loss_dialog._combo_preset.setCurrentText(key)
        qtbot.wait(50)
        assert "600" in loss_dialog._lbl_device_info.text()


class TestLossThermalDialogFieldValidators:
    def test_rds_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_rds.validator() is not None

    def test_eon_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_eon.validator() is not None

    def test_eoff_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_eoff.validator() is not None

    def test_vref_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_vref.validator() is not None

    def test_iref_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_iref.validator() is not None

    def test_rthja_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_rthja.validator() is not None

    def test_tjmax_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_tjmax.validator() is not None

    def test_vdc_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_vdc.validator() is not None

    def test_ipk_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_ipk.validator() is not None

    def test_irms_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_irms.validator() is not None

    def test_fpwm_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_fpwm.validator() is not None

    def test_tamb_field_has_validator(self, loss_dialog):
        assert loss_dialog._ed_tamb.validator() is not None


class TestLossThermalDialogAccessibleNames:
    def test_rds_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_rds.accessibleName() != ""

    def test_eon_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_eon.accessibleName() != ""

    def test_eoff_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_eoff.accessibleName() != ""

    def test_vref_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_vref.accessibleName() != ""

    def test_iref_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_iref.accessibleName() != ""

    def test_rthja_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_rthja.accessibleName() != ""

    def test_tjmax_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_tjmax.accessibleName() != ""

    def test_vdc_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_vdc.accessibleName() != ""

    def test_ipk_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_ipk.accessibleName() != ""

    def test_irms_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_irms.accessibleName() != ""

    def test_fpwm_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_fpwm.accessibleName() != ""

    def test_tamb_accessible_name(self, loss_dialog):
        assert loss_dialog._ed_tamb.accessibleName() != ""

    def test_modulation_combo_accessible_name(self, loss_dialog):
        assert loss_dialog._combo_mod.accessibleName() != ""

    def test_results_text_accessible_name(self, loss_dialog):
        assert loss_dialog._results_text.accessibleName() != ""

    def test_canvas_accessible_name(self, loss_dialog):
        assert loss_dialog._canvas.accessibleName() != ""

    def test_calc_button_accessible_name(self, loss_dialog):
        assert loss_dialog._btn_calc.accessibleName() != ""


class TestLossThermalDialogConfigPrefill:
    def test_vdc_prefilled_from_config(self, loss_dialog):
        assert loss_dialog._ed_vdc.text() == "400.0"

    def test_fpwm_prefilled_from_config(self, loss_dialog):
        assert loss_dialog._ed_fpwm.text() == "10000"

    def test_tamb_defaults_to_25(self, loss_dialog):
        assert loss_dialog._ed_tamb.text() == "25.0"


class TestLossThermalDialogCalculate:
    def test_calculate_updates_results_text(self, loss_dialog, qtbot):
        # Select a known preset and fill currents.
        loss_dialog._combo_preset.setCurrentText(
            "Infineon IPP60R099P7 — CoolMOS P7 (600 V / 20 A)"
        )
        loss_dialog._ed_ipk.setText("10.0")
        loss_dialog._ed_irms.setText("7.07")
        from PySide6.QtCore import Qt
        qtbot.mouseClick(loss_dialog._btn_calc, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        text = loss_dialog._results_text.toPlainText()
        assert "IPP60R099P7" in text or "Custom" in text or len(text) > 20

    def test_calculate_stores_result(self, loss_dialog, qtbot):
        loss_dialog._combo_preset.setCurrentText(
            "Infineon IPP60R099P7 — CoolMOS P7 (600 V / 20 A)"
        )
        loss_dialog._ed_ipk.setText("10.0")
        loss_dialog._ed_irms.setText("7.07")
        from PySide6.QtCore import Qt
        qtbot.mouseClick(loss_dialog._btn_calc, Qt.MouseButton.LeftButton)
        qtbot.wait(100)
        assert loss_dialog._result is not None
        assert isinstance(loss_dialog._result, LossThermalResult)

    def test_calculate_with_missing_field_shows_warning(
        self, loss_dialog, qtbot, qapp
    ):
        # Clear a required field so Calculate shows a warning.
        loss_dialog._ed_ipk.clear()
        loss_dialog._ed_irms.clear()
        with qtbot.waitSignal(qapp.focusChanged, timeout=2000, raising=False):
            pass
        # Trigger calculate; a QMessageBox.warning should appear.
        # We patch QMessageBox.warning to avoid blocking the test runner.
        from unittest.mock import patch

        with patch("svm_shaper.gui.QtWidgets.QMessageBox.warning") as mock_warn:
            loss_dialog._calculate()
            assert mock_warn.called

    def test_read_float_returns_none_for_blank(self, loss_dialog):
        from PySide6.QtWidgets import QLineEdit

        ed = QLineEdit()
        ed.setText("")
        assert loss_dialog._read_float(ed) is None

    def test_read_float_accepts_comma_decimal(self, loss_dialog):
        from PySide6.QtWidgets import QLineEdit

        ed = QLineEdit()
        ed.setText("3,14")
        assert abs(loss_dialog._read_float(ed) - 3.14) < 1e-9

    def test_modulation_combo_has_two_items(self, loss_dialog):
        assert loss_dialog._combo_mod.count() == 2
