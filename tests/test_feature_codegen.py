"""Tests for Feature: Embedded C Code Generator.

Atomic features covered:
  1. CCodeGenerator.generate() returns (header_str, source_str) for all 11 modulations.
  2. Header file contains mandatory structural elements:
       - include guard (#ifndef / #define / #endif)
       - Doxygen @file, @brief, @author block
       - Copyright and license block
       - MISRA C:2012 compliance section
       - PC-lint / FlexeLint compliance statement
       - Disclaimer paragraph
       - float32_t typedef + PWM_FLOAT32_DEFINED guard
       - Return-code enum with _OK, _ERR_NULL_PTR, _ERR_PARAM
       - Function declaration with correct return type and all parameters
  3. Source file contains mandatory structural elements:
       - @file Doxygen block
       - #include <math.h>
       - #include "pwm_<name>.h"
       - Static helper functions: fmaxf wrapper, fminf wrapper, clamp
       - Main API function definition matching the header declaration
       - Single-return / single-exit (MISRA Rule 15.5) pattern
       - Input validation (NULL pointer check + range check)
       - Status set to _OK on the happy path
  4. Per-modulation algorithm keywords are present in the correct source.
  5. Custom THIPWM has an extra inj_ratio parameter in both header and source.
  6. CodegenOptions fields propagate into the generated output.
  7. CCodeDialog instantiates and renders correctly (GUI smoke tests).
"""

from __future__ import annotations

import pytest

from svm_shaper.codegen import (
    CCodeGenerator,
    CodegenOptions,
    _MODULATION_NAMES,
    _MODULATION_DESCRIPTIONS,
)
from svm_shaper.modulations import ModulationMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_MODES = list(ModulationMode)


def _gen(mode: ModulationMode, **kw) -> tuple[str, str]:
    opts = CodegenOptions(**kw)
    return CCodeGenerator().generate(mode, opts)


# ===========================================================================
# Feature 1 — generate() returns non-empty (header, source) for every mode
# ===========================================================================


class TestGenerateAllModes:
    @pytest.mark.parametrize("mode", _ALL_MODES)
    def test_returns_two_strings(self, mode):
        header, source = _gen(mode)
        assert isinstance(header, str)
        assert isinstance(source, str)

    @pytest.mark.parametrize("mode", _ALL_MODES)
    def test_header_not_empty(self, mode):
        header, _ = _gen(mode)
        assert len(header) > 200

    @pytest.mark.parametrize("mode", _ALL_MODES)
    def test_source_not_empty(self, mode):
        _, source = _gen(mode)
        assert len(source) > 200


# ===========================================================================
# Feature 2 — Header structural elements
# ===========================================================================


class TestHeaderStructure:
    @pytest.fixture(params=_ALL_MODES)
    def header(self, request):
        h, _ = _gen(request.param)
        return h, request.param

    def test_include_guard_ifndef(self, header):
        h, mode = header
        mod = _MODULATION_NAMES[mode]
        assert f"#ifndef PWM_{mod}_H" in h

    def test_include_guard_define(self, header):
        h, mode = header
        mod = _MODULATION_NAMES[mode]
        assert f"#define PWM_{mod}_H" in h

    def test_include_guard_endif(self, header):
        h, mode = header
        mod = _MODULATION_NAMES[mode]
        assert f"#endif /* PWM_{mod}_H */" in h

    def test_doxygen_file_tag(self, header):
        h, _ = header
        assert "* @file" in h

    def test_doxygen_brief_tag(self, header):
        h, _ = header
        assert "* @brief" in h

    def test_doxygen_author_tag(self, header):
        h, _ = header
        assert "* @author" in h

    def test_doxygen_date_tag(self, header):
        h, _ = header
        assert "* @date" in h

    def test_copyright_present(self, header):
        h, _ = header
        assert "Copyright" in h

    def test_license_mit_present(self, header):
        h, _ = header
        assert "MIT License" in h

    def test_disclaimer_present(self, header):
        h, _ = header
        assert "Disclaimer" in h
        assert "warranty" in h.lower()

    def test_misra_compliance_section(self, header):
        h, _ = header
        assert "MISRA C:2012" in h

    def test_misra_rule_15_5_mentioned(self, header):
        h, _ = header
        assert "15.5" in h

    def test_lint_compliance_section(self, header):
        h, _ = header
        assert "PC-lint" in h or "FlexeLint" in h

    def test_stdint_include(self, header):
        h, _ = header
        assert "#include <stdint.h>" in h

    def test_stdbool_include(self, header):
        h, _ = header
        assert "#include <stdbool.h>" in h

    def test_math_h_include(self, header):
        h, _ = header
        assert "#include <math.h>" in h

    def test_float32_typedef(self, header):
        h, _ = header
        assert "typedef float float32_t" in h

    def test_float32_defined_guard(self, header):
        h, _ = header
        assert "PWM_FLOAT32_DEFINED" in h

    def test_ok_enum_member(self, header):
        h, mode = header
        mod = _MODULATION_NAMES[mode]
        assert f"PWM_{mod}_OK" in h

    def test_err_null_ptr_member(self, header):
        h, mode = header
        mod = _MODULATION_NAMES[mode]
        assert f"PWM_{mod}_ERR_NULL_PTR" in h

    def test_err_param_member(self, header):
        h, mode = header
        mod = _MODULATION_NAMES[mode]
        assert f"PWM_{mod}_ERR_PARAM" in h

    def test_function_declaration_present(self, header):
        h, mode = header
        mod = _MODULATION_NAMES[mode]
        assert f"PWM_{mod}_ComputeDuty(" in h

    def test_theta_rad_param_in_decl(self, header):
        h, _ = header
        assert "theta_rad" in h

    def test_mi_param_in_decl(self, header):
        h, _ = header
        assert "float32_t         mi," in h

    def test_p_duty_a_param_in_decl(self, header):
        h, _ = header
        assert "p_duty_a" in h

    def test_p_duty_b_param_in_decl(self, header):
        h, _ = header
        assert "p_duty_b" in h

    def test_p_duty_c_param_in_decl(self, header):
        h, _ = header
        assert "p_duty_c" in h

    def test_two_pi_constant(self, header):
        h, mode = header
        mod = _MODULATION_NAMES[mode]
        assert f"PWM_{mod}_TWO_PI" in h

    def test_two_pi_over_3_constant(self, header):
        h, mode = header
        mod = _MODULATION_NAMES[mode]
        assert f"PWM_{mod}_TWO_PI_OVER_3" in h

    def test_svm_analyst_version_in_header(self, header):
        h, _ = header
        assert "SVM Analyst" in h


# ===========================================================================
# Feature 3 — Source structural elements
# ===========================================================================


class TestSourceStructure:
    @pytest.fixture(params=_ALL_MODES)
    def source(self, request):
        _, s = _gen(request.param)
        return s, request.param

    def test_doxygen_file_tag(self, source):
        s, _ = source
        assert "* @file" in s

    def test_math_h_include(self, source):
        s, _ = source
        assert "#include <math.h>" in s

    def test_local_header_include(self, source):
        s, mode = source
        mod = _MODULATION_NAMES[mode].lower()
        assert f'#include "pwm_{mod}.h"' in s

    def test_fmaxf_wrapper_defined(self, source):
        s, mode = source
        mod = _MODULATION_NAMES[mode].lower()
        assert f"static float32_t pwm_{mod}_fmaxf" in s

    def test_fminf_wrapper_defined(self, source):
        s, mode = source
        mod = _MODULATION_NAMES[mode].lower()
        assert f"static float32_t pwm_{mod}_fminf" in s

    def test_clamp_helper_defined(self, source):
        s, mode = source
        mod = _MODULATION_NAMES[mode].lower()
        assert f"static float32_t pwm_{mod}_clamp" in s

    def test_compute_duty_defined(self, source):
        s, mode = source
        mod = _MODULATION_NAMES[mode]
        assert f"PWM_{mod}_ComputeDuty(" in s

    def test_null_pointer_check(self, source):
        s, _ = source
        assert "NULL" in s

    def test_mi_range_check(self, source):
        s, _ = source
        # mi out-of-range validation must be present
        assert "mi < 0.0f" in s or "mi > 1.0f" in s

    def test_ok_status_set(self, source):
        s, mode = source
        mod = _MODULATION_NAMES[mode]
        assert f"PWM_{mod}_OK" in s

    def test_single_return_statement(self, source):
        s, _ = source
        # MISRA 15.5: only one 'return' in the function body
        assert s.count("return status;") == 1

    def test_sinf_used(self, source):
        s, _ = source
        assert "sinf(" in s

    def test_no_dynamic_memory(self, source):
        s, _ = source
        assert "malloc" not in s
        assert "calloc" not in s
        assert "realloc" not in s
        assert "free" not in s

    def test_no_recursion_keyword(self, source):
        """There should be no recursive function calls present."""
        s, _ = source
        # A simple heuristic: no function calls itself
        assert "ComputeDuty(" not in s.split("ComputeDuty(")[1] if "ComputeDuty(" in s else True

    def test_comment_style_c(self, source):
        """All block comments must use /* */ style (no // C++-style comments)."""
        s, _ = source
        assert "//" not in s


# ===========================================================================
# Feature 4 — Per-modulation algorithm keywords in source
# ===========================================================================


class TestAlgorithmKeywords:
    def test_spwm_uses_half_mi(self):
        _, s = _gen(ModulationMode.SINUSOIDAL)
        assert "half_mi" in s

    def test_spwm_no_ucm(self):
        _, s = _gen(ModulationMode.SINUSOIDAL)
        assert "ucm" not in s

    def test_thipwm16_pregain(self):
        _, s = _gen(ModulationMode.THIPWM_1_6)
        assert "PREGAIN" in s

    def test_thipwm16_sin3(self):
        _, s = _gen(ModulationMode.THIPWM_1_6)
        assert "sin3" in s

    def test_thipwm16_inj_coeff_constant(self):
        _, s = _gen(ModulationMode.THIPWM_1_6)
        mod = _MODULATION_NAMES[ModulationMode.THIPWM_1_6]
        assert f"PWM_{mod}_INJ_COEFF" in s

    def test_thipwm14_pregain(self):
        _, s = _gen(ModulationMode.THIPWM_1_4)
        assert "PREGAIN" in s

    def test_thipwm14_inj_coeff_constant(self):
        _, s = _gen(ModulationMode.THIPWM_1_4)
        mod = _MODULATION_NAMES[ModulationMode.THIPWM_1_4]
        assert f"PWM_{mod}_INJ_COEFF" in s

    def test_svm_ucm_present(self):
        _, s = _gen(ModulationMode.SVM)
        assert "ucm" in s

    def test_svm_norm_applied(self):
        _, s = _gen(ModulationMode.SVM)
        assert "SVM_NORM" in s

    def test_dpwm120max_toffset_one_minus_tmax(self):
        _, s = _gen(ModulationMode.DPWM_120_MAX)
        assert "1.0f - tmax_s" in s

    def test_dpwm120min_toffset_minus_tmin(self):
        _, s = _gen(ModulationMode.DPWM_120_MIN)
        assert "-(tmin_s)" in s

    def test_dpwm60_1_uses_cond60(self):
        _, s = _gen(ModulationMode.DPWM_60_1)
        assert "use_max" in s or "tmax_s + tmin_s" in s

    def test_dpwm60_0_uses_cond60(self):
        _, s = _gen(ModulationMode.DPWM_60_0)
        assert "use_min" in s or "tmax_s + tmin_s" in s

    def test_dpwm60_2_phase_shift(self):
        _, s = _gen(ModulationMode.DPWM_60_2)
        mod = _MODULATION_NAMES[ModulationMode.DPWM_60_2]
        assert f"PWM_{mod}_PHASE_SHIFT" in s

    def test_dpwm30_3_sin6theta(self):
        _, s = _gen(ModulationMode.DPWM_30_3)
        assert "6.0f * theta_rad" in s

    def test_dpwm_clamp_applied(self):
        for mode in (
            ModulationMode.DPWM_120_MAX,
            ModulationMode.DPWM_120_MIN,
            ModulationMode.DPWM_60_1,
            ModulationMode.DPWM_60_0,
            ModulationMode.DPWM_60_2,
            ModulationMode.DPWM_30_3,
        ):
            _, s = _gen(mode)
            mod = _MODULATION_NAMES[mode].lower()
            assert f"pwm_{mod}_clamp(" in s


# ===========================================================================
# Feature 5 — Custom THIPWM has inj_ratio parameter
# ===========================================================================


class TestCustomThipwm:
    def test_header_has_inj_ratio_param(self):
        h, _ = _gen(ModulationMode.CUSTOM_THIPWM)
        assert "inj_ratio" in h

    def test_source_has_inj_ratio_param(self):
        _, s = _gen(ModulationMode.CUSTOM_THIPWM)
        assert "inj_ratio" in s

    def test_source_computes_inj_coeff_from_ratio(self):
        _, s = _gen(ModulationMode.CUSTOM_THIPWM)
        assert "inj_ratio * (1.0f / 6.0f)" in s

    def test_source_has_inj_ratio_range_check(self):
        _, s = _gen(ModulationMode.CUSTOM_THIPWM)
        assert "inj_ratio < 0.0f" in s or "inj_ratio > 1.0f" in s

    def test_pregain_present(self):
        _, s = _gen(ModulationMode.CUSTOM_THIPWM)
        assert "PREGAIN" in s

    def test_non_custom_modes_have_no_inj_ratio(self):
        for mode in (ModulationMode.SINUSOIDAL, ModulationMode.SVM,
                     ModulationMode.DPWM_120_MAX):
            h, s = _gen(mode)
            assert "inj_ratio" not in h
            assert "inj_ratio" not in s


# ===========================================================================
# Feature 6 — CodegenOptions propagate into output
# ===========================================================================


class TestCodegenOptions:
    def test_project_name_in_header(self):
        h, _ = _gen(ModulationMode.SVM, project_name="MotorDriveV2")
        assert "MotorDriveV2" in h

    def test_author_in_header(self):
        h, _ = _gen(ModulationMode.SVM, author="Jane Doe")
        assert "Jane Doe" in h

    def test_author_in_source(self):
        _, s = _gen(ModulationMode.SVM, author="Jane Doe")
        assert "Jane Doe" in s

    def test_organization_in_header(self):
        h, _ = _gen(ModulationMode.SVM, organization="Acme Corp")
        assert "Acme Corp" in h

    def test_version_in_header(self):
        h, _ = _gen(ModulationMode.SVM, svm_analyst_version="9.9.9")
        assert "9.9.9" in h

    def test_default_project_name_is_myproject(self):
        h, _ = _gen(ModulationMode.SVM)
        assert "MyProject" in h

    def test_default_author_is_amine(self):
        h, _ = _gen(ModulationMode.SVM)
        assert "Amine KHETTAT" in h


# ===========================================================================
# Feature 6b — Header constants are numerically correct
# ===========================================================================


class TestHeaderConstants:
    def test_two_pi_value_in_header(self):
        h, _ = _gen(ModulationMode.SINUSOIDAL)
        assert "6.28318530717958647692f" in h

    def test_two_pi_over_3_value_in_header(self):
        h, _ = _gen(ModulationMode.SINUSOIDAL)
        assert "2.09439510239319549231f" in h

    def test_svm_norm_value_in_header(self):
        h, _ = _gen(ModulationMode.SVM)
        assert "1.15470053837925152902f" in h

    def test_thipwm16_inj_coeff_value_in_header(self):
        h, _ = _gen(ModulationMode.THIPWM_1_6)
        assert "0.16666666666666666667f" in h

    def test_thipwm14_inj_coeff_value_in_header(self):
        h, _ = _gen(ModulationMode.THIPWM_1_4)
        assert "0.25f" in h

    def test_dpwm62_phase_shift_value_in_header(self):
        h, _ = _gen(ModulationMode.DPWM_60_2)
        assert "0.52359877559829887308f" in h


# ===========================================================================
# Feature 6c — Modulation names and descriptions tables
# ===========================================================================


class TestModulationTables:
    def test_all_modes_have_name_entry(self):
        for mode in _ALL_MODES:
            assert mode in _MODULATION_NAMES

    def test_all_modes_have_description_entry(self):
        for mode in _ALL_MODES:
            assert mode in _MODULATION_DESCRIPTIONS

    def test_all_names_are_nonempty_strings(self):
        for mode, name in _MODULATION_NAMES.items():
            assert isinstance(name, str) and len(name) > 0

    def test_all_descriptions_are_nonempty_strings(self):
        for mode, desc in _MODULATION_DESCRIPTIONS.items():
            assert isinstance(desc, str) and len(desc) > 0


# ===========================================================================
# Feature 7 — CCodeDialog GUI smoke tests
# ===========================================================================


class TestCCodeDialogGui:
    @pytest.fixture
    def dialog(self, qapp):
        from svm_shaper.gui import CCodeDialog

        dlg = CCodeDialog(
            modulation=ModulationMode.SVM,
            modulation_index=1.0,
            project_name="TestProject",
            author_name="Test Author",
        )
        return dlg

    def test_dialog_instantiates(self, dialog):
        assert dialog is not None

    def test_window_title_contains_modulation(self, dialog):
        assert "SVM" in dialog.windowTitle()

    def test_accessible_name_set(self, dialog):
        assert dialog.accessibleName() == "C code generator dialog"

    def test_has_two_tabs(self, dialog):
        assert dialog._tabs.count() == 2

    def test_header_tab_label(self, dialog):
        assert "Header" in dialog._tabs.tabText(0)

    def test_source_tab_label(self, dialog):
        assert "Source" in dialog._tabs.tabText(1)

    def test_header_content_non_empty(self, dialog):
        assert len(dialog._header_edit.toPlainText()) > 100

    def test_source_content_non_empty(self, dialog):
        assert len(dialog._source_edit.toPlainText()) > 100

    def test_save_header_button_present(self, dialog):
        assert dialog._save_header_btn is not None

    def test_save_source_button_present(self, dialog):
        assert dialog._save_source_btn is not None

    def test_copy_header_button_present(self, dialog):
        assert dialog._copy_header_btn is not None

    def test_copy_source_button_present(self, dialog):
        assert dialog._copy_source_btn is not None

    def test_all_modulations_dialog_smoke(self, qapp):
        from svm_shaper.gui import CCodeDialog

        for mode in _ALL_MODES:
            dlg = CCodeDialog(
                modulation=mode,
                modulation_index=1.0,
                project_name="P",
                author_name="A",
            )
            assert dlg._header_edit.toPlainText() != ""
            assert dlg._source_edit.toPlainText() != ""
