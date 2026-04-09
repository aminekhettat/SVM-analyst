"""C code generator for PWM modulation algorithms.

Generates MISRA C:2012-compliant, Doxygen-documented, lint-clean C source and
header files for direct integration into embedded applications.

The generated files contain:
- A public header (.h) with project metadata, MISRA/lint compliance statement,
  license block, type definitions, constants, and function declarations.
- A C implementation file (.c) with the full Doxygen-annotated algorithm.

All generated code:
- Targets C99 / MISRA C:2012 (mandatory + required rules).
- Uses only ``<math.h>`` (sinf, fmaxf, fminf), ``<stdint.h>``, and ``<stdbool.h>``.
- Avoids dynamic memory, recursion, VLAs, and all reserved/prohibited MISRA
  interfaces.
- Is annotated for PC-lint Plus / FlexeLint with zero expected warnings.

Author: Amine KHETTAT
Date: 2026-03-09
License: See LICENSE
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Tuple

from .modulations import ModulationMode

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

_MODULATION_NAMES: dict[ModulationMode, str] = {
    ModulationMode.SINUSOIDAL: "SPWM",
    ModulationMode.THIPWM_1_6: "THIPWM16",
    ModulationMode.THIPWM_1_4: "THIPWM14",
    ModulationMode.CUSTOM_THIPWM: "THIPWM_CUSTOM",
    ModulationMode.SVM: "SVM",
    ModulationMode.DPWM_120_MAX: "DPWM120_MAX",
    ModulationMode.DPWM_120_MIN: "DPWM120_MIN",
    ModulationMode.DPWM_60_1: "DPWM60_1",
    ModulationMode.DPWM_60_0: "DPWM60_0",
    ModulationMode.DPWM_60_2: "DPWM60_2",
    ModulationMode.DPWM_30_3: "DPWM30_3",
}

_MODULATION_DESCRIPTIONS: dict[ModulationMode, str] = {
    ModulationMode.SINUSOIDAL: (
        "Sinusoidal PWM (SPWM) — pure sine-wave carrier comparison.\n"
        " * Three-phase sinusoidal references with no common-mode injection.\n"
        " * Simple implementation; DC-bus utilisation limited to mi * Vdc/2."
    ),
    ModulationMode.THIPWM_1_6: (
        "Third Harmonic Injection PWM at 1/6 ratio (THIPWM 1/6).\n"
        " * A 1/6 third-harmonic common-mode signal is added to the three-phase\n"
        " * sine references, extending the linear modulation range by ~15.5%.\n"
        " * The 1.15 pre-gain ensures references remain within the carrier bounds."
    ),
    ModulationMode.THIPWM_1_4: (
        "Third Harmonic Injection PWM at 1/4 ratio (THIPWM 1/4).\n"
        " * Similar to THIPWM 1/6 but uses a larger 1/4 injection coefficient.\n"
        " * Improves harmonic symmetry at the cost of slightly higher harmonic\n"
        " * distortion at low modulation indices."
    ),
    ModulationMode.CUSTOM_THIPWM: (
        "Custom Third Harmonic Injection PWM.\n"
        " * The injection ratio is supplied at runtime via the inj_ratio parameter.\n"
        " * inj_ratio = 0.0 yields pure sinusoidal; inj_ratio = 1.0 is THIPWM 1/6."
    ),
    ModulationMode.SVM: (
        "Space Vector Modulation (SVM).\n"
        " * The common-mode offset is computed from the instantaneous max/min of\n"
        " * the three-phase references, then subtracted.  This is equivalent to\n"
        " * the standard SVM algorithm and minimises switching events per cycle.\n"
        " * References peak at sqrt(3)/2; a 1.1547 (2/sqrt(3)) factor normalises\n"
        " * them to the unity linear range."
    ),
    ModulationMode.DPWM_120_MAX: (
        "Discontinuous PWM 120 degrees — high-side clamping (DPWM_120_MAX).\n"
        " * One phase leg is clamped to 100% duty for 120 electrical degrees per\n"
        " * cycle, eliminating switching losses in that leg during clamping.\n"
        " * The clamped leg is always the one with the highest duty cycle."
    ),
    ModulationMode.DPWM_120_MIN: (
        "Discontinuous PWM 120 degrees — low-side clamping (DPWM_120_MIN).\n"
        " * One phase leg is clamped to 0% duty for 120 electrical degrees per\n"
        " * cycle, eliminating low-side switching losses in that leg."
    ),
    ModulationMode.DPWM_60_1: (
        "Discontinuous PWM 60 degrees — variant 1 (DPWM1).\n"
        " * Alternates between high-side and low-side clamping every 60 degrees\n"
        " * based on whether the sum of the highest and lowest switch times\n"
        " * exceeds the carrier period."
    ),
    ModulationMode.DPWM_60_0: (
        "Discontinuous PWM 60 degrees — variant 0 (DPWM0).\n"
        " * Similar to DPWM1 but with inverted clamping logic, redistributing\n"
        " * switching losses differently across the three phases."
    ),
    ModulationMode.DPWM_60_2: (
        "Discontinuous PWM 60 degrees — variant 2 (DPWM2), 30-degree shifted.\n"
        " * A 30-degree phase-shifted version of DPWM1.  The clamping decision\n"
        " * is based on sinf(theta - PI/6) for each electrical cycle."
    ),
    ModulationMode.DPWM_30_3: (
        "Discontinuous PWM 30 degrees — variant 3 (DPWM3).\n"
        " * Alternates between DPWM1 and DPWM0 clamping strategies every 30\n"
        " * electrical degrees, further distributing switching losses."
    ),
}


@dataclass
class CodegenOptions:
    """User-configurable metadata embedded in the generated file headers.

    Attributes
    ----------
    project_name:
        Name of the target embedded project (used in Doxygen @file block).
    author:
        Author / owner name written to the @author tag.
    organization:
        Company or organisation name written to the copyright line.
    svm_analyst_version:
        Version string of SVM Analyst that generated the code.
    custom_injection_percent:
        Only relevant for CUSTOM_THIPWM: default injection percentage
        to embed as a documentation note.  Range [0.0, 100.0].
    """

    project_name: str = "MyProject"
    author: str = "Amine KHETTAT"
    organization: str = ""
    svm_analyst_version: str = "1.4.5"
    custom_injection_percent: float = 100.0
    # Internal: filled by the generator
    _mod_name: str = field(default="", init=False, repr=False)
    _mod_enum: ModulationMode = field(
        default=ModulationMode.SINUSOIDAL, init=False, repr=False
    )


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------


class CCodeGenerator:
    """Generate MISRA C:2012 compliant C source and header for a modulation mode.

    Usage
    -----
    >>> gen = CCodeGenerator()
    >>> header, source = gen.generate(ModulationMode.SVM, options)
    """

    # Doxygen @defgroup name for all generated modules
    _DOXYGEN_GROUP = "pwm_codegen"

    def generate(
        self,
        modulation: ModulationMode,
        options: CodegenOptions,
    ) -> Tuple[str, str]:
        """Generate the header and C-source code strings.

        Parameters
        ----------
        modulation:
            The modulation algorithm to generate code for.
        options:
            Metadata and configuration for the generated files.

        Returns
        -------
        header_code : str
            Contents of the ``.h`` file.
        source_code : str
            Contents of the ``.c`` file.
        """
        opts = options
        opts._mod_name = _MODULATION_NAMES[modulation]  # noqa: SLF001
        opts._mod_enum = modulation  # noqa: SLF001

        header = self._build_header(modulation, opts)
        source = self._build_source(modulation, opts)
        return header, source

    # ------------------------------------------------------------------
    # Header builder
    # ------------------------------------------------------------------

    def _build_header(
        self, modulation: ModulationMode, opts: CodegenOptions
    ) -> str:
        mod = opts._mod_name  # noqa: SLF001
        mod_lower = mod.lower()
        guard = f"PWM_{mod}_H"
        today = datetime.date.today().isoformat()
        year = datetime.date.today().year
        desc = _MODULATION_DESCRIPTIONS[modulation]
        org = opts.organization if opts.organization else opts.author

        lines: list[str] = []

        # ---- File-level Doxygen block ----------------------------------
        lines += [
            "/**",
            f" * @file    pwm_{mod_lower}.h",
            f" * @brief   Three-phase PWM duty-cycle computation — {modulation.value} modulation.",
            f" * @author  {opts.author}",
            f" * @date    {today}",
            " * @version 1.0.0",
            " *",
            f" * @par Project",
            f" *   {opts.project_name}",
            " *",
            " * @par Generator",
            f" *   SVM Analyst v{opts.svm_analyst_version} — Embedded C Code Generator",
            " *   https://github.com/aminekhettat/SVM-analyst",
            " *",
            f" * @par Copyright",
            f" *   Copyright (c) {year} {org}",
            " *   All rights reserved.",
            " *",
            " * @par License",
            " *   MIT License",
            " *",
            " *   Permission is hereby granted, free of charge, to any person obtaining",
            " *   a copy of this software and associated documentation files (the",
            " *   \"Software\"), to deal in the Software without restriction, including",
            " *   without limitation the rights to use, copy, modify, merge, publish,",
            " *   distribute, sublicense, and/or sell copies of the Software, and to",
            " *   permit persons to whom the Software is furnished to do so, subject to",
            " *   the following conditions:",
            " *",
            " *   The above copyright notice and this permission notice shall be",
            " *   included in all copies or substantial portions of the Software.",
            " *",
            " *   THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND,",
            " *   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF",
            " *   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND",
            " *   NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS",
            " *   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN",
            " *   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN",
            " *   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE",
            " *   SOFTWARE.",
            " *",
            " * @par Disclaimer",
            " *   This code was generated automatically by SVM Analyst.  It is the",
            " *   user's responsibility to validate the algorithm for their specific",
            " *   hardware platform, operating conditions, and safety requirements.",
            " *   The generator author provides no warranty of fitness for any",
            " *   particular purpose.  Always verify generated code against datasheet",
            " *   specifications and applicable functional-safety standards before",
            " *   use in any safety-critical system.",
            " *",
            " * @par MISRA C:2012 Compliance",
            " *   This file is compliant with MISRA C:2012 (mandatory and required",
            " *   rules) with the following notes:",
            " *   - Rule  1.3: No undefined/critical unspecified behaviour.",
            " *   - Rule  2.2: No dead code — all conditional branches are reachable.",
            " *   - Rule  5.1-5.5: All identifiers are unique and non-conflicting.",
            " *   - Rule  8.4: Compatible declarations for all external objects.",
            " *   - Rule 10.1: Operands shall not be of an inappropriate essential type.",
            " *   - Rule 10.3: Values shall not be assigned to a narrower type.",
            " *   - Rule 10.4: Operands in arithmetic have the same essential type.",
            " *   - Rule 14.4: Controlling expressions are of essentially boolean type.",
            " *   - Rule 15.5: Functions have a single point of exit.",
            " *   - Rule 17.3: No implicit function declarations.",
            " *   - Rule 18.1: Pointer arithmetic stays within array bounds.",
            " *   - Rule 20.1: Standard headers included before other headers.",
            " *   - Rule 21.1: Macro names do not redefine reserved identifiers.",
            " *",
            " * @par Lint (PC-lint Plus / FlexeLint) Compliance",
            " *   Designed to compile without warnings under:",
            " *     PC-lint Plus 1.4 with au-misra-c-2012.lnt and au-misra-c-2012-strict.lnt",
            " *     FlexeLint 9.00L with standard MISRA configuration.",
            " *   Type-cast suppressions: none required.",
            " *   Deviation comments follow MISRA deviation permit format where used.",
            " *",
            " * @par Algorithm Description",
            f" *   {desc}",
            " */",
            "",
        ]

        # ---- Include guard ---------------------------------------------
        lines += [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "/* MISRA C:2012 Rule 20.1 — Standard headers before project headers. */",
            "#include <math.h>      /* sinf, fmaxf, fminf                          */",
            "#include <stdbool.h>   /* bool, true, false                           */",
            "#include <stdint.h>    /* uint8_t, uint16_t, float32_t (via typedef)  */",
            "",
            "/* --------------------------------------------------------------------------",
            " * Portable float32 typedef",
            " * --------------------------------------------------------------------------",
            " * If your platform already typedef-s float32_t (e.g. CMSIS, AUTOSAR),",
            " * define PWM_FLOAT32_DEFINED before including this header to suppress the",
            " * typedef below and avoid duplicate-type compiler errors.",
            " * -------------------------------------------------------------------------- */",
            "#ifndef PWM_FLOAT32_DEFINED",
            "/** @typedef float32_t",
            " *  @brief   32-bit IEEE 754 single-precision floating-point scalar. */",
            "typedef float float32_t;",
            "#define PWM_FLOAT32_DEFINED (1)",
            "#endif",
            "",
        ]

        # ---- Module constants ------------------------------------------
        lines += [
            "/* --------------------------------------------------------------------------",
            f" * Module-level constants — {mod}",
            " * -------------------------------------------------------------------------- */",
            "",
            "/** @defgroup PWM_{mod}_Constants Mathematical constants".format(mod=mod),
            " *  @ingroup  " + self._DOXYGEN_GROUP,
            " *  @{",
            " */",
            "",
            f"/** @brief Full circle in radians (2 * pi). */",
            f"#define PWM_{mod}_TWO_PI         (6.28318530717958647692f)",
            "",
            f"/** @brief Two-thirds of pi (2 * pi / 3), phase offset between phases. */",
            f"#define PWM_{mod}_TWO_PI_OVER_3  (2.09439510239319549231f)",
            "",
        ]

        # Mode-specific constants
        if modulation in (ModulationMode.THIPWM_1_6, ModulationMode.THIPWM_1_4,
                          ModulationMode.CUSTOM_THIPWM):
            if modulation == ModulationMode.THIPWM_1_6:
                inj = "0.16666666666666666667f  /* 1/6 */"
            elif modulation == ModulationMode.THIPWM_1_4:
                inj = "0.25f                    /* 1/4 */"
            else:
                inj = None

            lines += [
                f"/** @brief THIPWM pre-gain factor: 1.15 = 1/cos(pi/6) approx. */",
                f"#define PWM_{mod}_PREGAIN     (1.15470053837925152902f)",
                "",
            ]
            if inj is not None:
                lines += [
                    f"/** @brief Third-harmonic injection coefficient. */",
                    f"#define PWM_{mod}_INJ_COEFF  ({inj})",
                    "",
                ]

        if modulation in (ModulationMode.SVM,
                          ModulationMode.DPWM_120_MAX, ModulationMode.DPWM_120_MIN,
                          ModulationMode.DPWM_60_0, ModulationMode.DPWM_60_1,
                          ModulationMode.DPWM_60_2, ModulationMode.DPWM_30_3):
            lines += [
                "/** @brief SVM normalization factor: 2/sqrt(3) = 1.1547... */",
                f"#define PWM_{mod}_SVM_NORM    (1.15470053837925152902f)",
                "",
            ]

        if modulation == ModulationMode.DPWM_60_2:
            lines += [
                "/** @brief Phase shift for DPWM2 clamping decision: pi/6 = 30 degrees. */",
                f"#define PWM_{mod}_PHASE_SHIFT  (0.52359877559829887308f)",
                "",
            ]

        lines += ["/** @} */", "", ""]

        # ---- Status / return-code enum --------------------------------
        lines += [
            "/* --------------------------------------------------------------------------",
            " * Return codes",
            " * -------------------------------------------------------------------------- */",
            "",
            "/**",
            " * @brief  Return status codes for all PWM computation functions.",
            " * @note   MISRA C:2012 Rule 8.4 — all enumerators visible at translation-unit scope.",
            " */",
            "typedef enum",
            "{",
            f"    PWM_{mod}_OK            = 0,  /**< Operation completed successfully.     */",
            f"    PWM_{mod}_ERR_NULL_PTR  = 1,  /**< A required output pointer is NULL.    */",
            f"    PWM_{mod}_ERR_PARAM     = 2   /**< A numerical parameter is out of range.*/",
            f"}} pwm_{mod_lower}_status_t;",
            "",
            "",
        ]

        # ---- Function declaration(s) ----------------------------------
        lines += [
            "/* --------------------------------------------------------------------------",
            " * Public API",
            " * -------------------------------------------------------------------------- */",
            "",
        ]

        if modulation != ModulationMode.CUSTOM_THIPWM:
            lines += self._decl_block(mod, mod_lower, has_inj=False)
        else:
            lines += self._decl_block(mod, mod_lower, has_inj=True)

        # ---- Close guard ---------------------------------------------
        lines += [
            "",
            f"#endif /* {guard} */",
            "",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Declarations helper
    # ------------------------------------------------------------------

    def _decl_block(
        self, mod: str, mod_lower: str, has_inj: bool
    ) -> list[str]:
        """Return Doxygen-documented function declaration lines."""

        inj_param = ""
        inj_doc = ""
        if has_inj:
            inj_param = "\n    float32_t         inj_ratio,"
            inj_doc = (
                " * @param[in]  inj_ratio  Third-harmonic injection ratio [0.0, 1.0].\n"
                " *                        0.0 = sinusoidal; 1.0 = standard 1/6 injection.\n"
            )

        ret = f"pwm_{mod_lower}_status_t"

        return [
            "/**",
            f" * @brief   Compute three-phase PWM duty cycles for the {mod} modulation.",
            " *",
            " * @details The caller provides the instantaneous electrical angle and the",
            " *          desired modulation index.  The three duty-cycle outputs represent",
            " *          the ratio of on-time to the full PWM period for each phase leg:",
            " *          0.0 = permanently off, 1.0 = permanently on.",
            " *          These values map directly to the timer compare register as:",
            " *            CCR = (uint16_t)(duty * (ARR + 1u))",
            " *          where ARR is the auto-reload register value.",
            " *",
            " * @note    Thread-safety: the function is re-entrant (no static state).",
            " * @note    MISRA C:2012 Rule 15.5 — single point of exit enforced.",
            " * @note    Pointer arguments must not alias each other.",
            " *",
            f" * @param[in]  theta_rad  Electrical angle in radians.  Range: [0, 2*PI).",
            f" * @param[in]  mi         Modulation index.  Range: [0.0, 1.0].",
            f"{inj_doc}" if inj_doc else "",
            f" * @param[out] p_duty_a   Pointer to Phase A duty cycle result [0.0, 1.0].",
            f" * @param[out] p_duty_b   Pointer to Phase B duty cycle result [0.0, 1.0].",
            f" * @param[out] p_duty_c   Pointer to Phase C duty cycle result [0.0, 1.0].",
            f" * @return {ret}",
            f" *         - PWM_{mod}_OK            on success.",
            f" *         - PWM_{mod}_ERR_NULL_PTR  if any output pointer is NULL.",
            f" *         - PWM_{mod}_ERR_PARAM     if mi is outside [0.0, 1.0].",
            " */",
            f"{ret} PWM_{mod}_ComputeDuty(",
            f"    float32_t         theta_rad,",
            f"    float32_t         mi,{inj_param}",
            f"    float32_t * const p_duty_a,",
            f"    float32_t * const p_duty_b,",
            f"    float32_t * const p_duty_c",
            ");",
            "",
        ]

    # ------------------------------------------------------------------
    # Source builder
    # ------------------------------------------------------------------

    def _build_source(
        self, modulation: ModulationMode, opts: CodegenOptions
    ) -> str:
        mod = opts._mod_name  # noqa: SLF001
        mod_lower = mod.lower()
        today = datetime.date.today().isoformat()
        year = datetime.date.today().year
        org = opts.organization if opts.organization else opts.author

        lines: list[str] = []

        # ---- File-level Doxygen block ---------------------------------
        lines += [
            "/**",
            f" * @file    pwm_{mod_lower}.c",
            f" * @brief   Implementation — {modulation.value} PWM duty-cycle computation.",
            f" * @author  {opts.author}",
            f" * @date    {today}",
            " * @version 1.0.0",
            " *",
            f" * @par Copyright",
            f" *   Copyright (c) {year} {org}",
            " *   All rights reserved.",
            " *",
            " * @par License",
            " *   MIT License — see pwm_{mod_lower}.h for full text.",
            " *",
            " * @par MISRA C:2012 Compliance",
            " *   See pwm_{mod_lower}.h for the complete compliance statement.",
            " *   Rule deviations: none.",
            " */",
            "",
            "/* MISRA C:2012 Rule 20.1 — include project header last. */",
            "#include <math.h>    /* sinf, cosf, fmaxf, fminf */",
            f"#include \"pwm_{mod_lower}.h\"",
            "",
        ]

        # ---- Module-private helper functions --------------------------
        lines += self._build_helpers(modulation, mod, mod_lower)

        # ---- Main API function ----------------------------------------
        lines += self._build_compute_function(modulation, mod, mod_lower, opts)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helper functions (static, MISRA-compliant)
    # ------------------------------------------------------------------

    def _build_helpers(
        self, modulation: ModulationMode, mod: str, mod_lower: str
    ) -> list[str]:
        """Return static helper function definitions needed by this modulation."""

        lines: list[str] = [
            "/* --------------------------------------------------------------------------",
            " * Module-private helper functions",
            " * MISRA C:2012 Rule 8.7 — functions used only in this translation unit",
            " * are declared static.",
            " * -------------------------------------------------------------------------- */",
            "",
        ]

        # fmaxf wrapper ------------------------------------------------
        lines += [
            "/**",
            f" * @brief   Return the larger of two float32_t scalars.",
            " * @details Provided as a static wrapper to avoid direct use of fmaxf from",
            " *          <math.h> where the compiler does not guarantee NaN propagation.",
            " *          MISRA C:2012 Rule 8.7 — internal linkage.",
            " * @param[in] a  First operand.",
            " * @param[in] b  Second operand.",
            " * @return   The maximum of a and b.",
            " */",
            f"static float32_t pwm_{mod_lower}_fmaxf(float32_t a, float32_t b)",
            "{",
            "    return (a >= b) ? a : b;",
            "}",
            "",
            "/**",
            f" * @brief   Return the smaller of two float32_t scalars.",
            " * @details MISRA C:2012 Rule 8.7 — internal linkage.",
            " * @param[in] a  First operand.",
            " * @param[in] b  Second operand.",
            " * @return   The minimum of a and b.",
            " */",
            f"static float32_t pwm_{mod_lower}_fminf(float32_t a, float32_t b)",
            "{",
            "    return (a <= b) ? a : b;",
            "}",
            "",
            "/**",
            f" * @brief   Clamp a float32_t value to the inclusive range [lo, hi].",
            " * @details MISRA C:2012 Rule 8.7 — internal linkage.",
            " * @param[in] x   Value to clamp.",
            " * @param[in] lo  Lower bound (inclusive).",
            " * @param[in] hi  Upper bound (inclusive).",
            " * @return   Clamped value.",
            " */",
            f"static float32_t pwm_{mod_lower}_clamp("
            "float32_t x, float32_t lo, float32_t hi)",
            "{",
            f"    float32_t result = pwm_{mod_lower}_fmaxf(lo, pwm_{mod_lower}_fminf(x, hi));",
            "    return result;",
            "}",
            "",
        ]

        return lines

    # ------------------------------------------------------------------
    # Compute function body
    # ------------------------------------------------------------------

    def _build_compute_function(
        self,
        modulation: ModulationMode,
        mod: str,
        mod_lower: str,
        opts: CodegenOptions,
    ) -> list[str]:
        """Return lines for the main PWM_<MOD>_ComputeDuty function."""

        ret_type = f"pwm_{mod_lower}_status_t"
        has_inj = (modulation == ModulationMode.CUSTOM_THIPWM)

        # Function signature
        sig_lines: list[str] = [
            "/* --------------------------------------------------------------------------",
            " * Public API implementation",
            " * -------------------------------------------------------------------------- */",
            "",
            f"{ret_type} PWM_{mod}_ComputeDuty(",
            f"    float32_t         theta_rad,",
            f"    float32_t         mi,",
        ]
        if has_inj:
            sig_lines.append("    float32_t         inj_ratio,")
        sig_lines += [
            f"    float32_t * const p_duty_a,",
            f"    float32_t * const p_duty_b,",
            f"    float32_t * const p_duty_c",
            ")",
            "{",
        ]

        # Body
        body: list[str] = []

        # --- variable declarations (MISRA: declare before first use, C99) --
        body += [
            f"    {ret_type} status; /* MISRA C:2012 Rule 15.5: single exit. */",
            "",
            "    /* --- Input validation -------------------------------------------- */",
            "    if ((p_duty_a == NULL) || (p_duty_b == NULL) || (p_duty_c == NULL))",
            "    {",
            f"        status = PWM_{mod}_ERR_NULL_PTR;",
            "    }",
            "    else if ((mi < 0.0f) || (mi > 1.0f))",
            "    {",
            f"        status = PWM_{mod}_ERR_PARAM;",
            "    }",
        ]
        if has_inj:
            body += [
                "    else if ((inj_ratio < 0.0f) || (inj_ratio > 1.0f))",
                "    {",
                f"        status = PWM_{mod}_ERR_PARAM;",
                "    }",
            ]

        body += ["    else", "    {"]
        body += self._algorithm_body(modulation, mod, mod_lower)
        body += [
            f"        status = PWM_{mod}_OK;",
            "    }",
            "",
            "    return status;",
            "}",
            "",
        ]

        return sig_lines + body

    # ------------------------------------------------------------------
    # Algorithm bodies per modulation
    # ------------------------------------------------------------------

    def _algorithm_body(
        self, modulation: ModulationMode, mod: str, mod_lower: str
    ) -> list[str]:
        """Return the inner algorithm lines (indented 8 spaces) for the given mode."""

        if modulation == ModulationMode.SINUSOIDAL:
            return self._algo_spwm(mod, mod_lower)
        if modulation == ModulationMode.THIPWM_1_6:
            return self._algo_thipwm(mod, mod_lower, inj_token=f"PWM_{mod}_INJ_COEFF")
        if modulation == ModulationMode.THIPWM_1_4:
            return self._algo_thipwm(mod, mod_lower, inj_token=f"PWM_{mod}_INJ_COEFF")
        if modulation == ModulationMode.CUSTOM_THIPWM:
            return self._algo_thipwm_custom(mod, mod_lower)
        if modulation == ModulationMode.SVM:
            return self._algo_svm(mod, mod_lower)
        if modulation == ModulationMode.DPWM_120_MAX:
            return self._algo_dpwm(mod, mod_lower, variant="120_MAX")
        if modulation == ModulationMode.DPWM_120_MIN:
            return self._algo_dpwm(mod, mod_lower, variant="120_MIN")
        if modulation == ModulationMode.DPWM_60_1:
            return self._algo_dpwm(mod, mod_lower, variant="60_1")
        if modulation == ModulationMode.DPWM_60_0:
            return self._algo_dpwm(mod, mod_lower, variant="60_0")
        if modulation == ModulationMode.DPWM_60_2:
            return self._algo_dpwm(mod, mod_lower, variant="60_2")
        # DPWM_30_3
        return self._algo_dpwm(mod, mod_lower, variant="30_3")

    # ---- SPWM --------------------------------------------------------

    def _algo_spwm(self, mod: str, mod_lower: str) -> list[str]:
        return [
            "        /* --- Sinusoidal references --------------------------------------- */",
            "        const float32_t half_mi = 0.5f * mi;",
            "",
            "        /* Phase A: sin(theta) */",
            "        *p_duty_a = 0.5f + half_mi * sinf(theta_rad);",
            "",
            "        /* Phase B: sin(theta - 2*PI/3) */",
            f"        *p_duty_b = 0.5f + half_mi * sinf(theta_rad - PWM_{mod}_TWO_PI_OVER_3);",
            "",
            "        /* Phase C: sin(theta + 2*PI/3) */",
            f"        *p_duty_c = 0.5f + half_mi * sinf(theta_rad + PWM_{mod}_TWO_PI_OVER_3);",
        ]

    # ---- THIPWM common -----------------------------------------------

    def _algo_thipwm(
        self, mod: str, mod_lower: str, inj_token: str
    ) -> list[str]:
        return [
            "        /* --- Third-harmonic injection references ------------------------- */",
            "        /* Third-harmonic common-mode component: sin(3*theta). */",
            "        const float32_t sin3 = sinf(3.0f * theta_rad);",
            "",
            "        /* Phase references with 1.15 pre-gain and harmonic injection.      */",
            f"        /* THIPWM formula: v_x = PREGAIN * (sin(theta_x) + K * sin3)       */",
            f"        const float32_t va_ref = PWM_{mod}_PREGAIN *",
            f"            (sinf(theta_rad) + ({inj_token} * sin3));",
            f"        const float32_t vb_ref = PWM_{mod}_PREGAIN *",
            f"            (sinf(theta_rad - PWM_{mod}_TWO_PI_OVER_3) + ({inj_token} * sin3));",
            f"        const float32_t vc_ref = PWM_{mod}_PREGAIN *",
            f"            (sinf(theta_rad + PWM_{mod}_TWO_PI_OVER_3) + ({inj_token} * sin3));",
            "",
            "        /* Map to duty cycles.  References nominally peak at ±1.0.         */",
            "        const float32_t half_mi = 0.5f * mi;",
            "        *p_duty_a = 0.5f + half_mi * va_ref;",
            "        *p_duty_b = 0.5f + half_mi * vb_ref;",
            "        *p_duty_c = 0.5f + half_mi * vc_ref;",
        ]

    # ---- THIPWM Custom -----------------------------------------------

    def _algo_thipwm_custom(self, mod: str, mod_lower: str) -> list[str]:
        return [
            "        /* --- Custom third-harmonic injection ----------------------------- */",
            "        /* inj_ratio = 1.0 corresponds to the standard 1/6 coefficient.    */",
            f"        const float32_t inj_coeff = inj_ratio * (1.0f / 6.0f);",
            "",
            "        const float32_t sin3 = sinf(3.0f * theta_rad);",
            "",
            f"        const float32_t va_ref = PWM_{mod}_PREGAIN *",
            "            (sinf(theta_rad) + (inj_coeff * sin3));",
            f"        const float32_t vb_ref = PWM_{mod}_PREGAIN *",
            f"            (sinf(theta_rad - PWM_{mod}_TWO_PI_OVER_3) + (inj_coeff * sin3));",
            f"        const float32_t vc_ref = PWM_{mod}_PREGAIN *",
            f"            (sinf(theta_rad + PWM_{mod}_TWO_PI_OVER_3) + (inj_coeff * sin3));",
            "",
            "        const float32_t half_mi = 0.5f * mi;",
            "        *p_duty_a = 0.5f + half_mi * va_ref;",
            "        *p_duty_b = 0.5f + half_mi * vb_ref;",
            "        *p_duty_c = 0.5f + half_mi * vc_ref;",
        ]

    # ---- SVM ---------------------------------------------------------

    def _algo_svm(self, mod: str, mod_lower: str) -> list[str]:
        return [
            "        /* --- Space vector modulation ------------------------------------ */",
            "        /* Step 1: three-phase sinusoidal references.                       */",
            "        const float32_t va_sin = sinf(theta_rad);",
            f"        const float32_t vb_sin = sinf(theta_rad - PWM_{mod}_TWO_PI_OVER_3);",
            f"        const float32_t vc_sin = sinf(theta_rad + PWM_{mod}_TWO_PI_OVER_3);",
            "",
            "        /* Step 2: common-mode offset = (max + min) / 2.                   */",
            f"        const float32_t vmax = pwm_{mod_lower}_fmaxf(",
            f"            pwm_{mod_lower}_fmaxf(va_sin, vb_sin), vc_sin);",
            f"        const float32_t vmin = pwm_{mod_lower}_fminf(",
            f"            pwm_{mod_lower}_fminf(va_sin, vb_sin), vc_sin);",
            "        const float32_t ucm = 0.5f * (vmax + vmin);",
            "",
            "        /* Step 3: subtract common-mode offset.  References now peak at    */",
            "        /* sqrt(3)/2 ≈ 0.866.                                              */",
            "        const float32_t va_sv = va_sin - ucm;",
            "        const float32_t vb_sv = vb_sin - ucm;",
            "        const float32_t vc_sv = vc_sin - ucm;",
            "",
            "        /* Step 4: normalize to unity range using 2/sqrt(3) ≈ 1.1547.     */",
            f"        const float32_t va_norm = va_sv * PWM_{mod}_SVM_NORM;",
            f"        const float32_t vb_norm = vb_sv * PWM_{mod}_SVM_NORM;",
            f"        const float32_t vc_norm = vc_sv * PWM_{mod}_SVM_NORM;",
            "",
            "        /* Step 5: map to duty cycles [0, 1].                              */",
            "        const float32_t half_mi = 0.5f * mi;",
            "        *p_duty_a = 0.5f + half_mi * va_norm;",
            "        *p_duty_b = 0.5f + half_mi * vb_norm;",
            "        *p_duty_c = 0.5f + half_mi * vc_norm;",
        ]

    # ---- DPWM variants -----------------------------------------------

    def _algo_dpwm(
        self, mod: str, mod_lower: str, variant: str
    ) -> list[str]:
        lines: list[str] = [
            "        /* --- DPWM: start from SVM reference ------------------------------ */",
            "        const float32_t va_sin = sinf(theta_rad);",
            f"        const float32_t vb_sin = sinf(theta_rad - PWM_{mod}_TWO_PI_OVER_3);",
            f"        const float32_t vc_sin = sinf(theta_rad + PWM_{mod}_TWO_PI_OVER_3);",
            "",
            f"        const float32_t vmax = pwm_{mod_lower}_fmaxf(",
            f"            pwm_{mod_lower}_fmaxf(va_sin, vb_sin), vc_sin);",
            f"        const float32_t vmin = pwm_{mod_lower}_fminf(",
            f"            pwm_{mod_lower}_fminf(va_sin, vb_sin), vc_sin);",
            "        const float32_t ucm  = 0.5f * (vmax + vmin);",
            "",
            "        /* Normalised switch times: t in [0, 1] corresponding to 0%-100%.  */",
            "        const float32_t tas = 0.5f + 0.5f * (mi * (va_sin - ucm) *",
            f"            PWM_{mod}_SVM_NORM);",
            "        const float32_t tbs = 0.5f + 0.5f * (mi * (vb_sin - ucm) *",
            f"            PWM_{mod}_SVM_NORM);",
            "        const float32_t tcs = 0.5f + 0.5f * (mi * (vc_sin - ucm) *",
            f"            PWM_{mod}_SVM_NORM);",
            "",
            f"        const float32_t tmax_s = pwm_{mod_lower}_fmaxf(",
            f"            pwm_{mod_lower}_fmaxf(tas, tbs), tcs);",
            f"        const float32_t tmin_s = pwm_{mod_lower}_fminf(",
            f"            pwm_{mod_lower}_fminf(tas, tbs), tcs);",
            "",
        ]

        # variant-specific offset computation
        if variant == "120_MAX":
            lines += [
                "        /* DPWM_120_MAX: shift so that the highest switch-time = 1.0. */",
                "        const float32_t toffset = 1.0f - tmax_s;",
            ]
        elif variant == "120_MIN":
            lines += [
                "        /* DPWM_120_MIN: shift so that the lowest switch-time = 0.0. */",
                "        const float32_t toffset = -(tmin_s);",
            ]
        elif variant == "60_1":
            lines += [
                "        /* DPWM1: if (tmax + tmin) >= 1, use MAX clamping, else MIN. */",
                "        const bool use_max = ((tmax_s + tmin_s) >= 1.0f);",
                "        const float32_t toffset = (use_max == true) ?",
                "            (1.0f - tmax_s) : -(tmin_s);",
            ]
        elif variant == "60_0":
            lines += [
                "        /* DPWM0: inverse of DPWM1 clamping logic. */",
                "        const bool use_min = ((tmax_s + tmin_s) >= 1.0f);",
                "        const float32_t toffset = (use_min == true) ?",
                "            -(tmin_s) : (1.0f - tmax_s);",
            ]
        elif variant == "60_2":
            lines += [
                "        /* DPWM2: decision based on sin(theta - 30°). */",
                f"        const bool use_max = (sinf(theta_rad - PWM_{mod}_PHASE_SHIFT) >= 0.0f);",
                "        const float32_t toffset = (use_max == true) ?",
                "            (1.0f - tmax_s) : -(tmin_s);",
            ]
        else:  # 30_3
            lines += [
                "        /* DPWM3: alternates DPWM1 and DPWM0 every 30 electrical degrees. */",
                "        const bool cond_60   = ((tmax_s + tmin_s) >= 1.0f);",
                "        const float32_t off1 = (cond_60 == true) ?",
                "            (1.0f - tmax_s) : -(tmin_s);",
                "        const float32_t off0 = (cond_60 == true) ?",
                "            -(tmin_s) : (1.0f - tmax_s);",
                "        const bool use_dpwm1 = (sinf(6.0f * theta_rad) >= 0.0f);",
                "        const float32_t toffset = (use_dpwm1 == true) ? off1 : off0;",
            ]

        # Apply offset and clamp
        lines += [
            "",
            "        /* Apply offset and clamp results to [0.0, 1.0].                   */",
            f"        *p_duty_a = pwm_{mod_lower}_clamp(tas + toffset, 0.0f, 1.0f);",
            f"        *p_duty_b = pwm_{mod_lower}_clamp(tbs + toffset, 0.0f, 1.0f);",
            f"        *p_duty_c = pwm_{mod_lower}_clamp(tcs + toffset, 0.0f, 1.0f);",
        ]

        return lines
