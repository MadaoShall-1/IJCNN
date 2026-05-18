"""Tests for Type 2 Stage 2+3: VSO init, checkable classification, SymPy verification."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from parser.schemas import ProblemParseObject

from type2.schemas import FormulaEntry, FormulaSet, StepObject, VSOEntry
from type2.stage2 import (
    PHYSICS_CONSTANTS,
    _build_prior_summary,
    _extract_numeric,
    _infer_output_unit,
    _parse_output_values,
    _verifier_confidence,
    classify_checkable,
    init_vso,
    map_formula_vars_to_vso,
    sympy_verify_step,
)

_LIBRARY_PATH = Path(__file__).parent / "type2" / "formula_library.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parse_obj(
    *,
    problem_text: str = "",
    known_quantities: dict | None = None,
    vso: dict | None = None,
    step_plan: list | None = None,
) -> ProblemParseObject:
    return ProblemParseObject(
        problem_text=problem_text,
        domains=[],
        sub_domains=[],
        known_quantities=known_quantities or {},
        step_plan=step_plan or [],
        vso=vso or {},
    )


def _ohms_law_entry() -> FormulaEntry:
    return FormulaEntry(
        id="CKT-001",
        topic="circuits",
        subtopic="ohms_law",
        target_quantities=["V", "I", "R"],
        canonical_quantity_names=["electric_potential", "electric_current", "resistance"],
        text="V = I * R",
        formula="V = I * R",
        sympy_expr="Eq(V, I * R)",
        tool_dispatch="sympy",
        variables={
            "V": {"symbol": "V", "name": "voltage", "unit_symbol": "V", "unit_name": "volts"},
            "I": {"symbol": "I", "name": "current", "unit_symbol": "A", "unit_name": "amperes"},
            "R": {"symbol": "R", "name": "resistance", "unit_symbol": "Ω", "unit_name": "ohms"},
        },
    )


def _capacitor_energy_entry() -> FormulaEntry:
    return FormulaEntry(
        id="CKT-011",
        topic="circuits",
        subtopic="capacitor_energy",
        target_quantities=["U", "C", "V"],
        canonical_quantity_names=["energy", "capacitance", "electric_potential"],
        text="U = 0.5 * C * V^2",
        formula="U = 0.5 * C * V**2",
        sympy_expr="Eq(U, Rational(1, 2) * C * V**2)",
        tool_dispatch="sympy",
        variables={
            "U": {"symbol": "U", "name": "energy", "unit_symbol": "J", "unit_name": "joules"},
            "C": {"symbol": "C", "name": "capacitance", "unit_symbol": "F", "unit_name": "farads"},
            "V": {"symbol": "V", "name": "voltage", "unit_symbol": "V", "unit_name": "volts"},
        },
    )


def _make_vso(**kwargs) -> dict:
    """Build a VSO dict for testing."""
    vso = {}
    for name, val in kwargs.items():
        vso[name] = VSOEntry(value=val, unit_symbol="", unit_name="", defined_at="test", updated_at="test")
    return vso


# ---------------------------------------------------------------------------
# PHYSICS_CONSTANTS
# ---------------------------------------------------------------------------

class TestPhysicsConstants(unittest.TestCase):

    def test_g_present(self):
        self.assertIn("g", PHYSICS_CONSTANTS)

    def test_g_value_approx(self):
        self.assertAlmostEqual(PHYSICS_CONSTANTS["g"]["value"], 9.80665, places=3)

    def test_ke_present(self):
        self.assertIn("k_e", PHYSICS_CONSTANTS)

    def test_ke_value_approx(self):
        self.assertAlmostEqual(PHYSICS_CONSTANTS["k_e"]["value"], 8.9875517923e9, delta=1e5)

    def test_epsilon_0_present(self):
        self.assertIn("epsilon_0", PHYSICS_CONSTANTS)


# ---------------------------------------------------------------------------
# init_vso
# ---------------------------------------------------------------------------

class TestInitVso(unittest.TestCase):

    def test_constants_populated(self):
        parse_obj = _make_parse_obj()
        vso = init_vso(parse_obj)
        self.assertIn("g", vso)
        self.assertIn("k_e", vso)

    def test_known_quantities_added(self):
        parse_obj = _make_parse_obj(
            known_quantities={
                "R": {"value": 10.0, "unit_symbol": "Ω", "unit_name": "ohms"},
                "V": {"value": 5.0, "unit_symbol": "V", "unit_name": "volts"},
            }
        )
        vso = init_vso(parse_obj)
        self.assertIn("R", vso)
        self.assertAlmostEqual(vso["R"].value, 10.0)
        self.assertAlmostEqual(vso["V"].value, 5.0)

    def test_known_quantities_override_constants(self):
        parse_obj = _make_parse_obj(
            known_quantities={
                "g": {"value": 1.62, "unit_symbol": "m/s^2", "unit_name": "moon gravity"},
            }
        )
        vso = init_vso(parse_obj)
        self.assertAlmostEqual(vso["g"].value, 1.62)

    def test_known_quantities_defined_at_stage0(self):
        parse_obj = _make_parse_obj(
            known_quantities={"R": {"value": 5.0, "unit_symbol": "Ω", "unit_name": "ohms"}}
        )
        vso = init_vso(parse_obj)
        self.assertEqual(vso["R"].defined_at, "stage0")

    def test_non_numeric_value_skipped(self):
        parse_obj = _make_parse_obj(
            known_quantities={"X": {"value": "not_a_number", "unit_symbol": "", "unit_name": ""}}
        )
        vso = init_vso(parse_obj)
        self.assertNotIn("X", vso)

    def test_none_value_skipped(self):
        parse_obj = _make_parse_obj(
            known_quantities={"Y": {"unit_symbol": "m", "unit_name": "metres"}}
        )
        vso = init_vso(parse_obj)
        self.assertNotIn("Y", vso)

    def test_parse_obj_vso_populated(self):
        parse_obj = _make_parse_obj(
            vso={"I": {"value": 2.0, "unit_symbol": "A", "unit_name": "amperes",
                       "defined_at": "stage0", "updated_at": "stage0"}}
        )
        vso = init_vso(parse_obj)
        self.assertIn("I", vso)
        self.assertAlmostEqual(vso["I"].value, 2.0)

    def test_known_quantities_take_priority_over_vso_field(self):
        parse_obj = _make_parse_obj(
            known_quantities={"I": {"value": 3.0, "unit_symbol": "A", "unit_name": "amperes"}},
            vso={"I": {"value": 99.0, "unit_symbol": "A", "unit_name": "amperes",
                       "defined_at": "stage0", "updated_at": "stage0"}},
        )
        vso = init_vso(parse_obj)
        self.assertAlmostEqual(vso["I"].value, 3.0)

    def test_returns_vso_entries(self):
        parse_obj = _make_parse_obj()
        vso = init_vso(parse_obj)
        for val in vso.values():
            self.assertIsInstance(val, VSOEntry)


# ---------------------------------------------------------------------------
# classify_checkable
# ---------------------------------------------------------------------------

class TestClassifyCheckable(unittest.TestCase):

    def test_formula_application_checkable(self):
        self.assertTrue(classify_checkable(
            "formula_application", ["CKT-001"], {"V": 5.0, "R": 10.0}
        ))

    def test_calculation_checkable(self):
        self.assertTrue(classify_checkable(
            "calculation", ["CKT-001"], {"V": 5.0, "R": 10.0}
        ))

    def test_unit_conversion_checkable(self):
        self.assertTrue(classify_checkable(
            "unit_conversion", ["CKT-001"], {"V": 5.0}
        ))

    def test_setup_not_checkable(self):
        self.assertFalse(classify_checkable(
            "setup", ["CKT-001"], {"V": 5.0}
        ))

    def test_conclusion_not_checkable(self):
        self.assertFalse(classify_checkable(
            "conclusion", ["CKT-001"], {"V": 5.0}
        ))

    def test_empty_formula_ids_not_checkable(self):
        self.assertFalse(classify_checkable(
            "formula_application", [], {"V": 5.0}
        ))

    def test_none_value_not_checkable(self):
        self.assertFalse(classify_checkable(
            "formula_application", ["CKT-001"], {"V": 5.0, "R": None}
        ))

    def test_empty_input_vars_not_checkable(self):
        self.assertFalse(classify_checkable(
            "formula_application", ["CKT-001"], {}
        ))


# ---------------------------------------------------------------------------
# _extract_numeric
# ---------------------------------------------------------------------------

class TestExtractNumeric(unittest.TestCase):

    def test_simple_integer(self):
        self.assertAlmostEqual(_extract_numeric("5 V"), 5.0)

    def test_decimal(self):
        self.assertAlmostEqual(_extract_numeric("3.14 m/s"), 3.14)

    def test_scientific(self):
        self.assertAlmostEqual(_extract_numeric("1.6e-19 C"), 1.6e-19)

    def test_negative(self):
        self.assertAlmostEqual(_extract_numeric("-3.2 J"), -3.2)

    def test_no_number_returns_none(self):
        self.assertIsNone(_extract_numeric("no number here"))

    def test_empty_string(self):
        self.assertIsNone(_extract_numeric(""))

    def test_leading_whitespace(self):
        self.assertAlmostEqual(_extract_numeric("  10.5 ohms"), 10.5)

    def test_comma_decimal(self):
        self.assertAlmostEqual(_extract_numeric("3,14 m"), 3.14)


# ---------------------------------------------------------------------------
# sympy_verify_step
# ---------------------------------------------------------------------------

try:
    import sympy
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False


class TestSympyVerifyStep(unittest.TestCase):

    def setUp(self):
        self.ohm = _ohms_law_entry()

    @unittest.skipUnless(_HAS_SYMPY, "sympy not installed")
    def test_correct_ohms_law_voltage(self):
        # V = I * R = 0.5 * 10 = 5.0
        verdict, conf = sympy_verify_step(
            self.ohm, {"I": 0.5, "R": 10.0}, "5.0 V"
        )
        self.assertEqual(verdict, "CORRECT")
        self.assertAlmostEqual(conf, 1.0)

    @unittest.skipUnless(_HAS_SYMPY, "sympy not installed")
    def test_incorrect_ohms_law_voltage(self):
        verdict, conf = sympy_verify_step(
            self.ohm, {"I": 0.5, "R": 10.0}, "7.0 V"
        )
        self.assertEqual(verdict, "INCORRECT")
        self.assertAlmostEqual(conf, 0.0)

    @unittest.skipUnless(_HAS_SYMPY, "sympy not installed")
    def test_correct_ohms_law_current(self):
        # I = V / R = 12.0 / 4.0 = 3.0
        verdict, conf = sympy_verify_step(
            self.ohm, {"V": 12.0, "R": 4.0}, "3.0 A"
        )
        self.assertEqual(verdict, "CORRECT")

    @unittest.skipUnless(_HAS_SYMPY, "sympy not installed")
    def test_no_numeric_in_answer_returns_uncertain(self):
        verdict, conf = sympy_verify_step(
            self.ohm, {"I": 0.5, "R": 10.0}, "five volts"
        )
        self.assertEqual(verdict, "UNCERTAIN")
        self.assertAlmostEqual(conf, 0.5)

    def test_non_sympy_dispatch_returns_uncertain(self):
        entry = _ohms_law_entry()
        entry.tool_dispatch = "llm"
        verdict, conf = sympy_verify_step(entry, {"I": 0.5, "R": 10.0}, "5.0 V")
        self.assertEqual(verdict, "UNCERTAIN")

    def test_empty_sympy_expr_returns_uncertain(self):
        entry = _ohms_law_entry()
        entry.sympy_expr = ""
        verdict, conf = sympy_verify_step(entry, {"I": 0.5, "R": 10.0}, "5.0 V")
        self.assertEqual(verdict, "UNCERTAIN")

    @unittest.skipUnless(_HAS_SYMPY, "sympy not installed")
    def test_tolerance_accepted(self):
        # 5.0 vs 5.001 — should be within tolerance
        verdict, _ = sympy_verify_step(
            self.ohm, {"I": 0.5, "R": 10.0}, "5.001 V"
        )
        self.assertEqual(verdict, "CORRECT")


# ---------------------------------------------------------------------------
# map_formula_vars_to_vso
# ---------------------------------------------------------------------------

class TestMapFormulaVarsToVso(unittest.TestCase):

    def setUp(self):
        self.ohm = _ohms_law_entry()

    def test_direct_match(self):
        vso = _make_vso(V=12.0, I=2.0, R=6.0)
        result = map_formula_vars_to_vso(self.ohm, vso)
        self.assertAlmostEqual(result["V"], 12.0)
        self.assertAlmostEqual(result["I"], 2.0)
        self.assertAlmostEqual(result["R"], 6.0)

    def test_partial_match(self):
        vso = _make_vso(V=5.0, R=10.0)
        result = map_formula_vars_to_vso(self.ohm, vso)
        self.assertIn("V", result)
        self.assertIn("R", result)
        self.assertNotIn("I", result)

    def test_empty_vso_returns_empty(self):
        result = map_formula_vars_to_vso(self.ohm, {})
        self.assertEqual(result, {})

    def test_canonical_match_source_voltage(self):
        # "source_voltage" canonicalizes to "electric_potential" → maps to V
        vso = _make_vso(source_voltage=9.0, R=3.0)
        result = map_formula_vars_to_vso(self.ohm, vso)
        self.assertIn("V", result)
        self.assertAlmostEqual(result["V"], 9.0)


# ---------------------------------------------------------------------------
# _verifier_confidence
# ---------------------------------------------------------------------------

class TestVerifierConfidence(unittest.TestCase):

    def test_correct_not_repaired(self):
        self.assertAlmostEqual(_verifier_confidence("formula_application", "CORRECT"), 1.0)

    def test_correct_repaired(self):
        self.assertAlmostEqual(_verifier_confidence("formula_application", "CORRECT", was_repaired=True), 0.6)

    def test_correct_llm(self):
        self.assertAlmostEqual(_verifier_confidence("formula_application", "CORRECT", used_llm=True), 0.8)

    def test_uncertain(self):
        self.assertAlmostEqual(_verifier_confidence("formula_application", "UNCERTAIN"), 0.5)

    def test_incorrect(self):
        self.assertAlmostEqual(_verifier_confidence("formula_application", "INCORRECT"), 0.0)

    def test_setup_returns_0_9(self):
        self.assertAlmostEqual(_verifier_confidence("setup", "UNCERTAIN"), 0.9)

    def test_unit_conversion_not_checkable(self):
        # unit_conversion IS checkable by type; confidence depends on verdict
        conf = _verifier_confidence("unit_conversion", "CORRECT")
        self.assertAlmostEqual(conf, 1.0)


# ---------------------------------------------------------------------------
# _build_prior_summary
# ---------------------------------------------------------------------------

class TestBuildPriorSummary(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(_build_prior_summary([]), "")

    def test_single_step_with_answer(self):
        s = StepObject(step_id="s1", goal="Find V", type="formula_application")
        s.intermediate_answer = "5.0 V"
        result = _build_prior_summary([s])
        self.assertIn("s1", result)
        self.assertIn("5.0 V", result)

    def test_step_without_answer_excluded(self):
        s = StepObject(step_id="s1", goal="Find V", type="formula_application")
        result = _build_prior_summary([s])
        self.assertEqual(result, "")

    def test_multiple_steps(self):
        s1 = StepObject(step_id="s1", goal="Find R", type="formula_application")
        s1.intermediate_answer = "10 Ω"
        s2 = StepObject(step_id="s2", goal="Find I", type="formula_application")
        s2.intermediate_answer = "0.5 A"
        result = _build_prior_summary([s1, s2])
        self.assertIn("s1", result)
        self.assertIn("s2", result)


# ---------------------------------------------------------------------------
# _parse_output_values
# ---------------------------------------------------------------------------

class TestParseOutputValues(unittest.TestCase):

    def test_valid_json(self):
        result = _parse_output_values('{"V": 5.0, "I": 0.5}', {})
        self.assertAlmostEqual(result["V"], 5.0)
        self.assertAlmostEqual(result["I"], 0.5)

    def test_invalid_json_falls_back(self):
        result = _parse_output_values("not valid json", {"V": {}})
        self.assertIn("V", result)
        self.assertIsNone(result["V"])

    def test_non_numeric_values_excluded(self):
        result = _parse_output_values('{"V": 5.0, "label": "volts"}', {})
        self.assertIn("V", result)
        self.assertNotIn("label", result)

    def test_empty_json_object(self):
        result = _parse_output_values("{}", {"V": {}})
        self.assertEqual(result, {})

    def test_integer_values_accepted(self):
        result = _parse_output_values('{"R": 10}', {})
        self.assertAlmostEqual(result["R"], 10.0)


# ---------------------------------------------------------------------------
# _infer_output_unit
# ---------------------------------------------------------------------------

class TestInferOutputUnit(unittest.TestCase):

    def setUp(self):
        self.ohm = _ohms_law_entry()

    def test_exact_match_from_formula(self):
        vso = _make_vso()
        sym, name = _infer_output_unit("V", self.ohm, vso)
        self.assertEqual(sym, "V")
        self.assertEqual(name, "volts")

    def test_var_not_in_formula_falls_to_vso(self):
        vso = _make_vso()
        vso["Z"] = VSOEntry(value=1.0, unit_symbol="Ω", unit_name="ohms",
                             defined_at="test", updated_at="test")
        sym, name = _infer_output_unit("Z", None, vso)
        self.assertEqual(sym, "Ω")

    def test_unknown_var_returns_empty(self):
        sym, name = _infer_output_unit("xyz_unknown", None, {})
        self.assertEqual(sym, "")
        self.assertEqual(name, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
