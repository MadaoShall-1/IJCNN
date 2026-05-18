"""Tests for Type 2 Stage 6: Response Assembly."""

import unittest

from parser.schemas import ProblemParseObject

from type2.schemas import (
    DiagnosisObject,
    FormulaEntry,
    FormulaSet,
    StepObject,
    TraceObject,
)
from type2.stage6 import (
    _aggregate_confidence,
    _build_step_cot,
    _collect_fol_axioms,
    _collect_premises,
    build_response,
    extract_final_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(step_id: str, status: str = "OK", goal: str = "test",
          intermediate_answer: str = "5.0 V",
          formula_ids: list | None = None,
          confidence: float | None = None,
          thought: str = "", step_input: str = "") -> StepObject:
    s = StepObject(step_id=step_id, goal=goal, type="formula_application")
    s.status = status
    s.intermediate_answer = intermediate_answer
    s.formula_ids = formula_ids or []
    s.confidence = confidence
    s.thought = thought
    s.step_input = step_input
    return s


def _entry(fid: str, premise: str = "", fol: str = "") -> FormulaEntry:
    return FormulaEntry(
        id=fid, topic="circuits", subtopic="ohms_law",
        target_quantities=[], canonical_quantity_names=[],
        text="V = I * R", formula="V = I * R", sympy_expr="",
        tool_dispatch="sympy", variables={},
        premise_text=premise, fol_axiom=fol,
    )


def _make_formula_set(step_ids: list, entry_id: str = "CKT-001",
                      premise: str = "Ohm's Law: V=IR",
                      fol: str = "∀V ∀I ∀R (V = I*R)") -> FormulaSet:
    e = _entry(entry_id, premise=premise, fol=fol)
    return FormulaSet(
        formulas={sid: e for sid in step_ids},
        retrieval_confidence=0.9,
        path_index=0,
    )


def _make_trace(steps: list, status: str = "PASS",
                final_answer: str = "") -> TraceObject:
    t = TraceObject(problem_id="test-001", formula_path_index=0)
    t.steps = steps
    t.trace_status = status
    t.final_answer = final_answer
    return t


def _make_parse_obj() -> ProblemParseObject:
    return ProblemParseObject(
        problem_text="A 10 Ω resistor has 5 V across it. Find the current.",
        domains=["circuits"],
        sub_domains=[],
        known_quantities={},
        step_plan=[],
        vso={},
    )


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------

class TestExtractFinalAnswer(unittest.TestCase):

    def test_uses_trace_final_answer(self):
        trace = _make_trace([], final_answer="3.5 A")
        self.assertEqual(extract_final_answer(trace), "3.5 A")

    def test_falls_back_to_last_step_with_numeric(self):
        steps = [
            _step("s1", intermediate_answer="10 Ω"),
            _step("s2", intermediate_answer="0.5 A"),
        ]
        trace = _make_trace(steps, final_answer="")
        self.assertEqual(extract_final_answer(trace), "0.5 A")

    def test_empty_trace_no_answer_returns_empty(self):
        trace = _make_trace([], final_answer="")
        self.assertEqual(extract_final_answer(trace), "")

    def test_step_without_number_skipped(self):
        steps = [
            _step("s1", intermediate_answer="unknown"),
            _step("s2", intermediate_answer="2.0 W"),
        ]
        trace = _make_trace(steps, final_answer="")
        self.assertEqual(extract_final_answer(trace), "2.0 W")

    def test_strips_whitespace(self):
        trace = _make_trace([], final_answer="  4.2 V  ")
        self.assertEqual(extract_final_answer(trace), "4.2 V")


# ---------------------------------------------------------------------------
# _collect_premises / _collect_fol_axioms
# ---------------------------------------------------------------------------

class TestCollectPrereqs(unittest.TestCase):

    def test_premises_collected(self):
        fs = _make_formula_set(["s1", "s2"], premise="Ohm's Law: V=IR")
        premises = _collect_premises(fs)
        self.assertIn("Ohm's Law: V=IR", premises)

    def test_fol_axioms_collected(self):
        fs = _make_formula_set(["s1"], fol="∀V ∀I ∀R (V = I*R)")
        axioms = _collect_fol_axioms(fs)
        self.assertIn("∀V ∀I ∀R (V = I*R)", axioms)

    def test_duplicates_removed(self):
        # Two steps use the same formula → premise should appear once
        fs = _make_formula_set(["s1", "s2"], premise="Ohm's Law")
        premises = _collect_premises(fs)
        self.assertEqual(len([p for p in premises if p == "Ohm's Law"]), 1)

    def test_empty_premise_excluded(self):
        fs = _make_formula_set(["s1"], premise="")
        premises = _collect_premises(fs)
        self.assertEqual(premises, [])

    def test_none_entry_skipped(self):
        fs = FormulaSet(formulas={"s1": None}, retrieval_confidence=0.9, path_index=0)
        premises = _collect_premises(fs)
        self.assertEqual(premises, [])


# ---------------------------------------------------------------------------
# _build_step_cot
# ---------------------------------------------------------------------------

class TestBuildStepCot(unittest.TestCase):

    def test_includes_goal(self):
        steps = [_step("s1", goal="Find current")]
        cot = _build_step_cot(steps)
        self.assertIn("Find current", cot)

    def test_includes_answer(self):
        steps = [_step("s1", intermediate_answer="0.5 A")]
        cot = _build_step_cot(steps)
        self.assertIn("0.5 A", cot)

    def test_includes_step_work(self):
        steps = [_step("s1", step_input="I = V/R = 5/10")]
        cot = _build_step_cot(steps)
        self.assertIn("I = V/R = 5/10", cot)

    def test_empty_steps(self):
        self.assertEqual(_build_step_cot([]), "")

    def test_multiple_steps_ordered(self):
        steps = [_step("s1", goal="step A"), _step("s2", goal="step B")]
        cot = _build_step_cot(steps)
        self.assertLess(cot.index("step A"), cot.index("step B"))


# ---------------------------------------------------------------------------
# _aggregate_confidence
# ---------------------------------------------------------------------------

class TestAggregateConfidence(unittest.TestCase):

    def test_pass_trace_full_confidence(self):
        steps = [_step("s1", confidence=1.0), _step("s2", confidence=1.0)]
        conf = _aggregate_confidence(steps, "PASS", 1.0)
        self.assertAlmostEqual(conf, 1.0)

    def test_repaired_trace_multiplied(self):
        steps = [_step("s1", confidence=1.0)]
        conf = _aggregate_confidence(steps, "REPAIRED", 1.0)
        self.assertAlmostEqual(conf, 0.8)

    def test_fail_trace_reduced(self):
        steps = [_step("s1", confidence=1.0)]
        conf = _aggregate_confidence(steps, "FAIL", 1.0)
        self.assertAlmostEqual(conf, 0.3)

    def test_empty_steps_returns_zero(self):
        self.assertAlmostEqual(_aggregate_confidence([], "PASS", 1.0), 0.0)

    def test_capped_at_1(self):
        steps = [_step("s1", confidence=1.0)]
        conf = _aggregate_confidence(steps, "PASS", 1.5)
        self.assertLessEqual(conf, 1.0)

    def test_no_confidence_on_steps_uses_half(self):
        steps = [_step("s1", confidence=None)]
        conf = _aggregate_confidence(steps, "PASS", 1.0)
        self.assertAlmostEqual(conf, 0.5)


# ---------------------------------------------------------------------------
# build_response — integration
# ---------------------------------------------------------------------------

class TestBuildResponse(unittest.TestCase):

    def setUp(self):
        self.steps = [
            _step("s1", goal="Find I", intermediate_answer="0.5 A",
                  formula_ids=["CKT-001"], confidence=0.9),
        ]
        self.trace = _make_trace(self.steps, final_answer="0.5 A")
        self.fs = _make_formula_set(["s1"], premise="V=IR", fol="∀V ∀I ∀R (V=I*R)")
        self.parse_obj = _make_parse_obj()

    def test_answer_in_response(self):
        r = build_response(self.trace, self.parse_obj, self.fs)
        self.assertEqual(r["answer"], "0.5 A")

    def test_required_keys_present(self):
        r = build_response(self.trace, self.parse_obj, self.fs)
        for key in ("answer", "confidence", "chain_of_thought", "premises",
                    "fol_axioms", "trace_status", "formula_path_index", "steps", "diagnosis"):
            self.assertIn(key, r)

    def test_confidence_is_float_in_range(self):
        r = build_response(self.trace, self.parse_obj, self.fs)
        self.assertIsInstance(r["confidence"], float)
        self.assertGreaterEqual(r["confidence"], 0.0)
        self.assertLessEqual(r["confidence"], 1.0)

    def test_premises_list(self):
        r = build_response(self.trace, self.parse_obj, self.fs)
        self.assertIsInstance(r["premises"], list)
        self.assertIn("V=IR", r["premises"])

    def test_fol_axioms_list(self):
        r = build_response(self.trace, self.parse_obj, self.fs)
        self.assertIsInstance(r["fol_axioms"], list)

    def test_steps_summary_correct_length(self):
        r = build_response(self.trace, self.parse_obj, self.fs)
        self.assertEqual(len(r["steps"]), len(self.steps))

    def test_diagnosis_none_when_not_provided(self):
        r = build_response(self.trace, self.parse_obj, self.fs)
        self.assertIsNone(r["diagnosis"])

    def test_diagnosis_dict_when_provided(self):
        diag = DiagnosisObject()
        r = build_response(self.trace, self.parse_obj, self.fs, diag)
        self.assertIsInstance(r["diagnosis"], dict)

    def test_trace_status_passed_through(self):
        r = build_response(self.trace, self.parse_obj, self.fs)
        self.assertEqual(r["trace_status"], "PASS")

    def test_cot_not_empty(self):
        r = build_response(self.trace, self.parse_obj, self.fs)
        self.assertTrue(r["chain_of_thought"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
