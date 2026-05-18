"""Tests for Type 2 Stage 4: Error Structuring and Diagnosis."""

import unittest

from type2.schemas import (
    CotIssue,
    DiagnosisObject,
    FormulaEntry,
    FormulaSet,
    StepObject,
    TraceObject,
)
from type2.stage4 import (
    _check_cot_consistency_heuristic,
    _classify_step_error,
    _find_fws,
    _infer_global_error_type,
    diagnose_trace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(step_id: str, status: str = "OK", goal: str = "test",
          input_var: dict | None = None, output_var: dict | None = None,
          intermediate_answer: str = "5.0 V") -> StepObject:
    s = StepObject(step_id=step_id, goal=goal, type="formula_application")
    s.status = status
    s.input_var = input_var or {}
    s.output_var = output_var or {}
    s.intermediate_answer = intermediate_answer
    return s


def _formula_entry(fid: str = "CKT-001") -> FormulaEntry:
    return FormulaEntry(
        id=fid,
        topic="circuits",
        subtopic="ohms_law",
        target_quantities=["V", "I", "R"],
        canonical_quantity_names=["electric_potential", "electric_current", "resistance"],
        text="V = I * R",
        formula="V = I * R",
        sympy_expr="Eq(V, I * R)",
        tool_dispatch="sympy",
        variables={},
    )


def _make_formula_set(step_ids: list, entry_id: str = "CKT-001") -> FormulaSet:
    entry = _formula_entry(entry_id)
    return FormulaSet(
        formulas={sid: entry for sid in step_ids},
        retrieval_confidence=0.9,
        path_index=0,
    )


def _make_trace(steps: list, status: str = "FAIL") -> TraceObject:
    t = TraceObject(problem_id="test-001", formula_path_index=0)
    t.steps = steps
    t.trace_status = status
    return t


# ---------------------------------------------------------------------------
# _find_fws
# ---------------------------------------------------------------------------

class TestFindFws(unittest.TestCase):

    def test_first_wrong_step(self):
        steps = [_step("s1", "OK"), _step("s2", "WRONG"), _step("s3", "OK")]
        self.assertEqual(_find_fws(steps), 1)

    def test_no_wrong_step(self):
        steps = [_step("s1", "OK"), _step("s2", "REPAIRED")]
        self.assertIsNone(_find_fws(steps))

    def test_wrong_at_first_position(self):
        steps = [_step("s1", "WRONG"), _step("s2", "OK")]
        self.assertEqual(_find_fws(steps), 0)

    def test_empty_steps(self):
        self.assertIsNone(_find_fws([]))

    def test_incorrect_treated_as_wrong(self):
        steps = [_step("s1", "OK"), _step("s2", "INCORRECT")]
        self.assertEqual(_find_fws(steps), 1)


# ---------------------------------------------------------------------------
# _classify_step_error
# ---------------------------------------------------------------------------

class TestClassifyStepError(unittest.TestCase):

    def test_ok_step(self):
        self.assertEqual(_classify_step_error(_step("s1", "OK"), _formula_entry()), "OK")

    def test_repaired_step_is_ok(self):
        self.assertEqual(_classify_step_error(_step("s1", "REPAIRED"), _formula_entry()), "OK")

    def test_uncertain_step(self):
        self.assertEqual(_classify_step_error(_step("s1", "UNCERTAIN"), _formula_entry()), "UNCERTAIN")

    def test_wrong_with_no_formula_is_e1(self):
        step = _step("s1", "WRONG")
        step.intermediate_answer = ""
        self.assertEqual(_classify_step_error(step, None), "E1")

    def test_wrong_no_numeric_answer_is_e3(self):
        step = _step("s1", "WRONG", intermediate_answer="no number here")
        self.assertEqual(_classify_step_error(step, _formula_entry()), "E3")

    def test_wrong_with_numeric_answer_is_e3(self):
        step = _step("s1", "WRONG", intermediate_answer="5.0 V")
        self.assertEqual(_classify_step_error(step, _formula_entry()), "E3")


# ---------------------------------------------------------------------------
# _check_cot_consistency_heuristic
# ---------------------------------------------------------------------------

class TestCotConsistency(unittest.TestCase):

    def test_no_issues_when_all_ok(self):
        steps = [
            _step("s1", "OK", output_var={"V": 5.0}),
            _step("s2", "OK", input_var={"V": {}}),
        ]
        issues = _check_cot_consistency_heuristic(steps)
        self.assertEqual(issues, [])

    def test_wrong_output_used_in_next_step(self):
        s1 = _step("s1", "WRONG", output_var={"V": 7.0})
        s2 = _step("s2", "OK", input_var={"V": {}})
        issues = _check_cot_consistency_heuristic([s1, s2])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].step_id, "s2")

    def test_wrong_output_not_used_in_next_step(self):
        s1 = _step("s1", "WRONG", output_var={"Z": 7.0})
        s2 = _step("s2", "OK", input_var={"V": {}})
        issues = _check_cot_consistency_heuristic([s1, s2])
        self.assertEqual(issues, [])

    def test_empty_steps(self):
        self.assertEqual(_check_cot_consistency_heuristic([]), [])

    def test_issue_is_cot_issue_type(self):
        s1 = _step("s1", "WRONG", output_var={"V": 7.0})
        s2 = _step("s2", "OK", input_var={"V": {}})
        issues = _check_cot_consistency_heuristic([s1, s2])
        self.assertIsInstance(issues[0], CotIssue)


# ---------------------------------------------------------------------------
# _infer_global_error_type
# ---------------------------------------------------------------------------

class TestInferGlobalErrorType(unittest.TestCase):

    def test_e1_hint_preserved(self):
        steps = [_step("s1", "WRONG")]
        self.assertEqual(_infer_global_error_type(steps, 0, "E1"), "E1")

    def test_e4_hint_preserved(self):
        steps = [_step("s1", "WRONG")]
        self.assertEqual(_infer_global_error_type(steps, 0, "E4"), "E4")

    def test_e5_when_prior_step_feeds_fws(self):
        s1 = _step("s1", "OK", output_var={"V": 5.0})
        s2 = _step("s2", "WRONG", input_var={"V": {}})
        # fws_index=1, prior step s1 output V fed into s2 input V
        self.assertEqual(_infer_global_error_type([s1, s2], 1, "E3"), "E5")

    def test_e3_when_fws_is_first_step(self):
        steps = [_step("s1", "WRONG")]
        self.assertEqual(_infer_global_error_type(steps, 0, "E3"), "E3")

    def test_none_fws_returns_e3(self):
        self.assertEqual(_infer_global_error_type([], None, ""), "E3")


# ---------------------------------------------------------------------------
# diagnose_trace — integration
# ---------------------------------------------------------------------------

class TestDiagnoseTrace(unittest.TestCase):

    def test_empty_trace_returns_e6(self):
        trace = _make_trace([])
        fs = _make_formula_set([])
        diag = diagnose_trace(trace, fs)
        self.assertEqual(diag.global_error_type, "E6")

    def test_passing_trace_no_fws(self):
        steps = [_step("s1", "OK"), _step("s2", "OK")]
        trace = _make_trace(steps, status="PASS")
        fs = _make_formula_set(["s1", "s2"])
        diag = diagnose_trace(trace, fs)
        self.assertIsNone(diag.fws_index)
        self.assertIsNone(diag.global_error_type)

    def test_failed_trace_fws_identified(self):
        steps = [_step("s1", "OK"), _step("s2", "WRONG")]
        trace = _make_trace(steps)
        fs = _make_formula_set(["s1", "s2"])
        diag = diagnose_trace(trace, fs)
        self.assertEqual(diag.fws_index, 1)

    def test_step_labels_populated(self):
        steps = [_step("s1", "OK"), _step("s2", "WRONG")]
        trace = _make_trace(steps)
        fs = _make_formula_set(["s1", "s2"])
        diag = diagnose_trace(trace, fs)
        self.assertIn("s1", diag.step_labels)
        self.assertIn("s2", diag.step_labels)
        self.assertEqual(diag.step_labels["s1"], "OK")

    def test_fws_description_mentions_step_id(self):
        steps = [_step("s1", "OK"), _step("s2", "WRONG")]
        trace = _make_trace(steps)
        fs = _make_formula_set(["s1", "s2"])
        diag = diagnose_trace(trace, fs)
        self.assertIn("s2", diag.fws_description)

    def test_repair_hint_populated(self):
        steps = [_step("s1", "WRONG")]
        trace = _make_trace(steps)
        fs = _make_formula_set(["s1"])
        diag = diagnose_trace(trace, fs)
        self.assertTrue(diag.repair_hint)

    def test_e5_detected_for_propagated_error(self):
        s1 = _step("s1", "OK", output_var={"V": 5.0})
        s2 = _step("s2", "WRONG", input_var={"V": {}})
        trace = _make_trace([s1, s2])
        fs = _make_formula_set(["s1", "s2"])
        diag = diagnose_trace(trace, fs)
        self.assertEqual(diag.global_error_type, "E5")

    def test_cot_issues_populated_on_propagation(self):
        s1 = _step("s1", "WRONG", output_var={"V": 7.0})
        s2 = _step("s2", "OK", input_var={"V": {}})
        trace = _make_trace([s1, s2])
        fs = _make_formula_set(["s1", "s2"])
        diag = diagnose_trace(trace, fs)
        self.assertGreater(len(diag.cot_issues), 0)

    def test_returns_diagnosis_object(self):
        trace = _make_trace([_step("s1", "OK")])
        fs = _make_formula_set(["s1"])
        diag = diagnose_trace(trace, fs)
        self.assertIsInstance(diag, DiagnosisObject)

    def test_diagnosis_to_dict_has_required_fields(self):
        trace = _make_trace([_step("s1", "WRONG")])
        fs = _make_formula_set(["s1"])
        diag = diagnose_trace(trace, fs)
        d = diag.to_dict()
        self.assertIn("global_error_type", d)
        self.assertIn("fws_index", d)
        self.assertIn("step_labels", d)
        self.assertIn("repair_hint", d)
        self.assertIn("cot_issues", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
