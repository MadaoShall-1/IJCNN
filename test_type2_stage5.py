"""Tests for Type 2 Stage 5: FWS-Centered Repair."""

import copy
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock

from parser.schemas import ProblemParseObject

from type2.schemas import (
    DiagnosisObject,
    FormulaEntry,
    FormulaSet,
    StepObject,
    TraceObject,
    VSOEntry,
)
from type2.stage5 import (
    extract_stable_prefix,
    repair_trace,
    rollback_vso,
    select_repair_formula,
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


def _entry(fid: str) -> FormulaEntry:
    return FormulaEntry(
        id=fid, topic="circuits", subtopic="ohms_law",
        target_quantities=[], canonical_quantity_names=[],
        text="", formula="", sympy_expr="", tool_dispatch="sympy",
        variables={},
    )


def _make_formula_set(
    step_ids: list, entry_id: str = "CKT-001", path_index: int = 0
) -> FormulaSet:
    e = _entry(entry_id)
    return FormulaSet(
        formulas={sid: e for sid in step_ids},
        retrieval_confidence=0.9,
        path_index=path_index,
    )


def _make_trace(steps: list, status: str = "FAIL",
                vso_snapshots: dict | None = None) -> TraceObject:
    t = TraceObject(problem_id="test-001", formula_path_index=0)
    t.steps = steps
    t.trace_status = status
    t.vso_snapshots = vso_snapshots or {}
    return t


def _diagnosis(fws_index: int | None, error_type: str | None = "E3") -> DiagnosisObject:
    d = DiagnosisObject()
    d.fws_index = fws_index
    d.global_error_type = error_type
    return d


def _make_vso_snapshot(**kwargs):
    snap = {}
    for name, val in kwargs.items():
        snap[name] = {"value": val, "unit_symbol": "", "unit_name": "",
                      "defined_at": "test", "updated_at": "test"}
    return snap


# ---------------------------------------------------------------------------
# extract_stable_prefix
# ---------------------------------------------------------------------------

class TestExtractStablePrefix(unittest.TestCase):

    def test_no_fws_returns_all_steps(self):
        steps = [_step("s1"), _step("s2")]
        trace = _make_trace(steps, status="PASS")
        diag = _diagnosis(None)
        prefix = extract_stable_prefix(trace, diag)
        self.assertEqual(len(prefix), 2)

    def test_fws_at_index_1_returns_one_step(self):
        steps = [_step("s1"), _step("s2", "WRONG")]
        trace = _make_trace(steps)
        diag = _diagnosis(1)
        prefix = extract_stable_prefix(trace, diag)
        self.assertEqual(len(prefix), 1)
        self.assertEqual(prefix[0].step_id, "s1")

    def test_fws_at_index_0_returns_empty(self):
        steps = [_step("s1", "WRONG"), _step("s2")]
        trace = _make_trace(steps)
        diag = _diagnosis(0)
        prefix = extract_stable_prefix(trace, diag)
        self.assertEqual(prefix, [])

    def test_prefix_is_deep_copy(self):
        steps = [_step("s1"), _step("s2", "WRONG")]
        trace = _make_trace(steps)
        diag = _diagnosis(1)
        prefix = extract_stable_prefix(trace, diag)
        prefix[0].goal = "modified"
        # Original trace should be unchanged
        self.assertNotEqual(trace.steps[0].goal, "modified")


# ---------------------------------------------------------------------------
# rollback_vso
# ---------------------------------------------------------------------------

class TestRollbackVso(unittest.TestCase):

    def test_rollback_to_predecessor_snapshot(self):
        steps = [_step("s1"), _step("s2", "WRONG")]
        trace = _make_trace(steps, vso_snapshots={
            "s1": _make_vso_snapshot(V=5.0, R=10.0)
        })
        vso = rollback_vso(trace, "s2")
        self.assertIn("V", vso)
        self.assertAlmostEqual(vso["V"].value, 5.0)
        self.assertIn("R", vso)

    def test_fws_at_first_step_returns_empty(self):
        steps = [_step("s1", "WRONG")]
        trace = _make_trace(steps, vso_snapshots={})
        vso = rollback_vso(trace, "s1")
        self.assertEqual(vso, {})

    def test_unknown_fws_returns_empty(self):
        steps = [_step("s1")]
        trace = _make_trace(steps, vso_snapshots={})
        vso = rollback_vso(trace, "s_nonexistent")
        self.assertEqual(vso, {})

    def test_returns_vso_entries(self):
        steps = [_step("s1"), _step("s2", "WRONG")]
        trace = _make_trace(steps, vso_snapshots={
            "s1": _make_vso_snapshot(I=2.0)
        })
        vso = rollback_vso(trace, "s2")
        for v in vso.values():
            self.assertIsInstance(v, VSOEntry)

    def test_missing_predecessor_snapshot_returns_empty(self):
        steps = [_step("s1"), _step("s2", "WRONG")]
        trace = _make_trace(steps, vso_snapshots={})
        vso = rollback_vso(trace, "s2")
        self.assertEqual(vso, {})


# ---------------------------------------------------------------------------
# select_repair_formula
# ---------------------------------------------------------------------------

class TestSelectRepairFormula(unittest.TestCase):

    def test_non_e1_error_returns_none(self):
        fs = _make_formula_set(["s1"])
        diag = _diagnosis(0, "E3")
        result = select_repair_formula(fs, "s1", diag, all_formula_sets=[fs])
        self.assertIsNone(result)

    def test_e1_no_beam_returns_none(self):
        fs = _make_formula_set(["s1"])
        diag = _diagnosis(0, "E1")
        result = select_repair_formula(fs, "s1", diag, all_formula_sets=None)
        self.assertIsNone(result)

    def test_e1_returns_alternative_formula(self):
        fs1 = _make_formula_set(["s1"], entry_id="CKT-001", path_index=0)
        fs2 = _make_formula_set(["s1"], entry_id="CKT-002", path_index=1)
        diag = _diagnosis(0, "E1")
        result = select_repair_formula(fs1, "s1", diag, all_formula_sets=[fs1, fs2])
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "CKT-002")

    def test_e1_no_different_formula_returns_none(self):
        # Both formula sets have the same formula for s1
        fs1 = _make_formula_set(["s1"], entry_id="CKT-001")
        fs2 = _make_formula_set(["s1"], entry_id="CKT-001")
        diag = _diagnosis(0, "E1")
        result = select_repair_formula(fs1, "s1", diag, all_formula_sets=[fs1, fs2])
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# repair_trace
# ---------------------------------------------------------------------------

class TestRepairTrace(unittest.TestCase):

    def _mock_solver_pass(self):
        """Returns a solver whose forward() always returns a passing trace."""
        solver = MagicMock()
        def fake_forward(parse_obj, formula_set, problem_id, step_retry_limit):
            t = TraceObject(problem_id=problem_id, formula_path_index=0)
            t.steps = [_step("s2", "OK", intermediate_answer="3.0 A")]
            t.vso = {}
            t.vso_snapshots = {}
            t.final_answer = "3.0 A"
            t.trace_status = "PASS"
            return t
        solver.forward.side_effect = fake_forward
        return solver

    def _mock_solver_fail(self):
        solver = MagicMock()
        def fake_forward(parse_obj, formula_set, problem_id, step_retry_limit):
            t = TraceObject(problem_id=problem_id, formula_path_index=0)
            t.steps = [_step("s2", "WRONG")]
            t.vso = {}
            t.vso_snapshots = {}
            t.final_answer = ""
            t.trace_status = "FAIL"
            return t
        solver.forward.side_effect = fake_forward
        return solver

    def _make_parse_obj(self, step_ids):
        return ProblemParseObject(
            problem_text="test problem",
            domains=["circuits"],
            sub_domains=[],
            known_quantities={"V": {"value": 12.0, "unit_symbol": "V", "unit_name": "volts"}},
            step_plan=[{"step_id": sid, "type": "formula_application",
                        "goal": f"step {sid}", "input_var": {}, "output_var": {}}
                       for sid in step_ids],
            vso={},
        )

    def test_no_solver_returns_fail(self):
        trace = _make_trace([_step("s1", "WRONG")])
        fs = _make_formula_set(["s1"])
        parse_obj = self._make_parse_obj(["s1"])
        diag = _diagnosis(0)
        result = repair_trace(trace, fs, parse_obj, diag, solver=None)
        self.assertEqual(result.trace_status, "FAIL")

    def test_no_fws_returns_pass(self):
        trace = _make_trace([_step("s1", "OK")], status="PASS")
        fs = _make_formula_set(["s1"])
        parse_obj = self._make_parse_obj(["s1"])
        diag = _diagnosis(None)
        result = repair_trace(trace, fs, parse_obj, diag, solver=MagicMock())
        self.assertEqual(result.trace_status, "PASS")

    def test_successful_repair_returns_repaired(self):
        steps = [_step("s1", "OK"), _step("s2", "WRONG")]
        trace = _make_trace(steps, vso_snapshots={"s1": _make_vso_snapshot(V=12.0)})
        fs = _make_formula_set(["s1", "s2"])
        parse_obj = self._make_parse_obj(["s1", "s2"])
        diag = _diagnosis(1, "E3")
        result = repair_trace(trace, fs, parse_obj, diag, solver=self._mock_solver_pass())
        self.assertEqual(result.trace_status, "REPAIRED")

    def test_failed_repair_returns_fail(self):
        steps = [_step("s1", "OK"), _step("s2", "WRONG")]
        trace = _make_trace(steps, vso_snapshots={"s1": _make_vso_snapshot(V=12.0)})
        fs = _make_formula_set(["s1", "s2"])
        parse_obj = self._make_parse_obj(["s1", "s2"])
        diag = _diagnosis(1, "E3")
        result = repair_trace(trace, fs, parse_obj, diag, solver=self._mock_solver_fail())
        self.assertEqual(result.trace_status, "FAIL")

    def test_stable_prefix_preserved(self):
        steps = [_step("s1", "OK", intermediate_answer="10.0 Ω"),
                 _step("s2", "WRONG")]
        trace = _make_trace(steps, vso_snapshots={"s1": _make_vso_snapshot(R=10.0)})
        fs = _make_formula_set(["s1", "s2"])
        parse_obj = self._make_parse_obj(["s1", "s2"])
        diag = _diagnosis(1, "E3")
        result = repair_trace(trace, fs, parse_obj, diag, solver=self._mock_solver_pass())
        # First step should be the stable prefix step s1
        self.assertEqual(result.steps[0].step_id, "s1")
        self.assertEqual(result.steps[0].intermediate_answer, "10.0 Ω")

    def test_repaired_final_answer_from_suffix(self):
        steps = [_step("s1", "OK"), _step("s2", "WRONG")]
        trace = _make_trace(steps, vso_snapshots={"s1": _make_vso_snapshot(V=12.0)})
        fs = _make_formula_set(["s1", "s2"])
        parse_obj = self._make_parse_obj(["s1", "s2"])
        diag = _diagnosis(1, "E3")
        result = repair_trace(trace, fs, parse_obj, diag, solver=self._mock_solver_pass())
        self.assertEqual(result.final_answer, "3.0 A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
