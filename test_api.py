"""Tests for api.py: router dispatch, _dict_to_parse_obj, and _run_type2 stub."""

import time
import unittest
from unittest.mock import MagicMock, patch

from parser.schemas import ProblemParseObject
from router import detect_query_type
from api import _dict_to_parse_obj


# ---------------------------------------------------------------------------
# detect_query_type (router.py)
# ---------------------------------------------------------------------------

class TestDetectQueryType(unittest.TestCase):

    def test_explicit_type1(self):
        self.assertEqual(detect_query_type({"query_type": "type1"}), "type1")

    def test_explicit_type2(self):
        self.assertEqual(detect_query_type({"query_type": "type2"}), "type2")

    def test_premises_nl_implies_type1(self):
        self.assertEqual(detect_query_type({"premises-NL": ["P1"]}), "type1")

    def test_premises_implies_type1(self):
        self.assertEqual(detect_query_type({"premises": ["P1"]}), "type1")

    def test_default_is_type2(self):
        self.assertEqual(detect_query_type({"question": "Find the current."}), "type2")

    def test_empty_payload_is_type2(self):
        self.assertEqual(detect_query_type({}), "type2")

    def test_case_insensitive_type1(self):
        self.assertEqual(detect_query_type({"query_type": "TYPE1"}), "type1")


# ---------------------------------------------------------------------------
# _dict_to_parse_obj
# ---------------------------------------------------------------------------

class TestDictToParseObj(unittest.TestCase):

    def _minimal_dict(self, **kwargs):
        base = {
            "problem_text": "Find the current.",
            "domains": ["circuits"],
            "sub_domains": [],
            "domain_confidence": 0.8,
            "known_quantities": {"R": {"value": 10.0, "unit_symbol": "Ω"}},
            "conditions": [],
            "relations": [],
            "unknown_quantity": "I",
            "unknown_unit": "A",
            "step_plan": [],
            "plan_confidence": 0.7,
            "parser_warnings": [],
            "vso": {},
            "metadata": {},
        }
        base.update(kwargs)
        return base

    def test_returns_parse_object(self):
        obj = _dict_to_parse_obj(self._minimal_dict())
        self.assertIsInstance(obj, ProblemParseObject)

    def test_problem_text_preserved(self):
        obj = _dict_to_parse_obj(self._minimal_dict(problem_text="Test problem."))
        self.assertEqual(obj.problem_text, "Test problem.")

    def test_domains_preserved(self):
        obj = _dict_to_parse_obj(self._minimal_dict(domains=["circuits"]))
        self.assertEqual(obj.domains, ["circuits"])

    def test_known_quantities_preserved(self):
        kq = {"V": {"value": 5.0, "unit_symbol": "V"}}
        obj = _dict_to_parse_obj(self._minimal_dict(known_quantities=kq))
        self.assertIn("V", obj.known_quantities)

    def test_step_plan_preserved(self):
        plan = [{"step_id": "s1", "type": "setup", "goal": "extract", "input_var": {}, "output_var": {}}]
        obj = _dict_to_parse_obj(self._minimal_dict(step_plan=plan))
        self.assertEqual(len(obj.step_plan), 1)

    def test_missing_keys_use_defaults(self):
        # Minimal dict — no known_quantities, step_plan, etc.
        obj = _dict_to_parse_obj({"problem_text": "bare"})
        self.assertEqual(obj.domains, [])
        self.assertEqual(obj.step_plan, [])
        self.assertEqual(obj.known_quantities, {})

    def test_vso_preserved(self):
        vso = {"I": {"value": 2.0, "unit_symbol": "A", "unit_name": "amperes",
                     "defined_at": "stage0", "updated_at": "stage0"}}
        obj = _dict_to_parse_obj(self._minimal_dict(vso=vso))
        self.assertIn("I", obj.vso)


# ---------------------------------------------------------------------------
# _run_type2 — with mocked retriever and stub solver
# ---------------------------------------------------------------------------

class TestRunType2(unittest.TestCase):

    def _make_formula_set(self):
        from type2.schemas import FormulaEntry, FormulaSet
        entry = FormulaEntry(
            id="CKT-001", topic="circuits", subtopic="ohms_law",
            target_quantities=["V", "I", "R"],
            canonical_quantity_names=["electric_potential", "electric_current", "resistance"],
            text="V=IR", formula="V=I*R", sympy_expr="Eq(V, I*R)",
            tool_dispatch="sympy", variables={},
            premise_text="Ohm's Law", fol_axiom="",
        )
        return FormulaSet(formulas={"s1": entry}, retrieval_confidence=0.9, path_index=0)

    def _make_passing_trace(self, problem_id="test"):
        from type2.schemas import StepObject, TraceObject
        t = TraceObject(problem_id=problem_id, formula_path_index=0)
        s = StepObject(step_id="s1", goal="find I", type="formula_application")
        s.status = "OK"
        s.intermediate_answer = "0.5 A"
        s.confidence = 0.9
        t.steps = [s]
        t.final_answer = "0.5 A"
        t.trace_status = "PASS"
        t.vso = {}
        t.vso_snapshots = {}
        return t

    def test_returns_answer_on_pass(self):
        import api as _api
        fs = self._make_formula_set()
        trace = self._make_passing_trace()

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [fs]

        mock_solver = MagicMock()
        mock_solver.forward.return_value = trace

        with patch.object(_api, "_retriever", mock_retriever), \
             patch.object(_api, "_solve_trace", mock_solver):
            from config import SolverConfig
            result = _api._run_type2(
                {"question": "A 10 Ω resistor has 5 V. Find I.", "id": "q1"},
                SolverConfig(),
                t_start=time.monotonic(),
            )

        self.assertEqual(result.get("answer"), "0.5 A")
        self.assertIn("confidence", result)
        self.assertIn("stage0_cache_hit", result)

    def test_empty_problem_text_returns_error(self):
        import api as _api
        from config import SolverConfig
        result = _api._run_type2({"question": ""}, SolverConfig(), t_start=time.monotonic())
        self.assertIn("error", result)
        self.assertEqual(result["answer"], "")

    def test_no_formula_sets_returns_error(self):
        import api as _api
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        with patch.object(_api, "_retriever", mock_retriever):
            from config import SolverConfig
            result = _api._run_type2(
                {"question": "A 10 Ω resistor has 5 V. Find I."},
                SolverConfig(),
                t_start=time.monotonic(),
            )
        self.assertIn("error", result)

    def test_stage0_llm_fallback_config_passed_to_parser(self):
        import api as _api
        from config import SolverConfig

        mock_parse = MagicMock(return_value={
            "problem_text": "A hard physics problem.",
            "domains": ["electrostatics"],
            "sub_domains": [],
            "domain_confidence": 0.5,
            "known_quantities": {},
            "conditions": [],
            "relations": [],
            "unknown_quantity": "F",
            "unknown_unit": "N",
            "step_plan": [],
            "plan_confidence": 0.0,
            "parser_warnings": [],
            "vso": {},
            "metadata": {},
        })
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        with patch.object(_api, "_parse_stage0", mock_parse), \
             patch.object(_api, "_retriever", mock_retriever):
            _api._run_type2(
                {"question": "A hard physics problem.", "id": "q-llm"},
                SolverConfig(stage0_use_llm_fallback=True),
                t_start=time.monotonic(),
            )

        mock_parse.assert_called_once_with(
            "A hard physics problem.",
            use_llm_fallback=True,
        )

    def test_stage0_cache_hit_skips_parser(self):
        import api as _api
        from config import SolverConfig

        cached_parse = {
            "problem_text": "Cached physics problem.",
            "domains": ["circuits"],
            "sub_domains": [],
            "domain_confidence": 0.8,
            "known_quantities": {},
            "conditions": [],
            "relations": [],
            "unknown_quantity": "I",
            "unknown_unit": "A",
            "step_plan": [],
            "plan_confidence": 0.8,
            "parser_warnings": [],
            "vso": {},
            "metadata": {},
        }
        mock_parse = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        with patch.object(_api, "_stage0_cache", None), \
             patch.object(_api, "_stage0_cache_path", None), \
             patch.object(_api, "_load_stage0_cache", return_value={
                 "by_id": {"cached-id": cached_parse},
                 "by_text": {},
             }), \
             patch.object(_api, "_parse_stage0", mock_parse), \
             patch.object(_api, "_retriever", mock_retriever):
            result = _api._run_type2(
                {"question": "Cached physics problem.", "id": "cached-id"},
                SolverConfig(stage0_cache_enabled=True),
                t_start=time.monotonic(),
            )

        mock_parse.assert_not_called()
        self.assertIn("error", result)
        parsed_obj = mock_retriever.retrieve.call_args.args[0]
        self.assertTrue(parsed_obj.metadata["stage0_cache_hit"])
        self.assertEqual(parsed_obj.metadata["stage0_cache_key"], "id")

    def test_stage0_cache_can_be_disabled_per_request(self):
        import api as _api
        from config import SolverConfig

        live_parse = {
            "problem_text": "Live physics problem.",
            "domains": ["circuits"],
            "sub_domains": [],
            "domain_confidence": 0.8,
            "known_quantities": {},
            "conditions": [],
            "relations": [],
            "unknown_quantity": "I",
            "unknown_unit": "A",
            "step_plan": [],
            "plan_confidence": 0.8,
            "parser_warnings": [],
            "vso": {},
            "metadata": {},
        }
        mock_parse = MagicMock(return_value=live_parse)
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        with patch.object(_api, "_load_stage0_cache") as mock_cache, \
             patch.object(_api, "_parse_stage0", mock_parse), \
             patch.object(_api, "_retriever", mock_retriever):
            _api._run_type2(
                {
                    "question": "Live physics problem.",
                    "id": "live-id",
                    "use_stage0_cache": False,
                },
                SolverConfig(stage0_cache_enabled=True),
                t_start=time.monotonic(),
            )

        mock_cache.assert_not_called()
        mock_parse.assert_called_once()

    def test_no_solver_returns_fail_trace(self):
        import api as _api
        fs = self._make_formula_set()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [fs]
        with patch.object(_api, "_retriever", mock_retriever), \
             patch.object(_api, "_solve_trace", None):
            from config import SolverConfig
            result = _api._run_type2(
                {"question": "A 10 Ω resistor has 5 V. Find I."},
                SolverConfig(),
                t_start=time.monotonic(),
            )
        # Should not crash; returns a response dict
        self.assertIn("answer", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
