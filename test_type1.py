"""Type 1 pipeline smoke tests.

Exercises the deterministic parsing layer (no LLM required) and verifies that
the pipeline orchestrator wires together correctly.  DSPy modules are mocked
so this file runs without a vLLM instance.

Run:
    python test_type1.py
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from config import SolverConfig
from router import detect_query_type
from type1.parser import detect_question_format, detect_solver_route, parse_type1
from type1.schemas import QuestionFormat, SolverRoute


# ---------------------------------------------------------------------------
# Sample payloads matching the contest_details.txt schema
# ---------------------------------------------------------------------------

PAYLOAD_MCQ = {
    "query_type": "type1",
    "premises-NL": [
        "If a curriculum is well-structured and has exercises, it enhances student engagement.",
        "If a curriculum enhances student engagement and provides access to advanced resources, "
        "it enhances critical thinking.",
        "If a faculty prioritizes pedagogical training and curriculum development, "
        "the curriculum is well-structured.",
        "The faculty prioritizes pedagogical training and curriculum development.",
        "The curriculum has practical exercises.",
        "The curriculum provides access to advanced resources.",
    ],
    "questions": [
        "Based on the premises, what can we conclude about the curriculum?\n"
        "A. It enhances student engagement but not critical thinking\n"
        "B. It enhances critical thinking\n"
        "C. It needs more resources to enhance critical thinking\n"
        "D. It is well-structured but lacks exercises",
    ],
}

PAYLOAD_YES_NO_SYMBOLIC = {
    "premises-NL": [
        "All mammals are warm-blooded.",
        "All warm-blooded animals have a four-chambered heart.",
        "Whales are mammals.",
    ],
    "questions": [
        "Does the combination of faculty priorities and curriculum features lead to "
        "enhanced critical thinking?"
    ],
}

PAYLOAD_YES_NO_NUMERIC = {
    "premises-NL": [
        "If a student scores >= 80 on the final exam, they pass the course.",
        "Maria scored 85 on the final exam.",
    ],
    "questions": [
        "Can we conclude that Maria passes the course?",
    ],
}

PAYLOAD_OPEN_ENDED = {
    "premises-NL": [
        "Regulation 13: A student with 0 lab points cannot pass the course.",
        "The student was absent for the lab exam and received 0 lab points.",
    ],
    "questions": [
        "Based on the regulations, what is the student's course outcome?",
    ],
}

PAYLOAD_MULTI_QUESTION = {
    "premises-NL": [
        "All birds can fly.",
        "Penguins are birds.",
        "Penguins cannot fly.",
    ],
    "questions": [
        "Can we conclude that penguins can fly?",
        "Which of the following best describes penguins?\n"
        "A. They can fly\n"
        "B. They cannot fly\n"
        "C. They are not birds\n"
        "D. Uncertain",
    ],
}

PAYLOAD_MISSING_PREMISES = {
    "questions": ["Can we conclude that X implies Y?"],
}

PAYLOAD_ALT_FIELD_NAME = {
    "premises": ["All X are Y.", "Z is an X."],
    "question": "Can we conclude that Z is a Y?",
}

# Type 2 payload — should NOT be routed to Type 1
PAYLOAD_TYPE2 = {
    "question": "Calculate the equivalent resistance of three 6Ω resistors in parallel.",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRouter(unittest.TestCase):
    def test_explicit_type1(self):
        self.assertEqual(detect_query_type({"query_type": "type1", "x": 1}), "type1")

    def test_explicit_type2(self):
        self.assertEqual(detect_query_type({"query_type": "type2"}), "type2")

    def test_premises_nl_heuristic(self):
        self.assertEqual(detect_query_type(PAYLOAD_MCQ), "type1")

    def test_premises_alt_name(self):
        self.assertEqual(detect_query_type(PAYLOAD_ALT_FIELD_NAME), "type1")

    def test_default_type2(self):
        self.assertEqual(detect_query_type(PAYLOAD_TYPE2), "type2")

    def test_empty_payload(self):
        self.assertEqual(detect_query_type({}), "type2")


class TestQuestionFormatDetection(unittest.TestCase):
    def test_mcq_labeled_options(self):
        fmt, opts = detect_question_format(PAYLOAD_MCQ["questions"][0])
        self.assertEqual(fmt, QuestionFormat.MCQ)
        self.assertIn("A", opts)
        self.assertIn("B", opts)

    def test_yes_no_can_we_conclude(self):
        fmt, _ = detect_question_format("Can we conclude that Maria passes?")
        self.assertEqual(fmt, QuestionFormat.YES_NO_UNCERTAIN)

    def test_yes_no_does_imply(self):
        fmt, _ = detect_question_format("Does this combination imply enhanced thinking?")
        self.assertEqual(fmt, QuestionFormat.YES_NO_UNCERTAIN)

    def test_yes_no_is_it_true(self):
        fmt, _ = detect_question_format("Is it true that all mammals are warm-blooded?")
        self.assertEqual(fmt, QuestionFormat.YES_NO_UNCERTAIN)

    def test_yes_no_follows_from(self):
        fmt, _ = detect_question_format("Does it follow from the premises that Z is a Y?")
        self.assertEqual(fmt, QuestionFormat.YES_NO_UNCERTAIN)

    def test_open_ended(self):
        fmt, _ = detect_question_format(
            "Based on the regulations, what is the student's course outcome?"
        )
        self.assertEqual(fmt, QuestionFormat.OPEN_ENDED)

    def test_mcq_which_of_the_following(self):
        fmt, _ = detect_question_format(
            "Which of the following best describes the curriculum?"
        )
        self.assertEqual(fmt, QuestionFormat.MCQ)


class TestSolverRouteDetection(unittest.TestCase):
    def test_yes_no_symbolic_prover9(self):
        # No digits or comparison operators → Prover9
        route = detect_solver_route(
            ["All mammals are warm-blooded.", "Whales are mammals."],
            QuestionFormat.YES_NO_UNCERTAIN,
        )
        self.assertEqual(route, SolverRoute.PROVER9)

    def test_yes_no_numeric_z3(self):
        # Digits present → Z3
        route = detect_solver_route(
            ["If a student scores >= 80, they pass."],
            QuestionFormat.YES_NO_UNCERTAIN,
        )
        self.assertEqual(route, SolverRoute.Z3)

    def test_yes_no_inequality_z3(self):
        route = detect_solver_route(
            ["Students must complete > 50% of assignments to pass."],
            QuestionFormat.YES_NO_UNCERTAIN,
        )
        self.assertEqual(route, SolverRoute.Z3)

    def test_mcq_always_llm(self):
        route = detect_solver_route(
            ["All X are Y."],
            QuestionFormat.MCQ,
        )
        self.assertEqual(route, SolverRoute.LLM)

    def test_open_ended_always_llm(self):
        route = detect_solver_route(
            ["Any premise with >= 100 marks is sufficient."],
            QuestionFormat.OPEN_ENDED,
        )
        self.assertEqual(route, SolverRoute.LLM)


class TestParseType1(unittest.TestCase):
    def test_mcq_payload(self):
        obj = parse_type1(PAYLOAD_MCQ)
        self.assertEqual(len(obj.questions), 1)
        self.assertEqual(obj.questions[0].format, QuestionFormat.MCQ)
        self.assertEqual(obj.questions[0].solver_route, SolverRoute.LLM)
        self.assertEqual(len(obj.premises_nl), 6)
        self.assertEqual(obj.metadata["question_count"], 1)

    def test_yes_no_symbolic_payload(self):
        obj = parse_type1(PAYLOAD_YES_NO_SYMBOLIC)
        self.assertEqual(obj.questions[0].format, QuestionFormat.YES_NO_UNCERTAIN)
        self.assertEqual(obj.questions[0].solver_route, SolverRoute.PROVER9)

    def test_yes_no_numeric_payload(self):
        obj = parse_type1(PAYLOAD_YES_NO_NUMERIC)
        self.assertEqual(obj.questions[0].format, QuestionFormat.YES_NO_UNCERTAIN)
        self.assertEqual(obj.questions[0].solver_route, SolverRoute.Z3)

    def test_open_ended_payload(self):
        obj = parse_type1(PAYLOAD_OPEN_ENDED)
        self.assertEqual(obj.questions[0].format, QuestionFormat.OPEN_ENDED)
        self.assertEqual(obj.questions[0].solver_route, SolverRoute.LLM)

    def test_multi_question_payload(self):
        obj = parse_type1(PAYLOAD_MULTI_QUESTION)
        self.assertEqual(len(obj.questions), 2)
        self.assertEqual(obj.questions[0].format, QuestionFormat.YES_NO_UNCERTAIN)
        self.assertEqual(obj.questions[1].format, QuestionFormat.MCQ)
        self.assertEqual(obj.metadata["question_count"], 2)

    def test_missing_premises_warning(self):
        obj = parse_type1(PAYLOAD_MISSING_PREMISES)
        self.assertTrue(any("premises" in w.lower() for w in obj.parse_warnings))
        self.assertEqual(len(obj.premises_nl), 0)

    def test_alt_field_names(self):
        obj = parse_type1(PAYLOAD_ALT_FIELD_NAME)
        self.assertEqual(len(obj.premises_nl), 2)
        self.assertEqual(len(obj.questions), 1)

    def test_serialisation(self):
        obj = parse_type1(PAYLOAD_MCQ)
        d = obj.to_dict()
        self.assertIn("premises_nl", d)
        self.assertIn("questions", d)
        self.assertEqual(d["questions"][0]["format"], "mcq")
        # Should round-trip through JSON without error
        json.dumps(d)


class TestSolverConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SolverConfig()
        self.assertEqual(cfg.beam_n, 3)
        self.assertEqual(cfg.abstain_behavior, "best_effort")
        self.assertTrue(cfg.generate_fol)

    def test_from_dict_partial(self):
        cfg = SolverConfig.from_dict({"beam_n": 1, "generate_fol": False})
        self.assertEqual(cfg.beam_n, 1)
        self.assertFalse(cfg.generate_fol)
        self.assertTrue(cfg.generate_cot)  # unchanged default

    def test_from_dict_unknown_keys_ignored(self):
        cfg = SolverConfig.from_dict({"nonexistent_key": "whatever"})
        self.assertIsInstance(cfg, SolverConfig)

    def test_tier_boundaries(self):
        cfg = SolverConfig()
        self.assertEqual(cfg.tier(0.0), 0)
        self.assertEqual(cfg.tier(11.9), 0)
        self.assertEqual(cfg.tier(12.0), 1)
        self.assertEqual(cfg.tier(34.9), 1)
        self.assertEqual(cfg.tier(35.0), 2)
        self.assertEqual(cfg.tier(54.9), 2)
        self.assertEqual(cfg.tier(55.0), 3)

    def test_optional_fields_enabled(self):
        cfg = SolverConfig()
        self.assertTrue(cfg.optional_fields_enabled(0.0))
        self.assertTrue(cfg.optional_fields_enabled(11.0))
        self.assertFalse(cfg.optional_fields_enabled(35.0))


class TestZ3Validation(unittest.TestCase):
    """Tests for validate_z3_code — no z3 install required."""

    def setUp(self):
        from type1.z3_solver import validate_z3_code
        self.validate = validate_z3_code

    # --- Valid code ---

    def test_valid_dual_solver_code(self):
        code = """
from z3 import *
x = Bool('x')
s1 = Solver(); s1.set(timeout=5000)
s1.add(x)
s1.add(Not(x))
result_negated = s1.check()
s2 = Solver(); s2.set(timeout=5000)
s2.add(x)
result_positive = s2.check()
"""
        ok, err = self.validate(code)
        self.assertTrue(ok, err)

    # --- Syntax errors ---

    def test_syntax_error(self):
        ok, err = self.validate("from z3 import *\ndef bad(:\n  pass")
        self.assertFalse(ok)
        self.assertIn("SyntaxError", err)

    # --- Missing structural elements ---

    def test_missing_result_negated(self):
        code = """
from z3 import *
x = Bool('x')
s1 = Solver()
s1.add(Not(x))
s2 = Solver()
result_positive = s2.check()
"""
        ok, err = self.validate(code)
        self.assertFalse(ok)
        self.assertIn("result_negated", err)

    def test_missing_result_positive(self):
        code = """
from z3 import *
x = Bool('x')
s1 = Solver()
s1.add(Not(x))
result_negated = s1.check()
"""
        ok, err = self.validate(code)
        self.assertFalse(ok)
        self.assertIn("result_positive", err)

    def test_missing_not_call(self):
        code = """
from z3 import *
x = Bool('x')
s1 = Solver()
s1.add(x)
result_negated = s1.check()
s2 = Solver()
result_positive = s2.check()
"""
        ok, err = self.validate(code)
        self.assertFalse(ok)
        self.assertIn("Not(", err)

    def test_missing_solver(self):
        code = """
from z3 import *
result_negated = unsat
result_positive = sat
x = Not(True)
"""
        ok, err = self.validate(code)
        self.assertFalse(ok)
        self.assertIn("Solver()", err)


class TestZ3ResultMapping(unittest.TestCase):
    """Tests for _map_z3_results — requires z3-solver to be installed."""

    @classmethod
    def setUpClass(cls):
        try:
            import z3  # noqa: F401
            cls._z3_available = True
        except ImportError:
            cls._z3_available = False

    def _skip_if_no_z3(self):
        if not self._z3_available:
            self.skipTest("z3-solver not installed")

    def test_yes_when_negated_unsat(self):
        self._skip_if_no_z3()
        from type1.z3_solver import execute_z3_code
        # Premises: x is True. Conclusion: x. Negation: Not(x) → UNSAT → Yes
        code = """
from z3 import *
x = Bool('x')
premises = [x == True]
conclusion = x
s1 = Solver(); s1.set(timeout=5000)
for p in premises: s1.add(p)
s1.add(Not(conclusion))
result_negated = s1.check()
s2 = Solver(); s2.set(timeout=5000)
for p in premises: s2.add(p)
s2.add(conclusion)
result_positive = s2.check()
"""
        answer, _ = execute_z3_code(code)
        self.assertEqual(answer, "Yes")

    def test_no_when_conclusion_unsat(self):
        self._skip_if_no_z3()
        from type1.z3_solver import execute_z3_code
        # Premises: x is True. Conclusion: Not(x). conclusion itself → UNSAT → No
        code = """
from z3 import *
x = Bool('x')
premises = [x == True]
conclusion = Not(x)
s1 = Solver(); s1.set(timeout=5000)
for p in premises: s1.add(p)
s1.add(Not(conclusion))
result_negated = s1.check()
s2 = Solver(); s2.set(timeout=5000)
for p in premises: s2.add(p)
s2.add(conclusion)
result_positive = s2.check()
"""
        answer, _ = execute_z3_code(code)
        self.assertEqual(answer, "No")

    def test_uncertain_when_both_sat(self):
        self._skip_if_no_z3()
        from type1.z3_solver import execute_z3_code
        # Premises: empty. Conclusion: x (free variable). Both SAT → Uncertain
        code = """
from z3 import *
x = Bool('x')
conclusion = x
s1 = Solver(); s1.set(timeout=5000)
s1.add(Not(conclusion))
result_negated = s1.check()
s2 = Solver(); s2.set(timeout=5000)
s2.add(conclusion)
result_positive = s2.check()
"""
        answer, _ = execute_z3_code(code)
        self.assertEqual(answer, "Uncertain")

    def test_exec_error_returns_unknown(self):
        self._skip_if_no_z3()
        from type1.z3_solver import execute_z3_code
        code = "result_negated = undefined_name\nresult_positive = also_undefined"
        answer, err = execute_z3_code(code)
        self.assertEqual(answer, "UNKNOWN")
        self.assertIn("exec error", err)

    def test_syntax_error_caught(self):
        from type1.z3_solver import execute_z3_code
        # Invalid Python — exec will raise SyntaxError inside
        code = "from z3 import *\ndef bad(:\n  result_negated = unsat"
        answer, err = execute_z3_code(code)
        self.assertEqual(answer, "UNKNOWN")


class TestPipelinePhase1(unittest.TestCase):
    """Pipeline tests for Phase 1 additions: Z3 path + verifier pass."""

    def _make_mock_solver(self, answer="B", explanation="Test.", confidence=None):
        from type1.schemas import Type1Response
        mock = MagicMock()
        mock.solve_one.return_value = Type1Response(
            answer=answer,
            explanation=explanation,
            fol=["fol_line"],
            cot=["Step 1.", "Step 2."],
            premises=["Premise 1"],
            confidence=confidence,
        )
        return mock

    def _make_mock_z3(self, answer="Yes", used_z3=True):
        mock = MagicMock()
        mock.return_value = MagicMock(
            answer=answer if used_z3 else "",
            explanation="Z3 formal verification.",
            z3_code="from z3 import *\n# code here\nresult_negated=unsat\nresult_positive=sat",
            used_z3=used_z3,
            confidence=1.0 if used_z3 else None,
        )
        return mock

    def test_z3_path_used_for_yes_no_z3_route(self):
        """Z3 module is called for Yes/No questions with Z3 route."""
        from type1.pipeline import run
        with patch("type1.pipeline._run_z3_path") as mock_z3_path:
            from type1.schemas import Type1Response
            mock_z3_path.return_value = Type1Response(
                answer="Yes",
                explanation="Z3 says Yes.",
                fol=["z3 code"],
                confidence=1.0,
                premises=PAYLOAD_YES_NO_NUMERIC["premises-NL"],
            )
            result = run(PAYLOAD_YES_NO_NUMERIC)
        mock_z3_path.assert_called_once()
        self.assertEqual(result["answer"], "Yes")
        self.assertEqual(result.get("confidence"), 1.0)

    def test_z3_fallback_to_llm_on_none(self):
        """If Z3 path returns None, pipeline falls back to LLM solver."""
        from type1.pipeline import run
        llm_solver = self._make_mock_solver(answer="No")
        with patch("type1.pipeline._run_z3_path", return_value=None):
            result = run(PAYLOAD_YES_NO_NUMERIC, solver=llm_solver)
        self.assertEqual(result["answer"], "No")

    def test_z3_skipped_when_type1_use_z3_false(self):
        """Z3 path is skipped entirely when config.type1_use_z3=False."""
        from type1.pipeline import run
        cfg = SolverConfig(type1_use_z3=False)
        llm_solver = self._make_mock_solver(answer="Uncertain")
        with patch("type1.pipeline._run_z3_path") as mock_z3:
            result = run(PAYLOAD_YES_NO_NUMERIC, config=cfg, solver=llm_solver)
        mock_z3.assert_not_called()
        self.assertEqual(result["answer"], "Uncertain")

    def test_z3_skipped_for_mcq_question(self):
        """Z3 path is not invoked for MCQ questions."""
        from type1.pipeline import run
        with patch("type1.pipeline._run_z3_path") as mock_z3:
            run(PAYLOAD_MCQ, solver=self._make_mock_solver())
        mock_z3.assert_not_called()

    def test_verifier_called_for_mcq(self):
        """Verifier pass runs for MCQ questions when type1_verify=True."""
        from type1.pipeline import run
        with patch("type1.pipeline._run_verifier_pass") as mock_verify:
            from type1.schemas import Type1Response
            mock_verify.return_value = Type1Response(
                answer="B", explanation="Verified.", confidence=0.9
            )
            run(PAYLOAD_MCQ, solver=self._make_mock_solver())
        mock_verify.assert_called_once()

    def test_verifier_skipped_when_tier1_exceeded(self):
        """Verifier pass is skipped at Tier 1 (≥12 s)."""
        from type1.pipeline import run
        cfg = SolverConfig(timeout_tier1_seconds=0.0)
        with patch("type1.pipeline._run_verifier_pass") as mock_verify:
            run(PAYLOAD_MCQ, config=cfg, solver=self._make_mock_solver())
        mock_verify.assert_not_called()

    def test_verifier_skipped_when_type1_verify_false(self):
        """Verifier pass is skipped when type1_verify=False."""
        from type1.pipeline import run
        cfg = SolverConfig(type1_verify=False)
        with patch("type1.pipeline._run_verifier_pass") as mock_verify:
            run(PAYLOAD_MCQ, config=cfg, solver=self._make_mock_solver())
        mock_verify.assert_not_called()

    def test_verifier_skipped_for_yes_no_question(self):
        """Verifier pass does not run for Yes/No/Uncertain questions."""
        from type1.pipeline import run
        # yes_no route goes to Z3; stub it out so it returns a response
        with patch("type1.pipeline._run_z3_path") as mock_z3, \
             patch("type1.pipeline._run_verifier_pass") as mock_verify:
            from type1.schemas import Type1Response
            mock_z3.return_value = Type1Response(
                answer="Yes", explanation="Z3.", confidence=1.0
            )
            run(PAYLOAD_YES_NO_NUMERIC)
        mock_verify.assert_not_called()

    def test_confidence_propagated_from_z3(self):
        """confidence=1.0 appears in output when Z3 succeeds."""
        from type1.pipeline import run
        cfg = SolverConfig(type1_verify=False)  # isolate Z3 confidence
        with patch("type1.pipeline._run_z3_path") as mock_z3:
            from type1.schemas import Type1Response
            mock_z3.return_value = Type1Response(
                answer="Yes", explanation="Z3.", confidence=1.0,
                premises=PAYLOAD_YES_NO_NUMERIC["premises-NL"],
            )
            result = run(PAYLOAD_YES_NO_NUMERIC, config=cfg)
        self.assertEqual(result.get("confidence"), 1.0)


class TestPipelineWithMockedSolver(unittest.TestCase):
    """Verify the orchestrator wires correctly using a mocked Type1Solver."""

    def _make_mock_solver(self, answer="B", explanation="Test explanation."):
        from type1.schemas import Type1Response
        mock = MagicMock()
        mock.solve_one.return_value = Type1Response(
            answer=answer,
            explanation=explanation,
            fol=["fol_line"],
            cot=["Step 1: reason.", "Step 2: conclude."],
            premises=["Premise 1"],
            confidence=0.9,
        )
        return mock

    def test_happy_path_full_fields(self):
        from type1.pipeline import run
        result = run(PAYLOAD_MCQ, solver=self._make_mock_solver())
        self.assertEqual(result["answer"], "B")
        self.assertIn("explanation", result)
        self.assertIn("fol", result)
        self.assertIn("cot", result)
        self.assertIn("premises", result)

    def test_tier2_skips_optional_fields(self):
        from type1.pipeline import run
        # Tier 2 triggers at ≥35 s — simulate by setting very low thresholds.
        cfg = SolverConfig(
            timeout_tier2_seconds=0.0,
            latency_budget_seconds=55.0,
        )
        result = run(PAYLOAD_MCQ, config=cfg, solver=self._make_mock_solver())
        self.assertIn("answer", result)
        self.assertIn("explanation", result)
        self.assertNotIn("fol", result)
        self.assertNotIn("cot", result)

    def test_no_questions_returns_error(self):
        from type1.pipeline import run
        result = run({"premises-NL": ["Some premise."]})
        self.assertEqual(result["answer"], "")
        self.assertIn("error", result["explanation"].lower())

    def test_missing_premises_still_runs(self):
        from type1.pipeline import run
        result = run(PAYLOAD_MISSING_PREMISES, solver=self._make_mock_solver())
        self.assertEqual(result["answer"], "B")

    def test_generate_fol_false_omits_fol(self):
        from type1.pipeline import run
        cfg = SolverConfig(generate_fol=False)
        result = run(PAYLOAD_MCQ, config=cfg, solver=self._make_mock_solver())
        self.assertNotIn("fol", result)

    def test_response_is_json_serialisable(self):
        from type1.pipeline import run
        result = run(PAYLOAD_MCQ, solver=self._make_mock_solver())
        json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# Manual demo (run directly)
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Print the parse objects for each sample payload."""
    samples = [
        ("MCQ", PAYLOAD_MCQ),
        ("Yes/No symbolic (Prover9 route)", PAYLOAD_YES_NO_SYMBOLIC),
        ("Yes/No numeric (Z3 route)", PAYLOAD_YES_NO_NUMERIC),
        ("Open-ended", PAYLOAD_OPEN_ENDED),
        ("Multi-question", PAYLOAD_MULTI_QUESTION),
        ("Alt field names", PAYLOAD_ALT_FIELD_NAME),
    ]

    for label, payload in samples:
        print(f"\n{'=' * 60}")
        print(f"Sample: {label}")
        print("=" * 60)
        obj = parse_type1(payload)
        for i, q in enumerate(obj.questions):
            print(f"  Q{i + 1}: [{q.format.value}] route={q.solver_route.value}")
            print(f"       {q.text[:80].strip()!r}")
            if q.mcq_options:
                print(f"       options={list(q.mcq_options.keys())}")
        if obj.parse_warnings:
            print(f"  warnings: {obj.parse_warnings}")
        print(f"  metadata: {json.dumps(obj.metadata, indent=4)}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        unittest.main(argv=[a for a in sys.argv if a != "--demo"])
