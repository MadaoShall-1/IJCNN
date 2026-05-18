"""Z3 formal solver for Type 1 Yes/No/Uncertain questions (Phase 1).

Both SolverRoute.Z3 and SolverRoute.PROVER9 use this module in Phase 1.
Prover9 is deferred to Phase 2 (design §2.1); purely symbolic premises
fall through to Z3, which handles them adequately.

Fallback chain (design §3.1 Step 3a):
  Z3 code fails validation (up to _MAX_ATTEMPTS tries)  → LLM path
  Z3 returns UNKNOWN                                     → LLM path
  z3-solver package not installed                        → LLM path

Design §3.1 result mapping (dual-solver form):
  result_negated == unsat                          → "Yes"
  result_positive == unsat                         → "No"
  result_negated == sat AND result_positive == sat → "Uncertain"
  either == unknown / missing                      → "UNKNOWN" → LLM fallback
"""

from __future__ import annotations

import ast
import logging
import re
from typing import List, Optional, Tuple

try:
    import dspy as _dspy
    _DSPY_AVAILABLE = True
except ModuleNotFoundError:
    _dspy = None  # type: ignore[assignment]
    _DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Z3 availability guard — missing z3-solver never crashes the pipeline
# ---------------------------------------------------------------------------

try:
    import z3 as _z3
    _Z3_AVAILABLE = True
except ImportError:  # pragma: no cover
    _z3 = None  # type: ignore[assignment]
    _Z3_AVAILABLE = False


# ---------------------------------------------------------------------------
# DSPy Signature — LLM generates Z3 code (design §3.1 Step 3a ii)
# Only defined when dspy-ai is installed.
# ---------------------------------------------------------------------------

if _DSPY_AVAILABLE:
    class Z3Formalization(_dspy.Signature):
        """Translate logical premises and a yes/no question into dual-solver Z3 Python code.

        Your output must be complete, executable Python code using the z3 library.

        Required structure (strictly follow this):
        1. ``from z3 import *``
        2. Declare symbolic variables: use ``Bool``, ``Int``, or ``Real`` as appropriate.
        3. Encode each premise as a z3 formula (a Python expression, not a string).
        4. Create TWO solvers with timeouts:
             s1 = Solver(); s1.set(timeout=5000)   # tests Not(conclusion)
             s2 = Solver(); s2.set(timeout=5000)   # tests conclusion directly
        5. Add premises to both solvers.
        6. Add ``Not(conclusion_formula)`` to s1; add ``conclusion_formula`` to s2.
        7. Assign results:
             result_negated  = s1.check()
             result_positive = s2.check()

        Do NOT add any print statements or other side effects.
        """

        premises: list[str] = _dspy.InputField(
            desc="Natural language premises to encode as z3 constraints"
        )
        question: str = _dspy.InputField(
            desc=(
                "A Yes/No/Uncertain question. Extract the conclusion to test from it. "
                "Example: 'Can we conclude that X?' → conclusion is X."
            )
        )
        z3_code: str = _dspy.OutputField(
            desc=(
                "Complete executable Python code. Must contain: variable declarations, "
                "two Solver instances (s1 adds Not(conclusion), s2 adds conclusion), "
                "result_negated = s1.check(), result_positive = s2.check()."
            )
        )


# ---------------------------------------------------------------------------
# Pre-execution structural validation (design §3.1 Step 3a iii)
# ---------------------------------------------------------------------------

def validate_z3_code(code: str) -> Tuple[bool, str]:
    """Check LLM-generated Z3 code before execution.

    Validation rules (design §3.1):
      1. Code is syntactically valid Python.
      2. Both result variables are present (dual-solver structure).
      3. At least one ``Not(`` call exists (negation of conclusion in s1).
      4. At least one ``Solver()`` instantiation exists.

    Returns ``(is_valid, failure_reason)``.  On failure, the pipeline falls
    back to the LLM reasoning path without executing the code.
    """
    # 1. Syntax
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    # 2. Dual result variables
    if "result_negated" not in code:
        return False, "Missing 'result_negated' assignment"
    if "result_positive" not in code:
        return False, "Missing 'result_positive' assignment"

    # 3. Negation
    if "Not(" not in code:
        return False, "Missing Not() — conclusion must be negated in s1"

    # 4. Solver instantiation
    if "Solver()" not in code:
        return False, "Missing Solver() instantiation"

    return True, ""


# ---------------------------------------------------------------------------
# Z3 execution
# ---------------------------------------------------------------------------

def execute_z3_code(code: str) -> Tuple[str, str]:
    """Execute validated Z3 code and map results.

    Returns ``(answer, error_detail)``:
      answer is one of ``"Yes"``, ``"No"``, ``"Uncertain"``, ``"UNKNOWN"``
      error_detail is non-empty only on exception or missing result variable.

    Execution uses a pre-populated namespace containing all public z3 symbols
    so that ``from z3 import *`` in generated code resolves correctly.
    """
    if not _Z3_AVAILABLE:
        return "UNKNOWN", "z3-solver not installed (pip install z3-solver)"

    # Build execution namespace with all public z3 symbols + __builtins__
    exec_ns: dict = {
        name: getattr(_z3, name)
        for name in dir(_z3)
        if not name.startswith("_")
    }
    exec_ns["z3"] = _z3
    exec_ns["__builtins__"] = __builtins__

    try:
        exec(code, exec_ns)  # noqa: S102
    except Exception as exc:
        return "UNKNOWN", f"exec error: {exc}"

    return _map_z3_results(
        exec_ns.get("result_negated"),
        exec_ns.get("result_positive"),
    )


def _map_z3_results(result_neg, result_pos) -> Tuple[str, str]:
    """Map dual-solver outcomes to Yes/No/Uncertain/UNKNOWN (design §3.1 step iv)."""
    if not _Z3_AVAILABLE:
        return "UNKNOWN", "z3-solver not available"

    unsat = _z3.unsat
    sat = _z3.sat

    if result_neg is None or result_pos is None:
        return "UNKNOWN", "result variable(s) not set after exec"

    if result_neg == unsat:
        # Not(conclusion) is unsatisfiable → conclusion must hold
        return "Yes", ""

    if result_pos == unsat:
        # conclusion itself is unsatisfiable → conclusion cannot hold
        return "No", ""

    if result_neg == sat and result_pos == sat:
        # Premises neither prove nor disprove the conclusion
        return "Uncertain", ""

    return "UNKNOWN", f"unexpected Z3 results: negated={result_neg}, positive={result_pos}"


# ---------------------------------------------------------------------------
# DSPy Module (guarded — only when dspy-ai is installed)
# ---------------------------------------------------------------------------

_MAX_ATTEMPTS = 2
_CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

if _DSPY_AVAILABLE:
    class Z3SolverModule(_dspy.Module):
        """Phase 1 formal solver for Yes/No/Uncertain questions using Z3.

        Flow (design §3.1 Step 3a):
          1. LLM (logic-reasoner adapter) generates Z3 Python code.
          2. Pre-execution structural validation; retry once on failure.
          3. Execute with dual-solver; map UNSAT/SAT → Yes/No/Uncertain.
          4. On validation failure or UNKNOWN: return sentinel → pipeline
             falls back to the LLM reasoning path.

        A successful Z3 result yields confidence = 1.0 and uses the Z3 code
        itself as the ``fol`` output (design §3.1 Step 3a v).
        """

        def __init__(self) -> None:
            self.formalizer = _dspy.ChainOfThought(Z3Formalization)

        def forward(
            self,
            premises: List[str],
            question: str,
        ) -> "_dspy.Prediction":
            """Run Z3 formalization + execution.

            Returns a Prediction with:
              answer    – "Yes" / "No" / "Uncertain", or "" on fallback
              z3_code   – generated code (used as the fol field on success)
              used_z3   – True only if Z3 produced a deterministic answer
              confidence – 1.0 on success, None on fallback
            """
            last_code = ""
            last_error = ""

            for attempt in range(_MAX_ATTEMPTS):
                q_text = question
                if attempt > 0 and last_error:
                    q_text = (
                        f"{question}\n\n[Previous attempt was invalid: {last_error}. "
                        "Fix the issues and regenerate the code.]"
                    )

                pred = self.formalizer(premises=premises, question=q_text)
                last_code = _strip_fences(pred.z3_code)

                is_valid, validation_error = validate_z3_code(last_code)
                if not is_valid:
                    last_error = validation_error
                    logger.warning(
                        "Z3 code validation failed (attempt %d/%d): %s",
                        attempt + 1,
                        _MAX_ATTEMPTS,
                        validation_error,
                    )
                    continue

                answer, exec_error = execute_z3_code(last_code)
                if answer != "UNKNOWN":
                    logger.debug("Z3 produced deterministic answer: %s", answer)
                    return _dspy.Prediction(
                        answer=answer,
                        explanation=_build_explanation(answer, len(premises)),
                        z3_code=last_code,
                        used_z3=True,
                        confidence=1.0,
                    )

                logger.warning(
                    "Z3 returned UNKNOWN (attempt %d/%d): %s",
                    attempt + 1,
                    _MAX_ATTEMPTS,
                    exec_error,
                )
                break

            logger.info("Z3 solver falling back to LLM path after %d attempt(s).", attempt + 1)
            return _dspy.Prediction(
                answer="",
                explanation="",
                z3_code=last_code,
                used_z3=False,
                confidence=None,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences that LLMs commonly add."""
    m = _CODE_FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text.strip()


def _build_explanation(answer: str, n_premises: int) -> str:
    """Build a brief prose explanation for a Z3-derived answer."""
    premise_str = f"{n_premises} premise{'s' if n_premises != 1 else ''}"
    if answer == "Yes":
        return (
            f"Formal Z3 verification over {premise_str}: "
            "the conclusion must hold (negation is unsatisfiable)."
        )
    if answer == "No":
        return (
            f"Formal Z3 verification over {premise_str}: "
            "the conclusion cannot hold (it is unsatisfiable given the premises)."
        )
    return (
        f"Formal Z3 verification over {premise_str}: "
        "the premises neither prove nor disprove the conclusion "
        "(both the conclusion and its negation are satisfiable)."
    )
