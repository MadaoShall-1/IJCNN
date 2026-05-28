"""Type 2 Stage 6: Response Assembly (design §5 Stage 6).

Converts a (possibly repaired) TraceObject into the final competition response
dictionary matching the submission schema.

Always-available:
  extract_final_answer(trace)                → str
  build_response(trace, parse_obj, formula_set, diagnosis) → dict

DSPy-guarded:
  ExplanationAssemblerModule — LLM explanation generator
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from parser.schemas import ProblemParseObject

from .schemas import (
    DiagnosisObject,
    FormulaEntry,
    FormulaSet,
    StepObject,
    TraceObject,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import dspy as _dspy
    _DSPY_AVAILABLE = True
except ModuleNotFoundError:
    _dspy = None  # type: ignore[assignment]
    _DSPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(
    r"([+-]?\s*\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-z°Ω·/²³⁻^][^\s,;]*)?",
    re.UNICODE,
)


def extract_final_answer(trace: TraceObject) -> str:
    """Extract a clean final answer string from a trace.

    Checks, in order:
    1. trace.final_answer (set by Stage 2+3 or repair)
    2. Last step's intermediate_answer that contains a number
    3. Empty string if nothing usable found
    """
    if trace.final_answer and trace.final_answer.strip():
        return trace.final_answer.strip()

    for step in reversed(trace.steps):
        ans = (step.intermediate_answer or "").strip()
        if ans and _NUMERIC_RE.search(ans):
            return ans

    return ""


def _collect_premises(formula_set: FormulaSet) -> List[str]:
    """Gather unique premise_text strings from all formulas in the set."""
    seen = set()
    premises = []
    for entry in formula_set.formulas.values():
        if entry is None:
            continue
        pt = entry.premise_text.strip()
        if pt and pt not in seen:
            seen.add(pt)
            premises.append(pt)
    return premises


def _collect_fol_axioms(formula_set: FormulaSet) -> List[str]:
    """Gather unique FOL axiom strings from all formulas in the set."""
    seen = set()
    axioms = []
    for entry in formula_set.formulas.values():
        if entry is None:
            continue
        ax = entry.fol_axiom.strip()
        if ax and ax not in seen:
            seen.add(ax)
            axioms.append(ax)
    return axioms


def _build_step_cot(steps: List[StepObject]) -> str:
    """Construct a chain-of-thought narrative from the step list."""
    parts = []
    for i, step in enumerate(steps, 1):
        thought = step.thought or ""
        work = step.step_input or ""
        answer = step.intermediate_answer or ""
        line = f"Step {i} ({step.goal})"
        if thought:
            line += f": {thought}"
        if work:
            line += f" [{work}]"
        if answer:
            line += f" → {answer}"
        parts.append(line)
    return "\n".join(parts)


def _aggregate_confidence(
    steps: List[StepObject],
    trace_status: str,
    formula_set_confidence: float,
) -> float:
    """Compute an overall confidence score for the response.

    Weighted average of step confidences × retrieval confidence.
    Repaired traces receive a 0.8 multiplier.
    """
    if not steps:
        return 0.0

    step_confs = [s.confidence for s in steps if s.confidence is not None]
    if not step_confs:
        step_conf = 0.5
    else:
        step_conf = sum(step_confs) / len(step_confs)

    combined = step_conf * formula_set_confidence
    if trace_status == "REPAIRED":
        combined *= 0.8
    elif trace_status == "FAIL":
        combined *= 0.3

    return round(min(combined, 1.0), 4)


# ---------------------------------------------------------------------------
# Main assembly function
# ---------------------------------------------------------------------------

def build_response(
    trace: TraceObject,
    parse_obj: ProblemParseObject,
    formula_set: FormulaSet,
    diagnosis: Optional[DiagnosisObject] = None,
    *,
    explanation_assembler=None,
) -> Dict[str, Any]:
    """Assemble the final response dictionary for a Type 2 problem.

    Returns a dict matching the competition submission schema:
    {
        "answer": str,
        "confidence": float,
        "chain_of_thought": str,
        "premises": [str, ...],
        "fol_axioms": [str, ...],
        "trace_status": str,
        "formula_path_index": int,
        "steps": [ {step_id, goal, formula_ids, intermediate_answer, status}, ... ],
        "diagnosis": dict | null,
    }
    """
    answer = extract_final_answer(trace)

    premises = _collect_premises(formula_set)
    fol_axioms = _collect_fol_axioms(formula_set)
    cot = _build_step_cot(trace.steps)

    confidence = _aggregate_confidence(
        trace.steps, trace.trace_status, formula_set.retrieval_confidence
    )

    # LLM-generated explanation (optional)
    if explanation_assembler is not None:
        try:
            cot = explanation_assembler(
                problem_text=parse_obj.problem_text,
                answer=answer,
                steps_cot=cot,
            )
        except Exception as exc:
            logger.warning("Explanation assembler failed: %s", exc)

    steps_summary = [
        {
            "step_id": s.step_id,
            "goal": s.goal,
            "formula_ids": s.formula_ids,
            "intermediate_answer": s.intermediate_answer,
            "status": s.status,
        }
        for s in trace.steps
    ]

    response: Dict[str, Any] = {
        "answer": answer,
        "confidence": confidence,
        "chain_of_thought": cot,
        "premises": premises,
        "fol_axioms": fol_axioms,
        "trace_status": trace.trace_status,
        "formula_path_index": trace.formula_path_index,
        "steps": steps_summary,
        "diagnosis": diagnosis.to_dict() if diagnosis is not None else None,
    }

    return response


# ---------------------------------------------------------------------------
# DSPy module (guarded)
# ---------------------------------------------------------------------------

if _DSPY_AVAILABLE:

    class ExplanationAssemblerSignature(_dspy.Signature):
        """Rewrite a physics solution chain-of-thought into a clear, student-friendly explanation.

        Use full sentences.  Mention the formula used at each step.
        Keep the answer highlighted at the end.
        """

        problem_text: str = _dspy.InputField(desc="Original physics problem statement")
        answer: str = _dspy.InputField(desc="Final numeric answer with unit")
        steps_cot: str = _dspy.InputField(
            desc="Machine-generated chain-of-thought from the solver"
        )

        explanation: str = _dspy.OutputField(
            desc="Student-friendly explanation of the full solution (3–6 sentences)"
        )

    class ExplanationAssemblerModule(_dspy.Module):
        """LLM-based explanation assembler for Stage 6."""

        def __init__(self) -> None:
            self.assembler = _dspy.Predict(ExplanationAssemblerSignature)

        def forward(self, problem_text: str, answer: str, steps_cot: str) -> str:
            pred = self.assembler(
                problem_text=problem_text,
                answer=answer,
                steps_cot=steps_cot,
            )
            return str(pred.explanation).strip()
