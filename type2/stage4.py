"""Type 2 Stage 4: Error Structuring and Diagnosis (design §5 Stage 4).

Classifies a failed TraceObject into one of six error types (E1–E6), identifies
the First Wrong Step (FWS), and produces a DiagnosisObject for Stage 5 repair.

Error taxonomy (design §5 Stage 4):
  E1 — Formula selection error: wrong formula chosen for a step
  E2 — Variable mapping error: input/output variables incorrectly wired
  E3 — Arithmetic / computation error: correct formula, wrong numeric result
  E4 — Unit inconsistency: values mixed in incompatible units
  E5 — Chain propagation: an earlier error propagated into later steps
  E6 — Concept error: fundamentally wrong approach (not repairable by formula swap)

Always-available (no LLM dependency):
  diagnose_trace(trace, formula_set, parse_obj) → DiagnosisObject

DSPy-guarded:
  CotConsistencyModule — LLM-based chain-of-thought consistency checker
  DiagnosticReasonerModule — LLM-based global error typing
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from .schemas import (
    CotIssue,
    DiagnosisObject,
    FormulaEntry,
    FormulaSet,
    StepObject,
    TraceObject,
)
from .stage2 import _extract_numeric, map_formula_vars_to_vso, sympy_verify_step

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
# Deterministic helpers
# ---------------------------------------------------------------------------

def _find_fws(steps: List[StepObject]) -> Optional[int]:
    """Return the index of the First Wrong Step, or None if none."""
    for i, step in enumerate(steps):
        if step.status in ("WRONG", "INCORRECT"):
            return i
    return None


def _classify_step_error(
    step: StepObject,
    formula_entry: Optional[FormulaEntry],
) -> str:
    """Assign a step-level error label using deterministic rules.

    Returns one of: OK, WRONG, UNCERTAIN, or a preliminary E-type hint.
    """
    if step.status in (None, "OK", "REPAIRED"):
        return "OK"
    if step.status == "UNCERTAIN":
        return "UNCERTAIN"

    if formula_entry is None:
        return "E1"    # no formula found → likely formula selection error

    # Check whether the intermediate_answer has a numeric value
    if not _extract_numeric(step.intermediate_answer or ""):
        return "E3"    # no number produced → computation error

    # Check for obvious unit mismatch patterns (heuristic)
    answer_text = (step.intermediate_answer or "").lower()
    if re.search(r"\b(kg|g)\b", answer_text) and re.search(r"\b(m|cm|mm)\b", answer_text):
        return "E4"

    return "E3"    # default wrong step to arithmetic error


def _check_cot_consistency_heuristic(steps: List[StepObject]) -> List[CotIssue]:
    """Detect obvious chain-of-thought inconsistencies without LLM.

    Checks:
    1. A step claims an output value but the next step uses a different value
       for the same variable (propagation inconsistency).
    2. A step that is marked WRONG is immediately followed by a step that
       uses its output as input (propagation of wrong value).
    """
    issues: List[CotIssue] = []

    # Build output_value map per step
    for i, step in enumerate(steps):
        if step.status == "WRONG" and i + 1 < len(steps):
            next_step = steps[i + 1]
            # Check if any of the WRONG step's output_var names appear in next input_var
            wrong_outputs = set(step.output_var.keys())
            next_inputs = set(next_step.input_var.keys())
            overlap = wrong_outputs & next_inputs
            if overlap:
                issues.append(CotIssue(
                    step_id=next_step.step_id,
                    description=(
                        f"Step {next_step.step_id} uses output from WRONG step "
                        f"{step.step_id}: shared variables {overlap}."
                    ),
                ))

    return issues


def _infer_global_error_type(
    steps: List[StepObject],
    fws_index: Optional[int],
    fws_error_hint: str,
) -> str:
    """Infer the global error type from step-level signals.

    Hierarchy:
    - If no FWS found → no error (shouldn't be called for passing traces)
    - E1 if FWS had no formula match
    - E5 if FWS is not the first step and prior steps have propagated outputs
    - E3 otherwise (arithmetic / computation as default)
    """
    if fws_index is None:
        return "E3"

    if fws_error_hint in ("E1", "E4", "E6"):
        return fws_error_hint

    # E5 if the FWS is not the very first step
    if fws_index > 0:
        # Check if any prior step output fed into the FWS input
        fws = steps[fws_index]
        for prior in steps[:fws_index]:
            if set(prior.output_var.keys()) & set(fws.input_var.keys()):
                return "E5"

    return fws_error_hint or "E3"


# ---------------------------------------------------------------------------
# Main diagnosis function (always available)
# ---------------------------------------------------------------------------

def diagnose_trace(
    trace: TraceObject,
    formula_set: FormulaSet,
    *,
    use_llm_cot_check: bool = False,
    cot_checker=None,
    diagnostic_reasoner=None,
) -> DiagnosisObject:
    """Produce a DiagnosisObject for a (potentially failed) TraceObject.

    Parameters
    ----------
    trace:
        The trace to diagnose.  Works on both PASS and FAIL traces.
    formula_set:
        The formula set that was used to generate this trace.
    use_llm_cot_check:
        If True and ``cot_checker`` is provided, run the LLM consistency check.
    cot_checker:
        Instance of CotConsistencyModule (optional).
    diagnostic_reasoner:
        Instance of DiagnosticReasonerModule (optional).
    """
    diagnosis = DiagnosisObject()

    if not trace.steps:
        diagnosis.global_error_type = "E6"
        diagnosis.fws_description = "Empty trace — no steps were generated."
        return diagnosis

    # ── Step-level labels ────────────────────────────────────────────────────
    step_labels: Dict[str, str] = {}
    for step in trace.steps:
        fe = formula_set.formulas.get(step.step_id)
        label = _classify_step_error(step, fe)
        step_labels[step.step_id] = label
    diagnosis.step_labels = step_labels

    # ── First Wrong Step ─────────────────────────────────────────────────────
    fws_index = _find_fws(trace.steps)
    diagnosis.fws_index = fws_index

    if fws_index is not None:
        fws = trace.steps[fws_index]
        fws_hint = step_labels.get(fws.step_id, "E3")
        diagnosis.fws_error_type = fws_hint
        diagnosis.fws_description = (
            f"Step {fws.step_id} ({fws.goal!r}) was marked {fws.status}. "
            f"Verifier notes: {fws.verifier_notes or 'none'}."
        )

        # ── Global error type ────────────────────────────────────────────────
        global_type = _infer_global_error_type(trace.steps, fws_index, fws_hint)
        diagnosis.global_error_type = global_type

        # Repair hint
        if global_type == "E1":
            diagnosis.repair_hint = (
                "Replace the formula used in the FWS with an alternative candidate."
            )
        elif global_type == "E3":
            diagnosis.repair_hint = (
                "Re-execute the arithmetic in the FWS with correct substitution."
            )
        elif global_type == "E4":
            diagnosis.repair_hint = (
                "Convert all input variables to consistent SI units before applying the formula."
            )
        elif global_type == "E5":
            diagnosis.repair_hint = (
                "Roll back to the FWS snapshot, fix the FWS, and re-run subsequent steps."
            )
        elif global_type == "E6":
            diagnosis.repair_hint = (
                "Reclassify the problem domain and select a fundamentally different approach."
            )
    else:
        # Passing trace or no definitive FWS
        diagnosis.global_error_type = None
        diagnosis.fws_description = "No wrong step detected."

    # ── COT consistency check ────────────────────────────────────────────────
    cot_issues = _check_cot_consistency_heuristic(trace.steps)

    if use_llm_cot_check and cot_checker is not None:
        try:
            llm_issues = cot_checker(trace)
            cot_issues.extend(llm_issues)
        except Exception as exc:
            logger.warning("LLM COT check failed: %s", exc)

    diagnosis.cot_issues = cot_issues

    # ── Optional LLM diagnostic reasoner ────────────────────────────────────
    if diagnostic_reasoner is not None and fws_index is not None:
        try:
            reasoning_result = diagnostic_reasoner(trace, formula_set)
            if reasoning_result.get("global_error_type"):
                diagnosis.global_error_type = reasoning_result["global_error_type"]
            if reasoning_result.get("repair_hint"):
                diagnosis.repair_hint = reasoning_result["repair_hint"]
        except Exception as exc:
            logger.warning("Diagnostic reasoner failed: %s", exc)

    return diagnosis


# ---------------------------------------------------------------------------
# DSPy modules (guarded)
# ---------------------------------------------------------------------------

if _DSPY_AVAILABLE:

    class CotConsistencySignature(_dspy.Signature):
        """Check whether the chain-of-thought steps in a physics solution are internally consistent.

        Look for: skipped steps, incorrect variable reuse, contradictory intermediate results,
        and steps whose output is inconsistent with prior steps.
        """

        steps_summary: str = _dspy.InputField(
            desc="JSON list of {step_id, goal, step_input, intermediate_answer, status}"
        )
        problem_text: str = _dspy.InputField(desc="Original physics problem statement")

        issues_json: str = _dspy.OutputField(
            desc='JSON list of {step_id, description} for each inconsistency found. '
                 'Empty list [] if no issues.'
        )

    class CotConsistencyModule(_dspy.Module):
        """LLM-based chain-of-thought consistency checker for Stage 4."""

        def __init__(self) -> None:
            import json as _json
            self._json = _json
            self.checker = _dspy.Predict(CotConsistencySignature)

        def forward(self, trace: TraceObject) -> List[CotIssue]:
            import json as _json
            steps_data = [
                {
                    "step_id": s.step_id,
                    "goal": s.goal,
                    "step_input": s.step_input,
                    "intermediate_answer": s.intermediate_answer,
                    "status": s.status,
                }
                for s in trace.steps
            ]
            pred = self.checker(
                steps_summary=_json.dumps(steps_data, ensure_ascii=False),
                problem_text=getattr(trace, "problem_text", ""),
            )
            try:
                raw = _json.loads(pred.issues_json)
                return [CotIssue(step_id=item["step_id"], description=item["description"])
                        for item in raw if isinstance(item, dict)]
            except Exception:
                return []

    class DiagnosticReasonerSignature(_dspy.Signature):
        """Classify the global error type for a failed physics solution trace.

        Choose from:
          E1 — wrong formula selected
          E2 — variable mapping error
          E3 — arithmetic / computation error
          E4 — unit inconsistency
          E5 — error propagation from earlier step
          E6 — fundamental concept error
        """

        problem_text: str = _dspy.InputField(desc="Original physics problem")
        fws_description: str = _dspy.InputField(desc="Description of the First Wrong Step")
        steps_summary: str = _dspy.InputField(desc="JSON summary of all steps")

        global_error_type: str = _dspy.OutputField(
            desc="One of: E1 E2 E3 E4 E5 E6"
        )
        repair_hint: str = _dspy.OutputField(
            desc="One sentence describing how to repair this error"
        )

    class DiagnosticReasonerModule(_dspy.Module):
        """LLM-based global error classifier for Stage 4."""

        def __init__(self) -> None:
            self.reasoner = _dspy.Predict(DiagnosticReasonerSignature)

        def forward(self, trace: TraceObject, formula_set: FormulaSet) -> dict:
            import json as _json
            steps_data = [
                {"step_id": s.step_id, "goal": s.goal,
                 "intermediate_answer": s.intermediate_answer, "status": s.status}
                for s in trace.steps
            ]
            fws_index = _find_fws(trace.steps)
            fws_desc = ""
            if fws_index is not None:
                fws = trace.steps[fws_index]
                fws_desc = f"Step {fws.step_id}: {fws.goal} → status={fws.status}"

            pred = self.reasoner(
                problem_text=getattr(trace, "problem_text", ""),
                fws_description=fws_desc,
                steps_summary=_json.dumps(steps_data, ensure_ascii=False),
            )
            raw_type = str(pred.global_error_type).strip().upper()
            # Normalize to E1–E6
            match = re.search(r"E[1-6]", raw_type)
            global_type = match.group(0) if match else "E3"
            return {
                "global_error_type": global_type,
                "repair_hint": str(pred.repair_hint).strip(),
            }
