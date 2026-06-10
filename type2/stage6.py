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

import math
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class _AnswerRepair:
    answer: str
    error_type: str
    repair_hint: str
    suspicious_step_id: Optional[str] = None
    suspicious_step_goal: Optional[str] = None


def _answer_metadata(original_answer: str) -> Dict[str, Any]:
    return {
        "final_answer_verdict": "PASS",
        "final_answer_error_type": None,
        "answer_level_fws": False,
        "suspicious_step_id": None,
        "suspicious_step_goal": None,
        "repair_hint": None,
        "numeric_repair_attempted": False,
        "numeric_repair_accepted": False,
        "original_answer": original_answer,
        "repaired_answer": None,
    }


def _extract_numeric_value(text: Any) -> Optional[float]:
    match = _NUMERIC_RE.search(str(text or ""))
    if not match:
        return None
    try:
        return float(match.group(1).replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def _is_finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(float(value))


def _close(a: Optional[float], b: Optional[float], rel_tol: float = 1e-3, abs_tol: float = 1e-9) -> bool:
    if not _is_finite(a) or not _is_finite(b):
        return False
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def _vso_value(trace: TraceObject, names: Tuple[str, ...]) -> Optional[float]:
    for name in names:
        entry = trace.vso.get(name)
        value = entry.get("value") if isinstance(entry, dict) else getattr(entry, "value", None)
        if _is_finite(value):
            return float(value)
    return None


def _rlc_reactance_values(trace: TraceObject, parse_obj: ProblemParseObject) -> Tuple[Optional[float], Optional[float]]:
    XL = _vso_value(trace, ("X_L", "XL", "Z_L", "inductive_reactance"))
    XC = _vso_value(trace, ("X_C", "XC", "Z_C", "capacitive_reactance"))
    if _is_finite(XL) and _is_finite(XC):
        return XL, XC

    text = str(parse_obj.problem_text or "")
    patterns = {
        "XL": r"\bX\s*_?\s*L\b\s*=\s*([+-]?\d+(?:\.\d+)?)",
        "XC": r"\bX\s*_?\s*C\b\s*=\s*([+-]?\d+(?:\.\d+)?)",
    }
    values: Dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            values[key] = float(match.group(1))
    if "XL" in values and "XC" in values:
        return values["XL"], values["XC"]

    # Stage 0 sometimes labels XL/XC as positional resistance values R2/R3.
    R2 = _vso_value(trace, ("R2",))
    R3 = _vso_value(trace, ("R3",))
    if _is_finite(R2) and _is_finite(R3) and ("xl" in text.lower() and "xc" in text.lower()):
        return R2, R3
    return XL, XC


def _trace_text(trace: TraceObject, parse_obj: ProblemParseObject) -> str:
    step_text = " ".join(
        " ".join(
            str(part or "")
            for part in (
                step.step_id,
                step.goal,
                step.step_input,
                step.intermediate_answer,
                step.verifier_notes,
                " ".join(step.formula_ids),
            )
        )
        for step in trace.steps
    )
    return " ".join(
        [
            str(parse_obj.problem_text or ""),
            str(parse_obj.unknown_quantity or ""),
            str(parse_obj.unknown_unit or ""),
            " ".join(parse_obj.conditions or []),
            step_text,
        ]
    )


def _last_step(trace: TraceObject) -> Tuple[Optional[str], Optional[str]]:
    if not trace.steps:
        return None, None
    return trace.steps[-1].step_id, trace.steps[-1].goal


def _target_is(parse_obj: ProblemParseObject, names: Tuple[str, ...], text: str) -> bool:
    target = str(parse_obj.unknown_quantity or "")
    lowered = text.lower()
    if target in names:
        return True
    return any(name.lower() in lowered for name in names)


def _unit_for_target(parse_obj: ProblemParseObject, target: str) -> str:
    unit = str(parse_obj.unknown_unit or "")
    if unit:
        return unit
    return {"Z": "ohm", "I": "A", "P": "W", "P_avg": "W"}.get(target, "")


def _format_numeric(value: float, unit: str = "") -> str:
    return f"{value:.6g} {unit}".strip() if unit else f"{value:.6g}"


def _try_rlc_condition_repair(
    trace: TraceObject,
    parse_obj: ProblemParseObject,
    original_answer: str,
) -> Optional[_AnswerRepair]:
    text = _trace_text(trace, parse_obj)
    lowered = text.lower()
    if not any(marker in lowered for marker in ("not in resonance", "not at resonance", "not resonant")):
        return None

    R = _vso_value(trace, ("R", "R_eq", "resistance"))
    XL, XC = _rlc_reactance_values(trace, parse_obj)
    if not (_is_finite(R) and _is_finite(XL) and _is_finite(XC)):
        return None

    Z = math.sqrt(float(R) ** 2 + (float(XL) - float(XC)) ** 2)
    original_value = _extract_numeric_value(original_answer)
    target = str(parse_obj.unknown_quantity or "")
    repaired_value: Optional[float] = None
    unit = _unit_for_target(parse_obj, target)
    shortcut_value: Optional[float] = None

    if _target_is(parse_obj, ("Z", "impedance"), text):
        repaired_value = Z
        unit = unit or "ohm"
        shortcut_value = R
    elif _target_is(parse_obj, ("I", "current"), text):
        V = _vso_value(trace, ("V", "U", "V_rms", "voltage"))
        if not _is_finite(V):
            return None
        repaired_value = float(V) / Z
        unit = unit or "A"
        shortcut_value = float(V) / float(R) if R else None
    elif _target_is(parse_obj, ("cos_phi", "power_factor"), text) or "power factor" in lowered:
        repaired_value = float(R) / Z
        unit = ""
        shortcut_value = 1.0
    elif _target_is(parse_obj, ("P", "P_avg", "average_power"), text) or "average power" in lowered:
        V = _vso_value(trace, ("V", "U", "V_rms", "voltage"))
        if not _is_finite(V):
            return None
        I = float(V) / Z
        cos_phi = float(R) / Z
        repaired_value = float(V) * I * cos_phi
        unit = unit or "W"
        shortcut_value = (float(V) ** 2) / float(R) if R else None

    if not _is_finite(repaired_value) or not _close(original_value, shortcut_value, rel_tol=1e-2):
        return None
    if _close(original_value, repaired_value, rel_tol=1e-3):
        return None

    step_id, step_goal = _last_step(trace)
    return _AnswerRepair(
        answer=_format_numeric(float(repaired_value), unit),
        error_type="rlc_condition_violation",
        repair_hint="Question says the circuit is not in resonance; use Z=sqrt(R^2+(X_L-X_C)^2), not Z=R.",
        suspicious_step_id=step_id,
        suspicious_step_goal=step_goal,
    )


def _try_q_disambiguation_repair(
    trace: TraceObject,
    parse_obj: ProblemParseObject,
    original_answer: str,
) -> Optional[_AnswerRepair]:
    text = _trace_text(trace, parse_obj)
    lowered = text.lower()
    if not any(marker in lowered for marker in ("quality factor", "q factor", "resonance q")):
        return None

    R = _vso_value(trace, ("R", "resistance"))
    L = _vso_value(trace, ("L", "L_ind", "inductance"))
    C = _vso_value(trace, ("C", "C_cap", "capacitance"))
    omega0 = _vso_value(trace, ("omega_0", "omega0", "w0"))
    f0 = _vso_value(trace, ("f0", "f_0", "resonance_frequency"))
    bandwidth = _vso_value(trace, ("bandwidth", "delta_f", "Delta_f"))

    repaired_value: Optional[float] = None
    if _is_finite(omega0) and _is_finite(L) and _is_finite(R) and R:
        repaired_value = float(omega0) * float(L) / float(R)
    elif _is_finite(R) and _is_finite(L) and _is_finite(C) and R and C:
        repaired_value = (1.0 / float(R)) * math.sqrt(float(L) / float(C))
    elif _is_finite(f0) and _is_finite(bandwidth) and bandwidth:
        repaired_value = float(f0) / float(bandwidth)

    if not _is_finite(repaired_value):
        return None
    if _close(_extract_numeric_value(original_answer), repaired_value, rel_tol=1e-3):
        return None

    step_id, step_goal = _last_step(trace)
    return _AnswerRepair(
        answer=_format_numeric(float(repaired_value), ""),
        error_type="q_disambiguation",
        repair_hint="Question asks for quality factor Q, not electric charge Q.",
        suspicious_step_id=step_id,
        suspicious_step_goal=step_goal,
    )


def _try_wrong_target_repair(
    trace: TraceObject,
    parse_obj: ProblemParseObject,
    original_answer: str,
) -> Optional[_AnswerRepair]:
    target = str(parse_obj.unknown_quantity or "")
    if not target or target not in trace.vso:
        return None
    entry = trace.vso.get(target)
    if not isinstance(entry, dict) or not _is_finite(entry.get("value")):
        return None

    last_output = trace.steps[-1].output_var if trace.steps else {}
    current_value = _extract_numeric_value(original_answer)
    target_value = float(entry["value"])
    if _close(current_value, target_value, rel_tol=1e-3):
        return None
    if target in getattr(parse_obj, "known_quantities", {}) and _is_finite(current_value):
        return None
    if target in last_output:
        return None
    question_lower = str(parse_obj.problem_text or "").lower()
    original_lower = original_answer.lower()
    target_unit = str(entry.get("unit_symbol") or parse_obj.unknown_unit or "")
    if "%" in original_answer and ("percentage" in question_lower or "percent" in question_lower):
        return None
    if (
        "force" in question_lower
        and target == "E"
        and " n/c" in f" {target_unit.lower()}"
        and " n" in f" {original_lower}"
        and "n/c" not in original_lower
    ):
        return None
    if not original_answer.strip() or _is_finite(current_value):
        unit = target_unit
        step_id, step_goal = _last_step(trace)
        return _AnswerRepair(
            answer=_format_numeric(target_value, unit),
            error_type="wrong_target_variable",
            repair_hint="Final answer did not report parse_obj.unknown_quantity even though that value exists in VSO.",
            suspicious_step_id=step_id,
            suspicious_step_goal=step_goal,
        )
    return None


_UNIT_SCALE_MAP = {
    "mT": ("T", 1e3),
    "mH": ("H", 1e3),
    "uJ": ("J", 1e6),
    "uC": ("C", 1e6),
}


def _try_unit_scale_repair(
    trace: TraceObject,
    parse_obj: ProblemParseObject,
    original_answer: str,
) -> Optional[_AnswerRepair]:
    target = str(parse_obj.unknown_quantity or "")
    desired_unit = str(parse_obj.unknown_unit or "")
    if not target or target not in trace.vso or desired_unit not in _UNIT_SCALE_MAP:
        return None
    if desired_unit.lower() in original_answer.lower():
        return None

    entry = trace.vso.get(target)
    if not isinstance(entry, dict) or not _is_finite(entry.get("value")):
        return None
    base_unit, multiplier = _UNIT_SCALE_MAP[desired_unit]
    vso_unit = str(entry.get("unit_symbol") or "")
    if vso_unit and vso_unit != base_unit:
        return None
    si_value = float(entry["value"])
    if not _close(_extract_numeric_value(original_answer), si_value, rel_tol=1e-3):
        return None

    step_id, step_goal = _last_step(trace)
    return _AnswerRepair(
        answer=_format_numeric(si_value * multiplier, desired_unit),
        error_type="unit_scale_error",
        repair_hint="Final numeric value was correct in SI units but displayed in the wrong scale for the requested unit.",
        suspicious_step_id=step_id,
        suspicious_step_goal=step_goal,
    )


def _guarded_answer_repair(
    trace: TraceObject,
    parse_obj: ProblemParseObject,
    original_answer: str,
) -> Tuple[str, Dict[str, Any]]:
    metadata = _answer_metadata(original_answer)
    if trace.trace_status not in ("PASS", "REPAIRED"):
        metadata["final_answer_verdict"] = "FAIL"
        return original_answer, metadata

    for handler in (
        _try_rlc_condition_repair,
        _try_q_disambiguation_repair,
        _try_wrong_target_repair,
        _try_unit_scale_repair,
    ):
        repair = handler(trace, parse_obj, original_answer)
        if repair is None:
            continue
        metadata.update(
            {
                "final_answer_verdict": "SUSPICIOUS",
                "final_answer_error_type": repair.error_type,
                "answer_level_fws": True,
                "suspicious_step_id": repair.suspicious_step_id,
                "suspicious_step_goal": repair.suspicious_step_goal,
                "repair_hint": repair.repair_hint,
                "numeric_repair_attempted": True,
            }
        )
        if repair.answer and _extract_numeric_value(repair.answer) is not None:
            metadata["numeric_repair_accepted"] = True
            metadata["repaired_answer"] = repair.answer
            metadata["final_answer_verdict"] = "PASS"
            return repair.answer, metadata
        return original_answer, metadata

    return original_answer, metadata


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
    original_answer = extract_final_answer(trace)
    answer, answer_level_metadata = _guarded_answer_repair(
        trace, parse_obj, original_answer
    )
    trace.final_answer_verdict = answer_level_metadata.get("final_answer_verdict")
    trace.final_answer_error_type = answer_level_metadata.get("final_answer_error_type")
    trace.answer_level_fws = bool(answer_level_metadata.get("answer_level_fws"))
    trace.suspicious_step_id = answer_level_metadata.get("suspicious_step_id")
    trace.suspicious_step_goal = answer_level_metadata.get("suspicious_step_goal")
    trace.numeric_repair_attempted = bool(answer_level_metadata.get("numeric_repair_attempted"))
    trace.numeric_repair_accepted = bool(answer_level_metadata.get("numeric_repair_accepted"))

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
        "fws_diagnosis": {
            "first_wrong_step_id": trace.first_wrong_step_id,
            "first_wrong_step_goal": trace.first_wrong_step_goal,
            "first_wrong_step_type": trace.first_wrong_step_type,
            "first_wrong_error_type": trace.first_wrong_error_type,
            "first_wrong_feedback": trace.first_wrong_feedback,
            "repair_hint": trace.repair_hint,
            "diagnostic_tags": list(trace.diagnostic_tags),
        },
        "answer_level_verification": answer_level_metadata,
        "final_answer_check": {
            "verdict": answer_level_metadata.get("final_answer_verdict", "UNKNOWN"),
            "error_type": answer_level_metadata.get("final_answer_error_type"),
            "repair_attempted": bool(answer_level_metadata.get("numeric_repair_attempted")),
            "repair_accepted": bool(answer_level_metadata.get("numeric_repair_accepted")),
            "notes": answer_level_metadata.get("repair_hint"),
        },
        "deterministic_replay": {
            "attempted": trace.deterministic_replay_attempted,
            "recomputed_steps": trace.deterministic_replay_recomputed_steps,
            "final_rebound": trace.deterministic_replay_final_rebound,
            "original_answer": trace.deterministic_replay_original_answer,
        },
    }
    response.update(
        {
            "final_answer_verdict": answer_level_metadata["final_answer_verdict"],
            "final_answer_error_type": answer_level_metadata["final_answer_error_type"],
            "answer_level_fws": answer_level_metadata["answer_level_fws"],
            "suspicious_step_id": answer_level_metadata["suspicious_step_id"],
            "suspicious_step_goal": answer_level_metadata["suspicious_step_goal"],
            "repair_hint": answer_level_metadata["repair_hint"],
            "numeric_repair_attempted": answer_level_metadata["numeric_repair_attempted"],
            "numeric_repair_accepted": answer_level_metadata["numeric_repair_accepted"],
        }
    )

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
