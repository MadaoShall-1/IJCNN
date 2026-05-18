"""Type 2 pipeline data structures (design §4).

All structures are JSON-serializable dataclasses.  Downstream stages may add
fields but must not remove fields defined in prior stages (design §4 invariant).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Formula Library (§4.3)
# ---------------------------------------------------------------------------

@dataclass
class FormulaEntry:
    """One entry in the formula library (§4.3)."""

    id: str
    topic: str
    subtopic: str
    target_quantities: List[str]
    canonical_quantity_names: List[str]
    text: str
    formula: str
    sympy_expr: str
    tool_dispatch: str          # "sympy" | "scipy_<module>" | "wolfram" | "llm"
    variables: Dict[str, Dict[str, str]]
    conditions: List[str] = field(default_factory=list)
    fol_axiom: str = ""
    premise_text: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FormulaEntry":
        return cls(
            id=d["id"],
            topic=d["topic"],
            subtopic=d["subtopic"],
            target_quantities=list(d.get("target_quantities", [])),
            canonical_quantity_names=list(d.get("canonical_quantity_names", [])),
            text=d["text"],
            formula=d["formula"],
            sympy_expr=d.get("sympy_expr", ""),
            tool_dispatch=d.get("tool_dispatch", "sympy"),
            variables=dict(d.get("variables", {})),
            conditions=list(d.get("conditions", [])),
            fol_axiom=d.get("fol_axiom", ""),
            premise_text=d.get("premise_text", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Variable State Object (§4.2)
# ---------------------------------------------------------------------------

@dataclass
class VSOEntry:
    """One variable in the Variable State Object."""

    value: float
    unit_symbol: str
    unit_name: str
    defined_at: str     # step_id that first introduced this variable
    updated_at: str     # step_id of most recent write

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Step Object (§4.1)
# ---------------------------------------------------------------------------

@dataclass
class StepObject:
    """One reasoning step in the solution trace (§4.1)."""

    step_id: str
    goal: str
    type: str           # calculation | formula_application | unit_conversion | setup | conclusion

    formula_ids: List[str] = field(default_factory=list)
    input_var: Dict[str, Any] = field(default_factory=dict)
    output_var: Dict[str, Any] = field(default_factory=dict)
    step_input: str = ""
    intermediate_answer: str = ""
    thought: str = ""
    confidence: Optional[float] = None
    checkable: bool = False
    status: Optional[str] = None    # OK | WRONG | UNCERTAIN | REPAIRED | null
    verifier_notes: str = ""
    evaluator_response: List[Any] = field(default_factory=list)
    cot_consistent: Optional[str] = None   # CONSISTENT | INCONSISTENT | NOT_CHECKED | null

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Trace Object (§4.5)
# ---------------------------------------------------------------------------

@dataclass
class TraceObject:
    """Full solution trace produced by Stage 2+3 (§4.5)."""

    problem_id: str
    formula_path_index: int

    steps: List[StepObject] = field(default_factory=list)
    vso: Dict[str, Any] = field(default_factory=dict)          # variable_name → VSOEntry dict
    vso_snapshots: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # step_id → VSO snapshot
    final_answer: str = ""
    final_unit: str = ""
    trace_status: str = "FAIL"  # PASS | FAIL | REPAIRED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Diagnosis Object (§4.6)
# ---------------------------------------------------------------------------

@dataclass
class CotIssue:
    step_id: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisObject:
    """Stage 4 diagnosis attached to a failed Trace Object (§4.6)."""

    global_error_type: Optional[str] = None    # E1 | E2 | E3 | E4 | E5 | E6
    fws_index: Optional[int] = None
    fws_error_type: Optional[str] = None
    fws_description: str = ""
    repair_hint: str = ""
    step_labels: Dict[str, str] = field(default_factory=dict)   # step_id → OK/WRONG/UNCERTAIN
    cot_issues: List[CotIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Formula Set — Stage 1 output (§5 Stage 1)
# ---------------------------------------------------------------------------

@dataclass
class FormulaSet:
    """One candidate formula path returned by Stage 1 retrieval.

    ``formulas`` maps step_id → FormulaEntry (or None when no formula was
    found for that step).  Stage 2 uses the mapping to look up which formula
    applies to each ``formula_application`` step.
    """

    formulas: Dict[str, Optional[FormulaEntry]]     # step_id → FormulaEntry | None
    retrieval_confidence: float
    path_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formulas": {
                k: (v.to_dict() if v else None)
                for k, v in self.formulas.items()
            },
            "retrieval_confidence": self.retrieval_confidence,
            "path_index": self.path_index,
        }
