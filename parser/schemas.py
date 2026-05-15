"""JSON-serializable schemas for the Stage 0 physics parser."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Quantity:
    value: float
    unit_symbol: str
    unit_name: str
    dimension: str
    source_text: str
    normalized_value: Optional[float] = None
    normalized_unit_symbol: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepPlanItem:
    step_id: str
    goal: str
    type: str
    input_var: Dict[str, Any] = field(default_factory=dict)
    output_var: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerifierError:
    error_type: str
    description: str
    repair_hint: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerifierResult:
    status: str
    errors: List[VerifierError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": list(self.warnings),
        }


@dataclass
class ProblemParseObject:
    problem_text: str
    domains: List[str] = field(default_factory=list)
    sub_domains: List[str] = field(default_factory=list)
    domain_confidence: float = 0.0
    known_quantities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    unknown_quantity: Optional[str] = None
    unknown_unit: Optional[str] = None
    step_plan: List[Dict[str, Any]] = field(default_factory=list)
    plan_confidence: float = 0.0
    parser_warnings: List[str] = field(default_factory=list)
    vso: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
