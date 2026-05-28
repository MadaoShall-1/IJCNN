"""Schemas for the Type 1 (logic-based educational query) pipeline.

Mirrors the design doc §4 convention: all pipeline artifacts are JSON-serializable
dataclasses.  The Type 1 equivalents of ProblemParseObject (parser/schemas.py) live
here so the two dataset paths stay fully independent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QuestionFormat(str, Enum):
    """Expected answer format detected from the question text (design §3.1)."""

    MCQ = "mcq"
    YES_NO_UNCERTAIN = "yes_no_uncertain"
    OPEN_ENDED = "open_ended"


class SolverRoute(str, Enum):
    """Which solver to invoke for a given question (design §3.1).

    Z3 / PROVER9 are only active for YES_NO_UNCERTAIN questions.
    All other formats (and Phase 0 fallback) use the LLM path.
    """

    Z3 = "z3"
    PROVER9 = "prover9"
    LLM = "llm"


@dataclass
class Type1Question:
    """A single parsed question from a Type 1 record."""

    text: str
    format: QuestionFormat
    solver_route: SolverRoute
    # Populated only for MCQ questions: {"A": "option text", "B": "...", ...}
    mcq_options: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["format"] = self.format.value
        d["solver_route"] = self.solver_route.value
        return d


@dataclass
class Type1ParseObject:
    """Structured output of the Type 1 parsing stage.

    Equivalent role to ProblemParseObject for Type 2.  Premises are already
    structured in the API payload — no extraction is needed (design §3.1).
    """

    premises_nl: List[str]
    questions: List[Type1Question]
    # FOL premises if the caller supplied premises-FOL; otherwise empty.
    premises_fol: List[str] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "premises_nl": list(self.premises_nl),
            "premises_fol": list(self.premises_fol),
            "questions": [q.to_dict() for q in self.questions],
            "parse_warnings": list(self.parse_warnings),
            "metadata": dict(self.metadata),
        }


@dataclass
class Type1Response:
    """Single-question API response for a Type 1 query.

    Fields match the competition submission schema (contest_details.txt §Submission).
    """

    answer: str
    explanation: str
    fol: List[str] = field(default_factory=list)
    cot: List[str] = field(default_factory=list)
    premises: List[str] = field(default_factory=list)
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "answer": self.answer,
            "explanation": self.explanation,
        }
        if self.fol:
            d["fol"] = list(self.fol)
        if self.cot:
            d["cot"] = list(self.cot)
        if self.premises:
            d["premises"] = list(self.premises)
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d
