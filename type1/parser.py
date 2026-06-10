"""Type 1 (logic-based educational query) parser.

Implements the programmatic parsing stage described in design §3.1:

  1. Extract premises-NL, premises-FOL, and question(s) from the raw payload.
  2. Detect the answer format of each question (MCQ / Yes-No-Uncertain / Open-ended)
     using pattern matching — no LLM call at this stage.
  3. Decide which solver to route each question to (Z3 / Prover9 / LLM) based on
     premise content and question format.
  4. Return a Type1ParseObject ready for the downstream DSPy reasoning modules.

All logic here is deterministic.  LLM calls begin in dspy_modules.py.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .schemas import QuestionFormat, SolverRoute, Type1ParseObject, Type1Question

# ---------------------------------------------------------------------------
# Answer format detection patterns (design §3.1)
# ---------------------------------------------------------------------------

# MCQ — labeled option lines such as "A. text", "A) text", or "(A) text"
_MCQ_OPTION_RE = re.compile(
    r"(?:^|[\n\r])\s*\(?([A-D])\)?[.)]\s+\S",
    re.MULTILINE,
)

# MCQ — phrasing that implies a multiple-choice structure
_MCQ_PHRASE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bwhich\s+of\s+the\s+following\b", re.IGNORECASE),
    re.compile(
        r"\bselect\s+(?:the\s+)?(?:correct|best|most\s+appropriate)\b", re.IGNORECASE
    ),
    re.compile(r"\bchoose\s+(?:the\s+)?(?:correct|best)\b", re.IGNORECASE),
]

# Yes/No/Uncertain — from design §3.1 plus common exam variants
_YES_NO_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bis\s+it\s+true\b", re.IGNORECASE),
    re.compile(r"\bdoes\b.{0,60}\bimpl[yi]\b", re.IGNORECASE),
    re.compile(r"\bcan\s+we\s+conclude\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+it\s+follow\b", re.IGNORECASE),
    re.compile(r"\bcan\s+(?:it\s+be\s+)?(?:concluded|inferred|determined)\b", re.IGNORECASE),
    re.compile(
        r"\bis\s+(?:the\s+following\s+)?(?:statement|conclusion)\s+"
        r"(?:true|false|valid|correct)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bcan\s+(?:we|one)\s+(?:say|claim|assert)\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+(?:the\s+)?(?:combination|fact|set\s+of\s+premises)\b", re.IGNORECASE),
    re.compile(r"\bdo(?:es)?\s+(?:these|the)\s+premises\b", re.IGNORECASE),
    re.compile(r"\bfollows?\s+from\s+(?:the\s+)?premises\b", re.IGNORECASE),
]

# Z3 routing trigger — any digit or comparison operator in a premise string
# (design §3.1: "premises contain numeric values, inequalities, or thresholds → Z3")
_NUMERIC_PREMISE_RE = re.compile(r"[\d<>≤≥=]")


# ---------------------------------------------------------------------------
# MCQ option extraction
# ---------------------------------------------------------------------------

def _extract_mcq_options(question_text: str) -> Dict[str, str]:
    """Return a dict mapping option labels to their text for MCQ questions.

    Handles formats:
      "A. text"  "A) text"  "(A) text"  preceded by newline or 2+ spaces.
    """
    options: Dict[str, str] = {}
    # Build a pattern that anchors at line start or after >=2 spaces.
    # Lookahead stops at next option label or end-of-string.
    for m in re.finditer(
        r"(?:(?:^|[\n\r])\s*|\s{2,})\(?([A-D])\)?[.)]\s+(.+?)(?=\s*(?:\(?[A-D]\)?[.)]|$))",
        question_text,
        re.DOTALL | re.MULTILINE,
    ):
        label = m.group(1).upper()
        text = m.group(2).strip()
        if label not in options:  # keep first occurrence
            options[label] = text
    return options


# ---------------------------------------------------------------------------
# Public detection helpers
# ---------------------------------------------------------------------------

def detect_question_format(
    question_text: str,
) -> Tuple[QuestionFormat, Dict[str, str]]:
    """Classify a question string as MCQ, Yes/No/Uncertain, or Open-ended.

    Returns ``(format, mcq_options)`` where ``mcq_options`` is non-empty only
    for MCQ questions.

    Detection order matches design §3.1 priority:
      1. Explicit labeled options → MCQ
      2. MCQ-implying phrases    → MCQ
      3. Yes/No/Uncertain phrases → YES_NO_UNCERTAIN
      4. Default                  → OPEN_ENDED
    """
    # 1. Labeled option lines take priority (most reliable signal)
    if _MCQ_OPTION_RE.search(question_text):
        return QuestionFormat.MCQ, _extract_mcq_options(question_text)

    # 2. MCQ-implying phrases
    for pattern in _MCQ_PHRASE_PATTERNS:
        if pattern.search(question_text):
            return QuestionFormat.MCQ, {}

    # 3. Yes/No/Uncertain phrasing
    for pattern in _YES_NO_PATTERNS:
        if pattern.search(question_text):
            return QuestionFormat.YES_NO_UNCERTAIN, {}

    return QuestionFormat.OPEN_ENDED, {}


def detect_solver_route(
    premises: List[str],
    question_format: QuestionFormat,
) -> SolverRoute:
    """Select Z3, Prover9, or LLM for a given question.

    Per design §3.1:
      - Only YES_NO_UNCERTAIN questions are eligible for formal solvers.
      - MCQ and Open-ended always go to the LLM path.
      - For YES_NO_UNCERTAIN: premises with digits or comparison operators → Z3,
        otherwise (purely symbolic predicate premises) → Prover9.
    """
    if question_format != QuestionFormat.YES_NO_UNCERTAIN:
        return SolverRoute.LLM

    combined = " ".join(premises)
    if _NUMERIC_PREMISE_RE.search(combined):
        return SolverRoute.Z3
    return SolverRoute.PROVER9


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_type1(payload: Dict[str, Any]) -> Type1ParseObject:
    """Parse a raw Type 1 API payload into a structured Type1ParseObject.

    Accepts both the official field name ``premises-NL`` and the alternate
    ``premises``, and both ``questions`` (list) and ``question`` (string),
    per the design §3.1 router fallback note.
    """
    premises_nl: List[str] = _coerce_list(
        payload.get("premises-NL") or payload.get("premises")
    )
    premises_fol: List[str] = _coerce_list(
        payload.get("premises-FOL") or payload.get("premises-fol")
    )

    # Accept either a list of questions or a single question string.
    raw_questions: List[str] = _coerce_list(
        payload.get("questions") or payload.get("question") or payload.get("query")
    )
    options = _coerce_list(payload.get("options"))

    warnings: List[str] = []
    if not premises_nl:
        warnings.append(
            "No premises-NL found in payload; reasoning will be premise-free."
        )
    if not raw_questions:
        warnings.append("No questions found in payload.")

    parsed_questions: List[Type1Question] = []
    for q_text in raw_questions:
        q_text = _append_options(q_text, options)
        fmt, mcq_options = detect_question_format(q_text)
        if options:
            if {option.strip().lower() for option in options} == {"yes", "no", "uncertain"}:
                fmt = QuestionFormat.YES_NO_UNCERTAIN
            else:
                fmt = QuestionFormat.MCQ
                mcq_options = {chr(ord("A") + i): option for i, option in enumerate(options)}
        route = detect_solver_route(premises_nl, fmt)
        parsed_questions.append(
            Type1Question(
                text=q_text,
                format=fmt,
                solver_route=route,
                mcq_options=mcq_options,
            )
        )

    return Type1ParseObject(
        premises_nl=premises_nl,
        premises_fol=premises_fol,
        questions=parsed_questions,
        parse_warnings=warnings,
        metadata={
            "question_count": len(parsed_questions),
            "premise_count": len(premises_nl),
            "has_fol_premises": bool(premises_fol),
            "format_distribution": _count_by(parsed_questions, lambda q: q.format.value),
            "route_distribution": _count_by(parsed_questions, lambda q: q.solver_route.value),
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_list(value: Optional[Any]) -> List[str]:
    """Return value as a flat list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    return [str(value)]


def _append_options(question_text: str, options: List[str]) -> str:
    """Include official choice options in the text seen by the reasoner."""
    if not options:
        return question_text
    if all(option in question_text for option in options):
        return question_text
    option_text = " ".join(
        f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(options)
    )
    return f"{question_text}\nOptions: {option_text}"


def _count_by(items: List[Any], key_fn: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts
