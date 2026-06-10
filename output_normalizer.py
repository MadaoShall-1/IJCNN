"""Normalize raw pipeline outputs into the official competition response schema.

Official response format (always a single-element list):
[{
    "query_id": str,
    "answer": str,
    "unit": str,
    "explanation": str,
    "premises_used": list[int],
    "reasoning": dict | null,
}]
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Unit normalisation table (Unicode -> ASCII)
# ---------------------------------------------------------------------------

_UNIT_MAP = {
    "Ω": "ohm",    # Ω Greek capital omega
    "Ω": "ohm",    # Ω Ohm sign
    "μF": "uF",
    "µF": "uF",
    "μC": "uC",
    "µC": "uC",
    "μA": "uA",
    "µA": "uA",
    "m/s²": "m/s^2",
    "cm²": "cm^2",
}


def _normalize_unit(raw: str) -> str:
    s = raw.strip()
    for src, dst in _UNIT_MAP.items():
        s = s.replace(src, dst)
    s = s.replace("μ", "u").replace("µ", "u")
    s = s.replace("Ω", "ohm").replace("Ω", "ohm")
    s = s.replace("²", "^2").replace("³", "^3")
    return s


# ---------------------------------------------------------------------------
# Answer-option normalisation for Type 1
# ---------------------------------------------------------------------------

_YES_ALIASES = {"yes", "true"}
_NO_ALIASES = {"no", "false"}
_UNCERTAIN_ALIASES = {"uncertain", "unknown", "cannot determine", "indeterminate"}


def _normalize_answer_to_option(answer: str, options: List[str]) -> str:
    if not options:
        return answer.strip()

    lower_map = {o.lower().strip(): o for o in options}
    ans_lower = answer.strip().lower()

    if ans_lower in lower_map:
        return lower_map[ans_lower]

    if ans_lower in _YES_ALIASES and "Yes" in options:
        return "Yes"
    if ans_lower in _NO_ALIASES and "No" in options:
        return "No"
    if ans_lower in _UNCERTAIN_ALIASES and "Uncertain" in options:
        return "Uncertain"

    if "Uncertain" in options:
        return "Uncertain"
    return options[0]


# ---------------------------------------------------------------------------
# Type 1 output normalisation
# ---------------------------------------------------------------------------

def normalize_type1_output(
    query_id: str,
    raw: Dict[str, Any],
    options: List[str],
) -> Dict[str, Any]:
    answer = str(raw.get("answer", "")).strip()
    if options:
        answer = _normalize_answer_to_option(answer, options)

    premises_used = raw.get("premises_used")
    if not isinstance(premises_used, list):
        premises_used = raw.get("premises", [])
        if isinstance(premises_used, list) and premises_used and isinstance(premises_used[0], str):
            premises_used = []

    explanation = str(raw.get("explanation", "")).strip()
    if not explanation:
        explanation = "No detailed explanation available."

    reasoning = raw.get("reasoning")
    if reasoning is None:
        fol = raw.get("fol")
        cot = raw.get("cot")
        if fol or cot:
            reasoning = {}
            if fol:
                reasoning["fol"] = fol
            if cot:
                reasoning["cot"] = cot

    return {
        "query_id": query_id,
        "answer": answer,
        "unit": "",
        "explanation": explanation,
        "premises_used": premises_used if isinstance(premises_used, list) else [],
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Type 2 output normalisation
# ---------------------------------------------------------------------------

_NUMERIC_ANSWER_RE = re.compile(
    r"([+-]?\s*\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)"
)


def _split_answer_unit(raw_answer: str) -> tuple:
    raw_answer = raw_answer.strip()
    m = _NUMERIC_ANSWER_RE.match(raw_answer)
    if m:
        numeric = m.group(1).replace(" ", "")
        unit = raw_answer[m.end():].strip()
        return numeric, unit
    return raw_answer, ""


def normalize_type2_output(
    query_id: str,
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    raw_answer = str(raw.get("answer", "")).strip()
    numeric, answer_unit = _split_answer_unit(raw_answer)

    unit = str(raw.get("unit") or raw.get("unknown_unit") or answer_unit or "").strip()
    unit = _normalize_unit(unit)

    explanation = str(
        raw.get("explanation") or raw.get("chain_of_thought") or ""
    ).strip()
    if not explanation:
        explanation = "No detailed explanation available."

    reasoning = raw.get("reasoning")
    if reasoning is None:
        steps = raw.get("steps")
        cot = raw.get("chain_of_thought")
        if steps or cot:
            reasoning = {}
            if steps:
                reasoning["steps"] = steps
            if cot:
                reasoning["chain_of_thought"] = cot

    return {
        "query_id": query_id,
        "answer": numeric,
        "unit": unit,
        "explanation": explanation,
        "premises_used": [],
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def fallback_type1(query_id: str, options: List[str]) -> Dict[str, Any]:
    if options:
        answer = "Uncertain" if "Uncertain" in options else options[0]
    else:
        answer = ""
    return {
        "query_id": query_id,
        "answer": answer,
        "unit": "",
        "explanation": "The system could not derive a confident logical conclusion from the provided premises.",
        "premises_used": [],
        "reasoning": None,
    }


def fallback_type2(query_id: str) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "answer": "",
        "unit": "",
        "explanation": "The system could not compute a confident physics answer.",
        "premises_used": [],
        "reasoning": None,
    }


def fallback_unknown(query_id: str, qtype: str) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "answer": "",
        "unit": "",
        "explanation": f"Unknown query type: {qtype}",
        "premises_used": [],
        "reasoning": None,
    }
