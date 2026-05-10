# parser/validator.py

import re
from typing import List, Tuple

from .schema import ParsedQuestion
from .rule_preparser import UNIT_BY_TARGET


ALLOWED_TOPICS = {
    "unknown",
    "electric_circuit",
    "dynamics",
    "kinematics",
    "energy",
    "momentum",
    "waves",
    "thermodynamics",
    "optics",
}

ALLOWED_QUESTION_TYPES = {
    "calculation",
    "multiple_choice",
    "yes_no_uncertain",
    "open_ended",
}

ALLOWED_ANSWER_TYPES = {
    "unknown",
    "multiple_choice",
    "yes_no_uncertain",
    "numeric_value",
    "symbolic_expression",
    "open_ended",
}

NUMERIC_TARGETS = {
    "resistance", "voltage", "current", "net_force",
    "acceleration", "velocity", "distance",
    "kinetic_energy", "potential_energy", "work", "power",
    "momentum", "frequency", "wavelength", "maximum_compression"
}


GENERIC_RELATION_PATTERNS = [
    r"\bf\s*=\s*m\s*\*?\s*a\b",
    r"\bv\s*=\s*i\s*\*?\s*r\b",
    r"\br\s*=\s*v\s*/\s*i\b",
    r"\bp\s*=\s*w\s*/\s*t\b",
    r"\bp\s*=\s*i\s*\*?\s*v\b",
    r"\bke\s*=\s*1\s*/\s*2\s*m\s*v\^?2\b",
    r"\bpe\s*=\s*m\s*g\s*h\b",
    r"\b[a-z_ ]+energy\b.*\bequals\b.*\benergy\b",
]


def _relation_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_generic_relation(relation: str) -> bool:
    normalized = _relation_text(relation).replace("·", "*")
    return any(re.search(pattern, normalized) for pattern in GENERIC_RELATION_PATTERNS)


def keep_explicit_physical_relations(
    parsed: ParsedQuestion,
    question: str,
) -> ParsedQuestion:
    """
    physical_relations only stores relationships explicitly stated in the
    question text, not formulas needed to solve the question.
    """
    data = parsed.model_dump()
    question_text = _relation_text(question)
    kept = []

    for relation in data.get("physical_relations") or []:
        relation_text = _relation_text(str(relation))
        if not relation_text:
            continue
        if _is_generic_relation(relation_text):
            continue
        if relation_text in question_text:
            kept.append(relation)

    data["physical_relations"] = kept
    return ParsedQuestion(**data)


def validate_parsed_question(parsed: ParsedQuestion) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if parsed.domain != "physics":
        errors.append(f"domain should be physics, got {parsed.domain}")

    if parsed.topic not in ALLOWED_TOPICS:
        errors.append(f"invalid topic: {parsed.topic}")

    if parsed.question_type not in ALLOWED_QUESTION_TYPES:
        errors.append(f"invalid question_type: {parsed.question_type}")

    if parsed.answer_type not in ALLOWED_ANSWER_TYPES:
        errors.append(f"invalid answer_type: {parsed.answer_type}")

    if not parsed.target_quantity:
        errors.append("missing target_quantity")

    if parsed.answer_type == "multiple_choice" and not parsed.answer_options:
        errors.append("multiple_choice answer_type requires answer_options")

    if parsed.parser_confidence < 0 or parsed.parser_confidence > 1:
        errors.append("parser_confidence must be between 0 and 1")

    expected_unit = UNIT_BY_TARGET.get(parsed.target_quantity)
    if expected_unit and parsed.unit_expected not in {expected_unit, "unknown"}:
        errors.append(
            f"unit_expected mismatch for {parsed.target_quantity}: "
            f"expected {expected_unit}, got {parsed.unit_expected}"
        )

    return len(errors) == 0, errors


def repair_with_rule_hints(parsed: ParsedQuestion, rule_hints: dict) -> ParsedQuestion:
    data = parsed.model_dump()

    if data["topic"] == "unknown" and rule_hints.get("possible_topic") != "unknown":
        data["topic"] = rule_hints["possible_topic"]

    if data["subtopic"] == "unknown" and rule_hints.get("possible_subtopic") != "unknown":
        data["subtopic"] = rule_hints["possible_subtopic"]

    if data["target_quantity"] == "unknown" and rule_hints.get("possible_target_quantity") != "unknown":
        data["target_quantity"] = rule_hints["possible_target_quantity"]

    known = data.get("known_variables") or {}
    for key, value in (rule_hints.get("known_variables") or {}).items():
        known.setdefault(key, value)
    data["known_variables"] = known

    if not data.get("unknown_variables"):
        data["unknown_variables"] = rule_hints.get("unknown_variables", [])

    if data.get("unit_expected") in [None, "", "unknown"]:
        data["unit_expected"] = rule_hints.get("unit_expected", "unknown")

    if data.get("question_type") in [None, "", "unknown"]:
        data["question_type"] = rule_hints.get("question_type", "calculation")

    if data.get("answer_type") in [None, "", "unknown"]:
        data["answer_type"] = rule_hints.get("answer_type", "unknown")

    if not data.get("answer_options") and rule_hints.get("answer_options"):
        data["answer_options"] = rule_hints["answer_options"]

    data["requires_diagram_reasoning"] = bool(
        data.get("requires_diagram_reasoning")
        or rule_hints.get("requires_diagram_reasoning", False)
    )

    data["requires_formula_retrieval"] = True

    data["parser_confidence"] = max(
        float(data.get("parser_confidence", 0.0)),
        float(rule_hints.get("rule_confidence", 0.0)) * 0.8
    )

    

    return ParsedQuestion(**data)


def align_with_rule_hints(parsed: ParsedQuestion, rule_hints: dict) -> ParsedQuestion:
    """
    Keep deterministic parser fields authoritative after model parsing.
    Qwen may enrich free-text fields, but direct text classifications should
    stay consistent with the rule hints.
    """
    data = parsed.model_dump()

    for field, hint_key in [
        ("topic", "possible_topic"),
        ("subtopic", "possible_subtopic"),
        ("target_quantity", "possible_target_quantity"),
    ]:
        hint_value = rule_hints.get(hint_key)
        if hint_value and hint_value != "unknown":
            data[field] = hint_value

    for field in [
        "question_type",
        "answer_type",
        "unit_expected",
        "requires_diagram_reasoning",
        "answer_options",
    ]:
        hint_value = rule_hints.get(field)
        if hint_value is not None:
            data[field] = hint_value

    known = data.get("known_variables") or {}
    for key, value in (rule_hints.get("known_variables") or {}).items():
        known[key] = value
    data["known_variables"] = known

    hint_unknowns = rule_hints.get("unknown_variables") or []
    if hint_unknowns:
        data["unknown_variables"] = hint_unknowns

    data["parser_confidence"] = max(
        float(data.get("parser_confidence", 0.0)),
        float(rule_hints.get("rule_confidence", 0.0)),
    )

    return ParsedQuestion(**data)
