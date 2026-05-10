# parser/main.py

from typing import Optional, Dict, Any

from .schema import ParsedQuestion
from .rule_preparser import rule_preparse
from .qwen_parser import QwenStructuredParser
from .validator import (
    validate_parsed_question,
    repair_with_rule_hints,
    align_with_rule_hints,
    keep_explicit_physical_relations,
)


_parser_instance: Optional[QwenStructuredParser] = None


def get_qwen_parser(load_4bit: bool = True) -> QwenStructuredParser:
    global _parser_instance

    if _parser_instance is None:
        _parser_instance = QwenStructuredParser(load_4bit=load_4bit)

    return _parser_instance


def fallback_rule_only_parse(question: str, rule_hints: Dict[str, Any]) -> ParsedQuestion:
    """
    Last-resort parser if Qwen output is invalid.
    This keeps the pipeline alive.
    """

    return ParsedQuestion(
        domain="physics",
        topic=rule_hints.get("possible_topic", "unknown"),
        subtopic=rule_hints.get("possible_subtopic", "unknown"),
        question_type=rule_hints.get("question_type", "calculation"),
        target_quantity=rule_hints.get("possible_target_quantity", "unknown"),
        known_variables=rule_hints.get("known_variables", {}),
        unknown_variables=rule_hints.get("unknown_variables", []),
        answer_type=rule_hints.get("answer_type", "unknown"),
        unit_expected=rule_hints.get("unit_expected", "unknown"),
        requires_diagram_reasoning=rule_hints.get("requires_diagram_reasoning", False),
        requires_formula_retrieval=True,
        answer_options=rule_hints.get("answer_options"),
        implicit_conditions=[],
        physical_relations=[],
        parser_confidence=rule_hints.get("rule_confidence", 0.3),
    )


def parse_question(
    question: str,
    use_qwen: bool = True,
    load_4bit: bool = True,
) -> Dict[str, Any]:
    rule_hints = rule_preparse(question)

    parsed = None
    parser_source = "unknown"
    errors = []

    if use_qwen:
        try:
            qwen_parser = get_qwen_parser(load_4bit=load_4bit)
            parsed = qwen_parser.parse(question, rule_hints)
            parser_source = "qwen"
        except Exception as e:
            errors.append(f"qwen_parse_failed: {str(e)}")

    if parsed is None:
        parsed = fallback_rule_only_parse(question, rule_hints)
        parser_source = "rule_fallback"
    else:
        parsed = align_with_rule_hints(parsed, rule_hints)
        parsed = keep_explicit_physical_relations(parsed, question)
        parser_source += "+rule_align"

    ok, validation_errors = validate_parsed_question(parsed)

    if not ok:
        errors.extend(validation_errors)

        # Try deterministic repair.
        repaired = repair_with_rule_hints(parsed, rule_hints)
        repaired_ok, repaired_errors = validate_parsed_question(repaired)

        if repaired_ok:
            parsed = keep_explicit_physical_relations(repaired, question)
            ok = True
            parser_source += "+rule_repair"
        else:
            errors.extend([f"after_repair: {e}" for e in repaired_errors])

            # Final fallback.
            parsed = fallback_rule_only_parse(question, rule_hints)
            ok, final_errors = validate_parsed_question(parsed)
            errors.extend([f"final_fallback: {e}" for e in final_errors])
            parser_source = "rule_fallback_final"

    result = parsed.model_dump()
    result["parser_valid"] = ok
    result["parser_errors"] = errors
    result["parser_source"] = parser_source
    result["rule_hints"] = rule_hints

    return result
