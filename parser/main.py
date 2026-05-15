"""Stage 0 deterministic-first physics parser entrypoint."""

from __future__ import annotations

from typing import Any, Dict

from .condition_extractor import extract_conditions
from .domain_classifier import classify_domain
from .error_logger import log_verifier_failure
from .llm_fallback import LLMFallbackParser
from .parse_verifier import verify_parse
from .question_type_classifier import classify_question_type, QUESTION_TYPE_NUMERIC
from .rule_extractor import extract_quantities, extract_relations
from .target_detector import detect_target
from .template_fallback import propose_step_plan


SKELETON_TEMPLATE_NAME = "skeleton_placeholder"
# Intentionally < 0.5 so the verifier's low_confidence check still fires
# and the parse remains FAIL. The skeleton's purpose is to suppress the
# redundant 'invalid_final_step: empty step_plan' error so downstream
# failure analysis can cluster on the true root cause (missing template,
# missing extractor, etc.) rather than the same surface symptom.
SKELETON_CONFIDENCE = 0.30


def _empty_parse(problem_text: str) -> Dict[str, Any]:
    return {
        "problem_text": problem_text,
        "domains": [],
        "sub_domains": [],
        "domain_confidence": 0.0,
        "question_type": QUESTION_TYPE_NUMERIC,
        "question_type_confidence": 0.0,
        "question_type_triggers": [],
        "known_quantities": {},
        "conditions": [],
        "relations": [],
        "unknown_quantity": None,
        "unknown_unit": None,
        "step_plan": [],
        "plan_confidence": 0.0,
        "parser_warnings": [],
        "vso": {},
        "metadata": {
            "used_rule_based": False,
            "used_template_fallback": False,
            "used_template_names": [],
            "used_llm_fallback": False,
            "used_skeleton_fallback": False,
            "extracted_relation_count": 0,
            "extracted_uncertainty_count": 0,
            "coverage_ignored_numbers": [],
            "verifier_status": "FAIL",
            "verifier_errors": [],
        },
    }


def _build_vso(known_quantities: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {
        name: {
            "value": quantity["value"],
            "unit_symbol": quantity["unit_symbol"],
            "unit_name": quantity["unit_name"],
            "dimension": quantity["dimension"],
            "defined_at": "stage_0",
            "updated_at": "stage_0",
        }
        for name, quantity in known_quantities.items()
    }


def _apply_verifier(parse_object: Dict[str, Any]) -> Dict[str, Any]:
    result = verify_parse(parse_object)
    parse_object["metadata"]["verifier_status"] = result.status
    parse_object["metadata"]["verifier_errors"] = [error.to_dict() for error in result.errors]
    for warning in result.warnings:
        if warning not in parse_object["parser_warnings"]:
            parse_object["parser_warnings"].append(warning)
    return result.to_dict()


def _apply_template(parse_object: Dict[str, Any]) -> None:
    # Make sub-domain classifications visible to template matchers without
    # changing the propose_step_plan signature. Templates can then gate on
    # phrases like 'resonance' or 'parallel_circuit' irrespective of whether
    # they came from condition_extractor or domain_classifier.
    augmented_conditions = list(parse_object.get("conditions") or [])
    for sub in parse_object.get("sub_domains") or []:
        sub_str = str(sub)
        if sub_str and sub_str not in augmented_conditions:
            augmented_conditions.append(sub_str)
    plan, confidence = propose_step_plan(
        parse_object["known_quantities"],
        parse_object["unknown_quantity"],
        augmented_conditions,
        parse_object.get("relations", []),
    )
    if plan and confidence > float(parse_object.get("plan_confidence") or 0.0):
        parse_object["step_plan"] = plan
        parse_object["plan_confidence"] = confidence
        parse_object["metadata"]["used_template_fallback"] = True
        template_names = sorted({str(step.get("template_name")) for step in plan if step.get("template_name")})
        parse_object["metadata"]["used_template_names"] = template_names
        for step in plan:
            warning = step.get("parser_warning")
            if warning and warning not in parse_object["parser_warnings"]:
                parse_object["parser_warnings"].append(str(warning))


def _ensure_conclusion_step(parse_object: Dict[str, Any]) -> None:
    """Append a conclusion only after an executable step already exists."""
    target = parse_object.get("unknown_quantity")
    steps = parse_object.get("step_plan") or []
    if not target or not steps:
        if target and not steps:
            warning = "No executable step_plan generated; template coverage missing."
            if warning not in parse_object["parser_warnings"]:
                parse_object["parser_warnings"].append(warning)
        return
    if not any(step.get("type") in {"formula_application", "calculation"} for step in steps):
        return
    final_outputs = steps[-1].get("output_var") or {}
    if steps[-1].get("type") == "conclusion" and target in final_outputs:
        return
    steps.append(
        {
            "step_id": f"step_{len(steps) + 1}",
            "goal": f"Report the final value of {target}.",
            "type": "conclusion",
            "template_name": "ensure_conclusion_step",
            "input_var": {str(target): str(target)},
            "output_var": {str(target): str(target)},
            "confidence": 0.84,
        }
    )


def _ensure_skeleton_step_plan(parse_object: Dict[str, Any]) -> None:
    """Emit a labeled placeholder step plan when a target exists but no template fired.

    Trade-off:
      * Suppresses the redundant ``invalid_final_step: empty plan`` verifier error
        so failure clusters reflect the true root cause (missing template,
        unrecognized phrasing, etc.) rather than this shared downstream symptom.
      * Hands a structurally valid stub forward for Stage 1+ / LLM fallback to
        overwrite, instead of an empty list.

    Non-goals:
      * This is NOT a way to produce a PASS. The skeleton's plan_confidence is
        deliberately below the verifier's 0.5 threshold, so the parse still
        fails with ``low_confidence``. The PASS set must stay uncontaminated.

    Marker:
      ``metadata['used_skeleton_fallback'] = True``. Do NOT set
      ``used_template_fallback`` — that flag means a real formula template ran.
    """
    target = parse_object.get("unknown_quantity")
    if not target:
        return
    if parse_object.get("step_plan"):
        return
    if parse_object["metadata"].get("used_template_fallback"):
        return

    target_str = str(target)
    skeleton: list = [
        {
            "step_id": "step_1",
            "goal": f"Identify known quantities relevant to {target_str}.",
            "type": "setup",
            "template_name": SKELETON_TEMPLATE_NAME,
            "input_var": {},
            "output_var": {},
            "confidence": SKELETON_CONFIDENCE,
        },
        {
            "step_id": "step_2",
            "goal": (
                f"Apply the governing formula for {target_str} "
                "(template missing — Stage 1 must resolve)."
            ),
            "type": "formula_application",
            "formula_name": "TBD",
            "template_name": SKELETON_TEMPLATE_NAME,
            "input_var": {},
            "output_var": {target_str: "TBD"},
            "confidence": SKELETON_CONFIDENCE,
        },
        {
            "step_id": "step_3",
            "goal": f"Report the final value of {target_str}.",
            "type": "conclusion",
            "template_name": SKELETON_TEMPLATE_NAME,
            "input_var": {target_str: target_str},
            "output_var": {target_str: target_str},
            "confidence": SKELETON_CONFIDENCE,
        },
    ]
    parse_object["step_plan"] = skeleton
    parse_object["plan_confidence"] = SKELETON_CONFIDENCE
    parse_object["metadata"]["used_skeleton_fallback"] = True
    warning = (
        f"No matching template for target {target_str}; emitted skeleton placeholder."
    )
    if warning not in parse_object["parser_warnings"]:
        parse_object["parser_warnings"].append(warning)


def parse_problem(
    problem_text: str,
    use_llm_fallback: bool = False,
    log_failures: bool = True,
) -> Dict[str, Any]:
    """Parse a raw physics problem into a verified Stage 0 parse object."""
    if not isinstance(problem_text, str) or not problem_text.strip():
        raise ValueError("problem_text must be a non-empty string")

    parse_object = _empty_parse(problem_text)

    # Stage 0.4.1: question-type triage runs first so verifier can route correctly.
    qt_result = classify_question_type(problem_text)
    parse_object["question_type"] = qt_result["question_type"]
    parse_object["question_type_confidence"] = qt_result["question_type_confidence"]
    parse_object["question_type_triggers"] = qt_result["question_type_triggers"]

    known_quantities = extract_quantities(problem_text)
    parse_object["known_quantities"] = known_quantities
    parse_object["relations"] = extract_relations(problem_text)
    parse_object["metadata"]["extracted_relation_count"] = len(parse_object["relations"])
    parse_object["metadata"]["extracted_uncertainty_count"] = sum(
        1 for relation in parse_object["relations"] if relation.get("type") == "uncertainty"
    )
    parse_object["metadata"]["used_rule_based"] = True

    conditions, implied_quantities = extract_conditions(problem_text, known_quantities)
    parse_object["conditions"] = conditions
    if sum(1 for relation in parse_object["relations"] if relation.get("type") == "equation") >= 2 and "system_of_equations" not in parse_object["conditions"]:
        parse_object["conditions"].append("system_of_equations")
    parse_object["known_quantities"].update(implied_quantities)

    unknown_quantity, unknown_unit = detect_target(problem_text)
    parse_object["unknown_quantity"] = unknown_quantity
    parse_object["unknown_unit"] = unknown_unit

    domains, sub_domains, confidence = classify_domain(problem_text)
    parse_object["domains"] = domains
    parse_object["sub_domains"] = sub_domains
    parse_object["domain_confidence"] = confidence
    parse_object["vso"] = _build_vso(parse_object["known_quantities"])

    _apply_template(parse_object)
    _ensure_conclusion_step(parse_object)
    _ensure_skeleton_step_plan(parse_object)
    verifier_result = _apply_verifier(parse_object)
    if verifier_result["status"] in ("PASS", "PASS_NON_NUMERIC"):
        return parse_object

    if log_failures:
        log_verifier_failure(problem_text, parse_object, verifier_result)

    if not parse_object["metadata"]["used_template_fallback"]:
        _apply_template(parse_object)
        _ensure_conclusion_step(parse_object)
        _ensure_skeleton_step_plan(parse_object)
        verifier_result = _apply_verifier(parse_object)
        if verifier_result["status"] in ("PASS", "PASS_NON_NUMERIC"):
            return parse_object
        if log_failures:
            log_verifier_failure(problem_text, parse_object, verifier_result)

    if use_llm_fallback:
        fallback = LLMFallbackParser()
        parse_object = fallback.complete_parse(problem_text, parse_object, verifier_result["errors"])
        parse_object.setdefault("metadata", {})["used_llm_fallback"] = True
        verifier_result = _apply_verifier(parse_object)
        if verifier_result["status"] == "FAIL" and log_failures:
            log_verifier_failure(problem_text, parse_object, verifier_result)

    return parse_object


def parse_question(question: str, use_qwen: bool = False, load_4bit: bool = True) -> Dict[str, Any]:
    """Backward-compatible wrapper for older callers."""
    return parse_problem(question, use_llm_fallback=use_qwen)