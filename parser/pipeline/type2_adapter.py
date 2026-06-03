"""Adapter from Stage 0 parser output to Type2 world-model input."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any


SKELETON_TEMPLATE_NAME = "skeleton_placeholder"


@dataclass
class Type2WorldModelInput:
    problem_text: str
    question_type: str
    domains: list[str]
    sub_domains: list[str]
    domain_confidence: float
    known_quantities: dict[str, dict[str, Any]]
    conditions: list[str]
    relations: list[dict[str, Any]]
    target: str | None
    target_unit: str | None
    step_plan: list[dict[str, Any]]
    plan_confidence: float
    vso: dict[str, dict[str, Any]]
    parser_status: str
    parser_errors: list[dict[str, Any]]
    parser_warnings: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Type2AdapterDiagnostics:
    has_problem_text: bool
    has_target: bool
    has_known_quantities: bool
    has_step_plan: bool
    has_real_step_plan: bool
    uses_skeleton_fallback: bool
    verifier_status: str
    error_types: list[str]
    warning_count: int
    quantity_count: int
    relation_count: int
    condition_count: int
    step_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _copy_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return deepcopy(value)
    return []


def _copy_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    return {}


def _copy_nested_dict(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return deepcopy(value)


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def adapt_parse_object(parse_object: dict[str, Any]) -> Type2WorldModelInput:
    """Convert a Stage 0 parse object to a stable Type2 world-model input."""
    parse_copy = _copy_dict(parse_object)
    metadata = _copy_dict(parse_copy.get("metadata"))

    return Type2WorldModelInput(
        problem_text=_string_or_empty(parse_copy.get("problem_text")),
        question_type=_string_or_empty(parse_copy.get("question_type")),
        domains=[str(item) for item in _copy_list(parse_copy.get("domains"))],
        sub_domains=[str(item) for item in _copy_list(parse_copy.get("sub_domains"))],
        domain_confidence=_float_or_zero(parse_copy.get("domain_confidence")),
        known_quantities=_copy_nested_dict(parse_copy.get("known_quantities")),
        conditions=[str(item) for item in _copy_list(parse_copy.get("conditions"))],
        relations=_copy_list(parse_copy.get("relations")),
        target=_optional_string(parse_copy.get("unknown_quantity")),
        target_unit=_optional_string(parse_copy.get("unknown_unit")),
        step_plan=_copy_list(parse_copy.get("step_plan")),
        plan_confidence=_float_or_zero(parse_copy.get("plan_confidence")),
        vso=_copy_nested_dict(parse_copy.get("vso")),
        parser_status=_string_or_empty(metadata.get("verifier_status")),
        parser_errors=_copy_list(metadata.get("verifier_errors")),
        parser_warnings=[str(item) for item in _copy_list(parse_copy.get("parser_warnings"))],
        metadata=metadata,
    )


def diagnose_world_model_input(world_input: Type2WorldModelInput) -> Type2AdapterDiagnostics:
    """Summarize adapter output completeness for downstream Type2 stages."""
    has_problem_text = bool(world_input.problem_text.strip())
    has_target = world_input.target is not None and bool(str(world_input.target).strip())
    has_known_quantities = bool(world_input.known_quantities)
    has_step_plan = bool(world_input.step_plan)

    has_real_step_plan = any(
        step.get("template_name") != SKELETON_TEMPLATE_NAME
        for step in world_input.step_plan
        if isinstance(step, dict)
    )
    uses_skeleton_fallback = bool(world_input.metadata.get("used_skeleton_fallback")) or any(
        isinstance(step, dict) and step.get("template_name") == SKELETON_TEMPLATE_NAME
        for step in world_input.step_plan
    )

    error_types: list[str] = []
    seen_error_types: set[str] = set()
    for error in world_input.parser_errors:
        if not isinstance(error, dict):
            continue
        error_type = error.get("error_type")
        if error_type is None:
            continue
        error_type_str = str(error_type)
        if error_type_str not in seen_error_types:
            seen_error_types.add(error_type_str)
            error_types.append(error_type_str)

    return Type2AdapterDiagnostics(
        has_problem_text=has_problem_text,
        has_target=has_target,
        has_known_quantities=has_known_quantities,
        has_step_plan=has_step_plan,
        has_real_step_plan=has_real_step_plan,
        uses_skeleton_fallback=uses_skeleton_fallback,
        verifier_status=world_input.parser_status,
        error_types=error_types,
        warning_count=len(world_input.parser_warnings),
        quantity_count=len(world_input.known_quantities),
        relation_count=len(world_input.relations),
        condition_count=len(world_input.conditions),
        step_count=len(world_input.step_plan),
    )


def parse_and_adapt(
    problem_text: str,
    use_llm_fallback: bool = False,
    log_failures: bool = False,
) -> tuple[Type2WorldModelInput, Type2AdapterDiagnostics]:
    """Run Stage 0 parser, adapt its output, and return diagnostics."""
    try:
        from parser.main import parse_problem
    except ImportError:
        from main import parse_problem  # type: ignore

    parse_object = parse_problem(
        problem_text,
        use_llm_fallback=use_llm_fallback,
        log_failures=log_failures,
    )
    world_input = adapt_parse_object(parse_object)
    diagnostics = diagnose_world_model_input(world_input)
    return world_input, diagnostics
