"""Deterministic verification and feature building for Type2 candidates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from parser.pipeline.type2_adapter import SKELETON_TEMPLATE_NAME, Type2WorldModelInput
from parser.pipeline.type2_candidate_generator import Type2CandidateGenerationResult, Type2StepPlanCandidate


TYPE2_CANDIDATE_FEATURE_NAMES = [
    "target_match_score",
    "unit_match_score",
    "dimension_consistency_score",
    "known_quantity_coverage",
    "condition_coverage",
    "relation_coverage",
    "formula_target_alignment",
    "input_availability_score",
    "step_validity_score",
    "template_domain_score",
    "geometry_condition_score",
    "uses_skeleton_penalty",
    "missing_input_penalty",
    "invalid_output_penalty",
    "warning_penalty",
    "legacy_bonus",
    "deterministic_variant_bonus",
    "prior_confidence",
    "rank_hint",
    "step_count_score",
    "parser_status_score",
    "oversimplified_scalar_penalty",
    "overall_candidate_score",
]


CONSTANTS = {"k", "g", "pi", "epsilon_0", "mu_0", "c"}
ALIAS_GROUPS = [
    {"q", "Q", "charge"},
    {"q0", "q", "test_charge"},
    {"q1", "qA", "charge_A", "q"},
    {"q2", "qB", "charge_B"},
    {"q3", "qC", "test_charge", "charge_C"},
    {"r", "d", "distance"},
    {"r12", "AB", "BA", "d", "distance"},
    {"r13", "AC", "CA", "MA", "AM", "distance_to_q1", "d", "L"},
    {"r23", "BC", "CB", "MB", "BM", "distance_to_q2", "d2", "L"},
    {"L", "side", "side_length"},
]
PENALTY_FEATURES = {
    "uses_skeleton_penalty",
    "missing_input_penalty",
    "invalid_output_penalty",
    "warning_penalty",
    "oversimplified_scalar_penalty",
}


@dataclass
class Type2CandidateVerification:
    candidate_id: str
    source: str
    template_names: list[str]
    target: str | None
    target_unit: str | None
    score: float
    confidence: float
    feature_vector: list[float]
    feature_names: list[str]
    feature_values: dict[str, float]
    verifier_status: str
    verifier_errors: list[dict[str, Any]]
    verifier_warnings: list[str]
    selected_formula_names: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Type2CandidateVerificationResult:
    problem_text: str
    target: str | None
    target_unit: str | None
    verified_candidates: list[Type2CandidateVerification]
    selected_candidate_id: str | None
    verification_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_text": self.problem_text,
            "target": self.target,
            "target_unit": self.target_unit,
            "verified_candidates": [candidate.to_dict() for candidate in self.verified_candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "verification_summary": deepcopy(self.verification_summary),
        }


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _normalize_name(name: Any) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def _normalize_unit(unit: Any) -> str:
    normalized = str(unit or "").strip().lower()
    return {
        "ω": "ohm",
        "Ω": "ohm",
        "ohms": "ohm",
        "newton": "n",
        "joule": "j",
        "farad": "f",
        "coulomb": "c",
        "volt": "v",
        "ampere": "a",
    }.get(normalized, normalized)


def _unit_dimension(unit: Any) -> str | None:
    unit_norm = _normalize_unit(unit)
    mapping = {
        "n": "force",
        "j": "energy",
        "f": "capacitance",
        "c": "charge",
        "v": "voltage",
        "a": "current",
        "ohm": "resistance",
        "m": "length",
        "s": "time",
        "m/s": "velocity",
        "m/s^2": "acceleration",
        "m/s²": "acceleration",
        "%": "percent",
    }
    return mapping.get(unit_norm)


def _target_dimension(name: Any, unit: Any = None) -> str | None:
    unit_dim = _unit_dimension(unit)
    if unit_dim:
        return unit_dim
    target = _normalize_name(name)
    if target in {"f", "f_e", "f_net", "f_on_q3", "force"}:
        return "force"
    if target in {"u_cap", "u_after", "delta_u", "energy", "p"}:
        return "energy" if target.startswith("u") or target == "energy" else "power"
    if target in {"c", "c_cap", "c_after"}:
        return "capacitance"
    if target in {"q", "q1", "q2", "q3", "q0", "q_after"}:
        return "charge"
    if target in {"v", "u", "v_after", "voltage"}:
        return "voltage"
    if target in {"i", "current"}:
        return "current"
    if target in {"r", "resistance"}:
        return "resistance"
    if target in {"theta", "angle"}:
        return "angle"
    if target in {"d", "s", "x", "distance"}:
        return "length"
    if target in {"t", "time"}:
        return "time"
    if target in {"v_final", "v_0", "v_avg", "speed", "velocity"}:
        return "velocity"
    if target in {"a", "acceleration"}:
        return "acceleration"
    if target in {"abs_error", "rel_error", "percent_error", "mean_value", "random_error"}:
        return "measurement_error"
    return None


def _compatible_targets(a: Any, b: Any, unit: Any = None) -> bool:
    left = _normalize_name(a)
    right = _normalize_name(b)
    if left == right:
        return True
    aliases = [
        {"f_e", "f_net", "f_on_q3", "f", "force"},
        {"u", "v", "v_after", "voltage"},
        {"c", "c_cap", "c_after"},
        {"q", "q1", "q2", "q3", "q0", "q_after"},
        {"d", "s", "x", "distance"},
        {"v", "v_final", "v_0", "v_avg", "speed", "velocity"},
    ]
    if any(left in group and right in group for group in aliases):
        if {left, right} <= {"u", "v", "v_after", "voltage"}:
            return _unit_dimension(unit) == "voltage"
        if {left, right} <= {"q", "q1", "q2", "q3", "q0", "q_after"}:
            return _unit_dimension(unit) == "charge"
        return True
    return False


def _extract_formula_names(candidate: Type2StepPlanCandidate) -> list[str]:
    return [
        str(step.get("formula_name"))
        for step in candidate.step_plan
        if isinstance(step, dict) and step.get("formula_name")
    ]


def _formula_lhs_names(candidate: Type2StepPlanCandidate) -> list[str]:
    lhs_names: list[str] = []
    for formula in _extract_formula_names(candidate):
        if "=" in formula:
            lhs_names.append(formula.split("=", 1)[0].strip())
    return lhs_names


def _extract_template_names(candidate: Type2StepPlanCandidate) -> list[str]:
    names = list(candidate.template_names)
    for step in candidate.step_plan:
        if isinstance(step, dict) and step.get("template_name"):
            names.append(str(step.get("template_name")))
    return sorted({name for name in names if name})


def _extract_input_vars(candidate: Type2StepPlanCandidate) -> list[str]:
    inputs: list[str] = []
    for step in candidate.step_plan:
        if not isinstance(step, dict):
            continue
        input_var = step.get("input_var") or {}
        if isinstance(input_var, dict):
            inputs.extend(str(key) for key in input_var.keys())
        elif isinstance(input_var, list):
            inputs.extend(str(item) for item in input_var)
    return inputs


def _extract_output_vars(candidate: Type2StepPlanCandidate) -> list[str]:
    outputs: list[str] = []
    for step in candidate.step_plan:
        if not isinstance(step, dict):
            continue
        output_var = step.get("output_var") or {}
        if isinstance(output_var, dict):
            outputs.extend(str(key) for key in output_var.keys())
        elif isinstance(output_var, list):
            outputs.extend(str(item) for item in output_var)
    return outputs


def _final_output_vars(candidate: Type2StepPlanCandidate) -> list[str]:
    for step in reversed(candidate.step_plan):
        if not isinstance(step, dict):
            continue
        output_var = step.get("output_var") or {}
        if isinstance(output_var, dict) and output_var:
            return [str(key) for key in output_var.keys()]
    return []


def _known_aliases(world_input: Type2WorldModelInput) -> set[str]:
    aliases = {_normalize_name(name) for name in world_input.known_quantities}
    for name, quantity in world_input.known_quantities.items():
        aliases.add(_normalize_name(name))
        if isinstance(quantity, dict):
            dim = quantity.get("dimension")
            if dim == "voltage":
                aliases.update({"u", "v"})
            elif dim == "charge":
                aliases.add("q")
            elif dim == "length":
                aliases.update({"r", "d"})
            elif dim == "force":
                aliases.add("f")
    aliases.update(CONSTANTS)
    return aliases


def _expanded_available_names(names: set[str]) -> set[str]:
    expanded = {_normalize_name(name) for name in names}
    changed = True
    normalized_groups = [{_normalize_name(alias) for alias in group} for group in ALIAS_GROUPS]
    while changed:
        changed = False
        for group in normalized_groups:
            if expanded & group and not group <= expanded:
                expanded.update(group)
                changed = True
    return expanded


def _is_name_available(name: str, available: set[str]) -> bool:
    name_norm = _normalize_name(name)
    if name_norm in available:
        return True
    expanded = _expanded_available_names(available)
    return name_norm in expanded


def _charge_variable_count(world_input: Type2WorldModelInput) -> int:
    names = {_normalize_name(name) for name in world_input.known_quantities}
    text = _world_text(world_input)
    explicit_names = {"q1", "q2", "q3", "q0", "qa", "qb", "qc"}
    count = len(names & explicit_names)
    for marker in explicit_names:
        if marker in text:
            count += 1
    if "three charge" in text or "three electric charge" in text:
        count = max(count, 3)
    if "two charge" in text or "two electric charge" in text:
        count = max(count, 2)
    return count


def _oversimplified_scalar_penalty(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    templates = " ".join(_extract_template_names(candidate)).lower()
    if "scalar_coulomb_single" not in templates:
        return 0.0
    target = _normalize_name(world_input.target)
    text = _world_text(world_input)
    charge_count = _charge_variable_count(world_input)
    vector_words = ("net", "acting on", "force vector", "resultant", "magnitude of the electric force")
    if target in {"f_net", "f_on_q3", "f_e"} and any(word in text for word in vector_words) and charge_count >= 2:
        return 1.0
    if target in {"f_net", "f_on_q3", "f_e"} and charge_count >= 3:
        return 0.8
    geometry_or_pairwise = bool(world_input.relations) or any(word in text for word in ("triangle", "right-angled", "equilateral", "vertices", "perpendicular"))
    if target == "f_net" and geometry_or_pairwise:
        return 0.5
    return 0.0


def _family(candidate: Type2StepPlanCandidate) -> str:
    text = " ".join(_extract_template_names(candidate) + _extract_formula_names(candidate)).lower()
    if SKELETON_TEMPLATE_NAME in text:
        return "skeleton"
    if "coulomb" in text or "electric_field" in text:
        return "coulomb"
    if "capacitor" in text or "capacitance" in text or "dielectric" in text or "charge_definition" in text or "voltage_definition" in text:
        return "capacitor"
    if "ohms" in text or "circuit" in text or "power_" in text or "resistance" in text:
        return "circuit"
    if "force" in text or "resultant" in text or "newton" in text:
        return "force_resultant"
    if "velocity" in text or "acceleration" in text or "speed" in text or "kinematic" in text:
        return "kinematics"
    if "error" in text or "uncertainty" in text or "mean_value" in text:
        return "measurement_error"
    if "symbolic" in text or "boolean" in text or "non_numeric" in text:
        return "non_numeric"
    return "other"


def _world_text(world_input: Type2WorldModelInput) -> str:
    return world_input.problem_text.lower()


def _labels(world_input: Type2WorldModelInput) -> str:
    return " ".join(str(item).lower() for item in world_input.domains + world_input.sub_domains + world_input.conditions)


def _has_text_or_condition(world_input: Type2WorldModelInput, *keywords: str) -> bool:
    text = _world_text(world_input) + " " + _labels(world_input)
    return any(keyword.lower() in text for keyword in keywords)


def _target_match_score(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    if not candidate.target or not world_input.target:
        return 0.0
    if _normalize_name(candidate.target) == _normalize_name(world_input.target):
        return 1.0
    if _compatible_targets(candidate.target, world_input.target, world_input.target_unit):
        return 0.7
    final_outputs = _final_output_vars(candidate)
    if final_outputs and any(_compatible_targets(output, world_input.target, world_input.target_unit) for output in final_outputs):
        return 0.7
    if candidate.target:
        return 0.3
    return 0.0


def _unit_match_score(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    candidate_unit = _normalize_unit(candidate.target_unit)
    world_unit = _normalize_unit(world_input.target_unit)
    if candidate_unit and world_unit and candidate_unit == world_unit:
        return 1.0
    candidate_dim = _unit_dimension(candidate_unit) or _target_dimension(candidate.target, candidate.target_unit)
    world_dim = _unit_dimension(world_unit) or _target_dimension(world_input.target, world_input.target_unit)
    if candidate_dim and world_dim and candidate_dim == world_dim:
        return 0.8
    if (not candidate_unit or not world_unit) and (candidate_dim or world_dim):
        return 0.5
    if not candidate_unit and not world_unit:
        return 0.5
    return 0.0


def _dimension_consistency_score(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    family = _family(candidate)
    target_dim = _target_dimension(world_input.target, world_input.target_unit)
    family_dims = {
        "coulomb": {"force", "charge", "electric_field"},
        "capacitor": {"energy", "capacitance", "charge", "voltage"},
        "circuit": {"current", "voltage", "resistance", "power"},
        "force_resultant": {"force", "angle"},
        "kinematics": {"velocity", "acceleration", "length", "time"},
        "measurement_error": {"measurement_error", "percent"},
        "non_numeric": {target_dim or "symbolic"},
        "skeleton": {target_dim or "unknown"},
        "other": {target_dim or "unknown"},
    }
    if target_dim and target_dim in family_dims.get(family, set()):
        return 1.0
    if family in {"non_numeric", "skeleton", "other"}:
        return 0.7
    if target_dim is None:
        return 0.3
    return 0.0


def _known_quantity_coverage(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    required = [_normalize_name(name) for name in _extract_input_vars(candidate)]
    if not required:
        return 0.5
    available = set(_known_aliases(world_input))
    matched = 0
    prior_outputs = {_normalize_name(output) for output in _extract_output_vars(candidate)}
    for name in required:
        if _is_name_available(name, available):
            matched += 1
        elif _is_name_available(name, prior_outputs):
            matched += 1
    return matched / len(required)


def _condition_coverage(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    templates = " ".join(_extract_template_names(candidate)).lower()
    if "right_angle" in templates:
        return 1.0 if _has_text_or_condition(world_input, "right_angle", "90", "perpendicular") else 0.2
    if "equilateral" in templates:
        return 1.0 if _has_text_or_condition(world_input, "equilateral", "60") else 0.2
    if "collinear" in templates or "opposite" in templates:
        return 1.0 if _has_text_or_condition(world_input, "collinear", "straight", "opposite") else 0.2
    if "battery_connected" in templates:
        return 1.0 if _has_text_or_condition(world_input, "battery_connected", "connected to battery") else 0.2
    if "battery_disconnected" in templates:
        return 1.0 if _has_text_or_condition(world_input, "battery_disconnected", "disconnected") else 0.2
    if "angle" in templates:
        return 1.0 if ("theta" in world_input.known_quantities or _has_text_or_condition(world_input, "angle", "degrees")) else 0.2
    return 0.6


def _relation_coverage(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    if not world_input.relations:
        return 0.6
    candidate_names = {_normalize_name(name) for name in _extract_input_vars(candidate) + _extract_output_vars(candidate)}
    for relation in world_input.relations:
        if not isinstance(relation, dict):
            continue
        relation_names = {
            _normalize_name(value)
            for key, value in relation.items()
            if key in {"left", "right", "symbol", "variable"} and value is not None
        }
        if candidate_names & relation_names:
            return 1.0
    return 0.4


def _formula_target_alignment(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    final_outputs = _final_output_vars(candidate)
    lhs_names = _formula_lhs_names(candidate)
    formula_outputs = [
        str(key)
        for step in candidate.step_plan
        if isinstance(step, dict) and step.get("type") == "formula_application"
        for key in ((step.get("output_var") or {}).keys() if isinstance(step.get("output_var"), dict) else [])
    ]
    if lhs_names and any(_normalize_name(lhs) == _normalize_name(world_input.target) for lhs in lhs_names):
        return 1.0
    if lhs_names and any(_compatible_targets(lhs, world_input.target, world_input.target_unit) for lhs in lhs_names):
        return 0.8
    if any(_normalize_name(output) == _normalize_name(world_input.target) for output in final_outputs):
        return 0.3
    if any(_compatible_targets(output, world_input.target, world_input.target_unit) for output in formula_outputs):
        return 0.3
    if any(_normalize_name(output) == _normalize_name(world_input.target) for output in final_outputs):
        return 0.3
    return 0.0 if final_outputs else 0.3


def _input_availability_score(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    available = set(_known_aliases(world_input))
    checks = 0
    valid = 0
    for step in candidate.step_plan:
        if not isinstance(step, dict):
            continue
        inputs = []
        input_var = step.get("input_var") or {}
        if isinstance(input_var, dict):
            inputs = [_normalize_name(key) for key in input_var.keys()]
        elif isinstance(input_var, list):
            inputs = [_normalize_name(item) for item in input_var]
        for name in inputs:
            checks += 1
            if _is_name_available(name, available):
                valid += 1
        output_var = step.get("output_var") or {}
        if isinstance(output_var, dict):
            available.update(_normalize_name(key) for key in output_var.keys())
    return 0.5 if checks == 0 else valid / checks


def _step_validity_score(candidate: Type2StepPlanCandidate) -> float:
    if not candidate.step_plan:
        return 0.0
    if all(isinstance(step, dict) and step.get("template_name") == SKELETON_TEMPLATE_NAME for step in candidate.step_plan):
        return 0.2
    formula_steps = [step for step in candidate.step_plan if isinstance(step, dict) and step.get("type") == "formula_application"]
    setup_only = not formula_steps and any(isinstance(step, dict) and step.get("type") == "setup" for step in candidate.step_plan)
    if setup_only:
        return 0.5
    for step in formula_steps:
        if not step.get("formula_name") or not step.get("output_var"):
            return 0.0
    has_conclusion = any(isinstance(step, dict) and step.get("type") == "conclusion" for step in candidate.step_plan)
    if len(candidate.step_plan) >= 2 and has_conclusion:
        return 1.0
    if len(formula_steps) == 1:
        return 0.8
    return 0.5


def _template_domain_score(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    family = _family(candidate)
    labels = _labels(world_input)
    target_dim = _target_dimension(world_input.target, world_input.target_unit)
    matches = {
        "coulomb": "coulomb" in labels or "electrostatic" in labels or target_dim in {"force", "charge"},
        "capacitor": "capacitor" in labels or target_dim in {"energy", "capacitance", "charge", "voltage"},
        "circuit": "ohms" in labels or "circuit" in labels or target_dim in {"current", "resistance", "power", "voltage"},
        "force_resultant": target_dim in {"force", "angle"},
        "kinematics": "mechanics" in labels or target_dim in {"velocity", "acceleration", "length", "time"},
        "measurement_error": "measurement" in labels or target_dim == "measurement_error",
    }
    if matches.get(family):
        return 1.0
    if family in {"non_numeric", "skeleton", "other"}:
        return 0.6
    return 0.0 if target_dim else 0.2


def _geometry_condition_score(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    templates = " ".join(_extract_template_names(candidate)).lower()
    if any(key in templates for key in ("right_angle", "equilateral", "collinear", "opposite")):
        return 1.0 if _condition_coverage(world_input, candidate) >= 1.0 else 0.4
    if "vector_sum" in templates:
        return 0.7
    return 0.6


def _uses_skeleton_penalty(candidate: Type2StepPlanCandidate) -> float:
    return 1.0 if any(
        isinstance(step, dict) and step.get("template_name") == SKELETON_TEMPLATE_NAME
        for step in candidate.step_plan
    ) else 0.0


def _missing_input_penalty(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    return 1.0 - _input_availability_score(world_input, candidate)


def _invalid_output_penalty(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    alignment = _formula_target_alignment(world_input, candidate)
    target_score = _target_match_score(world_input, candidate)
    if alignment == 0.0 and target_score < 0.5:
        return 1.0
    if alignment < 0.5:
        return 0.5
    return 0.0


def _warning_penalty(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> float:
    candidate_warnings = []
    for step in candidate.step_plan:
        if isinstance(step, dict) and step.get("parser_warning"):
            candidate_warnings.append(str(step.get("parser_warning")))
    verifier_errors = [
        str(error.get("error_type", ""))
        for error in world_input.parser_errors
        if isinstance(error, dict)
    ]
    warnings = " ".join(world_input.parser_warnings + candidate_warnings + verifier_errors).lower()
    if not warnings:
        return 0.0
    if "low_confidence" in warnings or "low confidence" in warnings:
        return 0.3
    if "serious" in warnings or "invalid" in warnings:
        return 0.6
    if "geometry" in warnings or "ambiguous" in warnings:
        return 0.15
    return 0.1


def _parser_status_score(world_input: Type2WorldModelInput) -> float:
    status = str(world_input.parser_status or "").upper()
    if status == "PASS":
        return 1.0
    if status in {"WARN", "UNKNOWN", ""}:
        return 0.5
    return 0.2


def _step_count_score(candidate: Type2StepPlanCandidate) -> float:
    count = len(candidate.step_plan)
    if 2 <= count <= 5:
        return 1.0
    if count == 1 or 6 <= count <= 7:
        return 0.8
    if count >= 8:
        return 0.5
    return 0.0


def _compute_overall_score(feature_values: dict[str, float]) -> float:
    positive = (
        0.13 * feature_values["target_match_score"]
        + 0.10 * feature_values["unit_match_score"]
        + 0.12 * feature_values["dimension_consistency_score"]
        + 0.11 * feature_values["known_quantity_coverage"]
        + 0.08 * feature_values["condition_coverage"]
        + 0.06 * feature_values["relation_coverage"]
        + 0.12 * feature_values["formula_target_alignment"]
        + 0.10 * feature_values["input_availability_score"]
        + 0.08 * feature_values["step_validity_score"]
        + 0.07 * feature_values["template_domain_score"]
        + 0.04 * feature_values["geometry_condition_score"]
        + 0.03 * feature_values["prior_confidence"]
        + 0.02 * feature_values["rank_hint"]
        + feature_values["legacy_bonus"]
        + feature_values["deterministic_variant_bonus"]
    )
    penalty = (
        0.20 * feature_values["uses_skeleton_penalty"]
        + 0.15 * feature_values["missing_input_penalty"]
        + 0.20 * feature_values["invalid_output_penalty"]
        + 0.10 * feature_values["warning_penalty"]
        + 0.25 * feature_values["oversimplified_scalar_penalty"]
    )
    raw_overall = max(0.0, positive - penalty)
    overall = raw_overall
    if overall > 0.97:
        strict_checks = [
            feature_values["target_match_score"] == 1.0,
            feature_values["unit_match_score"] == 1.0,
            feature_values["dimension_consistency_score"] == 1.0,
            feature_values["formula_target_alignment"] == 1.0,
            feature_values["input_availability_score"] == 1.0,
            feature_values["missing_input_penalty"] == 0.0,
            feature_values["invalid_output_penalty"] == 0.0,
            feature_values["uses_skeleton_penalty"] == 0.0,
            feature_values["warning_penalty"] == 0.0,
            feature_values["condition_coverage"] == 1.0,
            feature_values["relation_coverage"] == 1.0,
            feature_values["geometry_condition_score"] == 1.0,
        ]
        strict_excellence_score = sum(1 for check in strict_checks if check) / len(strict_checks)
        raw_excess_score = _clip((raw_overall - 0.97) / 0.15)
        overall = 0.97 + 0.02 * strict_excellence_score + 0.009 * raw_excess_score
        overall = min(overall, 0.999)
    return _clip(overall)


def _build_feature_values(world_input: Type2WorldModelInput, candidate: Type2StepPlanCandidate) -> dict[str, float]:
    values = {
        "target_match_score": _target_match_score(world_input, candidate),
        "unit_match_score": _unit_match_score(world_input, candidate),
        "dimension_consistency_score": _dimension_consistency_score(world_input, candidate),
        "known_quantity_coverage": _known_quantity_coverage(world_input, candidate),
        "condition_coverage": _condition_coverage(world_input, candidate),
        "relation_coverage": _relation_coverage(world_input, candidate),
        "formula_target_alignment": _formula_target_alignment(world_input, candidate),
        "input_availability_score": _input_availability_score(world_input, candidate),
        "step_validity_score": _step_validity_score(candidate),
        "template_domain_score": _template_domain_score(world_input, candidate),
        "geometry_condition_score": _geometry_condition_score(world_input, candidate),
        "uses_skeleton_penalty": _uses_skeleton_penalty(candidate),
        "missing_input_penalty": _missing_input_penalty(world_input, candidate),
        "invalid_output_penalty": _invalid_output_penalty(world_input, candidate),
        "warning_penalty": _warning_penalty(world_input, candidate),
        "legacy_bonus": 0.08 if candidate.source == "legacy_parser_step_plan" and world_input.parser_status == "PASS" else (0.03 if candidate.source == "legacy_parser_step_plan" else 0.0),
        "deterministic_variant_bonus": 0.04 if candidate.source == "deterministic_variant" else 0.0,
        "prior_confidence": _clip(candidate.prior_confidence),
        "rank_hint": _clip(candidate.rank_hint),
        "step_count_score": _step_count_score(candidate),
        "parser_status_score": _parser_status_score(world_input),
        "oversimplified_scalar_penalty": _oversimplified_scalar_penalty(world_input, candidate),
    }
    values["overall_candidate_score"] = _compute_overall_score(values)
    return {name: _clip(values[name], 0.0, 1.0) if name not in {"legacy_bonus", "deterministic_variant_bonus"} else values[name] for name in TYPE2_CANDIDATE_FEATURE_NAMES}


def _verifier_errors(feature_values: dict[str, float]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def add(error_type: str, severity: str, message: str) -> None:
        errors.append({"error_type": error_type, "severity": severity, "message": message})

    if feature_values["missing_input_penalty"] > 0.35:
        add("missing_inputs", "medium", "Candidate requires inputs not available from known quantities or earlier steps.")
    if feature_values["target_match_score"] < 0.5:
        add("target_mismatch", "high", "Candidate target does not match the world-model target.")
    if feature_values["unit_match_score"] < 0.5:
        add("unit_mismatch", "medium", "Candidate target unit conflicts with the world-model target unit.")
    if feature_values["dimension_consistency_score"] < 0.5:
        add("dimension_mismatch", "high", "Candidate formula family appears inconsistent with target dimension.")
    if feature_values["condition_coverage"] < 0.5:
        add("weak_condition_support", "medium", "Candidate assumes a condition not explicit in the parse.")
    if feature_values["uses_skeleton_penalty"] > 0:
        add("skeleton_candidate", "medium", "Candidate contains skeleton placeholder steps.")
    if feature_values["oversimplified_scalar_penalty"] >= 0.8:
        add(
            "oversimplified_scalar_coulomb",
            "medium",
            "Scalar Coulomb candidate is likely insufficient for a multi-charge net-force/vector-force problem.",
        )
    return errors


def _verifier_warnings(feature_values: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    if feature_values["missing_input_penalty"] > 0.15:
        warnings.append("Some candidate inputs are unavailable or symbolic.")
    if feature_values["invalid_output_penalty"] > 0:
        warnings.append("Candidate output alignment with target is weak.")
    if feature_values["warning_penalty"] > 0:
        warnings.append("Parser or candidate warnings reduce confidence.")
    if feature_values["uses_skeleton_penalty"] > 0:
        warnings.append("Skeleton placeholder candidate should rank below concrete candidates.")
    if feature_values["oversimplified_scalar_penalty"] > 0:
        warnings.append("Scalar Coulomb candidate may oversimplify a multi-charge net-force problem.")
    return warnings


def _status(feature_values: dict[str, float]) -> str:
    score = feature_values["overall_candidate_score"]
    if score >= 0.65 and feature_values["invalid_output_penalty"] < 0.5 and feature_values["uses_skeleton_penalty"] == 0:
        return "PASS"
    if score >= 0.45:
        return "WARN"
    return "FAIL"


def _summary(verified: list[Type2CandidateVerification]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for candidate in verified:
        status_counts[candidate.verifier_status] = status_counts.get(candidate.verifier_status, 0) + 1
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
    score_margin = 0.0
    if len(verified) >= 2:
        score_margin = verified[0].score - verified[1].score
    elif verified:
        score_margin = verified[0].score
    return {
        "verified_candidate_count": len(verified),
        "candidate_verifier_status_counts": status_counts,
        "candidate_source_counts": source_counts,
        "selected_score": verified[0].score if verified else 0.0,
        "selected_confidence": verified[0].confidence if verified else 0.0,
        "score_margin": score_margin,
    }


def verify_step_plan_candidates(
    world_input: Type2WorldModelInput,
    generation_result: Type2CandidateGenerationResult,
) -> Type2CandidateVerificationResult:
    """Build deterministic verification features for each generated candidate."""
    raw_verified: list[Type2CandidateVerification] = []
    scores: dict[str, float] = {}

    for candidate in generation_result.candidates:
        feature_values = _build_feature_values(world_input, candidate)
        scores[candidate.candidate_id] = feature_values["overall_candidate_score"]
        raw_verified.append(
            Type2CandidateVerification(
                candidate_id=candidate.candidate_id,
                source=candidate.source,
                template_names=_extract_template_names(candidate),
                target=candidate.target,
                target_unit=candidate.target_unit,
                score=feature_values["overall_candidate_score"],
                confidence=0.0,
                feature_vector=[feature_values[name] for name in TYPE2_CANDIDATE_FEATURE_NAMES],
                feature_names=list(TYPE2_CANDIDATE_FEATURE_NAMES),
                feature_values=feature_values,
                verifier_status=_status(feature_values),
                verifier_errors=_verifier_errors(feature_values),
                verifier_warnings=_verifier_warnings(feature_values),
                selected_formula_names=_extract_formula_names(candidate),
                metadata={
                    "candidate_generation_metadata": deepcopy(candidate.metadata),
                    "candidate_family": _family(candidate),
                },
            )
        )

    sorted_scores = sorted(scores.values(), reverse=True)
    verified: list[Type2CandidateVerification] = []
    for item in raw_verified:
        second_best = sorted_scores[1] if len(sorted_scores) > 1 and item.score == sorted_scores[0] else (sorted_scores[0] if sorted_scores else item.score)
        confidence = _clip(0.6 * item.score + 0.4 * abs(item.score - second_best))
        item.confidence = confidence
        verified.append(item)

    verified.sort(key=lambda candidate: (-candidate.score, -candidate.confidence, candidate.candidate_id))
    selected_candidate_id = verified[0].candidate_id if verified else None

    return Type2CandidateVerificationResult(
        problem_text=world_input.problem_text,
        target=world_input.target,
        target_unit=world_input.target_unit,
        verified_candidates=verified,
        selected_candidate_id=selected_candidate_id,
        verification_summary=_summary(verified),
    )
