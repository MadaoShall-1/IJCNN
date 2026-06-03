"""Deterministic Type2 step-plan candidate generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from parser.pipeline.type2_adapter import SKELETON_TEMPLATE_NAME, Type2WorldModelInput
from parser.pipeline.type2_llm_canonicalizer import canonicalize_llm_fallback_candidate


@dataclass
class Type2StepPlanCandidate:
    candidate_id: str
    source: str
    template_names: list[str]
    target: str | None
    target_unit: str | None
    step_plan: list[dict[str, Any]]
    prior_confidence: float
    rank_hint: float
    generation_notes: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Type2CandidateGenerationResult:
    problem_text: str
    target: str | None
    target_unit: str | None
    candidates: list[Type2StepPlanCandidate]
    selected_candidate_id: str | None
    generation_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_text": self.problem_text,
            "target": self.target,
            "target_unit": self.target_unit,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "generation_summary": deepcopy(self.generation_summary),
        }


def _copy_step_plan(step_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return deepcopy(step_plan) if isinstance(step_plan, list) else []


def _infer_template_names(step_plan: list[dict[str, Any]], metadata: dict[str, Any]) -> list[str]:
    metadata_names = metadata.get("used_template_names")
    if isinstance(metadata_names, list) and metadata_names:
        return sorted({str(name) for name in metadata_names if name})
    return sorted(
        {
            str(step.get("template_name"))
            for step in step_plan
            if isinstance(step, dict) and step.get("template_name")
        }
    )


def _formula_step(
    step_id: str,
    goal: str,
    formula: str,
    inputs: list[str],
    output: str,
    template_name: str,
    confidence: float = 0.74,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "goal": goal,
        "type": "formula_application",
        "formula_name": formula,
        "template_name": template_name,
        "input_var": {name: name for name in inputs},
        "output_var": {output: formula},
        "confidence": confidence,
    }


def _setup_step(
    step_id: str,
    goal: str,
    template_name: str,
    outputs: dict[str, str] | None = None,
    confidence: float = 0.70,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "goal": goal,
        "type": "setup",
        "template_name": template_name,
        "input_var": {},
        "output_var": outputs or {},
        "confidence": confidence,
    }


def _conclusion_step(step_id: str, target: str | None, template_name: str) -> dict[str, Any]:
    output = str(target or "target")
    return {
        "step_id": step_id,
        "goal": f"Report the final value of {output}.",
        "type": "conclusion",
        "template_name": template_name,
        "input_var": {output: output},
        "output_var": {output: output},
        "confidence": 0.72,
    }


def _plan(template_name: str, steps: list[dict[str, Any]], target: str | None) -> list[dict[str, Any]]:
    if target and (not steps or steps[-1].get("type") != "conclusion"):
        steps.append(_conclusion_step(f"step_{len(steps) + 1}", target, template_name))
    return steps


def _make_candidate(
    candidates: list[Type2StepPlanCandidate],
    source: str,
    template_names: list[str],
    target: str | None,
    target_unit: str | None,
    step_plan: list[dict[str, Any]],
    prior_confidence: float,
    rank_hint: float,
    generation_notes: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    index = len(candidates)
    safe_source = source.replace(" ", "_")
    candidates.append(
        Type2StepPlanCandidate(
            candidate_id=f"candidate_{safe_source}_{index}",
            source=source,
            template_names=template_names,
            target=target,
            target_unit=target_unit,
            step_plan=_copy_step_plan(step_plan),
            prior_confidence=prior_confidence,
            rank_hint=rank_hint,
            generation_notes=list(generation_notes),
            metadata=deepcopy(metadata or {}),
        )
    )


def _text(world_input: Type2WorldModelInput) -> str:
    return world_input.problem_text.lower()


def _all_labels(world_input: Type2WorldModelInput) -> set[str]:
    return {str(item).lower() for item in world_input.domains + world_input.sub_domains + world_input.conditions}


def _domain_contains(world_input: Type2WorldModelInput, *keywords: str) -> bool:
    domains = [str(domain).lower() for domain in world_input.domains]
    return any(keyword.lower() in domain for domain in domains for keyword in keywords)


def _subdomain_contains(world_input: Type2WorldModelInput, *keywords: str) -> bool:
    subdomains = [str(subdomain).lower() for subdomain in world_input.sub_domains]
    return any(keyword.lower() in subdomain for subdomain in subdomains for keyword in keywords)


def _text_contains(world_input: Type2WorldModelInput, *keywords: str) -> bool:
    text = _text(world_input)
    return any(keyword.lower() in text for keyword in keywords)


def _legacy_templates_contain(world_input: Type2WorldModelInput, *keywords: str) -> bool:
    template_names = _infer_template_names(world_input.step_plan, world_input.metadata)
    return any(
        keyword.lower() in template_name.lower()
        for template_name in template_names
        for keyword in keywords
    )


def _quantity_names_contain(world_input: Type2WorldModelInput, *names: str) -> bool:
    known_names = {name.lower() for name in world_input.known_quantities}
    return any(name.lower() in known_names for name in names)


def _target_in(world_input: Type2WorldModelInput, *targets: str) -> bool:
    return world_input.target in set(targets)


def _has_template(world_input: Type2WorldModelInput, template_name: str) -> bool:
    return any(
        isinstance(step, dict) and step.get("template_name") == template_name
        for step in world_input.step_plan
    )


def _contains_condition(world_input: Type2WorldModelInput, condition: str) -> bool:
    condition = condition.lower()
    return condition in _all_labels(world_input) or condition in _text(world_input)


def _target_is(world_input: Type2WorldModelInput, *targets: str) -> bool:
    return _target_in(world_input, *targets)


def _has_quantity(world_input: Type2WorldModelInput, *names: str) -> bool:
    return all(name in world_input.known_quantities for name in names)


def _quantity_names_by_dimension(world_input: Type2WorldModelInput, dimension: str) -> list[str]:
    return [
        name
        for name, quantity in world_input.known_quantities.items()
        if isinstance(quantity, dict) and quantity.get("dimension") == dimension
    ]


def _has_any_quantity(world_input: Type2WorldModelInput, *names: str) -> bool:
    return any(name in world_input.known_quantities for name in names)


def _first_existing(world_input: Type2WorldModelInput, *names: str) -> str | None:
    for name in names:
        if name in world_input.known_quantities:
            return name
    return None


def _has_angle_near_90(world_input: Type2WorldModelInput) -> bool:
    quantity = world_input.known_quantities.get("theta")
    if not isinstance(quantity, dict):
        return False
    value = quantity.get("value")
    try:
        return abs(float(value) - 90.0) < 1e-6
    except (TypeError, ValueError):
        return False


def _trigger_coulomb(world_input: Type2WorldModelInput) -> bool:
    charges = _quantity_names_by_dimension(world_input, "charge")
    distances = _quantity_names_by_dimension(world_input, "length")
    charge_evidence = _quantity_names_contain(world_input, "q", "q1", "q2", "q3", "q0", "Q") or bool(charges)
    capacitor_target = _target_in(world_input, "U_cap", "C_cap", "Q", "V", "U_after", "V_after", "Q_after", "C_after", "delta_U")
    capacitor_evidence = (
        _subdomain_contains(world_input, "capacitor", "dielectric_capacitor", "capacitor_network")
        or _legacy_templates_contain(world_input, "capacitor", "capacitance", "dielectric")
        or _text_contains(world_input, "capacitor", "capacitance", "dielectric")
        or _quantity_names_contain(world_input, "C_cap")
    )
    explicit_coulomb_evidence = (
        _domain_contains(world_input, "coulomb", "electrostatic", "electric_force")
        or _subdomain_contains(world_input, "coulomb", "electrostatic", "electric_force")
        or _text_contains(world_input, "charge", "charges", "coulomb", "test charge")
        or _legacy_templates_contain(world_input, "coulomb", "electric_field")
    )
    strong_coulomb_evidence = (
        _domain_contains(world_input, "coulomb", "electrostatic", "electric_force")
        or _subdomain_contains(world_input, "electrostatic", "electric_force")
        or _text_contains(world_input, "coulomb", "test charge")
        or _legacy_templates_contain(world_input, "electric_field")
    )
    if capacitor_target and capacitor_evidence and not strong_coulomb_evidence:
        return False
    if _target_in(world_input, "F_net") and not charge_evidence and not strong_coulomb_evidence:
        return False
    force_or_field_target = _target_in(world_input, "F_e", "F_net", "F_on_q3", "q", "E")
    electric_evidence = (
        explicit_coulomb_evidence
        or (_domain_contains(world_input, "electricity") and force_or_field_target and charge_evidence)
    )
    if _target_in(world_input, "F_net", "F_on_q3") and not (charge_evidence or electric_evidence):
        return False
    return (
        electric_evidence
        or _target_in(world_input, "F_e", "q", "E")
        or (_target_in(world_input, "F_net", "F_on_q3") and charge_evidence)
        or (charge_evidence and bool(distances))
    )


def _add_coulomb_variants(
    world_input: Type2WorldModelInput,
    candidates: list[Type2StepPlanCandidate],
) -> None:
    if not _trigger_coulomb(world_input):
        return
    target = world_input.target or "F_e"
    charges = _quantity_names_by_dimension(world_input, "charge") or ["q1", "q2", "q3"]
    distances = _quantity_names_by_dimension(world_input, "length") or ["r"]

    _make_candidate(
        candidates,
        "deterministic_variant",
        ["scalar_coulomb_single"],
        world_input.target,
        world_input.target_unit,
        _plan(
            "scalar_coulomb_single",
            [_formula_step("step_1", "Apply scalar Coulomb force relation.", "F = k * abs(q1*q2) / r^2", charges[:2] + distances[:1] + ["k"], target, "scalar_coulomb_single")],
            world_input.target,
        ),
        0.68,
        0.68,
        ["Deterministic Coulomb scalar candidate."],
    )
    if len(charges) >= 3 or _target_is(world_input, "F_net", "F_on_q3"):
        _make_candidate(
            candidates,
            "deterministic_variant",
            ["coulomb_pairwise_vector_sum"],
            world_input.target,
            world_input.target_unit,
            _plan(
                "coulomb_pairwise_vector_sum",
                [
                    _formula_step("step_1", "Compute force between q1 and q3.", "F_13 = k * abs(q1*q3) / r13^2", ["q1", "q3", "r13", "k"], "F_13", "coulomb_pairwise_vector_sum"),
                    _formula_step("step_2", "Compute force between q2 and q3.", "F_23 = k * abs(q2*q3) / r23^2", ["q2", "q3", "r23", "k"], "F_23", "coulomb_pairwise_vector_sum"),
                    _formula_step("step_3", "Combine pairwise force vectors.", "F_net = vector_sum(F_13, F_23)", ["F_13", "F_23"], target, "coulomb_pairwise_vector_sum"),
                ],
                world_input.target,
            ),
            0.66,
            0.70,
            ["Deterministic Coulomb pairwise vector-sum candidate."],
        )
    if _contains_condition(world_input, "right_angle") or "perpendicular" in _text(world_input) or "90" in _text(world_input):
        _make_candidate(
            candidates,
            "deterministic_variant",
            ["coulomb_right_angle_resultant"],
            world_input.target,
            world_input.target_unit,
            _plan(
                "coulomb_right_angle_resultant",
                [
                    _formula_step("step_1", "Compute Coulomb pair force F_13.", "F_13 = k * abs(q1*q3) / r13^2", ["q1", "q3", "r13", "k"], "F_13", "coulomb_right_angle_resultant"),
                    _formula_step("step_2", "Compute Coulomb pair force F_23.", "F_23 = k * abs(q2*q3) / r23^2", ["q2", "q3", "r23", "k"], "F_23", "coulomb_right_angle_resultant"),
                    _formula_step("step_3", "Combine perpendicular force magnitudes.", "F_net = sqrt(F_13^2 + F_23^2)", ["F_13", "F_23"], target, "coulomb_right_angle_resultant"),
                ],
                world_input.target,
            ),
            0.69,
            0.73,
            ["Right-angle Coulomb resultant candidate."],
        )
    if _contains_condition(world_input, "equilateral_triangle"):
        _make_candidate(
            candidates,
            "deterministic_variant",
            ["coulomb_equilateral_resultant"],
            world_input.target,
            world_input.target_unit,
            _plan(
                "coulomb_equilateral_resultant",
                [
                    _formula_step("step_1", "Compute Coulomb pair force F_13.", "F_13 = k * abs(q1*q3) / r13^2", ["q1", "q3", "r13", "k"], "F_13", "coulomb_equilateral_resultant"),
                    _formula_step("step_2", "Compute Coulomb pair force F_23.", "F_23 = k * abs(q2*q3) / r23^2", ["q2", "q3", "r23", "k"], "F_23", "coulomb_equilateral_resultant"),
                    _formula_step("step_3", "Combine forces with 60 degree included angle.", "F_net = sqrt(F_13^2 + F_23^2 + 2*F_13*F_23*cos(60deg))", ["F_13", "F_23"], target, "coulomb_equilateral_resultant"),
                ],
                world_input.target,
            ),
            0.69,
            0.73,
            ["Equilateral-triangle Coulomb resultant candidate."],
        )
    if any(phrase in _text(world_input) for phrase in ("collinear", "straight line", "opposite sides")):
        _make_candidate(
            candidates,
            "deterministic_variant",
            ["coulomb_collinear_opposite"],
            world_input.target,
            world_input.target_unit,
            _plan(
                "coulomb_collinear_opposite",
                [
                    _formula_step("step_1", "Compute Coulomb pair force F_13.", "F_13 = k * abs(q1*q3) / r13^2", ["q1", "q3", "r13", "k"], "F_13", "coulomb_collinear_opposite"),
                    _formula_step("step_2", "Compute Coulomb pair force F_23.", "F_23 = k * abs(q2*q3) / r23^2", ["q2", "q3", "r23", "k"], "F_23", "coulomb_collinear_opposite"),
                    _formula_step("step_3", "Combine opposite collinear force magnitudes.", "F_net = abs(F_13 - F_23)", ["F_13", "F_23"], target, "coulomb_collinear_opposite"),
                ],
                world_input.target,
            ),
            0.69,
            0.72,
            ["Collinear-opposite Coulomb candidate."],
        )


def _add_force_resultant_variants(
    world_input: Type2WorldModelInput,
    candidates: list[Type2StepPlanCandidate],
) -> None:
    force_names = _quantity_names_by_dimension(world_input, "force")
    has_named_forces = _has_quantity(world_input, "F", "F2") or _has_quantity(world_input, "F1", "F2")
    force_text_evidence = _text_contains(
        world_input,
        "force",
        "forces",
        "resultant",
        "same direction",
        "opposite direction",
        "perpendicular",
        "angle",
    )
    if not ((_target_in(world_input, "F_net", "F_e", "theta") or force_text_evidence) and (has_named_forces or len(force_names) >= 2)):
        return
    if _has_quantity(world_input, "F", "F2"):
        f1, f2 = "F", "F2"
    elif _has_quantity(world_input, "F1", "F2"):
        f1, f2 = "F1", "F2"
    else:
        f1, f2 = force_names[0], force_names[1]

    _make_candidate(
        candidates,
        "deterministic_variant",
        ["force_collinear_sum"],
        world_input.target,
        world_input.target_unit,
        _plan("force_collinear_sum", [_formula_step("step_1", "Add collinear forces.", "F_net = F + F2", [f1, f2], world_input.target or "F_net", "force_collinear_sum")], world_input.target),
        0.64,
        0.64,
        ["Collinear force-sum candidate."],
    )
    if any(phrase in _text(world_input) for phrase in ("opposite direction", "opposite directions", "collinear")):
        _make_candidate(
            candidates,
            "deterministic_variant",
            ["force_collinear_difference"],
            world_input.target,
            world_input.target_unit,
            _plan("force_collinear_difference", [_formula_step("step_1", "Subtract opposite collinear forces.", "F_net = abs(F - F2)", [f1, f2], world_input.target or "F_net", "force_collinear_difference")], world_input.target),
            0.67,
            0.69,
            ["Opposite-direction force candidate."],
        )
    if "theta" in world_input.known_quantities:
        _make_candidate(
            candidates,
            "deterministic_variant",
            ["force_angle_resultant"],
            world_input.target,
            world_input.target_unit,
            _plan("force_angle_resultant", [_formula_step("step_1", "Compute resultant of two angled forces.", "F_net = sqrt(F^2 + F2^2 + 2*F*F2*cos(theta))", [f1, f2, "theta"], world_input.target or "F_net", "force_angle_resultant")], world_input.target),
            0.68,
            0.72,
            ["Angled-force resultant candidate."],
        )
    if _contains_condition(world_input, "right_angle") or _has_angle_near_90(world_input):
        _make_candidate(
            candidates,
            "deterministic_variant",
            ["force_right_angle_resultant"],
            world_input.target,
            world_input.target_unit,
            _plan("force_right_angle_resultant", [_formula_step("step_1", "Compute right-angle force resultant.", "F_net = sqrt(F^2 + F2^2)", [f1, f2], world_input.target or "F_net", "force_right_angle_resultant")], world_input.target),
            0.69,
            0.73,
            ["Right-angle force resultant candidate."],
        )
    if _target_is(world_input, "theta") or "direction" in _text(world_input):
        _make_candidate(
            candidates,
            "deterministic_variant",
            ["force_direction_angle"],
            world_input.target,
            world_input.target_unit,
            _plan("force_direction_angle", [_formula_step("step_1", "Compute direction from net vector components.", "theta = atan2(sum(F_y), sum(F_x))", [f1, f2], "theta", "force_direction_angle")], world_input.target),
            0.64,
            0.70,
            ["Force direction-angle candidate."],
        )


def _add_capacitor_variants(
    world_input: Type2WorldModelInput,
    candidates: list[Type2StepPlanCandidate],
) -> None:
    capacitor_evidence = (
        _subdomain_contains(world_input, "capacitor", "dielectric_capacitor", "capacitor_network")
        or _legacy_templates_contain(world_input, "capacitor", "capacitance", "dielectric")
        or _text_contains(world_input, "capacitor", "capacitance", "dielectric")
        or _quantity_names_contain(world_input, "C_cap")
    )
    if not (
        capacitor_evidence
        or (_target_in(world_input, "U_cap", "C_cap", "U_after", "V_after", "Q_after", "C_after", "delta_U") and _domain_contains(world_input, "electric"))
        or (_target_in(world_input, "Q", "V") and capacitor_evidence)
    ):
        return

    variants = [
        ("capacitor_energy", "Compute capacitor energy.", "U_cap = 0.5 * C_cap * V^2", ["C_cap", "V"], "U_cap", 0.70),
        ("capacitance_definition", "Compute capacitance from charge and voltage.", "C_cap = Q / V", ["Q", "V"], "C_cap", 0.67),
        ("charge_definition", "Compute capacitor charge.", "Q = C_cap * V", ["C_cap", "V"], "Q", 0.67),
        ("voltage_definition", "Compute capacitor voltage.", "V = Q / C_cap", ["Q", "C_cap"], "V", 0.67),
    ]
    for template_name, goal, formula, inputs, output, rank in variants:
        _make_candidate(
            candidates,
            "deterministic_variant",
            [template_name],
            world_input.target,
            world_input.target_unit,
            _plan(template_name, [_formula_step("step_1", goal, formula, inputs, output, template_name)], world_input.target),
            rank,
            rank,
            [f"{template_name} deterministic capacitor candidate."],
        )
    if _contains_condition(world_input, "battery_connected"):
        _make_candidate(
            candidates,
            "deterministic_variant",
            ["dielectric_battery_connected"],
            world_input.target,
            world_input.target_unit,
            _plan(
                "dielectric_battery_connected",
                [
                    _formula_step("step_1", "Compute capacitance after inserting dielectric.", "C_after = epsilon_r * C_cap", ["epsilon_r", "C_cap"], "C_after", "dielectric_battery_connected"),
                    _formula_step("step_2", "Battery keeps voltage fixed.", "V_after = V", ["V"], "V_after", "dielectric_battery_connected"),
                    _formula_step("step_3", "Compute charge after dielectric insertion.", "Q_after = epsilon_r * Q", ["epsilon_r", "Q"], "Q_after", "dielectric_battery_connected"),
                ],
                world_input.target,
            ),
            0.71,
            0.75,
            ["Battery-connected dielectric candidate."],
        )
    if _contains_condition(world_input, "battery_disconnected"):
        _make_candidate(
            candidates,
            "deterministic_variant",
            ["dielectric_battery_disconnected"],
            world_input.target,
            world_input.target_unit,
            _plan(
                "dielectric_battery_disconnected",
                [
                    _formula_step("step_1", "Compute capacitance after inserting dielectric.", "C_after = epsilon_r * C_cap", ["epsilon_r", "C_cap"], "C_after", "dielectric_battery_disconnected"),
                    _formula_step("step_2", "Disconnected capacitor preserves charge.", "Q_after = Q", ["Q"], "Q_after", "dielectric_battery_disconnected"),
                    _formula_step("step_3", "Compute voltage after dielectric insertion.", "V_after = V / epsilon_r", ["V", "epsilon_r"], "V_after", "dielectric_battery_disconnected"),
                ],
                world_input.target,
            ),
            0.71,
            0.75,
            ["Battery-disconnected dielectric candidate."],
        )


def _add_circuit_variants(world_input: Type2WorldModelInput, candidates: list[Type2StepPlanCandidate]) -> None:
    circuit_evidence = (
        _subdomain_contains(world_input, "ohms_law", "circuit", "resistor")
        or _legacy_templates_contain(world_input, "ohms", "power_from", "resistance")
        or _text_contains(world_input, "circuit", "resistor", "resistance", "ohm", "current", "voltage", "power")
        or _quantity_names_contain(world_input, "I", "R", "P", "P_total")
    )
    if not ((_target_in(world_input, "I", "V", "R", "P", "P_total") and circuit_evidence) or _subdomain_contains(world_input, "ohms_law")):
        return
    variants = [
        ("ohms_current", "Apply Ohm's law for current.", "I = V / R", ["V", "R"], "I", 0.68),
        ("ohms_voltage", "Apply Ohm's law for voltage.", "V = I * R", ["I", "R"], "V", 0.68),
        ("ohms_resistance", "Apply Ohm's law for resistance.", "R = V / I", ["V", "I"], "R", 0.68),
        ("power_vi", "Compute electric power from voltage and current.", "P = V * I", ["V", "I"], "P", 0.66),
        ("power_v2_over_r", "Compute resistive power from voltage and resistance.", "P = V^2 / R", ["V", "R"], "P", 0.66),
        ("power_i2r", "Compute resistive power from current and resistance.", "P = I^2 * R", ["I", "R"], "P", 0.66),
    ]
    for template_name, goal, formula, inputs, output, rank in variants:
        _make_candidate(
            candidates,
            "deterministic_variant",
            [template_name],
            world_input.target,
            world_input.target_unit,
            _plan(template_name, [_formula_step("step_1", goal, formula, inputs, output, template_name)], world_input.target),
            rank,
            rank,
            [f"{template_name} deterministic circuit candidate."],
        )


def _add_kinematics_variants(world_input: Type2WorldModelInput, candidates: list[Type2StepPlanCandidate]) -> None:
    electric_geometry = (
        _domain_contains(world_input, "electricity", "electrostatic", "coulomb")
        or _subdomain_contains(world_input, "electrostatic", "coulomb", "capacitor")
    )
    blocked_electric_targets = {"F_e", "F_net", "F_on_q3", "q", "C_cap", "U_cap", "Q", "V"}
    if world_input.target in blocked_electric_targets and electric_geometry:
        return

    motion_text = _text_contains(
        world_input,
        "speed",
        "velocity",
        "accelerates",
        "acceleration",
        "moves",
        "travels",
        "distance traveled",
        "time taken",
        "uniform acceleration",
    )
    kinematics_evidence = (
        _domain_contains(world_input, "mechanics", "kinematics")
        or _subdomain_contains(world_input, "kinematics", "motion", "uniform_acceleration", "average_speed")
        or _target_in(
            world_input,
            "v",
            "v_final",
            "v_0",
            "a",
            "d",
            "s",
            "x",
            "t",
            "v_avg",
            "distance",
            "speed",
            "velocity",
            "acceleration",
            "time",
        )
        or motion_text
    )
    if electric_geometry and not motion_text:
        return
    if not kinematics_evidence:
        return
    variants = [
        ("uniform_acceleration_velocity", "Apply constant-acceleration velocity relation.", "v_final = v_0 + a*t", ["v_0", "a", "t"], "v_final", 0.68),
        ("uniform_acceleration_displacement", "Apply constant-acceleration displacement relation.", "d = v_0*t + 0.5*a*t^2", ["v_0", "a", "t"], "d", 0.68),
        ("no_time_velocity_displacement", "Apply no-time velocity-displacement relation.", "v_final^2 = v_0^2 + 2*a*d", ["v_0", "a", "d"], "v_final", 0.66),
        ("constant_speed_distance", "Apply constant-speed distance relation.", "d = v*t", ["v", "t"], "d", 0.66),
        ("average_speed", "Compute average speed.", "v_avg = total_distance / total_time", ["total_distance", "total_time"], "v_avg", 0.64),
    ]
    for template_name, goal, formula, inputs, output, rank in variants:
        _make_candidate(
            candidates,
            "deterministic_variant",
            [template_name],
            world_input.target,
            world_input.target_unit,
            _plan(template_name, [_formula_step("step_1", goal, formula, inputs, output, template_name)], world_input.target),
            rank,
            rank,
            [f"{template_name} deterministic kinematics candidate."],
        )


def _add_measurement_error_variants(
    world_input: Type2WorldModelInput,
    candidates: list[Type2StepPlanCandidate],
) -> None:
    if not (
        _target_is(world_input, "abs_error", "rel_error", "percent_error", "mean_value", "random_error")
        or bool({"measurement_error", "error_propagation"} & _all_labels(world_input))
    ):
        return
    variants = [
        ("absolute_error", "Compute absolute error.", "abs_error = abs(measured_value - true_value)", ["measured_value", "true_value"], "abs_error", 0.68),
        ("relative_error", "Compute relative error.", "rel_error = abs_error / true_value", ["abs_error", "true_value"], "rel_error", 0.67),
        ("percent_error", "Compute percent error.", "percent_error = 100 * rel_error", ["rel_error"], "percent_error", 0.67),
        ("mean_value", "Compute mean of repeated measurements.", "mean_value = average(values)", ["values"], "mean_value", 0.66),
    ]
    for template_name, goal, formula, inputs, output, rank in variants:
        _make_candidate(
            candidates,
            "deterministic_variant",
            [template_name],
            world_input.target,
            world_input.target_unit,
            _plan(template_name, [_formula_step("step_1", goal, formula, inputs, output, template_name)], world_input.target),
            rank,
            rank,
            [f"{template_name} deterministic measurement-error candidate."],
        )


def _add_symbolic_or_boolean_variants(
    world_input: Type2WorldModelInput,
    candidates: list[Type2StepPlanCandidate],
) -> None:
    if world_input.question_type == "numeric_calc":
        return
    symbolic_plan = _copy_step_plan(world_input.step_plan) or [
        _setup_step(
            "step_1",
            "Set up symbolic relations from extracted quantities and conditions.",
            "symbolic_relation_candidate",
            {"symbolic_relation": "extracted_relations"},
        )
    ]
    _make_candidate(
        candidates,
        "non_numeric_symbolic_candidate",
        _infer_template_names(symbolic_plan, world_input.metadata) or ["symbolic_relation_candidate"],
        world_input.target,
        world_input.target_unit,
        symbolic_plan,
        0.62,
        0.64,
        ["Non-numeric symbolic relation candidate."],
    )
    _make_candidate(
        candidates,
        "non_numeric_boolean_candidate",
        ["boolean_check_candidate"],
        world_input.target,
        world_input.target_unit,
        [
            _setup_step(
                "step_1",
                "Evaluate whether the stated condition follows from extracted relations and physical constraints.",
                "boolean_check_candidate",
            )
        ],
        0.58,
        0.60,
        ["Non-numeric boolean-check candidate."],
    )


def _normalized_formula_sequence(candidate: Type2StepPlanCandidate) -> tuple[str, ...]:
    return tuple(
        " ".join(str(step.get("formula_name", "")).lower().split())
        for step in candidate.step_plan
        if isinstance(step, dict) and step.get("formula_name")
    )


def _dedupe_key(candidate: Type2StepPlanCandidate) -> tuple[str, tuple[str, ...] | tuple[tuple[str, ...], str | None]]:
    formula_sequence = _normalized_formula_sequence(candidate)
    if formula_sequence:
        return ("formula", formula_sequence)
    return ("template_target", (tuple(sorted(candidate.template_names)), candidate.target))


def deduplicate_candidates(
    candidates: list[Type2StepPlanCandidate],
    max_candidates: int = 8,
) -> list[Type2StepPlanCandidate]:
    """Deduplicate candidates deterministically and keep highest-rank variants."""
    legacy = [candidate for candidate in candidates if candidate.candidate_id == "candidate_legacy_0"]
    non_legacy = [candidate for candidate in candidates if candidate.candidate_id != "candidate_legacy_0"]

    best_by_key: dict[tuple[str, Any], Type2StepPlanCandidate] = {}
    for candidate in non_legacy:
        key = _dedupe_key(candidate)
        previous = best_by_key.get(key)
        if previous is None or (candidate.rank_hint, -len(candidate.candidate_id), candidate.candidate_id) > (
            previous.rank_hint,
            -len(previous.candidate_id),
            previous.candidate_id,
        ):
            best_by_key[key] = candidate

    deduped = legacy + list(best_by_key.values())
    deduped = sorted(deduped, key=lambda candidate: (-candidate.rank_hint, candidate.candidate_id))

    if not legacy:
        return deduped[:max_candidates]

    legacy_candidate = legacy[0]
    selected = deduped[:max_candidates]
    if legacy_candidate not in selected:
        selected = selected[: max(max_candidates - 1, 0)] + [legacy_candidate]
        selected = sorted(selected, key=lambda candidate: (-candidate.rank_hint, candidate.candidate_id))
    return selected


def _build_summary(candidates: list[Type2StepPlanCandidate], pre_dedup_count: int) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    template_counts: dict[str, int] = {}
    for candidate in candidates:
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
        for template_name in candidate.template_names:
            template_counts[template_name] = template_counts.get(template_name, 0) + 1
    return {
        "pre_dedup_candidate_count": pre_dedup_count,
        "candidate_count": len(candidates),
        "candidate_source_counts": source_counts,
        "candidate_template_counts": template_counts,
        "legacy_candidate_present": any(candidate.source == "legacy_parser_step_plan" for candidate in candidates),
    }


def _add_llm_fallback_canonicalized_variant(
    world_input: Type2WorldModelInput,
    candidates: list[Type2StepPlanCandidate],
) -> None:
    """Step 10: add a canonicalized candidate derived from an LLM-fallback parse.

    Deterministic candidates are left untouched. A new candidate is added only
    when the parse came from the LLM fallback (or a loose/skeleton plan) and at
    least one loose formula maps to a canonical executor formula.
    """
    if not bool(world_input.metadata.get("used_llm_fallback")):
        legacy_has_loose = any(
            isinstance(step, dict)
            and step.get("type") == "formula_application"
            and step.get("formula_name")
            and "=" not in str(step.get("formula_name"))
            for step in (world_input.step_plan or [])
        )
        if not legacy_has_loose:
            return

    source_candidate = next(
        (c for c in candidates if c.candidate_id == "candidate_legacy_0"), None
    )
    if source_candidate is None:
        return

    result = canonicalize_llm_fallback_candidate(world_input, source_candidate)
    if result.status not in {"PASS", "WARN"} or not result.canonical_formula_names:
        return

    _make_candidate(
        candidates,
        "llm_fallback_canonicalized",
        result.canonical_template_names or ["llm_fallback_canonicalized"],
        world_input.target,
        world_input.target_unit,
        result.canonical_step_plan,
        0.66,
        0.66,
        ["Canonicalized LLM-fallback step plan into executor-supported formulas."],
        metadata={
            "canonicalization_result": result.to_dict(),
            "llm_original_candidate": result.metadata.get("llm_original_candidate"),
        },
    )


def generate_step_plan_candidates(
    world_input: Type2WorldModelInput,
) -> Type2CandidateGenerationResult:
    """Generate deterministic step-plan candidates from adapted Stage 0 output."""
    candidates: list[Type2StepPlanCandidate] = []

    if world_input.step_plan:
        legacy_plan = _copy_step_plan(world_input.step_plan)
        template_names = _infer_template_names(legacy_plan, world_input.metadata)
        candidates.append(
            Type2StepPlanCandidate(
                candidate_id="candidate_legacy_0",
                source="legacy_parser_step_plan",
                template_names=template_names,
                target=world_input.target,
                target_unit=world_input.target_unit,
                step_plan=legacy_plan,
                prior_confidence=world_input.plan_confidence,
                rank_hint=world_input.plan_confidence,
                generation_notes=["Existing Stage 0 parser step_plan candidate."],
                metadata={
                    "parser_status": world_input.parser_status,
                    "used_skeleton_fallback": bool(world_input.metadata.get("used_skeleton_fallback")),
                    "verifier_errors": deepcopy(world_input.parser_errors),
                    "parser_warnings": deepcopy(world_input.parser_warnings),
                },
            )
        )

    _add_coulomb_variants(world_input, candidates)
    _add_force_resultant_variants(world_input, candidates)
    _add_capacitor_variants(world_input, candidates)
    _add_circuit_variants(world_input, candidates)
    _add_kinematics_variants(world_input, candidates)
    _add_measurement_error_variants(world_input, candidates)
    _add_symbolic_or_boolean_variants(world_input, candidates)
    _add_llm_fallback_canonicalized_variant(world_input, candidates)

    pre_dedup_count = len(candidates)
    candidates = deduplicate_candidates(candidates, max_candidates=8)
    selected_candidate_id = candidates[0].candidate_id if candidates else None

    return Type2CandidateGenerationResult(
        problem_text=world_input.problem_text,
        target=world_input.target,
        target_unit=world_input.target_unit,
        candidates=candidates,
        selected_candidate_id=selected_candidate_id,
        generation_summary=_build_summary(candidates, pre_dedup_count),
    )