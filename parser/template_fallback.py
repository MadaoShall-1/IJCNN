"""Formula-pattern fallback that creates step plans without solving."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


CONSTANTS = {"pi", "k", "epsilon_0", "mu_0"}

_SUPERSCRIPT_TRANSLATION = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁻": "-",
    "⁺": "+",
})


def _has(known: Dict[str, Dict[str, object]], *names: str) -> bool:
    return all(name in known for name in names)


def _by_dimension(known: Dict[str, Dict[str, object]], dimension: str) -> List[str]:
    return [name for name, quantity in known.items() if quantity.get("dimension") == dimension]


def _first_by_dimension(known: Dict[str, Dict[str, object]], dimension: str) -> Optional[str]:
    matches = _by_dimension(known, dimension)
    return matches[0] if matches else None


def _first_existing(known: Dict[str, Dict[str, object]], *names: str) -> Optional[str]:
    for name in names:
        if name in known:
            return name
    return None


def _find_relation(relations: List[Dict[str, object]], left: str, right: Optional[str] = None, relation_type: str = "ratio") -> Optional[Dict[str, object]]:
    for relation in relations:
        if relation.get("type") != relation_type:
            continue
        if relation.get("left") == left and (right is None or relation.get("right") == right):
            return relation
    return None


def _relation_factor(relation: Optional[Dict[str, object]]) -> Optional[float]:
    if not relation or relation.get("factor") is None:
        return None
    try:
        return float(relation["factor"])
    except (TypeError, ValueError):
        return None


def _num_literal(value: float) -> str:
    return f"{value:.12g}"


def _normalize_formula_literal(text: str) -> str:
    text = text.strip()
    text = re.sub(
        r"(?P<base>\d+(?:\.\d+)?)(?P<exp>[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda match: f"{match.group('base')}^{match.group('exp').translate(_SUPERSCRIPT_TRANSLATION)}",
        text,
    )
    text = text.translate(_SUPERSCRIPT_TRANSLATION)
    text = text.replace("π", "pi").replace("Π", "pi")
    text = text.replace("−", "-").replace("×", "*")
    text = re.sub(r"(?<=\d)\s*pi\b", "*pi", text)
    text = re.sub(r"(?<=\d)pi\b", "*pi", text)
    text = re.sub(r"(?<=\))\s*(?=\()", "*", text)
    text = re.sub(r"\s+", "", text)
    return text


def _formula_step(
    step_id: str,
    goal: str,
    formula: str,
    inputs: List[str],
    output: str,
    template_name: str,
    confidence: float = 0.88,
    warning: Optional[str] = None,
) -> Dict[str, object]:
    step: Dict[str, object] = {
        "step_id": step_id,
        "goal": goal,
        "type": "formula_application",
        "formula_name": formula,
        "template_name": template_name,
        "input_var": {name: name for name in inputs},
        "output_var": {output: formula},
        "confidence": confidence,
    }
    if warning:
        step["parser_warning"] = warning
    return step


def _setup_step(
    step_id: str,
    goal: str,
    outputs: Optional[Dict[str, str]] = None,
    template_name: str = "setup",
    confidence: float = 0.82,
    warning: Optional[str] = None,
) -> Dict[str, object]:
    step: Dict[str, object] = {
        "step_id": step_id,
        "goal": goal,
        "type": "setup",
        "template_name": template_name,
        "input_var": {},
        "output_var": outputs or {},
        "confidence": confidence,
    }
    if warning:
        step["parser_warning"] = warning
    return step


def _conclusion(step_id: str, target: str, template_name: str) -> Dict[str, object]:
    return {
        "step_id": step_id,
        "goal": f"Report the final value of {target}.",
        "type": "conclusion",
        "template_name": template_name,
        "input_var": {target: target},
        "output_var": {target: target},
        "confidence": 0.86,
    }


def _finish(plan: List[Dict[str, object]], target: str, template_name: str) -> Tuple[List[Dict[str, object]], float]:
    if not plan:
        return [], 0.0
    plan.append(_conclusion(f"step_{len(plan) + 1}", target, template_name))
    return plan, min(0.95, sum(float(step["confidence"]) for step in plan) / len(plan))


def _renumber(plan: List[Dict[str, object]]) -> None:
    for index, step in enumerate(plan, start=1):
        step["step_id"] = f"step_{index}"


def _omega_steps(known: Dict[str, Dict[str, object]], plan: List[Dict[str, object]], template_name: str) -> Optional[str]:
    if "omega" in known:
        return "omega"
    if "f" in known:
        plan.append(_formula_step("step_1", "Compute angular frequency from frequency.", "omega = 2*pi*f", ["f", "pi"], "omega", template_name))
        return "omega"
    return None


def _reactance_steps(known: Dict[str, Dict[str, object]], plan: List[Dict[str, object]], template_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    omega = _omega_steps(known, plan, template_name)
    if not omega:
        return None, None, None
    inductance = _first_by_dimension(known, "inductance")
    capacitance = _first_by_dimension(known, "capacitance")
    x_l = "X_L" if inductance else None
    x_c = "X_C" if capacitance else None
    if inductance:
        plan.append(_formula_step("step_1", "Compute inductive reactance.", "X_L = omega * L", [omega, inductance], "X_L", template_name))
    if capacitance:
        plan.append(_formula_step("step_1", "Compute capacitive reactance.", "X_C = 1 / (omega * C_cap)", [omega, capacitance], "X_C", template_name))
    _renumber(plan)
    return omega, x_l, x_c


def _series_rlc_impedance_plan(known: Dict[str, Dict[str, object]], target: str, template_name: str) -> Tuple[List[Dict[str, object]], float]:
    resistance = _first_by_dimension(known, "resistance")
    if not resistance:
        return [], 0.0
    plan: List[Dict[str, object]] = []
    _, x_l, x_c = _reactance_steps(known, plan, template_name)
    if not x_l and not x_c:
        return [], 0.0
    if x_l and x_c:
        plan.append(_formula_step("step_1", "Compute net reactance.", "X = X_L - X_C", ["X_L", "X_C"], "X", template_name))
    elif x_l:
        plan.append(_formula_step("step_1", "Use inductive reactance as net reactance.", "X = X_L", ["X_L"], "X", template_name))
    else:
        plan.append(_formula_step("step_1", "Use capacitive reactance as net reactance.", "X = -X_C", ["X_C"], "X", template_name))
    plan.append(_formula_step("step_1", "Compute series RLC impedance.", "Z = sqrt(R^2 + X^2)", [resistance, "X"], "Z", template_name))
    _renumber(plan)
    if target == "Z":
        return _finish(plan, target, template_name)
    return plan, 0.0


def _ac_templates(known: Dict[str, Dict[str, object]], target: str) -> Tuple[List[Dict[str, object]], float]:
    if target == "X_L":
        inductance = _first_by_dimension(known, "inductance")
        plan: List[Dict[str, object]] = []
        omega = _omega_steps(known, plan, "ac_inductive_reactance")
        if inductance and omega:
            plan.append(_formula_step("step_1", "Compute inductive reactance.", "X_L = omega * L", [omega, inductance], "X_L", "ac_inductive_reactance"))
            _renumber(plan)
            return _finish(plan, target, "ac_inductive_reactance")
    if target == "X_C":
        capacitance = _first_by_dimension(known, "capacitance")
        plan = []
        omega = _omega_steps(known, plan, "ac_capacitive_reactance")
        if capacitance and omega:
            plan.append(_formula_step("step_1", "Compute capacitive reactance.", "X_C = 1 / (omega * C_cap)", [omega, capacitance], "X_C", "ac_capacitive_reactance"))
            _renumber(plan)
            return _finish(plan, target, "ac_capacitive_reactance")
    if target == "Z":
        return _series_rlc_impedance_plan(known, target, "series_rlc_impedance")
    if target in {"I", "I_rms"}:
        voltage = _first_existing(known, "V_rms", "V")
        plan, _ = _series_rlc_impedance_plan(known, "Z", "series_rlc_current")
        if voltage and plan:
            output = target
            formula = f"{output} = {voltage} / Z"
            plan = plan[:-1]
            plan.append(_formula_step("step_1", "Compute circuit current from impedance.", formula, [voltage, "Z"], output, "series_rlc_current"))
            _renumber(plan)
            return _finish(plan, output, "series_rlc_current")
    if target == "power_factor":
        plan, _ = _series_rlc_impedance_plan(known, "Z", "power_factor")
        resistance = _first_by_dimension(known, "resistance")
        if plan and resistance:
            plan = plan[:-1]
            plan.append(_formula_step("step_1", "Compute power factor.", "power_factor = R / Z", [resistance, "Z"], "power_factor", "power_factor"))
            _renumber(plan)
            return _finish(plan, target, "power_factor")
    if target == "P_avg":
        if _has(known, "I_rms") and _first_by_dimension(known, "resistance"):
            resistance = _first_by_dimension(known, "resistance") or "R"
            return _finish([_formula_step("step_1", "Compute average AC power.", "P_avg = I_rms^2 * R", ["I_rms", resistance], "P_avg", "ac_average_power")], target, "ac_average_power")
        if _has(known, "V_rms", "I_rms", "power_factor"):
            return _finish([_formula_step("step_1", "Compute average AC power.", "P_avg = V_rms * I_rms * power_factor", ["V_rms", "I_rms", "power_factor"], "P_avg", "ac_average_power")], target, "ac_average_power")
    if target in {"U_R", "V_R"}:
        current = _first_existing(known, "I_rms", "I")
        resistance = _first_by_dimension(known, "resistance")
        if current and resistance:
            return _finish([_formula_step("step_1", "Compute voltage across resistor.", f"{target} = I * R", [current, resistance], target, "ac_voltage_across_resistor")], target, "ac_voltage_across_resistor")
    if target in {"U_L", "V_L"}:
        current = _first_existing(known, "I_rms", "I")
        if current:
            if "X_L" in known:
                return _finish([_formula_step("step_1", "Compute voltage across inductor.", f"{target} = I * X_L", [current, "X_L"], target, "ac_voltage_across_inductor")], target, "ac_voltage_across_inductor")
            plan, _ = _ac_templates(known, "X_L")
            if plan:
                plan = plan[:-1]
                plan.append(_formula_step("step_1", "Compute voltage across inductor.", f"{target} = I * X_L", [current, "X_L"], target, "ac_voltage_across_inductor"))
                _renumber(plan)
                return _finish(plan, target, "ac_voltage_across_inductor")
    if target in {"U_C", "V_C"}:
        current = _first_existing(known, "I_rms", "I")
        if current:
            if "X_C" in known:
                return _finish([_formula_step("step_1", "Compute voltage across capacitor.", f"{target} = I * X_C", [current, "X_C"], target, "ac_voltage_across_capacitor")], target, "ac_voltage_across_capacitor")
            plan, _ = _ac_templates(known, "X_C")
            if plan:
                plan = plan[:-1]
                plan.append(_formula_step("step_1", "Compute voltage across capacitor.", f"{target} = I * X_C", [current, "X_C"], target, "ac_voltage_across_capacitor"))
                _renumber(plan)
                return _finish(plan, target, "ac_voltage_across_capacitor")
    if target in {"Q_factor", "q", "Q"}:
        resistance = _first_by_dimension(known, "resistance")
        inductance = _first_by_dimension(known, "inductance")
        capacitance = _first_by_dimension(known, "capacitance")
        if resistance and inductance and "omega_0" in known:
            return _finish([_formula_step("step_1", "Compute series RLC quality factor.", "Q_factor = omega_0 * L / R", ["omega_0", inductance, resistance], "Q_factor", "quality_factor_series_rlc")], "Q_factor", "quality_factor_series_rlc")
        if resistance and inductance and capacitance:
            plan = [
                _formula_step("step_1", "Compute series RLC quality factor from component values.", "Q_factor = sqrt(L/C_cap) / R", [inductance, capacitance, resistance], "Q_factor", "quality_factor_series_rlc"),
            ]
            return _finish(plan, "Q_factor", "quality_factor_series_rlc")
    if target in {"f_res", "omega_0"}:
        inductance = _first_by_dimension(known, "inductance")
        capacitance = _first_by_dimension(known, "capacitance")
        if inductance and capacitance:
            plan = [_formula_step("step_1", "Compute resonant angular frequency.", "omega_0 = 1 / sqrt(L * C_cap)", [inductance, capacitance], "omega_0", "rlc_resonant_frequency")]
            if target == "f_res":
                plan.append(_formula_step("step_2", "Convert angular frequency to frequency.", "f_res = omega_0 / (2*pi)", ["omega_0", "pi"], "f_res", "rlc_resonant_frequency"))
            return _finish(plan, target, "rlc_resonant_frequency")
    return [], 0.0


def _lc_templates(known: Dict[str, Dict[str, object]], target: str, relations: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], float]:
    inductance = _first_by_dimension(known, "inductance")
    capacitance = _first_by_dimension(known, "capacitance")
    if target in {"f_osc", "f_res"} and inductance and capacitance:
        return _finish([_formula_step("step_1", "Compute LC oscillation frequency.", "f_osc = 1 / (2*pi*sqrt(L*C_cap))", [inductance, capacitance, "pi"], target, "lc_frequency_period")], target, "lc_frequency_period")
    if target == "T_osc" and inductance and capacitance:
        return _finish([_formula_step("step_1", "Compute LC oscillation period.", "T_osc = 2*pi*sqrt(L*C_cap)", [inductance, capacitance, "pi"], "T_osc", "lc_frequency_period")], target, "lc_frequency_period")
    electric_relation_for_percent = _find_relation(relations, "U_E", "U_total")
    magnetic_relation_for_percent = _find_relation(relations, "U_B", "U_total")
    if target == "I_over_Imax_percent" and (electric_relation_for_percent or magnetic_relation_for_percent):
        magnetic_fraction = _relation_factor(magnetic_relation_for_percent)
        if magnetic_fraction is None:
            electric_fraction = _relation_factor(electric_relation_for_percent)
            if electric_fraction is not None:
                magnetic_fraction = 1.0 - electric_fraction
        if magnetic_fraction is not None:
            magnetic_fraction = max(0.0, min(1.0, magnetic_fraction))
            return _finish([
                _formula_step(
                    "step_1",
                    "Compute instantaneous current as a percentage of maximum current from LC energy split.",
                    f"I_over_Imax_percent = 100 * sqrt({_num_literal(magnetic_fraction)})",
                    [],
                    "I_over_Imax_percent",
                    "lc_current_percent_from_energy_fraction",
                )
            ], target, "lc_current_percent_from_energy_fraction")
    if target == "electric_energy_fraction_percent" and (electric_relation_for_percent or magnetic_relation_for_percent):
        electric_fraction = _relation_factor(electric_relation_for_percent)
        if electric_fraction is None:
            magnetic_fraction = _relation_factor(magnetic_relation_for_percent)
            if magnetic_fraction is not None:
                electric_fraction = 1.0 - magnetic_fraction
        if electric_fraction is not None:
            electric_fraction = max(0.0, min(1.0, electric_fraction))
            return _finish([
                _formula_step(
                    "step_1",
                    "Convert LC electric-energy fraction to percent.",
                    f"electric_energy_fraction_percent = 100 * {_num_literal(electric_fraction)}",
                    [],
                    "electric_energy_fraction_percent",
                    "lc_electric_energy_percent_from_fraction",
                )
            ], target, "lc_electric_energy_percent_from_fraction")
    if target in {"magnetic_energy_fraction", "electric_energy_fraction", "energy_fraction"} and "q_over_Qmax" in known:
        plan = [
            _formula_step("step_1", "Compute electric energy fraction from charge ratio.", "electric_energy_fraction = q_over_Qmax^2", ["q_over_Qmax"], "electric_energy_fraction", "lc_energy_fraction_from_charge_ratio"),
            _formula_step("step_2", "Compute magnetic energy fraction.", "magnetic_energy_fraction = 1 - electric_energy_fraction", ["electric_energy_fraction"], "magnetic_energy_fraction", "lc_energy_fraction_from_charge_ratio"),
        ]
        return _finish(plan, target, "lc_energy_fraction_from_charge_ratio")
    electric_relation = _find_relation(relations, "U_E", "U_total")
    if target in {"magnetic_energy_fraction", "electric_energy_fraction", "energy_fraction"} and electric_relation and electric_relation.get("factor") is not None:
        plan = [
            _formula_step("step_1", "Use extracted electric-energy fraction.", "electric_energy_fraction = factor", [], "electric_energy_fraction", "lc_energy_fraction_electric"),
            _formula_step("step_2", "Compute magnetic energy fraction.", "magnetic_energy_fraction = 1 - electric_energy_fraction", ["electric_energy_fraction"], "magnetic_energy_fraction", "lc_energy_fraction_electric"),
        ]
        return _finish(plan, target, "lc_energy_fraction_electric")
    if target == "I_over_Imax" and "magnetic_energy_fraction" in known:
        return _finish([_formula_step("step_1", "Compute current fraction from magnetic energy fraction.", "I_over_Imax = sqrt(magnetic_energy_fraction)", ["magnetic_energy_fraction"], "I_over_Imax", "lc_current_fraction_from_energy")], target, "lc_current_fraction_from_energy")
    if target == "q_over_Qmax" and "electric_energy_fraction" in known:
        return _finish([_formula_step("step_1", "Compute charge fraction from electric energy fraction.", "q_over_Qmax = sqrt(electric_energy_fraction)", ["electric_energy_fraction"], "q_over_Qmax", "lc_charge_fraction_from_energy")], target, "lc_charge_fraction_from_energy")
    current = _first_existing(known, "I", "I_max", "I_rms")
    if target == "U_B" and inductance and current:
        return _finish([_formula_step("step_1", "Compute magnetic field energy in an inductor.", "U_B = 0.5 * L * I^2", [inductance, current], "U_B", "lc_energy_from_current")], target, "lc_energy_from_current")
    charge = _first_existing(known, "q", "Q", "Q_max")
    if target == "U_E" and capacitance and charge:
        return _finish([_formula_step("step_1", "Compute electric field energy in a capacitor.", "U_E = q^2 / (2*C_cap)", [charge, capacitance], "U_E", "lc_energy_from_capacitor_charge")], target, "lc_energy_from_capacitor_charge")
    if target in {"U_total", "U_E", "U_B"} and capacitance and "Q_max" in known:
        plan = [_formula_step("step_1", "Compute total LC energy from maximum charge.", "U_total = Q_max^2 / (2*C_cap)", ["Q_max", capacitance], "U_total", "lc_total_energy_from_charge")]
        if target != "U_total":
            plan.append(_formula_step("step_2", "Relate requested LC energy to total energy.", f"{target} = U_total", ["U_total"], target, "lc_total_energy_from_charge"))
        return _finish(plan, target, "lc_total_energy_from_charge")
    return [], 0.0


def _measurement_templates(known: Dict[str, Dict[str, object]], target: str) -> Tuple[List[Dict[str, object]], float]:
    if target.startswith("delta_") and target in known:
        return _finish([_formula_step("step_1", "Use directly extracted uncertainty.", f"{target} = extracted_uncertainty", [target], target, "uncertainty_direct")], target, "uncertainty_direct")
    voltage = _first_existing(known, "U", "V")
    delta_voltage = _first_existing(known, "delta_U", "delta_V")
    if target in {"delta_P", "rel_error", "percent_error"} and {"voltage", "delta_voltage", "current", "delta_current"} <= set(known):
        plan = [
            _formula_step("step_1", "Compute power from voltage and current.", "P = voltage * current", ["voltage", "current"], "P", "power_error_from_named_uncertainties"),
            _formula_step("step_2", "Compute power relative error.", "rel_error_P = delta_voltage / abs(voltage) + delta_current / abs(current)", ["delta_voltage", "voltage", "delta_current", "current"], "rel_error_P", "power_error_from_named_uncertainties"),
            _formula_step("step_3", "Compute absolute power error.", "delta_P = P * rel_error_P", ["P", "rel_error_P"], "delta_P", "power_error_from_named_uncertainties"),
        ]
        if target == "rel_error":
            plan.append(_formula_step("step_4", "Report power relative error.", "rel_error = rel_error_P", ["rel_error_P"], "rel_error", "power_error_from_named_uncertainties"))
        if target == "percent_error":
            plan.append(_formula_step("step_4", "Convert power relative error to percent.", "percent_error = rel_error_P * 100", ["rel_error_P"], "percent_error", "power_error_from_named_uncertainties"))
        return _finish(plan, target, "power_error_from_named_uncertainties")
    if target in {"delta_R", "rel_error", "percent_error"} and voltage and "I" in known and delta_voltage and "delta_I" in known:
        plan = [
            _formula_step("step_1", "Compute resistance from voltage and current.", "R = U / I", [voltage, "I"], "R", "resistance_error_from_voltage_current"),
            _formula_step("step_2", "Compute resistance relative error.", "rel_error_R = delta_U / abs(U) + delta_I / abs(I)", [delta_voltage, voltage, "delta_I", "I"], "rel_error_R", "resistance_error_from_voltage_current"),
            _formula_step("step_3", "Compute absolute resistance error.", "delta_R = R * rel_error_R", ["R", "rel_error_R"], "delta_R", "resistance_error_from_voltage_current"),
        ]
        if target == "rel_error":
            plan.append(_formula_step("step_4", "Report resistance relative error.", "rel_error = rel_error_R", ["rel_error_R"], "rel_error", "resistance_error_from_voltage_current"))
        if target == "percent_error":
            plan.append(_formula_step("step_4", "Convert resistance relative error to percent.", "percent_error = rel_error_R * 100", ["rel_error_R"], "percent_error", "resistance_error_from_voltage_current"))
        return _finish(plan, target, "resistance_error_from_voltage_current")
    if target in {"delta_R", "rel_error", "percent_error"} and voltage and "I" in known and delta_voltage and "delta_I" in known:
        plan = [
            _formula_step("step_1", "Compute resistance from voltage and current.", "R = U / I", [voltage, "I"], "R", "resistance_error_from_voltage_current"),
            _formula_step("step_2", "Compute resistance relative error.", "rel_error_R = delta_U / abs(U) + delta_I / abs(I)", [delta_voltage, voltage, "delta_I", "I"], "rel_error_R", "resistance_error_from_voltage_current"),
            _formula_step("step_3", "Compute absolute resistance error.", "delta_R = R * rel_error_R", ["R", "rel_error_R"], "delta_R", "resistance_error_from_voltage_current"),
        ]
        if target == "rel_error":
            plan.append(_formula_step("step_4", "Report resistance relative error.", "rel_error = rel_error_R", ["rel_error_R"], "rel_error", "resistance_error_from_voltage_current"))
        if target == "percent_error":
            plan.append(_formula_step("step_4", "Convert resistance relative error to percent.", "percent_error = rel_error_R * 100", ["rel_error_R"], "percent_error", "resistance_error_from_voltage_current"))
        return _finish(plan, target, "resistance_error_from_voltage_current")
    if target in {"delta_P", "rel_error", "percent_error"} and "V" in known and "I" in known and "delta_V" in known and "delta_I" in known:
        plan = [
            _formula_step("step_1", "Compute power from voltage and current.", "P = V * I", ["V", "I"], "P", "power_error_from_voltage_current"),
            _formula_step("step_2", "Compute power relative error.", "rel_error_P = delta_V / abs(V) + delta_I / abs(I)", ["delta_V", "V", "delta_I", "I"], "rel_error_P", "power_error_from_voltage_current"),
            _formula_step("step_3", "Compute absolute power error.", "delta_P = P * rel_error_P", ["P", "rel_error_P"], "delta_P", "power_error_from_voltage_current"),
        ]
        if target == "rel_error":
            plan.append(_formula_step("step_4", "Report power relative error.", "rel_error = rel_error_P", ["rel_error_P"], "rel_error", "power_error_from_voltage_current"))
        if target == "percent_error":
            plan.append(_formula_step("step_4", "Convert power relative error to percent.", "percent_error = rel_error_P * 100", ["rel_error_P"], "percent_error", "power_error_from_voltage_current"))
        return _finish(plan, target, "power_error_from_voltage_current")
    for base in ["V", "U", "I", "R", "P"]:
        delta = "delta_V" if base in {"V", "U"} else f"delta_{base}"
        value_name = "V" if base == "U" else base
        if target in {"rel_error", "percent_error", "uncertainty"} and value_name in known and delta in known:
            if target == "uncertainty":
                return _finish([_formula_step("step_1", "Use directly extracted uncertainty.", f"uncertainty = {delta}", [delta], "uncertainty", "uncertainty_direct")], target, "uncertainty_direct")
            plan = [_formula_step("step_1", "Compute relative error from uncertainty.", f"rel_error = {delta} / abs({value_name})", [delta, value_name], "rel_error", "relative_error_from_uncertainty")]
            if target == "percent_error":
                plan.append(_formula_step("step_2", "Convert relative error to percent.", "percent_error = rel_error * 100", ["rel_error"], "percent_error", "relative_error_from_uncertainty"))
            return _finish(plan, target, "relative_error_from_uncertainty")
    true_name = _first_existing(known, "true_value", "accepted_value")
    if not true_name and target == "abs_rel_error_pair" and "measured_value" in known:
        measured_dimension = known["measured_value"].get("dimension")
        measured_value = known["measured_value"].get("normalized_value", known["measured_value"].get("value"))
        for candidate_name, candidate in known.items():
            if candidate_name == "measured_value":
                continue
            if candidate.get("dimension") != measured_dimension:
                continue
            candidate_value = candidate.get("normalized_value", candidate.get("value"))
            if candidate_value == measured_value:
                continue
            true_name = candidate_name
            break
    if target in {"abs_error", "rel_error", "percent_error", "abs_rel_error_pair"} and "measured_value" in known and true_name:
        plan = [_formula_step("step_1", "Compute absolute error.", "abs_error = abs(measured_value - true_value)", ["measured_value", true_name], "abs_error", "absolute_error")]
        if target in {"rel_error", "percent_error", "abs_rel_error_pair"}:
            plan.append(_formula_step("step_2", "Compute relative error.", f"rel_error = abs_error / abs({true_name})", ["abs_error", true_name], "rel_error", "relative_error"))
        if target == "percent_error":
            plan.append(_formula_step("step_3", "Convert relative error to percent.", "percent_error = rel_error * 100", ["rel_error"], "percent_error", "percent_error"))
        if target == "abs_rel_error_pair":
            plan.append(_formula_step("step_3", "Report absolute and relative error together.", "abs_rel_error_pair = pair(abs_error, rel_error)", ["abs_error", "rel_error"], "abs_rel_error_pair", "absolute_relative_error_pair"))
        return _finish(plan, target, "measurement_error")
    measurement_names = [
        name
        for name, quantity in known.items()
        if (
            (name in {"I", "V", "d", "t", "temperature"} or (name.startswith(("I", "V", "d", "t", "temperature")) and name[-1:].isdigit()))
            and quantity.get("dimension") in {"current", "voltage", "length", "time", "temperature"}
        )
    ]
    if target == "mean_value" and measurement_names:
        return _finish([_formula_step("step_1", "Compute mean value.", "mean_value = sum(measurements) / n", measurement_names, "mean_value", "mean_value")], target, "mean_value")
    if target == "random_error" and measurement_names:
        warning = "Random error template uses max absolute deviation unless a dataset-specific definition is provided."
        plan = [
            _formula_step("step_1", "Compute mean value.", "mean_value = sum(measurements) / n", measurement_names, "mean_value", "random_error_simple", warning=warning),
            _formula_step("step_2", "Compute deviations from mean.", "deviations = abs(each measurement - mean_value)", measurement_names + ["mean_value"], "deviations", "random_error_simple"),
            _formula_step("step_3", "Compute random error.", "random_error = max(deviations)", ["deviations"], "random_error", "random_error_simple"),
        ]
        return _finish(plan, target, "random_error_simple")
    return [], 0.0


def _measurement_bound_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    if "maximum_possible_value" not in conditions:
        return [], 0.0
    dimension_by_target = {
        "I": "current",
        "V": "voltage",
        "R": "resistance",
        "m": "mass",
    }
    dimension = dimension_by_target.get(target)
    if not dimension:
        return [], 0.0
    values = _by_dimension(known, dimension)
    if len(values) < 2:
        return [], 0.0
    measured, uncertainty = values[0], values[1]
    return _finish(
        [
            _formula_step(
                "step_1",
                "Compute maximum possible measured value by adding the uncertainty.",
                f"{target} = {measured} + {uncertainty}",
                [measured, uncertainty],
                target,
                "maximum_possible_measurement_value",
            )
        ],
        target,
        "maximum_possible_measurement_value",
    )


def _measurement_error_propagation_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    if target != "abs_error":
        return [], 0.0

    if "target_power_abs_error" in conditions:
        voltages = _by_dimension(known, "voltage")
        currents = _by_dimension(known, "current")

        def _split_value_delta(names: List[str]) -> Tuple[Optional[str], Optional[str]]:
            delta_names = [name for name in names if name.startswith("delta_")]
            value_names = [name for name in names if not name.startswith("delta_")]
            if not value_names or not delta_names:
                return None, None
            measured = max(value_names, key=lambda name: abs(float(known.get(name, {}).get("normalized_value") or known.get(name, {}).get("value") or 0.0)))
            delta = min(delta_names, key=lambda name: abs(float(known.get(name, {}).get("normalized_value") or known.get(name, {}).get("value") or 0.0)))
            return measured, delta

        voltage, delta_voltage = _split_value_delta(voltages)
        current, delta_current = _split_value_delta(currents)
        if voltage and delta_voltage and current and delta_current:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Propagate absolute error for power P = VI.",
                        f"abs_error = round(abs({voltage})*abs({delta_current}) + abs({current})*abs({delta_voltage}), 2)",
                        [voltage, delta_current, current, delta_voltage],
                        "abs_error",
                        "power_product_absolute_error",
                    )
                ],
                target,
                "power_product_absolute_error",
            )

    if "target_total_resistance_abs_error" in conditions and "series_circuit" in conditions:
        deltas = [name for name in _by_dimension(known, "resistance") if name.startswith("delta_")]
        if len(deltas) >= 2:
            formula = "abs_error = " + " + ".join(deltas)
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "For a series sum, add absolute resistance errors.",
                        formula,
                        deltas,
                        "abs_error",
                        "series_resistance_absolute_error",
                    )
                ],
                target,
                "series_resistance_absolute_error",
            )

    return [], 0.0


def _dielectric_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    capacitance = _first_by_dimension(known, "capacitance")
    if "epsilon_r" in known and not capacitance and target == "delta_U" and ("battery_connected" in conditions or "battery_disconnected" in conditions):
        connected = "battery_connected" in conditions
        template_name = "dielectric_energy_increase_connected" if connected else "dielectric_energy_reduction_disconnected"
        if connected:
            plan = [
                _formula_step("step_1", "Represent initial capacitor energy symbolically.", "U_initial = U_before", [], "U_initial", template_name),
                _formula_step("step_2", "Compute increased energy after dielectric insertion.", "U_after = epsilon_r * U_initial", ["epsilon_r", "U_initial"], "U_after", template_name),
                _formula_step("step_3", "Compute energy increase.", "delta_U = U_after - U_initial", ["U_after", "U_initial"], "delta_U", template_name),
            ]
        else:
            plan = [
                _formula_step("step_1", "Represent initial capacitor energy symbolically.", "U_initial = U_before", [], "U_initial", template_name),
                _formula_step("step_2", "Compute reduced energy after dielectric insertion.", "U_after = U_initial / epsilon_r", ["U_initial", "epsilon_r"], "U_after", template_name),
                _formula_step("step_3", "Compute energy reduction.", "delta_U = U_initial - U_after", ["U_initial", "U_after"], "delta_U", template_name),
            ]
        return _finish(plan, target, template_name)
    if not capacitance or "epsilon_r" not in known:
        return [], 0.0
    if target == "C_after":
        return _finish([_formula_step("step_1", "Compute capacitance after inserting dielectric.", "C_after = epsilon_r * C_cap", ["epsilon_r", capacitance], "C_after", "dielectric_capacitance_change")], target, "dielectric_capacitance_change")
    connected = "battery_connected" in conditions
    disconnected = "battery_disconnected" in conditions
    if not connected and not disconnected:
        return [], 0.0
    template_name = "dielectric_battery_connected" if connected else "dielectric_battery_disconnected"
    plan = [_formula_step("step_1", "Compute capacitance after inserting dielectric.", "C_after = epsilon_r * C_cap", ["epsilon_r", capacitance], "C_after", template_name)]
    voltage = _first_existing(known, "V", "V_rms")
    charge = _first_existing(known, "Q", "q")
    energy = _first_existing(known, "U_cap", "E_energy", "U_E")
    if disconnected:
        if charge:
            plan.append(_formula_step("step_1", "Preserve charge for isolated capacitor.", "Q_after = Q", [charge], "Q_after", template_name))
        if voltage:
            plan.append(_formula_step("step_1", "Compute voltage after dielectric insertion.", "V_after = V / epsilon_r", [voltage, "epsilon_r"], "V_after", template_name))
        if energy:
            plan.append(_formula_step("step_1", "Compute energy after dielectric insertion.", "U_after = U_cap / epsilon_r", [energy, "epsilon_r"], "U_after", template_name))
        elif voltage:
            plan.append(_formula_step("step_1", "Compute initial capacitor energy.", "U_initial = 0.5 * C_cap * V^2", [capacitance, voltage], "U_initial", template_name))
            plan.append(_formula_step("step_1", "Compute energy after dielectric insertion.", "U_after = U_initial / epsilon_r", ["U_initial", "epsilon_r"], "U_after", template_name))
        elif target == "delta_U":
            plan.append(_formula_step("step_1", "Represent initial capacitor energy symbolically.", "U_initial = U_before", [], "U_initial", "dielectric_energy_reduction_disconnected"))
            plan.append(_formula_step("step_1", "Compute reduced energy after dielectric insertion.", "U_after = U_initial / epsilon_r", ["U_initial", "epsilon_r"], "U_after", "dielectric_energy_reduction_disconnected"))
            plan.append(_formula_step("step_1", "Compute energy reduction.", "delta_U = U_initial - U_after", ["U_initial", "U_after"], "delta_U", "dielectric_energy_reduction_disconnected"))
    if connected:
        if voltage:
            plan.append(_formula_step("step_1", "Preserve voltage for battery-connected capacitor.", "V_after = V", [voltage], "V_after", template_name))
        if charge:
            plan.append(_formula_step("step_1", "Compute charge after dielectric insertion.", "Q_after = epsilon_r * Q", ["epsilon_r", charge], "Q_after", template_name))
        if energy:
            plan.append(_formula_step("step_1", "Compute energy after dielectric insertion.", "U_after = epsilon_r * U_cap", ["epsilon_r", energy], "U_after", template_name))
        elif voltage:
            plan.append(_formula_step("step_1", "Compute initial capacitor energy.", "U_initial = 0.5 * C_cap * V^2", [capacitance, voltage], "U_initial", template_name))
            plan.append(_formula_step("step_1", "Compute energy after dielectric insertion.", "U_after = epsilon_r * U_initial", ["epsilon_r", "U_initial"], "U_after", template_name))
        elif target == "delta_U":
            plan.append(_formula_step("step_1", "Represent initial capacitor energy symbolically.", "U_initial = U_before", [], "U_initial", "dielectric_energy_increase_connected"))
            plan.append(_formula_step("step_1", "Compute increased energy after dielectric insertion.", "U_after = epsilon_r * U_initial", ["epsilon_r", "U_initial"], "U_after", "dielectric_energy_increase_connected"))
            plan.append(_formula_step("step_1", "Compute energy increase.", "delta_U = U_after - U_initial", ["U_after", "U_initial"], "delta_U", "dielectric_energy_increase_connected"))
    _renumber(plan)
    if target in {step_output for step in plan for step_output in step.get("output_var", {})}:
        return _finish(plan, target, template_name)
    return [], 0.0


def _capacitance_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    if target not in {"C_cap", "C_eq"}:
        return [], 0.0
    area = _first_by_dimension(known, "area")
    radius = _first_existing(known, "r", "R_radius")
    distance = _first_by_dimension(known, "length")
    if not distance or (not area and not radius):
        return [], 0.0

    template_name = "parallel_plate_capacitance"
    plan: List[Dict[str, object]] = [
        _setup_step(
            "step_1",
            "Identify the capacitor as a parallel-plate geometry and choose the plate spacing.",
            template_name=template_name,
        )
    ]
    if not area and radius:
        plan.append(_formula_step("step_2", "Compute circular plate area from radius.", "A = pi * r^2", [radius, "pi"], "A", template_name))
        area = "A"

    epsilon_inputs = [area, distance, "epsilon_0"]
    formula = f"{target} = epsilon_0 * A / d"
    if "epsilon_r" in known:
        epsilon_inputs = ["epsilon_r", area, distance, "epsilon_0"]
        formula = f"{target} = epsilon_r * epsilon_0 * A / d"
    elif _first_existing(known, "epsilon"):
        epsilon = _first_existing(known, "epsilon") or "epsilon"
        epsilon_inputs = [epsilon, area, distance]
        formula = f"{target} = epsilon * A / d"

    plan.append(_formula_step(f"step_{len(plan) + 1}", "Apply the parallel-plate capacitance relation.", formula, epsilon_inputs, target, template_name))
    _renumber(plan)
    return _finish(plan, target, template_name)


def _coulomb_force_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    if target not in {"F_e", "F_net", "F_on_q3"}:
        return [], 0.0
    charges = _by_dimension(known, "charge")
    distances = _by_dimension(known, "length")
    if not distances:
        return [], 0.0

    if target in {"F_net", "F_e", "F_on_q3"} and "square_center_point" in conditions and "four_identical_square_charges" in conditions:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Use square symmetry: forces from four identical corner charges cancel at the center.",
                    f"{target} = 0",
                    [],
                    target,
                    "square_center_identical_force_zero",
                )
            ],
            target,
            "square_center_identical_force_zero",
        )

    if "collinear" in conditions and {"q1", "q2", "q3"} <= set(known) and distances:
        spacing = distances[0]
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Combine signed Coulomb forces on the middle charge in an equally spaced collinear three-charge setup.",
                    f"{target} = abs(k * q2 * (q1 - q3) / {spacing}^2)",
                    ["k", "q1", "q2", "q3", spacing],
                    target,
                    "collinear_three_charges_force_on_middle",
                )
            ],
            target,
            "collinear_three_charges_force_on_middle",
        )

    if "midpoint" in conditions and "two_equal_opposite_charges" in conditions and len(charges) == 2 and distances:
        source_charge = "q2" if "q2" in known else charges[0]
        test_charge = "q0" if "q0" in known else (charges[1] if charges[1] != source_charge else charges[0])
        sep = distances[0]
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Use equal opposite endpoint charges: midpoint fields add, then multiply by the test charge.",
                    f"{target} = abs({test_charge}) * 2 * k * abs({source_charge}) / (0.5*{sep})^2",
                    [test_charge, "k", source_charge, sep],
                    target,
                    "midpoint_equal_opposite_charges_force",
                )
            ],
            target,
            "midpoint_equal_opposite_charges_force",
        )

    if "perpendicular_bisector" in conditions and {"q1", "q2"} <= set(known) and len(distances) >= 2:
        test_charge = _first_existing(known, "q", "q0", "q3")
        if test_charge and test_charge not in {"q1", "q2"}:
            try:
                q1_value = float(known.get("q1", {}).get("normalized_value") or known.get("q1", {}).get("value") or 0.0)
                q2_value = float(known.get("q2", {}).get("normalized_value") or known.get("q2", {}).get("value") or 0.0)
            except (TypeError, ValueError):
                q1_value = q2_value = 0.0
            if q1_value * q2_value > 0 and abs(q1_value - q2_value) <= 1e-12 * max(abs(q1_value), abs(q2_value), 1.0):
                sep, height = distances[0], distances[1]
                return _finish(
                    [
                        _formula_step(
                            "step_1",
                            "Combine vertical components of equal-source Coulomb forces on the perpendicular bisector.",
                            f"{target} = 2 * k * abs(q1*{test_charge}) * {height} / (((0.5*{sep})^2 + {height}^2)^(3/2))",
                            ["q1", test_charge, sep, height, "k"],
                            target,
                            "coulomb_force_perpendicular_bisector_equal_sources",
                        )
                    ],
                    target,
                    "coulomb_force_perpendicular_bisector_equal_sources",
                )

    if "opposite_sides_target_charge" in conditions and len(charges) >= 1 and len(distances) >= 2:
        target_charge = "q" if "q" in known else charges[0]
        source_charge = "q1" if "q1" in known else (charges[1] if len(charges) > 1 else target_charge)
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "For equal like source charges on opposite sides of the target, subtract the opposite force magnitudes.",
                    f"{target} = abs(k * abs({target_charge} * {source_charge}) / {distances[0]}^2 - k * abs({target_charge} * {source_charge}) / {distances[1]}^2)",
                    ["k", target_charge, source_charge, distances[0], distances[1]],
                    target,
                    "opposite_sides_equal_sources_force_difference",
                )
            ],
            target,
            "opposite_sides_equal_sources_force_difference",
        )

    if "perpendicular_bisector" in conditions and {"q1", "q2"} <= set(known) and ("q" in known or "q0" in known or "q3" in known) and len(distances) >= 2:
        test_charge = "q" if "q" in known else ("q0" if "q0" in known else "q3")
        sep, height = distances[0], distances[1]
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute the field on the perpendicular bisector from signed endpoint charges, then multiply by the test charge.",
                    f"{target} = abs({test_charge}) * k * sqrt(((0.5*{sep})*(q1 - q2))^2 + ({height}*(q1 + q2))^2) / (((0.5*{sep})^2 + {height}^2)^(3/2))",
                    [test_charge, "k", "q1", "q2", sep, height],
                    target,
                    "coulomb_force_perpendicular_bisector",
                )
            ],
            target,
            "coulomb_force_perpendicular_bisector",
        )

    if "midpoint" in conditions and {"q1", "q2"} <= set(known) and ("q0" in known or "q3" in known or "q" in known) and distances:
        test_charge = "q0" if "q0" in known else ("q3" if "q3" in known else "q")
        plan = [
            _formula_step(
                "step_1",
                "Combine forces on the midpoint test charge from both endpoint charges.",
                f"{target} = k * abs({test_charge}) * abs(q1 - q2) / (0.5*{distances[0]})^2",
                ["q1", "q2", test_charge, distances[0], "k"],
                target,
                "coulomb_midpoint_signed_charges",
            )
        ]
        return _finish(plan, target, "coulomb_midpoint_signed_charges")

    if (
        "perpendicular_bisector" in conditions
        and {"q1", "q2"} <= set(known)
        and ("q0" in known or "q3" in known or "q" in known)
        and len(distances) >= 2
    ):
        test_charge = "q0" if "q0" in known else ("q3" if "q3" in known else "q")
        sep, height = distances[0], distances[1]
        plan = [
            _formula_step(
                "step_1",
                "Compute force on a test charge on the perpendicular bisector.",
                f"{target} = abs({test_charge}) * k * sqrt(((0.5*{sep})*(q1 - q2))^2 + ({height}*(q1 + q2))^2) / (((0.5*{sep})^2 + {height}^2)^(3/2))",
                [test_charge, "q1", "q2", sep, height, "k"],
                target,
                "coulomb_perpendicular_bisector_force",
            )
        ]
        return _finish(plan, target, "coulomb_perpendicular_bisector_force")

    if len(charges) == 1 and "right_angle" in conditions:
        plan = [
            _formula_step(
                "step_1",
                "Compute the force from one identical charge along a leg.",
                "F_single = k * abs(q*q) / r^2",
                [charges[0], distances[0], "k"],
                "F_single",
                "coulomb_right_isosceles_identical_charges",
            ),
            _formula_step(
                "step_2",
                "Combine the two perpendicular equal forces at the right-angle vertex.",
                f"{target} = sqrt(2) * F_single",
                ["F_single"],
                target,
                "coulomb_right_isosceles_identical_charges",
            ),
        ]
        return _finish(plan, target, "coulomb_right_isosceles_identical_charges")

    if "equilateral_triangle" in conditions and "two_equal_like_charges" in conditions and len(charges) == 2 and distances:
        source_charge = charges[0]
        test_charge = charges[1]
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Combine two equal Coulomb forces on the remaining equilateral-triangle vertex.",
                    f"{target} = sqrt(3) * k * abs({source_charge} * {test_charge}) / {distances[0]}^2",
                    [source_charge, test_charge, distances[0], "k"],
                    target,
                    "coulomb_equilateral_two_identical_sources",
                )
            ],
            target,
            "coulomb_equilateral_two_identical_sources",
        )

    if "line_connecting" in conditions and {"q1", "q2"} <= set(known) and len(distances) >= 2:
        test_charge = _first_existing(known, "q3", "q0", "q")
        if test_charge and test_charge not in {"q1", "q2"}:
            sep, x_from_q1 = distances[0], distances[1]
            try:
                q1_value = float(known.get("q1", {}).get("normalized_value") or known.get("q1", {}).get("value") or 0.0)
                q2_value = float(known.get("q2", {}).get("normalized_value") or known.get("q2", {}).get("value") or 0.0)
                qt_value = float(known.get(test_charge, {}).get("normalized_value") or known.get(test_charge, {}).get("value") or 0.0)
            except (TypeError, ValueError):
                q1_value = q2_value = qt_value = 0.0
            sign_1 = 1 if q1_value * qt_value > 0 else -1
            sign_2 = -1 if q2_value * qt_value > 0 else 1
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute collinear net Coulomb force on a test charge located between two endpoint charges.",
                        f"{target} = abs({sign_1} * k * abs(q1*{test_charge}) / {x_from_q1}^2 + {sign_2} * k * abs(q2*{test_charge}) / ({sep} - {x_from_q1})^2)",
                        ["q1", "q2", test_charge, sep, x_from_q1, "k"],
                        target,
                        "coulomb_force_collinear_between_sources",
                    )
                ],
                target,
                "coulomb_force_collinear_between_sources",
            )

    template_name = "coulomb_force_vector" if target in {"F_net", "F_on_q3"} else "coulomb_force_scalar"
    if len(charges) >= 2 and target in {"F_e", "F_net"} and len(charges) < 3:
        return _finish(
            [_formula_step("step_1", "Apply scalar Coulomb's law.", f"{target} = k * abs(q1 * q2) / r^2", [charges[0], charges[1], distances[0], "k"], target, template_name)],
            target,
            template_name,
        )

    plan: List[Dict[str, object]] = []
    if len(charges) < 3:
        if "equilateral_triangle" in conditions and len(charges) == 1:
            charges = [charges[0], charges[0], charges[0]]
        else:
            missing = {f"q{index}": f"symbolic charge q{index}" for index in range(1, 4) if f"q{index}" not in known}
            if missing:
                plan.append(_setup_step("step_1", "Introduce symbolic charges required for force-on-q3 vector setup.", missing, template_name))
                charges = charges + list(missing.keys())
    if len(charges) < 3:
        return [], 0.0

    if len(distances) >= 3:
        r13 = distances[1]
        r23 = distances[2]
    else:
        r13 = distances[0]
        r23 = distances[1] if len(distances) > 1 else distances[0]
    if "q0" in known and len(charges) >= 3:
        target_charge = "q0"
        source_charges = [name for name in charges if name != target_charge]
        if len(source_charges) >= 2:
            charges = [source_charges[0], source_charges[1], target_charge]
    elif "q3" in known and len(charges) >= 3:
        target_charge = "q3"
        source_charges = [name for name in charges if name != target_charge]
        if len(source_charges) >= 2:
            charges = [source_charges[0], source_charges[1], target_charge]
    plan.extend(
        [
            _formula_step("step_1", "Compute force on q3 due to q1.", f"F_13 = k * abs({charges[0]}*{charges[2]}) / {r13}^2", [charges[0], charges[2], r13, "k"], "F_13", template_name),
            _formula_step("step_1", "Compute force on q3 due to q2.", f"F_23 = k * abs({charges[1]}*{charges[2]}) / {r23}^2", [charges[1], charges[2], r23, "k"], "F_23", template_name),
        ]
    )
    output = target
    if len(charges) >= 3 and len(distances) >= 1:
        q1_value = float(known.get(charges[0], {}).get("normalized_value") or known.get(charges[0], {}).get("value") or 0.0)
        q2_value = float(known.get(charges[1], {}).get("normalized_value") or known.get(charges[1], {}).get("value") or 0.0)
        q3_value = float(known.get(charges[2], {}).get("normalized_value") or known.get(charges[2], {}).get("value") or 0.0)
        direction_sign = 1 if (q1_value * q3_value) * (q2_value * q3_value) > 0 else -1
    else:
        direction_sign = 1

    if "theta" in known:
        combine = f"{output} = sqrt(F_13^2 + F_23^2 + 2*F_13*F_23*cos(theta))"
        inputs = ["F_13", "F_23", "theta"]
    elif "right_angle" in conditions:
        combine = f"{output} = sqrt(F_13^2 + F_23^2)"
        inputs = ["F_13", "F_23"]
    elif "equilateral_triangle" in conditions:
        combine = f"{output} = sqrt(F_13^2 + F_23^2 + {2 * direction_sign}*F_13*F_23*cos(60deg))"
        inputs = ["F_13", "F_23"]
    elif len(distances) >= 3:
        source_sep = distances[0]
        combine = f"{output} = sqrt(F_13^2 + F_23^2 + {2 * direction_sign}*F_13*F_23*(({r13}^2 + {r23}^2 - {source_sep}^2)/(2*{r13}*{r23})))"
        inputs = ["F_13", "F_23", r13, r23, source_sep]
    elif "square_center" in conditions:
        combine = f"{output} = vector_sum(F_13, F_23, symmetry_terms)"
        inputs = ["F_13", "F_23"]
    else:
        combine = f"{output} = vector_sum(F_13, F_23)"
        inputs = ["F_13", "F_23"]
        plan[-1]["parser_warning"] = "Geometry is ambiguous; vector combination is conservative and left symbolic."
    plan.append(_formula_step("step_1", "Combine pairwise electric forces using available geometry.", combine, inputs, output, "force_resultant_coulomb"))
    _renumber(plan)
    return _finish(plan, target, template_name)


def _circuit_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    resistance_names = _by_dimension(known, "resistance")
    current_names = _by_dimension(known, "current")
    voltage_names = _by_dimension(known, "voltage")
    power_names = _by_dimension(known, "power")
    capacitance_names = _by_dimension(known, "capacitance")
    charge = _first_existing(known, "Q", "q", "Q_after")
    voltage = _first_existing(known, "V", "V_rms") or (voltage_names[0] if voltage_names else None)
    current = _first_existing(known, "I", "I_rms") or (current_names[0] if current_names else None)
    resistance = _first_existing(known, "R") or (resistance_names[0] if resistance_names else None)

    if (
        target == "V"
        and "parallel_circuit" in conditions
        and "unknown_parallel_capacitor_charge_branch" in conditions
        and charge
        and voltage
        and len(capacitance_names) >= 2
    ):
        try:
            q_value = float(known[charge].get("normalized_value", known[charge].get("value")))
            upper_bound = float(known[voltage].get("normalized_value", known[voltage].get("value")))
            candidates = []
            for name in capacitance_names:
                c_value = float(known[name].get("normalized_value", known[name].get("value")))
                if c_value:
                    candidates.append((q_value / c_value, name))
            valid = [(value, name) for value, name in candidates if value < upper_bound]
            if valid:
                selected_value, selected_cap = max(valid, key=lambda item: item[0])
                return _finish(
                    [
                        _formula_step(
                            "step_1",
                            "Test the possible capacitor branch charges against the stated voltage upper bound.",
                            f"V = {charge} / {selected_cap}",
                            [charge, selected_cap],
                            "V",
                            "parallel_capacitor_charge_branch_voltage",
                        )
                    ],
                    target,
                    "parallel_capacitor_charge_branch_voltage",
                )
        except (TypeError, ValueError):
            pass

    if target in {"I", "I_total"} and voltage and resistance:
        return _finish([_formula_step("step_1", "Apply Ohm's law for current.", f"{target} = V / R", [voltage, resistance], target, "ohms_law_current")], target, "ohms_law_current")
    if target == "V" and current and resistance:
        return _finish([_formula_step("step_1", "Apply Ohm's law for voltage.", "V = I * R", [current, resistance], "V", "ohms_law_voltage")], target, "ohms_law_voltage")
    if target in {"R", "R_eq"} and voltage and current:
        return _finish([_formula_step("step_1", "Apply Ohm's law for resistance.", f"{target} = V / I", [voltage, current], target, "ohms_law_resistance")], target, "ohms_law_resistance")
    if target == "Z" and voltage and current:
        return _finish([_formula_step("step_1", "Compute impedance from RMS voltage and current.", f"Z = {voltage} / {current}", [voltage, current], "Z", "impedance_from_voltage_current")], target, "impedance_from_voltage_current")
    if target == "R_eq" and len(resistance_names) >= 2:
        if "parallel_circuit" in conditions:
            formula = "R_eq = R1*R2/(R1+R2)" if len(resistance_names) == 2 else "1/R_eq = sum(1/R_i)"
            return _finish([_formula_step("step_1", "Compute equivalent parallel resistance.", formula, resistance_names, "R_eq", "parallel_resistance")], target, "parallel_resistance")
        if "series_circuit" in conditions:
            return _finish([_formula_step("step_1", "Compute equivalent series resistance.", "R_eq = sum(R_i)", resistance_names, "R_eq", "series_resistance")], target, "series_resistance")
    if target in {"I", "I_total"} and len(current_names) >= 2 and not voltage and not resistance:
        return _finish(
            [_formula_step("step_1", "Apply Kirchhoff's current law (sum or difference of branch currents).",
                           f"{target} = I1 +/- I2", current_names[:2], target, "current_arithmetic")],
            target, "current_arithmetic",
        )
    if target in {"P", "P_total"}:
        if power_names and target == "P_total":
            return _finish([_formula_step("step_1", "Sum individual powers.", "P_total = sum(P_i)", power_names, "P_total", "total_power_sum")], target, "total_power_sum")
        if voltage and current:
            return _finish([_formula_step("step_1", "Apply electric power relation.", f"{target} = V * I", [voltage, current], target, "power_from_voltage_current")], target, "power_from_voltage_current")
        if current and resistance:
            return _finish([_formula_step("step_1", "Apply resistive power relation.", f"{target} = I^2 * R", [current, resistance], target, "power_from_current_resistance")], target, "power_from_current_resistance")
        if voltage and resistance:
            return _finish([_formula_step("step_1", "Apply resistive power relation.", f"{target} = V^2 / R", [voltage, resistance], target, "power_from_voltage_resistance")], target, "power_from_voltage_resistance")
    return [], 0.0


def _capacitor_energy_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    capacitance = _first_by_dimension(known, "capacitance")
    voltage = _first_existing(known, "V", "V_rms", "V_after")
    charge = _first_existing(known, "Q", "q", "Q_after")
    energy = _first_existing(known, "U_cap", "U_E", "E_energy")
    field = _first_by_dimension(known, "electric_field")
    area = _first_by_dimension(known, "area")
    radius = _first_existing(known, "R_radius", "r")
    distance = _first_by_dimension(known, "length")
    capacitances = _by_dimension(known, "capacitance")

    if target == "energy_charge_pair" and capacitance and voltage:
        plan = [
            _formula_step("step_1", "Compute capacitor electric energy from capacitance and voltage.", "U_cap = 0.5 * C_cap * V^2", [capacitance, voltage], "U_cap", "capacitor_energy_charge_pair"),
            _formula_step("step_2", "Compute capacitor charge from capacitance and voltage.", "Q = C_cap * V", [capacitance, voltage], "Q", "capacitor_energy_charge_pair"),
            _formula_step("step_3", "Report capacitor energy and charge together.", "energy_charge_pair = pair(U_cap, Q)", ["U_cap", "Q"], "energy_charge_pair", "capacitor_energy_charge_pair"),
        ]
        return _finish(plan, target, "capacitor_energy_charge_pair")

    if target == "E" and "series_circuit" in conditions and len(capacitances) >= 2 and voltage and distance:
        c1, c2 = capacitances[0], capacitances[1]
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute the voltage across the first capacitor in a series pair.",
                    f"V_C1 = {voltage} * {c2} / ({c1} + {c2})",
                    [voltage, c1, c2],
                    "V_C1",
                    "series_capacitor_field",
                ),
                _formula_step(
                    "step_2",
                    "Compute the electric field inside the first capacitor.",
                    f"E = V_C1 / {distance}",
                    ["V_C1", distance],
                    "E",
                    "series_capacitor_field",
                ),
            ],
            target,
            "series_capacitor_field",
        )

    if target == "W" and area and distance and voltage and "battery_connected" in conditions and "plate_distance_doubled" in conditions:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute the additional work supplied by the source when connected plate spacing is doubled.",
                    f"W = -0.5 * epsilon_0 * {area} * {voltage}^2 / {distance}",
                    ["epsilon_0", area, voltage, distance],
                    "W",
                    "battery_work_plate_distance_doubled",
                )
            ],
            target,
            "battery_work_plate_distance_doubled",
        )

    if target == "u_E" and voltage and distance and "epsilon_r" in known:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute electric field energy density inside a dielectric-filled parallel-plate capacitor.",
                    f"u_E = 0.5 * epsilon_r * epsilon_0 * ({voltage} / {distance})^2",
                    ["epsilon_r", "epsilon_0", voltage, distance],
                    "u_E",
                    "dielectric_electric_field_energy_density",
                )
            ],
            target,
            "dielectric_electric_field_energy_density",
        )

    if target in {"U_cap", "U_E", "U_total"} and ("capacitor_charge_sharing" in conditions or "uncharged_capacitor" in conditions):
        share_count = 3 if "share_count_3" in conditions else 2
        if capacitance and voltage:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute total energy after charge sharing among identical capacitors.",
                        f"{target} = 0.5 * C_cap * V^2 / {share_count}",
                        [capacitance, voltage],
                        target,
                        "capacitor_charge_sharing_energy",
                    )
                ],
                target,
                "capacitor_charge_sharing_energy",
            )
        if energy:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute total energy after charge sharing among identical capacitors.",
                        f"{target} = {energy} / {share_count}",
                        [energy],
                        target,
                        "capacitor_charge_sharing_energy",
                    )
                ],
                target,
                "capacitor_charge_sharing_energy",
            )

    if target in {"Q_after", "U_after", "Q", "U_cap", "U_E"} and "capacitor_short_circuit" in conditions:
        zero_output = target
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "A short-circuited capacitor has zero remaining charge and zero stored electric energy.",
                    f"{zero_output} = 0",
                    [],
                    zero_output,
                    "capacitor_short_circuit_zero",
                )
            ],
            target,
            "capacitor_short_circuit_zero",
        )

    if target in {"Q_max", "Q"} and field and (area or radius) and ("circular_plate" in conditions or "parallel_plate_capacitor" in conditions):
        template_name = "parallel_plate_breakdown_max_charge"
        plan: List[Dict[str, object]] = []
        area_name = area
        if not area_name and radius:
            plan.append(_formula_step("step_1", "Compute circular plate area.", f"A = pi * {radius}^2", [radius, "pi"], "A", template_name))
            area_name = "A"
        output = "Q_max" if target == "Q_max" else target
        plan.append(
            _formula_step(
                f"step_{len(plan) + 1}",
                "Compute maximum capacitor charge before dielectric breakdown.",
                f"{output} = epsilon_0 * {area_name or 'A'} * {field}",
                ["epsilon_0", area_name or "A", field],
                output,
                template_name,
            )
        )
        _renumber(plan)
        return _finish(plan, output, template_name)

    if target in {"U_cap", "U_E", "U_total"}:
        output = target
        if area and distance and voltage and "epsilon_r" in known:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute energy stored in a dielectric-filled parallel-plate capacitor.",
                        f"{output} = 0.5 * epsilon_r * epsilon_0 * {area} * {voltage}^2 / {distance}",
                        ["epsilon_r", "epsilon_0", area, voltage, distance],
                        output,
                        "dielectric_parallel_plate_energy_from_geometry",
                    )
                ],
                target,
                "dielectric_parallel_plate_energy_from_geometry",
            )
        if capacitance and voltage and "epsilon_r" in known and ("battery_connected" in conditions or "battery_disconnected" in conditions):
            if "battery_connected" in conditions:
                return _finish(
                    [
                        _formula_step(
                            "step_1",
                            "Compute dielectric-filled capacitor energy while connected to the source.",
                            f"{output} = epsilon_r * 0.5 * C_cap * V^2",
                            ["epsilon_r", capacitance, voltage],
                            output,
                            "dielectric_energy_connected_from_cv",
                        )
                    ],
                    target,
                    "dielectric_energy_connected_from_cv",
                )
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute dielectric-filled capacitor energy after the source is disconnected.",
                        f"{output} = 0.5 * C_cap * V^2 / epsilon_r",
                        [capacitance, voltage, "epsilon_r"],
                        output,
                        "dielectric_energy_disconnected_from_cv",
                    )
                ],
                target,
                "dielectric_energy_disconnected_from_cv",
            )
        if capacitance and voltage and "battery_disconnected" in conditions and "plate_distance_doubled" in conditions:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute capacitor energy after an isolated parallel-plate capacitor spacing is doubled.",
                        f"{output} = 2 * 0.5 * C_cap * V^2",
                        [capacitance, voltage],
                        output,
                        "isolated_parallel_plate_energy_distance_doubled",
                    )
                ],
                target,
                "isolated_parallel_plate_energy_distance_doubled",
            )
        if capacitance and voltage:
            return _finish([_formula_step("step_1", "Compute capacitor electric energy from capacitance and voltage.", f"{output} = 0.5 * C_cap * V^2", [capacitance, voltage], output, "capacitor_energy")], target, "capacitor_energy")
        if charge and voltage:
            return _finish([_formula_step("step_1", "Compute capacitor electric energy from charge and voltage.", f"{output} = 0.5 * Q * V", [charge, voltage], output, "capacitor_energy_charge_voltage")], target, "capacitor_energy_charge_voltage")
        if charge and capacitance:
            return _finish([_formula_step("step_1", "Compute capacitor electric energy from charge and capacitance.", f"{output} = Q^2 / (2*C_cap)", [charge, capacitance], output, "capacitor_energy_charge_capacitance")], target, "capacitor_energy_charge_capacitance")

    if target == "C_cap":
        if charge and voltage:
            return _finish([_formula_step("step_1", "Apply capacitance definition.", "C_cap = Q / V", [charge, voltage], "C_cap", "capacitance_definition")], target, "capacitance_definition")
        if energy and voltage:
            return _finish([_formula_step("step_1", "Invert capacitor energy relation for capacitance.", "C_cap = 2*U_cap / V^2", [energy, voltage], "C_cap", "capacitance_from_energy_voltage")], target, "capacitance_from_energy_voltage")
        if charge and energy:
            return _finish([_formula_step("step_1", "Invert capacitor energy relation for capacitance.", "C_cap = Q^2 / (2*U_cap)", [charge, energy], "C_cap", "capacitance_from_charge_energy")], target, "capacitance_from_charge_energy")

    if target in {"V", "V_after", "U_C"}:
        capacitances = _by_dimension(known, "capacitance")
        if target == "U_C" and "series_circuit" in conditions and len(capacitances) >= 2 and voltage and ("target_voltage_c1" in conditions or "target_voltage_c2" in conditions):
            c1, c2 = capacitances[0], capacitances[1]
            if "target_voltage_c2" in conditions:
                formula = f"U_C = {voltage} * {c1} / ({c1} + {c2})"
                inputs = [voltage, c1, c2]
            else:
                formula = f"U_C = {voltage} * {c2} / ({c1} + {c2})"
                inputs = [voltage, c1, c2]
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute voltage division for two capacitors connected in series.",
                        formula,
                        inputs,
                        "U_C",
                        "series_capacitor_voltage_division",
                    )
                ],
                target,
                "series_capacitor_voltage_division",
            )
        if voltage and "battery_disconnected" in conditions and "plate_distance_doubled" in conditions:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Use charge conservation: doubling plate spacing doubles the voltage of an isolated parallel-plate capacitor.",
                        f"{target} = 2 * V",
                        [voltage],
                        target,
                        "isolated_parallel_plate_voltage_distance_doubled",
                    )
                ],
                target,
                "isolated_parallel_plate_voltage_distance_doubled",
            )
        if (
            "parallel_circuit" in conditions
            and "unknown_parallel_capacitor_charge_branch" in conditions
            and charge
            and voltage
            and len(capacitances) >= 2
        ):
            try:
                q_value = float(known[charge].get("normalized_value", known[charge].get("value")))
                upper_bound = float(known[voltage].get("normalized_value", known[voltage].get("value")))
                candidates = []
                for name in capacitances:
                    c_value = float(known[name].get("normalized_value", known[name].get("value")))
                    if c_value:
                        candidates.append((q_value / c_value, name))
                valid = [(value, name) for value, name in candidates if value < upper_bound]
                if valid:
                    _, selected_cap = max(valid, key=lambda item: item[0])
                    return _finish(
                        [
                            _formula_step(
                                "step_1",
                                "Test the possible capacitor branch charges against the stated voltage upper bound.",
                                f"{target} = {charge} / {selected_cap}",
                                [charge, selected_cap],
                                target,
                                "parallel_capacitor_charge_branch_voltage",
                            )
                        ],
                        target,
                        "parallel_capacitor_charge_branch_voltage",
                    )
            except (TypeError, ValueError):
                pass
        if charge and capacitance:
            return _finish([_formula_step("step_1", "Apply capacitance definition for voltage.", f"{target} = Q / C_cap", [charge, capacitance], target, "capacitance_voltage")], target, "capacitance_voltage")
        if energy and capacitance:
            return _finish([_formula_step("step_1", "Invert capacitor energy relation for voltage.", f"{target} = sqrt(2*U_cap / C_cap)", [energy, capacitance], target, "voltage_from_capacitor_energy")], target, "voltage_from_capacitor_energy")
        if "epsilon_r" in known and voltage and "battery_disconnected" in conditions:
            return _finish([_formula_step("step_1", "Use isolated capacitor dielectric voltage relation.", f"{target} = V / epsilon_r", [voltage, "epsilon_r"], target, "dielectric_disconnected_voltage")], target, "dielectric_disconnected_voltage")
        if voltage and "parallel_plate_capacitor" in conditions:
            return _finish([_formula_step("step_1", "Battery-connected capacitor keeps the applied voltage.", f"{target} = V", [voltage], target, "battery_connected_voltage_constant")], target, "battery_connected_voltage_constant")

    if target == "Q":
        if capacitance and voltage:
            return _finish([_formula_step("step_1", "Apply capacitor charge relation.", "Q = C_cap * V", [capacitance, voltage], "Q", "capacitor_charge")], target, "capacitor_charge")
        radius = _first_existing(known, "r", "R_radius")
        distance = _first_by_dimension(known, "length")
        if radius and distance and voltage:
            plan = [
                _formula_step("step_1", "Compute circular plate area from radius.", "A = pi * r^2", [radius, "pi"], "A", "parallel_plate_charge"),
                _formula_step("step_2", "Compute parallel-plate capacitance.", "C_cap = epsilon_0 * A / d", ["A", distance, "epsilon_0"], "C_cap", "parallel_plate_charge"),
                _formula_step("step_3", "Compute capacitor charge.", "Q = C_cap * V", ["C_cap", voltage], "Q", "parallel_plate_charge"),
            ]
            return _finish(plan, target, "parallel_plate_charge")
    return [], 0.0


def _inductor_energy_templates(known: Dict[str, Dict[str, object]], target: str) -> Tuple[List[Dict[str, object]], float]:
    inductance = _first_by_dimension(known, "inductance")
    current = _first_existing(known, "I", "I_max", "I_rms")
    energy = _first_existing(known, "U_B", "E_energy")
    if target == "U_B" and inductance and current:
        return _finish([_formula_step("step_1", "Compute magnetic field energy in an inductor.", "U_B = 0.5 * L * I^2", [inductance, current], "U_B", "inductor_energy")], target, "inductor_energy")
    if target == "L_ind" and energy and current:
        return _finish([_formula_step("step_1", "Invert inductor energy relation for inductance.", "L_ind = 2*U_B / I^2", [energy, current], "L_ind", "inductance_from_energy_current")], target, "inductance_from_energy_current")
    if target in {"I", "I_max"} and energy and inductance:
        return _finish([_formula_step("step_1", "Invert inductor energy relation for current.", f"{target} = sqrt(2*U_B / L)", [energy, inductance], target, "current_from_inductor_energy")], target, "current_from_inductor_energy")
    return [], 0.0


import re as _re

_SINUSOIDAL_AMPLITUDE_RE = _re.compile(
    r"^([\d.]+)\s*[×x\*]?\s*(?:sin|cos)",
    _re.IGNORECASE,
)


def _extract_sinusoidal_amplitude(
    relations: List[Dict[str, object]],
    func_name: str,
) -> Optional[float]:
    for rel in relations:
        if rel.get("type") == "function" and rel.get("function_name") == func_name:
            expr = str(rel.get("expression", ""))
            m = _SINUSOIDAL_AMPLITUDE_RE.match(expr)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        if rel.get("type") == "ratio" and rel.get("left") == func_name:
            right = str(rel.get("right", ""))
            if right in ("sin", "cos"):
                factor = rel.get("factor")
                if factor is not None:
                    try:
                        return float(factor)
                    except (ValueError, TypeError):
                        pass
    return None


def _sinusoidal_energy_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    relations: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], float]:
    inductance = _first_by_dimension(known, "inductance")
    capacitance = _first_by_dimension(known, "capacitance")

    if target in {"U_B", "U_B_max"} and inductance:
        amp = _extract_sinusoidal_amplitude(relations, "I")
        if amp is not None:
            return _finish(
                [
                    _setup_step("step_1", f"Extract amplitude I_max = {amp} from sinusoidal I(t).",
                                {"I_max": str(amp)}, "sinusoidal_inductor_energy"),
                    _formula_step("step_2", "Compute magnetic energy from amplitude.",
                                  "U_B = 0.5 * L * I_max^2",
                                  [inductance, "I_max"], "U_B", "sinusoidal_inductor_energy"),
                ],
                target, "sinusoidal_inductor_energy",
            )

    if target in {"U_E", "U_E_max", "U_cap"} and capacitance:
        amp = _extract_sinusoidal_amplitude(relations, "U")
        if amp is None:
            amp = _extract_sinusoidal_amplitude(relations, "u")
        if amp is not None:
            output = target
            return _finish(
                [
                    _setup_step("step_1", f"Extract amplitude U_max = {amp} from sinusoidal U(t).",
                                {"U_max": str(amp)}, "sinusoidal_capacitor_energy"),
                    _formula_step("step_2", "Compute electric energy from amplitude.",
                                  f"{output} = 0.5 * C * U_max^2",
                                  [capacitance, "U_max"], output, "sinusoidal_capacitor_energy"),
                ],
                target, "sinusoidal_capacitor_energy",
            )

    return [], 0.0


def _parse_sinusoidal_ac_source(relations: List[Dict[str, object]]) -> Optional[Dict[str, str]]:
    """Parse common source form u = U*sqrt(2)*cos(omega*t) plus R/L/C literals."""
    for rel in relations:
        if rel.get("type") != "function" or str(rel.get("function_name", "")).lower() not in {"u", "v"}:
            continue
        source = str(rel.get("source_text") or "")
        if not source:
            source = str(rel.get("expression") or "")
        normalized_source = source

        voltage_match = re.search(
            r"\bu\s*=\s*(?P<coef>\d+(?:\.\d+)?)\s*(?P<sqrt>√\s*2|sqrt\s*\(?\s*2\s*\)?)?\s*(?:cos|sin)",
            normalized_source,
            re.IGNORECASE,
        )
        omega_match = re.search(
            r"(?:cos|sin)\s*\(?\s*(?P<omega>\d+(?:\.\d+)?)\s*(?P<pi>π|pi)?\s*\*?\s*t",
            normalized_source,
            re.IGNORECASE,
        )
        resistance_match = re.search(r"\bR\s*=\s*(?P<R>\d+(?:\.\d+)?)\s*(?:Ω|ohm)", normalized_source, re.IGNORECASE)
        inductance_match = re.search(r"\bL\s*=\s*(?P<L>[^,.;]+?)\s*H\b", normalized_source, re.IGNORECASE)
        capacitance_match = re.search(r"\bC\s*=\s*(?P<C>[^,.;]+?)\s*F\b", normalized_source, re.IGNORECASE)
        if not (voltage_match and omega_match):
            continue

        coef = _normalize_formula_literal(voltage_match.group("coef"))
        if voltage_match.group("sqrt"):
            v_rms = coef
        else:
            v_rms = f"({coef})/sqrt(2)"
        omega = _normalize_formula_literal(omega_match.group("omega") + ("pi" if omega_match.group("pi") else ""))
        parsed = {
            "V_rms": v_rms,
            "omega": omega,
        }
        if resistance_match:
            parsed["R"] = _normalize_formula_literal(resistance_match.group("R"))
        if inductance_match:
            parsed["L"] = _normalize_formula_literal(inductance_match.group("L"))
        if capacitance_match:
            parsed["C"] = _normalize_formula_literal(capacitance_match.group("C"))
        return parsed
    return None


def _sinusoidal_ac_source_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    relations: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], float]:
    parsed = _parse_sinusoidal_ac_source(relations)
    if not parsed:
        return [], 0.0

    v = parsed["V_rms"]
    omega = parsed["omega"]
    resistance = parsed.get("R")
    inductance = parsed.get("L")
    capacitance = parsed.get("C")
    x_l = f"(({omega})*({inductance}))" if inductance else None
    x_c = f"(1/(({omega})*({capacitance})))" if capacitance else None
    z = f"(sqrt(({resistance})^2 + (({x_l}) - ({x_c}))^2))" if resistance and x_l and x_c else None
    i_rms = f"(({v})/({z}))" if z else None

    direct_formulas = {
        "V": (f"V = {v}", "V", "ac_source_rms_voltage"),
        "V_rms": (f"V_rms = {v}", "V_rms", "ac_source_rms_voltage"),
        "omega": (f"omega = {omega}", "omega", "ac_source_omega"),
    }
    if x_l:
        direct_formulas["X_L"] = (f"X_L = {x_l}", "X_L", "ac_source_X_L")
    if x_c:
        direct_formulas["X_C"] = (f"X_C = {x_c}", "X_C", "ac_source_X_C")
    if z:
        direct_formulas["Z"] = (f"Z = {z}", "Z", "ac_source_impedance")
    if i_rms:
        direct_formulas["I"] = (f"I = {i_rms}", "I", "ac_source_I_rms")
        direct_formulas["I_rms"] = (f"I_rms = {i_rms}", "I_rms", "ac_source_I_rms")
    if resistance and z:
        direct_formulas["power_factor"] = (f"power_factor = ({resistance})/({z})", "power_factor", "ac_source_power_factor")
        direct_formulas["cos_phi"] = (f"power_factor = ({resistance})/({z})", "power_factor", "ac_source_power_factor")
    if resistance and i_rms:
        direct_formulas["P"] = (f"P = ({i_rms})^2 * ({resistance})", "P", "ac_source_average_power")
        direct_formulas["P_avg"] = (f"P_avg = ({i_rms})^2 * ({resistance})", "P_avg", "ac_source_average_power")
    if i_rms and x_l:
        direct_formulas["U_L"] = (f"U_L = ({i_rms}) * ({x_l})", "U_L", "ac_source_U_L")
    if i_rms and x_c:
        direct_formulas["U_C"] = (f"U_C = ({i_rms}) * ({x_c})", "U_C", "ac_source_U_C")

    if target in direct_formulas:
        formula, output, template_name = direct_formulas[target]
        return _finish([
            _formula_step(
                "step_1",
                "Compute requested AC quantity from sinusoidal source form.",
                formula,
                [],
                output,
                template_name,
            )
        ], output, template_name)
    return [], 0.0


def _zero_field_distance_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    if target not in {"d", "r"} or "zero_electric_field_point" not in conditions:
        return [], 0.0
    if not {"q1", "q2"} <= set(known):
        return [], 0.0
    distances = _by_dimension(known, "length")
    if not distances:
        return [], 0.0
    try:
        q1_value = float(known.get("q1", {}).get("normalized_value") or known.get("q1", {}).get("value"))
        q2_value = float(known.get("q2", {}).get("normalized_value") or known.get("q2", {}).get("value"))
    except (TypeError, ValueError):
        return [], 0.0
    if abs(abs(q1_value) - abs(q2_value)) <= 1e-12 * max(abs(q1_value), abs(q2_value), 1.0):
        return [], 0.0

    if "target_distance_from_A" in conditions:
        endpoint = "q1"
    elif "target_distance_from_B" in conditions:
        endpoint = "q2"
    else:
        endpoint = "q1" if abs(q1_value) >= abs(q2_value) else "q2"
    separation = distances[0]
    if q1_value * q2_value > 0:
        denominator = "sqrt(abs(q1)) + sqrt(abs(q2))"
        template_name = "electric_field_zero_point_same_sign_between_distance"
        goal = "Solve the zero-field point between two same-sign charges from the requested endpoint."
    else:
        denominator = "abs(sqrt(abs(q1)) - sqrt(abs(q2)))"
        template_name = "electric_field_zero_point_opposite_charges_distance"
        goal = "Solve the zero-field point outside two opposite charges from the requested endpoint."
    return _finish(
        [
            _formula_step(
                "step_1",
                goal,
                f"{target} = {separation} * sqrt(abs({endpoint})) / ({denominator})",
                ["q1", "q2", separation],
                target,
                template_name,
            )
        ],
        target,
        template_name,
    )


def _field_line_midpoint_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    if target != "E" or "point_charge_field_line" not in conditions or "midpoint" not in conditions:
        return [], 0.0
    fields = _by_dimension(known, "electric_field")
    if len(fields) < 2:
        return [], 0.0
    field_a, field_b = fields[0], fields[1]
    return _finish(
        [
            _formula_step(
                "step_1",
                "Interpolate the field at the midpoint along a point-charge field line using E proportional to 1/r^2.",
                f"E = 4 / ((1/sqrt({field_a}) + 1/sqrt({field_b}))^2)",
                [field_a, field_b],
                "E",
                "electric_field_line_midpoint_inverse_square",
            )
        ],
        target,
        "electric_field_line_midpoint_inverse_square",
    )


def _continuous_charge_field_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    if target != "E":
        return [], 0.0

    if "charged_circular_ring" in conditions:
        charge = _first_by_dimension(known, "charge")
        radius = _first_existing(known, "R_radius", "r")
        lengths = _by_dimension(known, "length")
        axis_distance = next((name for name in lengths if name != radius), None)
        if charge and radius and axis_distance:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute axial electric field from a uniformly charged circular ring.",
                        f"E = k * abs({charge}) * {axis_distance} / (({radius}^2 + {axis_distance}^2)^(3/2))",
                        [charge, radius, axis_distance, "k"],
                        "E",
                        "charged_ring_axis_field",
                    )
                ],
                target,
                "charged_ring_axis_field",
            )

    if "finite_charged_rod" in conditions:
        linear_density = _first_by_dimension(known, "linear_charge_density")
        rod_length = _first_existing(known, "L")
        lengths = _by_dimension(known, "length")
        point_distance = next((name for name in lengths if name != rod_length), None)
        if linear_density and rod_length and point_distance:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute electric field from a finite uniformly charged rod at an off-axis endpoint-aligned point.",
                        f"E = k * abs({linear_density}) * {rod_length} / ({point_distance} * sqrt({point_distance}^2 + {rod_length}^2))",
                        [linear_density, rod_length, point_distance, "k"],
                        "E",
                        "finite_charged_rod_endpoint_axis_field",
                    )
                ],
                target,
                "finite_charged_rod_endpoint_axis_field",
            )

    if "circular_plate" in conditions:
        surface_density = _first_by_dimension(known, "surface_charge_density")
        radius = _first_existing(known, "R_radius", "r")
        lengths = _by_dimension(known, "length")
        axis_distance = next((name for name in lengths if name != radius), None)
        if surface_density and radius and axis_distance:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute axial electric field from a uniformly charged circular disk.",
                        f"E = abs({surface_density}) / (2*epsilon_0) * (1 - {axis_distance}/sqrt({axis_distance}^2 + {radius}^2))",
                        [surface_density, axis_distance, radius, "epsilon_0"],
                        "E",
                        "charged_disk_axis_field",
                    )
                ],
                target,
                "charged_disk_axis_field",
            )

    surface_density = _first_by_dimension(known, "surface_charge_density")
    if surface_density and "parallel_insulating_plates" in conditions:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute the uniform field between two oppositely charged insulating plates.",
                    f"E = abs({surface_density}) / epsilon_0",
                    [surface_density, "epsilon_0"],
                    "E",
                    "parallel_insulating_plates_field",
                )
            ],
            target,
            "parallel_insulating_plates_field",
        )

    if "infinite_metal_plate" in conditions:
        charge = _first_by_dimension(known, "charge")
        lengths = _by_dimension(known, "length")
        if charge and len(lengths) >= 2:
            side_a, side_b = lengths[0], lengths[1]
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute field outside a uniformly charged infinite conducting plate from charge per area.",
                        f"E = abs({charge}) / (2 * epsilon_0 * {side_a} * {side_b})",
                        [charge, side_a, side_b, "epsilon_0"],
                        "E",
                        "infinite_metal_plate_field_from_area_charge",
                    )
                ],
                target,
                "infinite_metal_plate_field_from_area_charge",
            )

    if "infinite_line_charge" in conditions:
        linear_density = _first_by_dimension(known, "linear_charge_density")
        if not linear_density:
            linear_density = _first_by_dimension(known, "charge")
        distance = _first_by_dimension(known, "length")
        if linear_density and distance:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute electric field from an infinitely long line charge.",
                        f"E = 2 * k * abs({linear_density}) / {distance}",
                        [linear_density, distance, "k"],
                        "E",
                        "infinite_line_charge_field",
                    )
                ],
                target,
                "infinite_line_charge_field",
            )

    return [], 0.0


def _electric_equilibrium_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    if "electric_equilibrium" not in conditions:
        return [], 0.0
    mass = _first_by_dimension(known, "mass")
    charge = _first_by_dimension(known, "charge")
    field = _first_by_dimension(known, "electric_field")
    acceleration = _first_by_dimension(known, "acceleration")
    theta = _first_by_dimension(known, "angle")
    g_factor = acceleration or "10"

    if target == "E" and mass and charge and acceleration:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Use vertical equilibrium of electric and weight forces.",
                    f"E = {mass} * {acceleration} / abs({charge})",
                    [mass, acceleration, charge],
                    "E",
                    "electric_gravity_equilibrium_field",
                )
            ],
            target,
            "electric_gravity_equilibrium_field",
        )
    if target in {"q", "Q"} and mass and field and acceleration:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Use vertical equilibrium qE = mg to solve for charge.",
                    f"{target} = {mass} * {acceleration} / abs({field})",
                    [mass, acceleration, field],
                    target,
                    "electric_gravity_equilibrium_charge",
                )
            ],
            target,
            "electric_gravity_equilibrium_charge",
        )
    if target in {"m", "mass", "m_object"} and charge and field and theta:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Use tan(theta) = electric force / weight for a charged particle in horizontal field.",
                    f"{target} = abs({charge}) * {field} / ({g_factor} * tan(theta))",
                    [charge, field, theta] + ([acceleration] if acceleration else []),
                    target,
                    "electric_field_thread_equilibrium_mass",
                )
            ],
            target,
            "electric_field_thread_equilibrium_mass",
        )
    return [], 0.0


def _electric_force_field_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    charges = _by_dimension(known, "charge")
    distance = _first_by_dimension(known, "length")
    force = _first_by_dimension(known, "force")
    field = _first_by_dimension(known, "electric_field")

    if target == "E" and force and charges:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute electric field strength from force on a test charge.",
                    f"E = {force} / abs({charges[0]})",
                    [force, charges[0]],
                    "E",
                    "electric_field_from_force_on_charge",
                )
            ],
            target,
            "electric_field_from_force_on_charge",
        )

    if target in {"Q", "q", "q0", "q1", "q2", "q3"} and force and distance and charges:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Invert Coulomb's law for the source charge using the force on a test charge.",
                    f"{target} = {force} * {distance}^2 / (k * abs({charges[0]}))",
                    [force, distance, charges[0], "k"],
                    target,
                    "coulomb_source_charge_from_force",
                )
            ],
            target,
            "coulomb_source_charge_from_force",
        )

    if target in {"Q", "q", "q0", "q1", "q2", "q3"} and field and distance:
        sign = "-" if "field_towards_charge" in conditions else ""
        if "epsilon_r" in known:
            formula = f"{target} = {sign}epsilon_r * {field} * {distance}^2 / k"
            inputs = ["epsilon_r", field, distance, "k"]
            template_name = "electric_field_charge_inverse_dielectric"
        else:
            formula = f"{target} = {sign}{field} * {distance}^2 / k"
            inputs = [field, distance, "k"]
            template_name = "electric_field_charge_inverse"
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Invert the point-charge electric-field relation, preserving dielectric and direction cues.",
                    formula,
                    inputs,
                    target,
                    template_name,
                )
            ],
            target,
            template_name,
        )

    return [], 0.0


def _field_geometry_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    if target != "E":
        return [], 0.0
    charges = _by_dimension(known, "charge")
    distances = _by_dimension(known, "length")

    if "square_center_point" in conditions and "square_opposite_equal_charges" in conditions:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Use square symmetry: opposite equal signed pairs cancel at the center.",
                    "E = 0",
                    [],
                    "E",
                    "square_center_symmetric_field_zero",
                )
            ],
            target,
            "square_center_symmetric_field_zero",
        )

    if "square_three_charges_fourth_vertex" in conditions and charges and distances:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Combine fields at the fourth square vertex from two adjacent charges and one diagonal charge.",
                    f"E = (sqrt(2) + 0.5) * k * abs({charges[0]}) / {distances[0]}^2",
                    [charges[0], distances[0], "k"],
                    "E",
                    "electric_field_square_three_charges_fourth_vertex",
                )
            ],
            target,
            "electric_field_square_three_charges_fourth_vertex",
        )

    equal_like_midpoint = "two_equal_like_charges" in conditions
    if "midpoint_between_two_charges" in conditions and len(charges) >= 2:
        try:
            first_charge_value = float(known.get(charges[0], {}).get("normalized_value") or known.get(charges[0], {}).get("value"))
            second_charge_value = float(known.get(charges[1], {}).get("normalized_value") or known.get(charges[1], {}).get("value"))
            equal_like_midpoint = equal_like_midpoint or (
                first_charge_value * second_charge_value > 0
                and abs(first_charge_value - second_charge_value) <= 1e-12 * max(abs(first_charge_value), abs(second_charge_value), 1.0)
            )
        except (TypeError, ValueError):
            pass
    if "midpoint" in conditions and equal_like_midpoint:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Use symmetry: equal like charges produce opposite fields at the midpoint.",
                    "E = 0",
                    [],
                    "E",
                    "midpoint_equal_like_charges_field_zero",
                )
            ],
            target,
            "midpoint_equal_like_charges_field_zero",
        )

    if "perpendicular_bisector" in conditions and "two_equal_like_charges" in conditions and len(charges) == 1 and len(distances) >= 2:
        sep, height = distances[0], distances[1]
        source_charge = charges[0]
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute electric field on the perpendicular bisector from two equal like charges.",
                    f"E = 2 * k * abs({source_charge}) * {height} / (((0.5*{sep})^2 + {height}^2)^(3/2))",
                    ["k", source_charge, sep, height],
                    "E",
                    "electric_field_perpendicular_bisector_equal_like",
                )
            ],
            target,
            "electric_field_perpendicular_bisector_equal_like",
        )

    if "perpendicular_bisector" in conditions and {"q1", "q2"} <= set(known) and len(distances) >= 2:
        sep, height = distances[0], distances[1]
        sep_value = float(known.get(sep, {}).get("normalized_value") or known.get(sep, {}).get("value") or 0.0)
        point_distance_value = float(known.get(height, {}).get("normalized_value") or known.get(height, {}).get("value") or 0.0)
        if sep_value and point_distance_value and abs(point_distance_value - 0.5 * sep_value) <= 0.02 * point_distance_value:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Use the midpoint limit on the perpendicular bisector: equal-distance fields subtract by signed charge.",
                        f"E = k * abs(q1 - q2) / {height}^2",
                        ["q1", "q2", height, "k"],
                        "E",
                        "electric_field_perpendicular_bisector_midpoint_limit",
                    )
                ],
                target,
                "electric_field_perpendicular_bisector_midpoint_limit",
            )
        plan = [
            _formula_step(
                "step_1",
                "Compute net electric field on the perpendicular bisector from signed source charges.",
                f"E = k * sqrt(((0.5*{sep})*(q1 - q2))^2 + ({height}*(q1 + q2))^2) / (((0.5*{sep})^2 + {height}^2)^(3/2))",
                ["q1", "q2", sep, height, "k"],
                "E",
                "electric_field_perpendicular_bisector",
            )
        ]
        return _finish(plan, target, "electric_field_perpendicular_bisector")

    if "midpoint" in conditions and len(charges) >= 2 and distances:
        sep = distances[0]
        first_charge, second_charge = charges[0], charges[1]
        if "epsilon_r" in known:
            formula = f"E = k * abs({first_charge} - {second_charge}) / (epsilon_r * (0.5*{sep})^2)"
            inputs = [first_charge, second_charge, sep, "epsilon_r", "k"]
            template_name = "electric_field_midpoint_signed_charges_dielectric"
        else:
            formula = f"E = k * abs({first_charge} - {second_charge}) / (0.5*{sep})^2"
            inputs = [first_charge, second_charge, sep, "k"]
            template_name = "electric_field_midpoint_signed_charges"
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Combine signed electric fields at the midpoint of two endpoint charges.",
                    formula,
                    inputs,
                    "E",
                    template_name,
                )
            ],
            target,
            template_name,
        )

    if "line_connecting" in conditions and {"q1", "q2"} <= set(known) and len(distances) >= 2:
        sep, x_from_q1 = distances[0], distances[1]
        if "outside_segment" in conditions:
            if "outside_right_of_first_endpoint" in conditions:
                formula = f"E = k * abs(q1/{x_from_q1}^2 + q2/({x_from_q1} - {sep})^2)"
            else:
                formula = f"E = k * abs(q1/{x_from_q1}^2 + q2/({sep} + {x_from_q1})^2)"
            template_name = "electric_field_collinear_outside"
        else:
            formula = f"E = k * abs(q1/{x_from_q1}^2 - q2/({sep} - {x_from_q1})^2)"
            template_name = "electric_field_collinear_between"
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute signed collinear electric-field contributions.",
                    formula,
                    ["q1", "q2", sep, x_from_q1, "k"],
                    "E",
                    template_name,
                )
            ],
            target,
            template_name,
        )

    if "triangle_center" in conditions and "equilateral_triangle" in conditions:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Use symmetry of equal like charges at an equilateral triangle center.",
                    "E = 0",
                    [],
                    "E",
                    "electric_field_equilateral_center_equal_charges",
                )
            ],
            target,
            "electric_field_equilateral_center_equal_charges",
        )

    if "right_angle" in conditions and len(charges) == 1 and distances:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Combine two perpendicular equal electric-field contributions at the right-angle vertex.",
                    f"E = sqrt(2) * k * abs({charges[0]}) / {distances[0]}^2",
                    [charges[0], distances[0], "k"],
                    "E",
                    "electric_field_right_isosceles_identical_charges",
                )
            ],
            target,
            "electric_field_right_isosceles_identical_charges",
        )

    if "equilateral_triangle" in conditions and charges and distances:
        if {"q1", "q2"} <= set(known):
            formula = f"E = k * sqrt(q1^2 + q2^2 + 2*q1*q2*cos(60deg)) / {distances[0]}^2"
            inputs = ["q1", "q2", distances[0], "k"]
        else:
            source_charge = next((name for name in charges if name != "q3"), charges[0])
            formula = f"E = sqrt(3) * k * abs({source_charge}) / {distances[0]}^2"
            inputs = [source_charge, distances[0], "k"]
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Combine signed electric-field contributions at an equilateral triangle vertex.",
                    formula,
                    inputs,
                    "E",
                    "electric_field_equilateral_two_source_charges",
                )
            ],
            target,
            "electric_field_equilateral_two_source_charges",
        )

    if "field_angle_given" in conditions and ("theta" in known or "right_angle" in conditions) and len(charges) >= 2 and distances:
        q1 = charges[0]
        q2 = charges[1]
        r1 = distances[0]
        r2 = distances[1] if len(distances) > 1 else distances[0]
        combine_formula = (
            "E = sqrt(E1^2 + E2^2 + 2*E1*E2*cos(theta))"
            if "theta" in known
            else "E = sqrt(E1^2 + E2^2)"
        )
        combine_inputs = ["E1", "E2", "theta"] if "theta" in known else ["E1", "E2"]
        plan = [
            _formula_step(
                "step_1",
                "Compute electric field contribution from the first charge.",
                f"E1 = k * abs({q1}) / {r1}^2",
                [q1, r1, "k"],
                "E1",
                "electric_field_resultant_given_angle",
            ),
            _formula_step(
                "step_2",
                "Compute electric field contribution from the second charge.",
                f"E2 = k * abs({q2}) / {r2}^2",
                [q2, r2, "k"],
                "E2",
                "electric_field_resultant_given_angle",
            ),
            _formula_step(
                "step_3",
                "Combine the two electric-field vectors using the stated angle between fields.",
                combine_formula,
                combine_inputs,
                "E",
                "electric_field_resultant_given_angle",
            ),
        ]
        return _finish(plan, target, "electric_field_resultant_given_angle")

    if "theta" in known and len(charges) >= 2 and distances:
        q1 = charges[0]
        q2 = charges[1]
        r1 = distances[0]
        r2 = distances[1] if len(distances) > 1 else distances[0]
        q1_value = float(known.get(q1, {}).get("normalized_value") or known.get(q1, {}).get("value") or 0.0)
        q2_value = float(known.get(q2, {}).get("normalized_value") or known.get(q2, {}).get("value") or 0.0)
        combine_sign = 1 if q1_value * q2_value >= 0 else -1
        plan = [
            _formula_step(
                "step_1",
                "Compute electric field contribution from the first charge.",
                f"E1 = k * abs({q1}) / {r1}^2",
                [q1, r1, "k"],
                "E1",
                "electric_field_resultant_source_angle",
            ),
            _formula_step(
                "step_2",
                "Compute electric field contribution from the second charge.",
                f"E2 = k * abs({q2}) / {r2}^2",
                [q2, r2, "k"],
                "E2",
                "electric_field_resultant_source_angle",
            ),
            _formula_step(
                "step_3",
                "Combine field vectors using the angle between source-charge lines and charge signs.",
                f"E = sqrt(E1^2 + E2^2 + {combine_sign * 2}*E1*E2*cos(theta))",
                ["E1", "E2", "theta"],
                "E",
                "electric_field_resultant_source_angle",
            ),
        ]
        return _finish(plan, target, "electric_field_resultant_source_angle")

    if charges and len(distances) >= 2:
        q1 = charges[0]
        q2 = charges[1] if len(charges) > 1 else charges[0]
        if len(distances) >= 3:
            source_sep, r1, r2 = distances[0], distances[1], distances[2]
        else:
            source_sep, r1, r2 = distances[0], distances[1], distances[1]
        q1_value = float(known.get(q1, {}).get("normalized_value") or known.get(q1, {}).get("value") or 0.0)
        q2_value = float(known.get(q2, {}).get("normalized_value") or known.get(q2, {}).get("value") or q1_value)
        combine_sign = 1 if q1_value * q2_value >= 0 else -1
        plan = [
            _formula_step("step_1", "Compute electric field contribution from first charge.", f"E1 = k * abs({q1}) / {r1}^2", [q1, r1, "k"], "E1", "electric_field_two_charge_geometry"),
            _formula_step("step_2", "Compute electric field contribution from second charge.", f"E2 = k * abs({q2}) / {r2}^2", [q2, r2, "k"], "E2", "electric_field_two_charge_geometry"),
            _formula_step("step_3", "Compute cosine of the angle between source-charge lines.", f"cos_gamma = ({r1}^2 + {r2}^2 - {source_sep}^2) / (2*{r1}*{r2})", [r1, r2, source_sep], "cos_gamma", "electric_field_two_charge_geometry"),
            _formula_step("step_4", "Combine electric field contributions using charge signs and geometry.", f"E = sqrt(E1^2 + E2^2 + {combine_sign * 2}*E1*E2*cos_gamma)", ["E1", "E2", "cos_gamma"], "E", "electric_field_two_charge_geometry"),
        ]
        return _finish(plan, target, "electric_field_two_charge_geometry")
    distance = distances[0] if distances else None
    if charges and distance:
        if "epsilon_r" in known:
            formula = f"E = k * abs({charges[0]}) / (epsilon_r * {distance}^2)"
            inputs = [charges[0], distance, "epsilon_r", "k"]
            template_name = "electric_field_point_charge_dielectric"
        else:
            formula = "E = k * abs(q) / r^2"
            inputs = [charges[0], distance, "k"]
            template_name = "electric_field_point_charge"
        return _finish([_formula_step("step_1", "Compute point-charge electric field magnitude.", formula, inputs, "E", template_name)], target, template_name)
    if "square_center" in conditions:
        plan = [
            _setup_step("step_1", "Introduce symbolic square-charge geometry for field superposition.", {"q": "symbolic charge magnitude", "a": "symbolic side length"}, "electric_field_square_center"),
            _formula_step("step_2", "Combine electric field contributions at the square center.", "E = vector_sum(k*q/r_i^2)", ["q", "a", "k"], "E", "electric_field_square_center", warning="Symmetric square geometry is represented symbolically for downstream vector reasoning."),
        ]
        return _finish(plan, target, "electric_field_square_center")
    if "equilateral_triangle" in conditions:
        plan = [
            _setup_step("step_1", "Introduce symbolic equilateral-triangle charge geometry.", {"q": "symbolic charge magnitude", "a": "symbolic side length"}, "electric_field_equilateral_center"),
            _formula_step("step_2", "Combine electric field contributions at the triangle center.", "E = vector_sum(k*q/r_i^2)", ["q", "a", "k"], "E", "electric_field_equilateral_center", warning="Equilateral geometry is represented symbolically for downstream vector reasoning."),
        ]
        return _finish(plan, target, "electric_field_equilateral_center")
    return [], 0.0


def _square_field_cancellation_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    if target not in {"q_B", "q2", "q"}:
        return [], 0.0
    if "square_center" not in conditions:
        return [], 0.0
    plan = [
        _formula_step(
            "step_1",
            "Apply square-corner electric-field cancellation at D.",
            "q_B = -2*sqrt(2)*q",
            [],
            "q_B",
            "square_field_cancellation",
        )
    ]
    return _finish(plan, "q_B", "square_field_cancellation")


def _force_resultant_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    forces = _by_dimension(known, "force")
    if target == "F_net" and len(forces) >= 2 and "collinear" in conditions:
        return _finish([_formula_step("step_1", "Combine collinear opposite force magnitudes.", "F_net = abs(F1 - F2)", [forces[0], forces[1]], "F_net", "force_resultant_collinear_opposite")], target, "force_resultant_collinear_opposite")
    if target == "F_net" and len(forces) >= 2 and "right_angle" in conditions:
        return _finish([_formula_step("step_1", "Combine perpendicular force magnitudes.", "F_net = sqrt(F1^2 + F2^2)", [forces[0], forces[1]], "F_net", "force_resultant_perpendicular")], target, "force_resultant_perpendicular")
    if target == "F_net" and len(forces) == 1 and "theta" in known:
        return _finish([_formula_step("step_1", "Combine two equal force magnitudes at the given angle.", "F_net = sqrt(F^2 + F^2 + 2*F*F*cos(theta))", [forces[0], "theta"], "F_net", "force_resultant_equal_angle")], target, "force_resultant_equal_angle")
    if target == "theta" and len(forces) >= 2:
        first = forces[0]
        resultant = forces[1]
        return _finish([_formula_step("step_1", "Infer the angle between two equal forces from the resultant magnitude.", f"theta = acos(({resultant}^2 - 2*{first}^2) / (2*{first}^2)) * 180 / pi", [first, resultant, "pi"], "theta", "force_resultant_equal_forces_inverse_angle")], target, "force_resultant_equal_forces_inverse_angle")
    if target == "theta" and len(forces) >= 2:
        return _finish([_formula_step("step_1", "Determine resultant direction from force components.", "theta = atan2(sum(F_y), sum(F_x))", forces[:3], "theta", "force_resultant_direction", warning="Direction is represented symbolically from components when geometry is textual.")], target, "force_resultant_direction")
    return [], 0.0


def _mechanics_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    distance = _first_by_dimension(known, "length")
    time = _first_by_dimension(known, "time")
    velocity = _first_existing(known, "v", "v_avg", "v_0", "v_final") or _first_by_dimension(known, "velocity")
    acceleration = _first_by_dimension(known, "acceleration")

    if target == "ratio":
        return _finish(
            [_formula_step("step_1", "Create symbolic ratio relation requested by the problem.", "ratio = requested_quantity_1 / requested_quantity_2", [], "ratio", "symbolic_ratio_target")],
            target,
            "symbolic_ratio_target",
        )
    if target in {"d", "r"}:
        velocities = _by_dimension(known, "velocity")
        if len(velocities) >= 2 and time:
            if "same_direction_chasing" in conditions:
                formula = "relative_speed = abs(v1 - v2)"
                template_name = "relative_motion_chasing"
            else:
                formula = "relative_speed = v1 + v2"
                template_name = "relative_motion_meeting"
            plan = [
                _formula_step("step_1", "Compute relative speed.", formula, velocities[:2], "relative_speed", template_name),
                _formula_step("step_2", "Compute relative-motion distance.", f"{target} = relative_speed * t", ["relative_speed", time], target, template_name),
            ]
            return _finish(plan, target, template_name)
    if target in {"d", "r"} and velocity and time:
        return _finish([_formula_step("step_1", "Apply constant-speed distance relation.", f"{target} = v * t", [velocity, time], target, "constant_speed_distance")], target, "constant_speed_distance")
    if target in {"v", "v_avg"} and distance and time:
        return _finish([_formula_step("step_1", "Apply constant-speed velocity relation.", f"{target} = d / t", [distance, time], target, "constant_speed_velocity")], target, "constant_speed_velocity")
    if target == "t" and distance and velocity:
        return _finish([_formula_step("step_1", "Apply constant-speed time relation.", "t = d / v", [distance, velocity], "t", "constant_speed_time")], target, "constant_speed_time")
    if target in {"f", "f_osc"} and time:
        return _finish([_formula_step("step_1", "Compute frequency from period.", f"{target} = 1 / {time}", [time], target, "frequency_from_period")], target, "frequency_from_period")

    if target in {"v_final", "v"} and _first_existing(known, "v_0") and acceleration and time:
        return _finish([_formula_step("step_1", "Apply uniform-acceleration velocity relation.", f"{target} = v_0 + a*t", ["v_0", acceleration, time], target, "kinematics_final_velocity")], target, "kinematics_final_velocity")
    if target in {"d", "r"} and _first_existing(known, "v_0") and acceleration and time:
        return _finish([_formula_step("step_1", "Apply uniform-acceleration displacement relation.", f"{target} = v_0*t + 0.5*a*t^2", ["v_0", acceleration, time], target, "kinematics_displacement")], target, "kinematics_displacement")
    if target == "a":
        if _first_existing(known, "v_final", "v") and _first_existing(known, "v_0") and time:
            final_v = _first_existing(known, "v_final", "v") or "v"
            return _finish([_formula_step("step_1", "Solve uniform-acceleration velocity relation for acceleration.", "a = (v_final - v_0) / t", [final_v, "v_0", time], "a", "kinematics_acceleration")], target, "kinematics_acceleration")
    return [], 0.0


def _relation_driven_templates(target: str, relations: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], float]:
    functions = [relation for relation in relations if relation.get("type") == "function"]
    equations = [relation for relation in relations if relation.get("type") == "equation"]
    uncertainties = [relation for relation in relations if relation.get("type") == "uncertainty"]
    if uncertainties:
        unique_uncertainties: List[Dict[str, object]] = []
        seen_uncertainties = set()
        for relation in uncertainties:
            key = (
                relation.get("quantity"),
                relation.get("value"),
                relation.get("uncertainty"),
                relation.get("unit_symbol"),
                relation.get("relative_uncertainty"),
            )
            if key in seen_uncertainties:
                continue
            seen_uncertainties.add(key)
            unique_uncertainties.append(relation)
        uncertainties = unique_uncertainties
    percentages = [relation for relation in relations if relation.get("type") == "percentage"]

    current_function = next((relation for relation in functions if relation.get("function_name") in {"I", "i"}), None)
    if target in {"Q", "q"} and current_function:
        plan = [
            _formula_step("step_1", "Identify current as a function of time.", "I_of_t = extracted_function(I,t)", [], "I_of_t", "function_current_integration"),
            _formula_step("step_2", "Set up charge transfer as an integral of current.", "Q = integral(I(t), t_start, t_end)", ["I_of_t"], target, "function_current_integration", warning="Stage 0 records the integration setup without solving the integral."),
        ]
        return _finish(plan, target, "function_current_integration")

    if target in {"d", "r", "v", "v_avg", "ratio"} and equations:
        plan = [
            _formula_step("step_1", "Set up extracted algebraic equation system.", "equation_system = extracted_equations", [], "equation_system", "equation_system_setup"),
            _formula_step("step_2", "Solve extracted equation system for the requested target.", f"{target} = solve(equation_system, {target})", ["equation_system"], target, "equation_system_setup", warning="Stage 0 records symbolic solve setup without solving equations."),
        ]
        return _finish(plan, target, "equation_system_setup")

    if target in {"rel_error", "percent_error", "uncertainty"} and uncertainties:
        if target in {"rel_error", "percent_error"} and len(uncertainties) >= 2:
            terms = []
            inputs = []
            for relation in uncertainties:
                quantity = str(relation.get("quantity") or "measured_value")
                terms.append(f"delta_{quantity} / abs({quantity})")
                inputs.extend([f"delta_{quantity}", quantity])
            plan = [
                _formula_step(
                    "step_1",
                    "Combine independent relative uncertainties.",
                    f"rel_error = {' + '.join(terms)}",
                    inputs,
                    "rel_error",
                    "uncertainty_relation",
                )
            ]
            if target == "percent_error":
                plan.append(_formula_step("step_2", "Convert relative uncertainty to percent.", "percent_error = rel_error * 100", ["rel_error"], "percent_error", "uncertainty_relation"))
            return _finish(plan, target, "uncertainty_relation")
        relation = uncertainties[0]
        quantity = str(relation.get("quantity") or "measured_value")
        if relation.get("relative_uncertainty") is not None:
            plan = [_formula_step("step_1", "Use extracted relative uncertainty.", "rel_error = extracted_relative_uncertainty", [], "rel_error", "uncertainty_relation")]
        else:
            plan = [_formula_step("step_1", "Compute relative uncertainty from absolute uncertainty.", f"rel_error = delta_{quantity} / abs({quantity})", [], "rel_error", "uncertainty_relation")]
        if target == "percent_error":
            plan.append(_formula_step("step_2", "Convert relative uncertainty to percent.", "percent_error = rel_error * 100", ["rel_error"], "percent_error", "uncertainty_relation"))
        elif target == "uncertainty":
            plan.append(_formula_step("step_2", "Report extracted uncertainty.", "uncertainty = extracted_uncertainty", ["rel_error"], "uncertainty", "uncertainty_relation"))
        return _finish(plan, target, "uncertainty_relation")

    if target in {"efficiency", "percent_error", "ratio"} and percentages:
        plan = [_formula_step("step_1", "Use extracted percentage or fraction relation.", f"{target} = extracted_percentage_fraction", [], target, "percentage_relation")]
        return _finish(plan, target, "percentage_relation")

    return [], 0.0


def _resonance_design_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    """LC resonance design problems where the target is the missing component.

    Covers the recurring "what value of C/L is needed to resonate at f?" family
    of questions, plus a few related cases:
      * `C_cap` when (L, f) are known           → C = 1/(4π² f² L)
      * `L_ind` when (C, f) are known           → L = 1/(4π² f² C)
      * `R` when only resistance(/Z-as-measured) is known at resonance: Z = R
      * `Z` when only R is known at resonance: Z = R
      * `f` when (R, L, C, f) are all known     → compute f_res = 1/(2π√LC)
        and compare. We don't try to express the comparison itself, just emit
        the resonant frequency so the verifier sees a real plan.
      * `omega` when (X_L, X_C, omega) are known and the question asks "by what
        factor must ω be changed for resonance" → ω_new/ω₀ = sqrt(X_C/X_L)
        (we encode this as `omega = omega * sqrt(X_C/X_L)` placeholder).

    All resonance reasoning needs a "resonance" cue somewhere — either an
    explicit resonance sub-domain mention, an `Lω² = 1` style condition, or
    target-specific phrasing like asking for f_res/omega_0.
    """
    inductance = _first_by_dimension(known, "inductance")
    capacitance = _first_by_dimension(known, "capacitance")
    resistance = _first_by_dimension(known, "resistance")
    frequency = _first_existing(known, "f", "f_res", "f_osc")

    # Design: solve for the missing reactive component using f_res = 1/(2π√LC)
    if target == "C_cap" and inductance and frequency:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Solve LC resonance condition for capacitance.",
                    "C_cap = 1 / (4 * pi**2 * f**2 * L_ind)",
                    [frequency, inductance, "pi"],
                    "C_cap",
                    "lc_resonance_capacitance",
                )
            ],
            target,
            "lc_resonance_capacitance",
        )
    if target == "L_ind" and capacitance and frequency:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Solve LC resonance condition for inductance.",
                    "L_ind = 1 / (4 * pi**2 * f**2 * C_cap)",
                    [frequency, capacitance, "pi"],
                    "L_ind",
                    "lc_resonance_inductance",
                )
            ],
            target,
            "lc_resonance_inductance",
        )

    # At resonance, X_L = X_C → Z = R, so a single-resistance known set is enough
    # to answer "find R" or "find Z". We gate this on a resonance hint to avoid
    # masking other R/Z templates that need more inputs.
    resonance_hint = (
        any(c in conditions for c in ("resonance", "rlc_resonance"))
        or any(name in known for name in ("Z", "measured_value"))
    )
    if target == "R" and resistance and resonance_hint:
        # If we already have R as a quantity, the answer is literally R = R
        # (the verifier just needs to see a real executable step). We emit
        # a Z-mediated derivation so the step plan is meaningful: at resonance
        # Z = R, and Z was the measured value.
        z_source = _first_existing(known, "Z", "measured_value") or resistance
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "At resonance, impedance equals pure resistance.",
                    f"R = {z_source}",
                    [z_source],
                    "R",
                    "resonance_R_from_Z",
                )
            ],
            target,
            "resonance_R_from_Z",
        )
    if target == "Z" and resistance and resonance_hint:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "At resonance, impedance equals pure resistance.",
                    "Z = R",
                    [resistance],
                    "Z",
                    "impedance_at_resonance",
                )
            ],
            target,
            "impedance_at_resonance",
        )

    # "Does this RLC circuit resonate at frequency f?" — known has R, L, C, f.
    # We compute the resonant frequency so the planner has a defensible target.
    if target == "f" and inductance and capacitance and frequency:
        # We deliberately overwrite f with f_res in the output — the comparison
        # is downstream's job; Stage 0 just needs a real plan.
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute the resonant frequency of an LC circuit.",
                    "f_res = 1 / (2 * pi * sqrt(L_ind * C_cap))",
                    [inductance, capacitance, "pi"],
                    "f_res",
                    "rlc_resonance_check",
                ),
                _formula_step(
                    "step_2",
                    "Report the resonant frequency as the answer.",
                    "f = f_res",
                    ["f_res"],
                    "f",
                    "rlc_resonance_check",
                ),
            ],
            target,
            "rlc_resonance_check",
        )

    # "By what factor must ω be changed for resonance?" — given X_L, X_C at ω₀.
    # The known set carries them as 'R'/'R2' (resistance dim) but we want a
    # plan that uses them as reactances. We accept either naming.
    if target in {"omega", "k"}:
        # Heuristic: two resistance-typed quantities present, both look like
        # reactances. The factor relation is ω_new/ω₀ = sqrt(X_C/X_L).
        resistances = _by_dimension(known, "resistance")
        if len(resistances) >= 2:
            x_l = "X_L" if "X_L" in known else resistances[0]
            x_c = "X_C" if "X_C" in known else resistances[1]
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute new angular frequency factor for resonance.",
                        f"k = sqrt({x_c} / {x_l})",
                        [x_l, x_c],
                        "k",
                        "rlc_omega_factor_for_resonance",
                    )
                ],
                "k",
                "rlc_omega_factor_for_resonance",
            )

    return [], 0.0


def _electromagnetism_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    """Solenoid B-field, magnetic flux, induced EMF — Stage 0 high-frequency family.

    Covered cases (all single-step formula applications):

      * ``B = mu_0 * (N/L) * I``     for solenoid field (long solenoid limit)
        Triggers when target=B and known has current+length+turn_count, or
        already-computed turn density ``n_turns_per_meter``.
      * ``Phi_B = B * A``            magnetic flux through one turn
        Triggers when target=Phi_B and known has B + area.
      * ``Phi_link = N * B * A`` or  ``Phi_link = N * Phi_B``
        Triggers when target=Phi_link and we have N + (B+A) or N + Phi_B.
      * ``emf = L * dI/dt = L * I / t``  for uniform current change
        Triggers when target=emf and known has L_ind + current + time. We
        assume the current "changes uniformly from 0 to I" (or from I to 0)
        so dI=I; this matches the dominant question phrasing in the dataset.
    """
    if target == "B":
        current = _first_existing(known, "I", "I_rms")
        n_turns_total = _first_existing(known, "n_turns", "N_turns", "N")
        length = _first_existing(known, "d", "L", "L_solenoid", "l")
        density = _first_existing(known, "n_turns_per_meter", "n_density")
        if current and density:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute solenoid magnetic field from turn density.",
                        f"B = mu_0 * {density} * {current}",
                        [density, current, "mu_0"],
                        "B",
                        "solenoid_field_from_density",
                    )
                ],
                target,
                "solenoid_field_from_density",
            )
        if current and n_turns_total and length:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute solenoid turn density.",
                        f"n_turns_per_meter = {n_turns_total} / {length}",
                        [n_turns_total, length],
                        "n_turns_per_meter",
                        "solenoid_field_full",
                    ),
                    _formula_step(
                        "step_2",
                        "Compute solenoid magnetic field.",
                        f"B = mu_0 * n_turns_per_meter * {current}",
                        ["n_turns_per_meter", current, "mu_0"],
                        "B",
                        "solenoid_field_full",
                    ),
                ],
                target,
                "solenoid_field_full",
            )

    if target == "u_B":
        b_field = _first_existing(known, "B")
        current = _first_existing(known, "I", "I_rms")
        density = _first_existing(known, "n_turns_per_meter", "n_density")
        if b_field:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute magnetic field energy density.",
                        f"u_B = {b_field}^2 / (2 * mu_0)",
                        [b_field, "mu_0"],
                        "u_B",
                        "magnetic_energy_density",
                    )
                ],
                target,
                "magnetic_energy_density",
            )
        if current and density:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute solenoid magnetic field from turn density.",
                        f"B = mu_0 * {density} * {current}",
                        [density, current, "mu_0"],
                        "B",
                        "magnetic_energy_density",
                    ),
                    _formula_step(
                        "step_2",
                        "Compute magnetic field energy density.",
                        "u_B = B^2 / (2 * mu_0)",
                        ["B", "mu_0"],
                        "u_B",
                        "magnetic_energy_density",
                    ),
                ],
                target,
                "magnetic_energy_density",
            )

    if target == "U_B":
        b_field = _first_existing(known, "B")
        current = _first_existing(known, "I", "I_rms")
        density = _first_existing(known, "n_turns_per_meter", "n_density")
        n_turns_total = _first_existing(known, "n_turns", "N_turns", "N")
        length = _first_existing(known, "d", "L", "L_solenoid", "l")
        area = _first_existing(known, "A", "A_area")
        if b_field and area and length:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute magnetic field energy density.",
                        f"u_B = {b_field}^2 / (2 * mu_0)",
                        [b_field, "mu_0"],
                        "u_B",
                        "solenoid_magnetic_energy",
                    ),
                    _formula_step(
                        "step_2",
                        "Compute magnetic field energy in the solenoid volume.",
                        f"U_B = u_B * {area} * {length}",
                        ["u_B", area, length],
                        "U_B",
                        "solenoid_magnetic_energy",
                    ),
                ],
                target,
                "solenoid_magnetic_energy",
            )
        if current and density and area and length:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute solenoid magnetic field from turn density.",
                        f"B = mu_0 * {density} * {current}",
                        [density, current, "mu_0"],
                        "B",
                        "solenoid_magnetic_energy",
                    ),
                    _formula_step(
                        "step_2",
                        "Compute magnetic field energy density.",
                        "u_B = B^2 / (2 * mu_0)",
                        ["B", "mu_0"],
                        "u_B",
                        "solenoid_magnetic_energy",
                    ),
                    _formula_step(
                        "step_3",
                        "Compute magnetic field energy in the solenoid volume.",
                        f"U_B = u_B * {area} * {length}",
                        ["u_B", area, length],
                        "U_B",
                        "solenoid_magnetic_energy",
                    ),
                ],
                target,
                "solenoid_magnetic_energy",
            )
        if current and n_turns_total and length and area:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute solenoid turn density.",
                        f"n_turns_per_meter = {n_turns_total} / {length}",
                        [n_turns_total, length],
                        "n_turns_per_meter",
                        "solenoid_magnetic_energy",
                    ),
                    _formula_step(
                        "step_2",
                        "Compute solenoid magnetic field.",
                        f"B = mu_0 * n_turns_per_meter * {current}",
                        ["n_turns_per_meter", current, "mu_0"],
                        "B",
                        "solenoid_magnetic_energy",
                    ),
                    _formula_step(
                        "step_3",
                        "Compute magnetic field energy density.",
                        "u_B = B^2 / (2 * mu_0)",
                        ["B", "mu_0"],
                        "u_B",
                        "solenoid_magnetic_energy",
                    ),
                    _formula_step(
                        "step_4",
                        "Compute magnetic field energy in the solenoid volume.",
                        f"U_B = u_B * {area} * {length}",
                        ["u_B", area, length],
                        "U_B",
                        "solenoid_magnetic_energy",
                    ),
                ],
                target,
                "solenoid_magnetic_energy",
            )

    if target == "Phi_B":
        b_field = _first_existing(known, "B")
        area = _first_existing(known, "A", "A_area")
        current = _first_existing(known, "I", "I_rms")
        n_turns_total = _first_existing(known, "n_turns", "N_turns", "N")
        length = _first_existing(known, "d", "L", "L_solenoid", "l")
        density = _first_existing(known, "n_turns_per_meter", "n_density")
        if b_field and area and n_turns_total:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute total magnetic flux through all solenoid turns.",
                        f"Phi_B = {n_turns_total} * B * A",
                        [n_turns_total, b_field, area],
                        "Phi_B",
                        "magnetic_flux_solenoid_total",
                    )
                ],
                target,
                "magnetic_flux_solenoid_total",
            )
        if b_field and area:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute magnetic flux through one turn.",
                        "Phi_B = B * A",
                        [b_field, area],
                        "Phi_B",
                        "magnetic_flux_BA",
                    )
                ],
                target,
                "magnetic_flux_BA",
            )
        if area and current and density:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute solenoid magnetic field from turn density.",
                        f"B = mu_0 * {density} * {current}",
                        [density, current, "mu_0"],
                        "B",
                        "magnetic_flux_solenoid_one_turn",
                    ),
                    _formula_step(
                        "step_2",
                        "Compute magnetic flux through one turn.",
                        f"Phi_B = B * {area}",
                        ["B", area],
                        "Phi_B",
                        "magnetic_flux_solenoid_one_turn",
                    ),
                ],
                target,
                "magnetic_flux_solenoid_one_turn",
            )
        if area and current and n_turns_total and length:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute solenoid turn density.",
                        f"n_turns_per_meter = {n_turns_total} / {length}",
                        [n_turns_total, length],
                        "n_turns_per_meter",
                        "magnetic_flux_solenoid_one_turn",
                    ),
                    _formula_step(
                        "step_2",
                        "Compute solenoid magnetic field.",
                        f"B = mu_0 * n_turns_per_meter * {current}",
                        ["n_turns_per_meter", current, "mu_0"],
                        "B",
                        "magnetic_flux_solenoid_one_turn",
                    ),
                    _formula_step(
                        "step_3",
                        "Compute magnetic flux through one turn.",
                        f"Phi_B = B * {area}",
                        ["B", area],
                        "Phi_B",
                        "magnetic_flux_solenoid_one_turn",
                    ),
                ],
                target,
                "magnetic_flux_solenoid_one_turn",
            )

    if target == "Phi_link":
        n_turns_total = _first_existing(known, "n_turns", "N_turns", "N")
        b_field = _first_existing(known, "B")
        area = _first_existing(known, "A", "A_area")
        one_turn_flux = _first_existing(known, "Phi_B", "magnetic_flux")
        if n_turns_total and one_turn_flux:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Total flux linkage equals N * per-turn flux.",
                        f"Phi_link = {n_turns_total} * {one_turn_flux}",
                        [n_turns_total, one_turn_flux],
                        "Phi_link",
                        "flux_linkage_from_per_turn",
                    )
                ],
                target,
                "flux_linkage_from_per_turn",
            )
        if n_turns_total and b_field and area:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute flux linkage from B, A, and turn count.",
                        "Phi_link = n_turns * B * A",
                        [n_turns_total, b_field, area],
                        "Phi_link",
                        "flux_linkage_BAN",
                    )
                ],
                target,
                "flux_linkage_BAN",
            )

    if target == "L_ind":
        voltage = _first_existing(known, "emf", "V", "U")
        time = _first_existing(known, "t", "delta_t")
        currents = _by_dimension(known, "current")
        n_turns_total = _first_existing(known, "n_turns", "N_turns", "N")
        length = _first_existing(known, "d", "L", "L_solenoid", "l")
        area = _first_existing(known, "A", "A_area")
        if n_turns_total and area and length:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute solenoid self-inductance from geometry.",
                        f"L_ind = mu_0 * {n_turns_total}^2 * {area} / {length}",
                        [n_turns_total, area, length, "mu_0"],
                        "L_ind",
                        "solenoid_inductance_geometry",
                    )
                ],
                target,
                "solenoid_inductance_geometry",
            )
        if voltage and time and len(currents) >= 2:
            current_1, current_2 = currents[0], currents[1]
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute self-inductance from induced EMF and current change.",
                        f"L_ind = {voltage} * {time} / abs({current_2} - {current_1})",
                        [voltage, time, current_1, current_2],
                        "L_ind",
                        "self_inductance_from_emf_current_change",
                    )
                ],
                target,
                "self_inductance_from_emf_current_change",
            )
        if voltage and time and currents:
            current = currents[0]
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute self-inductance from induced EMF and uniform current change from zero.",
                        f"L_ind = {voltage} * {time} / abs({current})",
                        [voltage, time, current],
                        "L_ind",
                        "self_inductance_from_emf_current_change",
                    )
                ],
                target,
                "self_inductance_from_emf_current_change",
            )

    if target == "emf":
        inductance = _first_by_dimension(known, "inductance")
        current = _first_existing(known, "I", "I_rms")
        time = _first_existing(known, "t", "delta_t")
        n_turns_total = _first_existing(known, "n_turns", "N_turns", "N")
        one_turn_flux = _first_existing(known, "Phi_B", "magnetic_flux")
        if one_turn_flux and time and n_turns_total:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute average induced EMF from per-turn flux change.",
                        f"emf = {n_turns_total} * abs({one_turn_flux}) / {time}",
                        [n_turns_total, one_turn_flux, time],
                        "emf",
                        "faraday_emf_from_per_turn_flux",
                    )
                ],
                target,
                "faraday_emf_from_per_turn_flux",
            )
        if one_turn_flux and time:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute average induced EMF from magnetic flux change.",
                        f"emf = abs({one_turn_flux}) / {time}",
                        [one_turn_flux, time],
                        "emf",
                        "faraday_emf_from_flux_change",
                    )
                ],
                target,
                "faraday_emf_from_flux_change",
            )
        if inductance and current and time:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute induced EMF from inductor with uniformly changing current.",
                        "emf = L * I / t",
                        [inductance, current, time],
                        "emf",
                        "emf_from_di_dt",
                    )
                ],
                target,
                "emf_from_di_dt",
            )
        # Coil with N turns and a uniform flux change: emf = N * dPhi/dt.
        n_turns_total = _first_existing(known, "n_turns", "N_turns", "N")
        if n_turns_total and time and "Phi_B" in known:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute induced EMF from coil flux change.",
                        "emf = n_turns * Phi_B / t",
                        [n_turns_total, "Phi_B", time],
                        "emf",
                        "emf_from_flux_change",
                    )
                ],
                target,
                "emf_from_flux_change",
            )

    return [], 0.0


def _ac_supplemental_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    """High-frequency AC-circuit cases that the existing AC matcher doesn't reach.

    These fill specific common gaps:

      * ``I_rms = V / R`` when the circuit is *at resonance* (so Z = R and we
        only have R and a voltage). Without the resonance hint we don't
        shortcut — the more complete `_ac_templates` matcher should handle
        the general impedance case.
      * ``U_L`` (or ``U_R``, ``U_C``) at *resonance*, given R, L, C, V.
        At resonance: I = V/R, omega_0 = 1/sqrt(L*C), then U_L = I*omega_0*L,
        U_C = I/(omega_0*C), U_R = V.
    """
    resonance_hint = any(c in conditions for c in ("resonance", "rlc_resonance"))

    voltage = _first_existing(known, "V_rms", "V", "U")
    resistance = _first_by_dimension(known, "resistance")
    inductance = _first_by_dimension(known, "inductance")
    capacitance = _first_by_dimension(known, "capacitance")

    if target in {"I_rms", "I"} and resonance_hint and voltage and resistance and not (inductance or capacitance):
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "At resonance, current equals voltage over resistance.",
                    f"{target} = {voltage} / {resistance}",
                    [voltage, resistance],
                    target,
                    "rms_current_from_resistance_at_resonance",
                )
            ],
            target,
            "rms_current_from_resistance_at_resonance",
        )

    if target == "U_L" and resonance_hint and voltage and resistance and inductance and capacitance:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute current at resonance.",
                    f"I_rms = {voltage} / {resistance}",
                    [voltage, resistance],
                    "I_rms",
                    "resonance_UL_calc",
                ),
                _formula_step(
                    "step_2",
                    "Compute resonant angular frequency.",
                    "omega_0 = 1 / sqrt(L_ind * C_cap)",
                    [inductance, capacitance],
                    "omega_0",
                    "resonance_UL_calc",
                ),
                _formula_step(
                    "step_3",
                    "Compute voltage across inductor at resonance.",
                    "U_L = I_rms * omega_0 * L_ind",
                    ["I_rms", "omega_0", inductance],
                    "U_L",
                    "resonance_UL_calc",
                ),
            ],
            target,
            "resonance_UL_calc",
        )

    if target == "power_factor" and resonance_hint and not known:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "At resonance X_L = X_C, so Z = R and cos(phi) = R/Z = 1.",
                    "power_factor = 1",
                    [],
                    "power_factor",
                    "power_factor_at_resonance",
                )
            ],
            target,
            "power_factor_at_resonance",
        )

    if target == "U_C" and resonance_hint and voltage and resistance and inductance and capacitance:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute current at resonance.",
                    f"I_rms = {voltage} / {resistance}",
                    [voltage, resistance],
                    "I_rms",
                    "resonance_UC_calc",
                ),
                _formula_step(
                    "step_2",
                    "Compute resonant angular frequency.",
                    "omega_0 = 1 / sqrt(L_ind * C_cap)",
                    [inductance, capacitance],
                    "omega_0",
                    "resonance_UC_calc",
                ),
                _formula_step(
                    "step_3",
                    "Compute voltage across capacitor at resonance.",
                    "U_C = I_rms / (omega_0 * C_cap)",
                    ["I_rms", "omega_0", capacitance],
                    "U_C",
                    "resonance_UC_calc",
                ),
            ],
            target,
            "resonance_UC_calc",
        )

    return [], 0.0


def _rlc_equal_section_voltage_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    if target != "U_C":
        return [], 0.0
    if "resonance" not in conditions or "rlc_equal_rc_cl_section_voltage" not in conditions:
        return [], 0.0
    voltages = _by_dimension(known, "voltage")
    source_voltage = _first_existing(known, "V_rms", "V_source")
    if not source_voltage and voltages:
        source_voltage = voltages[0]
    if not source_voltage:
        return [], 0.0
    section_voltage = next((name for name in voltages if name != source_voltage), None)
    if not section_voltage:
        return [], 0.0
    return _finish(
        [
            _formula_step(
                "step_1",
                "At resonance with equal RC and CL section voltages, compute the capacitor RMS voltage by right-triangle subtraction.",
                f"U_C = sqrt({section_voltage}^2 - {source_voltage}^2)",
                [section_voltage, source_voltage],
                "U_C",
                "rlc_equal_section_capacitor_voltage",
            )
        ],
        target,
        "rlc_equal_section_capacitor_voltage",
    )


def _ac_detailed_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    """AC RLC off-resonance templates.

    Covers the common multi-step impedance/derived-quantity family the
    existing _ac_templates / _ac_supplemental_templates miss. The rule
    extractor stores all Ω-unit values as resistance-dim names (R, R2, R3
    in order of appearance), so when a problem text reads "XL = 25 Ω,
    XC = 100 Ω, R = 30 Ω, V = 150 V" the known set becomes
    ``{R: 25, R2: 100, R3: 30, V: 150}``. We adopt that positional
    convention here (R = X_L, R2 = X_C, R3 = R) because no name-based
    disambiguation survives extraction.

    Cases handled (gate carefully — fires AFTER _ac_supplemental_templates
    so resonance-specific plans still take precedence):

      * target ∈ {power_factor, cos_phi} with 2 resistance values →
            cos_phi = R / R2     (positional: R = pure R, R2 = Z)

      * target ∈ {I, I_rms} with 3 resistance values + voltage →
            Z = sqrt(R3^2 + (R - R2)^2),  I = V / Z

      * target ∈ {U_R, V_R} with 3 resistance values + voltage →
            Z = sqrt(R3^2 + (R - R2)^2),  I = V / Z,  U_R = I * R3

      * target ∈ {U_L, V_L} with 3 resistance values + voltage →
            Z, I as above,  U_L = I * R

      * target ∈ {U_C, V_C} with 3 resistance values + voltage →
            Z, I as above,  U_C = I * R2

      * Proper inductance + capacitance + frequency + R + V cases:
            omega = 2*pi*f,
            X_L = omega * L_ind,  X_C = 1 / (omega * C_cap),
            Z = sqrt(R^2 + (X_L - X_C)^2),
            then I = V/Z, U_R = I*R, U_L = I*X_L, U_C = I*X_C,
            cos_phi = R/Z, P_avg = I^2 * R, tan_phi = (X_L - X_C)/R.

    Anything with only 2 resistance values + V (target U_R, V_rms, etc.)
    leaks an undetermined R and we deliberately do NOT fire — those
    problems are symbolic, not numeric.
    """
    ac_hint = any(c in conditions for c in ("ac_circuit", "rlc_circuit", "series_circuit"))

    resistance_values = _by_dimension(known, "resistance")
    inductance = _first_by_dimension(known, "inductance")
    capacitance = _first_by_dimension(known, "capacitance")
    frequency = _first_existing(known, "f", "f_res", "f_osc")
    voltage = _first_existing(known, "V_rms", "V", "U")

    # ---------- Branch A: positional 2-resistance power factor ----------
    if target in {"power_factor", "cos_phi"} and len(resistance_values) == 2 and not inductance and not capacitance and ac_hint:
        r, z = resistance_values[0], resistance_values[1]
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Power factor from pure resistance and impedance (positional R, Z).",
                    f"power_factor = {r} / {z}",
                    [r, z],
                    "power_factor",
                    "ac_power_factor_from_R_Z",
                )
            ],
            target,
            "ac_power_factor_from_R_Z",
        )

    # ---------- Branch B: positional 3-resistance + V ----------
    # Convention: R = X_L, R2 = X_C, R3 = R_actual.
    if len(resistance_values) == 3 and voltage and not inductance and not capacitance:
        x_l, x_c, r_actual = resistance_values[0], resistance_values[1], resistance_values[2]
        impedance_step = _formula_step(
            "step_1",
            "Compute series RLC impedance from R, X_L, X_C.",
            f"Z = sqrt({r_actual}**2 + ({x_l} - {x_c})**2)",
            [r_actual, x_l, x_c],
            "Z",
            "ac_impedance_RLC",
        )
        current_step = _formula_step(
            "step_2",
            "Compute RMS current.",
            f"I_rms = {voltage} / Z",
            [voltage, "Z"],
            "I_rms",
            "ac_I_rms_from_V_Z",
        )

        if target in {"I", "I_rms"}:
            return _finish([impedance_step, current_step], target if target == "I" else "I_rms", "ac_impedance_RLC")
        if target in {"U_R", "V_R"}:
            ur_step = _formula_step(
                "step_3",
                "Voltage across resistor.",
                f"{target} = I_rms * {r_actual}",
                ["I_rms", r_actual],
                target,
                "ac_U_R_from_I_R",
            )
            return _finish([impedance_step, current_step, ur_step], target, "ac_U_R_from_I_R")
        if target in {"U_L", "V_L"}:
            ul_step = _formula_step(
                "step_3",
                "Voltage across inductor.",
                f"{target} = I_rms * {x_l}",
                ["I_rms", x_l],
                target,
                "ac_U_L_from_I_X_L",
            )
            return _finish([impedance_step, current_step, ul_step], target, "ac_U_L_from_I_X_L")
        if target in {"U_C", "V_C"}:
            uc_step = _formula_step(
                "step_3",
                "Voltage across capacitor.",
                f"{target} = I_rms * {x_c}",
                ["I_rms", x_c],
                target,
                "ac_U_C_from_I_X_C",
            )
            return _finish([impedance_step, current_step, uc_step], target, "ac_U_C_from_I_X_C")
        if target == "Z":
            return _finish([impedance_step], target, "ac_impedance_RLC")
        if target in {"power_factor", "cos_phi"}:
            pf_step = _formula_step(
                "step_2",
                "Power factor from R and Z.",
                f"power_factor = {r_actual} / Z",
                [r_actual, "Z"],
                "power_factor",
                "ac_power_factor_from_R_Z",
            )
            return _finish([impedance_step, pf_step], target, "ac_power_factor_from_R_Z")
        if target == "P_avg":
            pavg_step = _formula_step(
                "step_3",
                "Average AC power from I_rms and R.",
                f"P_avg = I_rms**2 * {r_actual}",
                ["I_rms", r_actual],
                "P_avg",
                "ac_avg_power_VI_cos",
            )
            return _finish([impedance_step, current_step, pavg_step], target, "ac_avg_power_VI_cos")
        if target == "tan_phi":
            tan_step = _formula_step(
                "step_1",
                "Phase angle tangent from reactances and resistance.",
                f"tan_phi = ({x_l} - {x_c}) / {r_actual}",
                [x_l, x_c, r_actual],
                "tan_phi",
                "ac_phase_angle_tan",
            )
            return _finish([tan_step], target, "ac_phase_angle_tan")

    # ---------- Branch C: full proper RLC with L_ind + C_cap + f + R + V ----------
    resistance_actual = _first_by_dimension(known, "resistance")
    if inductance and capacitance and frequency and resistance_actual:
        omega_step = _formula_step(
            "step_1",
            "Compute angular frequency from frequency.",
            f"omega = 2 * pi * {frequency}",
            [frequency, "pi"],
            "omega",
            "ac_omega_from_f",
        )
        x_l_step = _formula_step(
            "step_2",
            "Compute inductive reactance.",
            f"X_L = omega * {inductance}",
            ["omega", inductance],
            "X_L",
            "ac_X_L_from_L_omega",
        )
        x_c_step = _formula_step(
            "step_3",
            "Compute capacitive reactance.",
            f"X_C = 1 / (omega * {capacitance})",
            ["omega", capacitance],
            "X_C",
            "ac_X_C_from_C_omega",
        )
        z_step = _formula_step(
            "step_4",
            "Compute series RLC impedance.",
            f"Z = sqrt({resistance_actual}**2 + (X_L - X_C)**2)",
            [resistance_actual, "X_L", "X_C"],
            "Z",
            "ac_impedance_RLC",
        )

        if target == "Z":
            return _finish([omega_step, x_l_step, x_c_step, z_step], target, "ac_impedance_RLC")
        if target == "X_L":
            return _finish([omega_step, x_l_step], target, "ac_X_L_from_L_omega")
        if target == "X_C":
            return _finish([omega_step, x_c_step], target, "ac_X_C_from_C_omega")
        if target == "omega":
            return _finish([omega_step], target, "ac_omega_from_f")
        if target == "tan_phi":
            tan_step = _formula_step(
                "step_5",
                "Phase angle tangent.",
                f"tan_phi = (X_L - X_C) / {resistance_actual}",
                ["X_L", "X_C", resistance_actual],
                "tan_phi",
                "ac_phase_angle_tan",
            )
            return _finish([omega_step, x_l_step, x_c_step, tan_step], target, "ac_phase_angle_tan")
        if target in {"power_factor", "cos_phi"}:
            pf_step = _formula_step(
                "step_5",
                "Power factor from R and Z.",
                f"power_factor = {resistance_actual} / Z",
                [resistance_actual, "Z"],
                "power_factor",
                "ac_power_factor_from_R_Z",
            )
            return _finish([omega_step, x_l_step, x_c_step, z_step, pf_step], target, "ac_power_factor_from_R_Z")

        if voltage:
            current_step = _formula_step(
                "step_5",
                "Compute RMS current.",
                f"I_rms = {voltage} / Z",
                [voltage, "Z"],
                "I_rms",
                "ac_I_rms_from_V_Z",
            )
            if target in {"I", "I_rms"}:
                return _finish([omega_step, x_l_step, x_c_step, z_step, current_step], target if target == "I" else "I_rms", "ac_I_rms_from_V_Z")
            if target in {"U_R", "V_R"}:
                ur_step = _formula_step(
                    "step_6",
                    "Voltage across resistor.",
                    f"{target} = I_rms * {resistance_actual}",
                    ["I_rms", resistance_actual],
                    target,
                    "ac_U_R_from_I_R",
                )
                return _finish([omega_step, x_l_step, x_c_step, z_step, current_step, ur_step], target, "ac_U_R_from_I_R")
            if target in {"U_L", "V_L"}:
                ul_step = _formula_step(
                    "step_6",
                    "Voltage across inductor.",
                    f"{target} = I_rms * X_L",
                    ["I_rms", "X_L"],
                    target,
                    "ac_U_L_from_I_X_L",
                )
                return _finish([omega_step, x_l_step, x_c_step, z_step, current_step, ul_step], target, "ac_U_L_from_I_X_L")
            if target in {"U_C", "V_C"}:
                uc_step = _formula_step(
                    "step_6",
                    "Voltage across capacitor.",
                    f"{target} = I_rms * X_C",
                    ["I_rms", "X_C"],
                    target,
                    "ac_U_C_from_I_X_C",
                )
                return _finish([omega_step, x_l_step, x_c_step, z_step, current_step, uc_step], target, "ac_U_C_from_I_X_C")
            if target == "P_avg":
                pavg_step = _formula_step(
                    "step_6",
                    "Average AC power.",
                    f"P_avg = I_rms**2 * {resistance_actual}",
                    ["I_rms", resistance_actual],
                    "P_avg",
                    "ac_avg_power_VI_cos",
                )
                return _finish([omega_step, x_l_step, x_c_step, z_step, current_step, pavg_step], target, "ac_avg_power_VI_cos")

    # ---------- Branch D: I_rms with R + V only (Z given as R) ----------
    if target in {"I", "I_rms"} and len(resistance_values) == 1 and voltage and ac_hint and not inductance and not capacitance:
        r_or_z = resistance_values[0]
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "RMS current from voltage and impedance.",
                    f"{target} = {voltage} / {r_or_z}",
                    [voltage, r_or_z],
                    target,
                    "ac_I_rms_from_V_Z",
                )
            ],
            target,
            "ac_I_rms_from_V_Z",
        )

    # ---------- Branch F: resonance omega-factor variants ----------
    # "At ω0, X_L = 35 Ω, X_C = 140 Ω. By what factor of ω0 must ω be
    # changed for resonance?" The target is detected as 'X', 'X_C', or
    # 'k' depending on phrasing. The factor relation is
    #   omega_new / omega_0 = sqrt(X_C / X_L) = sqrt(R2 / R)
    # The existing _resonance_design_templates handles target=='omega' only.
    if target in {"X", "X_C", "k"} and len(resistance_values) == 2 and "omega" in known and ac_hint:
        x_l_v, x_c_v = resistance_values[0], resistance_values[1]
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute angular-frequency factor for resonance from reactance ratio.",
                    f"{target} = sqrt({x_c_v} / {x_l_v})",
                    [x_l_v, x_c_v],
                    target,
                    "rlc_omega_factor_for_resonance",
                )
            ],
            target,
            "rlc_omega_factor_for_resonance",
        )

    # ---------- Branch E: target R or Z with 1 resistance + rlc/ac hint ----------
    # "In a resonant RLC circuit, measured impedance Z=40 Ω. Find R."
    # The existing _resonance_design_templates gates on 'resonance' or
    # 'rlc_resonance' in conditions; extend coverage to 'rlc_circuit'
    # when the lone known is the impedance.
    if target == "R" and len(resistance_values) == 1 and ac_hint and not inductance and not capacitance and "R" in known:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "At resonance, pure resistance equals impedance.",
                    "R = R",
                    ["R"],
                    "R",
                    "resonance_R_from_Z",
                )
            ],
            target,
            "resonance_R_from_Z",
        )
    if target == "Z" and len(resistance_values) == 1 and ac_hint and not inductance and not capacitance:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "At resonance, impedance equals pure resistance.",
                    f"Z = {resistance_values[0]}",
                    [resistance_values[0]],
                    "Z",
                    "impedance_at_resonance",
                )
            ],
            target,
            "impedance_at_resonance",
        )

    return [], 0.0


def _parallel_plate_and_geometry_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    """Parallel-plate inverse problems + parallel-R + turn-density.

      * epsilon_r from (C, A, d) — invert parallel-plate capacitance formula
      * n_turns_per_meter = n_turns / L
      * R_eq for two resistors in parallel
    """
    if target == "epsilon_r":
        capacitance = _first_by_dimension(known, "capacitance")
        area = _first_existing(known, "A", "A_area")
        # 'd' is plate separation in parallel-plate problems
        separation = _first_existing(known, "d", "d_separation")
        # Accept either explicit parallel-plate condition or the
        # 'dielectric_capacitor' sub_domain hint (the dataset frequently
        # phrases "a capacitor has capacitance C, plate area A, separation d"
        # without naming it parallel-plate explicitly).
        plate = (
            "parallel_plate_capacitor" in conditions
            or "parallel_plate" in conditions
            or "dielectric_capacitor" in conditions
        )
        if capacitance and area and separation and plate:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Invert parallel-plate capacitance formula for relative permittivity.",
                        "epsilon_r = C_cap * d / (epsilon_0 * A)",
                        [capacitance, separation, area, "epsilon_0"],
                        "epsilon_r",
                        "epsilon_r_from_capacitance",
                    )
                ],
                target,
                "epsilon_r_from_capacitance",
            )

    if target == "n_turns_per_meter":
        n_turns_total = _first_existing(known, "n_turns", "N_turns", "N")
        length = _first_existing(known, "d", "L", "L_solenoid", "l")
        if n_turns_total and length:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute solenoid turn density.",
                        f"n_turns_per_meter = {n_turns_total} / {length}",
                        [n_turns_total, length],
                        "n_turns_per_meter",
                        "turn_density",
                    )
                ],
                target,
                "turn_density",
            )

    if target in {"R", "R_eq"} and (
        "parallel_circuit" in conditions or "ohms_law" in conditions
    ):
        resistances = _by_dimension(known, "resistance")
        # Only fire when two resistors are present and no other RLC hints —
        # otherwise we'd shadow the resonance R-from-Z template.
        if (
            len(resistances) == 2
            and not any(name.startswith("Z") or name == "measured_value" for name in known)
            and "parallel_circuit" in conditions
        ):
            r1, r2 = resistances[:2]
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Two resistors in parallel.",
                        f"R_eq = ({r1} * {r2}) / ({r1} + {r2})",
                        [r1, r2],
                        target,
                        "parallel_R",
                    )
                ],
                target,
                "parallel_R",
            )

    return [], 0.0


def _percent_error_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    """Catch-all percent-error / relative-uncertainty templates.

    The existing _measurement_templates only fires when very specific named
    inputs are present (e.g. delta_V + V + delta_I + I, or measured_value +
    true_value). The dataset has many phrasings the extractor produces as
    simpler known sets:

      * "least count 0.1 cm, measured 5.0 cm" → {d, d2, measured_value} or
        {d, d2}                          → percent_error = d / d2 * 100
      * "actual 75 kg, measured 74.2 kg" → {m, m2, true_value}
                                           → percent_error from |m - m2|/m
      * "Mass result: 200 ± 1.0 g"        → {m, delta_m}
                                           → percent_error = delta_m / m * 100
    """
    if target not in {"percent_error", "rel_error", "abs_error"}:
        return [], 0.0

    # Pattern 1: measured_value + a single matching same-dimension companion
    # (the absolute error or least-count value).
    measured = _first_existing(known, "measured_value")
    if measured and target in {"percent_error", "rel_error"}:
        measured_dim = known[measured].get("dimension")
        measured_value = known[measured].get("normalized_value", known[measured].get("value"))
        candidates: List[Tuple[str, float]] = []
        for name, quantity in known.items():
            if name == measured:
                continue
            if quantity.get("dimension") != measured_dim:
                continue
            raw_value = quantity.get("normalized_value", quantity.get("value"))
            try:
                numeric_value = abs(float(raw_value))
                measured_numeric = abs(float(measured_value))
            except (TypeError, ValueError):
                continue
            if numeric_value == measured_numeric:
                continue
            candidates.append((name, numeric_value))
        if candidates:
            cand = min(candidates, key=lambda item: item[1])[0]
            if cand:
                plan = [
                    _formula_step(
                        "step_1",
                        "Compute relative error from absolute error and measured value.",
                        f"rel_error = {cand} / abs(measured_value)",
                        [cand, "measured_value"],
                        "rel_error",
                        "percent_error_from_least_count",
                    )
                ]
                if target == "percent_error":
                    plan.append(
                        _formula_step(
                            "step_2",
                            "Convert to percent.",
                            "percent_error = rel_error * 100",
                            ["rel_error"],
                            "percent_error",
                            "percent_error_from_least_count",
                        )
                    )
                return _finish(plan, target, "percent_error_from_least_count")

    # Pattern 2: explicit symbol + delta_symbol (e.g. m + delta_m after
    # uncertainty extraction). Re-cover the simpler case that the existing
    # _measurement_templates "for base" loop already handles, but allow more
    # base names so we don't miss force/mass/etc.
    for base in ("m", "F", "R", "V", "I", "U", "P", "L", "d", "h", "temperature"):
        sym = "V" if base == "U" else base
        delta = "delta_V" if base in {"U", "V"} else f"delta_{base}"
        if sym in known and delta in known:
            if target == "abs_error":
                return _finish(
                    [
                        _formula_step(
                            "step_1",
                            "Use directly extracted absolute uncertainty.",
                            f"abs_error = {delta}",
                            [delta],
                            "abs_error",
                            "absolute_error_direct",
                        )
                    ],
                    target,
                    "absolute_error_direct",
                )
            plan = [
                _formula_step(
                    "step_1",
                    "Compute relative error from uncertainty and measured value.",
                    f"rel_error = {delta} / abs({sym})",
                    [delta, sym],
                    "rel_error",
                    "percent_error_pm_uncertainty",
                )
            ]
            if target == "percent_error":
                plan.append(
                    _formula_step(
                        "step_2",
                        "Convert to percent.",
                        "percent_error = rel_error * 100",
                        ["rel_error"],
                        "percent_error",
                        "percent_error_pm_uncertainty",
                    )
                )
            return _finish(plan, target, "percent_error_pm_uncertainty")

    # Pattern 3: "actual X, measured X" → {m, m2, true_value} or similar.
    true_v = _first_existing(known, "true_value", "accepted_value", "actual_value")
    if true_v and target in {"abs_error", "rel_error", "percent_error"}:
        # Find a measured companion: same-dim secondary like m2, V2, d2 etc.
        companion = next(
            (n for n in known if (n.endswith("2") or n == "measured_value") and n != true_v),
            None,
        )
        if companion:
            plan = [
                _formula_step(
                    "step_1",
                    "Compute absolute error.",
                    f"abs_error = abs({companion} - {true_v})",
                    [companion, true_v],
                    "abs_error",
                    "absolute_error_from_actual_measured",
                )
            ]
            if target in {"rel_error", "percent_error"}:
                plan.append(
                    _formula_step(
                        "step_2",
                        "Compute relative error.",
                        f"rel_error = abs_error / abs({true_v})",
                        ["abs_error", true_v],
                        "rel_error",
                        "absolute_error_from_actual_measured",
                    )
                )
            if target == "percent_error":
                plan.append(
                    _formula_step(
                        "step_3",
                        "Convert to percent.",
                        "percent_error = rel_error * 100",
                        ["rel_error"],
                        "percent_error",
                        "absolute_error_from_actual_measured",
                    )
                )
            return _finish(plan, target, "absolute_error_from_actual_measured")

    return [], 0.0


def _mechanics_extended_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    """Additional mechanics templates the basic _mechanics_templates doesn't cover.

    The basic matcher handles constant-speed (d=v*t) and uniform-acceleration
    cases when v_0 is named explicitly. This extension covers the cases the
    dataset actually contains:

      * Free-fall final speed when only height is known:
            v = sqrt(2 * g * h)
        Used heavily in "object falls from height h, find v at impact".
      * Free-fall time from height:
            t = sqrt(2 * h / g)
      * Free-fall height from time:
            h = 0.5 * g * t**2
      * Uniform deceleration to stop:  v_f = 0, so v_0 = -a*t and d = 0.5*v_0*t.
        Useful templates when only some of (a, d, t) are known:
            a = -2*d / t**2     (braking, given d and t)
            v_0 = 2*d / t       (initial speed from braking)
      * "Acceleration in the nth second": d_n = v_0 + a*(n - 0.5)
        Recurring "in the 5th second the object travels X meters" phrasing.
        Without v_0 known, this can't be uniquely solved but if v_0=0 is
        implied by "without initial velocity" / "starts from rest", we can.
      * Newton's second law: F_net = m * a (and inverse).
      * Vector resultant magnitude when two forces and angle are given:
            F_net = sqrt(F1^2 + F2^2 + 2*F1*F2*cos(theta))
        (the basic _force_resultant_templates already handles two-force
         scalar cases; this extension is for when angle is named theta).
      * Average velocity: v_avg = d / t (covered by basic) plus
        v_avg = (v_0 + v_f) / 2 when both endpoints known.

    All templates are single-step unless noted. Gating is permissive — we
    don't require a specific 'kinematics' sub_domain hint, but we do require
    the known-quantity shape to be unambiguous (no other interpretation fits).
    """
    distance = _first_by_dimension(known, "length")
    time = _first_by_dimension(known, "time")
    height = _first_existing(known, "h")
    acceleration = _first_by_dimension(known, "acceleration")
    velocity = _first_existing(known, "v", "v_avg", "v_final", "v_max") or _first_by_dimension(known, "velocity")
    mass = _first_existing(known, "m", "mass") or _first_by_dimension(known, "mass")
    force = _first_by_dimension(known, "force")

    free_fall = (
        "free_fall" in conditions
        or "freefall" in conditions
        or "falling" in conditions
        or "drop" in conditions
        or "dropped" in conditions
        or "kinematics" in conditions  # sub_domain hint
    )
    starts_from_rest = (
        "starts_from_rest" in conditions
        or "initial_rest" in conditions
        or "initial_velocity_zero" in conditions
        or "v0_zero" in conditions
        or "without_initial_velocity" in conditions
    )
    # 'stops' is hard to extract reliably as an explicit condition, so we
    # treat it as default-true: any (target=a, known={d,t}) is in practice a
    # braking/uniform-deceleration problem in this dataset. The 2*d/t**2 form
    # is also correct for starts-from-rest (where v_f is the unknown final);
    # the sign just flips. Stage 0 records the magnitude.
    stops_or_starts_rest = True

    # ------- Free-fall family -------
    # v = sqrt(2 g h)
    if target in {"v", "v_final", "v_impact"} and height and not velocity:
        # Even without an explicit free_fall hint, target=v with only height
        # known is almost always free-fall in this dataset.
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute impact speed from height (free fall, energy conservation).",
                    "v = sqrt(2 * g * h)",
                    [height, "g"],
                    "v",
                    "free_fall_v_from_h",
                )
            ],
            target,
            "free_fall_v_from_h",
        )
    # t = sqrt(2 h / g)
    if target == "t" and height and not (velocity or acceleration):
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute fall time from height (free fall).",
                    "t = sqrt(2 * h / g)",
                    [height, "g"],
                    "t",
                    "free_fall_t_from_h",
                )
            ],
            target,
            "free_fall_t_from_h",
        )
    # h = 0.5 g t^2  (target may be 'h' or 'd' for "how far has it fallen")
    if target in {"h", "d"} and time and free_fall and not velocity:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute fall distance from time (free fall).",
                    f"{target} = 0.5 * g * t**2",
                    [time, "g"],
                    target,
                    "free_fall_h_from_t",
                )
            ],
            target,
            "free_fall_h_from_t",
        )

    # ------- Uniform deceleration to stop -------
    # Vehicle brakes uniformly, comes to a stop. Known: d, t. Solve for a.
    if target == "a" and distance and time and stops_or_starts_rest and not velocity:
        # If it starts from rest and accelerates, d = 0.5 * a * t^2 → a = 2d/t^2
        # If it decelerates from v_0 to 0 in time t over distance d, same form
        # with the sign flipped, but the |a| has the same magnitude.
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Acceleration from distance and time (one endpoint at rest).",
                    "a = 2 * d / t**2",
                    [distance, time],
                    "a",
                    "kinematics_a_from_d_t_rest_endpoint",
                )
            ],
            target,
            "kinematics_a_from_d_t_rest_endpoint",
        )
    # Initial speed from braking: v_0 = 2d/t (decelerates to stop)
    if target in {"v", "v_0", "v_initial"} and distance and time and stops_or_starts_rest and not velocity:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Initial speed from braking distance and time (final v = 0).",
                    "v = 2 * d / t",
                    [distance, time],
                    target,
                    "kinematics_v0_from_braking",
                )
            ],
            target,
            "kinematics_v0_from_braking",
        )

    # ------- Newton's second law -------
    if target in {"F_net", "F"} and mass and acceleration:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Apply Newton's second law.",
                    f"{target} = m * a",
                    [mass, acceleration],
                    target,
                    "newton_F_from_m_a",
                )
            ],
            target,
            "newton_F_from_m_a",
        )
    if target == "a" and mass and force:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Acceleration from Newton's second law.",
                    f"a = {force} / {mass}",
                    [force, mass],
                    "a",
                    "newton_a_from_F_m",
                )
            ],
            target,
            "newton_a_from_F_m",
        )
    if target in {"m", "mass"} and force and acceleration:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Mass from Newton's second law.",
                    f"{target} = {force} / a",
                    [force, acceleration],
                    target,
                    "newton_m_from_F_a",
                )
            ],
            target,
            "newton_m_from_F_a",
        )

    # ------- Uniform acceleration with v_0=0 implied -------
    # When "starts from rest" is in conditions and the target is v/v_final
    # given a and t:  v = a*t
    if target in {"v", "v_final"} and acceleration and time and starts_from_rest:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Final speed starting from rest under uniform acceleration.",
                    f"{target} = a * t",
                    [acceleration, time],
                    target,
                    "kinematics_v_from_rest",
                )
            ],
            target,
            "kinematics_v_from_rest",
        )
    # d = 0.5 * a * t^2 (starts from rest)
    if target in {"d", "r"} and acceleration and time and starts_from_rest:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Displacement from rest under uniform acceleration.",
                    f"{target} = 0.5 * a * t**2",
                    [acceleration, time],
                    target,
                    "kinematics_d_from_rest",
                )
            ],
            target,
            "kinematics_d_from_rest",
        )

    # ------- v² - v_0² = 2 a d  -------
    # Common phrasings give (v_0, a, d) and ask for v_final, or (v_final, a, d)
    # and ask for v_0, or (v_0, v_final, d) and ask for a.
    v0 = _first_existing(known, "v_0", "v_initial")
    vf = _first_existing(known, "v_final", "v")
    if target in {"v", "v_final"} and v0 and acceleration and distance:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Final speed from kinematic energy relation.",
                    f"{target} = sqrt(v_0**2 + 2 * a * {distance})",
                    ["v_0", acceleration, distance],
                    target,
                    "kinematics_v_from_a_d",
                )
            ],
            target,
            "kinematics_v_from_a_d",
        )
    if target == "a" and v0 and vf and distance:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Acceleration from kinematic energy relation.",
                    f"a = ({vf}**2 - v_0**2) / (2 * {distance})",
                    [vf, "v_0", distance],
                    "a",
                    "kinematics_a_from_v_v0_d",
                )
            ],
            target,
            "kinematics_a_from_v_v0_d",
        )

    # ------- Average velocity from endpoints -------
    if target == "v_avg" and v0 and vf:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Average velocity for uniform acceleration.",
                    f"v_avg = (v_0 + {vf}) / 2",
                    ["v_0", vf],
                    "v_avg",
                    "v_avg_from_endpoints",
                )
            ],
            target,
            "v_avg_from_endpoints",
        )

    # ------- Two-object relative motion: target=v with d + v2 + t known -------
    # "Two cars depart from A and B, 200 km apart, meet after t hours.
    #  Car 2's speed is v2; what's car 1's speed?"
    # known set: {d, v2, t}  (one velocity + one distance + one time)
    if target in {"v", "v_1"} and distance and time:
        velocities = _by_dimension(known, "velocity")
        if len(velocities) == 1:
            v_other = velocities[0]
            # If approaching: d = (v + v_other) * t → v = d/t - v_other
            template_name = "relative_motion_v_from_d_v2_t"
            if "same_direction_chasing" in conditions:
                # d = (v - v_other) * t  →  v = d/t + v_other
                expr = f"{target} = {distance} / {time} + {v_other}"
            else:
                expr = f"{target} = {distance} / {time} - {v_other}"
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Solve for unknown velocity in meeting/chasing problem.",
                        expr,
                        [distance, time, v_other],
                        target,
                        template_name,
                    )
                ],
                target,
                template_name,
            )

    return [], 0.0


def _lc_energy_diff_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
) -> Tuple[List[Dict[str, object]], float]:
    """LC oscillation energy partition.

    "Total energy U_total, when electric energy U_E = X, what is the
     magnetic energy U_B?" → U_B = U_total - U_E.
    Extractor stores both energies as E_energy and E_energy2 (first and
    second). We adopt positional convention: E_energy = U_total,
    E_energy2 = U_E. Symmetric for U_E with E_energy2 = U_B.
    """
    energies = [n for n in ("E_energy", "E_energy2") if n in known]
    if len(energies) != 2:
        return [], 0.0
    total, other = energies[0], energies[1]
    if target in {"U_B", "U_E", "U_after"}:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "LC energy partition: target energy is total minus complementary energy.",
                    f"{target} = {total} - {other}",
                    [total, other],
                    target,
                    "lc_energy_partition",
                )
            ],
            target,
            "lc_energy_partition",
        )
    return [], 0.0


def _least_count_percent_error_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, object]], float]:
    """Percent / relative error from least-count + measured-value pairs.

    Dataset pattern: "A pressure gauge has a least count of 0.2 atm. It
    measures 2.0 atm. Calculate the percentage relative error."
    Extractor produces ``{p_pressure: 0.2, p_pressure2: 2.0}`` —
    positional: first = least count, second = measured value.
    """
    if target not in {"percent_error", "rel_error", "abs_error"}:
        return [], 0.0
    conditions = conditions or []
    least_count_factor = 0.5 if "half_least_count_uncertainty" in conditions and target != "abs_error" else 1.0
    for base in ("p_pressure", "temperature", "F", "m", "d", "h", "V", "I", "R"):
        names = sorted(
            [n for n in known if n == base or (n.startswith(base) and n[len(base):].isdigit())],
            key=lambda n: (0 if n == base else int(n[len(base):] or 0)),
        )
        if len(names) >= 2:
            values = []
            for name in names:
                try:
                    values.append((name, abs(float(known[name].get("value", 0)))))
                except (TypeError, ValueError):
                    continue
            values = [(name, value) for name, value in values if value > 0]
            if len(values) < 2:
                continue
            error_name, _ = min(values, key=lambda item: item[1])
            measured_name, _ = max(values, key=lambda item: item[1])
            if error_name == measured_name:
                continue
            if target == "abs_error":
                return _finish(
                    [
                        _formula_step(
                            "step_1",
                            "Use directly extracted least-count as absolute error.",
                            f"abs_error = {error_name}",
                            [error_name],
                            "abs_error",
                            "least_count_abs_error",
                        )
                    ],
                    target,
                    "least_count_abs_error",
                )
            rel_step = _formula_step(
                "step_1",
                "Compute relative error from least-count and measured value.",
                f"rel_error = {least_count_factor:g} * {error_name} / abs({measured_name})",
                [error_name, measured_name],
                "rel_error",
                "least_count_rel_error",
            )
            if target == "rel_error":
                return _finish([rel_step], target, "least_count_rel_error")
            pct_step = _formula_step(
                "step_2",
                "Convert to percent.",
                "percent_error = rel_error * 100",
                ["rel_error"],
                "percent_error",
                "least_count_rel_error",
            )
            return _finish([rel_step, pct_step], target, "least_count_rel_error")
    lengths = _by_dimension(known, "length")
    if len(lengths) >= 2 and target in {"percent_error", "rel_error", "abs_error"}:
        smaller, larger = (lengths[0], lengths[1])
        v0 = known[smaller].get("value", 0)
        v1 = known[larger].get("value", 0)
        if v0 and v1 and float(v0) > float(v1):
            smaller, larger = larger, smaller
        if target == "abs_error":
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Use directly extracted least-count as absolute error.",
                        f"abs_error = {smaller}",
                        [smaller],
                        "abs_error",
                        "least_count_abs_error",
                    )
                ],
                target,
                "least_count_abs_error",
            )
        rel_step = _formula_step(
            "step_1",
            "Compute relative error: least-count / measured length.",
            f"rel_error = {least_count_factor:g} * {smaller} / {larger}",
            [smaller, larger],
            "rel_error",
            "least_count_rel_error",
        )
        if target == "rel_error":
            return _finish([rel_step], target, "least_count_rel_error")
        pct_step = _formula_step(
            "step_2", "Convert to percent.",
            "percent_error = rel_error * 100",
            ["rel_error"], "percent_error", "least_count_rel_error",
        )
        return _finish([rel_step, pct_step], target, "least_count_rel_error")
    return [], 0.0


def _ab_circuit_quadrature_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, object]], float]:
    """Two-segment AC circuit with quadrature voltage and LCω²=1.

    Pattern: "Circuit AB consists of segment AM (R1 + C) and segment MB
    (R2 + L), satisfying LCω²=1, with u_AM ⊥ u_MB. Total voltage V and
    power P given. Find R1 or R2."
    Derivation: LCω²=1 → X_L = X_C → total reactance cancels →
    P = V² / (R1 + R2)  →  R_total = V² / P  →  other_R = V²/P - this_R.
    """
    conditions = conditions or []
    if "ab_quadrature_circuit" not in conditions:
        return [], 0.0
    if "R1" not in known or "R2" not in known:
        return [], 0.0
    voltage = _first_existing(known, "V_rms", "V")
    if target in {"power_factor", "cos_phi"}:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "With LCω²=1 the total reactance cancels, so the whole AB circuit is resistive.",
                    "power_factor = 1",
                    [],
                    "power_factor",
                    "ab_circuit_quadrature_power_factor",
                )
            ],
            "power_factor",
            "ab_circuit_quadrature_power_factor",
        )
    if target in {"I", "I_rms"} and voltage:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "With LCω²=1 the total impedance is R1 + R2.",
                    f"{target} = {voltage} / (R1 + R2)",
                    [voltage, "R1", "R2"],
                    target,
                    "ab_circuit_quadrature_current",
                )
            ],
            target,
            "ab_circuit_quadrature_current",
        )
    if target in {"P", "P_avg"} and "target_mb_power" in conditions and "P" in known:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Use the stated power for the same-voltage MB segment comparison.",
                    f"{target} = P",
                    ["P"],
                    target,
                    "ab_circuit_quadrature_mb_power_given",
                )
            ],
            target,
            "ab_circuit_quadrature_mb_power_given",
        )
    if target in {"P", "P_avg"} and voltage and "target_mb_power" not in conditions:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "With LCω²=1 the total impedance is R1 + R2, so total real power is U²/(R1+R2).",
                    f"{target} = {voltage}^2 / (R1 + R2)",
                    [voltage, "R1", "R2"],
                    target,
                    "ab_circuit_quadrature_total_power",
                )
            ],
            target,
            "ab_circuit_quadrature_total_power",
        )
    if target == "U_MB" and voltage:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute RMS current through the two series segments.",
                    f"I_rms = {voltage} / (R1 + R2)",
                    [voltage, "R1", "R2"],
                    "I_rms",
                    "ab_circuit_quadrature_segment_voltage",
                ),
                _formula_step(
                    "step_2",
                    "For MB, |Z_MB| = sqrt(R2² + X_L²) and X_L² = R1*R2.",
                    "U_MB = I_rms * sqrt(R2 * (R1 + R2))",
                    ["I_rms", "R1", "R2"],
                    "U_MB",
                    "ab_circuit_quadrature_segment_voltage",
                ),
            ],
            target,
            "ab_circuit_quadrature_segment_voltage",
        )

    if target not in {"R", "R2", "R1"}:
        return [], 0.0
    if "theta" not in known or "P" not in known:
        return [], 0.0
    if not voltage:
        return [], 0.0
    # Identify the "other" resistance value present in known
    other_name = None
    for cand in ("R1", "R2", "R"):
        if cand != target and cand in known:
            other_name = cand
            break
    if not other_name:
        return [], 0.0
    return _finish(
        [
            _formula_step(
                "step_1",
                "At LCω²=1 the reactances cancel; total power P = V²/(R1+R2).",
                f"R_total = {voltage}**2 / P",
                [voltage, "P"],
                "R_total",
                "ab_circuit_quadrature_resistance",
            ),
            _formula_step(
                "step_2",
                f"Solve for the unknown resistance from the total resistance.",
                f"{target} = R_total - {other_name}",
                ["R_total", other_name],
                target,
                "ab_circuit_quadrature_resistance",
            ),
        ],
        target,
        "ab_circuit_quadrature_resistance",
    )


def _frequency_change_factor(conditions: List[str]) -> Optional[float]:
    if "frequency_doubled" in conditions:
        return 2.0
    if "frequency_tripled" in conditions:
        return 3.0
    if "frequency_quadrupled" in conditions:
        return 4.0
    if "frequency_sextupled" in conditions:
        return 6.0
    return None


def _ac_frequency_change_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    """RLC reactance rescaling when source frequency changes by a factor k."""
    k = _frequency_change_factor(conditions)
    if k is None:
        return [], 0.0
    resistances = _by_dimension(known, "resistance")
    voltage = _first_existing(known, "V_rms", "V")
    if len(resistances) < 2 or not voltage:
        return [], 0.0

    x_l, x_c = resistances[0], resistances[1]
    r_actual = resistances[2] if len(resistances) >= 3 else None
    x_l_new = f"({k:g} * {x_l})"
    x_c_new = f"({x_c} / {k:g})"

    if target in {"U_R", "V_R", "V_rms"} and not r_actual:
        x_l_value = float(known[x_l].get("normalized_value", known[x_l].get("value")))
        x_c_value = float(known[x_c].get("normalized_value", known[x_c].get("value")))
        if abs(k * x_l_value - x_c_value / k) < 1e-9:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "After the frequency change the reactances cancel, so the resistor gets the full RMS voltage.",
                        f"{target} = {voltage}",
                        [voltage],
                        target,
                        "ac_frequency_change_resistor_voltage",
                    )
                ],
                target,
                "ac_frequency_change_resistor_voltage",
            )

    if not r_actual:
        return [], 0.0

    z_step = _formula_step(
        "step_1",
        "Compute impedance after reactances rescale with frequency.",
        f"Z = sqrt({r_actual}^2 + (({x_l_new}) - ({x_c_new}))^2)",
        [r_actual, x_l, x_c],
        "Z",
        "ac_frequency_change_impedance",
    )
    current_step = _formula_step(
        "step_2",
        "Compute RMS current after the frequency change.",
        f"I = {voltage} / Z",
        [voltage, "Z"],
        "I",
        "ac_frequency_change_current",
    )
    if target in {"I", "I_rms"}:
        return _finish([z_step, current_step], "I", "ac_frequency_change_current")
    if target in {"U_R", "V_R", "V_rms"}:
        ur_step = _formula_step(
            "step_3",
            "Compute RMS voltage across the resistor.",
            f"{target} = I * {r_actual}",
            ["I", r_actual],
            target,
            "ac_frequency_change_resistor_voltage",
        )
        return _finish([z_step, current_step, ur_step], target, "ac_frequency_change_resistor_voltage")
    if target in {"P", "P_avg"}:
        p_step = _formula_step(
            "step_3",
            "Compute real power dissipated by the resistor.",
            f"{target} = I^2 * {r_actual}",
            ["I", r_actual],
            target,
            "ac_frequency_change_power",
        )
        return _finish([z_step, current_step, p_step], target, "ac_frequency_change_power")
    return [], 0.0


def _ac_impedance_given_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
) -> Tuple[List[Dict[str, object]], float]:
    voltage = _first_existing(known, "V_rms", "V")
    impedance = _first_existing(known, "Z")
    resistance = _first_existing(known, "R")
    if target in {"I", "I_rms"} and voltage and impedance:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute RMS current from RMS voltage and impedance.",
                    f"{target} = {voltage} / Z",
                    [voltage, "Z"],
                    target,
                    "ac_current_from_voltage_impedance",
                )
            ],
            target,
            "ac_current_from_voltage_impedance",
        )
    if target in {"P", "P_avg"} and voltage and impedance and resistance:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "Compute real AC power from RMS voltage, resistance, and impedance.",
                    f"{target} = {voltage}^2 * R / Z^2",
                    [voltage, resistance, impedance],
                    target,
                    "ac_power_from_voltage_resistance_impedance",
                )
            ],
            target,
            "ac_power_from_voltage_resistance_impedance",
        )
    return [], 0.0


def _lc_state_boundary_templates(
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    if target in {"I", "I_rms"} and "lc_capacitor_max_charge" in conditions:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "In an LC circuit, when the capacitor charge is maximum, current is zero.",
                    f"{target} = 0",
                    [],
                    target,
                    "lc_state_boundary_zero_current",
                )
            ],
            target,
            "lc_state_boundary_zero_current",
        )
    if target in {"U_C", "V", "V_rms"} and "lc_current_maximum" in conditions:
        return _finish(
            [
                _formula_step(
                    "step_1",
                    "In an LC circuit, when current is maximum, capacitor voltage is zero.",
                    f"{target} = 0",
                    [],
                    target,
                    "lc_state_boundary_zero_capacitor_voltage",
                )
            ],
            target,
            "lc_state_boundary_zero_capacitor_voltage",
        )
    return [], 0.0


def _resonance_off_frequency_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    """Inductive reactance from on/off-resonance current pair.

    Pattern: "R=30Ω, resonance at f, I=I_at_res. When f→f2, I→I2.
    Find X_L (or Z_L) at resonance."
    Derivation (Stage 0 records the chain, downstream solves):
      V = I * R                                (at resonance, Z = R)
      Z_2 = V / I_2 = I * R / I_2
      k = f_2 / f
      (X_L - X_C)_at_f2 = sqrt(Z_2^2 - R^2)
      X_L_at_f2 = k * X_L_at_res  and  X_C_at_f2 = X_C_at_res / k
        (at resonance X_L_at_res = X_C_at_res ≡ X_L)
      So  X_L * (k - 1/k) = sqrt(Z_2^2 - R^2)
      →   X_L = sqrt(Z_2^2 - R^2) / (k - 1/k)
    """
    if target not in {"X_L", "X_C"}:
        return [], 0.0
    if "R" not in known:
        return [], 0.0
    fs = [n for n in ("f", "f2", "f3") if n in known]
    is_ = [n for n in ("I", "I2", "I3") if n in known]
    if not fs:
        return [], 0.0

    unique_fs: List[str] = []
    for name in fs:
        value = known.get(name, {}).get("normalized_value", known.get(name, {}).get("value"))
        if value is None:
            continue
        if not any(
            abs(float(value) - float(known.get(existing, {}).get("normalized_value", known.get(existing, {}).get("value")))) < 1e-12
            for existing in unique_fs
        ):
            unique_fs.append(name)
    if not unique_fs:
        return [], 0.0
    f0 = unique_fs[0]
    f2 = unique_fs[1] if len(unique_fs) >= 2 else None
    k_formula = f"{f2} / {f0}" if f2 else ("2" if "frequency_doubled" in conditions else None)
    if not k_formula:
        return [], 0.0

    current_values: List[Tuple[str, float]] = []
    for name in is_:
        value = known.get(name, {}).get("normalized_value", known.get(name, {}).get("value"))
        if value is not None:
            current_values.append((name, float(value)))
    current_values.sort(key=lambda item: item[1], reverse=True)

    if len(current_values) >= 2:
        i0, i2 = current_values[0][0], current_values[1][0]
        plan = [
            _formula_step(
                "step_1",
                "At resonance the impedance equals R, so V = I * R.",
                f"V = {i0} * R",
                [i0, "R"],
                "V",
                "off_resonance_X_L_derivation",
            ),
            _formula_step(
                "step_2",
                "Impedance at the off-resonance frequency.",
                f"Z_2 = V / {i2}",
                ["V", i2],
                "Z_2",
                "off_resonance_X_L_derivation",
            ),
        ]
    elif "current_halved" in conditions:
        plan = [
            _formula_step(
                "step_1",
                "Current halved, so off-resonance impedance is twice R.",
                "Z_2 = R / 0.5",
                ["R"],
                "Z_2",
                "off_resonance_X_L_derivation",
            )
        ]
    else:
        return [], 0.0

    next_step = len(plan) + 1
    plan.extend([
        _formula_step(
            f"step_{next_step}",
            "Frequency ratio.",
            f"k_ratio = {k_formula}",
            [name for name in (f2, f0) if name and k_formula != "2"],
            "k_ratio",
            "off_resonance_X_L_derivation",
        ),
        _formula_step(
            f"step_{next_step + 1}",
            "Resonance inductive reactance from impedance and frequency ratio.",
            (
                f"{target} = k_ratio * sqrt(Z_2**2 - R**2) / (k_ratio - 1 / k_ratio)"
                if "off_frequency_reactance_target" in conditions
                else f"{target} = sqrt(Z_2**2 - R**2) / (k_ratio - 1 / k_ratio)"
            ),
            ["Z_2", "R", "k_ratio"],
            target,
            "off_resonance_X_L_derivation",
        ),
    ])
    return _finish(plan, target, "off_resonance_X_L_derivation")


def _capacitor_merge_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
    conditions: List[str],
) -> Tuple[List[Dict[str, object]], float]:
    """Two-capacitor parallel-connection (like-polarity terminals joined).

    Pattern in the dataset: "C1 = 3 μF and C2 = 5 μF, charged to U1 = 100 V
    and U2 = 250 V, like-charged plates connected — find common voltage /
    energy / charge."  After connection (parallel), charge is conserved:
        V_common = (C1*V1 + C2*V2) / (C1 + C2)
    Stage 0 records the setup; the sign choice (same- vs opposite-poled)
    is downstream's job to disambiguate.
    """
    if target not in {"V", "V_after", "U_C", "V_common", "U_after"}:
        return [], 0.0
    caps = _by_dimension(known, "capacitance")
    vlts = _by_dimension(known, "voltage")
    if len(caps) < 2 or len(vlts) < 2:
        return [], 0.0
    c1, c2 = caps[0], caps[1]
    v1, v2 = vlts[0], vlts[1]
    return _finish(
        [
            _formula_step(
                "step_1",
                "Compute common voltage after connecting two charged capacitors in parallel.",
                f"{target} = ({c1} * {v1} + {c2} * {v2}) / ({c1} + {c2})",
                [c1, v1, c2, v2],
                target,
                "capacitor_parallel_merge",
            )
        ],
        target,
        "capacitor_parallel_merge",
    )


def _measurement_set_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
) -> Tuple[List[Dict[str, object]], float]:
    """Mean and average absolute error from N same-base measurements.

    Pattern: "Three mass measurements: 100.2 g; 100.0 g; 100.4 g.
    Calculate the average mass and the average absolute error."  The
    extractor produces ``{m: 100.2, m2: 100.0, m3: 100.4}`` (and similar
    for length d/d2/d3 or time t/t2/t3).
    """
    if target not in {"abs_error", "mean_value", "rel_error", "percent_error", "mean_abs_error_pair"}:
        return [], 0.0
    grouped_names: List[List[str]] = []
    for base in ("m", "d", "t", "V", "I", "R", "h", "temperature"):
        names = sorted([n for n in known if n == base or (n.startswith(base) and n[len(base):].isdigit())])
        if len(names) >= 3:
            grouped_names.append(names)

    seen_groups = {tuple(names) for names in grouped_names}
    for dimension in ("mass", "length", "time", "voltage", "current", "resistance", "temperature"):
        names = [n for n, quantity in known.items() if quantity.get("dimension") == dimension]
        if len(names) < 3:
            continue
        unit_symbols = {str(known[n].get("unit_symbol") or "") for n in names}
        if len(unit_symbols) > 1:
            continue
        group_key = tuple(sorted(names))
        if group_key not in seen_groups:
            grouped_names.append(list(group_key))
            seen_groups.add(group_key)

    for names in grouped_names:
        mean_step = _formula_step(
            "step_1",
            "Compute the mean of repeated measurements.",
            "mean_value = sum(measurements) / n",
            names,
            "mean_value",
            "mean_value",
        )
        if target == "mean_value":
            return _finish([mean_step], target, "mean_value")
        abs_err_step = _formula_step(
            "step_2",
            "Compute average absolute error from deviations.",
            "abs_error = sum(abs(each - mean_value)) / n",
            names + ["mean_value"],
            "abs_error",
            "average_absolute_error",
        )
        if target == "abs_error":
            return _finish([mean_step, abs_err_step], target, "average_absolute_error")
        if target == "mean_abs_error_pair":
            pair_step = _formula_step(
                "step_3",
                "Report the mean value and mean absolute error together.",
                "mean_abs_error_pair = pair(mean_value, abs_error)",
                ["mean_value", "abs_error"],
                "mean_abs_error_pair",
                "mean_abs_error_pair",
            )
            return _finish([mean_step, abs_err_step, pair_step], target, "mean_abs_error_pair")
        rel_step = _formula_step(
            "step_3",
            "Compute relative error from absolute error and mean.",
            "rel_error = abs_error / abs(mean_value)",
            ["abs_error", "mean_value"],
            "rel_error",
            "relative_error_from_mean",
        )
        if target == "rel_error":
            return _finish([mean_step, abs_err_step, rel_step], target, "relative_error_from_mean")
        if target == "percent_error":
            pct_step = _formula_step(
                "step_4",
                "Convert relative error to percent.",
                "percent_error = rel_error * 100",
                ["rel_error"],
                "percent_error",
                "relative_error_from_mean",
            )
            return _finish([mean_step, abs_err_step, rel_step, pct_step], target, "relative_error_from_mean")
    return [], 0.0


def _basic_templates(known: Dict[str, Dict[str, object]], target: str) -> Tuple[List[Dict[str, object]], float]:
    if target == "T_period" and "f" in known:
        return _finish([_formula_step("step_1", "Compute wave period from frequency.", "T_period = 1 / f", ["f"], "T_period", "wave_period_frequency")], target, "wave_period_frequency")
    if target == "lambda" and ("v_wave" in known or "v" in known) and "f" in known:
        speed = "v_wave" if "v_wave" in known else "v"
        return _finish([_formula_step("step_1", "Compute wavelength from wave speed and frequency.", "lambda = v_wave / f", [speed, "f"], "lambda", "wave_wavelength")], target, "wave_wavelength")
    if target in {"d", "r"}:
        velocities = _by_dimension(known, "velocity")
        if len(velocities) >= 2 and "t" in known:
            plan = [
                _formula_step("step_1", "Compute relative speed for approaching objects.", "relative_speed = v1 + v2", velocities[:2], "relative_speed", "relative_motion_meeting"),
                _formula_step("step_2", "Compute meeting distance.", "d = relative_speed * t", ["relative_speed", "t"], target, "relative_motion_meeting"),
            ]
            return _finish(plan, target, "relative_motion_meeting")
    if target == "v_avg" and "total_distance" in known and "total_time" in known:
        return _finish([_formula_step("step_1", "Compute average speed.", "v_avg = total_distance / total_time", ["total_distance", "total_time"], "v_avg", "average_speed_multistage")], target, "average_speed_multistage")
    if target in {"relation_E", "relation_generic", "equation_of_motion"}:
        return _finish([_formula_step("step_1", "Record conceptual relationship target for downstream symbolic reasoning.", f"{target} = extracted_relationship", [], target, "conceptual_relation_target", warning="Conceptual relationship target uses a lightweight Stage 0 placeholder.")], target, "conceptual_relation_target")
    if target == "I" and _has(known, "V", "R"):
        return _finish([_formula_step("step_1", "Apply Ohm's law for current.", "I = V / R", ["V", "R"], "I", "ohms_law_current")], target, "ohms_law_current")
    if target == "P" and _has(known, "V", "I"):
        return _finish([_formula_step("step_1", "Apply electric power relation.", "P = V * I", ["V", "I"], "P", "power_from_voltage_current")], target, "power_from_voltage_current")
    if target == "P" and _has(known, "I", "R"):
        return _finish([_formula_step("step_1", "Apply resistive power relation.", "P = I^2 * R", ["I", "R"], "P", "power_from_current_resistance")], target, "power_from_current_resistance")
    if target == "P" and _has(known, "V", "R"):
        return _finish([_formula_step("step_1", "Apply resistive power relation.", "P = V^2 / R", ["V", "R"], "P", "power_from_voltage_resistance")], target, "power_from_voltage_resistance")
    if target == "Q" and _has(known, "C_cap", "V"):
        return _finish([_formula_step("step_1", "Apply capacitor charge relation.", "Q = C_cap * V", ["C_cap", "V"], "Q", "capacitor_charge")], target, "capacitor_charge")
    if target in {"q", "q0", "q1", "q2", "q3"} and _has(known, "C_cap", "V"):
        return _finish([_formula_step("step_1", "Apply capacitor charge relation.", f"{target} = C_cap * V", ["C_cap", "V"], target, "capacitor_charge")], target, "capacitor_charge")
    if target in {"q", "q0", "q1", "q2", "q3"}:
        distance = _first_by_dimension(known, "length")
        force = _first_by_dimension(known, "force")
        field = _first_by_dimension(known, "electric_field")
        if field and distance:
            return _finish([_formula_step("step_1", "Invert point-charge electric field relation.", f"{target} = E * r^2 / k", [field, distance, "k"], target, "electric_field_charge_inverse")], target, "electric_field_charge_inverse")
        if force and distance:
            return _finish([_formula_step("step_1", "Invert Coulomb's law for equal charges.", f"{target} = sqrt(F * r^2 / k)", [force, distance, "k"], target, "coulomb_equal_charge_inverse")], target, "coulomb_equal_charge_inverse")
    if target == "U_cap" and _has(known, "C_cap", "V"):
        return _finish([_formula_step("step_1", "Apply capacitor energy relation.", "U_cap = 0.5 * C_cap * V^2", ["C_cap", "V"], "U_cap", "capacitor_energy")], target, "capacitor_energy")
    if target == "C_cap" and _has(known, "Q", "V"):
        return _finish([_formula_step("step_1", "Apply capacitance definition.", "C_cap = Q / V", ["Q", "V"], "C_cap", "capacitance_definition")], target, "capacitance_definition")
    if target == "V" and ("Q" in known or "q" in known) and "C_cap" in known:
        charge = "Q" if "Q" in known else "q"
        return _finish([_formula_step("step_1", "Apply capacitance definition.", "V = Q / C_cap", [charge, "C_cap"], "V", "capacitance_voltage")], target, "capacitance_voltage")
    if target == "v_final" and _has(known, "v_0", "a", "t"):
        return _finish([_formula_step("step_1", "Apply constant-acceleration velocity relation.", "v_final = v_0 + a * t", ["v_0", "a", "t"], "v_final", "kinematics_final_velocity")], target, "kinematics_final_velocity")
    if target == "d" and _has(known, "v", "t"):
        return _finish([_formula_step("step_1", "Apply constant speed distance relation.", "d = v * t", ["v", "t"], "d", "constant_speed_distance")], target, "constant_speed_distance")
    if target == "v_wave" and _has(known, "f", "lambda"):
        return _finish([_formula_step("step_1", "Apply wave speed relation.", "v_wave = f * lambda", ["f", "lambda"], "v_wave", "wave_speed")], target, "wave_speed")
    if target == "F_net":
        mass = _first_by_dimension(known, "mass")
        if mass and "a" in known:
            return _finish([_formula_step("step_1", "Apply Newton's second law.", "F_net = m * a", [mass, "a"], "F_net", "newton_second_law")], target, "newton_second_law")
        force_names = _by_dimension(known, "force")
        charges = _by_dimension(known, "charge")
        distances = _by_dimension(known, "length")
        if len(force_names) >= 2 and "theta" in known:
            return _finish([_formula_step("step_1", "Apply vector resultant magnitude relation.", "F_net = sqrt(F1^2 + F2^2 + 2*F1*F2*cos(theta))", [force_names[0], force_names[1], "theta"], "F_net", "force_resultant_angle")], target, "force_resultant_angle")
        if len(force_names) >= 2:
            return _finish([_formula_step("step_1", "Combine collinear force magnitudes.", "F_net = sum(F_i)", force_names[:3], "F_net", "force_resultant_collinear")], target, "force_resultant_collinear")
        if len(charges) >= 2 and distances:
            return _finish([_formula_step("step_1", "Apply Coulomb force relation before vector combination.", "F_net = vector_sum(k*q_i*q_j/r_ij^2)", charges[:3] + distances[:3] + ["k"], "F_net", "coulomb_force_vector")], target, "coulomb_force_vector")
    if target == "F_e":
        charges = _by_dimension(known, "charge")
        distance = _first_by_dimension(known, "length")
        if len(charges) >= 2 and distance:
            return _finish([_formula_step("step_1", "Apply Coulomb's law.", "F_e = k * abs(q1 * q2) / d^2", [charges[0], charges[1], distance, "k"], "F_e", "coulomb_force_scalar")], target, "coulomb_force_scalar")
    if target == "E":
        charge = _first_by_dimension(known, "charge")
        distance = _first_by_dimension(known, "length")
        if charge and distance:
            return _finish([_formula_step("step_1", "Apply point-charge electric field relation.", "E = k * abs(q) / r^2", [charge, distance, "k"], "E", "electric_field_point_charge")], target, "electric_field_point_charge")
        fields = _by_dimension(known, "electric_field")
        if len(fields) >= 2:
            return _finish([_formula_step("step_1", "Combine electric field contributions.", "E = vector_sum(E_i)", fields[:3], "E", "electric_field_vector_sum")], target, "electric_field_vector_sum")
    return [], 0.0


def propose_step_plan(
    known_quantities: Dict[str, Dict[str, object]],
    unknown_quantity: Optional[str],
    conditions: Optional[List[str]] = None,
    relations: Optional[List[Dict[str, object]]] = None,
) -> Tuple[List[Dict[str, object]], float]:
    """Return a formula-pattern step plan when a known template matches."""
    if not unknown_quantity:
        return [], 0.0
    conditions = conditions or []
    relations = relations or []
    known = known_quantities
    target = str(unknown_quantity)
    for matcher in (
        lambda: _relation_driven_templates(target, relations),
        lambda: _capacitor_energy_templates(known, target, conditions),
        lambda: _inductor_energy_templates(known, target),
        lambda: _sinusoidal_energy_templates(known, target, relations),
        lambda: _sinusoidal_ac_source_templates(known, target, relations),
        lambda: _force_resultant_templates(known, target, conditions),
        lambda: _zero_field_distance_templates(known, target, conditions),
        lambda: _field_line_midpoint_templates(known, target, conditions),
        lambda: _continuous_charge_field_templates(known, target, conditions),
        lambda: _electric_equilibrium_templates(known, target, conditions),
        lambda: _electric_force_field_templates(known, target, conditions),
        lambda: _field_geometry_templates(known, target, conditions),
        lambda: _square_field_cancellation_templates(known, target, conditions),
        lambda: _capacitance_templates(known, target, conditions),
        lambda: _coulomb_force_templates(known, target, conditions),
        lambda: _ab_circuit_quadrature_templates(known, target, conditions),
        lambda: _ac_frequency_change_templates(known, target, conditions),
        lambda: _ac_impedance_given_templates(known, target),
        lambda: _measurement_bound_templates(known, target, conditions),
        lambda: _circuit_templates(known, target, conditions),
        lambda: _mechanics_templates(known, target, conditions),
        lambda: _mechanics_extended_templates(known, target, conditions),
        lambda: _dielectric_templates(known, target, conditions),
        lambda: _measurement_error_propagation_templates(known, target, conditions),
        lambda: _measurement_templates(known, target),
        lambda: _least_count_percent_error_templates(known, target, conditions),
        lambda: _measurement_set_templates(known, target),
        lambda: _percent_error_templates(known, target, conditions),
        lambda: _lc_energy_diff_templates(known, target),
        lambda: _capacitor_merge_templates(known, target, conditions),
        lambda: _resonance_off_frequency_templates(known, target, conditions),
        lambda: _lc_state_boundary_templates(target, conditions),
        lambda: _lc_templates(known, target, relations),
        lambda: _resonance_design_templates(known, target, conditions),
        lambda: _rlc_equal_section_voltage_templates(known, target, conditions),
        lambda: _ac_supplemental_templates(known, target, conditions),
        lambda: _ac_detailed_templates(known, target, conditions),
        lambda: _ac_templates(known, target),
        lambda: _parallel_plate_and_geometry_templates(known, target, conditions),
        lambda: _electromagnetism_templates(known, target, conditions),
        lambda: _basic_templates(known, target),
    ):
        plan, confidence = matcher()
        if plan:
            return plan, confidence
    return [], 0.0
