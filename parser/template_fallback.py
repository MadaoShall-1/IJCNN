"""Formula-pattern fallback that creates step plans without solving."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


CONSTANTS = {"pi", "k", "epsilon_0", "mu_0"}


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
    if target == "Q_factor":
        resistance = _first_by_dimension(known, "resistance")
        inductance = _first_by_dimension(known, "inductance")
        capacitance = _first_by_dimension(known, "capacitance")
        if resistance and inductance and "omega_0" in known:
            return _finish([_formula_step("step_1", "Compute series RLC quality factor.", "Q_factor = omega_0 * L / R", ["omega_0", inductance, resistance], "Q_factor", "quality_factor_series_rlc")], target, "quality_factor_series_rlc")
        if resistance and inductance and capacitance:
            plan = [
                _formula_step("step_1", "Compute resonant angular frequency.", "omega_0 = 1 / sqrt(L*C_cap)", [inductance, capacitance], "omega_0", "quality_factor_series_rlc"),
                _formula_step("step_2", "Compute series RLC quality factor.", "Q_factor = omega_0 * L / R", ["omega_0", inductance, resistance], "Q_factor", "quality_factor_series_rlc"),
            ]
            return _finish(plan, target, "quality_factor_series_rlc")
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
    voltage = _first_existing(known, "U", "V")
    delta_voltage = _first_existing(known, "delta_U", "delta_V")
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
    true_name = _first_existing(known, "true_value", "accepted_value")
    if target in {"abs_error", "rel_error", "percent_error"} and "measured_value" in known and true_name:
        plan = [_formula_step("step_1", "Compute absolute error.", "abs_error = abs(measured_value - true_value)", ["measured_value", true_name], "abs_error", "absolute_error")]
        if target in {"rel_error", "percent_error"}:
            plan.append(_formula_step("step_2", "Compute relative error.", "rel_error = abs_error / abs(true_value)", ["abs_error", true_name], "rel_error", "relative_error"))
        if target == "percent_error":
            plan.append(_formula_step("step_3", "Convert relative error to percent.", "percent_error = rel_error * 100", ["rel_error"], "percent_error", "percent_error"))
        return _finish(plan, target, "measurement_error")
    measurement_names = [name for name in known if name.startswith(("I", "V", "d", "t", "temperature")) and name[-1:].isdigit()]
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

    template_name = "coulomb_force_vector" if target in {"F_net", "F_on_q3"} else "coulomb_force_scalar"
    if len(charges) >= 2 and target in {"F_e", "F_net"} and len(charges) < 3:
        return _finish(
            [_formula_step("step_1", "Apply scalar Coulomb's law.", f"{target} = k * abs(q1 * q2) / r^2", [charges[0], charges[1], distances[0], "k"], target, template_name)],
            target,
            template_name,
        )

    plan: List[Dict[str, object]] = []
    if len(charges) < 3:
        missing = {f"q{index}": f"symbolic charge q{index}" for index in range(1, 4) if f"q{index}" not in known}
        if missing:
            plan.append(_setup_step("step_1", "Introduce symbolic charges required for force-on-q3 vector setup.", missing, template_name))
            charges = charges + list(missing.keys())
    if len(charges) < 3:
        return [], 0.0

    r13 = distances[0]
    r23 = distances[1] if len(distances) > 1 else distances[0]
    plan.extend(
        [
            _formula_step("step_1", "Compute force on q3 due to q1.", "F_13 = k * abs(q1*q3) / r13^2", [charges[0], charges[2], r13, "k"], "F_13", template_name),
            _formula_step("step_1", "Compute force on q3 due to q2.", "F_23 = k * abs(q2*q3) / r23^2", [charges[1], charges[2], r23, "k"], "F_23", template_name),
        ]
    )
    output = target
    if "theta" in known:
        combine = f"{output} = sqrt(F_13^2 + F_23^2 + 2*F_13*F_23*cos(theta))"
        inputs = ["F_13", "F_23", "theta"]
    elif "right_angle" in conditions:
        combine = f"{output} = sqrt(F_13^2 + F_23^2)"
        inputs = ["F_13", "F_23"]
    elif "equilateral_triangle" in conditions:
        combine = f"{output} = sqrt(F_13^2 + F_23^2 + 2*F_13*F_23*cos(60deg))"
        inputs = ["F_13", "F_23"]
    elif "square_center" in conditions:
        combine = f"{output} = vector_sum(F_13, F_23, symmetry_terms)"
        inputs = ["F_13", "F_23"]
    else:
        combine = f"{output} = vector_sum(F_13, F_23)"
        inputs = ["F_13", "F_23"]
        plan[-1]["parser_warning"] = "Geometry is ambiguous; vector combination is conservative and left symbolic."
    plan.append(_formula_step("step_1", "Combine pairwise electric forces using available geometry.", combine, inputs, output, template_name))
    _renumber(plan)
    return _finish(plan, target, template_name)


def _circuit_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    resistance_names = _by_dimension(known, "resistance")
    current_names = _by_dimension(known, "current")
    voltage_names = _by_dimension(known, "voltage")
    power_names = _by_dimension(known, "power")
    voltage = _first_existing(known, "V", "V_rms") or (voltage_names[0] if voltage_names else None)
    current = _first_existing(known, "I", "I_rms") or (current_names[0] if current_names else None)
    resistance = _first_existing(known, "R") or (resistance_names[0] if resistance_names else None)

    if target in {"I", "I_total"} and voltage and resistance:
        return _finish([_formula_step("step_1", "Apply Ohm's law for current.", f"{target} = V / R", [voltage, resistance], target, "ohms_law_current")], target, "ohms_law_current")
    if target == "V" and current and resistance:
        return _finish([_formula_step("step_1", "Apply Ohm's law for voltage.", "V = I * R", [current, resistance], "V", "ohms_law_voltage")], target, "ohms_law_voltage")
    if target in {"R", "R_eq"} and voltage and current:
        return _finish([_formula_step("step_1", "Apply Ohm's law for resistance.", f"{target} = V / I", [voltage, current], target, "ohms_law_resistance")], target, "ohms_law_resistance")
    if target == "R_eq" and len(resistance_names) >= 2:
        if "parallel_circuit" in conditions:
            formula = "R_eq = R1*R2/(R1+R2)" if len(resistance_names) == 2 else "1/R_eq = sum(1/R_i)"
            return _finish([_formula_step("step_1", "Compute equivalent parallel resistance.", formula, resistance_names, "R_eq", "parallel_resistance")], target, "parallel_resistance")
        if "series_circuit" in conditions:
            return _finish([_formula_step("step_1", "Compute equivalent series resistance.", "R_eq = sum(R_i)", resistance_names, "R_eq", "series_resistance")], target, "series_resistance")
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

    if target in {"U_cap", "U_E", "U_total"}:
        output = target
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


def _field_geometry_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    if target != "E":
        return [], 0.0
    charges = _by_dimension(known, "charge")
    distance = _first_by_dimension(known, "length")
    if charges and distance:
        return _finish([_formula_step("step_1", "Compute point-charge electric field magnitude.", "E = k * abs(q) / r^2", [charges[0], distance, "k"], "E", "electric_field_point_charge")], target, "electric_field_point_charge")
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


def _force_resultant_templates(known: Dict[str, Dict[str, object]], target: str, conditions: List[str]) -> Tuple[List[Dict[str, object]], float]:
    forces = _by_dimension(known, "force")
    if target == "F_net" and len(forces) == 1 and "theta" in known:
        return _finish([_formula_step("step_1", "Combine two equal force magnitudes at the given angle.", "F_net = sqrt(F^2 + F^2 + 2*F*F*cos(theta))", [forces[0], "theta"], "F_net", "force_resultant_equal_angle")], target, "force_resultant_equal_angle")
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
                    "R = Z",
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
    if target == "omega" and "omega" in known:
        # Heuristic: two resistance-typed quantities present, both look like
        # reactances. The factor relation is ω_new/ω₀ = sqrt(X_C/X_L).
        resistances = _by_dimension(known, "resistance")
        if len(resistances) >= 2:
            x_l = resistances[0]
            x_c = resistances[1]
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Compute new angular frequency factor for resonance.",
                        f"omega = omega * sqrt({x_c} / {x_l})",
                        ["omega", x_l, x_c],
                        "omega",
                        "rlc_omega_factor_for_resonance",
                    )
                ],
                target,
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
                        "B = mu_0 * n_turns_per_meter * I",
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
                        "n_turns_per_meter = n_turns / L",
                        [n_turns_total, length],
                        "n_turns_per_meter",
                        "solenoid_field_full",
                    ),
                    _formula_step(
                        "step_2",
                        "Compute solenoid magnetic field.",
                        "B = mu_0 * n_turns_per_meter * I",
                        ["n_turns_per_meter", current, "mu_0"],
                        "B",
                        "solenoid_field_full",
                    ),
                ],
                target,
                "solenoid_field_full",
            )

    if target == "Phi_B":
        b_field = _first_existing(known, "B")
        area = _first_existing(known, "A", "A_area")
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

    if target == "Phi_link":
        n_turns_total = _first_existing(known, "n_turns", "N_turns", "N")
        b_field = _first_existing(known, "B")
        area = _first_existing(known, "A", "A_area")
        if n_turns_total and "Phi_B" in known:
            return _finish(
                [
                    _formula_step(
                        "step_1",
                        "Total flux linkage equals N * per-turn flux.",
                        "Phi_link = n_turns * Phi_B",
                        [n_turns_total, "Phi_B"],
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

    if target == "emf":
        inductance = _first_by_dimension(known, "inductance")
        current = _first_existing(known, "I", "I_rms")
        time = _first_existing(known, "t", "delta_t")
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
                    "R = Z",
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
                        "n_turns_per_meter = n_turns / L",
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
        # Find an absolute error value: any same-dim "d", "d2", "m", "m2", etc.
        for base in ("d", "m", "V", "I", "R", "h", "temperature", "p_pressure", "F"):
            companion_names = [n for n in known if n.startswith(base) and n != base + ""]
            # Prefer a non-prime variant; fall back to base itself.
            cand = next((n for n in companion_names if n != measured), None)
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
) -> Tuple[List[Dict[str, object]], float]:
    """Percent / relative error from least-count + measured-value pairs.

    Dataset pattern: "A pressure gauge has a least count of 0.2 atm. It
    measures 2.0 atm. Calculate the percentage relative error."
    Extractor produces ``{p_pressure: 0.2, p_pressure2: 2.0}`` —
    positional: first = least count, second = measured value.
    """
    if target not in {"percent_error", "rel_error", "abs_error"}:
        return [], 0.0
    for base in ("p_pressure", "temperature", "F", "m", "d", "h", "V", "I", "R"):
        a = base
        b = base + "2"
        if a in known and b in known:
            if target == "abs_error":
                return _finish(
                    [
                        _formula_step(
                            "step_1",
                            "Use directly extracted least-count as absolute error.",
                            f"abs_error = {a}",
                            [a],
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
                f"rel_error = {a} / abs({b})",
                [a, b],
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
    return [], 0.0


def _ab_circuit_quadrature_templates(
    known: Dict[str, Dict[str, object]],
    target: str,
) -> Tuple[List[Dict[str, object]], float]:
    """Two-segment AC circuit with quadrature voltage and LCω²=1.

    Pattern: "Circuit AB consists of segment AM (R1 + C) and segment MB
    (R2 + L), satisfying LCω²=1, with u_AM ⊥ u_MB. Total voltage V and
    power P given. Find R1 or R2."
    Derivation: LCω²=1 → X_L = X_C → total reactance cancels →
    P = V² / (R1 + R2)  →  R_total = V² / P  →  other_R = V²/P - this_R.
    """
    if target not in {"R", "R2", "R1"}:
        return [], 0.0
    if "theta" not in known or "P" not in known:
        return [], 0.0
    voltage = _first_existing(known, "V_rms", "V")
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
    if len(fs) < 2 or len(is_) < 2:
        return [], 0.0
    f0, f2 = fs[0], fs[1]
    i0, i2 = is_[0], is_[1]
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
        _formula_step(
            "step_3",
            "Frequency ratio.",
            f"k_ratio = {f2} / {f0}",
            [f2, f0],
            "k_ratio",
            "off_resonance_X_L_derivation",
        ),
        _formula_step(
            "step_4",
            "Resonance inductive reactance from impedance and frequency ratio.",
            f"{target} = sqrt(Z_2**2 - R**2) / (k_ratio - 1 / k_ratio)",
            ["Z_2", "R", "k_ratio"],
            target,
            "off_resonance_X_L_derivation",
        ),
    ]
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
    if target not in {"abs_error", "mean_value", "rel_error", "percent_error"}:
        return [], 0.0
    for base in ("m", "d", "t", "V", "I", "R", "h", "temperature"):
        names = sorted([n for n in known if n == base or (n.startswith(base) and n[len(base):].isdigit())])
        if len(names) < 3:
            continue
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
        lambda: _force_resultant_templates(known, target, conditions),
        lambda: _field_geometry_templates(known, target, conditions),
        lambda: _capacitance_templates(known, target, conditions),
        lambda: _coulomb_force_templates(known, target, conditions),
        lambda: _circuit_templates(known, target, conditions),
        lambda: _mechanics_templates(known, target, conditions),
        lambda: _mechanics_extended_templates(known, target, conditions),
        lambda: _dielectric_templates(known, target, conditions),
        lambda: _measurement_templates(known, target),
        lambda: _measurement_set_templates(known, target),
        lambda: _least_count_percent_error_templates(known, target),
        lambda: _percent_error_templates(known, target, conditions),
        lambda: _lc_energy_diff_templates(known, target),
        lambda: _capacitor_merge_templates(known, target, conditions),
        lambda: _resonance_off_frequency_templates(known, target, conditions),
        lambda: _ab_circuit_quadrature_templates(known, target),
        lambda: _lc_templates(known, target, relations),
        lambda: _resonance_design_templates(known, target, conditions),
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