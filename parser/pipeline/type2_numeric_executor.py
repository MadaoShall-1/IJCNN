"""Deterministic numeric executor for selected Type2 step plans."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from parser.pipeline.type2_adapter import Type2WorldModelInput


@dataclass
class Type2ExecutionStepResult:
    step_id: str
    formula_name: str | None
    template_name: str | None
    status: str
    input_values: dict[str, Any]
    output_values: dict[str, Any]
    numeric_value: float | None
    unit: str | None
    warnings: list[str]
    errors: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Type2NumericExecutionResult:
    status: str
    target: str | None
    unit: str | None
    numeric_value: float | None
    answer: str | None
    execution_trace: list[Type2ExecutionStepResult]
    computed_values: dict[str, Any]
    warnings: list[str]
    errors: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "target": self.target,
            "unit": self.unit,
            "numeric_value": self.numeric_value,
            "answer": self.answer,
            "execution_trace": [step.to_dict() for step in self.execution_trace],
            "computed_values": dict(self.computed_values),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


UNIT_FACTORS = {
    "c": ("C", 1.0),
    "mc": ("C", 1e-3),
    "μc": ("C", 1e-6),
    "μc": ("C", 1e-6),
    "µc": ("C", 1e-6),
    "uc": ("C", 1e-6),
    "nc": ("C", 1e-9),
    "f": ("F", 1.0),
    "μf": ("F", 1e-6),
    "μf": ("F", 1e-6),
    "µf": ("F", 1e-6),
    "uf": ("F", 1e-6),
    "nf": ("F", 1e-9),
    "m": ("m", 1.0),
    "cm": ("m", 1e-2),
    "mm": ("m", 1e-3),
    "n": ("N", 1.0),
    "v": ("V", 1.0),
    "a": ("A", 1.0),
    "ω": ("Ω", 1.0),
    "ohm": ("Ω", 1.0),
    "j": ("J", 1.0),
    "deg": ("deg", 1.0),
    "degree": ("deg", 1.0),
    "degrees": ("deg", 1.0),
    "°": ("deg", 1.0),
    "w": ("W", 1.0),
}
CONSTANTS = {
    "k": {"value": 8.9875517923e9, "unit": "N*m^2/C^2"},
    "pi": {"value": math.pi, "unit": None},
    "epsilon_0": {"value": 8.854187817e-12, "unit": "F/m"},
    "mu_0": {"value": 4 * math.pi * 1e-7, "unit": "T*m/A"},
    "g": {"value": 9.8, "unit": "m/s^2"},
}


def _normalize_var_name(name: Any) -> str:
    return str(name or "").strip().replace(" ", "_")


def _canonical_unit(unit: Any) -> str | None:
    if unit is None:
        return None
    text = str(unit).strip()
    if not text:
        return None
    return (
        text.replace("µ", "μ")
        .replace("Âµ", "μ")
        .replace("Î¼", "μ")
        .replace("Ω", "Ω")
        .replace("Î©", "Ω")
        .replace("â„¦", "Ω")
        .replace("°", "°")
        .replace("Â°", "°")
    )


def _parse_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    text = str(value).strip()
    text = (
        text.replace("×", "e")
        .replace("x", "e")
        .replace("X", "e")
        .replace("−", "-")
        .replace("⁻", "-")
        .replace("⁰", "0")
        .replace("¹", "1")
        .replace("²", "2")
        .replace("³", "3")
        .replace("⁴", "4")
        .replace("⁵", "5")
        .replace("⁶", "6")
        .replace("⁷", "7")
        .replace("⁸", "8")
        .replace("⁹", "9")
        .replace(" ", "")
    )
    text = re.sub(r"e10\^?(-?\d+)", r"e\1", text)
    pow_match = re.search(r"(^|[^0-9.])10\^?(-?\d+)", text)
    if pow_match and not re.search(r"\d(?:e|E)[-+]?\d", text):
        coefficient = text[: pow_match.start()].strip()
        coefficient_value = _parse_number(coefficient) if coefficient else 1.0
        if coefficient_value is None:
            coefficient_value = 1.0
        return coefficient_value * (10 ** int(pow_match.group(2)))
    match = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", text, re.IGNORECASE)
    if not match:
        pow_match = re.search(r"10\^(-?\d+)", text)
        if pow_match:
            return 10 ** int(pow_match.group(1))
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _split_value_unit(text: str) -> tuple[float | None, str | None]:
    value = _parse_number(text)
    unit_match = re.search(r"(μC|uC|mC|nC|C|μF|uF|nF|F|cm|mm|m|N|V|A|Ω|ohm|J|degrees|degree|deg|°|W)\b", text)
    return value, unit_match.group(1) if unit_match else None


def _get_quantity_value(quantity_obj: Any) -> float | None:
    if not isinstance(quantity_obj, dict):
        return _parse_number(quantity_obj)
    for key in ("si_value", "normalized_value", "value"):
        if key in quantity_obj:
            parsed = _parse_number(quantity_obj.get(key))
            if parsed is not None:
                return parsed
    for key in ("raw_text", "source_text", "text"):
        if key in quantity_obj:
            parsed, _ = _split_value_unit(str(quantity_obj.get(key)))
            if parsed is not None:
                return parsed
    return None


def _get_quantity_unit(quantity_obj: Any) -> str | None:
    if not isinstance(quantity_obj, dict):
        if isinstance(quantity_obj, str):
            _, unit = _split_value_unit(quantity_obj)
            return unit
        return None
    for key in ("si_unit", "normalized_unit_symbol", "unit", "unit_symbol", "symbol"):
        unit = quantity_obj.get(key)
        if unit:
            return str(unit)
    for key in ("raw_text", "source_text", "text"):
        if key in quantity_obj:
            _, unit = _split_value_unit(str(quantity_obj.get(key)))
            if unit:
                return unit
    return None


def _convert_to_si(value: float | None, unit: str | None) -> tuple[float | None, str | None]:
    if value is None:
        return None, _canonical_unit(unit)
    unit_norm = _canonical_unit(unit)
    if not unit_norm:
        return value, None
    unit_key = unit_norm.lower()
    if unit_key in UNIT_FACTORS:
        si_unit, factor = UNIT_FACTORS[unit_key]
        return value * factor, si_unit
    return value, unit_norm


def _env_item(value: float, unit: str | None = None) -> dict[str, Any]:
    return {"value": float(value), "unit": unit}


def _add_env_alias(env: dict[str, dict[str, Any]], alias: str, source: str) -> None:
    if source in env and alias not in env:
        env[alias] = dict(env[source])


def _execution_metadata(env: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = env.setdefault(
        "__meta__",
        {
            "value": None,
            "unit": None,
            "data": {
                "alias_resolution_log": [],
                "chained_equality_patches": [],
                "distance_alias_patches": [],
                "target_alias_used": None,
                "vector_sum_mode": None,
                "role_aware_coulomb_used": False,
                "role_aware_geometry_mode": None,
                "target_point": None,
                "target_charge_name": None,
                "source_points": [],
                "role_aware_pair_force_count": 0,
                "coulomb_scene_warnings": [],
                "coulomb_scene_failure_reasons": [],
                "law_of_cosines_used": False,
                "inferred_theta_deg": None,
                "theta_effective_deg": None,
                "charge_interaction_adjustment": None,
                "target_point_inference_source": None,
                "labeled_charge_patches": [],
                "executed_dispatch_names": [],
                "unsupported_dispatch_names": [],
                "role_aware_electric_field_used": False,
                "electric_field_vector_mode": None,
                "electric_field_pair_count": 0,
                "electric_field_scene": None,
                "step9_dispatch_names": [],
                "target_writeback_aliases": [],
                "parsed_function_amplitudes": {},
                "parsed_current_delta": None,
                "formula_output_units": {},
                "formula_alias_canonicalized": False,
            },
        },
    )
    return meta["data"]


def _log_alias(env: dict[str, dict[str, Any]], requested: str, resolved: str) -> None:
    if requested != resolved:
        _execution_metadata(env)["alias_resolution_log"].append({"requested": requested, "resolved": resolved})


def _add_aliases(env: dict[str, dict[str, Any]]) -> None:
    for name, item in list(env.items()):
        unit = item.get("unit")
        if unit == "V":
            if name == "U":
                _add_env_alias(env, "V", "U")
            if name == "V":
                _add_env_alias(env, "U", "V")
        if unit == "F":
            if name == "C":
                _add_env_alias(env, "C_cap", "C")
            if name == "C_cap":
                _add_env_alias(env, "C", "C_cap")
        if unit == "C":
            if name == "Q":
                _add_env_alias(env, "q", "Q")
            if name == "q":
                _add_env_alias(env, "Q", "q")
        if unit == "m":
            if name == "d":
                _add_env_alias(env, "r", "d")
                _add_env_alias(env, "r13", "d")
            if name == "r":
                _add_env_alias(env, "d", "r")
            if name == "d2":
                _add_env_alias(env, "r23", "d2")
            if name == "L":
                _add_env_alias(env, "side", "L")
                _add_env_alias(env, "side_length", "L")
                _add_env_alias(env, "r13", "L")
                _add_env_alias(env, "r23", "L")
    alias_pairs = {
        "AC": "r13", "CA": "r13", "MA": "r13", "AM": "r13",
        "BC": "r23", "CB": "r23", "MB": "r23", "BM": "r23",
        "d13": "r13", "d23": "r23", "r12": "d", "AB": "r12", "BA": "r12",
        "side": "L", "side_length": "L", "a": "L",
        "q0": "q3", "test_charge": "q3", "qC": "q3", "charge_C": "q3", "charge_M": "q3", "M_charge": "q3",
        "qo": "q0",
        "qA": "q1", "charge_A": "q1", "A_charge": "q1",
        "qB": "q2", "charge_B": "q2", "B_charge": "q2",
        "F_2": "F2", "F_b": "F2", "second_force": "F2",
        "angle": "theta",
    }
    for alias, source in alias_pairs.items():
        _add_env_alias(env, alias, source)
        _add_env_alias(env, source, alias)


def _patch_chained_equalities_from_text(env: dict[str, dict[str, Any]], problem_text: str) -> None:
    text = problem_text.replace("×", "x").replace("−", "-")
    charge_unit = r"(?:uC|μC|Î¼C|mC|nC|C)"
    number = r"[-+]?\d*\.?\d+(?:\s*(?:x|X|Ã—)\s*10\^?-?\d+|(?:e|E)-?\d+)?"
    patterns = [
        (("q1", "q2", "q3"), rf"q1\s*=\s*q2\s*=\s*q3\s*=\s*({number})\s*({charge_unit})"),
        (("q1", "q2"), rf"q1\s*=\s*q2\s*=\s*({number})\s*({charge_unit})"),
        (("q2", "q3"), rf"q2\s*=\s*q3\s*=\s*({number})\s*({charge_unit})"),
        (("qA", "qB", "qC"), rf"qA\s*=\s*qB\s*=\s*qC\s*=\s*({number})\s*({charge_unit})"),
    ]
    for names, pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        value = _parse_number(match.group(1))
        value, unit = _convert_to_si(value, match.group(2))
        if value is None:
            continue
        for name in names:
            if name not in env:
                env[name] = _env_item(value, unit)
                _execution_metadata(env)["chained_equality_patches"].append({"name": name, "value": value, "unit": unit})
    if re.search(r"identical charges|same magnitude", text, re.IGNORECASE) and "q" in env:
        for name in ("q1", "q2", "q3"):
            if name not in env:
                env[name] = dict(env["q"])
                _execution_metadata(env)["chained_equality_patches"].append({"name": name, "source": "q"})
    opposite_match = re.search(rf"q1\s*=\s*-\s*q2\s*=\s*({number}|10\^?-?\d+)\s*({charge_unit})", text, re.IGNORECASE)
    if opposite_match:
        value = _parse_number(opposite_match.group(1))
        value, unit = _convert_to_si(value, opposite_match.group(2))
        if value is not None:
            for name, signed_value in (("q1", value), ("q2", -value)):
                if name not in env:
                    env[name] = _env_item(signed_value, unit)
                    _execution_metadata(env)["chained_equality_patches"].append({"name": name, "value": signed_value, "unit": unit})


def _patch_labeled_charges_from_text(env: dict[str, dict[str, Any]], problem_text: str) -> None:
    text = problem_text.replace("Ã—", "x").replace("âˆ’", "-")
    charge_unit = r"(?:uC|μC|µC|Î¼C|ÃŽÂ¼C|mC|nC|C)"
    number = r"[-+]?\d*\.?\d+(?:\s*(?:x|X|×)\s*10\^?-?\d+|(?:e|E)-?\d+)?|10\^?-?\d+"

    def patch(name: str, raw_value: str, raw_unit: str, source: str) -> None:
        value = _parse_number(raw_value)
        value, unit = _convert_to_si(value, raw_unit)
        if value is None:
            return
        key = _normalize_var_name(name)
        if key not in env:
            env[key] = _env_item(value, unit)
            _execution_metadata(env)["labeled_charge_patches"].append({"name": key, "value": value, "unit": unit, "source": source})

    for charge_name, raw_value, raw_unit in re.findall(r"\b(q[A-CM]|q0|qo)\s*=\s*({})\s*({})".format(number, charge_unit), text, re.IGNORECASE):
        normalized = "q0" if charge_name.lower() == "qo" else charge_name[0] + charge_name[1:].upper()
        patch(normalized, raw_value, raw_unit, "labeled_charge_assignment")

    for point, raw_value, raw_unit in re.findall(r"\bcharge\s+at\s+(?:point\s+)?([A-CM])\s+is\s+({})\s*({})".format(number, charge_unit), text, re.IGNORECASE):
        patch(f"q{point.upper()}", raw_value, raw_unit, "charge_at_point")


def _patch_distance_aliases(env: dict[str, dict[str, Any]], world_input: Type2WorldModelInput) -> None:
    text = world_input.problem_text.lower()
    if ("equilateral_triangle" in world_input.conditions or "equilateral" in text):
        for source in ("L", "side", "side_length", "a", "d"):
            if source in env:
                for target in ("r13", "r23", "r12"):
                    if target not in env:
                        env[target] = dict(env[source])
                        _execution_metadata(env)["distance_alias_patches"].append({"name": target, "source": source})
                break
    if "perpendicular bisector" in text:
        base = _resolve_value(env, "AB", aliases=("r12", "d"))
        height = _resolve_value(env, "h", aliases=("d2", "d3"))
        if base is not None and height is not None:
            distance = math.sqrt((base / 2) ** 2 + height ** 2)
            for target in ("r13", "r23"):
                if target not in env:
                    env[target] = _env_item(distance, "m")
                    _execution_metadata(env)["distance_alias_patches"].append({"name": target, "source": "perpendicular_bisector"})


def _build_numeric_environment(world_input: Type2WorldModelInput) -> dict[str, dict[str, Any]]:
    env: dict[str, dict[str, Any]] = {name: dict(value) for name, value in CONSTANTS.items()}
    _execution_metadata(env)
    for name, quantity in world_input.known_quantities.items():
        value = _get_quantity_value(quantity)
        unit = _get_quantity_unit(quantity)
        value, unit = _convert_to_si(value, unit)
        if value is not None:
            env[_normalize_var_name(name)] = _env_item(value, unit)
    _patch_labeled_charges_from_text(env, world_input.problem_text)
    _patch_chained_equalities_from_text(env, world_input.problem_text)
    _add_aliases(env)
    _patch_distance_aliases(env, world_input)
    _add_aliases(env)
    if _resolve_value(env, "k", aliases=("k",)) is None:
        env["k"] = dict(CONSTANTS["k"])
    return env


def _alias_candidates(name: str) -> tuple[str, ...]:
    alias_map = {
        "q1": ("q1", "qA", "charge_A", "A_charge"),
        "q2": ("q2", "qB", "charge_B", "B_charge"),
        "q3": ("q3", "qC", "q0", "qo", "q", "test_charge", "charge_C", "charge_M", "M_charge"),
        "q0": ("q0", "qo", "q3", "q", "test_charge"),
        "qo": ("qo", "q0", "q3", "q", "test_charge"),
        "q": ("q", "q0", "qo", "q3", "test_charge"),
        "qA": ("qA", "q1"),
        "qB": ("qB", "q2"),
        "qC": ("qC", "q3", "q0", "qo", "q", "test_charge"),
        "r13": ("r13", "AC", "CA", "MA", "AM", "distance_to_q1", "d13"),
        "r23": ("r23", "BC", "CB", "MB", "BM", "distance_to_q2", "d23"),
        "r12": ("r12", "AB", "BA", "d", "distance_between_q1_q2"),
        "r1": ("r1", "r_1", "d1", "distance1", "AM", "AC", "AO", "AH", "r13"),
        "r2": ("r2", "r_2", "d2", "distance2", "BM", "BC", "BO", "BH", "r23"),
        "distance": ("distance", "d", "r"),
        "side": ("L", "side", "side_length", "a"),
        "E": ("E", "E_net", "E_total", "electric_field"),
        "E1": ("E1", "E_1", "E_A", "E_B"),
        "E2": ("E2", "E_2", "E_B", "E_C"),
        "V": ("V", "U", "voltage", "V_rms", "U_C"),
        "U": ("U", "V", "voltage"),
        "R": ("R", "resistance", "Z"),
        "R_total": ("R_total", "Rtotal", "R_eq"),
        "I": ("I", "current", "I_rms", "I_max"),
        "P": ("P", "power", "P_total"),
        "P_total": ("P_total", "P", "total_power"),
        "C_cap": ("C_cap", "C", "capacitance"),
        "L_ind": ("L_ind", "L", "inductance"),
        "L": ("L", "L_ind", "inductance"),
        "f": ("f", "frequency", "f_res", "f_osc"),
        "omega": ("omega", "omega_0"),
        "omega_0": ("omega_0", "omega"),
        "Z": ("Z", "R", "impedance"),
        "X_L": ("X_L", "XL"),
        "X_C": ("X_C", "XC"),
        "U_B": ("U_B", "W_L", "WL"),
        "U_E": ("U_E", "W_C", "WC", "U_cap"),
        "U_cap": ("U_cap", "U_E", "U_C"),
        "epsilon_r": ("epsilon_r", "eps_r", "kappa", "dielectric_constant"),
        "epsilon_0": ("epsilon_0", "eps0"),
        "A": ("A", "area"),
        "mean_value": ("mean_value", "mean", "average"),
        "abs_error": ("abs_error", "absolute_error", "delta"),
        "rel_error": ("rel_error", "relative_error"),
        "percent_error": ("percent_error", "percentage_error"),
        "F2": ("F2", "F_2", "F_b", "second_force"),
        "theta": ("theta", "angle"),
    }
    return alias_map.get(name, (name,))


def _resolve_value(env: dict[str, dict[str, Any]], name: str, aliases: tuple[str, ...] | None = None) -> float | None:
    candidates = aliases or _alias_candidates(name)
    for candidate in candidates:
        key = _normalize_var_name(candidate)
        if key in env and key != "__meta__" and env[key].get("value") is not None:
            _log_alias(env, name, key)
            return float(env[key]["value"])
    return None


def _resolve_distance(env: dict[str, dict[str, Any]], name: str, world_input: Type2WorldModelInput) -> float | None:
    value = _resolve_value(env, name)
    if value is not None:
        return value
    text = world_input.problem_text.lower()
    if ("equilateral_triangle" in world_input.conditions or "equilateral" in text) and name in {"r13", "r23", "r12"}:
        return _resolve_value(env, "side")
    return None


def _resolve_target_value(env: dict[str, dict[str, Any]], target: str | None, target_unit: str | None) -> tuple[float | None, str | None]:
    if not target:
        return None, target_unit
    candidates: list[str] = [target]
    normalized_target = _normalize_var_name(target)
    compatible_aliases = {
        "F_e": ("F_e", "F_net", "F_on_q3", "F"),
        "F_net": ("F_net", "F_e", "F_on_q3", "F"),
        "F_on_q3": ("F_on_q3", "F_net", "F_e", "F"),
        "E": ("E", "E_net", "E_total", "electric_field"),
        "E_net": ("E_net", "E", "E_total", "electric_field"),
        "E_total": ("E_total", "E_net", "E", "electric_field"),
        "electric_field": ("electric_field", "E", "E_net", "E_total"),
        "U_cap": ("U_cap", "U"),
        "U_B": ("U_B",),
        "U_E": ("U_E", "U_cap"),
        "U_C": ("U_C", "V", "U_cap"),
        "C_cap": ("C_cap", "C"),
        "C": ("C", "C_cap"),
        "Q": ("Q", "q"),
        "q": ("q", "Q"),
        "V": ("V", "U"),
        "U": ("U", "V"),
        "I": ("I", "current", "I_rms", "I_max"),
        "R": ("R", "Z"),
        "Z": ("Z", "R"),
        "L_ind": ("L_ind", "L"),
        "L": ("L", "L_ind"),
        "P": ("P", "P_total"),
        "P_total": ("P_total", "P"),
        # Step 9 cross-name writeback aliases.
        "f": ("f", "f_res", "f_osc", "frequency"),
        "f_res": ("f_res", "f_osc", "f", "frequency"),
        "f_osc": ("f_osc", "f_res", "f", "frequency"),
        "omega": ("omega", "omega_0"),
        "omega_0": ("omega_0", "omega"),
        "T_osc": ("T_osc", "T", "period"),
        "U_cap": ("U_cap", "U_E", "U_total", "U"),
        "U_E": ("U_E", "U_cap", "U_total"),
        "U_total": ("U_total", "U_E", "U_cap"),
        "U_B": ("U_B", "W_L"),
        "U_C": ("U_C", "V", "U_cap"),
        "I_max": ("I_max", "I"),
        "I_rms": ("I_rms", "I"),
        "n_turns_per_meter": ("n_turns_per_meter", "n"),
        "Phi_B": ("Phi_B", "Phi"),
        "emf": ("emf",),
        "B": ("B",),
        "X_L": ("X_L", "XL"),
        "X_C": ("X_C", "XC"),
        "Z_2": ("Z_2", "Z2"),
        "power_factor": ("power_factor", "cos_phi"),
        "Q_factor": ("Q_factor", "quality_factor"),
    }
    candidates.extend(compatible_aliases.get(normalized_target, ()))
    for candidate in candidates:
        key = _normalize_var_name(candidate)
        if key not in env or key == "__meta__" or env[key].get("value") is None:
            continue
        unit = env[key].get("unit") or target_unit
        if normalized_target == "U_cap" and unit != "J":
            continue
        if normalized_target == "C_cap" and unit != "F":
            continue
        if normalized_target in {"Q", "q"} and unit != "C":
            continue
        if normalized_target.startswith("F") and unit != "N":
            continue
        if normalized_target in {"E", "E_net", "E_total", "electric_field"} and unit not in {"V/m", "N/C"}:
            continue
        if normalized_target in {"V", "U", "U_C", "V_after"} and target_unit and _canonical_unit(target_unit) == "V" and unit != "V":
            continue
        if normalized_target in {"U_cap", "U_B", "U_E"} and target_unit and _canonical_unit(target_unit) == "J" and unit != "J":
            continue
        if key != normalized_target:
            _execution_metadata(env)["target_alias_used"] = {"target": normalized_target, "alias": key}
            wb = _execution_metadata(env)["target_writeback_aliases"]
            entry = {"target": normalized_target, "alias": key}
            if entry not in wb:
                wb.append(entry)
            _log_alias(env, normalized_target, key)
        return float(env[key]["value"]), unit
    return None, target_unit


def _charge_value(env: dict[str, dict[str, Any]], name: str) -> float | None:
    return _resolve_value(env, name, aliases=(name, *_alias_candidates(name)))


def _charge_name_exists(env: dict[str, dict[str, Any]], name: str) -> bool:
    return _charge_value(env, name) is not None


def _extract_charge_point_assignments(problem_text: str, env: dict[str, dict[str, Any]]) -> dict[str, str]:
    text = problem_text
    text_l = text.lower()
    assignments: dict[str, str] = {}

    for point, charge in (("A", "qA"), ("B", "qB"), ("C", "qC")):
        if _charge_name_exists(env, charge) and re.search(rf"\b{charge}\b", text, re.IGNORECASE):
            assignments[point] = charge

    if re.search(r"q1.*q2.*q3.*(?:points?|vertices).*a.*b.*c|vertices\s+of.*triangle\s+abc|points?\s+a\s*(?:,|and)?\s*b\s*(?:,|and)?\s*c", text_l, re.IGNORECASE | re.DOTALL):
        for point, charge in (("A", "q1"), ("B", "q2"), ("C", "q3")):
            if _charge_name_exists(env, charge):
                assignments.setdefault(point, charge)
    elif re.search(r"q1.*q2.*(?:points?|vertices).*a.*b|points?\s+a\s*(?:and|,)\s*b", text_l, re.IGNORECASE | re.DOTALL):
        for point, charge in (("A", "q1"), ("B", "q2")):
            if _charge_name_exists(env, charge):
                assignments.setdefault(point, charge)

    for charge, point in re.findall(r"\b(q[0-9abc]|qo|q)\b\s*(?:is\s+)?(?:placed|located|at)\s+(?:at\s+)?(?:point\s+)?([A-Z])\b", text, re.IGNORECASE):
        charge_name = "q0" if charge.lower() == "qo" else charge[0] + charge[1:].upper() if len(charge) == 2 and charge[1].isalpha() and charge[1].islower() else charge
        if charge_name == "q":
            charge_name = "q"
        if _charge_name_exists(env, charge_name):
            assignments[point.upper()] = charge_name

    for charge, point in re.findall(r"\b(q[0-9abc]|qo|q)\b\s*=\s*[^.?,;]+?\s+(?:is\s+)?(?:placed|located)\s+at\s+(?:point\s+)?([A-Z])\b", text, re.IGNORECASE):
        charge_name = "q0" if charge.lower() == "qo" else charge[0] + charge[1:].upper() if len(charge) == 2 and charge[1].isalpha() and charge[1].islower() else charge
        if _charge_name_exists(env, charge_name):
            assignments[point.upper()] = charge_name

    for point, charge in re.findall(r"\bcharge\s+(q[0-9abc]|qo|q)\s+(?:is\s+)?(?:placed|located)\s+at\s+(?:point\s+)?([A-Z])\b", text, re.IGNORECASE):
        charge_name = "q0" if charge.lower() == "qo" else charge[0] + charge[1:].upper() if len(charge) == 2 and charge[1].isalpha() and charge[1].islower() else charge
        if _charge_name_exists(env, charge_name):
            assignments[point.upper()] = charge_name

    if re.search(r"\b(?:midpoint|center|centre)\b", text_l):
        for point in ("M", "H", "O"):
            if re.search(rf"\b{point.lower()}\b", text_l) and _charge_name_exists(env, "q0") and re.search(r"\bq0\b|\bqo\b", text_l):
                assignments.setdefault(point, "q0")
            elif re.search(rf"\b{point.lower()}\b", text_l) and _charge_name_exists(env, "q") and re.search(r"\btest charge\b|\bcharge q\b|\bq\s*=", text_l):
                assignments.setdefault(point, "q")

    if _charge_name_exists(env, "q0") and re.search(r"\b(?:midpoint|center|centre)\b", text_l) and re.search(r"\bq0\b|\bqo\b", text_l):
        assignments.setdefault("M", "q0")
    elif _charge_name_exists(env, "q") and re.search(r"\btest charge|charge q\b|q\s*=", text_l) and re.search(r"\b(?:midpoint|center|centre)\b", text_l):
        assignments.setdefault("M", "q")

    if not assignments and all(label in text_l for label in ("q1", "q2", "q3")):
        assignments = {"q1": "q1", "q2": "q2", "q3": "q3"}
    elif not assignments and all(label in text_l for label in ("q1", "q2")) and re.search(r"\bq0\b|\bqo\b|\btest charge\b", text_l):
        assignments = {"q1": "q1", "q2": "q2", "M": "q0"}

    return assignments


def _point_for_charge(assignments: dict[str, str], charge_name: str) -> str | None:
    normalized = _normalize_var_name(charge_name).lower()
    exact_aliases = {
        "qo": {"q0", "qo"},
        "q0": {"q0", "qo"},
        "q": {"q", "q0", "qo"},
        "qa": {"qa", "q1"},
        "qb": {"qb", "q2"},
        "qc": {"qc", "q3"},
    }
    aliases = exact_aliases.get(normalized, {normalized})
    for point, assigned_charge in assignments.items():
        if _normalize_var_name(assigned_charge).lower() in aliases:
            return point
    return None


def _infer_target_point(problem_text: str, env: dict[str, dict[str, Any]], assignments: dict[str, str], target: str | None = None) -> str | None:
    text = problem_text
    text_l = text.lower()
    target_l = (target or "").lower()
    meta = _execution_metadata(env)

    for point in re.findall(r"(?:force|net force|electric force|resultant force)[^.?\n]*(?:acting|exerted)?\s*(?:on|upon)\s+(?:the\s+)?charge\s+at\s+(?:point\s+)?([A-Z])\b", text, re.IGNORECASE):
        if point.upper() in assignments:
            meta["target_point_inference_source"] = "force_on_point_pattern"
            return point.upper()

    charge_patterns = [
        r"(?:force|net force|electric force|resultant force)[^.?\n]*(?:acting|exerted)?\s*(?:on|upon)\s+(?:(?:a|the)\s+)?(?:(?:test|electric)\s+)?(?:charge\s+)?(q[0-9abc]|qo|q)\b",
        r"exerted\s+by\s+q\d\s+and\s+q\d\s+on\s+(q[0-9abc]|qo|q)\b",
        r"\bacting\s+on\s+(q[0-9abc]|qo|q)\b",
        r"\bon\s+(q[0-9abc]|qo|q)\b",
    ]
    for pattern in charge_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            continue
        charge = matches[-1]
        charge_name = "q0" if charge.lower() == "qo" else charge[0] + charge[1:].upper() if len(charge) == 2 and charge[1].isalpha() and charge[1].islower() else charge
        point = _point_for_charge(assignments, charge_name)
        if point:
            meta["target_point_inference_source"] = "force_on_charge_pattern"
            return point

    if "f_on_q3" in target_l:
        point = _point_for_charge(assignments, "q3")
        if point:
            meta["target_point_inference_source"] = "target_alias_f_on_q3"
            return point
    if "q0" in target_l or "qo" in target_l:
        point = _point_for_charge(assignments, "q0")
        if point:
            meta["target_point_inference_source"] = "target_alias_q0"
            return point

    for phrase_point in ("A", "B", "C", "M", "H", "O"):
        if re.search(rf"(?:acting|exerted)\s+on\s+(?:the\s+)?charge\s+at\s+(?:point\s+)?{phrase_point.lower()}\b", text_l) and phrase_point in assignments:
            meta["target_point_inference_source"] = "force_on_point_pattern"
            return phrase_point

    if re.search(r"\btest charge\b|\bcharge q\b|\bq0\b|\bqo\b", text_l):
        for charge in ("q0", "q"):
            point = _point_for_charge(assignments, charge)
            if point:
                meta["target_point_inference_source"] = "test_charge_pattern"
                return point
    return None


def _infer_distances_by_pair(problem_text: str, env: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    text_l = problem_text.lower()
    distances: dict[str, dict[str, Any]] = {}
    pairs = ("AB", "AC", "BC", "AM", "BM", "CM", "AH", "BH", "CH", "AO", "BO", "CO")

    def add_pair(pair: str, value: float | None, source: str) -> None:
        if value is None:
            return
        distances[pair] = {"name": pair, "value": value, "source": source}
        distances[pair[::-1]] = {"name": pair[::-1], "value": value, "source": source}

    distance_unit = r"(?:cm|mm|m)"
    number = r"[-+]?\d*\.?\d+(?:\s*(?:x|X|×)\s*10\^?-?\d+|(?:e|E)-?\d+)?|10\^?-?\d+"
    for pair, raw_value, raw_unit in re.findall(rf"\b([A-Z]{{2}})\s*=\s*({number})\s*({distance_unit})", problem_text, re.IGNORECASE):
        pair = pair.upper()
        if pair in pairs or pair[::-1] in pairs:
            value = _parse_number(raw_value)
            value, _ = _convert_to_si(value, raw_unit)
            add_pair(pair, value, "text_pair_distance")

    for pair in pairs:
        value = _resolve_value(env, pair, aliases=(pair, pair[::-1], pair.lower(), pair[::-1].lower()))
        if value is not None and pair not in distances:
            add_pair(pair, value, pair)

    ab = distances.get("AB", {}).get("value")
    ac = distances.get("AC", {}).get("value")
    bc = distances.get("BC", {}).get("value")
    if "right" in text_l and " at a" in text_l:
        if ab is not None and bc is not None and bc > ab and ac is None:
            add_pair("AC", math.sqrt(max(bc * bc - ab * ab, 0.0)), "right_triangle_completion")
        elif ac is not None and bc is not None and bc > ac and ab is None:
            add_pair("AB", math.sqrt(max(bc * bc - ac * ac, 0.0)), "right_triangle_completion")
        elif ab is not None and ac is not None and bc is None:
            add_pair("BC", math.sqrt(ab * ab + ac * ac), "right_triangle_completion")

    ab = distances.get("AB", {}).get("value")
    if ab is None:
        ab = _resolve_value(env, "r12", aliases=("r12", "d", "r", "a", "L"))
    if ab is not None and re.search(r"\bmidpoint\b|\bcenter\b|\bcentre\b", text_l):
        for point in ("M", "H", "O"):
            if point.lower() in text_l or point == "M":
                add_pair(f"A{point}", ab / 2, "midpoint_AB")
                add_pair(f"B{point}", ab / 2, "midpoint_AB")

    if all(label in text_l for label in ("q1", "q2", "q3")):
        d = _resolve_value(env, "r12", aliases=("r12", "d", "r", "a", "L"))
        if d is not None:
            if "straight line" in text_l and "apart" in text_l:
                add_pair("q1q2", d, "linear_adjacent")
                add_pair("q2q3", d, "linear_adjacent")
                add_pair("q1q3", d * 2, "linear_outer")
            if "midpoint" in text_l:
                add_pair("q1q3", d / 2, "midpoint_q1q2")
                add_pair("q2q3", d / 2, "midpoint_q1q2")

    side = _resolve_value(env, "side", aliases=("a", "L", "side", "side_length", "d"))
    if side is not None and ("isosceles right" in text_l or "legs" in text_l):
        add_pair("q1q3", side, "isosceles_right_leg")
        add_pair("q2q3", side, "isosceles_right_leg")
    if side is not None and "equidistant from a and b" in text_l:
        add_pair("AM", side, "equidistant")
        add_pair("BM", side, "equidistant")

    return distances


def _infer_coulomb_geometry(problem_text: str, world_input: Type2WorldModelInput) -> str:
    text_l = problem_text.lower()
    conditions = " ".join(world_input.conditions).lower()
    combined = f"{text_l} {conditions}"
    if "perpendicular bisector" in combined:
        return "perpendicular_bisector"
    if "equilateral" in combined:
        return "equilateral"
    if "right" in combined or "perpendicular" in combined or "90" in combined:
        return "right_angle"
    if "same direction" in combined:
        return "same_direction"
    if "collinear" in combined or "straight line" in combined or "opposite sides" in combined or "midpoint" in combined:
        return "collinear"
    return "unknown"


def _lookup_distance_for_points(source_point: str, target_point: str, distances_by_pair: dict[str, dict[str, Any]]) -> tuple[str | None, float | None]:
    pair = f"{source_point}{target_point}"
    item = distances_by_pair.get(pair) or distances_by_pair.get(pair[::-1])
    if item:
        return str(item.get("name") or pair), float(item["value"])
    return None, None


def _infer_angle_between_pair_forces(scene: dict[str, Any], world_input: Type2WorldModelInput, env: dict[str, dict[str, Any]]) -> tuple[float | None, str | None]:
    pair_forces = scene.get("pair_forces", []) or []
    if len(pair_forces) != 2:
        return None, None
    source_1 = str(pair_forces[0].get("source_point") or "")
    source_2 = str(pair_forces[1].get("source_point") or "")
    target_point = str(scene.get("target_point") or "")
    distances = scene.get("metadata", {}).get("distances_by_pair", {}) or {}
    _, d1 = _lookup_distance_for_points(source_1, target_point, distances)
    _, d2 = _lookup_distance_for_points(source_2, target_point, distances)
    _, d12 = _lookup_distance_for_points(source_1, source_2, distances)
    if d1 is None or d2 is None or d12 is None or d1 <= 0 or d2 <= 0 or d12 <= 0:
        return None, None
    denominator = 2 * d1 * d2
    if denominator == 0:
        return None, None
    cos_theta = (d1 * d1 + d2 * d2 - d12 * d12) / denominator
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    return theta, f"law_of_cosines_angle_at_{target_point}"


def extract_coulomb_scene(world_input: Type2WorldModelInput, env: dict[str, dict[str, Any]]) -> dict[str, Any]:
    assignments = _extract_charge_point_assignments(world_input.problem_text, env)
    target_point = _infer_target_point(world_input.problem_text, env, assignments, world_input.target)
    distances_by_pair = _infer_distances_by_pair(world_input.problem_text, env)
    geometry = _infer_coulomb_geometry(world_input.problem_text, world_input)
    warnings: list[str] = []
    failure_reasons: list[str] = []

    if not assignments:
        failure_reasons.append("no_charge_point_assignments")
    if not target_point:
        failure_reasons.append("target_point_unresolved")
    if target_point and target_point not in assignments:
        failure_reasons.append("target_charge_unresolved")

    target_charge_name = assignments.get(target_point or "")
    target_charge_value = _charge_value(env, target_charge_name) if target_charge_name else None
    if target_charge_name and target_charge_value is None:
        failure_reasons.append("target_charge_value_missing")

    source_points = [
        point
        for point, charge_name in assignments.items()
        if point != target_point and _normalize_var_name(charge_name).lower() != _normalize_var_name(target_charge_name).lower()
    ]
    pair_forces: list[dict[str, Any]] = []
    k = _resolve_value(env, "k", aliases=("k",))
    if k is None:
        failure_reasons.append("k_missing")

    if target_point and target_charge_name and target_charge_value is not None and k is not None:
        for source_point in source_points:
            source_charge_name = assignments[source_point]
            source_charge_value = _charge_value(env, source_charge_name)
            distance_name, distance_value = _lookup_distance_for_points(source_point, target_point, distances_by_pair)
            if source_charge_value is None:
                warnings.append(f"Missing charge value for source point {source_point}.")
                continue
            if distance_value is None or distance_value == 0:
                warnings.append(f"Missing distance between source point {source_point} and target point {target_point}.")
                continue
            force_value = k * abs(source_charge_value * target_charge_value) / (distance_value * distance_value)
            pair_forces.append(
                {
                    "source_point": source_point,
                    "target_point": target_point,
                    "source_charge_name": source_charge_name,
                    "target_charge_name": target_charge_name,
                    "distance_name": distance_name,
                    "distance_value": distance_value,
                    "force_name": f"F_{source_point}{target_point}",
                    "force_value": force_value,
                    "source_charge_value": source_charge_value,
                    "target_charge_value": target_charge_value,
                }
            )

    if not pair_forces and not failure_reasons:
        failure_reasons.append("no_pair_forces_executed")

    return {
        "target_point": target_point,
        "target_charge_name": target_charge_name,
        "target_charge_value": target_charge_value,
        "source_points": source_points,
        "pair_forces": pair_forces,
        "geometry": geometry,
        "warnings": warnings,
        "metadata": {
            "charges_by_point": dict(assignments),
            "distances_by_pair": distances_by_pair,
            "failure_reasons": failure_reasons,
        },
    }


def _combine_role_aware_pair_forces(scene: dict[str, Any], world_input: Type2WorldModelInput, env: dict[str, dict[str, Any]]) -> tuple[float | None, str, str | None]:
    pair_forces = scene.get("pair_forces", []) or []
    geometry = str(scene.get("geometry") or "unknown")
    if len(pair_forces) != 2:
        return None, geometry, "Role-aware Coulomb scene extracted pair forces, but geometry is insufficient for final vector magnitude."
    f1 = float(pair_forces[0]["force_value"])
    f2 = float(pair_forces[1]["force_value"])
    if geometry == "right_angle":
        return math.sqrt(f1 * f1 + f2 * f2), geometry, None
    if geometry == "equilateral":
        return math.sqrt(f1 * f1 + f2 * f2 + 2 * f1 * f2 * math.cos(math.radians(60))), geometry, None
    if geometry in {"collinear", "perpendicular_bisector"}:
        q1 = float(pair_forces[0]["source_charge_value"])
        q2 = float(pair_forces[1]["source_charge_value"])
        if q1 * q2 < 0:
            return f1 + f2, "same_direction", None
        return abs(f1 - f2), "collinear_opposite", None
    if geometry == "same_direction":
        return f1 + f2, geometry, None
    theta, theta_mode = _infer_angle_between_pair_forces(scene, world_input, env)
    if theta is not None:
        target_value = float(pair_forces[0]["target_charge_value"])
        relation_1 = float(pair_forces[0]["source_charge_value"]) * target_value
        relation_2 = float(pair_forces[1]["source_charge_value"]) * target_value
        same_interaction_type = relation_1 * relation_2 > 0
        theta_effective = theta if same_interaction_type else math.pi - theta
        adjustment = "same_interaction_type" if same_interaction_type else "opposite_interaction_type_pi_minus_theta"
        value = math.sqrt(max(f1 * f1 + f2 * f2 + 2 * f1 * f2 * math.cos(theta_effective), 0.0))
        metadata = _execution_metadata(env)
        metadata["law_of_cosines_used"] = True
        metadata["vector_sum_mode"] = "law_of_cosines"
        metadata["inferred_theta_rad"] = theta
        metadata["inferred_theta_deg"] = math.degrees(theta)
        metadata["theta_effective_rad"] = theta_effective
        metadata["theta_effective_deg"] = math.degrees(theta_effective)
        metadata["charge_interaction_adjustment"] = adjustment
        return value, theta_mode or "law_of_cosines", "Using geometric line angle for force magnitude; charge signs may affect direction."
    return None, geometry, "Role-aware Coulomb scene extracted pair forces, but geometry is insufficient for final vector magnitude."


def _is_force_like_target(target: str | None) -> bool:
    return bool(target and _normalize_var_name(target).lower().startswith("f"))


def _write_force_aliases(env: dict[str, dict[str, Any]], target: str | None, value: float) -> None:
    names = [target] if target else []
    names.extend(["F_net", "F_e", "F_on_q3"])
    for name in names:
        if not name:
            continue
        key = _normalize_var_name(name)
        if key not in env or env[key].get("unit") == "N":
            env[key] = _env_item(value, "N")


def _execute_role_aware_coulomb_scene(world_input: Type2WorldModelInput, env: dict[str, dict[str, Any]], target: str | None) -> Type2ExecutionStepResult:
    scene = extract_coulomb_scene(world_input, env)
    final_force, geometry_mode, combine_warning = _combine_role_aware_pair_forces(scene, world_input, env)
    metadata = _execution_metadata(env)
    metadata["role_aware_coulomb_used"] = True
    metadata["role_aware_geometry_mode"] = geometry_mode
    metadata["target_point"] = scene.get("target_point")
    metadata["target_charge_name"] = scene.get("target_charge_name")
    metadata["source_points"] = list(scene.get("source_points", []) or [])
    metadata["role_aware_pair_force_count"] = len(scene.get("pair_forces", []) or [])
    metadata["coulomb_scene_warnings"] = list(scene.get("warnings", []) or [])
    metadata["coulomb_scene_failure_reasons"] = list(scene.get("metadata", {}).get("failure_reasons", []) or [])
    metadata["coulomb_scene"] = scene

    outputs: dict[str, Any] = {}
    for index, pair_force in enumerate(scene.get("pair_forces", []) or [], start=1):
        force_name = str(pair_force["force_name"])
        force_value = float(pair_force["force_value"])
        _set(env, force_name, force_value, "N")
        outputs[force_name] = force_value
        if index == 1:
            _set(env, "F_13", force_value, "N")
        elif index == 2:
            _set(env, "F_23", force_value, "N")

    warnings = list(scene.get("warnings", []) or [])
    if combine_warning:
        warnings.append(combine_warning)
    if final_force is not None:
        _write_force_aliases(env, target or world_input.target, final_force)
        outputs[target or world_input.target or "F_net"] = final_force
        return _step_result(
            {"step_id": "role_aware_coulomb_scene", "type": "formula_application", "formula_name": "role_aware_coulomb_scene", "template_name": "role_aware_coulomb"},
            "PASS",
            {},
            outputs,
            final_force,
            "N",
            warnings=warnings,
            metadata={"role_aware_coulomb_used": True, "role_aware_geometry_mode": geometry_mode},
        )
    status = "WARN" if outputs else "FAIL"
    errors = []
    if not outputs:
        errors.append(
            {
                "error_type": "coulomb_scene_unresolved",
                "severity": "medium",
                "message": "Role-aware Coulomb scene could not execute pair forces.",
                "failure_reasons": metadata["coulomb_scene_failure_reasons"],
            }
        )
    return _step_result(
        {"step_id": "role_aware_coulomb_scene", "type": "formula_application", "formula_name": "role_aware_coulomb_scene", "template_name": "role_aware_coulomb"},
        status,
        {},
        outputs,
        None,
        "N",
        warnings=warnings,
        errors=errors,
        metadata={"role_aware_coulomb_used": True, "role_aware_geometry_mode": geometry_mode},
    )


def _plan_has_coulomb_pairwise(selected_step_plan: list[dict[str, Any]]) -> bool:
    for step in selected_step_plan:
        if not isinstance(step, dict):
            continue
        text = f"{step.get('formula_name') or ''} {step.get('template_name') or ''}".lower()
        if "coulomb" in text or "q1*q3" in text or "q2*q3" in text or "vector_sum" in text:
            return True
    return False


def _has_coulomb_missing_errors(errors: list[dict[str, Any]]) -> bool:
    missing_names = {"q1", "q2", "q3", "q0", "q", "r", "r13", "r23"}
    for error in errors:
        if error.get("error_type") != "missing_inputs":
            continue
        if any(str(name) in missing_names for name in error.get("missing_vars", []) or []):
            return True
    return False


def _value(env: dict[str, dict[str, Any]], *names: str) -> float | None:
    for name in names:
        value = _resolve_value(env, name)
        if value is not None:
            return value
    return None


def _unit(env: dict[str, dict[str, Any]], *names: str) -> str | None:
    for name in names:
        key = _normalize_var_name(name)
        if key in env:
            return env[key].get("unit")
    return None


def _set(env: dict[str, dict[str, Any]], name: str, value: float, unit: str | None) -> None:
    key = _normalize_var_name(name)
    env[key] = _env_item(value, unit)
    if unit == "V" and key in {"V", "U", "voltage", "U_C", "V_after"}:
        for alias in ("V", "U", "voltage"):
            if alias not in env:
                env[alias] = _env_item(value, unit)
    if unit == "F" and key in {"C", "C_cap", "C_after", "capacitance"}:
        for alias in ("C", "C_cap"):
            if alias not in env:
                env[alias] = _env_item(value, unit)
    if unit == "H" and key in {"L", "L_ind", "inductance"}:
        for alias in ("L", "L_ind"):
            if alias not in env:
                env[alias] = _env_item(value, unit)
    if unit in {"V/m", "N/C"} and key in {"E", "E_net", "E_total", "electric_field"}:
        for alias in ("E", "E_net", "E_total", "electric_field"):
            if alias not in env:
                env[alias] = _env_item(value, unit)
    if unit == "W" and key in {"P", "P_total", "power"}:
        for alias in ("P", "P_total"):
            if alias not in env:
                env[alias] = _env_item(value, unit)
    _add_aliases(env)


def _formula_text(step: dict[str, Any]) -> str:
    return str(step.get("formula_name") or "").replace("**", "^")


def _target_output(step: dict[str, Any]) -> str | None:
    output_var = step.get("output_var")
    if isinstance(output_var, dict) and output_var:
        return str(next(iter(output_var.keys())))
    return None


def _formula_lhs(step: dict[str, Any]) -> str | None:
    formula = str(step.get("formula_name") or "")
    if "=" not in formula:
        return None
    return formula.split("=", 1)[0].strip()


def _force_output_name(output: str | None, fallback: str = "F") -> str:
    return output if output in {"F", "F_e", "F_net", "F_on_q3", "F_13", "F_23"} else fallback


def _has_condition(world_input: Type2WorldModelInput, *terms: str) -> bool:
    blob = " ".join([world_input.problem_text] + world_input.conditions + world_input.domains + world_input.sub_domains).lower()
    return any(term.lower() in blob for term in terms)


def _infer_output_unit(output_var: str | None, formula_name: str | None, target_unit: str | None = None) -> str | None:
    if target_unit:
        return _canonical_unit(target_unit)
    name = _normalize_var_name(output_var).lower()
    formula = str(formula_name or "").lower()
    if name.startswith("f") or "force" in formula:
        return "N"
    if name in {"e", "e_net", "e_total", "electric_field"} or formula.startswith("e=") or "electric field" in formula:
        return "V/m"
    if name in {"u_c", "v", "v_after"} and "sqrt" in formula:
        return "V"
    if name.startswith("u") or "energy" in formula:
        return "J"
    if name in {"c", "c_cap", "c_after"}:
        return "F"
    if name.startswith("q"):
        return "C"
    if name in {"v", "u", "v_after"}:
        return "V"
    if name.startswith("i"):
        return "A"
    if name.startswith("p"):
        return "W"
    if name in {"l", "l_ind"}:
        return "H"
    if name in {"f", "f_osc", "f_res"}:
        return "Hz"
    if name == "t_osc":
        return "s"
    if name.startswith("omega"):
        return "rad/s"
    if name in {"z", "x_l", "x_c"}:
        return "ohm"
    if name.startswith("r"):
        return "Ω"
    return None


def _format_answer(value: float, unit: str | None) -> str:
    abs_value = abs(value)
    if abs_value != 0 and (abs_value < 1e-4 or abs_value >= 1e6):
        text = f"{value:.6g}"
    else:
        text = f"{value:.6g}"
    return f"{text} {unit}" if unit else text


def _step_result(
    step: dict[str, Any],
    status: str,
    input_values: dict[str, Any],
    output_values: dict[str, Any],
    numeric_value: float | None,
    unit: str | None,
    warnings: list[str] | None = None,
    errors: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Type2ExecutionStepResult:
    return Type2ExecutionStepResult(
        step_id=str(step.get("step_id") or ""),
        formula_name=step.get("formula_name"),
        template_name=step.get("template_name"),
        status=status,
        input_values=input_values,
        output_values=output_values,
        numeric_value=numeric_value,
        unit=unit,
        warnings=warnings or [],
        errors=errors or [],
        metadata=metadata or {},
    )


def _inputs_for_step(step: dict[str, Any], env: dict[str, dict[str, Any]]) -> dict[str, Any]:
    input_var = step.get("input_var") or {}
    keys = list(input_var.keys()) if isinstance(input_var, dict) else list(input_var or [])
    return {str(key): env.get(_normalize_var_name(key), {}).get("value") for key in keys}


def _close_aliases(env: dict[str, dict[str, Any]], missing_vars: list[str]) -> dict[str, list[str]]:
    available = [key for key in env if key != "__meta__"]
    close: dict[str, list[str]] = {}
    for var in missing_vars:
        aliases = list(_alias_candidates(var))
        close[var] = [alias for alias in aliases if _normalize_var_name(alias) in available]
    return close


def _missing_error(message: str, missing_vars: list[str] | None = None, formula_name: str | None = None, env: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    missing_vars = missing_vars or []
    return {
        "error_type": "missing_inputs",
        "severity": "medium",
        "message": message,
        "missing_vars": missing_vars,
        "formula_name": formula_name,
        "available_close_aliases": _close_aliases(env or {}, missing_vars),
    }


def _missing_from_values(named_values: dict[str, float | None], env: dict[str, dict[str, Any]], formula_name: str, message: str) -> dict[str, Any]:
    missing_vars = [name for name, value in named_values.items() if value is None]
    return _missing_error(message, missing_vars, formula_name, env)


def _mark_executed(env: dict[str, dict[str, Any]], formula_name: str) -> None:
    names = _execution_metadata(env)["executed_dispatch_names"]
    if formula_name not in names:
        names.append(formula_name)


def _mark_unsupported(env: dict[str, dict[str, Any]], formula_name: str) -> None:
    names = _execution_metadata(env)["unsupported_dispatch_names"]
    if formula_name not in names:
        names.append(formula_name)


def _v(env: dict[str, dict[str, Any]]) -> float | None:
    return _value(env, "V", "U", "voltage", "V_rms")


def _r(env: dict[str, dict[str, Any]], *names: str) -> float | None:
    return _value(env, *(names or ("R", "resistance")))


def _i(env: dict[str, dict[str, Any]], *names: str) -> float | None:
    return _value(env, *(names or ("I", "current", "I_rms")))


def _cap(env: dict[str, dict[str, Any]]) -> float | None:
    return _value(env, "C_cap", "C", "capacitance")


def _ind(env: dict[str, dict[str, Any]]) -> float | None:
    return _value(env, "L_ind", "L", "inductance")


def _freq(env: dict[str, dict[str, Any]]) -> float | None:
    return _value(env, "f", "frequency", "f_res", "f_osc")


def _safe_sqrt(value: float) -> float:
    return math.sqrt(max(value, 0.0))


def _numeric_list_from_env(env: dict[str, dict[str, Any]], prefix: str) -> list[float]:
    values: list[float] = []
    for key, item in env.items():
        if key == "__meta__":
            continue
        if key.lower().startswith(prefix.lower()) and item.get("value") is not None:
            values.append(float(item["value"]))
    return values


def _mark_step9(env: dict[str, dict[str, Any]], formula: str) -> None:
    names = _execution_metadata(env)["step9_dispatch_names"]
    if formula not in names:
        names.append(formula)


def _record_output_unit(env: dict[str, dict[str, Any]], name: Any, unit: str | None) -> None:
    if name and unit:
        _execution_metadata(env)["formula_output_units"][str(name)] = unit


# Energy is stored by the parser under E_energy (generic) as well as the
# template-specific U_cap/U_E/U_B names, so resolution must span all of them.
_CAP_ENERGY_ALIASES = (
    "U_cap", "U_E", "U_total", "E_energy", "energy",
    "stored_energy", "electric_field_energy", "electric_energy", "W_C",
)
_MAG_ENERGY_ALIASES = (
    "U_B", "W_L", "E_energy", "energy", "magnetic_field_energy", "magnetic_energy",
)


def _resolve_cap_energy(env: dict[str, dict[str, Any]]) -> float | None:
    return _value(env, *_CAP_ENERGY_ALIASES)


def _resolve_mag_energy(env: dict[str, dict[str, Any]]) -> float | None:
    return _value(env, *_MAG_ENERGY_ALIASES)


def _parse_amplitude(env: dict[str, dict[str, Any]], text: str, kind: str) -> float | None:
    """Parse the peak amplitude from a sinusoidal voltage/current function.

    Handles forms like ``U(t) = 200cos(500t)``, ``U = 100 sin(1000t)``,
    ``U(t) = 250 x sin(1000t) V`` and ``I(t) = 2sin(100*pi*t)``. Returns the
    leading coefficient (peak value). No eval is used.
    """
    if kind == "voltage":
        symbols = "uv"
    else:
        symbols = "i"
    pattern = re.compile(
        r"(?<![a-z])[" + symbols + r"]\s*(?:_?max)?\s*(?:\(\s*t\s*\))?\s*="
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:[x*\u00d7\u00b7]\s*)?(?:sin|cos)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        amplitude = float(match.group(1))
        _execution_metadata(env)["parsed_function_amplitudes"][kind] = amplitude
        return amplitude
    return None


def _parse_current_delta(env: dict[str, dict[str, Any]], text: str) -> tuple[float | None, str | None]:
    """Determine |delta I| for an induced-EMF problem.

    Priority: explicit env I & I2 -> |I - I2|; else parse 'from X to Y' in the
    text; else fall back to the magnitude of I (current that drops to/from 0).
    """
    i1 = _value(env, "I", "current", "I_initial")
    i2 = _value(env, "I2", "I_final", "I_2")
    if i1 is not None and i2 is not None:
        delta = abs(i1 - i2)
        _execution_metadata(env)["parsed_current_delta"] = {"delta": delta, "source": "env_I_I2"}
        return delta, "env_I_I2"
    match = re.search(
        r"from\s+([0-9]+(?:\.[0-9]+)?)\s*a?\s*to\s+([0-9]+(?:\.[0-9]+)?)",
        text, re.IGNORECASE,
    )
    if match:
        delta = abs(float(match.group(1)) - float(match.group(2)))
        _execution_metadata(env)["parsed_current_delta"] = {"delta": delta, "source": "text_from_to"}
        return delta, "text_from_to"
    if i1 is not None:
        _execution_metadata(env)["parsed_current_delta"] = {"delta": abs(i1), "source": "env_I_to_zero"}
        return abs(i1), "env_I_to_zero"
    return None, None


def _area_m2(env: dict[str, dict[str, Any]]) -> float | None:
    """Resolve a cross-sectional area in m^2, converting cm^2/mm^2 if needed."""
    area = _value(env, "A", "area")
    if area is None:
        return None
    unit = (_unit(env, "A", "area") or "").lower()
    if unit in {"cm^2", "cm2"}:
        return area * 1e-4
    if unit in {"mm^2", "mm2"}:
        return area * 1e-6
    return area


# Unambiguous loose LLM formula names -> canonical formula strings. Ambiguous
# families (capacitor energy, capacitance, Coulomb scalar-vs-vector) are NOT
# listed here; those are resolved by the canonicalizer with input/target context
# so the executor backstop never guesses a variant.
_LOOSE_FORMULA_ALIASES: dict[str, str] = {
    "magnetic_field_solenoid": "B = mu_0 * n_turns_per_meter * I",
    "magnetic_field": "B = mu_0 * n_turns_per_meter * I",
    "magnetic_flux": "Phi_B = B * A",
    "magnetic_flux_solenoid": "Phi_B = B * A",
    "flux_linkage": "Phi_B = B * A",
    "magnetic_flux_linkage": "Phi_B = B * A",
    "emf_from_flux_change": "emf = L * I / t",
    "emf_self_inductance": "emf = L * I / t",
    "faradays_law": "emf = L * I / t",
    "turn_density": "n_turns_per_meter = n_turns / L",
    "impedance_series": "Z = sqrt(R^2 + (X_L - X_C)^2)",
    "impedance_rlc": "Z = sqrt(R^2 + (X_L - X_C)^2)",
    "impedance_of_series_rlc": "Z = sqrt(R^2 + (X_L - X_C)^2)",
    "current_rms": "I_rms = V / Z",
    "current_in_ac_circuit": "I_rms = V / Z",
    "inductive_reactance": "X_L = omega * L_ind",
    "capacitive_reactance": "X_C = 1 / (omega * C_cap)",
    "resonance_condition": "f_res = 1 / (2*pi*sqrt(L_ind*C_cap))",
    "resonant_frequency": "f_res = 1 / (2*pi*sqrt(L_ind*C_cap))",
    "resonance_frequency": "f_res = 1 / (2*pi*sqrt(L_ind*C_cap))",
    "angular_frequency": "omega = 2*pi*f",
    "power_factor_formula": "power_factor = R / Z",
    "electric_field_point_charge": "E = k * abs(q) / r^2",
    "voltage_across_resistor": "U_R = I_rms * R",
    "voltage_across_capacitor": "U_C = I_rms * X_C",
}


def _normalize_loose_formula(formula: str) -> str | None:
    """Map an unambiguous loose semantic formula name to a canonical formula.

    Only fires for names with no '=' (i.e. not already an equation)."""
    token = formula.strip().lower()
    if "=" in token:
        return None
    token = token.replace(" ", "_").replace("-", "_")
    return _LOOSE_FORMULA_ALIASES.get(token)


def _execute_formula_step(step: dict[str, Any], env: dict[str, dict[str, Any]], world_input: Type2WorldModelInput) -> Type2ExecutionStepResult:
    formula = _formula_text(step)
    template = str(step.get("template_name") or "")
    # Part 5 backstop: canonicalize an unambiguous loose formula name before dispatch.
    _loose_canonical = _normalize_loose_formula(formula)
    if _loose_canonical is not None:
        _execution_metadata(env)["formula_alias_canonicalized"] = True
        formula = _loose_canonical
    output = _target_output(step)
    lhs = _formula_lhs(step)
    if _loose_canonical is not None and "=" in formula:
        lhs = formula.split("=", 1)[0].strip()
    inputs = _inputs_for_step(step, env)
    formula_l = formula.lower().replace(" ", "")
    unit = _infer_output_unit(output, formula, world_input.target_unit if output == world_input.target else None)

    try:
        value: float | None = None
        out_name = output
        warnings: list[str] = []

        if formula_l.startswith("e=k*abs(q)/r^2") or formula_l.startswith("e=k*abs(q)/r**2"):
            q = _value(env, "q", "Q", "q1", "q2", "q3", "qA", "qB", "qC")
            r_value = _value(env, "r", "d", "distance", "r1", "r13", "AC", "AM", "AO", "AH")
            k = _value(env, "k")
            if q is None or r_value is None or k is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V/m", errors=[_missing_from_values({"q": q, "r": r_value, "k": k}, env, formula, "Missing q, r, or k.")])
            value, out_name, unit = k * abs(q) / (r_value * r_value), lhs or output or "E", "V/m"
        elif formula_l == "q=e*r^2/k":
            field = _value(env, "E", "E_net", "electric_field")
            r_value = _value(env, "r", "d", "distance")
            k = _value(env, "k")
            if field is None or r_value is None or k is None:
                return _step_result(step, "FAIL", inputs, {}, None, "C", errors=[_missing_from_values({"E": field, "r": r_value, "k": k}, env, formula, "Missing E, r, or k.")])
            value, out_name, unit = field * r_value * r_value / k, lhs or output or "q", "C"
        elif formula_l.startswith("e1=k*abs(q1)/r1^2") or formula_l.startswith("e_1=k*abs(q1)/r1^2"):
            q = _value(env, "q1", "qA")
            r_value = _value(env, "r1", "d1", "AC", "AM", "AO", "AH", "r13")
            k = _value(env, "k")
            if q is None or r_value is None or k is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V/m", errors=[_missing_from_values({"q1": q, "r1": r_value, "k": k}, env, formula, "Missing q1, r1, or k.")])
            value, out_name, unit = k * abs(q) / (r_value * r_value), lhs or output or "E1", "V/m"
        elif formula_l.startswith("e2=k*abs(q2)/r2^2") or formula_l.startswith("e_2=k*abs(q2)/r2^2"):
            q = _value(env, "q2", "qB")
            r_value = _value(env, "r2", "d2", "BC", "BM", "BO", "BH", "r23")
            k = _value(env, "k")
            if q is None or r_value is None or k is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V/m", errors=[_missing_from_values({"q2": q, "r2": r_value, "k": k}, env, formula, "Missing q2, r2, or k.")])
            value, out_name, unit = k * abs(q) / (r_value * r_value), lhs or output or "E2", "V/m"
        elif "vector_sum(e1,e2" in formula_l or "vector_sum(e_i)" in formula_l:
            _execution_metadata(env)["role_aware_electric_field_used"] = True
            e1 = _value(env, "E1", "E_1")
            e2 = _value(env, "E2", "E_2")
            if e1 is None or e2 is None:
                _execution_metadata(env)["electric_field_vector_mode"] = "symbolic_missing_pair_fields"
                return _step_result(step, "WARN", inputs, {}, None, "V/m", warnings=["Electric field vector sum left symbolic because pair fields are unavailable."])
            if _has_condition(world_input, "midpoint", "same sign", "equal like"):
                value = 0.0
                _execution_metadata(env)["electric_field_vector_mode"] = "symmetric_midpoint_cancel"
            elif _has_condition(world_input, "right", "perpendicular", "90"):
                value = _safe_sqrt(e1 * e1 + e2 * e2)
                _execution_metadata(env)["electric_field_vector_mode"] = "right_angle"
            elif _has_condition(world_input, "equilateral", "60"):
                value = _safe_sqrt(e1 * e1 + e2 * e2 + 2 * e1 * e2 * math.cos(math.radians(60)))
                _execution_metadata(env)["electric_field_vector_mode"] = "equilateral"
            elif _has_condition(world_input, "collinear", "opposite sides", "straight line"):
                q1 = _value(env, "q1", "qA")
                q2 = _value(env, "q2", "qB")
                value = e1 + e2 if q1 is not None and q2 is not None and q1 * q2 < 0 else abs(e1 - e2)
                _execution_metadata(env)["electric_field_vector_mode"] = "collinear"
            else:
                return _step_result(step, "WARN", inputs, {}, None, "V/m", warnings=["Ambiguous electric field vector geometry; left symbolic."])
            _execution_metadata(env)["role_aware_electric_field_used"] = True
            _execution_metadata(env)["electric_field_pair_count"] = 2
            value, out_name, unit = value, lhs or output or "E", "V/m"
        elif "vector_sum(k*q/r_i^2)" in formula_l:
            _execution_metadata(env)["role_aware_electric_field_used"] = True
            _execution_metadata(env)["electric_field_vector_mode"] = "symbolic_geometry"
            return _step_result(step, "WARN", inputs, {}, None, "V/m", warnings=["Electric field vector geometry requires explicit source distances; left symbolic."])
        elif formula_l in {"i=v/r", "i=u/r"}:
            v = _v(env)
            r_value = _r(env)
            if v is None or r_value is None:
                return _step_result(step, "FAIL", inputs, {}, None, "A", errors=[_missing_from_values({"V": v, "R": r_value}, env, formula, "Missing V/U or R.")])
            value, out_name, unit = v / r_value, lhs or output or "I", "A"
        elif formula_l in {"v=i*r", "u=i*r"}:
            current = _i(env)
            r_value = _r(env)
            if current is None or r_value is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V", errors=[_missing_from_values({"I": current, "R": r_value}, env, formula, "Missing I or R.")])
            value, out_name, unit = current * r_value, lhs or output or "V", "V"
        elif formula_l in {"r=v/i", "r=u/i"}:
            v = _v(env)
            current = _i(env)
            if v is None or current is None:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_from_values({"V": v, "I": current}, env, formula, "Missing V/U or I.")])
            value, out_name, unit = v / current, lhs or output or "R", "ohm"
        elif formula_l in {"p=v^2/r", "p=u^2/r"}:
            v = _v(env)
            r_value = _r(env)
            if v is None or r_value is None:
                return _step_result(step, "FAIL", inputs, {}, None, "W", errors=[_missing_from_values({"V": v, "R": r_value}, env, formula, "Missing V/U or R.")])
            value, out_name, unit = v * v / r_value, lhs or output or "P", "W"
        elif formula_l in {"p=v*i", "p=u*i"}:
            v = _v(env)
            current = _i(env)
            if v is None or current is None:
                return _step_result(step, "FAIL", inputs, {}, None, "W", errors=[_missing_from_values({"V": v, "I": current}, env, formula, "Missing V/U or I.")])
            value, out_name, unit = v * current, lhs or output or "P", "W"
        elif formula_l == "p=i^2*r":
            current = _i(env)
            r_value = _r(env)
            if current is None or r_value is None:
                return _step_result(step, "FAIL", inputs, {}, None, "W", errors=[_missing_from_values({"I": current, "R": r_value}, env, formula, "Missing I or R.")])
            value, out_name, unit = current * current * r_value, lhs or output or "P", "W"
        elif formula_l in {"r_total=v^2/p", "r_total=u^2/p", "r_total=v_rms^2/p"}:
            v = _v(env)
            power = _value(env, "P", "power")
            if v is None or power is None:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_from_values({"V": v, "P": power}, env, formula, "Missing V/U or P.")])
            value, out_name, unit = v * v / power, lhs or output or "R_total", "ohm"
        elif formula_l in {"r=r_total-r2", "r2=r_total-r1"}:
            total = _value(env, "R_total", "Rtotal")
            other = _value(env, "R2" if formula_l.startswith("r=") else "R1")
            if total is None or other is None:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_from_values({"R_total": total, "R_other": other}, env, formula, "Missing R_total or branch resistance.")])
            value, out_name, unit = total - other, lhs or output or ("R" if formula_l.startswith("r=") else "R2"), "ohm"
        elif "p_total=sum(p_i)" in formula_l or "total_power=sum" in formula_l:
            powers = _numeric_list_from_env(env, "P")
            if not powers:
                return _step_result(step, "WARN", inputs, {}, None, "W", warnings=["Power terms are unavailable; total power left symbolic."])
            value, out_name, unit = sum(powers), lhs or output or "P_total", "W"
        elif formula_l == "a=pi*r^2":
            radius = _value(env, "r", "radius")
            if radius is None:
                return _step_result(step, "FAIL", inputs, {}, None, "m^2", errors=[_missing_from_values({"r": radius}, env, formula, "Missing radius.")])
            value, out_name, unit = math.pi * radius * radius, lhs or output or "A", "m^2"
        elif formula_l == "c_cap=epsilon_0*a/d":
            eps0 = _value(env, "epsilon_0")
            area = _value(env, "A", "area")
            d = _value(env, "d", "distance")
            if eps0 is None or area is None or d is None:
                return _step_result(step, "FAIL", inputs, {}, None, "F", errors=[_missing_from_values({"epsilon_0": eps0, "A": area, "d": d}, env, formula, "Missing epsilon_0, A, or d.")])
            value, out_name, unit = eps0 * area / d, lhs or output or "C_cap", "F"
        elif formula_l == "c_cap=epsilon_r*epsilon_0*a/d":
            epsr = _value(env, "epsilon_r")
            eps0 = _value(env, "epsilon_0")
            area = _value(env, "A", "area")
            d = _value(env, "d", "distance")
            if epsr is None or eps0 is None or area is None or d is None:
                return _step_result(step, "FAIL", inputs, {}, None, "F", errors=[_missing_from_values({"epsilon_r": epsr, "epsilon_0": eps0, "A": area, "d": d}, env, formula, "Missing epsilon_r, epsilon_0, A, or d.")])
            value, out_name, unit = epsr * eps0 * area / d, lhs or output or "C_cap", "F"
        elif formula_l == "epsilon_r=c_cap*d/(epsilon_0*a)":
            c = _cap(env)
            d = _value(env, "d", "distance")
            eps0 = _value(env, "epsilon_0")
            area = _value(env, "A", "area")
            if c is None or d is None or eps0 is None or area is None:
                return _step_result(step, "FAIL", inputs, {}, None, None, errors=[_missing_from_values({"C_cap": c, "d": d, "epsilon_0": eps0, "A": area}, env, formula, "Missing C_cap, d, epsilon_0, or A.")])
            value, out_name, unit = c * d / (eps0 * area), lhs or output or "epsilon_r", None
        elif formula_l == "c_after=epsilon_r*c_cap":
            epsr = _value(env, "epsilon_r")
            c = _cap(env)
            if epsr is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "F", errors=[_missing_from_values({"epsilon_r": epsr, "C_cap": c}, env, formula, "Missing epsilon_r or C_cap.")])
            value, out_name, unit = epsr * c, lhs or output or "C_after", "F"
        elif formula_l == "v_after=v/epsilon_r":
            v = _v(env)
            epsr = _value(env, "epsilon_r")
            if v is None or epsr is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V", errors=[_missing_from_values({"V": v, "epsilon_r": epsr}, env, formula, "Missing V or epsilon_r.")])
            value, out_name, unit = v / epsr, lhs or output or "V_after", "V"
        elif formula_l in {"q_after=q", "v=v", "u_c=v"}:
            source = _value(env, "Q", "q") if formula_l.startswith("q") else _v(env)
            if source is None:
                return _step_result(step, "FAIL", inputs, {}, None, "C" if formula_l.startswith("q") else "V", errors=[_missing_error("Missing identity source value.")])
            value, out_name, unit = source, lhs or output or ("Q_after" if formula_l.startswith("q") else "V"), "C" if formula_l.startswith("q") else "V"
        elif "0.5" in formula_l and "c_cap" in formula_l and "v^2" in formula_l:
            c = _value(env, "C_cap", "C")
            v = _value(env, "V", "U")
            if c is None or v is None:
                return _step_result(step, "FAIL", inputs, {}, None, unit, errors=[_missing_error("Missing C_cap or V.", ["C_cap", "V"], formula, env)])
            value, out_name, unit = 0.5 * c * v * v, output or "U_cap", "J"
        elif formula_l.startswith("c_cap=q/v") or "c_cap=q/v" in formula_l:
            q = _value(env, "Q", "q")
            v = _value(env, "V", "U")
            if q is None or v is None:
                return _step_result(step, "FAIL", inputs, {}, None, "F", errors=[_missing_error("Missing Q or V.", ["Q", "V"], formula, env)])
            value, out_name, unit = q / v, output or "C_cap", "F"
        elif formula_l.startswith("q=c_cap*v") or "q=c_cap*v" in formula_l:
            c = _value(env, "C_cap", "C")
            v = _value(env, "V", "U")
            if c is None or v is None:
                return _step_result(step, "FAIL", inputs, {}, None, "C", errors=[_missing_error("Missing C_cap or V.", ["C_cap", "V"], formula, env)])
            value, out_name, unit = c * v, output or "Q", "C"
        elif formula_l.startswith("v=q/c_cap") or "v=q/c_cap" in formula_l:
            q = _value(env, "Q", "q")
            c = _value(env, "C_cap", "C")
            if q is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V", errors=[_missing_error("Missing Q or C_cap.", ["Q", "C_cap"], formula, env)])
            value, out_name, unit = q / c, output or "V", "V"
        elif formula_l in {"c_cap=2*u_cap/v^2", "c_cap=2*u_e/v^2", "c_cap=2*energy/v^2", "c_cap=2*e_energy/v^2"}:
            energy = _resolve_cap_energy(env)
            v = _v(env)
            if energy is None or v is None:
                return _step_result(step, "FAIL", inputs, {}, None, "F", errors=[_missing_from_values({"U_cap": energy, "V": v}, env, formula, "Missing capacitor energy or V.")])
            _mark_step9(env, formula)
            value, out_name, unit = 2 * energy / (v * v), lhs or output or "C_cap", "F"
        elif formula_l in {"v=sqrt(2*u_cap/c_cap)", "u_c=sqrt(2*u_cap/c_cap)", "v=sqrt(2*u_e/c_cap)", "u_c=sqrt(2*u_e/c_cap)", "v=sqrt(2*energy/c_cap)"}:
            energy = _resolve_cap_energy(env)
            c = _cap(env)
            if energy is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V", errors=[_missing_from_values({"U_cap": energy, "C_cap": c}, env, formula, "Missing capacitor energy or C_cap.")])
            _mark_step9(env, formula)
            value, out_name, unit = _safe_sqrt(2 * energy / c), lhs or output or ("U_C" if formula_l.startswith("u_c") else "V"), "V"
        elif formula_l in {"u_cap=q^2/(2*c_cap)", "u_e=q^2/(2*c_cap)", "u_total=q_max^2/(2*c_cap)", "u_total=q^2/(2*c_cap)", "u_e=q_max^2/(2*c_cap)"}:
            q = _value(env, "Q_max", "Q", "q", "q_max", "charge")
            c = _cap(env)
            if q is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "J", errors=[_missing_from_values({"Q": q, "C_cap": c}, env, formula, "Missing Q/Q_max or C_cap.")])
            _mark_step9(env, formula)
            value, out_name, unit = q * q / (2 * c), lhs or output or ("U_total" if "u_total" in formula_l else "U_cap"), "J"
        elif formula_l == "u_cap=0.5*q*v":
            q = _value(env, "Q", "q")
            v = _v(env)
            if q is None or v is None:
                return _step_result(step, "FAIL", inputs, {}, None, "J", errors=[_missing_from_values({"Q": q, "V": v}, env, formula, "Missing Q or V.")])
            value, out_name, unit = 0.5 * q * v, lhs or output or "U_cap", "J"
        elif formula_l in {"u_b=0.5*l*i^2", "u_b=0.5*l*i_max^2", "u_b=0.5*l_ind*i^2", "u_b=0.5*l_ind*i_max^2"}:
            inductance = _ind(env)
            current = _value(env, "I_max", "I", "current")
            if current is None:
                current = _parse_amplitude(env, world_input.problem_text, "current")
            if inductance is None or current is None:
                return _step_result(step, "FAIL", inputs, {}, None, "J", errors=[_missing_from_values({"L": inductance, "I_max": current}, env, formula, "Missing L/L_ind or I/I_max.")])
            if "i_max" in formula_l or _value(env, "I_max", "I", "current") is None:
                _mark_step9(env, formula)
            value, out_name, unit = 0.5 * inductance * current * current, lhs or output or "U_B", "J"
        elif formula_l == "l_ind=2*u_b/i^2":
            energy = _value(env, "U_B")
            current = _i(env)
            if energy is None or current is None:
                return _step_result(step, "FAIL", inputs, {}, None, "H", errors=[_missing_from_values({"U_B": energy, "I": current}, env, formula, "Missing U_B or I.")])
            value, out_name, unit = 2 * energy / (current * current), lhs or output or "L_ind", "H"
        elif formula_l in {"i=sqrt(2*u_b/l)", "i_max=sqrt(2*u_b/l)"}:
            energy = _value(env, "U_B")
            inductance = _ind(env)
            if energy is None or inductance is None:
                return _step_result(step, "FAIL", inputs, {}, None, "A", errors=[_missing_from_values({"U_B": energy, "L": inductance}, env, formula, "Missing U_B or L.")])
            value, out_name, unit = _safe_sqrt(2 * energy / inductance), lhs or output or ("I_max" if formula_l.startswith("i_max") else "I"), "A"
        elif formula_l in {"f_osc=1/(2*pi*sqrt(l*c_cap))", "f_res=1/(2*pi*sqrt(l_ind*c_cap))"}:
            inductance = _ind(env)
            c = _cap(env)
            if inductance is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "Hz", errors=[_missing_from_values({"L": inductance, "C_cap": c}, env, formula, "Missing L/L_ind or C_cap.")])
            value, out_name, unit = 1 / (2 * math.pi * math.sqrt(inductance * c)), lhs or output or ("f_res" if formula_l.startswith("f_res") else "f_osc"), "Hz"
        elif formula_l == "t_osc=2*pi*sqrt(l*c_cap)":
            inductance = _ind(env)
            c = _cap(env)
            if inductance is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "s", errors=[_missing_from_values({"L": inductance, "C_cap": c}, env, formula, "Missing L or C_cap.")])
            value, out_name, unit = 2 * math.pi * math.sqrt(inductance * c), lhs or output or "T_osc", "s"
        elif formula_l == "omega_0=1/sqrt(l_ind*c_cap)":
            inductance = _ind(env)
            c = _cap(env)
            if inductance is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "rad/s", errors=[_missing_from_values({"L_ind": inductance, "C_cap": c}, env, formula, "Missing L_ind or C_cap.")])
            value, out_name, unit = 1 / math.sqrt(inductance * c), lhs or output or "omega_0", "rad/s"
        elif formula_l == "c_cap=1/(4*pi^2*f^2*l_ind)":
            freq = _freq(env)
            inductance = _ind(env)
            if freq is None or inductance is None:
                return _step_result(step, "FAIL", inputs, {}, None, "F", errors=[_missing_from_values({"f": freq, "L_ind": inductance}, env, formula, "Missing f or L_ind.")])
            value, out_name, unit = 1 / (4 * math.pi * math.pi * freq * freq * inductance), lhs or output or "C_cap", "F"
        elif formula_l == "l_ind=1/(4*pi^2*f^2*c_cap)":
            freq = _freq(env)
            c = _cap(env)
            if freq is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "H", errors=[_missing_from_values({"f": freq, "C_cap": c}, env, formula, "Missing f or C_cap.")])
            value, out_name, unit = 1 / (4 * math.pi * math.pi * freq * freq * c), lhs or output or "L_ind", "H"
        elif formula_l in {"r=z", "z=r"}:
            source = _value(env, "Z") if formula_l.startswith("r=") else _value(env, "R")
            if source is None:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_error("Missing impedance/resistance identity source.")])
            value, out_name, unit = source, lhs or output or ("R" if formula_l.startswith("r=") else "Z"), "ohm"
        elif formula_l in {"i_rms=v/r", "i_rms=v_rms/r"}:
            v = _v(env)
            r_value = _r(env)
            if v is None or r_value is None:
                return _step_result(step, "FAIL", inputs, {}, None, "A", errors=[_missing_from_values({"V": v, "R": r_value}, env, formula, "Missing V/V_rms or R.")])
            value, out_name, unit = v / r_value, lhs or output or "I_rms", "A"
        elif formula_l == "u_l=i_rms*omega_0*l_ind":
            current = _value(env, "I_rms", "I")
            omega0 = _value(env, "omega_0", "omega")
            inductance = _ind(env)
            if current is None or omega0 is None or inductance is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V", errors=[_missing_from_values({"I_rms": current, "omega_0": omega0, "L_ind": inductance}, env, formula, "Missing I_rms, omega_0, or L_ind.")])
            value, out_name, unit = current * omega0 * inductance, lhs or output or "U_L", "V"
        elif formula_l in {"x_l=omega*l", "x_l=omega*l_ind"}:
            omega = _value(env, "omega", "omega_0")
            inductance = _ind(env)
            if omega is None or inductance is None:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_from_values({"omega": omega, "L": inductance}, env, formula, "Missing omega or L.")])
            value, out_name, unit = omega * inductance, lhs or output or "X_L", "ohm"
        elif formula_l == "x_c=1/(omega*c_cap)":
            omega = _value(env, "omega", "omega_0")
            c = _cap(env)
            if omega is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_from_values({"omega": omega, "C_cap": c}, env, formula, "Missing omega or C_cap.")])
            value, out_name, unit = 1 / (omega * c), lhs or output or "X_C", "ohm"
        elif formula_l in {"omega=2*pi*f", "omega=2*pi*f"}:
            freq = _freq(env)
            if freq is None:
                return _step_result(step, "FAIL", inputs, {}, None, "rad/s", errors=[_missing_from_values({"f": freq}, env, formula, "Missing f.")])
            value, out_name, unit = 2 * math.pi * freq, lhs or output or "omega", "rad/s"
        elif formula_l == "z=sqrt(r^2+(x_l-x_c)^2)":
            r_value = _r(env)
            xl = _value(env, "X_L")
            xc = _value(env, "X_C")
            if r_value is None or xl is None or xc is None:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_from_values({"R": r_value, "X_L": xl, "X_C": xc}, env, formula, "Missing R, X_L, or X_C.")])
            value, out_name, unit = _safe_sqrt(r_value * r_value + (xl - xc) ** 2), lhs or output or "Z", "ohm"
        elif formula_l == "power_factor=r/z":
            r_value = _r(env)
            z = _value(env, "Z")
            if r_value is None or z is None:
                return _step_result(step, "FAIL", inputs, {}, None, None, errors=[_missing_from_values({"R": r_value, "Z": z}, env, formula, "Missing R or Z.")])
            value, out_name, unit = r_value / z, lhs or output or "power_factor", None
        elif formula_l == "power_factor=1":
            value, out_name, unit = 1.0, lhs or output or "power_factor", None
        elif "sqrt(f*r^2/k)" in formula_l:
            force = _value(env, "F", "F_e", "F_net")
            r = _value(env, "r", "d")
            k = _value(env, "k")
            if force is None or r is None or k is None:
                return _step_result(step, "FAIL", inputs, {}, None, "C", errors=[_missing_from_values({"F": force, "r": r, "k": k}, env, formula, "Missing F, r, or k.")])
            value, out_name, unit = math.sqrt(force * r * r / k), output or "q", "C"
        elif "abs(q1*q3)" in formula_l:
            q1 = _resolve_value(env, "q1")
            q3 = _resolve_value(env, "q3")
            r13 = _resolve_distance(env, "r13", world_input)
            k = _value(env, "k")
            if q1 is None or q3 is None or r13 is None or k is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_from_values({"q1": q1, "q3": q3, "r13": r13, "k": k}, env, formula, "Missing q1, q3, r13, or k.")])
            value, out_name, unit = k * abs(q1 * q3) / (r13 * r13), lhs or output or "F_13", "N"
        elif "abs(q2*q3)" in formula_l:
            q2 = _resolve_value(env, "q2")
            q3 = _resolve_value(env, "q3")
            r23 = _resolve_distance(env, "r23", world_input)
            k = _value(env, "k")
            if q2 is None or q3 is None or r23 is None or k is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_from_values({"q2": q2, "q3": q3, "r23": r23, "k": k}, env, formula, "Missing q2, q3, r23, or k.")])
            value, out_name, unit = k * abs(q2 * q3) / (r23 * r23), lhs or output or "F_23", "N"
        elif "abs(q1*q2)" in formula_l or "abs(q1 * q2)" in formula.lower():
            q1 = _resolve_value(env, "q1")
            q2 = _resolve_value(env, "q2")
            r = _resolve_distance(env, "r12", world_input) or _resolve_value(env, "r", aliases=("r", "d", "AB", "BA"))
            k = _value(env, "k")
            if q1 is None or q2 is None or r is None or k is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_from_values({"q1": q1, "q2": q2, "r": r, "k": k}, env, formula, "Missing q1, q2, r, or k.")])
            value, out_name, unit = k * abs(q1 * q2) / (r * r), _force_output_name(output, lhs or "F"), "N"
        elif "f_net=f+f2" in formula_l:
            f = _value(env, "F", "F1")
            f2 = _value(env, "F2", "F_2")
            if f is None or f2 is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_error("Missing F or F2.")])
            value, out_name, unit = f + f2, lhs or output or "F_net", "N"
        elif "abs(f-f2)" in formula_l:
            f = _value(env, "F", "F1")
            f2 = _value(env, "F2", "F_2")
            if f is None or f2 is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_error("Missing F or F2.")])
            value, out_name, unit = abs(f - f2), lhs or output or "F_net", "N"
        elif "sqrt(f^2+f^2+2*f*f*cos(theta))" in formula_l:
            f = _value(env, "F", "F1")
            theta = _value(env, "theta", "angle")
            if f is None or theta is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_error("Missing F or theta.")])
            theta_rad = math.radians(theta) if _unit(env, "theta", "angle") == "deg" else theta
            value, out_name, unit = math.sqrt(f * f + f * f + 2 * f * f * math.cos(theta_rad)), lhs or output or "F_net", "N"
        elif "sqrt(f^2+f2^2+2*f*f2*cos(theta))" in formula_l:
            f = _value(env, "F", "F1")
            f2 = _value(env, "F2", "F_2")
            theta = _value(env, "theta", "angle")
            if f is None or f2 is None or theta is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_error("Missing F, F2, or theta.")])
            theta_rad = math.radians(theta) if _unit(env, "theta", "angle") == "deg" else theta
            value, out_name, unit = math.sqrt(f * f + f2 * f2 + 2 * f * f2 * math.cos(theta_rad)), lhs or output or "F_net", "N"
        elif "sqrt(f^2+f2^2)" in formula_l:
            f = _value(env, "F", "F1")
            f2 = _value(env, "F2", "F_2")
            if f is None or f2 is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_error("Missing F or F2.")])
            value, out_name, unit = math.sqrt(f * f + f2 * f2), lhs or output or "F_net", "N"
        elif "sqrt(f_13^2+f_23^2+2*f_13*f_23*cos(60" in formula_l:
            f13 = _value(env, "F_13")
            f23 = _value(env, "F_23")
            if f13 is None or f23 is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_error("Missing F_13 or F_23.")])
            _execution_metadata(env)["vector_sum_mode"] = "equilateral"
            value, out_name, unit = math.sqrt(f13 * f13 + f23 * f23 + 2 * f13 * f23 * math.cos(math.radians(60))), lhs or output or "F_net", "N"
        elif "sqrt(f_13^2+f_23^2)" in formula_l:
            f13 = _value(env, "F_13")
            f23 = _value(env, "F_23")
            if f13 is None or f23 is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_error("Missing F_13 or F_23.")])
            _execution_metadata(env)["vector_sum_mode"] = "right_angle"
            value, out_name, unit = math.sqrt(f13 * f13 + f23 * f23), lhs or output or "F_net", "N"
        elif "abs(f_13-f_23)" in formula_l:
            f13 = _value(env, "F_13")
            f23 = _value(env, "F_23")
            if f13 is None or f23 is None:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_error("Missing F_13 or F_23.")])
            _execution_metadata(env)["vector_sum_mode"] = "collinear_opposite"
            value, out_name, unit = abs(f13 - f23), lhs or output or "F_net", "N"
        elif "vector_sum(f_13,f_23" in formula_l or "vector_sum(k*q_i*q_j/r_ij^2)" in formula_l:
            f13 = _value(env, "F_13")
            f23 = _value(env, "F_23")
            if f13 is None or f23 is None:
                _execution_metadata(env)["vector_sum_mode"] = "symbolic_missing_pair_forces"
                return _step_result(step, "WARN", inputs, {}, None, "N", warnings=["Vector sum left symbolic because pair forces are unavailable."])
            if _has_condition(world_input, "right_angle", "right-angled", "perpendicular", "90"):
                value = math.sqrt(f13 * f13 + f23 * f23)
                _execution_metadata(env)["vector_sum_mode"] = "right_angle"
            elif _has_condition(world_input, "equilateral", "60"):
                value = math.sqrt(f13 * f13 + f23 * f23 + 2 * f13 * f23 * math.cos(math.radians(60)))
                _execution_metadata(env)["vector_sum_mode"] = "equilateral"
            elif _has_condition(world_input, "collinear", "opposite"):
                value = abs(f13 - f23)
                _execution_metadata(env)["vector_sum_mode"] = "collinear_opposite"
            elif _has_condition(world_input, "same direction"):
                value = f13 + f23
                _execution_metadata(env)["vector_sum_mode"] = "same_direction"
            else:
                _execution_metadata(env)["vector_sum_mode"] = "symbolic_ambiguous"
                return _step_result(step, "WARN", inputs, {}, None, "N", warnings=["Ambiguous vector geometry; left symbolic."])
            out_name, unit = lhs or output or "F_net", "N"
        elif "sum(f_i)" in formula_l:
            if _value(env, "F") is not None and _value(env, "F2") is not None:
                forces = [_value(env, "F") or 0.0, _value(env, "F2") or 0.0]
            else:
                seen_values: set[float] = set()
                forces = []
                for key, item in env.items():
                    if key.lower().startswith("f") and item.get("unit") == "N":
                        value_key = round(float(item["value"]), 12)
                        if value_key not in seen_values:
                            seen_values.add(value_key)
                            forces.append(float(item["value"]))
            if not forces:
                return _step_result(step, "FAIL", inputs, {}, None, "N", errors=[_missing_error("No force variables available.")])
            value, out_name, unit = sum(forces), lhs or output or "F_net", "N"
        elif formula_l == "mean_value=sum(measurements)/n":
            measurements = _numeric_list_from_env(env, "measurement")
            n = _value(env, "n")
            if not measurements:
                return _step_result(step, "WARN", inputs, {}, None, None, warnings=["Measurement array unavailable; mean left symbolic."])
            value, out_name, unit = sum(measurements) / (n or len(measurements)), lhs or output or "mean_value", None
        elif formula_l == "abs_error=abs(measured_value-true_value)":
            measured = _value(env, "measured_value", "measurement")
            true = _value(env, "true_value", "accepted_value")
            if measured is None or true is None:
                return _step_result(step, "WARN", inputs, {}, None, None, warnings=["Measured or true value unavailable; absolute error left symbolic."])
            value, out_name, unit = abs(measured - true), lhs or output or "abs_error", None
        elif formula_l == "abs_error=sum(abs(each-mean_value))/n":
            measurements = _numeric_list_from_env(env, "measurement")
            mean_value = _value(env, "mean_value")
            n = _value(env, "n")
            if not measurements or mean_value is None:
                return _step_result(step, "WARN", inputs, {}, None, None, warnings=["Measurement array unavailable; average absolute error left symbolic."])
            value, out_name, unit = sum(abs(item - mean_value) for item in measurements) / (n or len(measurements)), lhs or output or "abs_error", None
        elif formula_l in {
            "rel_error=abs_error/abs(true_value)",
            "rel_error=delta_as/abs(as)",
            "rel_error=d/abs(d2)",
            "rel_error=delta_voltage/abs(voltage)",
            "rel_error=delta_length/abs(length)",
            "rel_error=delta_current/abs(current)",
        }:
            if "abs_error" in formula_l:
                numerator = _value(env, "abs_error")
                denominator = _value(env, "true_value")
            elif "delta_as" in formula_l:
                numerator = _value(env, "delta_as")
                denominator = _value(env, "as")
            elif "d/abs(d2)" in formula_l:
                numerator = _value(env, "d")
                denominator = _value(env, "d2")
            elif "delta_voltage" in formula_l:
                numerator = _value(env, "delta_voltage")
                denominator = _value(env, "voltage", "V")
            elif "delta_length" in formula_l:
                numerator = _value(env, "delta_length")
                denominator = _value(env, "length", "L")
            else:
                numerator = _value(env, "delta_current")
                denominator = _value(env, "current", "I")
            if numerator is None or denominator is None:
                return _step_result(step, "WARN", inputs, {}, None, None, warnings=["Relative error inputs unavailable; left symbolic."])
            value, out_name, unit = numerator / abs(denominator), lhs or output or "rel_error", None
        elif formula_l in {"percent_error=rel_error*100", "percent_error=100*rel_error"}:
            rel = _value(env, "rel_error", "relative_error")
            if rel is None:
                return _step_result(step, "WARN", inputs, {}, None, "%", warnings=["Relative error unavailable; percent error left symbolic."])
            value, out_name, unit = rel * 100, lhs or output or "percent_error", "%"
        elif formula_l == "random_error=max(deviations)":
            deviations = _numeric_list_from_env(env, "deviation")
            if not deviations:
                return _step_result(step, "WARN", inputs, {}, None, None, warnings=["Deviation array unavailable; random error left symbolic."])
            value, out_name, unit = max(deviations), lhs or output or "random_error", None
        elif "atan2" in formula_l:
            return _step_result(step, "WARN", inputs, {}, None, "deg", warnings=["Direction components are unavailable; theta left symbolic."])
        # ---------------------------------------------------------------- #
        # Step 9: low-risk executor coverage expansion                     #
        # ---------------------------------------------------------------- #
        elif formula_l in {"u_cap=0.5*c*u_max^2", "u_e=0.5*c*u_max^2", "u_cap=0.5*c_cap*u_max^2", "u_e=0.5*c_cap*u_max^2"}:
            c = _cap(env)
            u_max = _value(env, "U_max", "V_max")
            if u_max is None:
                u_max = _parse_amplitude(env, world_input.problem_text, "voltage")
            if c is None or u_max is None:
                return _step_result(step, "FAIL", inputs, {}, None, "J", errors=[_missing_from_values({"C_cap": c, "U_max": u_max}, env, formula, "Missing C/C_cap or U_max.")])
            _mark_step9(env, formula)
            value, out_name, unit = 0.5 * c * u_max * u_max, lhs or output or ("U_E" if formula_l.startswith("u_e") else "U_cap"), "J"
        elif formula_l in {"u_c=q/c_cap", "v=q/c_cap"}:
            q = _value(env, "Q", "q", "Q_max", "charge")
            c = _cap(env)
            if q is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V", errors=[_missing_from_values({"Q": q, "C_cap": c}, env, formula, "Missing Q or C_cap.")])
            _mark_step9(env, formula)
            value, out_name, unit = q / c, lhs or output or ("U_C" if formula_l.startswith("u_c") else "V"), "V"
        elif formula_l in {"f=f_res", "f=f_osc", "f_res=f", "f_res=f_osc"}:
            source = _value(env, "f_res", "f_osc", "f", "frequency")
            if source is None:
                return _step_result(step, "FAIL", inputs, {}, None, "Hz", errors=[_missing_from_values({"f_res": source}, env, formula, "Missing resonant frequency.")])
            _mark_step9(env, formula)
            value, out_name, unit = source, lhs or output or "f", "Hz"
        elif formula_l in {"omega=omega_0", "omega_0=omega"}:
            source = _value(env, "omega_0", "omega")
            if source is None:
                return _step_result(step, "FAIL", inputs, {}, None, "rad/s", errors=[_missing_from_values({"omega_0": source}, env, formula, "Missing omega.")])
            _mark_step9(env, formula)
            value, out_name, unit = source, lhs or output or "omega", "rad/s"
        elif formula_l in {"omega_0=2*pi*f_res", "omega_0=2*pi*f", "omega=2*pi*f_res"}:
            freq = _value(env, "f_res", "f", "f_osc", "frequency")
            if freq is None:
                return _step_result(step, "FAIL", inputs, {}, None, "rad/s", errors=[_missing_from_values({"f_res": freq}, env, formula, "Missing f_res.")])
            _mark_step9(env, formula)
            value, out_name, unit = 2 * math.pi * freq, lhs or output or "omega_0", "rad/s"
        elif formula_l in {"f_res=omega_0/(2*pi)", "f=omega_0/(2*pi)"}:
            omega0 = _value(env, "omega_0", "omega")
            if omega0 is None:
                return _step_result(step, "FAIL", inputs, {}, None, "Hz", errors=[_missing_from_values({"omega_0": omega0}, env, formula, "Missing omega_0.")])
            _mark_step9(env, formula)
            value, out_name, unit = omega0 / (2 * math.pi), lhs or output or "f_res", "Hz"
        elif formula_l in {"omega_0=1/sqrt(l*c_cap)", "omega=1/sqrt(l*c_cap)", "omega=1/sqrt(l_ind*c_cap)"}:
            inductance = _ind(env)
            c = _cap(env)
            if inductance is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "rad/s", errors=[_missing_from_values({"L": inductance, "C_cap": c}, env, formula, "Missing L/L_ind or C_cap.")])
            _mark_step9(env, formula)
            value, out_name, unit = 1 / math.sqrt(inductance * c), lhs or output or "omega_0", "rad/s"
        elif formula_l in {"f_osc=1/(2*pi*sqrt(l_ind*c_cap))", "f_res=1/(2*pi*sqrt(l*c_cap))", "f=1/(2*pi*sqrt(l*c_cap))", "f=1/(2*pi*sqrt(l_ind*c_cap))"}:
            inductance = _ind(env)
            c = _cap(env)
            if inductance is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "Hz", errors=[_missing_from_values({"L": inductance, "C_cap": c}, env, formula, "Missing L/L_ind or C_cap.")])
            _mark_step9(env, formula)
            value, out_name, unit = 1 / (2 * math.pi * math.sqrt(inductance * c)), lhs or output or ("f_res" if formula_l.startswith("f_res") else ("f" if formula_l.startswith("f=") else "f_osc")), "Hz"
        elif formula_l == "t_osc=2*pi*sqrt(l_ind*c_cap)":
            inductance = _ind(env)
            c = _cap(env)
            if inductance is None or c is None:
                return _step_result(step, "FAIL", inputs, {}, None, "s", errors=[_missing_from_values({"L_ind": inductance, "C_cap": c}, env, formula, "Missing L_ind or C_cap.")])
            _mark_step9(env, formula)
            value, out_name, unit = 2 * math.pi * math.sqrt(inductance * c), lhs or output or "T_osc", "s"
        elif formula_l in {"i_rms=v/z", "i=v/z", "i_rms=v_rms/z"}:
            v = _v(env)
            z = _value(env, "Z", "impedance", "R")
            if v is None or z is None:
                return _step_result(step, "FAIL", inputs, {}, None, "A", errors=[_missing_from_values({"V": v, "Z": z}, env, formula, "Missing V or Z.")])
            _mark_step9(env, formula)
            value, out_name, unit = v / z, lhs or output or "I_rms", "A"
        elif formula_l in {"z_2=v/i2", "z_2=v/i_2", "z2=v/i2"}:
            v = _v(env)
            i2 = _value(env, "I2", "I_2")
            if v is None or i2 is None:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_from_values({"V": v, "I2": i2}, env, formula, "Missing V or I2.")])
            _mark_step9(env, formula)
            value, out_name, unit = v / i2, lhs or output or "Z_2", "ohm"
        elif formula_l in {"x_c=sqrt(r2/r)", "x=sqrt(r2/r)", "k=sqrt(r2/r)", "x_l=sqrt(r2/r)"}:
            r2 = _value(env, "R2", "R_2")
            r_value = _value(env, "R", "R1", "resistance")
            if r2 is None or r_value is None or r_value == 0:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_from_values({"R2": r2, "R": r_value}, env, formula, "Missing R2 or R.")])
            _mark_step9(env, formula)
            default_name = {"x_c": "X_C", "x_l": "X_L", "k": "k"}.get(formula_l.split("=", 1)[0], "X")
            value, out_name, unit = _safe_sqrt(r2 / r_value), lhs or output or default_name, "ohm"
        elif formula_l in {"power_factor=r1/r2", "power_factor=r/r2"}:
            numerator = _value(env, "R1", "R") if formula_l == "power_factor=r1/r2" else _value(env, "R", "R1")
            r2 = _value(env, "R2", "R_2")
            if numerator is None or r2 is None or r2 == 0:
                return _step_result(step, "FAIL", inputs, {}, None, None, errors=[_missing_from_values({"R": numerator, "R2": r2}, env, formula, "Missing R/R1 or R2.")])
            _mark_step9(env, formula)
            value, out_name, unit = numerator / r2, lhs or output or "power_factor", None
        elif formula_l == "z=sqrt(r3^2+(r-r2)^2)":
            r3 = _value(env, "R3", "R_3")
            r_value = _value(env, "R", "R1")
            r2 = _value(env, "R2", "R_2")
            if r3 is None or r_value is None or r2 is None:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_from_values({"R3": r3, "R": r_value, "R2": r2}, env, formula, "Missing R3, R, or R2.")])
            _mark_step9(env, formula)
            value, out_name, unit = _safe_sqrt(r3 * r3 + (r_value - r2) ** 2), lhs or output or "Z", "ohm"
        elif formula_l in {"k_ratio=f2/f", "k_ratio=f2/f_res"}:
            f2 = _value(env, "f2", "f_2")
            f1 = _value(env, "f", "f_res", "f_osc", "frequency")
            if f2 is None or f1 is None or f1 == 0:
                return _step_result(step, "FAIL", inputs, {}, None, None, errors=[_missing_from_values({"f2": f2, "f": f1}, env, formula, "Missing f2 or f.")])
            _mark_step9(env, formula)
            value, out_name, unit = f2 / f1, lhs or output or "k_ratio", None
        elif formula_l == "x_l=sqrt(z_2^2-r^2)/(k_ratio-1/k_ratio)":
            z2 = _value(env, "Z_2", "Z2")
            r_value = _r(env)
            k_ratio = _value(env, "k_ratio")
            if z2 is None or r_value is None or k_ratio is None or k_ratio == 0 or (k_ratio - 1 / k_ratio) == 0:
                return _step_result(step, "FAIL", inputs, {}, None, "ohm", errors=[_missing_from_values({"Z_2": z2, "R": r_value, "k_ratio": k_ratio}, env, formula, "Missing Z_2, R, or k_ratio.")])
            _mark_step9(env, formula)
            value, out_name, unit = _safe_sqrt(z2 * z2 - r_value * r_value) / (k_ratio - 1 / k_ratio), lhs or output or "X_L", "ohm"
        elif formula_l == "q_factor=omega_0*l/r":
            omega0 = _value(env, "omega_0", "omega")
            inductance = _ind(env)
            r_value = _r(env)
            if omega0 is None or inductance is None or r_value is None or r_value == 0:
                return _step_result(step, "FAIL", inputs, {}, None, None, errors=[_missing_from_values({"omega_0": omega0, "L": inductance, "R": r_value}, env, formula, "Missing omega_0, L, or R.")])
            _mark_step9(env, formula)
            value, out_name, unit = omega0 * inductance / r_value, lhs or output or "Q_factor", None
        elif formula_l == "omega=omega*sqrt(r2/r)":
            old_omega = _value(env, "omega", "omega_0")
            r2 = _value(env, "R2", "R_2")
            r_value = _value(env, "R", "R1", "resistance")
            if old_omega is None or r2 is None or r_value is None or r_value == 0:
                return _step_result(step, "FAIL", inputs, {}, None, "rad/s", errors=[_missing_from_values({"omega": old_omega, "R2": r2, "R": r_value}, env, formula, "Missing omega, R2, or R.")])
            new_omega = old_omega * _safe_sqrt(r2 / r_value)
            _mark_step9(env, formula)
            _execution_metadata(env)["omega_scaled_from_previous"] = True
            _set(env, "omega_after", new_omega, "rad/s")
            value, out_name, unit = new_omega, lhs or output or "omega", "rad/s"
        # --- Magnetic field / flux / EMF ---
        elif formula_l in {"n_turns_per_meter=n_turns/l", "n=n_turns/l", "n_turns_per_meter=n/l"}:
            n_turns = _value(env, "n_turns", "N", "turns")
            # L here is a length (unit m), not inductance (unit H).
            length = _value(env, "L_length", "length")
            if length is None and _unit(env, "L") != "H":
                length = _value(env, "L")
            if n_turns is None or length in (None, 0):
                return _step_result(step, "FAIL", inputs, {}, None, "1/m", errors=[_missing_from_values({"n_turns": n_turns, "L_length": length}, env, formula, "Missing n_turns or length L.")])
            _mark_step9(env, formula)
            value, out_name, unit = n_turns / length, lhs or output or "n_turns_per_meter", "1/m"
        elif formula_l in {"b=mu_0*n_turns_per_meter*i", "b=mu_0*n*i"}:
            mu0 = _value(env, "mu_0")
            density = _value(env, "n_turns_per_meter", "n")
            current = _i(env)
            if density is None:
                n_turns = _value(env, "n_turns", "N", "turns")
                length = _value(env, "L_length", "length", "L")
                density = (n_turns / length) if (n_turns is not None and length not in (None, 0)) else None
            if mu0 is None or density is None or current is None:
                return _step_result(step, "FAIL", inputs, {}, None, "T", errors=[_missing_from_values({"n_turns_per_meter": density, "I": current}, env, formula, "Missing turn density or I.")])
            _mark_step9(env, formula)
            value, out_name, unit = mu0 * density * current, lhs or output or "B", "T"
        elif formula_l in {"b=mu_0*n_turns*i/l_length", "b=mu_0*n*i/l", "b=mu_0*n_turns*i/l"}:
            mu0 = _value(env, "mu_0")
            n_turns = _value(env, "n_turns", "N", "turns")
            current = _i(env)
            length = _value(env, "L_length", "length", "L")
            if mu0 is None or n_turns is None or current is None or length in (None, 0):
                return _step_result(step, "FAIL", inputs, {}, None, "T", errors=[_missing_from_values({"n_turns": n_turns, "I": current, "L": length}, env, formula, "Missing n_turns, I, or length.")])
            _mark_step9(env, formula)
            value, out_name, unit = mu0 * n_turns * current / length, lhs or output or "B", "T"
        elif formula_l in {"phi_b=b*a", "phi=b*a", "phi_b=b*a*n_turns"}:
            b = _value(env, "B", "magnetic_field")
            area = _area_m2(env)
            if b is None or area is None:
                return _step_result(step, "FAIL", inputs, {}, None, "Wb", errors=[_missing_from_values({"B": b, "A": area}, env, formula, "Missing B or A.")])
            _mark_step9(env, formula)
            flux = b * area
            if "n_turns" in formula_l:
                n_turns = _value(env, "n_turns", "N", "turns") or 1.0
                flux *= n_turns
            value, out_name, unit = flux, lhs or output or "Phi_B", "Wb"
        elif formula_l in {"emf=l*i/t", "emf=l_ind*i/t", "emf=l_ind*delta_i/t", "emf=l*abs(delta_i)/t", "emf=l_ind*abs(i_final-i_initial)/t", "emf=l*delta_i/t"}:
            inductance = _ind(env)
            t = _value(env, "t", "time", "delta_t")
            delta_i, _src = _parse_current_delta(env, world_input.problem_text)
            if inductance is None or t in (None, 0) or delta_i is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V", errors=[_missing_from_values({"L": inductance, "delta_I": delta_i, "t": t}, env, formula, "Missing L, current change, or t.")])
            _mark_step9(env, formula)
            value, out_name, unit = inductance * abs(delta_i) / t, lhs or output or "emf", "V"
        elif formula_l in {"u_r=i_rms*r", "u_r=i*r", "v_r=i_rms*r"}:
            current = _value(env, "I_rms", "I", "current")
            r_value = _r(env)
            if current is None or r_value is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V", errors=[_missing_from_values({"I_rms": current, "R": r_value}, env, formula, "Missing I_rms or R.")])
            _mark_step9(env, formula)
            value, out_name, unit = current * r_value, lhs or output or "U_R", "V"
        elif formula_l in {"u_c=i_rms*x_c", "v_c=i_rms*x_c"}:
            current = _value(env, "I_rms", "I", "current")
            xc = _value(env, "X_C", "XC")
            if current is None or xc is None:
                return _step_result(step, "FAIL", inputs, {}, None, "V", errors=[_missing_from_values({"I_rms": current, "X_C": xc}, env, formula, "Missing I_rms or X_C.")])
            _mark_step9(env, formula)
            value, out_name, unit = current * xc, lhs or output or "U_C", "V"
        else:
            _mark_unsupported(env, formula)
            return _step_result(step, "WARN", inputs, {}, None, unit, warnings=[f"Unsupported formula dispatch: {formula}"])

        if out_name and value is not None:
            _set(env, out_name, value, unit)
            _mark_executed(env, formula)
            return _step_result(step, "PASS", inputs, {out_name: value}, value, unit)
        return _step_result(step, "WARN", inputs, {}, None, unit, warnings=["Formula matched but produced no output."])
    except Exception as exc:
        return _step_result(
            step,
            "FAIL",
            inputs,
            {},
            None,
            unit,
            errors=[{"error_type": "execution_exception", "severity": "high", "message": repr(exc)}],
        )


def _execute_conclusion_step(step: dict[str, Any], env: dict[str, dict[str, Any]], target: str | None, target_unit: str | None) -> Type2ExecutionStepResult:
    output_var = step.get("output_var") or {}
    candidate_targets = []
    if target:
        candidate_targets.append(target)
    if isinstance(output_var, dict):
        candidate_targets.extend(str(key) for key in output_var.keys())
    for name in candidate_targets:
        key = _normalize_var_name(name)
        if key in env:
            item = env[key]
            return _step_result(step, "PASS", {key: item.get("value")}, {key: item.get("value")}, item.get("value"), item.get("unit") or target_unit)
        value, unit = _resolve_target_value(env, name, target_unit)
        if value is not None:
            key = _normalize_var_name(name)
            return _step_result(step, "PASS", {key: value}, {key: value}, value, unit)
    return _step_result(step, "WARN", {}, {}, None, target_unit, warnings=["Conclusion target not found in computed environment."])


def execute_selected_step_plan(
    world_input: Type2WorldModelInput,
    selected_step_plan: list[dict[str, Any]],
    target: str | None = None,
    target_unit: str | None = None,
) -> Type2NumericExecutionResult:
    target = target or world_input.target
    target_unit = target_unit or world_input.target_unit
    env = _build_numeric_environment(world_input)
    trace: list[Type2ExecutionStepResult] = []
    warnings: list[str] = []
    errors: list[dict[str, Any]] = []
    executed_formula_count = 0

    for step in selected_step_plan:
        if not isinstance(step, dict):
            continue
        if step.get("type") == "formula_application":
            result = _execute_formula_step(step, env, world_input)
            if result.status == "PASS":
                executed_formula_count += 1
            trace.append(result)
        elif step.get("type") == "conclusion":
            trace.append(_execute_conclusion_step(step, env, target, target_unit))
        else:
            trace.append(_step_result(step, "WARN", {}, {}, None, None, warnings=["Non-formula setup step skipped by numeric executor."]))
        warnings.extend(trace[-1].warnings)
        errors.extend(trace[-1].errors)

    should_try_role_aware = _plan_has_coulomb_pairwise(selected_step_plan) and (
        _is_force_like_target(target) or _has_coulomb_missing_errors(errors)
    )
    if should_try_role_aware:
        before_value, _ = _resolve_target_value(env, target, target_unit)
        if before_value is None or _has_coulomb_missing_errors(errors):
            role_result = _execute_role_aware_coulomb_scene(world_input, env, target)
            trace.append(role_result)
            if role_result.status == "PASS":
                executed_formula_count += max(1, int(_execution_metadata(env).get("role_aware_pair_force_count") or 0))
            warnings.extend(role_result.warnings)
            errors.extend(role_result.errors)

    numeric_value, unit = _resolve_target_value(env, target, target_unit)
    if numeric_value is not None and _execution_metadata(env).get("role_aware_coulomb_used"):
        errors = [error for error in errors if error.get("error_type") not in {"missing_inputs", "target_unresolved"}]
        warnings = [warning for warning in warnings if warning != "Conclusion target not found in computed environment."]
    output_unit = unit or _infer_output_unit(target, None, target_unit)
    answer = _format_answer(numeric_value, output_unit) if numeric_value is not None else None
    high_errors = [error for error in errors if error.get("severity") == "high"]
    if numeric_value is not None and not high_errors:
        status = "PASS"
    elif executed_formula_count > 0 or _execution_metadata(env).get("role_aware_coulomb_used"):
        status = "WARN"
    else:
        status = "FAIL"
    if numeric_value is None:
        errors.append({"error_type": "target_unresolved", "severity": "medium", "message": "Numeric executor could not resolve target."})

    return Type2NumericExecutionResult(
        status=status,
        target=target,
        unit=output_unit,
        numeric_value=numeric_value,
        answer=answer,
        execution_trace=trace,
        computed_values={key: value.get("value") for key, value in env.items() if key != "__meta__"},
        warnings=list(dict.fromkeys(warnings)),
        errors=errors,
        metadata={
            "executed_formula_count": executed_formula_count,
            "step_count": len(selected_step_plan),
            **_execution_metadata(env),
        },
    )