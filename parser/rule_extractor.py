"""Deterministic extraction of explicit numeric physics quantities."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from .unit_normalizer import UNIT_DEFINITIONS, get_unit_info, normalize_quantity


SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
DOT_SCI_NUMBER_RE = r"[-+]?(?:\d+\.\d+|\d+)\.10\s*(?:\^\s*[-+]?\d+|[⁻-]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
SCI_NUMBER_RE = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)\s*(?:×|Ã—|x|\*)\s*10\s*(?:\^\s*[-+]?\d+|[⁻-]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
POW10_NUMBER_RE = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)?10\s*(?:\^\s*[-+]?\d+|[⁻-]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
NUMBER_RE = rf"{DOT_SCI_NUMBER_RE}|{SCI_NUMBER_RE}|{POW10_NUMBER_RE}|[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?|[-+]?\d+\s*/\s*\d+"
UNIT_ALIASES = sorted(UNIT_DEFINITIONS, key=len, reverse=True)
UNIT_RE = "|".join(re.escape(unit) for unit in UNIT_ALIASES)

# Stage 0.5: accept clean UTF-8 forms in addition to mojibake variants
# already present in the historical logs.
SUPERSCRIPT_DIGITS = {
    **str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-"),
    ord("¹"): "1",
    ord("²"): "2",
    ord("³"): "3",
}
EXPONENT_RE = r"(?:\^\s*\{?\s*[-+]?\d+\s*\}?|[⁻-]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+|[â»-]?[â°Â¹Â²Â³â´âµâ¶â·â¸â¹]+)"
DOT_SCI_NUMBER_RE = rf"[-+]?(?:\d+\.\d+|\d+)\.10\s*{EXPONENT_RE}"
SCI_NUMBER_RE = rf"[-+]?(?:\d+\.\d+|\d+|\.\d+)\s*(?:×|Ã—|Ãƒâ€”|x|\*|·)\s*10\s*{EXPONENT_RE}"
POW10_NUMBER_RE = rf"[-+]?(?:\d+\.\d+|\d+|\.\d+)?10\s*{EXPONENT_RE}"
NUMBER_RE = rf"{DOT_SCI_NUMBER_RE}|{SCI_NUMBER_RE}|{POW10_NUMBER_RE}|[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?|[-+]?\d+\s*/\s*\d+"


def _parse_number(raw: str) -> float:
    compact = (
        raw.strip()
        .replace(" ", "")
        .replace("{", "")
        .replace("}", "")
        .replace("×", "*")
        .replace("·", "*")
        .translate(SUPERSCRIPT_DIGITS)
    )
    compact = re.sub(r"(?<=\d)\.10(?=\^|[-⁻])", "*10", compact)
    sci_match = re.fullmatch(r"([-+]?(?:\d+\.\d+|\d+|\.\d+))(?:×|Ã—|x|\*)10(?:\^?([-+]?\d+))", compact)
    if sci_match:
        return float(sci_match.group(1)) * (10 ** int(sci_match.group(2)))
    # pow10: a bare-"10^n" or "<coef>10^n" form. The '^' is REQUIRED here,
    # otherwise plain integers like '1000' got greedy-matched as
    # coefficient='' + '10' + exponent='00' → 10**0 == 1 (bug).
    pow10_match = re.fullmatch(r"([-+]?(?:\d+\.\d+|\d+|\.\d+)?)10\^([-+]?\d+)", compact)
    if pow10_match:
        coefficient = pow10_match.group(1)
        if coefficient in {"", "+", "-"}:
            coefficient = f"{coefficient}1"
        return float(coefficient) * (10 ** int(pow10_match.group(2)))
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        return float(numerator.strip()) / float(denominator.strip())
    try:
        return float(raw)
    except ValueError:
        numeric_prefix = re.match(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", raw.strip())
        if numeric_prefix:
            return float(numeric_prefix.group(0))
        raise


def _clean_var(name: str) -> str:
    return name.replace("_", "").strip()


def _relation(
    relation_type: str,
    left: str,
    right: Optional[str],
    source_text: str,
    factor: Optional[float] = None,
    value: Optional[float] = None,
    unit_symbol: Optional[str] = None,
) -> Dict[str, object]:
    return {
        "type": relation_type,
        "left": left,
        "right": right,
        "factor": factor,
        "value": value,
        "unit_symbol": unit_symbol,
        "source_text": source_text.strip(),
    }


def _variable_tokens(expression: str) -> List[str]:
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expression)
    skip = {"sin", "cos", "tan", "sqrt", "exp", "log", "ln", "pi", "abs"}
    return sorted({token for token in tokens if token not in skip and not token[0].isdigit()})


def _function_dimension(name: str, context: str, unit_symbol: Optional[str] = None) -> str:
    raw = name.strip()
    lowered = context.lower()
    if unit_symbol:
        info = get_unit_info(unit_symbol, context)
        if info:
            return str(info["dimension"])
    if raw in {"I", "i"}:
        return "current"
    if raw in {"U", "V"}:
        if any(word in lowered for word in ("energy", "stored", "internal", "potential energy")):
            return "energy"
        return "voltage"
    if raw in {"q", "Q"}:
        return "charge"
    if raw in {"x", "d", "s"}:
        return "length"
    if raw == "v":
        return "velocity"
    if raw == "a":
        return "acceleration"
    if raw == "E":
        if any(word in lowered for word in ("energy", "stored")):
            return "energy"
        return "electric_field"
    if raw == "B":
        return "magnetic_field"
    return "unknown"


def _normalize_expression(expression: str) -> str:
    normalized = expression.strip()
    normalized = normalized.replace("π", "pi").replace("Ï€", "pi")
    normalized = normalized.replace("−", "-").replace("âˆ’", "-")
    normalized = normalized.replace("^", "**")
    normalized = normalized.replace(" ", "")
    normalized = re.sub(r"(?<![A-Za-z])e\{([^{}]+)\}", r"exp(\1)", normalized)
    normalized = re.sub(r"(?<![A-Za-z])e\*\*\{([^{}]+)\}", r"exp(\1)", normalized)
    normalized = re.sub(r"(?<![A-Za-z])e\*\*\(([^()]+)\)", r"exp(\1)", normalized)
    normalized = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", normalized)
    normalized = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", normalized)
    return normalized


def _base_variable_for_dimension(dimension: str, context: str) -> str:
    text = context.lower()
    if dimension == "area":
        return "A"
    if dimension == "mass":
        return "m_object" if "block" in text or "car" in text else "m"
    if dimension == "length":
        if re.search(r"\bR\s*=?\s*$|\bR\s*=", context) and any(token in text for token in ("radius", "circular", "circle", "disk", "sphere")):
            return "R_radius"
        if any(token in text for token in ("radius", "circular", "circle", "disk", "sphere")):
            return "r"
        if "height" in text:
            return "h"
        if "wavelength" in text:
            return "lambda"
        if "length" in text:
            return "L"
        return "d"
    if dimension == "time":
        return "t"
    if dimension == "velocity":
        if "wave" in text:
            return "v_wave"
        return "v"
    if dimension == "acceleration":
        return "a"
    if dimension == "force":
        return "F"
    if dimension == "charge":
        if "maximum charge" in text:
            return "Q_max"
        label = re.search(r"\b(q\d+|q|Q)\s*=?\s*$", context)
        return label.group(1) if label else "q"
    if dimension == "voltage":
        if "rms" in text or "root-mean-square" in text:
            return "V_rms"
        return "V"
    if dimension == "current":
        if "maximum current" in text:
            return "I_max"
        if "rms" in text or "root-mean-square" in text:
            return "I_rms"
        return "I"
    if dimension == "resistance":
        label = re.search(r"\b(R\d+|R)\s*=?\s*$", context)
        return label.group(1) if label else "R"
    if dimension == "capacitance":
        return "C_cap"
    if dimension == "inductance":
        return "L_ind"
    if dimension == "frequency":
        return "f"
    if dimension == "angular_frequency":
        return "omega"
    if dimension == "electric_field":
        return "E"
    if dimension == "magnetic_field":
        return "B"
    if dimension == "energy":
        if re.search(r"\bU\s*=?\s*$|\benergy\s+U\b", context, re.IGNORECASE):
            return "U_E"
        if "kinetic" in text:
            return "KE"
        if "potential" in text:
            return "PE"
        return "E_energy"
    if dimension == "power":
        return "P"
    if dimension == "angle":
        return "theta"
    if dimension == "pressure":
        if "atmospheric" in text:
            return "p_atm"
        return "p_pressure"
    if dimension == "volume":
        return "V_volume"
    if dimension == "specific_heat_capacity":
        if "water" in text:
            return "c_water"
        if "ice" in text:
            return "c_ice"
        if "copper" in text:
            return "c_copper"
        if "aluminum" in text or "aluminium" in text:
            return "c_aluminum"
        return "c_heat"
    if dimension == "specific_latent_heat":
        if "fusion" in text or "melt" in text:
            return "L_fusion"
        if "vapor" in text or "boil" in text:
            return "L_vapor"
        return "L_latent"
    if dimension == "count":
        if "turn" in text:
            return "n_turns"
        if "oscillat" in text:
            return "n_osc"
        return "n_count"
    if dimension == "turn_density":
        return "n_turns_per_meter"
    return dimension


def _dedupe_name(base: str, counts: Dict[str, int], existing: Iterable[str]) -> str:
    if re.search(r"\d+$", base) and base not in existing:
        counts[base] += 1
        return base
    counts[base] += 1
    if counts[base] == 1 and base not in existing:
        return base
    return f"{base}{counts[base]}"


def _window(text: str, start: int, end: int, radius: int = 40) -> str:
    return text[max(0, start - radius): min(len(text), end + radius)]


def extract_quantities(problem_text: str) -> Dict[str, Dict[str, object]]:
    """Extract explicit numeric quantities and infer variable names from local context."""
    if not isinstance(problem_text, str):
        raise TypeError("problem_text must be a string")

    quantities: Dict[str, Dict[str, object]] = {}
    counts: Dict[str, int] = defaultdict(int)
    pattern = re.compile(
        rf"(?<![A-Za-z_])(?P<value>{NUMBER_RE})\s*\(?(?P<unit>{UNIT_RE})\)?(?![A-Za-z/])",
        re.IGNORECASE,
    )

    for match in pattern.finditer(problem_text):
        raw_value = match.group("value")
        raw_unit = match.group("unit")
        context = _window(problem_text, match.start(), match.end())
        info = get_unit_info(raw_unit, context)
        if not info:
            continue
        value = _parse_number(raw_value)
        normalized_value, normalized_unit = normalize_quantity(value, raw_unit, context)
        dimension = str(info["dimension"])
        local_context = problem_text[max(0, match.start() - 30): min(len(problem_text), match.end() + 30)]
        before_value = problem_text[max(0, match.start() - 30): match.start()]
        naming_context = before_value if dimension in {"charge", "resistance"} else local_context
        base_name = _base_variable_for_dimension(dimension, naming_context)
        name = _dedupe_name(base_name, counts, quantities)
        quantities[name] = {
            "value": value,
            "unit_symbol": str(info["unit_symbol"]),
            "unit_name": str(info["unit_name"]),
            "dimension": str(info["dimension"]),
            "source_text": match.group(0).strip(),
            "normalized_value": normalized_value,
            "normalized_unit_symbol": normalized_unit,
        }

    _extract_dimensionless_quantities(problem_text, quantities, counts)
    _extract_named_frequency_quantities(problem_text, quantities)
    _extract_ratio_quantities(problem_text, quantities, counts)
    _extract_error_measurements(problem_text, quantities, counts)
    _extract_uncertainty_quantities(problem_text, quantities)
    return quantities


def extract_relations(problem_text: str) -> List[Dict[str, object]]:
    """Extract symbolic relations such as ratios, uncertainties, ranges, and coordinates."""
    relations: List[Dict[str, object]] = []
    relations.extend(_extract_function_relations(problem_text))
    relations.extend(_extract_symbolic_ratio_relations(problem_text))
    relations.extend(_extract_word_ratio_relations(problem_text))
    relations.extend(_extract_fraction_relations(problem_text))
    relations.extend(_extract_percentage_relations(problem_text))
    relations.extend(_extract_uncertainty_relations(problem_text))
    relations.extend(_extract_equation_relations(problem_text))
    relations.extend(_extract_motion_relations(problem_text))
    relations.extend(_extract_range_relations(problem_text))
    relations.extend(_extract_coordinate_relations(problem_text))
    return relations


def _add_dimensionless(
    quantities: Dict[str, Dict[str, object]],
    counts: Dict[str, int],
    base_name: str,
    value: float,
    source_text: str,
) -> None:
    name = _dedupe_name(base_name, counts, quantities)
    quantities[name] = {
        "value": value,
        "unit_symbol": "",
        "unit_name": "dimensionless",
        "dimension": "dimensionless",
        "source_text": source_text.strip(),
        "normalized_value": value,
        "normalized_unit_symbol": None,
    }


def _add_named_quantity(
    quantities: Dict[str, Dict[str, object]],
    name: str,
    value: float,
    unit_symbol: str,
    unit_name: str,
    dimension: str,
    source_text: str,
) -> None:
    normalized_value, normalized_unit = normalize_quantity(value, unit_symbol, source_text) if unit_symbol else (value, None)
    quantities[name] = {
        "value": value,
        "unit_symbol": unit_symbol,
        "unit_name": unit_name,
        "dimension": dimension,
        "source_text": source_text.strip(),
        "normalized_value": normalized_value,
        "normalized_unit_symbol": normalized_unit,
    }


def _extract_named_frequency_quantities(problem_text: str, quantities: Dict[str, Dict[str, object]]) -> None:
    """Extract named f/omega assignments that may omit an explicit unit."""
    patterns = [
        ("omega", "rad/s", "radian per second", "angular_frequency", rf"(?:angular frequency|omega|ω)\s*(?:=|is)?\s*(?P<value>{NUMBER_RE})"),
        ("f", "Hz", "hertz", "frequency", rf"(?:\bfrequency\b|\bf\b)\s*(?:=|is)?\s*(?P<value>{NUMBER_RE})\s*(?:Hz|hertz)?"),
    ]
    for name, unit_symbol, unit_name, dimension, pattern in patterns:
        if name in quantities:
            continue
        for match in re.finditer(pattern, problem_text, re.IGNORECASE):
            try:
                value = _parse_number(match.group("value"))
            except ValueError:
                continue
            _add_named_quantity(quantities, name, value, unit_symbol, unit_name, dimension, match.group(0))
            break


def _extract_dimensionless_quantities(
    problem_text: str,
    quantities: Dict[str, Dict[str, object]],
    counts: Dict[str, int],
) -> None:
    """Extract common dimensionless constants/factors that affect templates."""
    rules = [
        ("epsilon_r", rf"(?:dielectric constant|relative permittivity|epsilon_r|kappa|κ|Îµ_r|ε_r|Îµ|ε)(?:\s+(?!determine\b|calculate\b|find\b)[A-Za-z]+){{0,8}}\s*(?:=|is|of)?\s*(?P<value>{NUMBER_RE})"),
        ("factor", rf"(?:factor of|by a factor of|increases by|decreases by)\s*(?P<value>{NUMBER_RE})"),
        ("k", rf"\bk\s*=\s*(?P<value>{NUMBER_RE})"),
    ]
    occupied_spans = [quantity.get("source_text", "") for quantity in quantities.values()]
    for base_name, pattern in rules:
        for match in re.finditer(pattern, problem_text, re.IGNORECASE):
            source = match.group(0)
            if any(source in str(existing) or str(existing) in source for existing in occupied_spans if existing):
                continue
            try:
                value = _parse_number(match.group("value"))
            except ValueError:
                continue
            _add_dimensionless(quantities, counts, base_name, value, source)


def _extract_ratio_quantities(
    problem_text: str,
    quantities: Dict[str, Dict[str, object]],
    counts: Dict[str, int],
) -> None:
    """Extract simple LC charge-to-maximum-charge ratios."""
    lowered = problem_text.lower()
    ratio_rules = [
        (r"half (?:the )?maximum charge", 0.5),
        (r"one third of (?:the )?maximum charge", 1.0 / 3.0),
        (r"one fourth of (?:the )?maximum charge", 0.25),
        (r"one quarter of (?:the )?maximum charge", 0.25),
        (r"charge\s+is\s+q\s*=\s*q\s*/\s*2", 0.5),
        (r"charge\s+is\s+q\s*=\s*Q\s*/\s*2", 0.5),
    ]
    for pattern, value in ratio_rules:
        match = re.search(pattern, lowered)
        if match:
            _add_dimensionless(quantities, counts, "q_over_Qmax", value, problem_text[match.start():match.end()])
            return
    numeric = re.search(r"charge\s+is\s+(?P<value>\d+(?:\.\d+)?)\s+of (?:the )?maximum charge", lowered)
    if numeric:
        _add_dimensionless(quantities, counts, "q_over_Qmax", float(numeric.group("value")), problem_text[numeric.start():numeric.end()])


def _extract_error_measurements(
    problem_text: str,
    quantities: Dict[str, Dict[str, object]],
    counts: Dict[str, int],
) -> None:
    """Add measured/true/accepted aliases for error-analysis templates."""
    label_map = {
        "measured": "measured_value",
        "true": "true_value",
        "accepted": "accepted_value",
        "actual": "true_value",
    }
    pattern = re.compile(
        rf"\b(?P<label>measured|true|accepted|actual)(?:\s+\w+){{0,3}}\s+(?:is|was|=)\s*(?P<value>{NUMBER_RE})\s*\(?(?P<unit>{UNIT_RE})?\)?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(problem_text):
        raw_unit = match.group("unit") or ""
        info = get_unit_info(raw_unit, match.group(0)) if raw_unit else None
        value = _parse_number(match.group("value"))
        normalized_value, normalized_unit = normalize_quantity(value, raw_unit, match.group(0)) if info else (value, None)
        name = label_map[match.group("label").lower()]
        quantities[name] = {
            "value": value,
            "unit_symbol": str(info["unit_symbol"]) if info else raw_unit,
            "unit_name": str(info["unit_name"]) if info else "unknown",
            "dimension": str(info["dimension"]) if info else "unknown",
            "source_text": match.group(0).strip(),
            "normalized_value": normalized_value,
            "normalized_unit_symbol": normalized_unit,
        }


def _symbol_to_var(symbol: str, context: str = "") -> str:
    raw = symbol.strip().replace("_", "")
    mapping = {"U": "V", "u": "V", "V": "V", "I": "I", "R": "R", "Q": "Q", "q": "q"}
    if raw in mapping:
        return mapping[raw]
    if raw.lower().startswith("u") and raw[1:].isdigit():
        return f"V{raw[1:]}"
    return raw


def _delta_name(var_name: str) -> str:
    if var_name == "V":
        return "delta_V"
    return f"delta_{var_name}"


def _extract_uncertainty_quantities(problem_text: str, quantities: Dict[str, Dict[str, object]]) -> None:
    """Extract X and delta_X from forms like U = 6.0 ± 0.1 V."""
    pattern = re.compile(
        rf"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s*(?:=|is|was)?\s*(?P<value>{NUMBER_RE})\s*(?:±|\+/-|plus or minus)\s*(?P<delta>{NUMBER_RE})\s*\(?(?P<unit>{UNIT_RE})\)?",
        re.IGNORECASE,
    )
    label_pattern = re.compile(
        rf"\b(?P<label>voltage|current|resistance)\s+(?:is|was)\s*(?P<value>{NUMBER_RE})\s*(?:±|\+/-|plus or minus)\s*(?P<delta>{NUMBER_RE})\s*\(?(?P<unit>{UNIT_RE})\)?",
        re.IGNORECASE,
    )
    for pattern_obj in [pattern, label_pattern]:
        for match in pattern_obj.finditer(problem_text):
            symbol = match.groupdict().get("symbol")
            if not symbol:
                symbol = {"voltage": "V", "current": "I", "resistance": "R"}[match.group("label").lower()]
            var_name = _symbol_to_var(symbol, match.group(0))
            unit = match.group("unit")
            info = get_unit_info(unit, match.group(0))
            if not info:
                continue
            value = _parse_number(match.group("value"))
            delta = _parse_number(match.group("delta"))
            normalized_value, normalized_unit = normalize_quantity(value, unit, match.group(0))
            normalized_delta, _ = normalize_quantity(delta, unit, match.group(0))
            quantities[var_name] = {
                "value": value,
                "unit_symbol": str(info["unit_symbol"]),
                "unit_name": str(info["unit_name"]),
                "dimension": str(info["dimension"]),
                "source_text": match.group(0).strip(),
                "normalized_value": normalized_value,
                "normalized_unit_symbol": normalized_unit,
            }
            quantities[_delta_name(var_name)] = {
                "value": delta,
                "unit_symbol": str(info["unit_symbol"]),
                "unit_name": str(info["unit_name"]),
                "dimension": str(info["dimension"]),
                "source_text": match.group(0).strip(),
                "normalized_value": normalized_delta,
                "normalized_unit_symbol": normalized_unit,
            }


def _extract_function_relations(problem_text: str) -> List[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    pattern = re.compile(
        rf"\b(?P<name>I|U|V|q|Q|x|d|s|v|a|E|B)\s*\(\s*(?P<ivar>[A-Za-z])\s*\)\s*=\s*(?P<expr>[^.;,\n?]+?)(?:\s*(?P<unit>{UNIT_RE}))?(?=$|[.;,\n?])",
        re.IGNORECASE,
    )
    for match in pattern.finditer(problem_text):
        name = match.group("name")
        unit = match.group("unit")
        expr = match.group("expr").strip()
        if unit and expr.endswith(unit):
            expr = expr[: -len(unit)].strip()
        source = match.group(0).strip()
        relations.append(
            {
                "type": "function",
                "function_name": name,
                "independent_var": match.group("ivar"),
                "expression": _normalize_expression(expr),
                "dimension": _function_dimension(name, _window(problem_text, match.start(), match.end(), 80), unit),
                "unit_symbol": str(get_unit_info(unit, source)["unit_symbol"]) if unit and get_unit_info(unit, source) else unit,
                "source_text": source,
            }
        )
    return relations


def _extract_percentage_relations(problem_text: str) -> List[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    pattern = re.compile(rf"(?P<value>{NUMBER_RE})\s*(?:%|percent\b|per cent\b)", re.IGNORECASE)
    for match in pattern.finditer(problem_text):
        raw_percent = _parse_number(match.group("value"))
        context = _window(problem_text, match.start(), match.end(), 50).lower()
        quantity = "percentage"
        if "efficien" in context:
            quantity = "efficiency"
        elif "uncertainty" in context:
            quantity = "uncertainty"
        elif "error" in context:
            quantity = "error"
        relations.append(
            {
                "type": "percentage",
                "quantity": quantity,
                "value": raw_percent / 100.0,
                "raw_percent": raw_percent,
                "source_text": match.group(0).strip(),
            }
        )
    return relations


def _extract_fraction_relations(problem_text: str) -> List[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    for match in re.finditer(r"(?<!\d)(?P<frac>-?\d+\s*/\s*\d+)(?!\d)", problem_text):
        value = _parse_number(match.group("frac"))
        context = _window(problem_text, match.start(), match.end(), 50).lower()
        quantity = "fraction"
        if "electric" in context and "energy" in context:
            quantity = "electric_energy_fraction"
        elif "magnetic" in context and "energy" in context:
            quantity = "magnetic_energy_fraction"
        elif "time" in context:
            quantity = "time_fraction"
        elif "distance" in context:
            quantity = "distance_fraction"
        relations.append(
            {
                "type": "percentage",
                "quantity": quantity,
                "value": value,
                "raw_percent": value * 100.0,
                "source_text": match.group(0).strip(),
            }
        )
    return relations


def _extract_symbolic_ratio_relations(problem_text: str) -> List[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    var = r"[A-Za-z][A-Za-z_]*\d*"
    patterns = [
        re.compile(rf"\b(?P<left>{var})\s*=\s*(?P<factor>{NUMBER_RE})\s*(?P<right>{var})\b"),
        re.compile(rf"\b(?P<left>{var})\s*=\s*(?P<right>{var})\s*/\s*(?P<den>{NUMBER_RE})\b"),
        re.compile(rf"\b(?P<right>{var})\s*=\s*(?P<left>{var})\s*/\s*(?P<den>{NUMBER_RE})\b"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(problem_text):
            left = _clean_var(match.group("left"))
            right = _clean_var(match.group("right"))
            if "den" in match.groupdict() and match.group("den"):
                factor = 1.0 / _parse_number(match.group("den"))
            else:
                factor = _parse_number(match.group("factor"))
            relations.append(_relation("ratio", left, right, match.group(0), factor=factor))
    return relations


def _extract_word_ratio_relations(problem_text: str) -> List[Dict[str, object]]:
    lowered = problem_text.lower()
    relations: List[Dict[str, object]] = []
    phrase_rules = [
        (r"the electric energy is 1/4 of the total energy", "U_E", "U_total", 0.25),
        (r"electric energy is 1/4 of total energy", "U_E", "U_total", 0.25),
        (r"the electric energy is one fourth of the total energy", "U_E", "U_total", 0.25),
        (r"the charge is half the maximum charge", "q", "Q_max", 0.5),
        (r"charge is half the maximum charge", "q", "Q_max", 0.5),
        (r"half the maximum charge", "q", "Q_max", 0.5),
        (r"one third of the total energy", "energy_fraction", "U_total", 1.0 / 3.0),
        (r"one fourth of total energy", "energy_fraction", "U_total", 0.25),
        (r"1/4 of total energy", "energy_fraction", "U_total", 0.25),
    ]
    for pattern, left, right, factor in phrase_rules:
        match = re.search(pattern, lowered)
        if match:
            relations.append(_relation("ratio", left, right, problem_text[match.start():match.end()], factor=factor))
    generic = re.search(r"\b(twice|three times|half) the (charge|resistance|total energy|maximum charge)\b", lowered)
    if generic:
        factor = {"twice": 2.0, "three times": 3.0, "half": 0.5}[generic.group(1)]
        right = {"charge": "q", "resistance": "R", "total energy": "U_total", "maximum charge": "Q_max"}[generic.group(2)]
        relations.append(_relation("ratio", right, right, problem_text[generic.start():generic.end()], factor=factor))
    word_factor = {
        "twice": 2.0,
        "double": 2.0,
        "triple": 3.0,
        "half": 0.5,
        "one half": 0.5,
        "three times": 3.0,
        "four times": 4.0,
        "one third": 1.0 / 3.0,
        "one third of": 1.0 / 3.0,
        "two thirds": 2.0 / 3.0,
        "quarter": 0.25,
        "one quarter": 0.25,
        "three quarters": 0.75,
    }
    ordinal_var = {"first": "q1", "second": "q2", "third": "q3"}
    for phrase, factor in word_factor.items():
        match = re.search(rf"\b(?:q1|the first charge|the charge q1)\s+is\s+{re.escape(phrase)}\s+(?:of\s+)?(?:q2|the second|the charge q2)\b", lowered)
        if match:
            relations.append(_relation("ratio", "q1", "q2", problem_text[match.start():match.end()], factor=factor))
        match = re.search(rf"\b(?P<left>[A-Za-z]\d*)\s+is\s+{re.escape(phrase)}\s+(?:of\s+)?(?P<right>[A-Za-z]\d*)\b", lowered)
        if match:
            relations.append(_relation("ratio", _clean_var(match.group("left")), _clean_var(match.group("right")), problem_text[match.start():match.end()], factor=factor))
        match = re.search(rf"\bthe\s+(first|second|third)\s+(?:charge|resistor|capacitor)\s+is\s+{re.escape(phrase)}\s+(?:of\s+)?the\s+(first|second|third)\b", lowered)
        if match and match.group(1) in ordinal_var and match.group(2) in ordinal_var:
            relations.append(_relation("ratio", ordinal_var[match.group(1)], ordinal_var[match.group(2)], problem_text[match.start():match.end()], factor=factor))
    return relations


def _extract_uncertainty_relations(problem_text: str) -> List[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    pattern = re.compile(
        rf"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s*(?:=|is|was)?\s*(?P<value>{NUMBER_RE})\s*(?:±|\+/-|plus or minus)\s*(?P<delta>{NUMBER_RE})\s*\(?(?P<unit>{UNIT_RE})\)?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(problem_text):
        var_name = _symbol_to_var(match.group("symbol"), match.group(0))
        relations.append(
            _relation(
                "uncertainty",
                var_name,
                _delta_name(var_name),
                match.group(0),
                value=_parse_number(match.group("delta")),
                unit_symbol=match.group("unit"),
            )
        )
    return relations


def _extract_range_relations(problem_text: str) -> List[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    pattern = re.compile(rf"\bfrom\s+(?P<start>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})?\s+to\s+(?P<end>{NUMBER_RE})\s*(?P<unit2>{UNIT_RE})?", re.IGNORECASE)
    for match in pattern.finditer(problem_text):
        relations.append(_relation("range", "range_start", "range_end", match.group(0), value=_parse_number(match.group("start")), unit_symbol=match.group("unit") or match.group("unit2")))
    return relations


def _extract_coordinate_relations(problem_text: str) -> List[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    pattern = re.compile(r"\(\s*(?P<x>[-+]?\d+(?:\.\d+)?)\s*,\s*(?P<y>[-+]?\d+(?:\.\d+)?)\s*\)")
    for match in pattern.finditer(problem_text):
        relations.append(_relation("coordinate", "x", "y", match.group(0), value=_parse_number(match.group("x"))))
    return relations


def _extract_uncertainty_relations(problem_text: str) -> List[Dict[str, object]]:  # type: ignore[no-redef]
    """Extract absolute and relative uncertainty relations in a Stage 0.6 shape."""
    relations: List[Dict[str, object]] = []
    patterns = [
        re.compile(
            rf"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s*(?:=|is|was)?\s*(?P<value>{NUMBER_RE})\s*(?:±|Â±|\+/-|plus or minus)\s*(?P<delta>{NUMBER_RE})\s*\(?(?P<unit>{UNIT_RE})\)?",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s*=\s*\(\s*(?P<value>{NUMBER_RE})\s*(?:±|Â±|\+/-|plus or minus)\s*(?P<delta>{NUMBER_RE})\s*\)\s*(?P<unit>{UNIT_RE})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<label>voltage|current|resistance|length|mass|force)\s+(?:=|is|was)?\s*(?P<value>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})\s*(?:±|Â±|\+/-|plus or minus)\s*(?P<delta>{NUMBER_RE})\s*(?P<unit2>{UNIT_RE})?",
            re.IGNORECASE,
        ),
    ]
    label_symbols = {"voltage": "V", "current": "I", "resistance": "R", "length": "d", "mass": "m", "force": "F"}
    for pattern in patterns:
        for match in pattern.finditer(problem_text):
            symbol = match.groupdict().get("symbol") or label_symbols.get(match.groupdict().get("label", "").lower(), "uncertainty")
            var_name = _symbol_to_var(symbol, match.group(0))
            unit = match.groupdict().get("unit2") or match.group("unit")
            relations.append(
                {
                    "type": "uncertainty",
                    "quantity": var_name,
                    "value": _parse_number(match.group("value")),
                    "uncertainty": _parse_number(match.group("delta")),
                    "unit_symbol": unit,
                    "relative_uncertainty": None,
                    "source_text": match.group(0).strip(),
                    "left": var_name,
                    "right": _delta_name(var_name),
                }
            )
    percent_pattern = re.compile(
        rf"\b(?:relative|percentage|percent)?\s*uncertainty(?:\s+(?:of|in)\s+(?P<label>voltage|current|resistance|length|mass|force))?\s*(?:is|=)?\s*(?P<percent>{NUMBER_RE})\s*(?:%|percent\b|per cent\b)",
        re.IGNORECASE,
    )
    for match in percent_pattern.finditer(problem_text):
        label = match.group("label")
        var_name = _symbol_to_var(label_symbols.get(label.lower(), "uncertainty"), match.group(0)) if label else "uncertainty"
        raw_percent = _parse_number(match.group("percent"))
        relations.append(
            {
                "type": "uncertainty",
                "quantity": var_name,
                "value": None,
                "uncertainty": None,
                "unit_symbol": None,
                "relative_uncertainty": raw_percent / 100.0,
                "source_text": match.group(0).strip(),
                "left": var_name,
                "right": _delta_name(var_name),
            }
        )
    return relations


def _extract_equation_relations(problem_text: str) -> List[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    equation_re = re.compile(
        r"\b(?P<equation>[A-Za-z][A-Za-z0-9_]*(?:\s*[*+\-/]\s*[A-Za-z0-9_]+)*\s*(?:\+|\-)\s*[A-Za-z][A-Za-z0-9_]*(?:\s*[*+\-/]\s*[A-Za-z0-9_]+)*\s*=\s*[A-Za-z0-9_]+(?:\s*[*+\-/]\s*[A-Za-z0-9_]+)*)"
    )
    ratio_like = re.compile(rf"^\s*[A-Za-z][A-Za-z_]*\d*\s*=\s*(?:{NUMBER_RE})\s*[A-Za-z][A-Za-z_]*\d*\s*$")
    for match in equation_re.finditer(problem_text):
        equation = match.group("equation").strip()
        if ratio_like.match(equation):
            continue
        relations.append(
            {
                "type": "equation",
                "equation": equation,
                "variables": _variable_tokens(equation),
                "source_text": equation,
            }
        )
    return relations


def _extract_motion_relations(problem_text: str) -> List[Dict[str, object]]:
    lowered = problem_text.lower()
    relations: List[Dict[str, object]] = []
    if any(phrase in lowered for phrase in ("move toward each other", "move towards each other", "toward each other", "towards each other", "meet after")):
        relations.append({"type": "equation", "equation": "d = (v1 + v2) * t", "variables": ["d", "v1", "v2", "t"], "source_text": "two vehicles meet after time t"})
    if any(phrase in lowered for phrase in ("catch up", "catches up", "overtakes", "same direction")):
        relations.append({"type": "equation", "equation": "d = abs(v1 - v2) * t", "variables": ["d", "v1", "v2", "t"], "source_text": "same-direction chasing relation"})
    if "downstream" in lowered or "upstream" in lowered:
        relations.append({"type": "equation", "equation": "v_down = v_boat + v_current", "variables": ["v_down", "v_boat", "v_current"], "source_text": "downstream speed relation"})
        relations.append({"type": "equation", "equation": "v_up = v_boat - v_current", "variables": ["v_up", "v_boat", "v_current"], "source_text": "upstream speed relation"})
    if "average speed" in lowered:
        relations.append({"type": "equation", "equation": "v_avg = total_distance / total_time", "variables": ["v_avg", "total_distance", "total_time"], "source_text": "average speed relation"})
    return relations


def _extract_uncertainty_quantities(problem_text: str, quantities: Dict[str, Dict[str, object]]) -> None:  # type: ignore[no-redef]
    """Extract measured values and absolute uncertainties into known_quantities."""
    patterns = [
        re.compile(
            rf"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s*(?:=|is|was)?\s*(?P<value>{NUMBER_RE})\s*(?:±|Â±|\+/-|plus or minus)\s*(?P<delta>{NUMBER_RE})\s*\(?(?P<unit>{UNIT_RE})\)?",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s*=\s*\(\s*(?P<value>{NUMBER_RE})\s*(?:±|Â±|\+/-|plus or minus)\s*(?P<delta>{NUMBER_RE})\s*\)\s*(?P<unit>{UNIT_RE})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<label>voltage|current|resistance|length|mass|force)\s+(?:=|is|was)?\s*(?P<value>{NUMBER_RE})\s*(?P<unit>{UNIT_RE})\s*(?:±|Â±|\+/-|plus or minus)\s*(?P<delta>{NUMBER_RE})\s*(?P<unit2>{UNIT_RE})?",
            re.IGNORECASE,
        ),
    ]
    label_symbols = {"voltage": "V", "current": "I", "resistance": "R", "length": "d", "mass": "m", "force": "F"}
    for pattern in patterns:
        for match in pattern.finditer(problem_text):
            symbol = match.groupdict().get("symbol") or label_symbols.get(match.groupdict().get("label", "").lower(), "uncertainty")
            var_name = _symbol_to_var(symbol, match.group(0))
            unit = match.groupdict().get("unit2") or match.group("unit")
            info = get_unit_info(unit, match.group(0))
            if not info:
                continue
            value = _parse_number(match.group("value"))
            delta = _parse_number(match.group("delta"))
            normalized_value, normalized_unit = normalize_quantity(value, unit, match.group(0))
            normalized_delta, _ = normalize_quantity(delta, unit, match.group(0))
            quantities[var_name] = {
                "value": value,
                "unit_symbol": str(info["unit_symbol"]),
                "unit_name": str(info["unit_name"]),
                "dimension": str(info["dimension"]),
                "source_text": match.group(0).strip(),
                "normalized_value": normalized_value,
                "normalized_unit_symbol": normalized_unit,
            }
            quantities[_delta_name(var_name)] = {
                "value": delta,
                "unit_symbol": str(info["unit_symbol"]),
                "unit_name": str(info["unit_name"]),
                "dimension": str(info["dimension"]),
                "source_text": match.group(0).strip(),
                "normalized_value": normalized_delta,
                "normalized_unit_symbol": normalized_unit,
            }


def _extract_function_relations(problem_text: str) -> List[Dict[str, object]]:  # type: ignore[no-redef]
    """Extract function-style physical quantities without treating decimal dots as sentence stops."""
    relations: List[Dict[str, object]] = []
    head_re = re.compile(r"\b(?P<name>I|U|V|q|Q|x|d|s|v|a|E|B)\s*\(\s*(?P<ivar>[A-Za-z])\s*\)\s*=", re.IGNORECASE)
    unit_tail_re = re.compile(rf"^(?P<expr>.+?)\s+(?P<unit>{UNIT_RE})$", re.IGNORECASE)
    for head in head_re.finditer(problem_text):
        cursor = head.end()
        end = len(problem_text)
        index = cursor
        while index < len(problem_text):
            char = problem_text[index]
            if char in {";", "\n", "?"}:
                end = index
                break
            if char == "." and (index + 1 == len(problem_text) or problem_text[index + 1].isspace()):
                end = index
                break
            index += 1
        body = problem_text[cursor:end].strip()
        unit = None
        expr = body
        unit_match = unit_tail_re.match(body)
        if unit_match:
            expr = unit_match.group("expr").strip()
            unit = unit_match.group("unit")
        source = problem_text[head.start():end].strip()
        info = get_unit_info(unit, source) if unit else None
        relations.append(
            {
                "type": "function",
                "function_name": head.group("name"),
                "independent_var": head.group("ivar"),
                "expression": _normalize_expression(expr),
                "dimension": _function_dimension(head.group("name"), _window(problem_text, head.start(), end, 80), unit),
                "unit_symbol": str(info["unit_symbol"]) if info else unit,
                "source_text": source,
            }
        )
    assignment_re = re.compile(
        rf"\b(?P<name>I|i|U|V|q|Q|x|d|s|v|a|E|B)\s*=\s*(?P<expr>(?:(?![.;?\n]).)*(?:sin|cos|tan|exp|Ï€|π|omega|ω|t)(?:(?![.;?\n]).)*?)(?:\s+(?P<unit>{UNIT_RE}))?(?=$|[.;?\n])",
        re.IGNORECASE,
    )
    for match in assignment_re.finditer(problem_text):
        if "(" in problem_text[max(0, match.start() - 3):match.start() + 3]:
            continue
        unit = match.group("unit")
        expr = match.group("expr").strip()
        if unit and expr.endswith(unit):
            expr = expr[: -len(unit)].strip()
        source = match.group(0).strip()
        info = get_unit_info(unit, source) if unit else None
        relations.append(
            {
                "type": "function",
                "function_name": match.group("name"),
                "independent_var": "t",
                "expression": _normalize_expression(expr),
                "dimension": _function_dimension(match.group("name"), _window(problem_text, match.start(), match.end(), 80), unit),
                "unit_symbol": str(info["unit_symbol"]) if info else unit,
                "source_text": source,
            }
        )
    return relations