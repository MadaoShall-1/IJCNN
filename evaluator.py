from __future__ import annotations

import math
import re
from typing import Any


UNIT_ALIASES: dict[str, tuple[str, float]] = {
    "": ("", 1.0),
    "-": ("dimensionless", 1.0),
    "none": ("dimensionless", 1.0),
    "dimensionless": ("dimensionless", 1.0),
    "ohm": ("ohm", 1.0),
    "ohms": ("ohm", 1.0),
    "omega": ("ohm", 1.0),
    "ω": ("ohm", 1.0),
    "Ω": ("ohm", 1.0),
    "kohm": ("ohm", 1e3),
    "kω": ("ohm", 1e3),
    "kΩ": ("ohm", 1e3),
    "v": ("V", 1.0),
    "volt": ("V", 1.0),
    "volts": ("V", 1.0),
    "mv": ("V", 1e-3),
    "kv": ("V", 1e3),
    "a": ("A", 1.0),
    "amp": ("A", 1.0),
    "ampere": ("A", 1.0),
    "amperes": ("A", 1.0),
    "ma": ("A", 1e-3),
    "μa": ("A", 1e-6),
    "µa": ("A", 1e-6),
    "ua": ("A", 1e-6),
    "j": ("J", 1.0),
    "joule": ("J", 1.0),
    "joules": ("J", 1.0),
    "mj": ("J", 1e-3),
    "μj": ("J", 1e-6),
    "µj": ("J", 1e-6),
    "uj": ("J", 1e-6),
    "microjoule": ("J", 1e-6),
    "microjoules": ("J", 1e-6),
    "nj": ("J", 1e-9),
    "kj": ("J", 1e3),
    "w": ("W", 1.0),
    "watt": ("W", 1.0),
    "watts": ("W", 1.0),
    "mw": ("W", 1e-3),
    "kw": ("W", 1e3),
    "f": ("F", 1.0),
    "farad": ("F", 1.0),
    "farads": ("F", 1.0),
    "mf": ("F", 1e-3),
    "μf": ("F", 1e-6),
    "µf": ("F", 1e-6),
    "uf": ("F", 1e-6),
    "microfarad": ("F", 1e-6),
    "microfarads": ("F", 1e-6),
    "nf": ("F", 1e-9),
    "pf": ("F", 1e-12),
    "c": ("C", 1.0),
    "coulomb": ("C", 1.0),
    "coulombs": ("C", 1.0),
    "mc": ("C", 1e-3),
    "μc": ("C", 1e-6),
    "µc": ("C", 1e-6),
    "uc": ("C", 1e-6),
    "nc": ("C", 1e-9),
    "n": ("N", 1.0),
    "newton": ("N", 1.0),
    "newtons": ("N", 1.0),
    "mn": ("N", 1e-3),
    "m": ("m", 1.0),
    "meter": ("m", 1.0),
    "meters": ("m", 1.0),
    "cm": ("m", 1e-2),
    "mm": ("m", 1e-3),
    "km": ("m", 1e3),
    "s": ("s", 1.0),
    "second": ("s", 1.0),
    "seconds": ("s", 1.0),
}


NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
FRACTION_PATTERN = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*/\s*([-+]?\d*\.?\d+)\s*$")
LATEX_SQRT_PATTERN = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
SAFE_EXPRESSION_PATTERN = re.compile(r"^[0-9eE+\-*/(). sqrtpi]+$")
UNICODE_SUPERSCRIPT_TRANS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
SUPERSCRIPT_EXPONENT_PATTERN = re.compile(r"10([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)")


def normalize_number(value: str) -> float | None:
    text = _clean_number_text(str(value))
    if not text:
        return None

    fraction_match = FRACTION_PATTERN.match(text)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        if denominator == 0:
            return None
        return numerator / denominator

    expression_value = _try_evaluate_numeric_expression(text)
    if expression_value is not None:
        return expression_value

    number_match = NUMBER_PATTERN.search(text)
    if not number_match:
        return None

    try:
        return float(number_match.group(0))
    except ValueError:
        return None


def normalize_unit(unit: str) -> str:
    canonical, _ = _unit_info(unit)
    return canonical


def compare_answer(
    pred_answer: str,
    pred_unit: str,
    gold_answer: str,
    gold_unit: str,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
) -> dict[str, Any]:
    pred_answer_text = str(pred_answer).strip()
    gold_answer_text = str(gold_answer).strip()
    normalized_pred_unit, pred_unit_scale = _unit_info(pred_unit)
    normalized_gold_unit, gold_unit_scale = _unit_info(gold_unit)

    pred_answer_num = normalize_number(pred_answer_text)
    gold_answer_num = normalize_number(gold_answer_text)

    converted_pred_answer_num = pred_answer_num
    converted_gold_answer_num = gold_answer_num
    if pred_answer_num is not None and gold_answer_num is not None:
        if normalized_pred_unit == normalized_gold_unit:
            converted_pred_answer_num = pred_answer_num * pred_unit_scale
            converted_gold_answer_num = gold_answer_num * gold_unit_scale

        numeric_match = math.isclose(
            converted_pred_answer_num,
            converted_gold_answer_num,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
    else:
        numeric_match = pred_answer_text == gold_answer_text

    exact_match = pred_answer_text == gold_answer_text and normalized_pred_unit == normalized_gold_unit
    unit_match = normalized_pred_unit == normalized_gold_unit

    return {
        "exact_match": exact_match,
        "numeric_match": numeric_match,
        "unit_match": unit_match,
        "correct": numeric_match and unit_match,
        "pred_answer_num": pred_answer_num,
        "gold_answer_num": gold_answer_num,
        "converted_pred_answer_num": converted_pred_answer_num,
        "converted_gold_answer_num": converted_gold_answer_num,
        "normalized_pred_unit": normalized_pred_unit,
        "normalized_gold_unit": normalized_gold_unit,
    }


def _clean_number_text(value: str) -> str:
    text = value.strip().replace(",", "")
    text = SUPERSCRIPT_EXPONENT_PATTERN.sub(lambda match: "10**" + match.group(1).translate(UNICODE_SUPERSCRIPT_TRANS), text)
    text = text.translate(UNICODE_SUPERSCRIPT_TRANS)
    text = text.replace("−", "-").replace("–", "-")
    text = text.replace("×", "*").replace("·", "*").replace("⋅", "*")
    text = text.replace("\\times", "*")
    text = LATEX_SQRT_PATTERN.sub(r"sqrt(\1)", text)
    text = re.sub(r"(\d)\s*sqrt\(", r"\1*sqrt(", text)
    text = re.sub(r"10\s*\^\s*([-+]?\d+)", r"10**\1", text)
    return text.strip()


def _try_evaluate_numeric_expression(text: str) -> float | None:
    if not any(token in text for token in ("*", "/", "sqrt", "pi", "**")):
        return None
    if not SAFE_EXPRESSION_PATTERN.match(text):
        return None

    try:
        import sympy as sp

        value = sp.N(sp.sympify(text, locals={"sqrt": sp.sqrt, "pi": sp.pi}))
        if value.is_real:
            return float(value)
    except Exception:
        return None

    return None


def _unit_info(unit: str) -> tuple[str, float]:
    text = str(unit).strip()
    if not text:
        return "", 1.0

    compact = text.replace(" ", "").replace(".", "").strip()
    candidates = [compact, compact.casefold(), text, text.casefold()]
    for candidate in candidates:
        if candidate in UNIT_ALIASES:
            return UNIT_ALIASES[candidate]

    normalized = (
        compact.casefold()
        .replace("î©", "ohm")
        .replace("ï‰", "ohm")
        .replace("ω", "ohm")
        .replace("Ω", "ohm")
    )
    if normalized in UNIT_ALIASES:
        return UNIT_ALIASES[normalized]

    if normalized.endswith("s") and normalized[:-1] in UNIT_ALIASES:
        return UNIT_ALIASES[normalized[:-1]]

    return text, 1.0
