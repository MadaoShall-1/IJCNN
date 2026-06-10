"""Evaluate final pipeline answers against dataset gold answers.

This evaluator is intentionally separate from pipeline trace status.  It asks:
"Did the final answer match the gold answer?", with numeric/unit-aware matching
for calculation problems and normalized text matching for formulas/choices.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parser.unit_normalizer import get_unit_info  # noqa: E402


SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
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
    }
)

NUMERIC_CUE_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?|[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
FORMULA_CUE_RE = re.compile(
    r"[=√ππωΩμ²³₀₁₂₃₄₅₆₇₈₉₊₋^*/]|"
    r"\b(?:sin|cos|tan|sqrt|pi|max|min|maximum|minimum)\b",
    re.IGNORECASE,
)
CHOICE_CUE_RE = re.compile(r"\b(which of the following|choose|select|option)\b", re.IGNORECASE)

CONCEPT_UNITS = {"", "-", "—"}
MATH_NAMES = {"sqrt", "sin", "cos", "tan", "pi", "max", "min"}
ROUND_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_unicode(text: object) -> str:
    s = str(text or "")
    s = re.sub(r"10\s*([⁻⁺+-]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", r"10^\1", s)
    return (
        s.translate(SUPERSCRIPT_TRANSLATION)
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("×", "*")
        .replace("·", "*")
        .replace("π", "pi")
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("Ω", "Î©")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("×", "*")
        .replace("·", "*")
        .replace("π", "pi")
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("Ω", "Ω")
    )


def _normalize_expression_text(text: object) -> str:
    s = _normalize_unicode(text)
    s = s.replace("\ufffd", "*")
    s = s.replace("×", "*")
    s = s.replace("^", "**")
    s = re.sub(r"(?<=[\d)])\s*[?*]\s*(?=10\b)", "*", s)
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=10\b)", "*", s)
    s = s.replace("\\sqrt", "sqrt")
    s = re.sub(r"sqrt\s*\{\s*([0-9.]+)\s*\}", r"sqrt(\1)", s)
    s = re.sub(r"√\s*\{?\s*([0-9.]+)\s*\}?", r"sqrt(\1)", s)
    s = re.sub(r"10\s*\*\*\s*\{\s*([+-]?\d+)\s*\}", r"10**\1", s)
    s = re.sub(r"(\)|\d)\s*\?\s*(10\b)", r"\1*\2", s)
    # Dataset notation such as "45.10^{5}" means 45*10^5.
    s = re.sub(r"(?<=\d)\s*\.\s*(?=10\s*\*\*)", "*", s)
    s = re.sub(r"(?<=\d)\s*(?=sqrt\()", "*", s)
    s = re.sub(r"(?<=\d)\s*(?=pi\b)", "*", s)
    return s


def _safe_eval(expr: str) -> Optional[float]:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None
    allowed = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Name,
        ast.Load,
    }
    for node in ast.walk(tree):
        if type(node) not in allowed:
            return None
        if isinstance(node, ast.Name) and node.id not in {"sqrt", "pi"}:
            return None
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "sqrt":
                return None
    try:
        value = eval(compile(tree, "<answer-expr>", "eval"), {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})  # noqa: S307
    except Exception:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


ATOM = r"(?:10\s*\*\*\s*[-+]?\d+|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|sqrt\([^)]*\)|pi)"
EXPR_RE = re.compile(rf"[-+]?{ATOM}(?:\s*(?:\*\*|[*/+-])\s*{ATOM})*")


def extract_numeric_value(text: object) -> Optional[float]:
    """Extract and evaluate the first concrete numeric expression."""
    s = _normalize_expression_text(text)
    for match in EXPR_RE.finditer(s):
        expr = match.group(0).strip()
        if not re.search(r"\d|pi", expr):
            continue
        value = _safe_eval(expr)
        if value is not None:
            return value
    return None


def _extract_unit_after_number(text: object) -> str:
    s = _normalize_unicode(text)
    match = re.search(
        r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?\s*"
        r"([A-Za-zμuΩ°/%^*²³/-]+)",
        s,
    )
    return match.group(1) if match else ""


def _normalize_unit_text(unit: object) -> str:
    return (
        str(unit or "")
        .strip()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("Ω", "ohm")
        .replace("ohms", "ohm")
        .replace(" ", "")
        .lower()
    )


def _get_unit_info_compat(unit: object) -> Optional[Dict[str, object]]:
    raw = str(unit or "").strip()
    candidates = [
        raw,
        raw.replace("\ufffd", "u").replace("?", "u"),
        raw.replace("μ", "u").replace("µ", "u"),
        raw.replace("Ω", "ohm"),
        raw.replace("μ", "Î¼").replace("µ", "Î¼"),
    ]
    for candidate in candidates:
        info = get_unit_info(candidate)
        if info:
            return info
    return None


def _rounding_abs_tolerance(question: object) -> Optional[float]:
    q = str(question or "").lower()
    match = re.search(r"round(?:ed)?(?:\s+\w+){0,5}\s+to\s+(?:(\d+)|(zero|one|two|three|four|five|six))\s+decimal\s+places?", q)
    if not match:
        return None
    decimals = int(match.group(1)) if match.group(1) is not None else ROUND_WORDS[match.group(2)]
    return 0.5 * (10 ** -decimals)


def numeric_answer_match(
    predicted: object,
    expected: object,
    expected_unit: object,
    question: object = "",
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[Optional[bool], str, Optional[float], Optional[float]]:
    pred_value = extract_numeric_value(predicted)
    expected_value = extract_numeric_value(expected)
    if pred_value is None or expected_value is None:
        return None, "missing_numeric", pred_value, expected_value

    if math.isclose(pred_value, expected_value, rel_tol=rel_tol, abs_tol=abs_tol):
        return True, "numeric_direct", pred_value, expected_value

    rounding_tol = _rounding_abs_tolerance(question)
    if rounding_tol is not None and math.isclose(pred_value, expected_value, rel_tol=0.0, abs_tol=rounding_tol):
        return True, "numeric_rounding_direct", pred_value, expected_value

    pred_unit = _extract_unit_after_number(predicted)
    pred_info = _get_unit_info_compat(pred_unit) if pred_unit else None
    expected_info = _get_unit_info_compat(str(expected_unit or "")) if expected_unit else None
    if pred_info and expected_info and pred_info.get("dimension") == expected_info.get("dimension"):
        pred_si = pred_value * float(pred_info.get("to_si", 1.0))
        expected_si = expected_value * float(expected_info.get("to_si", 1.0))
        if math.isclose(pred_si, expected_si, rel_tol=rel_tol, abs_tol=max(abs_tol, abs(expected_si) * rel_tol)):
            return True, "numeric_unit_converted", pred_si, expected_si
        if rounding_tol is not None:
            expected_unit_scale = float(expected_info.get("to_si", 1.0))
            if math.isclose(pred_si, expected_si, rel_tol=0.0, abs_tol=rounding_tol * expected_unit_scale):
                return True, "numeric_rounding_unit_converted", pred_si, expected_si

    return False, "numeric_mismatch", pred_value, expected_value


def _contains_unresolved_symbolic_variable(answer: object, unit: object) -> bool:
    """Detect formulas such as sqrt(2)*F0 that are not concrete numeric answers."""
    s = _normalize_expression_text(answer).lower()
    if "=" in s:
        s = s.split("=", 1)[1]
    unit_norm = _normalize_unit_text(unit)
    for token in re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", s):
        token_norm = token.lower()
        if token_norm in MATH_NAMES:
            continue
        if token_norm in unit_norm:
            continue
        if get_unit_info(token):
            continue
        return True
    return False


def normalize_text_answer(text: object) -> str:
    s = _normalize_expression_text(text).lower()
    s = s.replace("ω", "omega").replace("w0", "w0")
    s = re.sub(r"\s+", "", s)
    return re.sub(r"[^a-z0-9=+\-*/^().]", "", s)


def text_answer_match(predicted: object, expected: object) -> tuple[bool, str]:
    pred = normalize_text_answer(predicted)
    exp = normalize_text_answer(expected)
    if not pred or not exp:
        return False, "missing_text"
    if pred == exp:
        return True, "text_exact"
    if exp in pred or pred in exp:
        return True, "text_contains"
    return False, "text_mismatch"


def classify_expected(question: object, answer: object, unit: object) -> str:
    q = str(question or "")
    a = str(answer or "").strip()
    unit_text = str(unit or "").strip()
    choice_like = bool(CHOICE_CUE_RE.search(q))
    has_numeric = bool(NUMERIC_CUE_RE.search(a))
    has_formula = bool(FORMULA_CUE_RE.search(a))
    numeric_value = extract_numeric_value(a)
    unresolved_symbol = _contains_unresolved_symbolic_variable(a, unit) if (has_numeric or has_formula) else False
    mostly_numeric = bool(re.fullmatch(r"\s*[-+0-9.,eE×xX*^{}\s\\/sqrtpiπ√()]+\s*", a))
    if numeric_value is not None and not unresolved_symbol:
        return "numeric_calc"
    if numeric_value is not None and unit_text not in CONCEPT_UNITS and not unresolved_symbol:
        return "numeric_calc"
    if has_numeric and mostly_numeric:
        return "numeric_calc"
    if has_formula:
        return "symbolic_formula"
    if choice_like:
        return "choice"
    return "concept_text"


def evaluate(
    results_path: Path,
    output_dir: Path,
    *,
    rel_tol: float,
    abs_tol: float,
) -> Dict[str, Any]:
    records = list(_read_jsonl(results_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    category_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    status_by_category: Dict[str, Counter[str]] = defaultdict(Counter)
    wrong_records: list[Dict[str, Any]] = []
    trace_pass_answer_wrong = 0
    repair_attempted = 0
    repair_accepted = 0
    repair_accepted_correct = 0
    repair_accepted_wrong = 0
    numeric_wrong_repaired = 0
    numeric_wrong_unrepaired = 0
    answer_repair_error_counts: Counter[str] = Counter()

    for record in records:
        payload = record.get("payload") or {}
        result = record.get("result") or {}
        answer_level = result.get("answer_level_verification") or {}
        dataset_id = record.get("dataset_id") or payload.get("id")
        category = classify_expected(payload.get("question"), payload.get("answer"), payload.get("unit"))
        predicted = result.get("answer")
        expected = payload.get("answer")
        expected_unit = payload.get("unit")

        if category == "numeric_calc":
            is_match, method, pred_value, expected_value = numeric_answer_match(
                predicted,
                expected,
                expected_unit,
                payload.get("question"),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        elif category in {"symbolic_formula", "choice"}:
            text_ok, method = text_answer_match(predicted, expected)
            is_match = text_ok
            pred_value = None
            expected_value = None
        else:
            # Concept text is reported for visibility but excluded from the
            # objective calculation accuracy unless --include-concepts is added
            # later.
            text_ok, method = text_answer_match(predicted, expected)
            is_match = text_ok
            pred_value = None
            expected_value = None

        category_counts[category] += 1
        category_counts["all"] += 1
        status_by_category[category][str(record.get("status") or "UNKNOWN")] += 1
        method_counts[f"{category}:{method}"] += 1
        if answer_level.get("numeric_repair_attempted"):
            repair_attempted += 1
        if answer_level.get("numeric_repair_accepted"):
            repair_accepted += 1
        if answer_level.get("final_answer_error_type"):
            answer_repair_error_counts[str(answer_level.get("final_answer_error_type"))] += 1
        if is_match is True:
            correct_counts[category] += 1
            correct_counts["all"] += 1
            if answer_level.get("numeric_repair_accepted"):
                repair_accepted_correct += 1
        else:
            if str(result.get("trace_status") or "").upper() in {"PASS", "REPAIRED"}:
                trace_pass_answer_wrong += 1
            if category == "numeric_calc":
                if answer_level.get("numeric_repair_accepted"):
                    numeric_wrong_repaired += 1
                else:
                    numeric_wrong_unrepaired += 1
            if answer_level.get("numeric_repair_accepted"):
                repair_accepted_wrong += 1
            wrong_records.append(
                {
                    "dataset_id": dataset_id,
                    "category": category,
                    "method": method,
                    "pipeline_status": record.get("status"),
                    "hybrid_source": result.get("hybrid_source"),
                    "question": payload.get("question"),
                    "expected_answer": expected,
                    "expected_unit": expected_unit,
                    "predicted_answer": predicted,
                    "predicted_value": pred_value,
                    "expected_value": expected_value,
                    "trace_status": result.get("trace_status"),
                    "confidence": result.get("confidence"),
                    "answer_level_verification": answer_level,
                }
            )

    objective_categories = {"numeric_calc", "symbolic_formula", "choice"}
    objective_total = sum(category_counts[cat] for cat in objective_categories)
    objective_correct = sum(correct_counts[cat] for cat in objective_categories)

    def _rate(correct: int, total: int) -> float:
        return round(correct / total, 4) if total else 0.0

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in sorted(category_counts):
        if category == "all":
            continue
        total = category_counts[category]
        correct = correct_counts[category]
        by_category[category] = {
            "total": total,
            "correct": correct,
            "wrong": total - correct,
            "accuracy": _rate(correct, total),
            "pipeline_status_counts": dict(status_by_category[category].most_common()),
        }

    summary: Dict[str, Any] = {
        "results_path": str(results_path),
        "total": category_counts["all"],
        "objective_categories": sorted(objective_categories),
        "objective_total": objective_total,
        "objective_correct": objective_correct,
        "objective_wrong": objective_total - objective_correct,
        "objective_accuracy": _rate(objective_correct, objective_total),
        "by_category": by_category,
        "method_counts": dict(method_counts.most_common()),
        "answer_level_repair": {
            "trace_pass_answer_wrong": trace_pass_answer_wrong,
            "numeric_repair_attempted": repair_attempted,
            "numeric_repair_accepted": repair_accepted,
            "numeric_repair_accepted_correct": repair_accepted_correct,
            "numeric_repair_accepted_wrong": repair_accepted_wrong,
            "numeric_wrong_repaired_accepted": numeric_wrong_repaired,
            "numeric_wrong_unrepaired": numeric_wrong_unrepaired,
            "answer_repair_error_type_counts": dict(answer_repair_error_counts.most_common()),
        },
        "outputs": {
            "summary": str(output_dir / "answer_eval_summary.json"),
            "wrong": str(output_dir / "answer_eval_wrong.jsonl"),
            "wrong_objective": str(output_dir / "answer_eval_wrong_objective.jsonl"),
            "answer_level_repair_summary": str(output_dir / "answer_level_repair_summary.json"),
        },
    }

    (output_dir / "answer_eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "answer_level_repair_summary.json").write_text(
        json.dumps(summary["answer_level_repair"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "answer_eval_wrong.jsonl", wrong_records)
    _write_jsonl(
        output_dir / "answer_eval_wrong_objective.jsonl",
        (record for record in wrong_records if record["category"] in objective_categories),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Path to api_results.jsonl")
    parser.add_argument("--output-dir", default=None, help="Directory for evaluator outputs")
    parser.add_argument("--rel-tol", type=float, default=1e-2)
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "answer_eval"
    summary = evaluate(results_path, output_dir, rel_tol=args.rel_tol, abs_tol=args.abs_tol)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
