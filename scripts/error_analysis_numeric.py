"""Deep error analysis for Type2 numeric failures.

Reads api_results.jsonl + answer_eval_wrong.jsonl and classifies each numeric
failure into root-cause categories to guide targeted fixes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_answers import extract_numeric_value, numeric_answer_match  # noqa: E402


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _get_nested(d: Dict, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_VECTOR_KEYWORDS = re.compile(
    r"(resultant|net force|vector|perpendicular|bisector|triangle|square|"
    r"equilateral|isosceles|right.?angle|vertex|vertices|diagonal|center|centroid|"
    r"midpoint|foot of the altitude)",
    re.IGNORECASE,
)

_RLC_KEYWORDS = re.compile(
    r"(RLC|quality factor|resonan|inductor|capacitor|coil|impedance|reactance)",
    re.IGNORECASE,
)

_UNIT_MISMATCH_SCALES = [1e-12, 1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9, 1e12]


def classify_error(
    record: Dict[str, Any],
    result_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify a single wrong-answer record into an error category."""
    question = str(record.get("question") or "")
    expected_answer = record.get("expected_answer")
    predicted_answer = record.get("predicted_answer")
    expected_unit = record.get("expected_unit")
    trace_status = str(record.get("trace_status") or "")
    hybrid_source = str(record.get("hybrid_source") or "")
    dataset_id = str(record.get("dataset_id") or "")
    pred_value = record.get("predicted_value")
    exp_value = record.get("expected_value")
    confidence = record.get("confidence")

    # Extract trace details from result_record if available
    parse_obj = {}
    steps = []
    formula_ids = []
    diagnosis = {}
    if result_record:
        result = result_record.get("result") or {}
        parse_obj = result_record.get("payload") or {}
        steps = result.get("steps") or []
        formula_ids = []
        for s in steps:
            formula_ids.extend(s.get("formula_ids") or [])
        diagnosis = result.get("diagnosis") or {}

    category = "unclassified"
    subcategory = ""
    detail = ""

    # F. answer_format_error: value is close after unit scaling
    if pred_value is not None and exp_value is not None and exp_value != 0:
        for scale in _UNIT_MISMATCH_SCALES:
            if math.isclose(pred_value / scale, exp_value, rel_tol=0.02):
                category = "answer_format_error"
                subcategory = f"unit_scale_{scale:g}"
                detail = f"pred/scale={pred_value/scale:g} vs expected={exp_value:g}"
                break
        if category == "answer_format_error":
            pass
        elif pred_value != 0 and math.isclose(pred_value, exp_value, rel_tol=0.05):
            category = "answer_format_error"
            subcategory = "rounding"
            detail = f"pred={pred_value:g} vs expected={exp_value:g} (within 5%)"

    # D. verifier_false_pass: trace says PASS but answer is wrong
    if category == "unclassified" and trace_status == "PASS" and pred_value is not None and exp_value is not None:
        category = "verifier_false_pass"
        subcategory = "pass_but_wrong"

    # E. llm_fallback_error
    if category == "verifier_false_pass" and hybrid_source == "llm_fallback":
        category = "llm_fallback_error"
        if pred_value is not None and exp_value is not None:
            if pred_value == 0:
                subcategory = "fallback_returned_zero"
            elif str(predicted_answer) and not any(c.isdigit() for c in str(predicted_answer)[:20]):
                subcategory = "fallback_non_numeric"
            else:
                subcategory = "fallback_wrong_value"
        else:
            subcategory = "fallback_no_numeric"

    # Refine verifier_false_pass for deterministic source
    if category == "verifier_false_pass" and hybrid_source == "deterministic":
        # Check for vector geometry issues
        if _VECTOR_KEYWORDS.search(question):
            category = "deterministic_solver_error"
            subcategory = "vector_geometry"
            detail = "vector composition likely wrong or missing"
        elif _RLC_KEYWORDS.search(question) and pred_value is not None and exp_value is not None:
            if "Q" in str(predicted_answer) or dataset_id.startswith("CH"):
                category = "formula_retrieval_error"
                subcategory = "Q_disambiguation"
                detail = "Q may be quality factor, not charge"
            else:
                category = "deterministic_solver_error"
                subcategory = "rlc_calculation"
        elif pred_value is not None and exp_value is not None and exp_value != 0:
            ratio = pred_value / exp_value if exp_value != 0 else float('inf')
            if 1.5 < abs(ratio) < 3.0 or 0.3 < abs(ratio) < 0.7:
                category = "deterministic_solver_error"
                subcategory = "factor_error"
                detail = f"ratio={ratio:.3f} (possible missing factor)"
            elif abs(ratio) > 100 or abs(ratio) < 0.01:
                category = "parser_error"
                subcategory = "wrong_target_or_unit"
                detail = f"ratio={ratio:.3g} (target/unit mismatch likely)"
            else:
                category = "deterministic_solver_error"
                subcategory = "calculation_error"

    # A. parser_error: trace FAIL often means parser didn't extract enough
    if category == "unclassified" and trace_status == "FAIL":
        if hybrid_source == "llm_fallback":
            category = "llm_fallback_error"
            subcategory = "fallback_also_failed"
        else:
            category = "parser_error"
            subcategory = "incomplete_parse"

    # Further subcategorize vector problems
    if category == "deterministic_solver_error" and subcategory == "vector_geometry":
        q_lower = question.lower()
        if "right isosceles" in q_lower or "isosceles right" in q_lower:
            subcategory = "right_isosceles_vector"
        elif "equilateral" in q_lower:
            subcategory = "equilateral_vector"
        elif "square" in q_lower:
            subcategory = "square_geometry"
        elif "perpendicular bisector" in q_lower:
            subcategory = "perpendicular_bisector"
        elif "midpoint" in q_lower:
            subcategory = "midpoint_field"
        elif "collinear" in q_lower or "straight line" in q_lower:
            subcategory = "collinear_charges"

    return {
        "dataset_id": dataset_id,
        "category": category,
        "subcategory": subcategory,
        "detail": detail,
        "question": question[:300],
        "expected_answer": expected_answer,
        "expected_unit": expected_unit,
        "predicted_answer": predicted_answer,
        "predicted_value": pred_value,
        "expected_value": exp_value,
        "trace_status": trace_status,
        "hybrid_source": hybrid_source,
        "confidence": confidence,
        "formula_ids": formula_ids[:5] if formula_ids else [],
    }


def analyze(
    results_path: Path,
    wrong_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    wrong_records = _read_jsonl(wrong_path)

    # Build lookup from dataset_id to full result record
    result_lookup: Dict[str, Dict[str, Any]] = {}
    if results_path.exists():
        for rec in _read_jsonl(results_path):
            did = rec.get("dataset_id") or _get_nested(rec, "payload", "id")
            if did:
                result_lookup[str(did)] = rec

    # Filter to numeric_calc only
    numeric_wrong = [r for r in wrong_records if r.get("category") == "numeric_calc"]

    classified: List[Dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for rec in numeric_wrong:
        did = str(rec.get("dataset_id") or "")
        result_rec = result_lookup.get(did)
        error = classify_error(rec, result_rec)
        classified.append(error)
        cat = error["category"]
        sub = f"{cat}:{error['subcategory']}" if error["subcategory"] else cat
        category_counts[cat] += 1
        subcategory_counts[sub] += 1
        source_counts[error["hybrid_source"]] += 1

    # Count totals from the full results
    total_results = len(result_lookup)
    total_numeric_wrong = len(numeric_wrong)

    # Compute accuracy from eval summary if available
    eval_summary_path = wrong_path.parent / "answer_eval_summary.json"
    eval_summary = {}
    if eval_summary_path.exists():
        eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))

    numeric_total = _get_nested(eval_summary, "by_category", "numeric_calc", "total", default=0)
    numeric_correct = _get_nested(eval_summary, "by_category", "numeric_calc", "correct", default=0)
    numeric_accuracy = numeric_correct / numeric_total if numeric_total else 0

    # Recommended fixes ordered by count
    fix_recommendations = []
    for sub, count in subcategory_counts.most_common():
        if count >= 3:
            fix_recommendations.append({
                "pattern": sub,
                "count": count,
                "expected_impact": f"+{count} potential fixes ({count/max(numeric_total,1)*100:.1f}%)",
            })

    summary = {
        "total_evaluated": total_results,
        "numeric_total": numeric_total,
        "numeric_correct": numeric_correct,
        "numeric_accuracy": round(numeric_accuracy, 4),
        "numeric_wrong": total_numeric_wrong,
        "category_counts": dict(category_counts.most_common()),
        "subcategory_counts": dict(subcategory_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "fix_recommendations": fix_recommendations[:15],
    }

    # Write outputs
    (output_dir / "api_error_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_jsonl(output_dir / "error_cases.jsonl", classified)
    _write_report(output_dir / "api_error_report.md", summary, classified, subcategory_counts)

    return summary


def _write_report(
    path: Path,
    summary: Dict[str, Any],
    classified: List[Dict[str, Any]],
    subcategory_counts: Counter,
) -> None:
    lines = [
        "# Type2 Numeric Error Analysis Report",
        "",
        f"- Total evaluated: {summary['total_evaluated']}",
        f"- Numeric total: {summary['numeric_total']}",
        f"- Numeric correct: {summary['numeric_correct']}",
        f"- **Numeric accuracy: {summary['numeric_accuracy']:.2%}**",
        f"- Numeric failures: {summary['numeric_wrong']}",
        "",
        "## Error Category Distribution",
        "",
        "| Category | Count | % of Failures |",
        "|----------|-------|---------------|",
    ]
    total_wrong = summary["numeric_wrong"] or 1
    for cat, count in sorted(summary["category_counts"].items(), key=lambda x: -x[1]):
        pct = count / total_wrong * 100
        lines.append(f"| {cat} | {count} | {pct:.1f}% |")
    lines.append("")

    lines.extend([
        "## Subcategory Breakdown",
        "",
        "| Subcategory | Count |",
        "|-------------|-------|",
    ])
    for sub, count in subcategory_counts.most_common(30):
        lines.append(f"| {sub} | {count} |")
    lines.append("")

    lines.extend([
        "## Source Distribution (among failures)",
        "",
        "| Source | Count |",
        "|--------|-------|",
    ])
    for src, count in sorted(summary["source_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {src} | {count} |")
    lines.append("")

    lines.extend([
        "## Recommended Fixes (by expected impact)",
        "",
    ])
    for rec in summary.get("fix_recommendations", []):
        lines.append(f"1. **{rec['pattern']}** — {rec['count']} cases ({rec['expected_impact']})")
    lines.append("")

    # Sample errors per category
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for e in classified:
        key = f"{e['category']}:{e['subcategory']}" if e['subcategory'] else e['category']
        if len(by_cat[key]) < 5:
            by_cat[key].append(e)

    lines.extend(["## Sample Errors by Category", ""])
    for key in sorted(by_cat, key=lambda k: -subcategory_counts.get(k, 0)):
        lines.append(f"### {key}")
        lines.append("")
        for e in by_cat[key]:
            lines.append(
                f"- **{e['dataset_id']}** [{e['hybrid_source']}] "
                f"expected=`{e['expected_answer']} {e.get('expected_unit', '')}` "
                f"got=`{e['predicted_answer']}`"
            )
            if e.get("detail"):
                lines.append(f"  - {e['detail']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        required=True,
        help="Path to api_results.jsonl",
    )
    parser.add_argument(
        "--wrong",
        required=True,
        help="Path to answer_eval_wrong.jsonl (from evaluate_answers.py)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: error_analysis/ next to --wrong)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    wrong_path = Path(args.wrong)
    output_dir = Path(args.output_dir) if args.output_dir else wrong_path.parent.parent / "error_analysis"

    summary = analyze(results_path, wrong_path, output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
