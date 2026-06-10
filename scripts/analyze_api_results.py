"""Analyze full API pipeline JSONL results."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parser.unit_normalizer import get_unit_info  # noqa: E402

try:
    from evaluate_answers import classify_expected, numeric_answer_match, text_answer_match  # noqa: E402
except Exception:  # noqa: BLE001
    classify_expected = None  # type: ignore[assignment]
    numeric_answer_match = None  # type: ignore[assignment]
    text_answer_match = None  # type: ignore[assignment]


_NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


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


def _first_number(value: object) -> Optional[float]:
    match = _NUMERIC_RE.search(str(value or ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def _normalize_unit(value: object) -> str:
    unit = str(value or "").strip().lower()
    unit = unit.replace("μ", "u").replace("µ", "u")
    unit = unit.replace("ohms", "ohm").replace("ohms", "ohm")
    unit = re.sub(r"\s+", "", unit)
    return unit


def _answer_number_match(
    predicted: object,
    expected: object,
    expected_unit: object,
    rel_tol: float,
    abs_tol: float,
) -> Optional[bool]:
    pred_num = _first_number(predicted)
    exp_num = _first_number(expected)
    if pred_num is None or exp_num is None:
        return None

    if math.isclose(pred_num, exp_num, rel_tol=rel_tol, abs_tol=abs_tol):
        return True

    pred_text = str(predicted or "")
    pred_unit_match = re.search(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?\s*([A-Za-zμµΩÂ°/\^\*\-²³]+)", pred_text)
    pred_unit = pred_unit_match.group(1) if pred_unit_match else ""
    pred_info = get_unit_info(pred_unit) if pred_unit else None
    exp_info = get_unit_info(str(expected_unit or "")) if expected_unit else None
    if pred_info and exp_info and pred_info.get("dimension") == exp_info.get("dimension"):
        pred_si = pred_num * float(pred_info.get("to_si", 1.0))
        exp_si = exp_num * float(exp_info.get("to_si", 1.0))
        if math.isclose(pred_si, exp_si, rel_tol=rel_tol, abs_tol=max(abs_tol, abs(exp_si) * rel_tol)):
            return True

    # Common metric-prefix mismatch: expected dataset may store micro value as
    # 5.86 with unit uF while model returns 0.00000586 F.
    for scale in (1e-12, 1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9, 1e12):
        if math.isclose(pred_num / scale, exp_num, rel_tol=rel_tol, abs_tol=abs_tol):
            return True
    return False


def _first_bad_step(steps: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for step in steps:
        if str(step.get("status", "")).upper() not in {"OK", "PASS", "REPAIRED"}:
            return step
    return None


def _formula_ids(steps: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for step in steps:
        for formula_id in step.get("formula_ids") or []:
            ids.append(str(formula_id))
    return ids


def _final_answer_check(result: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    raw = (
        record.get("final_answer_check")
        or result.get("final_answer_check")
        or result.get("answer_level_verification")
        or {}
    )
    return {
        "verdict": raw.get("verdict") or raw.get("final_answer_verdict") or result.get("final_answer_verdict") or "UNKNOWN",
        "error_type": raw.get("error_type") or raw.get("final_answer_error_type") or result.get("final_answer_error_type"),
        "repair_attempted": bool(raw.get("repair_attempted") or raw.get("numeric_repair_attempted") or result.get("numeric_repair_attempted")),
        "repair_accepted": bool(raw.get("repair_accepted") or raw.get("numeric_repair_accepted") or result.get("numeric_repair_accepted")),
        "notes": raw.get("notes") or raw.get("repair_hint") or result.get("repair_hint"),
    }


def _gold_correct(
    payload: Dict[str, Any],
    predicted_answer: object,
    rel_tol: float,
    abs_tol: float,
) -> tuple[Optional[bool], str, str]:
    if classify_expected is None:
        answer_match = _answer_number_match(
            predicted_answer,
            payload.get("answer"),
            payload.get("unit"),
            rel_tol,
            abs_tol,
        )
        if answer_match is None:
            return None, "unknown", "not_numeric_or_missing_expected"
        return answer_match, "numeric_calc", "numeric_match" if answer_match else "numeric_mismatch"

    category = classify_expected(payload.get("question"), payload.get("answer"), payload.get("unit"))
    if category == "numeric_calc" and numeric_answer_match is not None:
        is_match, method, _pred_value, _expected_value = numeric_answer_match(
            predicted_answer,
            payload.get("answer"),
            payload.get("unit"),
            payload.get("question"),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        return is_match, category, method
    if text_answer_match is not None:
        is_match, method = text_answer_match(predicted_answer, payload.get("answer"))
        return is_match, category, method
    return None, category, "unknown"


def _predicted_unit(predicted: object) -> str:
    text = str(predicted or "")
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?\s*([A-Za-zÎ¼ÂµμΩ/\^\*\-²³]+)", text)
    return match.group(1) if match else ""


def _classify_false_pass(
    record: Dict[str, Any],
    result: Dict[str, Any],
    category: str,
    match_method: str,
    final_answer_check: Dict[str, Any],
    rel_tol: float,
    abs_tol: float,
) -> tuple[str, str]:
    payload = record.get("payload") or {}
    question = str(payload.get("question") or "").lower()
    predicted = result.get("answer")
    expected = payload.get("answer")
    expected_unit = payload.get("unit")
    explicit_error = final_answer_check.get("error_type")
    known_buckets = {
        "unit_scale_error",
        "rounding_error",
        "wrong_target_variable",
        "q_disambiguation",
        "rlc_condition_violation",
        "formula_condition_violation",
        "numeric_mismatch",
        "unknown_false_pass",
    }
    if explicit_error in known_buckets:
        return str(explicit_error), "final_answer_check reported this error type"

    pred_num = _first_number(predicted)
    exp_num = _first_number(expected)
    if category == "numeric_calc" and pred_num is None:
        return "answer_format_wrong", "final answer has no parseable number"

    pred_unit = _predicted_unit(predicted)
    exp_info = get_unit_info(str(expected_unit or "")) if expected_unit else None
    pred_info = get_unit_info(pred_unit) if pred_unit else None
    if pred_info and exp_info and pred_info.get("dimension") != exp_info.get("dimension"):
        return "wrong_target_variable", f"predicted unit {pred_unit} is incompatible with expected unit {expected_unit}"
    if pred_info and exp_info and pred_info.get("dimension") == exp_info.get("dimension") and pred_num is not None and exp_num is not None:
        pred_si = pred_num * float(pred_info.get("to_si", 1.0))
        exp_si = exp_num * float(exp_info.get("to_si", 1.0))
        if math.isclose(pred_si, exp_si, rel_tol=rel_tol, abs_tol=max(abs_tol, abs(exp_si) * rel_tol)):
            return "unit_scale_error", "numeric value differs only by unit scaling"

    if pred_num is not None and exp_num is not None:
        if math.isclose(pred_num, exp_num, rel_tol=5e-2, abs_tol=max(abs_tol, abs(exp_num) * 5e-2)):
            return "rounding_error", "numeric values are close under a relaxed 5% tolerance"

    if ("quality factor" in question or "q factor" in question) and (pred_unit.lower() in {"c", "uc", "μc"}):
        return "q_disambiguation", "question asks for quality factor but answer looks like charge"

    trace_text = " ".join(
        str(part or "")
        for step in (result.get("steps") or [])
        for part in (step.get("goal"), step.get("intermediate_answer"), step.get("formula_ids"))
    ).lower()
    if any(marker in question for marker in ("not in resonance", "not at resonance", "not resonant", "off resonance")):
        if "z=r" in trace_text.replace(" ", "") or "resonance" in trace_text:
            return "rlc_condition_violation", "non-resonance problem appears to use a resonance shortcut"

    if "condition" in question and "violat" in trace_text:
        return "formula_condition_violation", "trace indicates formula condition issue"

    if match_method in {"numeric_mismatch", "missing_numeric"}:
        return "numeric_mismatch", "numeric answer does not match gold under evaluator tolerance"
    return "unknown_false_pass", "no deterministic false-pass bucket matched"


def _suggest_false_pass_fix(bucket: str) -> str:
    suggestions = {
        "unit_scale_error": "Check final display unit only; preserve SI value and add a guarded unit-format repair if deterministic.",
        "rounding_error": "Adjust final formatting precision or evaluator tolerance; do not change physics calculation.",
        "wrong_target_variable": "Ensure final answer is selected from parse_obj.unknown_quantity / VSO target variable.",
        "q_disambiguation": "Route quality-factor Q to Q_factor formulas, not electric charge.",
        "rlc_condition_violation": "Apply off-resonance impedance/current/power formulas when non-resonance markers are present.",
        "formula_condition_violation": "Add a formula-condition guard before accepting the trace.",
        "numeric_mismatch": "Inspect as a solver or gold-data issue; do not auto-repair without a deterministic pattern.",
        "answer_format_wrong": "Normalize final answer formatting so the evaluator can parse the intended value.",
        "unknown_false_pass": "Manual review required.",
    }
    return suggestions.get(bucket, "Manual review required.")


def analyze(
    results_path: Path,
    output_dir: Path,
    rel_tol: float,
    abs_tol: float,
    sample_limit: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    records = list(_read_jsonl(results_path))
    total = len(records)

    status_counts: Counter[str] = Counter()
    trace_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    formula_path_counts: Counter[str] = Counter()
    first_bad_step_counts: Counter[str] = Counter()
    first_bad_goal_counts: Counter[str] = Counter()
    formula_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    answer_match_counts: Counter[str] = Counter()
    latency_values: List[float] = []
    confidence_values: List[float] = []
    failures: List[Dict[str, Any]] = []
    soft_step_warnings: List[Dict[str, Any]] = []
    answer_mismatches: List[Dict[str, Any]] = []
    by_failure_reason: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    failure_reason_counts: Counter[str] = Counter()
    soft_step_counts: Counter[str] = Counter()
    fws_error_counts: Counter[str] = Counter()
    final_correct_counts: Counter[str] = Counter()
    false_pass_bucket_counts: Counter[str] = Counter()
    false_pass_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    false_pass_records: List[Dict[str, Any]] = []
    unit_wrong_rows = 0
    answer_format_wrong_rows = 0

    for record in records:
        payload = record.get("payload") or {}
        result = record.get("result") or {}
        steps = result.get("steps") or []
        fws_diagnosis = result.get("fws_diagnosis") or {}
        status = str(record.get("status") or "UNKNOWN")
        trace_status = str(result.get("trace_status") or "UNKNOWN")
        query_type = str(result.get("query_type") or payload.get("query_type") or "unknown")
        error = str(result.get("error") or "")
        expected_answer = payload.get("answer")
        expected_unit = payload.get("unit")
        predicted_answer = result.get("answer")
        first_bad = _first_bad_step(steps)
        final_answer_check = _final_answer_check(result, record)

        status_counts[status] += 1
        trace_counts[trace_status] += 1
        query_counts[query_type] += 1
        if error:
            error_counts[error] += 1
        formula_path_counts[str(result.get("formula_path_index", "none"))] += 1
        for formula_id in _formula_ids(steps):
            formula_counts[formula_id] += 1
        if expected_unit:
            unit_counts[_normalize_unit(expected_unit)] += 1
        if isinstance(result.get("latency_seconds"), (int, float)):
            latency_values.append(float(result["latency_seconds"]))
        if isinstance(result.get("confidence"), (int, float)):
            confidence_values.append(float(result["confidence"]))
        fws_error_type = fws_diagnosis.get("first_wrong_error_type")
        if fws_error_type:
            fws_error_counts[str(fws_error_type)] += 1

        answer_match = _answer_number_match(predicted_answer, expected_answer, expected_unit, rel_tol, abs_tol)
        if answer_match is True:
            answer_match_counts["numeric_match"] += 1
        elif answer_match is False:
            answer_match_counts["numeric_mismatch"] += 1
        else:
            answer_match_counts["not_numeric_or_missing_expected"] += 1

        final_correct, category, match_method = _gold_correct(
            payload,
            predicted_answer,
            rel_tol,
            abs_tol,
        )
        if final_correct is True:
            final_correct_counts["correct"] += 1
        elif final_correct is False:
            final_correct_counts["wrong"] += 1
        else:
            final_correct_counts["unknown"] += 1
        if category == "numeric_calc" and _first_number(predicted_answer) is None:
            answer_format_wrong_rows += 1

        if answer_match is False:
            mismatch = {
                "dataset_id": record.get("dataset_id"),
                "row_index": record.get("row_index"),
                "question": payload.get("question"),
                "expected_answer": expected_answer,
                "expected_unit": expected_unit,
                "predicted_answer": predicted_answer,
                "trace_status": trace_status,
                "formula_path_index": result.get("formula_path_index"),
                "formula_ids": _formula_ids(steps),
                "confidence": result.get("confidence"),
                "fws_diagnosis": fws_diagnosis,
            }
            answer_mismatches.append(mismatch)

        trace_pass = trace_status.upper() in {"PASS", "REPAIRED"} and status == "PASS"
        if trace_pass and final_correct is False:
            bucket, cause = _classify_false_pass(
                record,
                result,
                category,
                match_method,
                final_answer_check,
                rel_tol,
                abs_tol,
            )
            false_pass_bucket_counts[bucket] += 1
            if bucket in {"unit_scale_error", "wrong_target_variable"}:
                unit_wrong_rows += 1
            if bucket == "answer_format_wrong":
                answer_format_wrong_rows += 1
            false_pass = {
                "dataset_id": record.get("dataset_id"),
                "row_index": record.get("row_index"),
                "bucket": bucket,
                "suspected_cause": cause,
                "question": payload.get("question"),
                "expected_answer": expected_answer,
                "expected_unit": expected_unit,
                "predicted_answer": predicted_answer,
                "trace_status": trace_status,
                "pipeline_status": status,
                "final_answer_verdict": final_answer_check.get("verdict"),
                "final_answer_check": final_answer_check,
                "match_method": match_method,
                "suggested_fix": _suggest_false_pass_fix(bucket),
            }
            false_pass_records.append(false_pass)
            if len(false_pass_examples[bucket]) < sample_limit:
                false_pass_examples[bucket].append(false_pass)

        if first_bad:
            key = str(first_bad.get("step_id") or "unknown")
            goal = str(first_bad.get("goal") or "unknown")
            first_bad_step_counts[key] += 1
            first_bad_goal_counts[goal] += 1

        has_bad_step = first_bad is not None
        hard_failure = status != "PASS" or trace_status == "FAIL" or bool(error)
        if hard_failure:
            if error:
                reason = f"error: {error}"
            elif first_bad:
                reason = f"bad_step:{first_bad.get('status')}:{first_bad.get('goal')}"
            else:
                reason = f"trace_status:{trace_status}"
            failure = {
                "dataset_id": record.get("dataset_id"),
                "row_index": record.get("row_index"),
                "reason": reason,
                "question": payload.get("question"),
                "expected_answer": expected_answer,
                "expected_unit": expected_unit,
                "predicted_answer": predicted_answer,
                "trace_status": trace_status,
                "formula_path_index": result.get("formula_path_index"),
                "formula_ids": _formula_ids(steps),
                "first_bad_step": first_bad,
                "fws_diagnosis": fws_diagnosis,
                "confidence": result.get("confidence"),
                "latency_seconds": result.get("latency_seconds"),
                "error": error,
            }
            failures.append(failure)
            failure_reason_counts[reason] += 1
            if len(by_failure_reason[reason]) < sample_limit:
                by_failure_reason[reason].append(failure)
        elif has_bad_step:
            reason = f"soft_bad_step:{first_bad.get('status')}:{first_bad.get('goal')}"
            soft_step_counts[reason] += 1
            soft_step_warnings.append(
                {
                    "dataset_id": record.get("dataset_id"),
                    "row_index": record.get("row_index"),
                    "reason": reason,
                    "question": payload.get("question"),
                    "expected_answer": expected_answer,
                    "expected_unit": expected_unit,
                    "predicted_answer": predicted_answer,
                    "trace_status": trace_status,
                    "formula_path_index": result.get("formula_path_index"),
                    "formula_ids": _formula_ids(steps),
                    "first_bad_step": first_bad,
                    "fws_diagnosis": fws_diagnosis,
                    "confidence": result.get("confidence"),
                    "latency_seconds": result.get("latency_seconds"),
                }
            )

    def _avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    false_pass_summary = {
        "results_path": str(results_path),
        "total_rows": total,
        "trace_pass_rows": trace_counts.get("PASS", 0) + trace_counts.get("REPAIRED", 0),
        "trace_fail_rows": trace_counts.get("FAIL", 0),
        "final_answer_correct_rows": final_correct_counts.get("correct", 0),
        "final_answer_wrong_rows": final_correct_counts.get("wrong", 0),
        "final_answer_unknown_rows": final_correct_counts.get("unknown", 0),
        "trace_pass_but_final_wrong_rows": len(false_pass_records),
        "numeric_wrong_rows": sum(1 for r in false_pass_records if r.get("match_method") in {"numeric_mismatch", "missing_numeric"}),
        "unit_wrong_rows": unit_wrong_rows,
        "answer_format_wrong_rows": answer_format_wrong_rows,
        "false_pass_error_buckets": dict(false_pass_bucket_counts.most_common()),
        "examples": {bucket: examples for bucket, examples in false_pass_examples.items()},
        "outputs": {
            "summary": str(output_dir / "false_pass_summary.json"),
            "report": str(output_dir / "false_pass_report.md"),
            "records": str(output_dir / "false_pass_records.jsonl"),
        },
    }

    summary = {
        "results_path": str(results_path),
        "total": total,
        "status_counts": dict(status_counts.most_common()),
        "trace_status_counts": dict(trace_counts.most_common()),
        "query_type_counts": dict(query_counts.most_common()),
        "answer_match_counts": dict(answer_match_counts.most_common()),
        "pass_rate": round(status_counts.get("PASS", 0) / total, 4) if total else 0.0,
        "numeric_match_rate": round(answer_match_counts.get("numeric_match", 0) / total, 4) if total else 0.0,
        "average_latency_seconds": _avg(latency_values),
        "average_confidence": _avg(confidence_values),
        "top_errors": dict(error_counts.most_common(20)),
        "formula_path_counts": dict(formula_path_counts.most_common()),
        "top_formula_ids": dict(formula_counts.most_common(25)),
        "top_units": dict(unit_counts.most_common(25)),
        "first_bad_step_counts": dict(first_bad_step_counts.most_common(20)),
        "first_bad_goal_counts": dict(first_bad_goal_counts.most_common(20)),
        "fws_error_type_counts": dict(fws_error_counts.most_common(20)),
        "final_answer_counts": dict(final_correct_counts.most_common()),
        "false_pass_summary": {
            "trace_pass_but_final_wrong_rows": len(false_pass_records),
            "false_pass_error_buckets": dict(false_pass_bucket_counts.most_common()),
            "unit_wrong_rows": unit_wrong_rows,
            "answer_format_wrong_rows": answer_format_wrong_rows,
        },
        "soft_step_warning_counts": dict(soft_step_counts.most_common(20)),
        "failure_reason_counts": dict(failure_reason_counts.most_common()),
        "outputs": {
            "summary": str(output_dir / "api_error_summary.json"),
            "failures": str(output_dir / "api_error_failures.jsonl"),
            "soft_step_warnings": str(output_dir / "api_soft_step_warnings.jsonl"),
            "answer_mismatches": str(output_dir / "api_answer_mismatches.jsonl"),
            "report": str(output_dir / "api_error_report.md"),
            "false_pass_summary": str(output_dir / "false_pass_summary.json"),
            "false_pass_report": str(output_dir / "false_pass_report.md"),
        },
    }

    _write_jsonl(output_dir / "api_error_failures.jsonl", failures)
    _write_jsonl(output_dir / "api_soft_step_warnings.jsonl", soft_step_warnings)
    _write_jsonl(output_dir / "api_answer_mismatches.jsonl", answer_mismatches)
    _write_jsonl(output_dir / "false_pass_records.jsonl", false_pass_records)
    (output_dir / "api_error_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "false_pass_summary.json").write_text(
        json.dumps(false_pass_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(output_dir / "api_error_report.md", summary, by_failure_reason, answer_mismatches, sample_limit)
    _write_false_pass_report(output_dir / "false_pass_report.md", false_pass_summary, sample_limit)
    return summary


def _write_report(
    path: Path,
    summary: Dict[str, Any],
    by_failure_reason: dict[str, list[Dict[str, Any]]],
    answer_mismatches: List[Dict[str, Any]],
    sample_limit: int,
) -> None:
    lines = [
        "# API Error Analysis",
        "",
        f"- Results: `{summary['results_path']}`",
        f"- Total: {summary['total']}",
        f"- PASS rate: {summary['pass_rate']}",
        f"- Numeric match rate: {summary['numeric_match_rate']}",
        f"- Average latency seconds: {summary['average_latency_seconds']}",
        "",
        "## Status Counts",
        "",
        "```json",
        json.dumps(summary["status_counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Trace Status Counts",
        "",
        "```json",
        json.dumps(summary["trace_status_counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Top Failure Reasons",
        "",
    ]
    for reason, samples in by_failure_reason.items():
        lines.append(f"### {reason}")
        lines.append("")
        for sample in samples[:sample_limit]:
            lines.append(
                f"- `{sample.get('dataset_id')}` row {sample.get('row_index')}: "
                f"expected `{sample.get('expected_answer')} {sample.get('expected_unit')}`, "
                f"got `{sample.get('predicted_answer')}`"
            )
            question = str(sample.get("question") or "")
            if question:
                lines.append(f"  Question: {question[:240]}")
        lines.append("")

    lines.extend(["## Answer Mismatch Samples", ""])
    for sample in answer_mismatches[:sample_limit]:
        lines.append(
            f"- `{sample.get('dataset_id')}` row {sample.get('row_index')}: "
            f"expected `{sample.get('expected_answer')} {sample.get('expected_unit')}`, "
            f"got `{sample.get('predicted_answer')}`, formulas `{sample.get('formula_ids')}`"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_false_pass_report(
    path: Path,
    false_pass_summary: Dict[str, Any],
    sample_limit: int,
) -> None:
    lines = [
        "# False-Pass Analysis",
        "",
        f"- Results: `{false_pass_summary['results_path']}`",
        f"- Total rows: {false_pass_summary['total_rows']}",
        f"- Trace PASS/REPAIRED rows: {false_pass_summary['trace_pass_rows']}",
        f"- Trace FAIL rows: {false_pass_summary['trace_fail_rows']}",
        f"- Final answer correct rows: {false_pass_summary['final_answer_correct_rows']}",
        f"- Final answer wrong rows: {false_pass_summary['final_answer_wrong_rows']}",
        f"- Trace PASS but final answer wrong rows: {false_pass_summary['trace_pass_but_final_wrong_rows']}",
        f"- Numeric wrong rows: {false_pass_summary['numeric_wrong_rows']}",
        f"- Unit wrong rows: {false_pass_summary['unit_wrong_rows']}",
        f"- Answer format wrong rows: {false_pass_summary['answer_format_wrong_rows']}",
        "",
        "## Top False-Pass Buckets",
        "",
        "```json",
        json.dumps(false_pass_summary["false_pass_error_buckets"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Examples",
        "",
    ]
    examples = false_pass_summary.get("examples") or {}
    for bucket, bucket_examples in examples.items():
        lines.append(f"### {bucket}")
        lines.append("")
        lines.append(f"Suggested fix: {_suggest_false_pass_fix(bucket)}")
        lines.append("")
        for sample in bucket_examples[:sample_limit]:
            lines.append(
                f"- `{sample.get('dataset_id')}` row {sample.get('row_index')}: "
                f"predicted `{sample.get('predicted_answer')}`, "
                f"gold `{sample.get('expected_answer')} {sample.get('expected_unit') or ''}`, "
                f"trace `{sample.get('trace_status')}`, "
                f"verdict `{sample.get('final_answer_verdict')}`"
            )
            lines.append(f"  Cause: {sample.get('suspected_cause')}")
            question = str(sample.get("question") or "")
            if question:
                lines.append(f"  Question: {question[:240]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze full API pipeline JSONL results.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rel-tol", type=float, default=1e-2)
    parser.add_argument("--abs-tol", type=float, default=1e-9)
    parser.add_argument("--sample-limit", type=int, default=10)
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "error_analysis"
    summary = analyze(
        results_path=results_path,
        output_dir=output_dir,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        sample_limit=args.sample_limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
