"""Evaluate Type 2 Stage 1 formula retrieval over saved Stage 0 artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from type2.pipeline import _dict_to_parse_obj
from type2.stage1 import FormulaRetriever


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


def _formula_steps(parse: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        step
        for step in parse.get("step_plan", []) or []
        if isinstance(step, dict) and step.get("type") == "formula_application"
    ]


def _tokens(text: object) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", str(text or "")):
        token = raw.lower().strip("_")
        if not token:
            continue
        tokens.add(token)
        for part in re.split(r"_+", token):
            if part:
                tokens.add(part)
                alpha_prefix = re.match(r"[a-z]+", part)
                if alpha_prefix:
                    tokens.add(alpha_prefix.group(0))
        alpha_prefix = re.match(r"[a-z]+", token)
        if alpha_prefix:
            tokens.add(alpha_prefix.group(0))
    return tokens


def _formula_overlap(step: Dict[str, Any], entry: Any) -> float:
    expected = _tokens(step.get("formula_name"))
    if not expected:
        return 1.0
    candidate = _tokens(f"{entry.formula} {entry.sympy_expr} {entry.text}")
    if not candidate:
        return 0.0
    return len(expected & candidate) / len(expected)


def evaluate(results_path: Path, output_dir: Path, beam_n: int = 3) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    retriever = FormulaRetriever()

    total = 0
    parse_pass = 0
    parse_fail = 0
    total_formula_steps = 0
    matched_formula_steps = 0
    none_formula_steps = 0
    no_formula_step_records: List[Dict[str, Any]] = []
    suspicious_records: List[Dict[str, Any]] = []

    formula_counts: Counter[str] = Counter()
    template_formula_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    confidence_hist: Counter[str] = Counter()
    path_count_hist: Counter[str] = Counter()
    retrieval_confidences: List[float] = []

    formula_by_template: Dict[str, Counter[str]] = defaultdict(Counter)

    for record in _read_jsonl(results_path):
        total += 1
        parse = record.get("parse") or {}
        metadata = parse.get("metadata") or {}
        status = metadata.get("verifier_status", "FAIL")
        if status == "PASS":
            parse_pass += 1
        elif status == "FAIL":
            parse_fail += 1

        formula_steps = _formula_steps(parse)
        total_formula_steps += len(formula_steps)
        for domain in parse.get("domains") or ["unknown"]:
            domain_counts[str(domain)] += 1
        for step in formula_steps:
            template_counts[str(step.get("template_name") or "unknown")] += 1

        try:
            parse_obj = _dict_to_parse_obj(parse)
            formula_sets = retriever.retrieve(parse_obj, beam_n=beam_n)
        except Exception as exc:
            suspicious_records.append(
                {
                    "dataset_id": record.get("dataset_id"),
                    "row_index": record.get("row_index"),
                    "question": record.get("question"),
                    "reason": f"stage1_exception: {exc}",
                    "parse_status": status,
                }
            )
            continue

        path_count_hist[str(len(formula_sets))] += 1
        best = formula_sets[0] if formula_sets else None
        if best is None:
            if formula_steps:
                suspicious_records.append(
                    {
                        "dataset_id": record.get("dataset_id"),
                        "row_index": record.get("row_index"),
                        "question": record.get("question"),
                        "reason": "no_formula_sets",
                        "parse_status": status,
                    }
                )
            continue

        retrieval_confidences.append(float(best.retrieval_confidence))
        conf = float(best.retrieval_confidence)
        if conf >= 0.8:
            confidence_hist[">=0.8"] += 1
        elif conf >= 0.5:
            confidence_hist["0.5-0.8"] += 1
        elif conf > 0:
            confidence_hist["0-0.5"] += 1
        else:
            confidence_hist["0"] += 1

        step_by_id = {step.get("step_id"): step for step in formula_steps}
        for step_id, step in step_by_id.items():
            entry = best.formulas.get(step_id)
            template = str(step.get("template_name") or "unknown")
            if entry is None:
                none_formula_steps += 1
                no_formula_step_records.append(
                    {
                        "dataset_id": record.get("dataset_id"),
                        "row_index": record.get("row_index"),
                        "question": record.get("question"),
                        "parse_status": status,
                        "domains": parse.get("domains"),
                        "sub_domains": parse.get("sub_domains"),
                        "unknown_quantity": parse.get("unknown_quantity"),
                        "step_id": step_id,
                        "step_goal": step.get("goal"),
                        "template_name": template,
                        "input_var": step.get("input_var"),
                        "output_var": step.get("output_var"),
                        "retrieval_confidence": best.retrieval_confidence,
                    }
                )
                continue

            matched_formula_steps += 1
            formula_counts[entry.id] += 1
            template_formula_key = f"{template} -> {entry.id}"
            template_formula_counts[template_formula_key] += 1
            formula_by_template[template][entry.id] += 1

            overlap = _formula_overlap(step, entry)
            if overlap < 0.4:
                suspicious_records.append(
                    {
                        "dataset_id": record.get("dataset_id"),
                        "row_index": record.get("row_index"),
                        "question": record.get("question"),
                        "reason": "low_formula_name_overlap",
                        "template_name": template,
                        "step_formula_name": step.get("formula_name"),
                        "formula_id": entry.id,
                        "formula": entry.formula,
                        "sympy_expr": entry.sympy_expr,
                        "overlap": round(overlap, 4),
                        "retrieval_confidence": best.retrieval_confidence,
                        "step_goal": step.get("goal"),
                    }
                )

            if template != "unknown":
                template_tokens = set(template.lower().split("_"))
                formula_tokens = {entry.topic.lower(), entry.subtopic.lower(), entry.id.lower()}
                if not template_tokens & set(entry.subtopic.lower().split("_")) and conf < 0.2:
                    suspicious_records.append(
                        {
                            "dataset_id": record.get("dataset_id"),
                            "row_index": record.get("row_index"),
                            "question": record.get("question"),
                            "reason": "low_confidence_template_formula_mismatch",
                            "template_name": template,
                            "formula_id": entry.id,
                            "formula_topic": entry.topic,
                            "formula_subtopic": entry.subtopic,
                            "retrieval_confidence": best.retrieval_confidence,
                            "step_goal": step.get("goal"),
                        }
                    )

    average_confidence = (
        sum(retrieval_confidences) / len(retrieval_confidences)
        if retrieval_confidences
        else 0.0
    )
    formula_step_coverage = (
        matched_formula_steps / total_formula_steps
        if total_formula_steps
        else 0.0
    )

    summary: Dict[str, Any] = {
        "results_path": str(results_path),
        "total_records": total,
        "parse_pass_records": parse_pass,
        "parse_fail_records": parse_fail,
        "total_formula_steps": total_formula_steps,
        "matched_formula_steps": matched_formula_steps,
        "none_formula_steps": none_formula_steps,
        "formula_step_coverage": round(formula_step_coverage, 4),
        "average_retrieval_confidence": round(average_confidence, 4),
        "confidence_hist": dict(confidence_hist),
        "path_count_hist": dict(path_count_hist),
        "top_formulas": dict(formula_counts.most_common(25)),
        "top_templates": dict(template_counts.most_common(25)),
        "top_template_formula_pairs": dict(template_formula_counts.most_common(40)),
        "domain_counts": dict(domain_counts.most_common()),
        "templates_to_formulas": {
            template: dict(counter.most_common(10))
            for template, counter in sorted(formula_by_template.items())
        },
        "outputs": {
            "summary": str(output_dir / "stage1_summary.json"),
            "no_formula_steps": str(output_dir / "stage1_no_formula_steps.jsonl"),
            "suspicious": str(output_dir / "stage1_suspicious.jsonl"),
        },
    }

    (output_dir / "stage1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "stage1_no_formula_steps.jsonl", no_formula_step_records)
    _write_jsonl(output_dir / "stage1_suspicious.jsonl", suspicious_records)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 formula retrieval on saved Stage 0 results.")
    parser.add_argument("--results", default="outputs/stage0_with_llm_v2/stage0_results.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage1_eval")
    parser.add_argument("--beam-n", type=int, default=3)
    args = parser.parse_args()

    summary = evaluate(
        results_path=(ROOT / args.results).resolve(),
        output_dir=(ROOT / args.output_dir).resolve(),
        beam_n=args.beam_n,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
