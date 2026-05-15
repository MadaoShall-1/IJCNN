"""Run the Stage 0 parser over a dataset and write parse artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parser.main import parse_problem


def _load_csv(path: Path, text_field: str, id_field: str) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    if text_field not in rows[0]:
        raise ValueError(f"Text field '{text_field}' not found. Available fields: {list(rows[0])}")
    return rows


def _write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_dataset(
    dataset_path: Path,
    output_dir: Path,
    text_field: str = "question",
    id_field: str = "id",
    limit: int | None = None,
    use_llm_fallback: bool = False,
) -> Dict[str, object]:
    """Run Stage 0 on a CSV dataset and write results, failures, and summary."""
    rows = _load_csv(dataset_path, text_field=text_field, id_field=id_field)
    if limit is not None:
        rows = rows[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "stage0_results.jsonl"
    failures_path = output_dir / "stage0_failures.jsonl"
    summary_path = output_dir / "stage0_summary.json"

    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    template_name_counts: Counter[str] = Counter()
    relation_type_counts: Counter[str] = Counter()
    extracted_relation_total = 0
    extracted_uncertainty_total = 0
    results: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for index, row in enumerate(rows, start=1):
        problem_text = row.get(text_field, "")
        item_id = row.get(id_field) or str(index)
        try:
            parse = parse_problem(
                problem_text,
                use_llm_fallback=use_llm_fallback,
                log_failures=False,
            )
        except Exception as exc:
            parse = {
                "problem_text": problem_text,
                "metadata": {
                    "verifier_status": "FAIL",
                    "verifier_errors": [
                        {
                            "error_type": "parser_exception",
                            "description": str(exc),
                            "repair_hint": "Inspect parser exception and add a guard.",
                        }
                    ],
                },
            }

        status = parse.get("metadata", {}).get("verifier_status", "FAIL")
        status_counts[str(status)] += 1
        for domain in parse.get("domains", []) or ["unknown"]:
            domain_counts[str(domain)] += 1
        target = parse.get("unknown_quantity") or "null"
        target_counts[str(target)] += 1
        metadata = parse.get("metadata", {})
        extracted_relation_total += int(metadata.get("extracted_relation_count", 0) or 0)
        extracted_uncertainty_total += int(metadata.get("extracted_uncertainty_count", 0) or 0)
        relation_type_counts.update(str(relation.get("type", "unknown")) for relation in parse.get("relations", []) or [])
        if metadata.get("used_template_fallback"):
            fallback_counts["template"] += 1
            template_name_counts.update(str(name) for name in metadata.get("used_template_names", []))
        if metadata.get("used_llm_fallback"):
            fallback_counts["llm"] += 1

        errors = metadata.get("verifier_errors", [])
        for error in errors:
            error_counts[str(error.get("error_type", "unknown"))] += 1

        record = {
            "dataset_id": item_id,
            "row_index": index,
            "question": problem_text,
            "answer": row.get("answer"),
            "unit": row.get("unit"),
            "parse": parse,
        }
        results.append(record)
        if status != "PASS":
            failures.append(record)

    total = len(rows)
    summary: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "total": total,
        "pass": status_counts.get("PASS", 0),
        "fail": status_counts.get("FAIL", 0),
        "pass_rate": round(status_counts.get("PASS", 0) / total, 4) if total else 0.0,
        "status_counts": dict(status_counts),
        "error_counts": dict(error_counts.most_common()),
        "domain_counts": dict(domain_counts.most_common()),
        "target_counts": [
            {"target": target, "count": count}
            for target, count in target_counts.most_common(25)
        ],
        "fallback_counts": dict(fallback_counts),
        "template_name_counts": dict(template_name_counts.most_common()),
        "relation_type_counts": dict(relation_type_counts.most_common()),
        "extracted_relation_total": extracted_relation_total,
        "extracted_uncertainty_total": extracted_uncertainty_total,
        "outputs": {
            "results": str(results_path),
            "failures": str(failures_path),
            "summary": str(summary_path),
        },
    }

    _write_jsonl(results_path, results)
    _write_jsonl(failures_path, failures)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 0 parser over a physics dataset.")
    parser.add_argument("--dataset", default="Dataset/Physics_Problems_Text_Only.csv")
    parser.add_argument("--output-dir", default="outputs/stage0")
    parser.add_argument("--text-field", default="question")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-llm-fallback", action="store_true")
    args = parser.parse_args()

    summary = run_dataset(
        dataset_path=(ROOT / args.dataset).resolve(),
        output_dir=(ROOT / args.output_dir).resolve(),
        text_field=args.text_field,
        id_field=args.id_field,
        limit=args.limit,
        use_llm_fallback=args.use_llm_fallback,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
