#!/usr/bin/env python3
"""Evaluate Type1 records against a running EXACT /predict endpoint."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


MCQ_RE = re.compile(
    r"(?:^|\n)\s*([A-D])\.\s*(.*?)(?=(?:\n\s*[A-D]\.\s*)|$)",
    re.S,
)


def extract_mcq_labels(question: str) -> list[str]:
    labels = [match.group(1) for match in MCQ_RE.finditer(question)]
    return labels if len(labels) >= 2 else []


def normalize_gold(answer: Any, options: list[str]) -> str:
    value = str(answer).strip()
    aliases = {"false": "No", "true": "Yes", "unknown": "Uncertain"}
    return aliases.get(value.lower(), value)


def post_predict(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, list) or len(data) != 1:
        raise RuntimeError(f"unexpected response shape: {data!r}")
    return data[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the first N Type1 records.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../Logic_Based_Educational_Queries.json"),
    )
    parser.add_argument("--records", type=int, default=50)
    parser.add_argument("--url", default="http://127.0.0.1:8080/predict")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("type1_50_record_eval_results.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("type1_50_record_eval_summary.json"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    correct = 0
    total = 0
    failures = 0

    for record_index, record in enumerate(data[: args.records]):
        premises = record.get("premises-NL") or record.get("premises") or []
        questions = record.get("questions") or []
        answers = record.get("answers") or []
        for question_index, (question, gold_raw) in enumerate(zip(questions, answers)):
            options = extract_mcq_labels(str(question))
            gold = normalize_gold(gold_raw, options)
            if options and "Uncertain" not in options:
                options.append("Uncertain")
            if options and gold not in options:
                options.append(gold)
            if not options:
                options = ["Yes", "No", "Uncertain"]
            payload = {
                "query_id": f"T1_R{record_index:04d}_Q{question_index:02d}",
                "type": "type1",
                "query": question,
                "premises": premises,
                "options": options,
            }
            total += 1
            try:
                pred = post_predict(args.url, payload, args.timeout)
                predicted = str(pred.get("answer", "")).strip()
                is_correct = predicted == gold
                correct += int(is_correct)
                row = {
                    "record_index": record_index,
                    "question_index": question_index,
                    "query_id": payload["query_id"],
                    "options": options,
                    "gold": gold,
                    "predicted": predicted,
                    "correct": is_correct,
                    "premises_used": pred.get("premises_used"),
                    "explanation": pred.get("explanation"),
                }
            except Exception as exc:
                failures += 1
                row = {
                    "record_index": record_index,
                    "question_index": question_index,
                    "query_id": payload["query_id"],
                    "options": options,
                    "gold": gold,
                    "predicted": "",
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)

    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "records_requested": args.records,
        "questions_evaluated": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "request_failures": failures,
        "output": str(args.output),
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
