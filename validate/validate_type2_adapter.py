"""Validate Stage 0 to Type2 adapter output over a CSV dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parser.pipeline.type2_adapter import parse_and_adapt


TEXT_COLUMN_CANDIDATES = ("problem_text", "question", "text")


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path.resolve()

    root_path = ROOT / raw_path
    if root_path.exists():
        return root_path.resolve()

    dataset_path = ROOT / "Dataset" / path.name
    if not path.parent.parts and dataset_path.exists():
        return dataset_path.resolve()

    return path.resolve()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _detect_text_column(rows: list[dict[str, str]]) -> str:
    if not rows:
        raise ValueError("Input CSV is empty.")

    fieldnames = list(rows[0].keys())
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate

    for fieldname in fieldnames:
        if any(isinstance(row.get(fieldname), str) and row.get(fieldname, "").strip() for row in rows):
            return fieldname

    raise ValueError(f"No string-like problem text column found. Available fields: {fieldnames}")


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _mean(total: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return total / denominator


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = len(rows)
    failed_count = sum(1 for row in rows if "error" in row)
    successful_diagnostics = [
        row["diagnostics"]
        for row in rows
        if "error" not in row and isinstance(row.get("diagnostics"), dict)
    ]

    verifier_status_counts: Counter[str] = Counter()
    top_error_types: Counter[str] = Counter()
    quantity_total = 0
    relation_total = 0
    condition_total = 0
    step_total = 0

    has_target_count = 0
    has_known_quantities_count = 0
    has_step_plan_count = 0
    has_real_step_plan_count = 0
    skeleton_fallback_count = 0

    for diagnostics in successful_diagnostics:
        if diagnostics.get("has_target"):
            has_target_count += 1
        if diagnostics.get("has_known_quantities"):
            has_known_quantities_count += 1
        if diagnostics.get("has_step_plan"):
            has_step_plan_count += 1
        if diagnostics.get("has_real_step_plan"):
            has_real_step_plan_count += 1
        if diagnostics.get("uses_skeleton_fallback"):
            skeleton_fallback_count += 1

        verifier_status_counts[str(diagnostics.get("verifier_status", ""))] += 1
        top_error_types.update(str(error_type) for error_type in diagnostics.get("error_types", []))
        quantity_total += int(diagnostics.get("quantity_count", 0) or 0)
        relation_total += int(diagnostics.get("relation_count", 0) or 0)
        condition_total += int(diagnostics.get("condition_count", 0) or 0)
        step_total += int(diagnostics.get("step_count", 0) or 0)

    return {
        "has_target_rate": _rate(has_target_count, evaluated),
        "has_known_quantities_rate": _rate(has_known_quantities_count, evaluated),
        "has_step_plan_rate": _rate(has_step_plan_count, evaluated),
        "has_real_step_plan_rate": _rate(has_real_step_plan_count, evaluated),
        "skeleton_fallback_rate": _rate(skeleton_fallback_count, evaluated),
        "verifier_status_counts": dict(verifier_status_counts),
        "top_error_types": dict(top_error_types.most_common()),
        "mean_quantity_count": _mean(quantity_total, evaluated),
        "mean_relation_count": _mean(relation_total, evaluated),
        "mean_condition_count": _mean(condition_total, evaluated),
        "mean_step_count": _mean(step_total, evaluated),
        "failed_count": failed_count,
    }


def validate_adapter(
    input_path: Path,
    limit: int,
    output_path: Path,
    use_llm_fallback: bool = False,
) -> dict[str, Any]:
    rows = _load_csv(input_path)
    text_column = _detect_text_column(rows)
    selected_rows = rows[: max(limit, 0)]

    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(selected_rows, start=1):
        problem_text = row.get(text_column, "")
        try:
            world_input, diagnostics = parse_and_adapt(
                problem_text,
                use_llm_fallback=use_llm_fallback,
                log_failures=False,
            )
            output_rows.append(
                {
                    "row_index": row_index,
                    "problem_text": problem_text,
                    "world_input": world_input.to_dict(),
                    "diagnostics": diagnostics.to_dict(),
                }
            )
        except Exception as exc:
            output_rows.append(
                {
                    "row_index": row_index,
                    "problem_text": problem_text,
                    "error": repr(exc),
                }
            )

    result = {
        "input_path": str(input_path),
        "evaluated": len(output_rows),
        "summary": _build_summary(output_rows),
        "rows": output_rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Type2 Stage 0 adapter.")
    parser.add_argument("--input", default="Physics_Problems_Text_Only.csv")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="type2_adapter_validation.json")
    parser.add_argument("--use-llm-fallback", action="store_true")
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)
    output_path = Path(args.output).resolve()
    result = validate_adapter(
        input_path=input_path,
        limit=args.limit,
        output_path=output_path,
        use_llm_fallback=args.use_llm_fallback,
    )
    print(json.dumps({"evaluated": result["evaluated"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
