"""Validate the end-to-end Type2 candidate pipeline over a CSV dataset."""

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

from parser.pipeline.type2_candidate_pipeline import Type2CandidatePipelineConfig, run_type2_candidate_pipeline


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


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    intermediate = result.get("intermediate_outputs", {}) if isinstance(result.get("intermediate_outputs"), dict) else {}
    return {
        "problem_text": result.get("problem_text"),
        "final_answer": result.get("final_answer"),
        "parser_status": result.get("parser_status"),
        "pipeline_status": result.get("pipeline_status"),
        "pipeline_warnings": result.get("pipeline_warnings", []),
        "pipeline_errors": result.get("pipeline_errors", []),
        "adapter_diagnostics": result.get("adapter_diagnostics", {}),
        "generation_summary": result.get("generation_summary", {}),
        "verification_summary": result.get("verification_summary", {}),
        "ranking_summary": result.get("ranking_summary", {}),
        "selected_candidate": result.get("selected_candidate"),
        "compact_scoreboard": intermediate.get("compact_scoreboard", []),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_summary(rows: list[dict[str, Any]], config: Type2CandidatePipelineConfig) -> dict[str, Any]:
    pipeline_status_counts: Counter[str] = Counter()
    parser_status_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    selected_source_counts: Counter[str] = Counter()
    selected_template_counts: Counter[str] = Counter()
    top_pipeline_warnings: Counter[str] = Counter()
    top_pipeline_error_types: Counter[str] = Counter()
    confidences: list[float] = []
    low_confidence_count = 0
    low_rank_margin_count = 0
    numeric_answer_count = 0
    symbolic_trace_count = 0
    unresolved_count = 0
    failed_count = 0

    for row in rows:
        pipeline_status = str(row.get("pipeline_status", ""))
        pipeline_status_counts[pipeline_status] += 1
        if pipeline_status == "ERROR":
            failed_count += 1
        parser_status_counts[str(row.get("parser_status", ""))] += 1

        final_answer = row.get("final_answer", {}) if isinstance(row.get("final_answer"), dict) else {}
        answer_type = str(final_answer.get("answer_type", ""))
        answer_type_counts[answer_type] += 1
        confidence = float(final_answer.get("confidence", 0.0) or 0.0)
        confidences.append(confidence)
        if confidence < config.low_confidence_threshold:
            low_confidence_count += 1
        if answer_type == "numeric":
            numeric_answer_count += 1
        if answer_type == "symbolic_trace":
            symbolic_trace_count += 1
        if answer_type == "unresolved":
            unresolved_count += 1

        source = final_answer.get("source")
        if source:
            selected_source_counts[str(source)] += 1
        for template_name in final_answer.get("template_names", []) or []:
            selected_template_counts[str(template_name)] += 1

        ranking_summary = row.get("ranking_summary", {}) if isinstance(row.get("ranking_summary"), dict) else {}
        rank_margin = float(ranking_summary.get("rank_margin", 0.0) or 0.0)
        if rank_margin < config.min_rank_margin_for_high_confidence:
            low_rank_margin_count += 1

        for warning in row.get("pipeline_warnings", []) or []:
            top_pipeline_warnings[str(warning)] += 1
        for error in row.get("pipeline_errors", []) or []:
            if isinstance(error, dict):
                top_pipeline_error_types[str(error.get("error_type", "unknown"))] += 1

    return {
        "pipeline_status_counts": dict(pipeline_status_counts.most_common()),
        "parser_status_counts": dict(parser_status_counts.most_common()),
        "answer_type_counts": dict(answer_type_counts.most_common()),
        "selected_source_counts": dict(selected_source_counts.most_common()),
        "selected_template_counts": dict(selected_template_counts.most_common()),
        "mean_confidence": _mean(confidences),
        "low_confidence_count": low_confidence_count,
        "low_rank_margin_count": low_rank_margin_count,
        "numeric_answer_count": numeric_answer_count,
        "symbolic_trace_count": symbolic_trace_count,
        "unresolved_count": unresolved_count,
        "failed_count": failed_count,
        "top_pipeline_warnings": dict(top_pipeline_warnings.most_common()),
        "top_pipeline_error_types": dict(top_pipeline_error_types.most_common()),
    }


def validate_pipeline(
    input_path: Path,
    limit: int,
    output_path: Path,
    use_llm_fallback: bool = False,
    max_candidates: int = 8,
    compact: bool = False,
) -> dict[str, Any]:
    rows = _load_csv(input_path)
    text_column = _detect_text_column(rows)
    selected_rows = rows[: max(limit, 0)]
    config = Type2CandidatePipelineConfig(
        use_llm_fallback=use_llm_fallback,
        log_failures=False,
        max_candidates=max_candidates,
        include_intermediate_outputs=not compact,
        include_scoreboards=True,
    )

    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(selected_rows, start=1):
        problem_text = row.get(text_column, "")
        result = run_type2_candidate_pipeline(problem_text, config)
        row_result = result.to_dict()
        if compact:
            row_result = _compact_result(row_result)
        row_result["row_index"] = row_index
        output_rows.append(row_result)

    result = {
        "input_path": str(input_path),
        "evaluated": len(output_rows),
        "summary": _build_summary(output_rows, config),
        "rows": output_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Type2 candidate pipeline.")
    parser.add_argument("--input", default="Physics_Problems_Text_Only.csv")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="type2_candidate_pipeline_validation.json")
    parser.add_argument("--use-llm-fallback", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    result = validate_pipeline(
        input_path=_resolve_input_path(args.input),
        limit=args.limit,
        output_path=Path(args.output).resolve(),
        use_llm_fallback=args.use_llm_fallback,
        max_candidates=args.max_candidates,
        compact=args.compact,
    )
    print(json.dumps({"evaluated": result["evaluated"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
