"""Validate Type2 numeric execution through the candidate pipeline."""

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
    execution_status_counts: Counter[str] = Counter()
    selected_source_counts: Counter[str] = Counter()
    selected_template_counts: Counter[str] = Counter()
    top_execution_warnings: Counter[str] = Counter()
    top_execution_error_types: Counter[str] = Counter()
    numeric_by_template_counts: Counter[str] = Counter()
    symbolic_by_template_counts: Counter[str] = Counter()
    fail_by_template_counts: Counter[str] = Counter()
    target_unresolved_by_template_counts: Counter[str] = Counter()
    missing_input_by_variable_counts: Counter[str] = Counter()
    point_target_inference_counts: Counter[str] = Counter()
    coulomb_scene_failure_reasons: Counter[str] = Counter()
    confidences: list[float] = []
    low_confidence_count = 0
    numeric_answer_count = 0
    symbolic_trace_count = 0
    unresolved_count = 0
    failed_count = 0
    vector_sum_symbolic_count = 0
    vector_sum_executed_count = 0
    chained_equality_patch_count = 0
    distance_alias_patch_count = 0
    target_alias_resolution_count = 0
    role_aware_coulomb_used_count = 0
    role_aware_coulomb_numeric_count = 0
    role_aware_coulomb_symbolic_count = 0
    role_aware_coulomb_ambiguous_count = 0
    k_missing_count = 0
    pair_force_executed_count = 0
    law_of_cosines_composition_count = 0
    law_of_cosines_numeric_count = 0
    target_point_unresolved_count = 0
    no_charge_point_assignments_count = 0
    no_pair_forces_executed_count = 0
    labeled_charge_patch_count = 0
    force_on_point_pattern_count = 0
    force_on_charge_pattern_count = 0
    charge_sign_adjustment_count = 0

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
        template_names = [str(template_name) for template_name in final_answer.get("template_names", []) or []]
        if not template_names:
            template_names = ["unknown"]
        for template_name in template_names:
            selected_template_counts[str(template_name)] += 1
            if answer_type == "numeric":
                numeric_by_template_counts[template_name] += 1
            if answer_type == "symbolic_trace":
                symbolic_by_template_counts[template_name] += 1
        metadata = final_answer.get("metadata", {}) if isinstance(final_answer.get("metadata"), dict) else {}
        execution_result = metadata.get("execution_result", {}) if isinstance(metadata.get("execution_result"), dict) else {}
        if execution_result:
            execution_status = str(execution_result.get("status", ""))
            execution_status_counts[execution_status] += 1
            if execution_status == "FAIL":
                for template_name in template_names:
                    fail_by_template_counts[template_name] += 1
            warning_suggests_symbolic_vector = False
            for warning in execution_result.get("warnings", []) or []:
                top_execution_warnings[str(warning)] += 1
                if "Vector sum left symbolic" in str(warning) or "Ambiguous vector geometry" in str(warning):
                    warning_suggests_symbolic_vector = True
            for error in execution_result.get("errors", []) or []:
                if isinstance(error, dict):
                    error_type = str(error.get("error_type", "unknown"))
                    top_execution_error_types[error_type] += 1
                    if error_type == "target_unresolved":
                        for template_name in template_names:
                            target_unresolved_by_template_counts[template_name] += 1
                    if error_type == "missing_inputs":
                        for variable in error.get("missing_vars", []) or []:
                            missing_input_by_variable_counts[str(variable)] += 1
                            if str(variable) == "k":
                                k_missing_count += 1
            execution_metadata = execution_result.get("metadata", {}) if isinstance(execution_result.get("metadata"), dict) else {}
            vector_sum_mode = execution_metadata.get("vector_sum_mode")
            if vector_sum_mode in {"right_angle", "equilateral", "collinear_opposite", "same_direction"}:
                vector_sum_executed_count += 1
            elif vector_sum_mode in {"symbolic_ambiguous", "symbolic_missing_pair_forces"}:
                vector_sum_symbolic_count += 1
            elif warning_suggests_symbolic_vector:
                vector_sum_symbolic_count += 1
            chained_equality_patch_count += len(execution_metadata.get("chained_equality_patches", []) or [])
            distance_alias_patch_count += len(execution_metadata.get("distance_alias_patches", []) or [])
            if execution_metadata.get("target_alias_used"):
                target_alias_resolution_count += 1
            if execution_metadata.get("role_aware_coulomb_used"):
                role_aware_coulomb_used_count += 1
                if answer_type == "numeric":
                    role_aware_coulomb_numeric_count += 1
                else:
                    role_aware_coulomb_symbolic_count += 1
                geometry_mode = str(execution_metadata.get("role_aware_geometry_mode") or "unknown")
                if geometry_mode in {"unknown", "perpendicular_bisector"} or execution_metadata.get("coulomb_scene_warnings"):
                    role_aware_coulomb_ambiguous_count += 1
                target_point = execution_metadata.get("target_point")
                if target_point:
                    point_target_inference_counts[str(target_point)] += 1
                for reason in execution_metadata.get("coulomb_scene_failure_reasons", []) or []:
                    coulomb_scene_failure_reasons[str(reason)] += 1
                    if str(reason) == "target_point_unresolved":
                        target_point_unresolved_count += 1
                    if str(reason) == "no_charge_point_assignments":
                        no_charge_point_assignments_count += 1
                    if str(reason) == "no_pair_forces_executed":
                        no_pair_forces_executed_count += 1
                pair_force_executed_count += int(execution_metadata.get("role_aware_pair_force_count") or 0)
            if execution_metadata.get("law_of_cosines_used"):
                law_of_cosines_composition_count += 1
                if answer_type == "numeric":
                    law_of_cosines_numeric_count += 1
            labeled_charge_patch_count += len(execution_metadata.get("labeled_charge_patches", []) or [])
            inference_source = str(execution_metadata.get("target_point_inference_source") or "")
            if inference_source == "force_on_point_pattern":
                force_on_point_pattern_count += 1
            if inference_source == "force_on_charge_pattern":
                force_on_charge_pattern_count += 1
            if execution_metadata.get("charge_interaction_adjustment"):
                charge_sign_adjustment_count += 1

    return {
        "pipeline_status_counts": dict(pipeline_status_counts.most_common()),
        "parser_status_counts": dict(parser_status_counts.most_common()),
        "answer_type_counts": dict(answer_type_counts.most_common()),
        "numeric_answer_count": numeric_answer_count,
        "symbolic_trace_count": symbolic_trace_count,
        "unresolved_count": unresolved_count,
        "execution_status_counts": dict(execution_status_counts.most_common()),
        "mean_confidence": _mean(confidences),
        "low_confidence_count": low_confidence_count,
        "selected_source_counts": dict(selected_source_counts.most_common()),
        "selected_template_counts": dict(selected_template_counts.most_common()),
        "top_execution_warnings": dict(top_execution_warnings.most_common()),
        "top_execution_error_types": dict(top_execution_error_types.most_common()),
        "numeric_by_template_counts": dict(numeric_by_template_counts.most_common()),
        "symbolic_by_template_counts": dict(symbolic_by_template_counts.most_common()),
        "fail_by_template_counts": dict(fail_by_template_counts.most_common()),
        "target_unresolved_by_template_counts": dict(target_unresolved_by_template_counts.most_common()),
        "missing_input_by_variable_counts": dict(missing_input_by_variable_counts.most_common()),
        "vector_sum_symbolic_count": vector_sum_symbolic_count,
        "vector_sum_executed_count": vector_sum_executed_count,
        "chained_equality_patch_count": chained_equality_patch_count,
        "distance_alias_patch_count": distance_alias_patch_count,
        "target_alias_resolution_count": target_alias_resolution_count,
        "role_aware_coulomb_used_count": role_aware_coulomb_used_count,
        "role_aware_coulomb_numeric_count": role_aware_coulomb_numeric_count,
        "role_aware_coulomb_symbolic_count": role_aware_coulomb_symbolic_count,
        "role_aware_coulomb_ambiguous_count": role_aware_coulomb_ambiguous_count,
        "k_missing_count": k_missing_count,
        "point_target_inference_counts": dict(point_target_inference_counts.most_common()),
        "coulomb_scene_failure_reasons": dict(coulomb_scene_failure_reasons.most_common()),
        "pair_force_executed_count": pair_force_executed_count,
        "law_of_cosines_composition_count": law_of_cosines_composition_count,
        "law_of_cosines_numeric_count": law_of_cosines_numeric_count,
        "target_point_unresolved_count": target_point_unresolved_count,
        "no_charge_point_assignments_count": no_charge_point_assignments_count,
        "no_pair_forces_executed_count": no_pair_forces_executed_count,
        "labeled_charge_patch_count": labeled_charge_patch_count,
        "force_on_point_pattern_count": force_on_point_pattern_count,
        "force_on_charge_pattern_count": force_on_charge_pattern_count,
        "charge_sign_adjustment_count": charge_sign_adjustment_count,
        "failed_count": failed_count,
    }


def validate_numeric_executor(
    input_path: Path,
    limit: int,
    output_path: Path,
    use_llm_fallback: bool = False,
    max_candidates: int = 8,
    execute_numeric: bool = True,
    compact: bool = False,
) -> dict[str, Any]:
    rows = _load_csv(input_path)
    text_column = _detect_text_column(rows)
    selected_rows = rows[: max(limit, 0)]
    config = Type2CandidatePipelineConfig(
        use_llm_fallback=use_llm_fallback,
        log_failures=False,
        max_candidates=max_candidates,
        execute_numeric=execute_numeric,
        include_intermediate_outputs=not compact,
        include_scoreboards=True,
    )
    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(selected_rows, start=1):
        result = run_type2_candidate_pipeline(row.get(text_column, ""), config)
        row_result = result.to_dict()
        if compact:
            row_result = _compact_result(row_result)
        row_result["row_index"] = row_index
        output_rows.append(row_result)

    output = {
        "input_path": str(input_path),
        "evaluated": len(output_rows),
        "summary": _build_summary(output_rows, config),
        "rows": output_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Type2 numeric executor.")
    parser.add_argument("--input", default="Physics_Problems_Text_Only.csv")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="type2_numeric_executor_validation.json")
    parser.add_argument("--use-llm-fallback", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--no-execute-numeric", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    result = validate_numeric_executor(
        input_path=_resolve_input_path(args.input),
        limit=args.limit,
        output_path=Path(args.output).resolve(),
        use_llm_fallback=args.use_llm_fallback,
        max_candidates=args.max_candidates,
        execute_numeric=not args.no_execute_numeric,
        compact=args.compact,
    )
    print(json.dumps({"evaluated": result["evaluated"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
