"""Validate Type2 candidate verification and feature vectors over a CSV dataset."""

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
from parser.pipeline.type2_candidate_generator import (
    Type2CandidateGenerationResult,
    deduplicate_candidates,
    generate_step_plan_candidates,
)
from parser.pipeline.type2_candidate_verifier import (
    PENALTY_FEATURES,
    TYPE2_CANDIDATE_FEATURE_NAMES,
    verify_step_plan_candidates,
)


TEXT_COLUMN_CANDIDATES = ("problem_text", "question", "text")
LOW_SCORE_FEATURES = {
    "target_match_score",
    "unit_match_score",
    "dimension_consistency_score",
    "known_quantity_coverage",
    "condition_coverage",
    "relation_coverage",
    "formula_target_alignment",
    "input_availability_score",
    "step_validity_score",
    "template_domain_score",
    "geometry_condition_score",
}


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


def _apply_max_candidates(
    result: Type2CandidateGenerationResult,
    max_candidates: int,
) -> Type2CandidateGenerationResult:
    candidates = deduplicate_candidates(result.candidates, max_candidates=max_candidates)
    selected_candidate_id = candidates[0].candidate_id if candidates else None
    summary = dict(result.generation_summary)
    summary["candidate_count"] = len(candidates)
    summary["max_candidates"] = max_candidates
    return Type2CandidateGenerationResult(
        problem_text=result.problem_text,
        target=result.target,
        target_unit=result.target_unit,
        candidates=candidates,
        selected_candidate_id=selected_candidate_id,
        generation_summary=summary,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _score_margin(verified_candidates: list[dict[str, Any]]) -> float:
    if len(verified_candidates) >= 2:
        return float(verified_candidates[0]["score"]) - float(verified_candidates[1]["score"])
    if verified_candidates:
        return float(verified_candidates[0]["score"])
    return 0.0


def _top_positive_features(feature_values: dict[str, float]) -> list[dict[str, float]]:
    items = [
        (name, value)
        for name, value in feature_values.items()
        if name not in PENALTY_FEATURES and name != "overall_candidate_score" and value > 0
    ]
    items.sort(key=lambda item: (-item[1], item[0]))
    return [{"feature": name, "value": value} for name, value in items[:5]]


def _top_negative_features(feature_values: dict[str, float]) -> list[dict[str, float]]:
    negatives: list[tuple[str, float]] = []
    for name in PENALTY_FEATURES:
        value = float(feature_values.get(name, 0.0))
        if value > 0.15:
            negatives.append((name, value))
    for name in LOW_SCORE_FEATURES:
        value = float(feature_values.get(name, 1.0))
        if value < 0.5:
            negatives.append((name, value))
    negatives.sort(key=lambda item: (-item[1] if item[0] in PENALTY_FEATURES else item[1], item[0]))
    return [{"feature": name, "value": value} for name, value in negatives[:5]]


def _candidate_scoreboard(verification_result_dict: dict[str, Any]) -> list[dict[str, Any]]:
    scoreboard: list[dict[str, Any]] = []
    for candidate in verification_result_dict.get("verified_candidates", []):
        feature_values = candidate.get("feature_values", {})
        scoreboard.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "source": candidate.get("source"),
                "template_names": candidate.get("template_names", []),
                "score": candidate.get("score"),
                "confidence": candidate.get("confidence"),
                "verifier_status": candidate.get("verifier_status"),
                "oversimplified_scalar_penalty": feature_values.get("oversimplified_scalar_penalty", 0.0),
                "warning_penalty": feature_values.get("warning_penalty", 0.0),
                "missing_input_penalty": feature_values.get("missing_input_penalty", 0.0),
                "input_availability_score": feature_values.get("input_availability_score", 0.0),
                "formula_target_alignment": feature_values.get("formula_target_alignment", 0.0),
                "top_negative_features": _top_negative_features(feature_values),
                "top_positive_features": _top_positive_features(feature_values),
            }
        )
    return scoreboard


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = len(rows)
    failed_count = sum(1 for row in rows if "error" in row)
    successful_rows = [row for row in rows if "error" not in row]

    candidate_counts: list[int] = []
    verified_candidate_counts: list[int] = []
    selected_scores: list[float] = []
    selected_confidences: list[float] = []
    score_margins: list[float] = []
    selected_candidate_source_counts: Counter[str] = Counter()
    selected_candidate_template_counts: Counter[str] = Counter()
    verifier_status_counts: Counter[str] = Counter()
    candidate_verifier_status_counts: Counter[str] = Counter()
    top_error_types: Counter[str] = Counter()
    feature_totals: Counter[str] = Counter()
    feature_count = 0
    legacy_selected_count = 0
    deterministic_selected_count = 0
    skeleton_selected_count = 0
    selected_score_1_count = 0
    selected_score_ge_0_99_count = 0
    margin_lt_0_01_count = 0
    margin_lt_0_05_count = 0
    oversimplified_scalar_penalty_count = 0

    for row in successful_rows:
        adapter = row.get("adapter_diagnostics", {})
        verifier_status_counts[str(adapter.get("verifier_status", ""))] += 1

        generation = row.get("candidate_generation_result", {})
        generation_candidates = generation.get("candidates", []) if isinstance(generation, dict) else []
        candidate_counts.append(len(generation_candidates))

        verification = row.get("candidate_verification_result", {})
        verified = verification.get("verified_candidates", []) if isinstance(verification, dict) else []
        verified_candidate_counts.append(len(verified))
        selected_candidate_id = verification.get("selected_candidate_id") if isinstance(verification, dict) else None
        margin = _score_margin(verified)
        score_margins.append(margin)
        if margin < 0.01:
            margin_lt_0_01_count += 1
        if margin < 0.05:
            margin_lt_0_05_count += 1

        selected = None
        for candidate in verified:
            candidate_verifier_status_counts[str(candidate.get("verifier_status", ""))] += 1
            for error in candidate.get("verifier_errors", []) or []:
                if isinstance(error, dict):
                    top_error_types[str(error.get("error_type", ""))] += 1
            feature_values = candidate.get("feature_values", {})
            for name in TYPE2_CANDIDATE_FEATURE_NAMES:
                feature_totals[name] += float(feature_values.get(name, 0.0))
            if float(feature_values.get("oversimplified_scalar_penalty", 0.0)) > 0:
                oversimplified_scalar_penalty_count += 1
            feature_count += 1
            if candidate.get("candidate_id") == selected_candidate_id:
                selected = candidate

        if selected:
            source = str(selected.get("source", ""))
            selected_candidate_source_counts[source] += 1
            if source == "legacy_parser_step_plan":
                legacy_selected_count += 1
            if source == "deterministic_variant":
                deterministic_selected_count += 1
            selected_scores.append(float(selected.get("score", 0.0)))
            selected_confidences.append(float(selected.get("confidence", 0.0)))
            selected_score = float(selected.get("score", 0.0))
            if selected_score == 1.0:
                selected_score_1_count += 1
            if selected_score >= 0.99:
                selected_score_ge_0_99_count += 1
            templates = selected.get("template_names", []) or []
            for template_name in templates:
                selected_candidate_template_counts[str(template_name)] += 1
            if any(str(template_name) == "skeleton_placeholder" for template_name in templates):
                skeleton_selected_count += 1

    feature_means = {
        name: (feature_totals[name] / feature_count if feature_count else 0.0)
        for name in TYPE2_CANDIDATE_FEATURE_NAMES
    }

    return {
        "mean_candidate_count": _mean([float(value) for value in candidate_counts]),
        "mean_verified_candidate_count": _mean([float(value) for value in verified_candidate_counts]),
        "selected_candidate_source_counts": dict(selected_candidate_source_counts.most_common()),
        "selected_candidate_template_counts": dict(selected_candidate_template_counts.most_common()),
        "verifier_status_counts": dict(verifier_status_counts.most_common()),
        "candidate_verifier_status_counts": dict(candidate_verifier_status_counts.most_common()),
        "mean_selected_score": _mean(selected_scores),
        "mean_selected_confidence": _mean(selected_confidences),
        "mean_score_margin": _mean(score_margins),
        "selected_score_1_count": selected_score_1_count,
        "selected_score_ge_0_99_count": selected_score_ge_0_99_count,
        "margin_lt_0_01_count": margin_lt_0_01_count,
        "margin_lt_0_05_count": margin_lt_0_05_count,
        "oversimplified_scalar_penalty_count": oversimplified_scalar_penalty_count,
        "mean_oversimplified_scalar_penalty": feature_means.get("oversimplified_scalar_penalty", 0.0),
        "mean_warning_penalty": feature_means.get("warning_penalty", 0.0),
        "mean_missing_input_penalty": feature_means.get("missing_input_penalty", 0.0),
        "legacy_selected_rate": _rate(legacy_selected_count, evaluated),
        "deterministic_variant_selected_rate": _rate(deterministic_selected_count, evaluated),
        "skeleton_selected_count": skeleton_selected_count,
        "top_error_types": dict(top_error_types.most_common()),
        "feature_means": feature_means,
        "failed_count": failed_count,
    }


def validate_candidate_verifier(
    input_path: Path,
    limit: int,
    output_path: Path,
    use_llm_fallback: bool = False,
    max_candidates: int = 8,
) -> dict[str, Any]:
    rows = _load_csv(input_path)
    text_column = _detect_text_column(rows)
    selected_rows = rows[: max(limit, 0)]

    output_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(selected_rows, start=1):
        problem_text = row.get(text_column, "")
        try:
            world_input, adapter_diagnostics = parse_and_adapt(
                problem_text,
                use_llm_fallback=use_llm_fallback,
                log_failures=False,
            )
            generation_result = generate_step_plan_candidates(world_input)
            generation_result = _apply_max_candidates(generation_result, max_candidates=max_candidates)
            verification_result = verify_step_plan_candidates(world_input, generation_result)

            verification_result_dict = verification_result.to_dict()
            output_rows.append(
                {
                    "row_index": row_index,
                    "problem_text": problem_text,
                    "adapter_diagnostics": adapter_diagnostics.to_dict(),
                    "candidate_generation_result": generation_result.to_dict(),
                    "candidate_verification_result": verification_result_dict,
                    "candidate_scoreboard": _candidate_scoreboard(verification_result_dict),
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
    parser = argparse.ArgumentParser(description="Validate Type2 candidate verifier.")
    parser.add_argument("--input", default="Physics_Problems_Text_Only.csv")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="type2_candidate_verifier_validation.json")
    parser.add_argument("--use-llm-fallback", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()

    result = validate_candidate_verifier(
        input_path=_resolve_input_path(args.input),
        limit=args.limit,
        output_path=Path(args.output).resolve(),
        use_llm_fallback=args.use_llm_fallback,
        max_candidates=args.max_candidates,
    )
    print(json.dumps({"evaluated": result["evaluated"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
