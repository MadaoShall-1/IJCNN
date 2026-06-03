"""Validate Type2 rule-based candidate ranking over a CSV dataset."""

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
from parser.pipeline.type2_candidate_ranker import rank_verified_candidates
from parser.pipeline.type2_candidate_verifier import verify_step_plan_candidates


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


def _rank_margin(ranked: list[dict[str, Any]]) -> float:
    if len(ranked) >= 2:
        return float(ranked[0].get("rank_score", 0.0)) - float(ranked[1].get("rank_score", 0.0))
    if ranked:
        return float(ranked[0].get("rank_score", 0.0))
    return 0.0


def _problem_has_explicit_geometry(problem_text: str) -> bool:
    text = problem_text.lower()
    return any(term in text for term in ("right_angle", "right-angled", "right angled", "perpendicular", "90", "90°", "equilateral", "60", "60°", "collinear", "straight line", "opposite sides", "opposite direction", "angle"))


def _is_geometry_specific(candidate: dict[str, Any]) -> bool:
    templates = " ".join(str(name).lower() for name in candidate.get("template_names", []) or [])
    return any(term in templates for term in ("right_angle", "equilateral", "collinear", "angle_resultant", "collinear_opposite", "collinear_difference"))


def _is_generic_vector(candidate: dict[str, Any]) -> bool:
    templates = " ".join(str(name).lower() for name in candidate.get("template_names", []) or [])
    return "vector_sum" in templates or "pairwise_vector_sum" in templates


def _has_geometry_specific_candidate(ranked: list[dict[str, Any]]) -> bool:
    return any(_is_geometry_specific(candidate) for candidate in ranked)


def _ranking_scoreboard(ranking_result_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate.get("candidate_id"),
            "source": candidate.get("source"),
            "template_names": candidate.get("template_names", []),
            "base_score": candidate.get("base_score"),
            "rank_score": candidate.get("rank_score"),
            "rank_adjustments": candidate.get("rank_adjustments", {}),
            "selection_reasons": candidate.get("selection_reasons", []),
            "rejection_reasons": candidate.get("rejection_reasons", []),
        }
        for candidate in ranking_result_dict.get("ranked_candidates", [])
    ]


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = len(rows)
    failed_count = sum(1 for row in rows if "error" in row)
    successful_rows = [row for row in rows if "error" not in row]

    candidate_counts: list[float] = []
    verified_counts: list[float] = []
    rank_scores: list[float] = []
    rank_margins: list[float] = []
    selected_candidate_source_counts: Counter[str] = Counter()
    selected_candidate_template_counts: Counter[str] = Counter()
    ranker_selected_source_counts: Counter[str] = Counter()
    ranker_selected_template_counts: Counter[str] = Counter()
    top_rank_adjustments: Counter[str] = Counter()
    top_rejection_reasons: Counter[str] = Counter()
    legacy_selected = 0
    deterministic_selected = 0
    rank_margin_lt_0_01_count = 0
    rank_margin_lt_0_05_count = 0
    geometry_specific_selected_count = 0
    generic_vector_selected_in_geometry_count = 0
    oversimplified_scalar_selected_count = 0
    skeleton_selected_count = 0

    for row in successful_rows:
        generation = row.get("candidate_generation_result", {})
        verification = row.get("candidate_verification_result", {})
        ranking = row.get("candidate_ranking_result", {})
        generated = generation.get("candidates", []) if isinstance(generation, dict) else []
        verified = verification.get("verified_candidates", []) if isinstance(verification, dict) else []
        ranked = ranking.get("ranked_candidates", []) if isinstance(ranking, dict) else []
        candidate_counts.append(float(len(generated)))
        verified_counts.append(float(len(verified)))

        verifier_selected_id = verification.get("selected_candidate_id") if isinstance(verification, dict) else None
        for candidate in verified:
            if candidate.get("candidate_id") != verifier_selected_id:
                continue
            selected_candidate_source_counts[str(candidate.get("source", ""))] += 1
            for template_name in candidate.get("template_names", []) or []:
                selected_candidate_template_counts[str(template_name)] += 1

        selected = ranking.get("selected_candidate") if isinstance(ranking, dict) else None
        margin = _rank_margin(ranked)
        rank_margins.append(margin)
        if margin < 0.01:
            rank_margin_lt_0_01_count += 1
        if margin < 0.05:
            rank_margin_lt_0_05_count += 1

        for candidate in ranked:
            for key, value in (candidate.get("rank_adjustments", {}) or {}).items():
                if value:
                    top_rank_adjustments[str(key)] += 1
            for reason in candidate.get("rejection_reasons", []) or []:
                top_rejection_reasons[str(reason)] += 1

        if not selected:
            continue
        source = str(selected.get("source", ""))
        ranker_selected_source_counts[source] += 1
        if source == "legacy_parser_step_plan":
            legacy_selected += 1
        if source == "deterministic_variant":
            deterministic_selected += 1
        rank_scores.append(float(selected.get("rank_score", 0.0)))
        templates = selected.get("template_names", []) or []
        for template_name in templates:
            ranker_selected_template_counts[str(template_name)] += 1
        if any(str(template_name) == "skeleton_placeholder" for template_name in templates):
            skeleton_selected_count += 1
        if float((selected.get("feature_values", {}) or {}).get("oversimplified_scalar_penalty", 0.0)) >= 0.5:
            oversimplified_scalar_selected_count += 1
        explicit_geometry = _problem_has_explicit_geometry(str(row.get("problem_text", "")))
        if explicit_geometry and _is_geometry_specific(selected):
            geometry_specific_selected_count += 1
        if explicit_geometry and _is_generic_vector(selected) and _has_geometry_specific_candidate(ranked):
            generic_vector_selected_in_geometry_count += 1

    return {
        "mean_candidate_count": _mean(candidate_counts),
        "mean_verified_candidate_count": _mean(verified_counts),
        "selected_candidate_source_counts": dict(selected_candidate_source_counts.most_common()),
        "selected_candidate_template_counts": dict(selected_candidate_template_counts.most_common()),
        "ranker_selected_source_counts": dict(ranker_selected_source_counts.most_common()),
        "ranker_selected_template_counts": dict(ranker_selected_template_counts.most_common()),
        "legacy_selected_rate": _rate(legacy_selected, evaluated),
        "deterministic_variant_selected_rate": _rate(deterministic_selected, evaluated),
        "mean_rank_score": _mean(rank_scores),
        "mean_rank_margin": _mean(rank_margins),
        "rank_margin_lt_0_01_count": rank_margin_lt_0_01_count,
        "rank_margin_lt_0_05_count": rank_margin_lt_0_05_count,
        "geometry_specific_selected_count": geometry_specific_selected_count,
        "generic_vector_selected_in_geometry_count": generic_vector_selected_in_geometry_count,
        "oversimplified_scalar_selected_count": oversimplified_scalar_selected_count,
        "skeleton_selected_count": skeleton_selected_count,
        "top_rank_adjustments": dict(top_rank_adjustments.most_common()),
        "top_rejection_reasons": dict(top_rejection_reasons.most_common()),
        "failed_count": failed_count,
    }


def validate_candidate_ranker(
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
            ranking_result = rank_verified_candidates(world_input, verification_result)
            ranking_result_dict = ranking_result.to_dict()

            output_rows.append(
                {
                    "row_index": row_index,
                    "problem_text": problem_text,
                    "adapter_diagnostics": adapter_diagnostics.to_dict(),
                    "candidate_generation_result": generation_result.to_dict(),
                    "candidate_verification_result": verification_result.to_dict(),
                    "candidate_ranking_result": ranking_result_dict,
                    "ranking_scoreboard": _ranking_scoreboard(ranking_result_dict),
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
    parser = argparse.ArgumentParser(description="Validate Type2 rule-based candidate ranker.")
    parser.add_argument("--input", default="Physics_Problems_Text_Only.csv")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="type2_candidate_ranker_validation.json")
    parser.add_argument("--use-llm-fallback", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()

    result = validate_candidate_ranker(
        input_path=_resolve_input_path(args.input),
        limit=args.limit,
        output_path=Path(args.output).resolve(),
        use_llm_fallback=args.use_llm_fallback,
        max_candidates=args.max_candidates,
    )
    print(json.dumps({"evaluated": result["evaluated"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
