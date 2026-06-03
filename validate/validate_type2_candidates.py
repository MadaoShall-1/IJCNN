"""Validate deterministic Type2 candidate generation over a CSV dataset."""

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


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _candidate_family(candidate: dict[str, Any]) -> str:
    names = [str(name).lower() for name in candidate.get("template_names", []) or []]
    source = str(candidate.get("source", "")).lower()
    joined = " ".join(names + [source])
    if "skeleton_placeholder" in names:
        return "skeleton"
    if "coulomb" in joined or "electric_field" in joined:
        return "coulomb"
    if "capacitor" in joined or "capacitance" in joined or "dielectric" in joined or "charge_definition" in joined or "voltage_definition" in joined:
        return "capacitor"
    if "ohms" in joined or "circuit" in joined or "power_" in joined or "resistance" in joined:
        return "circuit"
    if "acceleration" in joined or "speed" in joined or "kinematic" in joined or "velocity" in joined:
        return "kinematics"
    if "error" in joined or "uncertainty" in joined or "mean_value" in joined:
        return "measurement_error"
    if "force_" in joined or "resultant" in joined or "newton" in joined:
        return "force_resultant"
    if "non_numeric" in joined or "symbolic" in joined or "boolean" in joined:
        return "non_numeric"
    return "other"


def _world_labels(world_input: Any) -> str:
    labels = list(getattr(world_input, "domains", []) or []) + list(getattr(world_input, "sub_domains", []) or [])
    return " ".join(str(label).lower() for label in labels)


def _world_text(world_input: Any) -> str:
    return str(getattr(world_input, "problem_text", "") or "").lower()


def _world_target(world_input: Any) -> str:
    return str(getattr(world_input, "target", "") or "")


def _known_names(world_input: Any) -> set[str]:
    return {str(name) for name in (getattr(world_input, "known_quantities", {}) or {})}


def _has_charge_evidence(world_input: Any) -> bool:
    known = _known_names(world_input)
    text = _world_text(world_input)
    return bool({"q", "q1", "q2", "q3", "q0", "Q"} & known) or any(
        phrase in text for phrase in ("charge", "charges", "coulomb", "test charge")
    )


def _has_capacitor_evidence(world_input: Any) -> bool:
    known = _known_names(world_input)
    labels = _world_labels(world_input)
    text = _world_text(world_input)
    return (
        bool({"C_cap", "Q", "V"} & known)
        or any(keyword in labels for keyword in ("capacitor", "capacitance", "dielectric"))
        or any(keyword in text for keyword in ("capacitor", "capacitance", "dielectric"))
    )


def _has_circuit_evidence(world_input: Any) -> bool:
    known = _known_names(world_input)
    labels = _world_labels(world_input)
    text = _world_text(world_input)
    return (
        bool({"V", "I", "R", "P", "P_total"} & known)
        or any(keyword in labels for keyword in ("ohms_law", "circuit", "resistor"))
        or any(keyword in text for keyword in ("circuit", "resistor", "resistance", "ohm", "current", "voltage", "power"))
    )


def _is_electrostatic_or_capacitor_problem(world_input: Any) -> bool:
    labels = _world_labels(world_input)
    text = _world_text(world_input)
    return any(keyword in labels or keyword in text for keyword in ("electrostatic", "coulomb", "capacitor", "capacitance"))


def _is_pure_force_resultant_problem(world_input: Any) -> bool:
    labels = _world_labels(world_input)
    text = _world_text(world_input)
    target = _world_target(world_input)
    force_words = any(keyword in text for keyword in ("force", "forces", "resultant", "same direction", "opposite direction", "perpendicular", "angle"))
    electric_words = any(keyword in labels or keyword in text for keyword in ("coulomb", "electrostatic", "charge", "test charge"))
    return (target in {"F_net", "theta"} or force_words) and not electric_words and not _has_charge_evidence(world_input)


def _candidate_quality_diagnostics(world_input: Any, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    suspected_cross_domain: list[dict[str, Any]] = []

    for candidate in candidates:
        family = _candidate_family(candidate)
        family_counts[family] += 1
        reason = None
        if family == "kinematics" and _is_electrostatic_or_capacitor_problem(world_input):
            reason = "kinematics_candidate_in_electrostatics_or_capacitor_problem"
        elif family == "coulomb" and _is_pure_force_resultant_problem(world_input):
            reason = "coulomb_candidate_in_pure_force_resultant_problem"
        elif family == "capacitor" and not _has_capacitor_evidence(world_input):
            reason = "capacitor_candidate_without_capacitor_evidence"
        elif family == "circuit" and not _has_circuit_evidence(world_input):
            reason = "circuit_candidate_without_circuit_evidence"

        if reason:
            suspected_cross_domain.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "source": candidate.get("source"),
                    "template_names": candidate.get("template_names", []),
                    "family": family,
                    "reason": reason,
                }
            )

    return {
        "candidate_family_counts": dict(family_counts.most_common()),
        "suspected_cross_domain_candidates": suspected_cross_domain,
    }


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = len(rows)
    failed_count = sum(1 for row in rows if "error" in row)
    successful_rows = [row for row in rows if "error" not in row]

    candidate_counts: list[int] = []
    legacy_present = 0
    skeleton_legacy = 0
    candidate_source_counts: Counter[str] = Counter()
    candidate_template_counts: Counter[str] = Counter()
    selected_candidate_source_counts: Counter[str] = Counter()
    selected_candidate_template_counts: Counter[str] = Counter()
    candidate_family_counts: Counter[str] = Counter()
    cross_domain_candidate_count = 0
    rows_with_cross_domain_candidates = 0

    for row in successful_rows:
        result = row.get("candidate_generation_result", {})
        candidates = result.get("candidates", []) if isinstance(result, dict) else []
        selected_candidate_id = result.get("selected_candidate_id") if isinstance(result, dict) else None
        candidate_counts.append(len(candidates))
        diagnostics = row.get("candidate_quality_diagnostics", {})
        row_cross_domain = diagnostics.get("suspected_cross_domain_candidates", []) if isinstance(diagnostics, dict) else []
        cross_domain_candidate_count += len(row_cross_domain)
        if row_cross_domain:
            rows_with_cross_domain_candidates += 1
        family_counts = diagnostics.get("candidate_family_counts", {}) if isinstance(diagnostics, dict) else {}
        candidate_family_counts.update({str(key): int(value) for key, value in family_counts.items()})

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            source = str(candidate.get("source", ""))
            candidate_source_counts[source] += 1
            for template_name in candidate.get("template_names", []) or []:
                candidate_template_counts[str(template_name)] += 1
            if source == "legacy_parser_step_plan":
                legacy_present += 1
                metadata = candidate.get("metadata", {})
                if isinstance(metadata, dict) and metadata.get("used_skeleton_fallback") is True:
                    skeleton_legacy += 1
            if candidate.get("candidate_id") == selected_candidate_id:
                selected_candidate_source_counts[source] += 1
                for template_name in candidate.get("template_names", []) or []:
                    selected_candidate_template_counts[str(template_name)] += 1

    single_candidate_count = sum(1 for count in candidate_counts if count == 1)
    multi_candidate_count = sum(1 for count in candidate_counts if count > 1)

    return {
        "mean_candidate_count": _mean(candidate_counts),
        "min_candidate_count": min(candidate_counts) if candidate_counts else 0,
        "max_candidate_count": max(candidate_counts) if candidate_counts else 0,
        "single_candidate_rate": _rate(single_candidate_count, evaluated),
        "multi_candidate_rate": _rate(multi_candidate_count, evaluated),
        "legacy_candidate_present_rate": _rate(legacy_present, evaluated),
        "skeleton_legacy_rate": _rate(skeleton_legacy, evaluated),
        "candidate_source_counts": dict(candidate_source_counts.most_common()),
        "candidate_template_counts": dict(candidate_template_counts.most_common()),
        "selected_candidate_source_counts": dict(selected_candidate_source_counts.most_common()),
        "selected_candidate_template_counts": dict(selected_candidate_template_counts.most_common()),
        "cross_domain_candidate_count": cross_domain_candidate_count,
        "cross_domain_candidate_rate": _rate(cross_domain_candidate_count, sum(candidate_counts)),
        "rows_with_cross_domain_candidates": rows_with_cross_domain_candidates,
        "candidate_family_counts": dict(candidate_family_counts.most_common()),
        "failed_count": failed_count,
    }


def validate_candidates(
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
            candidate_result = generate_step_plan_candidates(world_input)
            candidate_result = _apply_max_candidates(candidate_result, max_candidates=max_candidates)
            candidate_result_dict = candidate_result.to_dict()
            quality_diagnostics = _candidate_quality_diagnostics(
                world_input,
                candidate_result_dict["candidates"],
            )
            output_rows.append(
                {
                    "row_index": row_index,
                    "problem_text": problem_text,
                    "adapter_diagnostics": adapter_diagnostics.to_dict(),
                    "candidate_generation_result": candidate_result_dict,
                    "candidate_quality_diagnostics": quality_diagnostics,
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
    parser = argparse.ArgumentParser(description="Validate Type2 deterministic candidate generation.")
    parser.add_argument("--input", default="Physics_Problems_Text_Only.csv")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="type2_candidate_validation.json")
    parser.add_argument("--use-llm-fallback", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()

    result = validate_candidates(
        input_path=_resolve_input_path(args.input),
        limit=args.limit,
        output_path=Path(args.output).resolve(),
        use_llm_fallback=args.use_llm_fallback,
        max_candidates=args.max_candidates,
    )
    print(json.dumps({"evaluated": result["evaluated"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
