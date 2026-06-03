"""Rule-based final ranking for verified Type2 step-plan candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from parser.pipeline.type2_adapter import Type2WorldModelInput
from parser.pipeline.type2_candidate_verifier import Type2CandidateVerification, Type2CandidateVerificationResult


@dataclass
class Type2RankedCandidate:
    candidate_id: str
    source: str
    template_names: list[str]
    base_score: float
    rank_score: float
    rank_adjustments: dict[str, float]
    selection_reasons: list[str]
    rejection_reasons: list[str]
    verifier_status: str
    feature_values: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Type2CandidateRankingResult:
    problem_text: str
    target: str | None
    target_unit: str | None
    ranked_candidates: list[Type2RankedCandidate]
    selected_candidate_id: str | None
    selected_candidate: dict[str, Any] | None
    ranking_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_text": self.problem_text,
            "target": self.target,
            "target_unit": self.target_unit,
            "ranked_candidates": [candidate.to_dict() for candidate in self.ranked_candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate": self.selected_candidate,
            "ranking_summary": dict(self.ranking_summary),
        }


def _clip_score(value: float) -> float:
    return max(0.0, min(1.25, float(value)))


def _text(world_input: Type2WorldModelInput) -> str:
    return str(world_input.problem_text or "").lower()


def _conditions(world_input: Type2WorldModelInput) -> str:
    return " ".join(str(item).lower() for item in world_input.conditions + world_input.domains + world_input.sub_domains)


def _problem_blob(world_input: Type2WorldModelInput) -> str:
    return _text(world_input) + " " + _conditions(world_input)


def _problem_has_right_angle(world_input: Type2WorldModelInput) -> bool:
    blob = _problem_blob(world_input)
    return any(term in blob for term in ("right_angle", "right-angled", "right angled", "perpendicular", "90", "90°"))


def _problem_has_equilateral(world_input: Type2WorldModelInput) -> bool:
    blob = _problem_blob(world_input)
    return "equilateral" in blob or "60" in blob or "60°" in blob


def _problem_has_collinear(world_input: Type2WorldModelInput) -> bool:
    blob = _problem_blob(world_input)
    return any(term in blob for term in ("collinear", "straight line", "opposite sides", "opposite direction", "opposite directions"))


def _problem_has_angle(world_input: Type2WorldModelInput) -> bool:
    blob = _problem_blob(world_input)
    return "theta" in world_input.known_quantities or any(term in blob for term in ("angle", "degrees", "°", "perpendicular", "60", "90"))


def _problem_has_perpendicular_bisector(world_input: Type2WorldModelInput) -> bool:
    return "perpendicular bisector" in _problem_blob(world_input)


def _has_explicit_geometry(world_input: Type2WorldModelInput) -> bool:
    return (
        _problem_has_right_angle(world_input)
        or _problem_has_equilateral(world_input)
        or _problem_has_collinear(world_input)
        or _problem_has_perpendicular_bisector(world_input)
        or _problem_has_angle(world_input)
    )


def _candidate_has_template(candidate: Type2CandidateVerification | Type2RankedCandidate, *template_keywords: str) -> bool:
    templates = " ".join(str(name).lower() for name in candidate.template_names)
    formulas = " ".join(str(name).lower() for name in getattr(candidate, "selected_formula_names", []) or [])
    blob = templates + " " + formulas
    return any(keyword.lower() in blob for keyword in template_keywords)


def _normalize_formula_name(name: str) -> str:
    return " ".join(str(name or "").lower().replace(" ", "").split())


def _candidate_formula_sequence(candidate: Type2CandidateVerification) -> tuple[str, ...]:
    return tuple(_normalize_formula_name(name) for name in candidate.selected_formula_names)


def _status_priority(status: str) -> int:
    return {"PASS": 2, "WARN": 1, "FAIL": 0}.get(str(status).upper(), 0)


def _is_geometry_specific(candidate: Type2CandidateVerification | Type2RankedCandidate) -> bool:
    return _candidate_has_template(
        candidate,
        "right_angle",
        "equilateral",
        "collinear",
        "angle_resultant",
        "collinear_opposite",
        "collinear_difference",
    )


def _is_generic_vector(candidate: Type2CandidateVerification | Type2RankedCandidate) -> bool:
    return _candidate_has_template(candidate, "vector_sum", "pairwise_vector_sum")


def _add_adjustment(adjustments: dict[str, float], key: str, value: float) -> None:
    if value:
        adjustments[key] = adjustments.get(key, 0.0) + value


def _rank_adjustments(
    world_input: Type2WorldModelInput,
    candidate: Type2CandidateVerification,
) -> tuple[dict[str, float], list[str], list[str]]:
    feature_values = dict(candidate.feature_values)
    adjustments: dict[str, float] = {}
    selection_reasons: list[str] = []
    rejection_reasons: list[str] = []
    explicit_geometry = _has_explicit_geometry(world_input)

    if _problem_has_right_angle(world_input):
        if _candidate_has_template(candidate, "coulomb_right_angle_resultant", "force_right_angle_resultant"):
            _add_adjustment(adjustments, "geometry_specific_boost", 0.030)
            selection_reasons.append("Geometry-specific right-angle candidate matches explicit problem condition.")
        elif _is_generic_vector(candidate):
            _add_adjustment(adjustments, "generic_vector_explicit_geometry_boost", 0.005)
    if _problem_has_equilateral(world_input) and _candidate_has_template(candidate, "coulomb_equilateral_resultant"):
        _add_adjustment(adjustments, "geometry_specific_boost", 0.030)
        selection_reasons.append("Equilateral candidate matches explicit geometry.")
    if _problem_has_angle(world_input) and _candidate_has_template(candidate, "force_angle_resultant"):
        _add_adjustment(adjustments, "geometry_specific_boost", 0.025)
        selection_reasons.append("Angle-resultant candidate matches angle evidence.")
    if _problem_has_collinear(world_input):
        if _candidate_has_template(candidate, "coulomb_collinear_opposite"):
            _add_adjustment(adjustments, "geometry_specific_boost", 0.030)
            selection_reasons.append("Collinear Coulomb candidate matches explicit geometry.")
        if _candidate_has_template(candidate, "force_collinear_difference"):
            _add_adjustment(adjustments, "geometry_specific_boost", 0.025)
            selection_reasons.append("Force-difference candidate matches opposite-direction evidence.")

    if explicit_geometry and _is_generic_vector(candidate):
        _add_adjustment(adjustments, "generic_vector_fallback_penalty", -0.015)
        rejection_reasons.append("Generic vector fallback is less specific than explicit geometry.")

    oversimplified = float(feature_values.get("oversimplified_scalar_penalty", 0.0))
    if oversimplified >= 0.8:
        _add_adjustment(adjustments, "oversimplified_scalar_penalty", -0.150)
        rejection_reasons.append("Scalar Coulomb candidate is too simple for multi-charge/vector-force problem.")
    elif oversimplified >= 0.5:
        _add_adjustment(adjustments, "oversimplified_scalar_penalty", -0.080)
        rejection_reasons.append("Scalar Coulomb candidate may be too simple for this force problem.")

    missing = float(feature_values.get("missing_input_penalty", 0.0))
    if missing >= 0.60:
        _add_adjustment(adjustments, "missing_input_penalty_amplified", -0.120)
        rejection_reasons.append("Candidate has substantial missing-input risk.")
    elif missing >= 0.30:
        _add_adjustment(adjustments, "missing_input_penalty_amplified", -0.060)
        rejection_reasons.append("Candidate has moderate missing-input risk.")

    invalid_output = float(feature_values.get("invalid_output_penalty", 0.0))
    if invalid_output >= 0.80:
        _add_adjustment(adjustments, "invalid_output_penalty_amplified", -0.180)
        rejection_reasons.append("Candidate output conflicts with requested target.")
    elif invalid_output >= 0.50:
        _add_adjustment(adjustments, "invalid_output_penalty_amplified", -0.100)
        rejection_reasons.append("Candidate output alignment is weak.")

    status = str(candidate.verifier_status).upper()
    if status == "PASS":
        _add_adjustment(adjustments, "verifier_status_adjustment", 0.010)
        selection_reasons.append("Verifier status PASS.")
    elif status == "WARN":
        _add_adjustment(adjustments, "verifier_status_adjustment", -0.030)
        rejection_reasons.append("Verifier status WARN.")
    else:
        _add_adjustment(adjustments, "verifier_status_adjustment", -0.120)
        rejection_reasons.append("Verifier status FAIL.")

    clean = (
        feature_values.get("warning_penalty", 0.0) == 0
        and feature_values.get("missing_input_penalty", 0.0) == 0
        and feature_values.get("invalid_output_penalty", 0.0) == 0
        and feature_values.get("uses_skeleton_penalty", 0.0) == 0
        and feature_values.get("formula_target_alignment", 0.0) >= 0.95
        and feature_values.get("input_availability_score", 0.0) >= 0.95
    )
    if clean:
        _add_adjustment(adjustments, "clean_candidate_boost", 0.015)
        selection_reasons.append("Candidate has clean verifier profile.")

    if candidate.source == "llm_fallback_canonicalized" and clean and status == "PASS":
        _add_adjustment(adjustments, "llm_canonicalized_clean_boost", 0.015)
        selection_reasons.append("Canonicalized LLM candidate has clean PASS profile.")

    if (
        candidate.source == "legacy_parser_step_plan"
        and feature_values.get("parser_status_score", 0.0) == 1.0
        and feature_values.get("warning_penalty", 0.0) == 0
        and feature_values.get("missing_input_penalty", 0.0) == 0
        and feature_values.get("formula_target_alignment", 0.0) >= 0.95
    ):
        _add_adjustment(adjustments, "legacy_stable_trace_bonus", 0.010)
        selection_reasons.append("Legacy candidate preserves stable PASS parser trace.")

    if feature_values.get("uses_skeleton_penalty", 0.0) > 0:
        _add_adjustment(adjustments, "skeleton_hard_rejection", -0.300)
        rejection_reasons.append("Skeleton placeholder candidate is rejected for final ranking.")

    return adjustments, selection_reasons, rejection_reasons


def _apply_equivalence_reasons(ranked: list[Type2RankedCandidate], verified: list[Type2CandidateVerification]) -> None:
    by_id = {candidate.candidate_id: candidate for candidate in verified}
    grouped: dict[tuple[str, ...], list[Type2RankedCandidate]] = {}
    for candidate in ranked:
        verified_candidate = by_id.get(candidate.candidate_id)
        if not verified_candidate:
            continue
        formula_sequence = _candidate_formula_sequence(verified_candidate)
        if formula_sequence:
            grouped.setdefault(formula_sequence, []).append(candidate)

    for equivalents in grouped.values():
        if len(equivalents) < 2:
            continue
        legacy_clean = [
            candidate
            for candidate in equivalents
            if candidate.source == "legacy_parser_step_plan"
            and candidate.verifier_status == "PASS"
            and candidate.feature_values.get("warning_penalty", 0.0) == 0
        ]
        if legacy_clean:
            best_legacy = sorted(legacy_clean, key=lambda candidate: (-candidate.rank_score, candidate.candidate_id))[0]
            best_legacy.selection_reasons.append("Equivalent formula path; legacy candidate preferred for stable parser trace.")
        else:
            best = sorted(equivalents, key=lambda candidate: (-candidate.rank_score, candidate.candidate_id))[0]
            best.selection_reasons.append("Equivalent formula path; deterministic candidate preferred due to cleaner feature profile.")


def _apply_geometry_near_tie_bonus(world_input: Type2WorldModelInput, ranked: list[Type2RankedCandidate]) -> None:
    if not _has_explicit_geometry(world_input):
        return
    legacy_scores = [
        candidate.rank_score
        for candidate in ranked
        if candidate.source == "legacy_parser_step_plan"
    ]
    if not legacy_scores:
        return
    best_legacy_score = max(legacy_scores)
    for candidate in ranked:
        if candidate.source != "deterministic_variant" or not _is_geometry_specific(candidate):
            continue
        if candidate.rank_score >= best_legacy_score - 0.02:
            candidate.rank_adjustments["deterministic_geometry_near_tie_bonus"] = (
                candidate.rank_adjustments.get("deterministic_geometry_near_tie_bonus", 0.0) + 0.015
            )
            candidate.rank_score = _clip_score(candidate.rank_score + 0.015)
            candidate.selection_reasons.append("Deterministic geometry-specific candidate wins near-tie with legacy trace.")


def _summary(ranked: list[Type2RankedCandidate]) -> dict[str, Any]:
    if not ranked:
        return {"ranked_candidate_count": 0, "selected_rank_score": 0.0, "rank_margin": 0.0}
    margin = ranked[0].rank_score - ranked[1].rank_score if len(ranked) > 1 else ranked[0].rank_score
    return {
        "ranked_candidate_count": len(ranked),
        "selected_rank_score": ranked[0].rank_score,
        "rank_margin": margin,
        "selected_source": ranked[0].source,
        "selected_templates": ranked[0].template_names,
    }


def rank_verified_candidates(
    world_input: Type2WorldModelInput,
    verification_result: Type2CandidateVerificationResult,
) -> Type2CandidateRankingResult:
    """Apply deterministic final ranking adjustments to verified candidates."""
    ranked: list[Type2RankedCandidate] = []
    for candidate in verification_result.verified_candidates:
        adjustments, selection_reasons, rejection_reasons = _rank_adjustments(world_input, candidate)
        base_score = float(candidate.score)
        rank_score = _clip_score(base_score + sum(adjustments.values()))
        ranked.append(
            Type2RankedCandidate(
                candidate_id=candidate.candidate_id,
                source=candidate.source,
                template_names=list(candidate.template_names),
                base_score=base_score,
                rank_score=rank_score,
                rank_adjustments=adjustments,
                selection_reasons=selection_reasons,
                rejection_reasons=rejection_reasons,
                verifier_status=("FAIL" if candidate.feature_values.get("uses_skeleton_penalty", 0.0) > 0 else candidate.verifier_status),
                feature_values=dict(candidate.feature_values),
                metadata={
                    "selected_formula_names": list(candidate.selected_formula_names),
                    "verifier_metadata": dict(candidate.metadata),
                },
            )
        )

    _apply_equivalence_reasons(ranked, verification_result.verified_candidates)
    _apply_geometry_near_tie_bonus(world_input, ranked)
    ranked.sort(
        key=lambda candidate: (
            -candidate.rank_score,
            -candidate.base_score,
            -_status_priority(candidate.verifier_status),
            candidate.candidate_id,
        )
    )
    selected = ranked[0].to_dict() if ranked else None
    return Type2CandidateRankingResult(
        problem_text=verification_result.problem_text,
        target=verification_result.target,
        target_unit=verification_result.target_unit,
        ranked_candidates=ranked,
        selected_candidate_id=ranked[0].candidate_id if ranked else None,
        selected_candidate=selected,
        ranking_summary=_summary(ranked),
    )