"""End-to-end deterministic Type2 candidate pipeline wrapper."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from parser.pipeline.type2_adapter import Type2WorldModelInput, diagnose_world_model_input, parse_and_adapt
from parser.pipeline.type2_candidate_generator import (
    Type2CandidateGenerationResult,
    Type2StepPlanCandidate,
    deduplicate_candidates,
    generate_step_plan_candidates,
)
from parser.pipeline.type2_candidate_ranker import Type2CandidateRankingResult, rank_verified_candidates
from parser.pipeline.type2_candidate_verifier import Type2CandidateVerification, Type2CandidateVerificationResult, verify_step_plan_candidates
from parser.pipeline.type2_answer_sanity_checker import Type2SanityCheckConfig, sanity_check_final_answer
from parser.pipeline.type2_numeric_executor import Type2NumericExecutionResult, execute_selected_step_plan


@dataclass
class Type2CandidatePipelineConfig:
    use_llm_fallback: bool = False
    log_failures: bool = False
    max_candidates: int = 8
    execute_numeric: bool = True
    include_intermediate_outputs: bool = True
    include_scoreboards: bool = True
    rank_confidence_margin_scale: float = 0.20
    low_confidence_threshold: float = 0.55
    min_rank_margin_for_high_confidence: float = 0.05
    enable_sanity_check: bool = True
    downgrade_critical_numeric: bool = True
    downgrade_high_risk_numeric: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Type2FinalAnswer:
    answer: str | None
    unit: str | None
    target: str | None
    numeric_value: float | None
    symbolic_expression: str | None
    answer_type: str
    confidence: float
    source_candidate_id: str | None
    source: str | None
    template_names: list[str]
    step_plan: list[dict[str, Any]]
    explanation: list[str]
    warnings: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Type2CandidatePipelineResult:
    problem_text: str
    final_answer: Type2FinalAnswer
    parser_status: str
    pipeline_status: str
    pipeline_warnings: list[str]
    pipeline_errors: list[dict[str, Any]]
    adapter_diagnostics: dict[str, Any]
    generation_summary: dict[str, Any]
    verification_summary: dict[str, Any]
    ranking_summary: dict[str, Any]
    selected_candidate: dict[str, Any] | None
    intermediate_outputs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_text": self.problem_text,
            "final_answer": self.final_answer.to_dict(),
            "parser_status": self.parser_status,
            "pipeline_status": self.pipeline_status,
            "pipeline_warnings": list(self.pipeline_warnings),
            "pipeline_errors": deepcopy(self.pipeline_errors),
            "adapter_diagnostics": deepcopy(self.adapter_diagnostics),
            "generation_summary": deepcopy(self.generation_summary),
            "verification_summary": deepcopy(self.verification_summary),
            "ranking_summary": deepcopy(self.ranking_summary),
            "selected_candidate": deepcopy(self.selected_candidate),
            "intermediate_outputs": deepcopy(self.intermediate_outputs),
        }


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _apply_max_candidates(
    generation_result: Type2CandidateGenerationResult,
    max_candidates: int,
) -> Type2CandidateGenerationResult:
    candidates = deduplicate_candidates(generation_result.candidates, max_candidates=max_candidates)
    selected_candidate_id = candidates[0].candidate_id if candidates else None
    summary = dict(generation_result.generation_summary)
    summary["candidate_count"] = len(candidates)
    summary["max_candidates"] = max_candidates
    return Type2CandidateGenerationResult(
        problem_text=generation_result.problem_text,
        target=generation_result.target,
        target_unit=generation_result.target_unit,
        candidates=candidates,
        selected_candidate_id=selected_candidate_id,
        generation_summary=summary,
    )


def _find_generation_candidate(
    generation_result: Type2CandidateGenerationResult,
    candidate_id: str | None,
) -> Type2StepPlanCandidate | None:
    return next((candidate for candidate in generation_result.candidates if candidate.candidate_id == candidate_id), None)


def _find_verified_candidate(
    verification_result: Type2CandidateVerificationResult,
    candidate_id: str | None,
) -> Type2CandidateVerification | None:
    return next((candidate for candidate in verification_result.verified_candidates if candidate.candidate_id == candidate_id), None)


def _rank_margin(ranking_result: Type2CandidateRankingResult) -> float:
    ranked = ranking_result.ranked_candidates
    if len(ranked) >= 2:
        return ranked[0].rank_score - ranked[1].rank_score
    if ranked:
        return ranked[0].rank_score
    return 0.0


def _extract_symbolic_expression(step_plan: list[dict[str, Any]], target: str | None) -> str | None:
    if not step_plan:
        return None
    if target:
        for step in reversed(step_plan):
            output_var = step.get("output_var") if isinstance(step, dict) else None
            if isinstance(output_var, dict) and target in output_var:
                value = output_var[target]
                return str(value) if value is not None else None
    for step in reversed(step_plan):
        if not isinstance(step, dict) or step.get("type") != "formula_application":
            continue
        output_var = step.get("output_var")
        if isinstance(output_var, dict) and output_var:
            return str(next(iter(output_var.values())))
    return None


def _extract_numeric_value(selected_candidate: Type2StepPlanCandidate | None) -> float | None:
    if not selected_candidate:
        return None
    metadata_value = selected_candidate.metadata.get("numeric_value") if isinstance(selected_candidate.metadata, dict) else None
    if isinstance(metadata_value, (int, float)):
        return float(metadata_value)
    for step in selected_candidate.step_plan:
        if not isinstance(step, dict):
            continue
        for key in ("numeric_value", "result", "value"):
            value = step.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _step_explanations(step_plan: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, step in enumerate(step_plan, start=1):
        if not isinstance(step, dict):
            continue
        goal = step.get("goal") or step.get("type") or "Process selected step."
        formula = step.get("formula_name")
        if formula:
            lines.append(f"Step {index}: {goal} using {formula}.")
        else:
            lines.append(f"Step {index}: {goal}.")
    return lines


def build_compact_scoreboard(
    generation_result: Type2CandidateGenerationResult,
    verification_result: Type2CandidateVerificationResult,
    ranking_result: Type2CandidateRankingResult,
) -> list[dict[str, Any]]:
    verifier_by_id = {candidate.candidate_id: candidate for candidate in verification_result.verified_candidates}
    ranked_by_id = {candidate.candidate_id: candidate for candidate in ranking_result.ranked_candidates}
    rows: list[dict[str, Any]] = []
    for candidate in generation_result.candidates:
        verified = verifier_by_id.get(candidate.candidate_id)
        ranked = ranked_by_id.get(candidate.candidate_id)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "source": candidate.source,
                "template_names": list(candidate.template_names),
                "verifier_score": verified.score if verified else None,
                "rank_score": ranked.rank_score if ranked else None,
                "verifier_status": verified.verifier_status if verified else None,
                "rank_adjustments": dict(ranked.rank_adjustments) if ranked else {},
                "selection_reasons": list(ranked.selection_reasons) if ranked else [],
                "rejection_reasons": list(ranked.rejection_reasons) if ranked else [],
            }
        )
    rows.sort(key=lambda row: (-(row["rank_score"] or 0.0), str(row["candidate_id"])))
    return rows


def build_final_answer(
    world_input: Type2WorldModelInput,
    generation_result: Type2CandidateGenerationResult,
    verification_result: Type2CandidateVerificationResult,
    ranking_result: Type2CandidateRankingResult,
    config: Type2CandidatePipelineConfig,
) -> Type2FinalAnswer:
    selected_ranked = ranking_result.ranked_candidates[0] if ranking_result.ranked_candidates else None
    selected_candidate_id = ranking_result.selected_candidate_id
    selected_generated = _find_generation_candidate(generation_result, selected_candidate_id)
    selected_verified = _find_verified_candidate(verification_result, selected_candidate_id)
    step_plan = deepcopy(selected_generated.step_plan) if selected_generated else []
    target = world_input.target
    unit = world_input.target_unit
    numeric_value = _extract_numeric_value(selected_generated)
    symbolic_expression = _extract_symbolic_expression(step_plan, target)
    execution_result: Type2NumericExecutionResult | None = None
    execution_explanations: list[str] = []

    if config.execute_numeric and selected_generated:
        execution_result = execute_selected_step_plan(world_input, step_plan, target, unit)
        if execution_result.numeric_value is not None:
            numeric_value = execution_result.numeric_value
            unit = execution_result.unit
        for index, executed_step in enumerate(execution_result.execution_trace, start=1):
            output_name = next(iter(executed_step.output_values.keys()), None)
            output_value = executed_step.numeric_value
            if executed_step.formula_name and output_name and output_value is not None:
                suffix = f" {executed_step.unit}" if executed_step.unit else ""
                execution_explanations.append(
                    f"Executed step {index}: {executed_step.formula_name} -> {output_name} = {output_value:.6g}{suffix}."
                )

    if numeric_value is not None:
        if execution_result and execution_result.answer:
            answer = execution_result.answer
        else:
            answer = f"{numeric_value} {unit}" if unit else str(numeric_value)
        answer_type = "numeric"
    elif symbolic_expression:
        answer = symbolic_expression
        answer_type = "symbolic_trace"
    elif step_plan:
        answer = target
        answer_type = "selected_step_plan"
    else:
        answer = target if target else None
        answer_type = "unresolved"

    rank_margin = _rank_margin(ranking_result)
    selected_rank_score = selected_ranked.rank_score if selected_ranked else 0.0
    base = _clip(selected_rank_score)
    margin_component = min(1.0, rank_margin / config.rank_confidence_margin_scale) if config.rank_confidence_margin_scale else 0.0
    confidence = 0.75 * base + 0.25 * margin_component
    verifier_status = selected_verified.verifier_status if selected_verified else ""
    if verifier_status == "WARN":
        confidence *= 0.85
    if verifier_status == "FAIL":
        confidence *= 0.60
    confidence = _clip(confidence)

    warnings: list[str] = []
    if rank_margin < config.min_rank_margin_for_high_confidence:
        warnings.append("Low rank margin; selected candidate is close to alternatives.")
    if selected_verified:
        warnings.extend(selected_verified.verifier_warnings)
    warnings.extend(world_input.parser_warnings)
    if execution_result and execution_result.numeric_value is None:
        warnings.extend(execution_result.warnings)
        warnings.append("Numeric execution attempted but could not resolve the final target.")
    if answer_type != "numeric" and not config.execute_numeric:
        warnings.append("Numeric execution is disabled.")
    elif answer_type != "numeric":
        warnings.append("Numeric execution did not produce a final numeric answer.")

    template_names = list(selected_ranked.template_names if selected_ranked else (selected_generated.template_names if selected_generated else []))
    explanation = [
        f"Parsed target: {target} with unit {unit}.",
        f"Generated {len(generation_result.candidates)} candidate step plans.",
    ]
    if selected_ranked:
        explanation.append(f"Selected candidate {selected_ranked.candidate_id} from {selected_ranked.source}.")
        explanation.append(f"Selected templates: {', '.join(template_names) if template_names else 'none'}.")
        explanation.extend(selected_ranked.selection_reasons)
    if execution_result and execution_result.numeric_value is not None:
        explanation.append("Numeric execution succeeded using the selected step plan.")
        explanation.extend(execution_explanations)
    else:
        explanation.extend(_step_explanations(step_plan))
    if config.execute_numeric and numeric_value is None:
        explanation.append("Numeric execution attempted but could not resolve the final target.")
    elif numeric_value is None:
        explanation.append("This stage selects the formula/step-plan trace; numeric execution is not yet enabled.")

    metadata = {
        "selected_rank_score": selected_rank_score,
        "selected_base_score": selected_ranked.base_score if selected_ranked else 0.0,
        "rank_margin": rank_margin,
        "verifier_status": verifier_status,
        "feature_values": deepcopy(selected_verified.feature_values) if selected_verified else {},
        "rank_adjustments": deepcopy(selected_ranked.rank_adjustments) if selected_ranked else {},
        "selected_formula_names": list(selected_verified.selected_formula_names) if selected_verified else [],
    }
    if execution_result:
        metadata["execution_result"] = execution_result.to_dict()

    return Type2FinalAnswer(
        answer=answer,
        unit=unit,
        target=target,
        numeric_value=numeric_value,
        symbolic_expression=symbolic_expression,
        answer_type=answer_type,
        confidence=confidence,
        source_candidate_id=selected_candidate_id,
        source=selected_ranked.source if selected_ranked else None,
        template_names=template_names,
        step_plan=step_plan,
        explanation=explanation,
        warnings=warnings,
        metadata=metadata,
    )


def _apply_sanity_check(
    final_answer: Type2FinalAnswer,
    world_input: Type2WorldModelInput,
    ranking_result: Type2CandidateRankingResult | None,
    config: Type2CandidatePipelineConfig,
) -> Type2FinalAnswer:
    if not config.enable_sanity_check or final_answer.answer_type != "numeric":
        return final_answer
    sanity_config = Type2SanityCheckConfig(
        enable_downgrade=True,
        downgrade_on_critical=config.downgrade_critical_numeric,
        downgrade_on_high_risk=config.downgrade_high_risk_numeric,
    )
    sanity_result = sanity_check_final_answer(final_answer, world_input, ranking_result, sanity_config)
    final_answer.metadata["sanity_check"] = sanity_result.to_dict()
    if sanity_result.status in {"WARN", "FAIL"}:
        for reason in sanity_result.reasons:
            final_answer.warnings.append(f"Sanity check: {reason}.")
    final_answer.confidence = _clip(final_answer.confidence * sanity_result.confidence_multiplier)
    if sanity_result.should_downgrade_to_symbolic:
        final_answer.metadata["downgraded_numeric_answer"] = {
            "answer": final_answer.answer,
            "numeric_value": final_answer.numeric_value,
            "unit": final_answer.unit,
        }
        final_answer.answer_type = "symbolic_trace"
        final_answer.numeric_value = None
        final_answer.answer = final_answer.symbolic_expression or final_answer.target
        final_answer.warnings.append("Numeric answer downgraded by sanity checker.")
    final_answer.warnings = list(dict.fromkeys(final_answer.warnings))
    return final_answer


def _pipeline_status(
    parser_status: str,
    final_answer: Type2FinalAnswer,
    ranking_result: Type2CandidateRankingResult | None,
    config: Type2CandidatePipelineConfig,
    exception_occurred: bool = False,
) -> str:
    if exception_occurred or not ranking_result or not ranking_result.selected_candidate_id:
        return "ERROR"
    if final_answer.answer_type == "unresolved":
        return "ERROR"
    execution_result = final_answer.metadata.get("execution_result", {})
    execution_status = execution_result.get("status") if isinstance(execution_result, dict) else None
    sanity_check = final_answer.metadata.get("sanity_check", {})
    sanity_status = sanity_check.get("status") if isinstance(sanity_check, dict) else None
    sanity_accepts = sanity_check.get("should_accept_numeric", True) if isinstance(sanity_check, dict) else True
    sanity_risk = sanity_check.get("risk_level") if isinstance(sanity_check, dict) else None
    if (
        parser_status == "PASS"
        and final_answer.answer_type == "numeric"
        and execution_status == "PASS"
        and sanity_status in {None, "PASS", "WARN"}
        and sanity_accepts
        and final_answer.confidence >= config.low_confidence_threshold
    ):
        return "WARN" if sanity_risk in {"MEDIUM", "HIGH"} else "OK"
    if parser_status != "PASS" or final_answer.confidence < config.low_confidence_threshold or final_answer.answer_type in {"symbolic_trace", "selected_step_plan"}:
        return "WARN"
    return "OK"


def _pipeline_warnings(
    world_input: Type2WorldModelInput | None,
    adapter_diagnostics: dict[str, Any],
    generation_result: Type2CandidateGenerationResult | None,
    ranking_result: Type2CandidateRankingResult | None,
    final_answer: Type2FinalAnswer,
    config: Type2CandidatePipelineConfig,
) -> list[str]:
    warnings: list[str] = []
    if not adapter_diagnostics.get("has_target", False):
        warnings.append("Parser did not detect target.")
    if generation_result and len(generation_result.candidates) == 1:
        warnings.append("Only one candidate was generated.")
    if ranking_result and _rank_margin(ranking_result) < config.min_rank_margin_for_high_confidence:
        warnings.append("Low rank margin.")
    if final_answer.answer_type != "numeric":
        warnings.append("Pipeline selected a trace but did not execute numeric computation.")
    execution_result = final_answer.metadata.get("execution_result", {})
    if isinstance(execution_result, dict) and execution_result.get("status") in {"WARN", "FAIL"}:
        warnings.append(f"Numeric execution status is {execution_result.get('status')}.")
    verifier_status = final_answer.metadata.get("verifier_status")
    if verifier_status in {"WARN", "FAIL"}:
        warnings.append(f"Selected verifier status is {verifier_status}.")
    if world_input:
        warnings.extend(world_input.parser_warnings)
    return list(dict.fromkeys(warnings + final_answer.warnings))


def _empty_final_answer(problem_text: str, error: str | None = None) -> Type2FinalAnswer:
    warnings = ["Pipeline failed before selecting a candidate."]
    if error:
        warnings.append(error)
    return Type2FinalAnswer(
        answer=None,
        unit=None,
        target=None,
        numeric_value=None,
        symbolic_expression=None,
        answer_type="unresolved",
        confidence=0.0,
        source_candidate_id=None,
        source=None,
        template_names=[],
        step_plan=[],
        explanation=[f"Unable to build Type2 candidate trace for: {problem_text}"],
        warnings=warnings,
        metadata={},
    )


def run_type2_candidate_pipeline(
    problem_text: str,
    config: Type2CandidatePipelineConfig | None = None,
) -> Type2CandidatePipelineResult:
    config = config or Type2CandidatePipelineConfig()
    intermediate_outputs: dict[str, Any] = {"config": config.to_dict()}
    world_input: Type2WorldModelInput | None = None
    adapter_diagnostics: dict[str, Any] = {}
    generation_result: Type2CandidateGenerationResult | None = None
    verification_result: Type2CandidateVerificationResult | None = None
    ranking_result: Type2CandidateRankingResult | None = None

    try:
        world_input, diagnostics = parse_and_adapt(
            problem_text,
            use_llm_fallback=config.use_llm_fallback,
            log_failures=config.log_failures,
        )
        adapter_diagnostics = diagnostics.to_dict()
        intermediate_outputs["world_input"] = world_input.to_dict()
        intermediate_outputs["adapter_diagnostics"] = adapter_diagnostics

        generation_result = generate_step_plan_candidates(world_input)
        if len(generation_result.candidates) > config.max_candidates:
            generation_result = _apply_max_candidates(generation_result, config.max_candidates)
        intermediate_outputs["candidate_generation_result"] = generation_result.to_dict()

        if not generation_result.candidates:
            raise ValueError("Candidate generation produced no candidates.")

        verification_result = verify_step_plan_candidates(world_input, generation_result)
        intermediate_outputs["candidate_verification_result"] = verification_result.to_dict()

        ranking_result = rank_verified_candidates(world_input, verification_result)
        intermediate_outputs["candidate_ranking_result"] = ranking_result.to_dict()

        final_answer = build_final_answer(world_input, generation_result, verification_result, ranking_result, config)
        final_answer = _apply_sanity_check(final_answer, world_input, ranking_result, config)
        if config.include_scoreboards:
            intermediate_outputs["compact_scoreboard"] = build_compact_scoreboard(generation_result, verification_result, ranking_result)
        if not config.include_intermediate_outputs:
            intermediate_outputs = {
                "config": config.to_dict(),
                "compact_scoreboard": intermediate_outputs.get("compact_scoreboard", []),
            }

        parser_status = world_input.parser_status
        pipeline_status = _pipeline_status(parser_status, final_answer, ranking_result, config)
        warnings = _pipeline_warnings(world_input, adapter_diagnostics, generation_result, ranking_result, final_answer, config)
        return Type2CandidatePipelineResult(
            problem_text=problem_text,
            final_answer=final_answer,
            parser_status=parser_status,
            pipeline_status=pipeline_status,
            pipeline_warnings=warnings,
            pipeline_errors=[],
            adapter_diagnostics=adapter_diagnostics,
            generation_summary=generation_result.generation_summary,
            verification_summary=verification_result.verification_summary,
            ranking_summary=ranking_result.ranking_summary,
            selected_candidate=ranking_result.selected_candidate,
            intermediate_outputs=intermediate_outputs,
        )
    except Exception as exc:
        error = {"error_type": "pipeline_exception", "message": repr(exc)}
        final_answer = _empty_final_answer(problem_text, repr(exc))
        if world_input:
            adapter_diagnostics = adapter_diagnostics or diagnose_world_model_input(world_input).to_dict()
        if generation_result:
            intermediate_outputs["candidate_generation_result"] = generation_result.to_dict()
        if verification_result:
            intermediate_outputs["candidate_verification_result"] = verification_result.to_dict()
        if ranking_result:
            intermediate_outputs["candidate_ranking_result"] = ranking_result.to_dict()
        return Type2CandidatePipelineResult(
            problem_text=problem_text,
            final_answer=final_answer,
            parser_status=world_input.parser_status if world_input else "ERROR",
            pipeline_status="ERROR",
            pipeline_warnings=list(final_answer.warnings),
            pipeline_errors=[error],
            adapter_diagnostics=adapter_diagnostics,
            generation_summary=generation_result.generation_summary if generation_result else {},
            verification_summary=verification_result.verification_summary if verification_result else {},
            ranking_summary=ranking_result.ranking_summary if ranking_result else {},
            selected_candidate=ranking_result.selected_candidate if ranking_result else None,
            intermediate_outputs=intermediate_outputs,
        )


def solve_type2_problem_candidate_pipeline(problem_text: str) -> dict[str, Any]:
    return run_type2_candidate_pipeline(problem_text).to_dict()
