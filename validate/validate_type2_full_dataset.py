"""Run full-dataset validation for the deterministic Type2 candidate pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parser.pipeline.type2_candidate_pipeline import Type2CandidatePipelineConfig, run_type2_candidate_pipeline


TEXT_COLUMN_CANDIDATES = ("problem_text", "question", "text")
SAMPLE_GROUPS = (
    "errors",
    "symbolic_trace",
    "target_unresolved",
    "missing_inputs",
    "risky_numeric",
    "parser_fail",
    "low_confidence",
)
VECTOR_EXECUTED_MODES = {"right_angle", "equilateral", "collinear_opposite", "same_direction", "law_of_cosines"}
VECTOR_SYMBOLIC_MODES = {"symbolic_ambiguous", "symbolic_missing_pair_forces"}


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(counter.most_common())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _execution_result(final_answer: dict[str, Any]) -> dict[str, Any]:
    metadata = _as_dict(final_answer.get("metadata"))
    return _as_dict(metadata.get("execution_result"))


def _execution_metadata(execution_result: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(execution_result.get("metadata"))


def _sanity_meta(final_answer: dict[str, Any]) -> dict[str, Any]:
    sanity_check = _as_dict(_as_dict(final_answer.get("metadata")).get("sanity_check"))
    return _as_dict(sanity_check.get("metadata"))


def _canon_info(final_answer: dict[str, Any]) -> tuple[bool, str | None, list[Any], list[str], list[str]]:
    """Return (used, status, mappings, original_llm_formulas, canonical_formulas)
    for the selected candidate when it came from LLM-fallback canonicalization."""
    meta = _as_dict(final_answer.get("metadata"))
    canon = _as_dict(meta.get("canonicalization_result"))
    if not canon:
        source = str(final_answer.get("source") or "")
        if source == "llm_fallback_canonicalized":
            return True, "PASS", [], [], _as_list(final_answer.get("template_names"))
        return False, None, [], [], []
    return (
        True,
        canon.get("status"),
        _as_list(canon.get("mapping_log")),
        _as_list(canon.get("original_formula_names")),
        _as_list(canon.get("canonical_formula_names")),
    )


def _rank_margin(result: dict[str, Any], final_answer: dict[str, Any]) -> float:
    ranking_summary = _as_dict(result.get("ranking_summary"))
    metadata = _as_dict(final_answer.get("metadata"))
    value = ranking_summary.get("rank_margin", metadata.get("rank_margin", 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _template_names(final_answer: dict[str, Any]) -> list[str]:
    names = [str(name) for name in _as_list(final_answer.get("template_names"))]
    return names or ["unknown"]


def _execution_error_types(execution_result: dict[str, Any]) -> list[str]:
    error_types: list[str] = []
    for error in _as_list(execution_result.get("errors")):
        if isinstance(error, dict):
            error_types.append(str(error.get("error_type", "unknown")))
    return error_types


def _is_extreme_numeric(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    number = float(value)
    return number == 0.0 or not math.isfinite(number) or abs(number) > 1e12


def _risky_numeric_reasons(
    row: dict[str, Any],
    final_answer: dict[str, Any],
    execution_result: dict[str, Any],
    execution_metadata: dict[str, Any],
    rank_margin: float,
    config: Type2CandidatePipelineConfig,
) -> list[str]:
    if final_answer.get("answer_type") != "numeric":
        return []
    reasons: list[str] = []
    final_warnings = [str(item) for item in _as_list(final_answer.get("warnings"))]
    execution_warnings = [str(item) for item in _as_list(execution_result.get("warnings"))]
    all_warnings = final_warnings + execution_warnings + [str(item) for item in _as_list(row.get("pipeline_warnings"))]
    if any("Using geometric line angle for force magnitude; charge signs may affect direction." in warning for warning in all_warnings):
        reasons.append("geometric_line_angle_warning")
    if execution_metadata.get("law_of_cosines_used") and execution_metadata.get("charge_interaction_adjustment"):
        reasons.append("law_of_cosines_charge_adjustment")
    verifier_status = _as_dict(final_answer.get("metadata")).get("verifier_status")
    selected_candidate = _as_dict(row.get("selected_candidate"))
    verifier_status = verifier_status or selected_candidate.get("verifier_status")
    if verifier_status in {"WARN", "FAIL"}:
        reasons.append(f"verifier_status_{verifier_status}")
    if rank_margin < 0.01:
        reasons.append("low_rank_margin")
    if row.get("parser_status") == "FAIL":
        reasons.append("parser_status_fail")
    if execution_warnings:
        reasons.append("execution_warnings_present")
    numeric_value = final_answer.get("numeric_value")
    if _is_extreme_numeric(numeric_value):
        reasons.append("numeric_value_zero_or_extreme")
    unit = final_answer.get("unit") or execution_result.get("unit")
    target = final_answer.get("target")
    if target and not unit:
        reasons.append("numeric_target_unit_missing")
    return list(dict.fromkeys(reasons))


def _compact_row(row_index: int, result: dict[str, Any], risk_reasons: list[str]) -> dict[str, Any]:
    final_answer = _as_dict(result.get("final_answer"))
    execution_result = _execution_result(final_answer)
    execution_metadata = _execution_metadata(execution_result)
    ranking_summary = _as_dict(result.get("ranking_summary"))
    metadata = {
        "role_aware_coulomb_used": bool(execution_metadata.get("role_aware_coulomb_used")),
        "role_aware_electric_field_used": bool(execution_metadata.get("role_aware_electric_field_used")),
        "law_of_cosines_used": bool(execution_metadata.get("law_of_cosines_used")),
        "vector_sum_mode": execution_metadata.get("vector_sum_mode"),
        "electric_field_vector_mode": execution_metadata.get("electric_field_vector_mode"),
        "target_point": execution_metadata.get("target_point"),
        "target_charge_name": execution_metadata.get("target_charge_name"),
        "executed_dispatch_names": _as_list(execution_metadata.get("executed_dispatch_names")),
        "unsupported_dispatch_names": _as_list(execution_metadata.get("unsupported_dispatch_names")),
        "step9_dispatch_names": _as_list(execution_metadata.get("step9_dispatch_names")),
        "target_writeback_aliases": _as_list(execution_metadata.get("target_writeback_aliases")),
        "parsed_function_amplitudes": execution_metadata.get("parsed_function_amplitudes") or {},
        "parsed_current_delta": execution_metadata.get("parsed_current_delta"),
    }
    return {
        "row_index": row_index,
        "problem_text": result.get("problem_text"),
        "parser_status": result.get("parser_status"),
        "pipeline_status": result.get("pipeline_status"),
        "target": final_answer.get("target"),
        "target_unit": final_answer.get("unit") or execution_result.get("unit"),
        "answer": final_answer.get("answer"),
        "answer_type": final_answer.get("answer_type"),
        "numeric_value": final_answer.get("numeric_value"),
        "unit": final_answer.get("unit") or execution_result.get("unit"),
        "confidence": final_answer.get("confidence"),
        "selected_source": final_answer.get("source") or ranking_summary.get("selected_source"),
        "selected_templates": _as_list(final_answer.get("template_names")),
        "execution_status": execution_result.get("status"),
        "execution_warnings": _as_list(execution_result.get("warnings")),
        "execution_error_types": _execution_error_types(execution_result),
        "pipeline_warnings": _as_list(result.get("pipeline_warnings")),
        "rank_margin": _rank_margin(result, final_answer),
        "is_risky_numeric": bool(risk_reasons),
        "risky_numeric_reasons": risk_reasons,
        "sanity_status": _as_dict(final_answer.get("metadata")).get("sanity_check", {}).get("status")
        if isinstance(_as_dict(final_answer.get("metadata")).get("sanity_check"), dict)
        else None,
        "sanity_risk_level": _as_dict(final_answer.get("metadata")).get("sanity_check", {}).get("risk_level")
        if isinstance(_as_dict(final_answer.get("metadata")).get("sanity_check"), dict)
        else None,
        "sanity_should_accept_numeric": _as_dict(final_answer.get("metadata")).get("sanity_check", {}).get("should_accept_numeric")
        if isinstance(_as_dict(final_answer.get("metadata")).get("sanity_check"), dict)
        else None,
        "sanity_downgraded": bool(_as_dict(final_answer.get("metadata")).get("downgraded_numeric_answer")),
        "sanity_reasons": _as_dict(final_answer.get("metadata")).get("sanity_check", {}).get("reasons", [])
        if isinstance(_as_dict(final_answer.get("metadata")).get("sanity_check"), dict)
        else [],
        "sanity_legacy_risk_level": _sanity_meta(final_answer).get("legacy_risk_level"),
        "sanity_hard_risk_reasons": _sanity_meta(final_answer).get("hard_risk_reasons", []),
        "sanity_soft_risk_reasons": _sanity_meta(final_answer).get("soft_risk_reasons", []),
        "sanity_unit_alias_resolved": _sanity_meta(final_answer).get("unit_alias_resolved", False),
        "sanity_expected_zero_accepted": _sanity_meta(final_answer).get("expected_zero_accepted", False),
        "canonicalization_used": _canon_info(final_answer)[0],
        "canonicalization_status": _canon_info(final_answer)[1],
        "canonicalization_mappings": _canon_info(final_answer)[2],
        "original_llm_formula_names": _canon_info(final_answer)[3],
        "canonical_formula_names": _canon_info(final_answer)[4],
        "metadata": metadata,
    }


def _add_sample(samples: dict[str, list[dict[str, Any]]], group: str, row: dict[str, Any], limit: int) -> None:
    if len(samples[group]) < limit:
        samples[group].append(row)


def _summarize(rows: list[dict[str, Any]], config: Type2CandidatePipelineConfig, sample_limit: int) -> dict[str, Any]:
    pipeline_status_counts: Counter[str] = Counter()
    parser_status_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    execution_status_counts: Counter[str] = Counter()
    selected_source_counts: Counter[str] = Counter()
    selected_template_counts: Counter[str] = Counter()
    numeric_by_template_counts: Counter[str] = Counter()
    symbolic_by_template_counts: Counter[str] = Counter()
    fail_by_template_counts: Counter[str] = Counter()
    target_unresolved_by_template_counts: Counter[str] = Counter()
    missing_input_by_variable_counts: Counter[str] = Counter()
    top_execution_error_types: Counter[str] = Counter()
    top_execution_warnings: Counter[str] = Counter()
    adapter_error_type_counts: Counter[str] = Counter()
    adapter_warning_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    target_unit_counts: Counter[str] = Counter()
    risky_numeric_by_template_counts: Counter[str] = Counter()
    risky_numeric_reasons: Counter[str] = Counter()
    unsupported_dispatch_by_formula: Counter[str] = Counter()
    step9_dispatch_by_formula: Counter[str] = Counter()
    sanity_status_counts: Counter[str] = Counter()
    sanity_risk_level_counts: Counter[str] = Counter()
    downgraded_numeric_by_template_counts: Counter[str] = Counter()
    high_risk_numeric_by_template_counts: Counter[str] = Counter()
    critical_numeric_by_template_counts: Counter[str] = Counter()
    sanity_reason_counts: Counter[str] = Counter()
    hard_risk_reason_counts: Counter[str] = Counter()
    soft_risk_reason_counts: Counter[str] = Counter()

    confidences: list[float] = []
    numeric_answer_count = 0
    symbolic_trace_count = 0
    unresolved_count = 0
    failed_count = 0
    low_confidence_count = 0
    role_aware_coulomb_used_count = 0
    role_aware_coulomb_numeric_count = 0
    law_of_cosines_composition_count = 0
    law_of_cosines_numeric_count = 0
    vector_sum_executed_count = 0
    vector_sum_symbolic_count = 0
    k_missing_count = 0
    risky_numeric_count = 0
    electric_field_numeric_count = 0
    electric_field_symbolic_count = 0
    ohm_power_numeric_count = 0
    parallel_plate_numeric_count = 0
    lc_rlc_numeric_count = 0
    measurement_numeric_count = 0
    magnetic_flux_emf_numeric_count = 0
    capacitor_inverse_numeric_count = 0
    step9_dispatch_count = 0
    step9_target_writeback_count = 0
    llm_fallback_selected_count = 0
    llm_fallback_numeric_count = 0
    llm_fallback_symbolic_count = 0
    llm_fallback_canonicalized_candidate_count = 0
    llm_fallback_canonicalized_selected_count = 0
    llm_fallback_canonicalized_numeric_count = 0
    raw_llm_unsupported_dispatch_count = 0
    canonicalized_unsupported_dispatch_count = 0
    canonicalization_status_counts: Counter[str] = Counter()
    canonicalization_mapping_counts: Counter[str] = Counter()
    canonicalization_failed_counts: Counter[str] = Counter()
    canonicalization_warning_counts: Counter[str] = Counter()
    role_aware_electric_field_used_count = 0
    role_aware_electric_field_numeric_count = 0
    accepted_numeric_count = 0
    downgraded_numeric_count = 0
    critical_numeric_count = 0
    high_risk_numeric_count = 0
    medium_risk_numeric_count = 0
    low_risk_numeric_count = 0
    downgraded_by_hard_risk_count = 0
    downgraded_by_soft_risk_count = 0
    unit_alias_resolved_count = 0
    expected_zero_accepted_count = 0
    high_to_medium_reclassified_count = 0
    high_to_low_reclassified_count = 0
    accepted_high_risk_count = 0
    samples = {group: [] for group in SAMPLE_GROUPS}

    for row in rows:
        pipeline_status = str(row.get("pipeline_status", ""))
        parser_status = str(row.get("parser_status", ""))
        final_answer = _as_dict(row.get("final_answer"))
        execution_result = _execution_result(final_answer)
        execution_metadata = _execution_metadata(execution_result)
        answer_type = str(final_answer.get("answer_type", ""))
        template_names = _template_names(final_answer)
        confidence = float(final_answer.get("confidence", 0.0) or 0.0)
        rank_margin = _rank_margin(row, final_answer)
        risk_reasons = _as_list(row.get("risky_numeric_reasons"))
        compact = _as_dict(row.get("compact_row"))
        sanity_check = _as_dict(_as_dict(final_answer.get("metadata")).get("sanity_check"))
        sanity_status = str(sanity_check.get("status", "")) if sanity_check else ""
        sanity_risk = str(sanity_check.get("risk_level", "")) if sanity_check else ""
        sanity_accepts = bool(sanity_check.get("should_accept_numeric")) if sanity_check else answer_type == "numeric"
        sanity_downgraded = bool(_as_dict(final_answer.get("metadata")).get("downgraded_numeric_answer"))
        if sanity_status:
            sanity_status_counts[sanity_status] += 1
        if sanity_risk:
            sanity_risk_level_counts[sanity_risk] += 1
            if sanity_risk == "CRITICAL":
                critical_numeric_count += 1
            elif sanity_risk == "HIGH":
                high_risk_numeric_count += 1
            elif sanity_risk == "MEDIUM":
                medium_risk_numeric_count += 1
            elif sanity_risk == "LOW":
                low_risk_numeric_count += 1
        for reason in _as_list(sanity_check.get("reasons")):
            sanity_reason_counts[str(reason)] += 1
        sanity_meta = _as_dict(sanity_check.get("metadata"))
        for reason in _as_list(sanity_meta.get("hard_risk_reasons")):
            hard_risk_reason_counts[str(reason)] += 1
        for reason in _as_list(sanity_meta.get("soft_risk_reasons")):
            soft_risk_reason_counts[str(reason)] += 1
        if sanity_meta.get("unit_alias_resolved"):
            unit_alias_resolved_count += 1
        if sanity_meta.get("expected_zero_accepted"):
            expected_zero_accepted_count += 1
        if sanity_meta.get("reclassified_high_to_medium"):
            high_to_medium_reclassified_count += 1
        if sanity_meta.get("reclassified_high_to_low"):
            high_to_low_reclassified_count += 1
        if sanity_meta.get("accepted_high_risk"):
            accepted_high_risk_count += 1
        if sanity_downgraded:
            if sanity_meta.get("downgrade_cause") == "soft":
                downgraded_by_soft_risk_count += 1
            else:
                downgraded_by_hard_risk_count += 1

        pipeline_status_counts[pipeline_status] += 1
        parser_status_counts[parser_status] += 1
        answer_type_counts[answer_type] += 1
        confidences.append(confidence)
        if confidence < config.low_confidence_threshold:
            low_confidence_count += 1
            _add_sample(samples, "low_confidence", compact, sample_limit)
        if pipeline_status == "ERROR":
            failed_count += 1
            _add_sample(samples, "errors", compact, sample_limit)
        if parser_status == "FAIL":
            _add_sample(samples, "parser_fail", compact, sample_limit)
        if answer_type == "numeric":
            numeric_answer_count += 1
            if sanity_accepts:
                accepted_numeric_count += 1
        elif answer_type == "symbolic_trace":
            symbolic_trace_count += 1
            _add_sample(samples, "symbolic_trace", compact, sample_limit)
        elif answer_type == "unresolved":
            unresolved_count += 1

        target = final_answer.get("target")
        if target:
            target_counts[str(target)] += 1
        unit = final_answer.get("unit") or execution_result.get("unit")
        if unit:
            target_unit_counts[str(unit)] += 1
        source = final_answer.get("source") or _as_dict(row.get("ranking_summary")).get("selected_source")
        if source:
            selected_source_counts[str(source)] += 1
        for template_name in template_names:
            selected_template_counts[template_name] += 1
            if answer_type == "numeric":
                numeric_by_template_counts[template_name] += 1
            if answer_type == "symbolic_trace":
                symbolic_by_template_counts[template_name] += 1
            if risk_reasons:
                risky_numeric_by_template_counts[template_name] += 1
            if sanity_downgraded:
                downgraded_numeric_by_template_counts[template_name] += 1
            if sanity_risk == "HIGH":
                high_risk_numeric_by_template_counts[template_name] += 1
            if sanity_risk == "CRITICAL":
                critical_numeric_by_template_counts[template_name] += 1
        if sanity_downgraded:
            downgraded_numeric_count += 1
        template_blob = " ".join(template_names)
        if "electric_field" in template_blob:
            if answer_type == "numeric":
                electric_field_numeric_count += 1
            else:
                electric_field_symbolic_count += 1
        if answer_type == "numeric" and any(term in template_blob for term in ("ohms", "power_from_voltage", "power_v2", "power_from_current", "total_power")):
            ohm_power_numeric_count += 1
        if answer_type == "numeric" and any(term in template_blob for term in ("parallel_plate", "epsilon_r", "dielectric")):
            parallel_plate_numeric_count += 1
        if answer_type == "numeric" and any(term in template_blob for term in ("lc_", "rlc", "inductor", "resonance", "impedance", "reactance", "power_factor")):
            lc_rlc_numeric_count += 1
        if answer_type == "numeric" and any(term in template_blob for term in ("error", "mean_value", "measurement", "least_count", "uncertainty")):
            measurement_numeric_count += 1
        if answer_type == "numeric" and (
            any(term in template_blob for term in ("magnetic", "flux", "emf", "solenoid", "turn_density", "field_from_density", "field_full"))
            or final_answer.get("target") in {"B", "Phi_B", "emf", "n_turns_per_meter"}
        ):
            magnetic_flux_emf_numeric_count += 1
        if answer_type == "numeric" and any(
            term in template_blob for term in ("capacitance_from_energy", "voltage_from_capacitor", "capacitor_inverse", "energy_charge_capacitance", "charge_capacitance")
        ):
            capacitor_inverse_numeric_count += 1
        step9_names = _as_list(execution_metadata.get("step9_dispatch_names"))
        if step9_names:
            step9_dispatch_count += len(step9_names)
            for fname in step9_names:
                step9_dispatch_by_formula[str(fname)] += 1
        if _as_list(execution_metadata.get("target_writeback_aliases")):
            step9_target_writeback_count += 1

        # --- Step 10: LLM fallback canonicalization diagnostics ---
        selected_source = str(final_answer.get("source") or "")
        is_raw_llm = bool(final_answer.get("metadata", {}).get("used_llm_fallback")) or "llm_fallback" in " ".join(template_names)
        if is_raw_llm and selected_source != "llm_fallback_canonicalized":
            llm_fallback_selected_count += 1
            if answer_type == "numeric":
                llm_fallback_numeric_count += 1
            else:
                llm_fallback_symbolic_count += 1
            for warning in _as_list(execution_result.get("warnings")):
                if str(warning).startswith("Unsupported formula dispatch:"):
                    raw_llm_unsupported_dispatch_count += 1
        canon_used, canon_status, canon_maps, _orig_f, canon_f = _canon_info(final_answer)
        if canon_used:
            llm_fallback_canonicalized_candidate_count += 1
            if canon_status:
                canonicalization_status_counts[str(canon_status)] += 1
            for entry in canon_maps:
                if isinstance(entry, dict):
                    st = str(entry.get("status"))
                    if st == "mapped":
                        canonicalization_mapping_counts[str(entry.get("canonical_formula"))] += 1
                    elif st in {"skipped", "unmapped"}:
                        canonicalization_failed_counts[str(entry.get("source"))] += 1
        if selected_source == "llm_fallback_canonicalized":
            llm_fallback_canonicalized_selected_count += 1
            if answer_type == "numeric":
                llm_fallback_canonicalized_numeric_count += 1
            for warning in _as_list(execution_result.get("warnings")):
                if str(warning).startswith("Unsupported formula dispatch:"):
                    canonicalized_unsupported_dispatch_count += 1

        execution_status = str(execution_result.get("status", ""))
        if execution_status:
            execution_status_counts[execution_status] += 1
        if execution_status == "FAIL":
            for template_name in template_names:
                fail_by_template_counts[template_name] += 1

        warning_suggests_symbolic_vector = False
        for warning in _as_list(execution_result.get("warnings")):
            warning_text = str(warning)
            top_execution_warnings[warning_text] += 1
            if "Vector sum left symbolic" in warning_text or "Ambiguous vector geometry" in warning_text:
                warning_suggests_symbolic_vector = True
            if warning_text.startswith("Unsupported formula dispatch:"):
                unsupported_dispatch_by_formula[warning_text.replace("Unsupported formula dispatch:", "").strip()] += 1
        for error in _as_list(execution_result.get("errors")):
            if not isinstance(error, dict):
                continue
            error_type = str(error.get("error_type", "unknown"))
            top_execution_error_types[error_type] += 1
            if error_type == "target_unresolved":
                _add_sample(samples, "target_unresolved", compact, sample_limit)
                for template_name in template_names:
                    target_unresolved_by_template_counts[template_name] += 1
            if error_type == "missing_inputs":
                _add_sample(samples, "missing_inputs", compact, sample_limit)
                for variable in _as_list(error.get("missing_vars")):
                    variable_name = str(variable)
                    missing_input_by_variable_counts[variable_name] += 1
                    if variable_name == "k":
                        k_missing_count += 1

        adapter_diagnostics = _as_dict(row.get("adapter_diagnostics"))
        for error_type in _as_list(adapter_diagnostics.get("error_types")):
            adapter_error_type_counts[str(error_type)] += 1
        warning_count = int(adapter_diagnostics.get("warning_count", 0) or 0)
        if warning_count:
            adapter_warning_counts[f"warning_count_{warning_count}"] += 1

        vector_sum_mode = execution_metadata.get("vector_sum_mode")
        if vector_sum_mode in VECTOR_EXECUTED_MODES:
            vector_sum_executed_count += 1
        elif vector_sum_mode in VECTOR_SYMBOLIC_MODES or warning_suggests_symbolic_vector:
            vector_sum_symbolic_count += 1
        if execution_metadata.get("role_aware_coulomb_used"):
            role_aware_coulomb_used_count += 1
            if answer_type == "numeric":
                role_aware_coulomb_numeric_count += 1
        if execution_metadata.get("law_of_cosines_used"):
            law_of_cosines_composition_count += 1
            if answer_type == "numeric":
                law_of_cosines_numeric_count += 1
        if execution_metadata.get("role_aware_electric_field_used"):
            role_aware_electric_field_used_count += 1
            if answer_type == "numeric":
                role_aware_electric_field_numeric_count += 1

        if risk_reasons:
            risky_numeric_count += 1
            _add_sample(samples, "risky_numeric", compact, sample_limit)
            for reason in risk_reasons:
                risky_numeric_reasons[str(reason)] += 1

    total = len(rows)
    return {
        "evaluated": total,
        "pipeline_status_counts": _counter_dict(pipeline_status_counts),
        "parser_status_counts": _counter_dict(parser_status_counts),
        "answer_type_counts": _counter_dict(answer_type_counts),
        "execution_status_counts": _counter_dict(execution_status_counts),
        "numeric_answer_count": numeric_answer_count,
        "symbolic_trace_count": symbolic_trace_count,
        "unresolved_count": unresolved_count,
        "failed_count": failed_count,
        "numeric_rate": _rate(numeric_answer_count, total),
        "symbolic_rate": _rate(symbolic_trace_count, total),
        "error_rate": _rate(failed_count, total),
        "selected_source_counts": _counter_dict(selected_source_counts),
        "selected_template_counts": _counter_dict(selected_template_counts),
        "numeric_by_template_counts": _counter_dict(numeric_by_template_counts),
        "symbolic_by_template_counts": _counter_dict(symbolic_by_template_counts),
        "fail_by_template_counts": _counter_dict(fail_by_template_counts),
        "target_unresolved_by_template_counts": _counter_dict(target_unresolved_by_template_counts),
        "missing_input_by_variable_counts": _counter_dict(missing_input_by_variable_counts),
        "top_execution_error_types": _counter_dict(top_execution_error_types),
        "top_execution_warnings": _counter_dict(top_execution_warnings),
        "adapter_error_type_counts": _counter_dict(adapter_error_type_counts),
        "adapter_warning_counts": _counter_dict(adapter_warning_counts),
        "target_counts": _counter_dict(target_counts),
        "target_unit_counts": _counter_dict(target_unit_counts),
        "role_aware_coulomb_used_count": role_aware_coulomb_used_count,
        "role_aware_coulomb_numeric_count": role_aware_coulomb_numeric_count,
        "law_of_cosines_composition_count": law_of_cosines_composition_count,
        "law_of_cosines_numeric_count": law_of_cosines_numeric_count,
        "vector_sum_executed_count": vector_sum_executed_count,
        "vector_sum_symbolic_count": vector_sum_symbolic_count,
        "k_missing_count": k_missing_count,
        "low_confidence_count": low_confidence_count,
        "mean_confidence": _mean(confidences),
        "median_confidence": median(confidences) if confidences else 0.0,
        "risky_numeric_count": risky_numeric_count,
        "risky_numeric_rate": _rate(risky_numeric_count, numeric_answer_count),
        "risky_numeric_by_template_counts": _counter_dict(risky_numeric_by_template_counts),
        "risky_numeric_reasons": _counter_dict(risky_numeric_reasons),
        "sanity_status_counts": _counter_dict(sanity_status_counts),
        "sanity_risk_level_counts": _counter_dict(sanity_risk_level_counts),
        "accepted_numeric_count": accepted_numeric_count,
        "downgraded_numeric_count": downgraded_numeric_count,
        "critical_numeric_count": critical_numeric_count,
        "high_risk_numeric_count": high_risk_numeric_count,
        "medium_risk_numeric_count": medium_risk_numeric_count,
        "low_risk_numeric_count": low_risk_numeric_count,
        "accepted_numeric_rate": _rate(accepted_numeric_count, numeric_answer_count),
        "downgraded_numeric_by_template_counts": _counter_dict(downgraded_numeric_by_template_counts),
        "high_risk_numeric_by_template_counts": _counter_dict(high_risk_numeric_by_template_counts),
        "critical_numeric_by_template_counts": _counter_dict(critical_numeric_by_template_counts),
        "sanity_reason_counts": _counter_dict(sanity_reason_counts),
        "hard_risk_reason_counts": _counter_dict(hard_risk_reason_counts),
        "soft_risk_reason_counts": _counter_dict(soft_risk_reason_counts),
        "downgraded_by_hard_risk_count": downgraded_by_hard_risk_count,
        "downgraded_by_soft_risk_count": downgraded_by_soft_risk_count,
        "unit_alias_resolved_count": unit_alias_resolved_count,
        "expected_zero_accepted_count": expected_zero_accepted_count,
        "high_to_medium_reclassified_count": high_to_medium_reclassified_count,
        "high_to_low_reclassified_count": high_to_low_reclassified_count,
        "accepted_high_risk_count": accepted_high_risk_count,
        "unsupported_dispatch_count": sum(unsupported_dispatch_by_formula.values()),
        "unsupported_dispatch_by_formula": _counter_dict(unsupported_dispatch_by_formula),
        "electric_field_numeric_count": electric_field_numeric_count,
        "electric_field_symbolic_count": electric_field_symbolic_count,
        "ohm_power_numeric_count": ohm_power_numeric_count,
        "parallel_plate_numeric_count": parallel_plate_numeric_count,
        "lc_rlc_numeric_count": lc_rlc_numeric_count,
        "magnetic_flux_emf_numeric_count": magnetic_flux_emf_numeric_count,
        "capacitor_inverse_numeric_count": capacitor_inverse_numeric_count,
        "step9_dispatch_count": step9_dispatch_count,
        "step9_dispatch_by_formula": _counter_dict(step9_dispatch_by_formula),
        "step9_target_writeback_count": step9_target_writeback_count,
        "llm_fallback_selected_count": llm_fallback_selected_count,
        "llm_fallback_numeric_count": llm_fallback_numeric_count,
        "llm_fallback_symbolic_count": llm_fallback_symbolic_count,
        "llm_fallback_canonicalized_candidate_count": llm_fallback_canonicalized_candidate_count,
        "llm_fallback_canonicalized_selected_count": llm_fallback_canonicalized_selected_count,
        "llm_fallback_canonicalized_numeric_count": llm_fallback_canonicalized_numeric_count,
        "raw_llm_unsupported_dispatch_count": raw_llm_unsupported_dispatch_count,
        "canonicalized_unsupported_dispatch_count": canonicalized_unsupported_dispatch_count,
        "canonicalization_status_counts": _counter_dict(canonicalization_status_counts),
        "canonicalization_mapping_counts": _counter_dict(canonicalization_mapping_counts),
        "canonicalization_failed_counts": _counter_dict(canonicalization_failed_counts),
        "canonicalization_warning_counts": _counter_dict(canonicalization_warning_counts),
        "measurement_numeric_count": measurement_numeric_count,
        "role_aware_electric_field_used_count": role_aware_electric_field_used_count,
        "role_aware_electric_field_numeric_count": role_aware_electric_field_numeric_count,
        "sampled_rows": samples,
    }


def validate_full_dataset(
    input_path: Path,
    output_path: Path,
    max_candidates: int = 8,
    use_llm_fallback: bool = False,
    compact: bool = False,
    sample_failures: int = 100,
    save_jsonl: Path | None = None,
    baseline_json: Path | None = None,
    downgrade_high_risk_numeric: bool = False,
) -> dict[str, Any]:
    rows = _load_csv(input_path)
    text_column = _detect_text_column(rows)
    config = Type2CandidatePipelineConfig(
        use_llm_fallback=use_llm_fallback,
        log_failures=False,
        max_candidates=max_candidates,
        execute_numeric=True,
        include_intermediate_outputs=not compact,
        include_scoreboards=True,
        downgrade_high_risk_numeric=downgrade_high_risk_numeric,
    )
    metric_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    jsonl_handle = save_jsonl.open("w", encoding="utf-8") if save_jsonl else None
    try:
        for row_index, csv_row in enumerate(rows, start=1):
            problem_text = csv_row.get(text_column, "")
            try:
                result = run_type2_candidate_pipeline(problem_text, config).to_dict()
            except Exception as exc:
                result = {
                    "problem_text": problem_text,
                    "parser_status": "ERROR",
                    "pipeline_status": "ERROR",
                    "pipeline_warnings": [],
                    "pipeline_errors": [{"error_type": "pipeline_exception", "message": repr(exc)}],
                    "adapter_diagnostics": {"error_types": ["pipeline_exception"], "warning_count": 0},
                    "ranking_summary": {},
                    "selected_candidate": None,
                    "final_answer": {
                        "answer": None,
                        "unit": None,
                        "target": None,
                        "numeric_value": None,
                        "answer_type": "unresolved",
                        "confidence": 0.0,
                        "source": None,
                        "template_names": [],
                        "warnings": [repr(exc)],
                        "metadata": {},
                    },
                }
            result["row_index"] = row_index
            final_answer = _as_dict(result.get("final_answer"))
            execution_result = _execution_result(final_answer)
            execution_metadata = _execution_metadata(execution_result)
            rank_margin = _rank_margin(result, final_answer)
            risk_reasons = _risky_numeric_reasons(result, final_answer, execution_result, execution_metadata, rank_margin, config)
            result["is_risky_numeric"] = bool(risk_reasons)
            result["risky_numeric_reasons"] = risk_reasons
            compact_row = _compact_row(row_index, result, risk_reasons)
            result["compact_row"] = compact_row
            metric_rows.append(result)
            output_row = compact_row if compact else result
            output_rows.append(output_row)
            if jsonl_handle:
                jsonl_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    finally:
        if jsonl_handle:
            jsonl_handle.close()

    summary = _summarize(metric_rows, config, sample_failures)
    if baseline_json and baseline_json.exists():
        baseline_payload = json.loads(baseline_json.read_text(encoding="utf-8"))
        baseline_summary = _as_dict(baseline_payload.get("summary"))
        baseline_numeric = _as_dict(baseline_summary.get("numeric_by_template_counts"))
        current_numeric = _as_dict(summary.get("numeric_by_template_counts"))
        gains: dict[str, int] = {}
        for template_name, current_value in current_numeric.items():
            gains[str(template_name)] = int(current_value or 0) - int(baseline_numeric.get(template_name, 0) or 0)
        summary["numeric_gain_by_template"] = dict(sorted(gains.items(), key=lambda item: item[1], reverse=True))
    else:
        summary["numeric_gain_by_template"] = {}
    result_payload = {
        "input": str(input_path),
        "output": str(output_path),
        "jsonl_output": str(save_jsonl) if save_jsonl else None,
        "text_column": text_column,
        "config": {
            "max_candidates": max_candidates,
            "use_llm_fallback": use_llm_fallback,
            "compact": compact,
            "sample_failures": sample_failures,
            "downgrade_high_risk_numeric": downgrade_high_risk_numeric,
        },
        "summary": summary,
    }
    if not compact:
        result_payload["rows"] = output_rows
    output_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Type2 pipeline on the full dataset.")
    parser.add_argument("--input", default="Physics_Problems_Text_Only.csv")
    parser.add_argument("--output", default="type2_full_dataset_validation.json")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--use-llm-fallback", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--sample-failures", type=int, default=100)
    parser.add_argument("--save-jsonl", default="type2_full_dataset_rows.jsonl")
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--downgrade-high-risk-numeric", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = _resolve_input_path(args.input)
    output_path = Path(args.output).resolve()
    save_jsonl = Path(args.save_jsonl).resolve() if args.save_jsonl else None
    baseline_json = Path(args.baseline_json).resolve() if args.baseline_json else None
    payload = validate_full_dataset(
        input_path=input_path,
        output_path=output_path,
        max_candidates=args.max_candidates,
        use_llm_fallback=args.use_llm_fallback,
        compact=args.compact,
        sample_failures=max(args.sample_failures, 0),
        save_jsonl=save_jsonl,
        baseline_json=baseline_json,
        downgrade_high_risk_numeric=args.downgrade_high_risk_numeric,
    )
    print(json.dumps({"evaluated": payload["summary"]["evaluated"], "summary": payload["summary"]}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()