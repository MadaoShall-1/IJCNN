"""Deterministic sanity checks for Type2 final numeric answers.

Step 7.1 risk-reason calibration:
- HIGH-risk reasons are split into hard-risk and soft-risk. Hard-risk reasons keep
  the answer at HIGH (and may downgrade in strict mode). Soft-risk reasons are
  demoted to MEDIUM: they reduce confidence but do not automatically downgrade.
- Unit comparison uses alias normalization (ohm/Ohm, V/m/N/C, J/N*m, Hz/(1/s),
  Wb/T*m^2) and a precise (non-greedy) target -> family map, removing false
  unit_target_mismatch and missing-unit reports for dimensionless targets.
- Zero values are only treated as risky when not physically expected. Symmetric
  field/force cancellations (midpoint, center of square/equilateral, equal
  charges) are accepted at LOW/MEDIUM risk.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Type2SanityCheckResult:
    status: str
    should_accept_numeric: bool
    should_downgrade_to_symbolic: bool
    confidence_multiplier: float
    risk_level: str
    reasons: list[str]
    checks: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Type2SanityCheckConfig:
    enable_downgrade: bool = True
    downgrade_on_critical: bool = True
    downgrade_on_high_risk: bool = False
    low_rank_margin_threshold: float = 0.01
    extreme_value_abs_threshold: float = 1e30
    tiny_nonzero_threshold: float = 1e-30
    require_unit_for_numeric: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Target families (precise, non-greedy). Only families that have a meaningful unit
# expectation are listed; everything else returns None (no unit check is performed).
# --------------------------------------------------------------------------------------

FORCE_TARGETS = {"F", "F_net", "F_e", "F_on_q3", "F12", "F13", "F23", "F_total", "F_res", "F_C"}
FIELD_TARGETS = {"E", "E_net", "E_total", "E_field", "electric_field", "E_x", "E_y"}
CAPACITANCE_TARGETS = {"C", "C_cap", "C_after", "C_eq", "C_series", "C_parallel", "C_total"}
ENERGY_TARGETS = {"U_cap", "U_B", "U_E", "U_total", "W_done", "energy", "E_stored"}
POWER_TARGETS = {"P", "P_total", "P_avg", "P_diss", "P_R", "power"}
CURRENT_TARGETS = {"I", "I_rms", "I1", "I2", "I3", "I_0", "I_total", "current"}
RESISTANCE_TARGETS = {"R", "R_total", "R1", "R2", "R3", "R_eq", "Z", "Z_total", "X_L", "X_C", "reactance", "impedance"}
FREQUENCY_TARGETS = {"f", "f_res", "f_osc", "frequency"}
INDUCTANCE_TARGETS = {"L", "L_ind", "inductance"}
MAGNETIC_FIELD_TARGETS = {"B", "B_field", "magnetic_field"}
FLUX_TARGETS = {"Phi_B", "Phi", "flux", "magnetic_flux"}
ANGLE_TARGETS = {"theta", "phi", "alpha", "beta", "angle"}

# Dimensionless targets: no unit family and missing unit is acceptable.
DIMENSIONLESS_TARGETS = {
    "epsilon_r",
    "percent_error",
    "rel_error",
    "relative_error",
    "power_factor",
    "k_ratio",
    "ratio",
    "efficiency",
    "n_turns",
    "refractive_index",
    "Q_factor",
    "quality_factor",
}

# Targets where a missing unit should not be penalised (dimensionless plus
# measurement-error templates that legitimately omit the unit).
UNIT_OPTIONAL_TARGETS = DIMENSIONLESS_TARGETS | {
    "abs_error",
    "absolute_error",
    "uncertainty",
    "error",
    "delta",
}


def _target_family(target: str | None) -> str | None:
    name = str(target or "")
    if not name:
        return None
    if name in DIMENSIONLESS_TARGETS:
        return None
    if name in FORCE_TARGETS:
        return "force"
    if name in FIELD_TARGETS:
        return "field"
    if name in CAPACITANCE_TARGETS:
        return "capacitance"
    if name in ENERGY_TARGETS:
        return "energy"
    if name in POWER_TARGETS:
        return "power"
    if name in CURRENT_TARGETS:
        return "current"
    if name in RESISTANCE_TARGETS:
        return "resistance"
    if name in FREQUENCY_TARGETS:
        return "frequency"
    if name in INDUCTANCE_TARGETS:
        return "inductance"
    if name in MAGNETIC_FIELD_TARGETS:
        return "magnetic_field"
    if name in FLUX_TARGETS:
        return "flux"
    if name in ANGLE_TARGETS:
        return "angle"
    return None


# Canonical unit per family.
UNIT_EXPECTATIONS = {
    "force": {"N"},
    "field": {"V/m"},
    "capacitance": {"F"},
    "energy": {"J"},
    "power": {"W"},
    "current": {"A"},
    "resistance": {"ohm"},
    "frequency": {"Hz"},
    "inductance": {"H"},
    "magnetic_field": {"T"},
    "flux": {"Wb"},
    "angle": {"rad"},
}


# Alias groups: each member normalizes to the canonical key. The mojibake "Î©" is
# kept because some upstream encodings emit it for the ohm sign.
UNIT_ALIAS_GROUPS = {
    "ohm": {"ohm", "ohms", "Ohm", "Ω", "Î©", "\u03a9", "\u2126"},
    "Hz": {"Hz", "hz", "1/s", "s^-1", "s**-1", "/s", "sec^-1"},
    "J": {"J", "N*m", "N·m", "N m", "Nm", "N*m^1", "joule", "joules"},
    "V/m": {"V/m", "N/C", "V m^-1", "V/m^1"},
    "Wb": {"Wb", "T*m^2", "T·m^2", "T m^2", "T*m**2", "weber"},
    "rad": {"rad", "radian", "radians", ""},
}


def _canonical_unit(unit: Any) -> str | None:
    if unit is None:
        return None
    raw = str(unit).strip()
    for canonical, members in UNIT_ALIAS_GROUPS.items():
        if raw in members:
            return canonical
    return raw


def _unit_compatibility(unit: Any, family: str | None) -> dict[str, Any]:
    """Return compatibility info for a unit against a family expectation."""
    expected = UNIT_EXPECTATIONS.get(family) if family else None
    if not expected:
        return {"checked": False, "compatible": True, "alias_used": False}
    raw = None if unit is None else str(unit).strip()
    canonical = _canonical_unit(unit)
    direct = raw in expected
    via_alias = (not direct) and (canonical in expected)
    compatible = direct or via_alias
    return {
        "checked": True,
        "compatible": compatible,
        "alias_used": bool(via_alias and compatible),
        "raw": raw,
        "canonical": canonical,
        "expected": sorted(expected),
    }


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _metadata(final_answer: Any) -> dict[str, Any]:
    value = _get(final_answer, "metadata", {})
    return value if isinstance(value, dict) else {}


def _execution_result(final_answer: Any) -> dict[str, Any]:
    value = _metadata(final_answer).get("execution_result", {})
    return value if isinstance(value, dict) else {}


def _execution_metadata(final_answer: Any) -> dict[str, Any]:
    value = _execution_result(final_answer).get("metadata", {})
    return value if isinstance(value, dict) else {}


def _ranking_margin(ranking_result: Any, final_answer: Any) -> float:
    metadata = _metadata(final_answer)
    if "rank_margin" in metadata:
        return float(metadata.get("rank_margin") or 0.0)
    ranked = _get(ranking_result, "ranked_candidates", []) if ranking_result else []
    if len(ranked) >= 2:
        return float(_get(ranked[0], "rank_score", 0.0) - _get(ranked[1], "rank_score", 0.0))
    if ranked:
        return float(_get(ranked[0], "rank_score", 0.0))
    return 0.0


def _is_expected_zero(final_answer: Any, world_input: Any, target: str | None, family: str | None) -> bool:
    execution_metadata = _execution_metadata(final_answer)
    templates = " ".join(str(item) for item in (_get(final_answer, "template_names", []) or [])).lower()
    text = str(_get(world_input, "problem_text", "") or "").lower()
    mode_values = {
        str(execution_metadata.get("vector_sum_mode") or ""),
        str(execution_metadata.get("electric_field_vector_mode") or ""),
        str(execution_metadata.get("role_aware_geometry_mode") or ""),
    }
    if any("cancel" in value or "symmetric" in value for value in mode_values):
        return True
    if execution_metadata.get("symmetric_cancellation") or execution_metadata.get("expected_zero"):
        return True
    if any(token in templates for token in ("symmetric", "midpoint", "square_center", "cancellation")):
        return True

    target_name = str(target or "")
    field_or_force = family in {"field", "force"} or target_name in {
        "E",
        "E_net",
        "E_total",
        "F_net",
        "F_e",
        "F_on_q3",
    }
    if not field_or_force:
        return False

    symmetry_cues = (
        "midpoint" in text
        or "mid-point" in text
        or "centre" in text
        or "center" in text
        or "equidistant" in text
        or "equilateral" in text
        or "symmetric" in text
        or "symmetrical" in text
    )
    if symmetry_cues:
        return True
    # Equal same-sign charges configuration also produces genuine cancellation.
    if ("square" in text or "equilateral" in text) and ("centre" in text or "center" in text):
        return True
    return False


# Severity ranking helpers.
_SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
_RANK_SEVERITY = {value: key for key, value in _SEVERITY_RANK.items()}


def sanity_check_final_answer(
    final_answer: Any,
    world_input: Any,
    ranking_result: Any | None = None,
    config: Type2SanityCheckConfig | None = None,
) -> Type2SanityCheckResult:
    config = config or Type2SanityCheckConfig()
    checks: dict[str, Any] = {}
    # Each entry: (reason, severity, risk_class). risk_class is "hard" or "soft".
    entries: list[tuple[str, str, str]] = []
    # High-severity signals that were suppressed by calibration (alias/zero/dimensionless/
    # family correction). Tracked only for diagnostics / reclassification accounting.
    suppressed_high_signals: list[str] = []
    unit_alias_used: list[str] = []
    expected_zero_accepted = False

    def add(reason: str, severity: str, risk_class: str, detail: Any = True) -> None:
        checks[reason] = detail
        entries.append((reason, severity, risk_class))

    answer_type = _get(final_answer, "answer_type")
    numeric_value = _get(final_answer, "numeric_value")
    target = _get(final_answer, "target")
    unit = _get(final_answer, "unit") or _execution_result(final_answer).get("unit")
    metadata = _metadata(final_answer)
    execution_result = _execution_result(final_answer)
    execution_metadata = _execution_metadata(final_answer)
    execution_status = str(execution_result.get("status") or "")
    template_names = [str(item) for item in (_get(final_answer, "template_names", []) or [])]
    template_blob = " ".join(template_names).lower()
    target_name = str(target) if target is not None else None
    family = _target_family(target_name)

    # ---- numeric value sanity ---------------------------------------------------------
    numeric_float: float | None = None
    if answer_type == "numeric" and numeric_value is None:
        add("numeric_answer_missing_value", "CRITICAL", "hard")
    if numeric_value is not None:
        try:
            numeric_float = float(numeric_value)
        except (TypeError, ValueError):
            numeric_float = math.nan
        if math.isnan(numeric_float) or math.isinf(numeric_float):
            add("numeric_value_nan_or_inf", "CRITICAL", "hard", numeric_value)
        elif abs(numeric_float) > config.extreme_value_abs_threshold:
            add("numeric_value_extreme", "CRITICAL", "hard", numeric_float)
        elif numeric_float == 0.0:
            if _is_expected_zero(final_answer, world_input, target_name, family):
                expected_zero_accepted = True
                suppressed_high_signals.append("unexpected_zero_numeric")
                add("expected_zero_accepted", "LOW", "soft")
            else:
                add("unexpected_zero_numeric", "HIGH", "hard")
        elif 0.0 < abs(numeric_float) < config.tiny_nonzero_threshold:
            add("tiny_nonzero_numeric", "HIGH", "hard", numeric_float)
        if numeric_float is not None and not math.isnan(numeric_float) and family and numeric_float < 0 and family in {
            "force",
            "field",
            "capacitance",
            "energy",
            "power",
            "resistance",
            "frequency",
            "inductance",
            "magnetic_field",
        }:
            add("negative_magnitude_target", "CRITICAL", "hard", numeric_float)

    # ---- unit presence ----------------------------------------------------------------
    unit_optional = target_name in UNIT_OPTIONAL_TARGETS
    if config.require_unit_for_numeric and answer_type == "numeric" and target and not unit:
        if unit_optional:
            suppressed_high_signals.append("numeric_physical_answer_missing_unit")
        else:
            add("numeric_physical_answer_missing_unit", "HIGH", "hard")
    if not target:
        add("target_missing", "HIGH", "hard")

    # ---- parser status ----------------------------------------------------------------
    parser_status = _get(world_input, "parser_status") or metadata.get("parser_status")
    if parser_status == "FAIL":
        clean_numeric = (
            answer_type == "numeric"
            and numeric_value is not None
            and bool(unit)
            and bool(target)
            and execution_status in {"", "PASS"}
            and not (numeric_float is not None and (math.isnan(numeric_float) or math.isinf(numeric_float)))
        )
        if clean_numeric:
            add("parser_status_fail", "HIGH", "soft")
        else:
            add("parser_status_fail", "HIGH", "hard")

    # ---- unit/target compatibility ----------------------------------------------------
    compatibility = _unit_compatibility(unit, family)
    if answer_type == "numeric" and compatibility.get("checked"):
        if compatibility.get("compatible"):
            if compatibility.get("alias_used"):
                unit_alias_used.append(str(unit))
                suppressed_high_signals.append("unit_target_mismatch")
        else:
            add(
                "unit_target_mismatch",
                "HIGH",
                "hard",
                {"target": target, "unit": unit, "expected": compatibility.get("expected")},
            )

    # ---- execution warnings -----------------------------------------------------------
    execution_warnings = [str(item) for item in execution_result.get("warnings", []) or []]
    warning_blob = " ".join(execution_warnings).lower()
    law_of_cosines_used = bool(execution_metadata.get("law_of_cosines_used"))
    charge_adjustment = execution_metadata.get("charge_interaction_adjustment")
    unit_target_consistent = (not compatibility.get("checked")) or compatibility.get("compatible")

    serious_warning_tokens = ("unresolved target", "missing input", "missing_inputs")
    serious_warning_present = any(token in warning_blob for token in serious_warning_tokens)
    # Unsupported dispatch only matters when the execution did not cleanly resolve.
    if "unsupported formula dispatch" in warning_blob and execution_status not in {"", "PASS"}:
        serious_warning_present = True

    if execution_warnings:
        if serious_warning_present and answer_type == "numeric":
            add("serious_execution_warning", "HIGH", "hard", execution_warnings)
        else:
            add("execution_warnings_present", "MEDIUM", "soft", execution_warnings)

    if any("Using geometric line angle for force magnitude" in warning for warning in execution_warnings):
        if law_of_cosines_used and unit_target_consistent and not (charge_adjustment and not unit_target_consistent):
            add("geometric_line_angle_warning", "HIGH", "soft")
        else:
            add("geometric_line_angle_warning", "HIGH", "hard")

    if any("Ambiguous vector geometry" in warning or "left symbolic" in warning for warning in execution_warnings):
        # On a numeric answer the ambiguity was safely resolved to a magnitude.
        if answer_type == "numeric" and numeric_value is not None and execution_status in {"", "PASS"}:
            add("ambiguous_vector_geometry_warning", "HIGH", "soft")
        else:
            add("ambiguous_vector_geometry_warning", "HIGH", "hard")

    if law_of_cosines_used and charge_adjustment:
        if unit_target_consistent:
            add("law_of_cosines_charge_adjustment", "HIGH", "soft", charge_adjustment)
        else:
            add("law_of_cosines_charge_adjustment", "HIGH", "hard", charge_adjustment)

    # ---- verifier status --------------------------------------------------------------
    verifier_status = metadata.get("verifier_status")
    if verifier_status == "WARN":
        add("verifier_status_warn", "MEDIUM", "soft")
    elif verifier_status == "FAIL":
        add("verifier_status_fail", "HIGH", "hard")

    # ---- rank margin ------------------------------------------------------------------
    rank_margin = _ranking_margin(ranking_result, final_answer)
    if rank_margin < config.low_rank_margin_threshold:
        add("low_rank_margin", "MEDIUM", "soft", rank_margin)

    # ---- template / target compatibility ----------------------------------------------
    if answer_type == "numeric":
        if "skeleton_placeholder" in template_names:
            add("skeleton_placeholder_numeric", "CRITICAL", "hard")
        if "boolean_check_candidate" in template_names:
            add("boolean_candidate_numeric", "CRITICAL", "hard")
        if any("conceptual" in template for template in template_names):
            add("conceptual_template_numeric", "HIGH", "hard")
        if ("coulomb" in template_blob or "electric_field" in template_blob) and charge_adjustment:
            if unit_target_consistent:
                add("vector_template_with_charge_adjustment", "HIGH", "soft")
            else:
                add("vector_template_with_charge_adjustment", "HIGH", "hard")

    # ---- aggregate severities ---------------------------------------------------------
    hard_risk_reasons = [reason for reason, severity, risk_class in entries if risk_class == "hard" and _SEVERITY_RANK[severity] >= _SEVERITY_RANK["HIGH"]]
    soft_high_reasons = [reason for reason, severity, risk_class in entries if risk_class == "soft" and severity == "HIGH"]

    has_critical = any(severity == "CRITICAL" for _, severity, _ in entries)
    has_hard_high = any(severity == "HIGH" and risk_class == "hard" for _, severity, risk_class in entries)
    has_high_class = any(severity == "HIGH" for _, severity, _ in entries)  # includes soft-high
    has_medium = any(severity == "MEDIUM" for _, severity, _ in entries)

    # Effective risk level after calibration: soft-HIGH demotes to MEDIUM.
    if has_critical:
        risk_level = "CRITICAL"
    elif has_hard_high:
        risk_level = "HIGH"
    elif soft_high_reasons or has_medium:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # "Legacy" level for reclassification accounting: treat every HIGH (hard or soft)
    # as HIGH and re-include suppressed HIGH signals (alias/zero/dimensionless).
    legacy_high = has_high_class or bool(suppressed_high_signals)
    if has_critical:
        legacy_risk_level = "CRITICAL"
    elif legacy_high:
        legacy_risk_level = "HIGH"
    elif has_medium:
        legacy_risk_level = "MEDIUM"
    else:
        legacy_risk_level = "LOW"

    reclassified_high_to_medium = legacy_risk_level == "HIGH" and risk_level == "MEDIUM"
    reclassified_high_to_low = legacy_risk_level == "HIGH" and risk_level == "LOW"

    if risk_level == "CRITICAL":
        status = "FAIL"
        confidence_multiplier = 0.30
    elif risk_level == "HIGH":
        status = "WARN"
        confidence_multiplier = 0.65
    elif risk_level == "MEDIUM":
        status = "WARN"
        # A MEDIUM that resulted from demoting a soft-HIGH keeps slightly lower confidence
        # than a clean MEDIUM, so calibration stays meaningful.
        confidence_multiplier = 0.78 if soft_high_reasons else 0.85
    else:
        status = "PASS"
        confidence_multiplier = 1.0

    should_accept_numeric = risk_level in {"LOW", "MEDIUM"} or (
        risk_level == "HIGH" and not config.downgrade_on_high_risk
    )
    if risk_level == "CRITICAL":
        should_accept_numeric = False

    should_downgrade = bool(
        config.enable_downgrade
        and (
            (risk_level == "CRITICAL" and config.downgrade_on_critical)
            or (risk_level == "HIGH" and config.downgrade_on_high_risk)
        )
    )
    # Downgrades, by construction, are always driven by hard / critical reasons.
    if should_downgrade:
        downgrade_cause = "hard" if (risk_level == "CRITICAL" or has_hard_high) else "soft"
    else:
        downgrade_cause = None

    accepted_high_risk = risk_level == "HIGH" and should_accept_numeric

    reasons = list(dict.fromkeys(reason for reason, _, _ in entries))
    soft_risk_reasons = list(dict.fromkeys(reason for reason, _, risk_class in entries if risk_class == "soft"))
    hard_risk_reasons = list(dict.fromkeys(hard_risk_reasons))

    return Type2SanityCheckResult(
        status=status,
        should_accept_numeric=should_accept_numeric,
        should_downgrade_to_symbolic=should_downgrade,
        confidence_multiplier=confidence_multiplier,
        risk_level=risk_level,
        reasons=reasons,
        checks=checks,
        metadata={
            "config": config.to_dict(),
            "rank_margin": rank_margin,
            "template_names": template_names,
            "target_family": family,
            "hard_risk_reasons": hard_risk_reasons,
            "soft_risk_reasons": soft_risk_reasons,
            "soft_high_reasons": list(dict.fromkeys(soft_high_reasons)),
            "suppressed_high_signals": list(dict.fromkeys(suppressed_high_signals)),
            "unit_alias_used": unit_alias_used,
            "unit_alias_resolved": bool(unit_alias_used),
            "expected_zero_accepted": expected_zero_accepted,
            "legacy_risk_level": legacy_risk_level,
            "reclassified_high_to_medium": reclassified_high_to_medium,
            "reclassified_high_to_low": reclassified_high_to_low,
            "accepted_high_risk": accepted_high_risk,
            "downgrade_cause": downgrade_cause,
        },
    )