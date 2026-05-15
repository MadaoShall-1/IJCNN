"""Verifier gate for Stage 0 parse objects."""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from .schemas import VerifierError, VerifierResult
from .target_detector import detect_target


NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z_])")
COMPATIBLE_DIMENSIONS = {
    "m": {"mass"},
    "m_object": {"mass"},
    "v": {"velocity"},
    "v_final": {"velocity"},
    "v_0": {"velocity"},
    "v_wave": {"velocity"},
    "a": {"acceleration"},
    "R": {"resistance"},
    "Z": {"resistance"},
    "X": {"resistance"},
    "X_L": {"resistance"},
    "X_C": {"resistance"},
    "I": {"current"},
    "I_rms": {"current"},
    "I_max": {"current"},
    "V": {"voltage"},
    "V_rms": {"voltage"},
    "V_after": {"voltage"},
    "C_cap": {"capacitance"},
    "C_eq": {"capacitance"},
    "C_after": {"capacitance"},
    "L_ind": {"inductance"},
    "q": {"charge"},
    "Q": {"charge"},
    "Q_max": {"charge"},
    "Q_after": {"charge"},
    "f": {"frequency"},
    "f_res": {"frequency"},
    "f_osc": {"frequency"},
    "omega": {"angular_frequency"},
    "omega_0": {"angular_frequency"},
    "lambda": {"length"},
    "r": {"length"},
    "d": {"length"},
    "h": {"length"},
    "L": {"length"},
    "F": {"force"},
    "F_net": {"force"},
    "F_e": {"force"},
    "F_on_q3": {"force"},
    "E": {"electric_field", "energy"},
    "B": {"magnetic_field"},
    "P": {"power"},
    "P_total": {"power"},
    "P_each": {"power"},
    "P_max": {"power"},
    "P_avg": {"power"},
    "KE": {"energy"},
    "PE": {"energy"},
    "E_energy": {"energy"},
    "theta": {"angle"},
    "direction": {"angle", "unknown"},
    "phi": {"angle"},
    "power_factor": {"dimensionless"},
    "epsilon_r": {"dimensionless"},
    "q_over_Qmax": {"dimensionless"},
    "ratio": {"dimensionless", "unknown"},
    "efficiency": {"dimensionless", "unknown"},
    "energy_fraction": {"dimensionless"},
    "magnetic_energy_fraction": {"dimensionless"},
    "electric_energy_fraction": {"dimensionless"},
    "U_after": {"energy"},
    "U_E": {"energy"},
    "U_B": {"energy"},
    "U_total": {"energy"},
    "delta_U": {"energy", "voltage", "unknown"},
    "delta_R": {"resistance", "unknown"},
    "delta_P": {"power", "unknown"},
    "delta_V": {"voltage"},
    "delta_I": {"current"},
    "uncertainty": {"unknown", "dimensionless", "length", "time", "mass", "velocity", "acceleration", "voltage", "current", "resistance", "energy", "power", "temperature"},
    "relation_E": {"unknown"},
    "relation_generic": {"unknown"},
    "equation_of_motion": {"unknown"},
    "I_over_Imax": {"dimensionless"},
    "v_avg": {"velocity"},
    "abs_error": {"unknown", "length", "time", "mass", "velocity", "acceleration", "voltage", "current", "resistance", "energy", "power", "temperature"},
    "rel_error": {"dimensionless", "unknown"},
    "percent_error": {"dimensionless", "unknown"},
    "measured_value": {"unknown", "length", "time", "mass", "velocity", "acceleration", "voltage", "current", "resistance", "energy", "power", "temperature"},
    "true_value": {"unknown", "length", "time", "mass", "velocity", "acceleration", "voltage", "current", "resistance", "energy", "power", "temperature"},
    "accepted_value": {"unknown", "length", "time", "mass", "velocity", "acceleration", "voltage", "current", "resistance", "energy", "power", "temperature"},
    "mean_value": {"unknown", "length", "time", "mass", "velocity", "acceleration", "voltage", "current", "resistance", "energy", "power", "temperature"},
    "random_error": {"unknown", "dimensionless", "length", "time", "mass", "velocity", "acceleration", "voltage", "current", "resistance", "energy", "power", "temperature"},
    "Phi_link": {"unknown", "magnetic_flux"},
}
CONSTANTS = {"g", "k", "pi", "epsilon_0", "mu_0"}


def _err(error_type: str, description: str, repair_hint: str) -> VerifierError:
    return VerifierError(error_type=error_type, description=description, repair_hint=repair_hint)


def _base_name(name: str) -> str:
    if re.match(r"q\d+$", name):
        return "q"
    if re.match(r"R\d+$", name):
        return "R"
    if re.match(r"X_[LC]\d+$", name):
        return name[:3]
    if re.match(r"C\d+$", name):
        return "C_cap"
    if re.match(r"v\d+$", name):
        return "v"
    if re.match(r"d\d+$", name):
        return "d"
    return name


def _verify_non_numeric(parse_object: Dict[str, object]) -> VerifierResult:
    """Lightweight verifier path for non-numeric question types.

    Stage 0.4.1: when the question_type classifier marks a problem as
    boolean_check or symbolic_derivation, we cannot fairly judge it by the
    numeric verifier (which expects a step_plan ending in unknown_quantity).
    Instead we do a minimal sanity check: the problem text must be present,
    and either some known quantities were extracted or some domain was
    identified. We do NOT require a step_plan, target, or coverage of every
    numeric token.

    Returns PASS_NON_NUMERIC on success.
    """
    errors: List[VerifierError] = []
    warnings: List[str] = []
    text = str(parse_object.get("problem_text", ""))
    if not text.strip():
        errors.append(_err("empty_problem_text", "problem_text is empty.", "Provide a non-empty problem."))

    known = parse_object.get("known_quantities") or {}
    domains = parse_object.get("domains") or []
    relations = parse_object.get("relations") or []
    if not known and not relations and (not domains or domains == ["unknown"]):
        # Nothing useful extracted at all — this is suspicious even for non-numeric.
        warnings.append(
            "Non-numeric problem with no extracted quantities, relations, or domain. "
            "Stage 0 has little structure to pass forward."
        )

    # Dimension sanity check still applies to anything that was extracted.
    for name, quantity in known.items():
        base = _base_name(name)
        expected = COMPATIBLE_DIMENSIONS.get(base)
        if not isinstance(quantity, dict):
            continue
        dimension = str(quantity.get("dimension"))
        if expected and dimension not in expected:
            errors.append(
                _err(
                    "wrong_unit",
                    f"{name} has dimension {dimension}, expected one of {sorted(expected)}.",
                    "Fix variable naming or unit disambiguation.",
                )
            )

    status = "PASS_NON_NUMERIC" if not errors else "FAIL"
    return VerifierResult(status=status, errors=errors, warnings=warnings)


def verify_parse(parse_object: Dict[str, object]) -> VerifierResult:
    """Run all parse verifier checks and return PASS / PASS_NON_NUMERIC / FAIL.

    Stage 0.4.1: when question_type != 'numeric_calc', this dispatches to a
    lightweight non-numeric verifier so that boolean and symbolic problems are
    not penalized for the absence of a numeric step plan.
    """
    question_type = parse_object.get("question_type") or "numeric_calc"
    if question_type != "numeric_calc":
        return _verify_non_numeric(parse_object)

    errors: List[VerifierError] = []
    warnings: List[str] = []
    text = str(parse_object.get("problem_text", ""))
    known: Dict[str, Dict[str, object]] = parse_object.get("known_quantities", {}) or {}
    relations: List[Dict[str, object]] = parse_object.get("relations", []) or []

    extracted_values = {str(quantity.get("value")) for quantity in known.values()}
    extracted_values.update(str(int(quantity["value"])) for quantity in known.values() if isinstance(quantity.get("value"), float) and float(quantity["value"]).is_integer())
    relation_values = {str(relation.get(key)) for relation in relations for key in ("factor", "value") if relation.get(key) is not None}
    ignored_numbers: List[str] = []

    # Pre-compute spans of contexts where numeric tokens should be ignored:
    #   * clock times: "7:30 AM", "6 PM", "8:15 AM", "4 o'clock"
    #   * resonance condition pivots: "LCω² = 1", "LC*omega^2 = 1"
    #   * scientific-notation tails picked up by NUMBER_RE alone
    #     (e.g. the '10' from '...10^-7 C' is part of a sci-notation literal
    #     that the extractor already accepted as a whole; we shouldn't report
    #     missing for the bare '10')
    ignore_spans: List[Tuple[int, int]] = []
    for pat in (
        r"\b\d{1,2}\s*:\s*\d{2}\s*(?:[AP]\.?M\.?|am|pm)?",  # 7:30 AM, 7:00, 10:00 PM
        r"\b\d{1,2}\s+(?:[AP]\.?M\.?|am|pm)\b",             # 6 AM, 8 PM
        r"\b\d{1,2}\s+o[\u2019']clock\b",                   # 4 o'clock, 5 o’clock
        r"LC\s*[\u03c9w][\u00b2\u00b3\^2]?\s*=\s*1",        # LCω² = 1, LCω^2 = 1
        r"\bLC\s*\*?\s*omega\s*\^?\s*2\s*=\s*1",
        r"[-+]?(?:\d+\.\d+|\d+|\.\d+)\s*(?:×|x|\*)\s*10\s*\^?\s*[-+]?[\d⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+",  # 4.10^-10, 2 × 10^5
        # Sequence labels: "Car 1", "Vehicle 2", "Option 1", "Question 3"
        r"\b(?:Car|Vehicle|Object|Sphere|Wagon|Truck|Train|Ship|Boat|Person|Body|Bus|Lamp|Block|Particle|Capacitor|Resistor|Inductor|Coil|Spring|Mass|Question|Option|Part|Problem|Exercise|Case|Scenario|Step|Figure|Example|Method|Solution|Sample|Test|Try)\s+\d+\b",
        r"\bM\d+\b|\bP\d+\b",          # M1, M2 (sphere labels)
        # Hyphenated compounds where digit is an attribute: "2-hour", "4-meter"
        r"\b\d+(?:\.\d+)?-(?:hour|hours|second|seconds|minute|minutes|day|days|meter|metre|meters|metres|cm|km|kg|gram|grams|liter|liters|litre|litres|year|years)\b",
        # Trig function arguments — angular frequency inside cos/sin (the
        # amplitude is the value before the function call, but we extract
        # those separately. The number inside the parens is the angular
        # frequency / phase and is already covered by relations).
        r"(?:cos|sin|tan|cot)\s*[\u00b2\u00b3\^2]?\s*\([^)]+\)",
        # μ₀ = 4π×10⁻⁷ (vacuum permeability constant declaration; the value
        # is a physical constant, not a problem-specific quantity).
        r"\u03bc[_\u2080\u2081\u20820]?\s*=\s*4\s*[\u03c0pi]+\s*[×x\*]\s*10\s*[\^\u207b\u00b9-]?\s*[\d\u207b\u00b9\u2070\u2074\u2075\u2076\u2077]*",
    ):
        for m in re.finditer(pat, text, re.IGNORECASE):
            ignore_spans.append(m.span())

    def _in_ignored_span(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in ignore_spans)

    for token in NUMBER_RE.finditer(text):
        value = token.group(0)
        start, end = token.span()
        before = text[max(0, token.start() - 1): token.start()]
        after = text[token.end(): min(len(text), token.end() + 2)]
        if before.isalpha():
            ignored_numbers.append(value)
            continue
        if before in {"^", "e", "E", "-"}:
            ignored_numbers.append(value)
            continue
        if after.startswith(".") or before == ".":
            ignored_numbers.append(value)
            continue
        if re.search(r"[A-Za-z_][A-Za-z_]*$", text[max(0, start - 3):start]) and value.isdigit():
            ignored_numbers.append(value)
            continue
        if re.match(r"^\s*\.", text[end: end + 2]) or re.match(r"^\s*[A-D][\.\)]", text[max(0, start - 3): end + 2]):
            ignored_numbers.append(value)
            continue
        if _in_ignored_span(start, end):
            ignored_numbers.append(value)
            continue
        # ':' on either side strongly suggests a clock time or a ratio
        # delimiter rather than a measurable quantity.
        if before == ":" or after.startswith(":"):
            ignored_numbers.append(value)
            continue
        if (
            value not in extracted_values
            and value not in relation_values
            and not any(value in str(quantity.get("source_text", "")) for quantity in known.values())
            and not any(value in str(relation.get("source_text", "")) for relation in relations)
        ):
            warnings.append(f"Numeric token {value} was not covered by known_quantities.")
            errors.append(_err("missing_quantity", f"Numeric token {value} was not extracted.", "Add or refine a unit/context extraction rule."))
    metadata = parse_object.get("metadata")
    if isinstance(metadata, dict):
        metadata["coverage_ignored_numbers"] = ignored_numbers[:50]

    for name, quantity in known.items():
        base = _base_name(name)
        expected = COMPATIBLE_DIMENSIONS.get(base)
        dimension = str(quantity.get("dimension"))
        if expected and dimension not in expected:
            errors.append(_err("wrong_unit", f"{name} has dimension {dimension}, expected one of {sorted(expected)}.", "Fix variable naming or unit disambiguation."))

    unknown = parse_object.get("unknown_quantity")
    detected_unknown, _ = detect_target(text)
    if unknown is None:
        errors.append(_err("missing_target", "unknown_quantity is null.", "Add a target detector phrase rule or semantic fallback."))
    elif detected_unknown and unknown != detected_unknown:
        errors.append(_err("target_mismatch", f"unknown_quantity is {unknown}, but deterministic target detector found {detected_unknown}.", "Align target phrase mapping with the question cue."))

    available: Set[str] = set(known) | CONSTANTS
    steps: List[Dict[str, object]] = parse_object.get("step_plan", []) or []
    for index, step in enumerate(steps, start=1):
        is_skeleton = step.get("template_name") == "skeleton_placeholder"
        if step.get("step_id") != f"step_{index}":
            errors.append(_err("invalid_dependency", f"Expected step_{index}, got {step.get('step_id')}.", "Renumber step ids sequentially."))
        if is_skeleton:
            # Skeleton stubs intentionally carry placeholder inputs/outputs
            # ('TBD' etc.). Skip dependency analysis so they don't generate
            # spurious invalid_dependency errors; low_confidence already
            # signals that this parse is not actually solved.
            available.update((step.get("output_var") or {}).keys())
            continue
        for var in (step.get("input_var") or {}).keys():
            if var not in available:
                errors.append(_err("invalid_dependency", f"Step {step.get('step_id')} depends on unavailable variable {var}.", "Use known quantities, constants, or earlier outputs only."))
        available.update((step.get("output_var") or {}).keys())

    if steps:
        skeleton_present = any(
            step.get("template_name") == "skeleton_placeholder" for step in steps
        )
        # An executable step is a real (non-skeleton) formula/calculation step.
        has_executable_step = any(
            step.get("type") in {"formula_application", "calculation"}
            and step.get("template_name") != "skeleton_placeholder"
            for step in (steps[:-1] or steps)
        )
        if not has_executable_step:
            if skeleton_present:
                # Skeleton placeholder is already in the plan. Don't double-
                # report invalid_final_step; low_confidence will fail the parse.
                warnings.append(
                    "Skeleton placeholder step_plan; real template missing."
                )
            else:
                errors.append(
                    _err(
                        "invalid_final_step",
                        "Step plan has no executable formula or calculation step.",
                        "Add a real formula_application/calculation step before conclusion.",
                    )
                )
        final_step = steps[-1]
        outputs = final_step.get("output_var") or {}
        if final_step.get("type") != "conclusion" or (unknown and unknown not in outputs):
            errors.append(_err("invalid_final_step", "Last step is not a conclusion for unknown_quantity.", "Append a conclusion step whose output_var includes unknown_quantity."))
    elif unknown:
        if str(unknown).startswith("relation_") or unknown == "equation_of_motion":
            warnings.append(f"{unknown} is a conceptual relationship target; a lightweight relation plan is acceptable.")
        else:
            errors.append(_err("invalid_final_step", "No step plan was produced.", "Add a template or fallback step plan."))
            warnings.append("No executable step_plan generated; template coverage missing.")

    used_llm = bool((parse_object.get("metadata") or {}).get("used_llm_fallback"))
    if float(parse_object.get("domain_confidence") or 0.0) < 0.5 and not used_llm:
        errors.append(_err("low_confidence", "domain_confidence is below 0.5.", "Improve domain keywords or use fallback recovery."))
    if float(parse_object.get("plan_confidence") or 0.0) < 0.5 and not used_llm:
        errors.append(_err("low_confidence", "plan_confidence is below 0.5.", "Add a matching formula template or use fallback recovery."))

    return VerifierResult(status="PASS" if not errors else "FAIL", errors=errors, warnings=warnings)