"""Analyze Stage 0 verifier failures and suggest parser rule expansions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


CUE_PHRASES = [
    "find",
    "calculate",
    "determine",
    "compute",
    "what is",
    "what are",
    "how much is",
    "how many",
    "evaluate",
    "derive",
]

TARGET_NOUN_PHRASES = [
    "percentage relative error",
    "relative error",
    "absolute error",
    "potential difference",
    "equivalent capacitance",
    "equivalent resistance",
    "electric field strength",
    "electric field",
    "electric force",
    "coulomb force",
    "resultant force",
    "net force",
    "force acting on",
    "energy stored",
    "magnetic field",
    "magnetic flux",
    "induced emf",
    "electromotive force",
    "resonant frequency",
    "current",
    "voltage",
    "resistance",
    "power",
    "charge",
    "capacitance",
    "inductance",
    "speed",
    "velocity",
    "acceleration",
    "distance",
    "displacement",
    "separation",
    "time",
    "wavelength",
    "frequency",
    "period",
    "angle",
    "direction",
]

TARGET_MAPPING = [
    (["equivalent resistance"], "R_eq"),
    (["equivalent capacitance"], "C_eq"),
    (["percentage relative error", "relative error"], "rel_error"),
    (["absolute error"], "abs_error"),
    (["charge stored"], "Q"),
    (["find q", "charge"], "q"),
    (["potential difference", "voltage"], "V"),
    (["current"], "I"),
    (["resistance"], "R"),
    (["power"], "P"),
    (["capacitance"], "C_cap"),
    (["electric field", "field strength"], "E"),
    (["force acting on q3"], "F_on_q3"),
    (["electric force", "coulomb force"], "F_e"),
    (["resultant force", "net force"], "F_net"),
    (["energy stored"], "U_cap"),
    (["magnetic field"], "B"),
    (["magnetic flux"], "Phi_B"),
    (["induced emf", "electromotive force"], "emf"),
    (["inductance"], "L_ind"),
    (["resonant frequency"], "f_res"),
    (["speed", "velocity"], "v"),
    (["acceleration"], "a"),
    (["distance", "separation"], "r"),
    (["wavelength"], "lambda"),
    (["frequency"], "f"),
    (["period"], "T_period"),
    (["angle", "direction"], "theta"),
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def infer_error_type(error: Dict[str, Any]) -> str:
    """Infer a verifier error type when the field is missing."""
    explicit = error.get("error_type")
    if explicit:
        return str(explicit)
    text = " ".join(str(error.get(key, "")) for key in ("description", "repair_hint", "message", "error")).lower()
    if "target" in text:
        return "target_mismatch" if "mismatch" in text else "missing_target"
    if "final step" in text or "conclusion" in text:
        return "invalid_final_step"
    if "quantity" in text or "numeric" in text:
        return "missing_quantity"
    if "confidence" in text:
        return "low_confidence"
    if "dependency" in text:
        return "invalid_dependency"
    return "unknown"


def normalize_error(error: Any) -> Dict[str, str]:
    """Normalize an error object to error_type, description, and repair_hint."""
    if isinstance(error, str):
        raw = {"description": error}
    elif isinstance(error, dict):
        raw = error
    else:
        raw = {"description": str(error)}
    return {
        "error_type": infer_error_type(raw),
        "description": str(raw.get("description") or raw.get("message") or raw.get("error") or ""),
        "repair_hint": str(raw.get("repair_hint") or raw.get("hint") or ""),
    }


def extract_errors(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract verifier errors from all known Stage 0 record layouts."""
    candidates = [
        _as_dict(record.get("verifier_result")).get("errors"),
        _as_dict(record.get("metadata")).get("verifier_errors"),
        _as_dict(_as_dict(record.get("parse_object")).get("metadata")).get("verifier_errors"),
        _as_dict(_as_dict(record.get("parse")).get("metadata")).get("verifier_errors"),
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [normalize_error(error) for error in candidate]
    return []


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw JSONL record to problem_text, parse, errors, and metadata."""
    parse = _as_dict(record.get("parse") or record.get("parse_object"))
    metadata = _as_dict(record.get("metadata")) or _as_dict(parse.get("metadata"))
    problem_text = (
        record.get("problem_text")
        or record.get("question")
        or parse.get("problem_text")
        or ""
    )
    return {
        "problem_text": str(problem_text),
        "parse": parse,
        "errors": extract_errors(record),
        "metadata": metadata,
        "raw": record,
    }


def load_failure_records(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Read JSONL failure records, skipping malformed lines and counting them."""
    records: List[Dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(raw, dict):
                malformed += 1
                continue
            records.append(normalize_record(raw))
    return records, malformed


def clean_phrase(text: str) -> str:
    """Normalize extracted phrase text for counting."""
    text = re.sub(r"[\?\.,;:\)\(\[\]\"]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_question_phrases(problem_text: str) -> List[str]:
    """Extract cue-following question phrases and known target noun phrases."""
    lowered = problem_text.lower()
    phrases: List[str] = []
    for cue in CUE_PHRASES:
        for match in re.finditer(rf"\b{re.escape(cue)}\b", lowered):
            after = lowered[match.end():]
            words = re.findall(r"[a-zA-Z0-9_′']+", after)[:12]
            if words:
                phrases.append(clean_phrase(" ".join(words)))
    for phrase in TARGET_NOUN_PHRASES:
        if phrase in lowered:
            phrases.append(phrase)
    force_match = re.search(r"\bforce acting on\s+(q\d+|q[abc]?|charge at [a-z])", lowered)
    if force_match:
        phrases.append(clean_phrase(force_match.group(0)))
    find_q = re.search(r"\bfind\s+(q\d*|q[abc]?|charge)\b", lowered)
    if find_q:
        phrases.append(clean_phrase(find_q.group(0)))
    return [phrase for phrase in phrases if phrase]


def has_error(record: Dict[str, Any], *error_types: str) -> bool:
    wanted = set(error_types)
    return any(error.get("error_type") in wanted for error in record["errors"])


def step_plan_summary(step_plan: Any) -> List[Dict[str, Any]]:
    """Create a compact, JSON-safe summary of a step plan."""
    summary: List[Dict[str, Any]] = []
    for step in _as_list(step_plan)[:5]:
        if not isinstance(step, dict):
            continue
        summary.append(
            {
                "step_id": step.get("step_id"),
                "type": step.get("type"),
                "inputs": list(_as_dict(step.get("input_var")).keys()),
                "outputs": list(_as_dict(step.get("output_var")).keys()),
            }
        )
    return summary


def classify_invalid_final_step(parse: Dict[str, Any]) -> str:
    """Classify the likely cause of an invalid_final_step error."""
    unknown = parse.get("unknown_quantity")
    if unknown is None or str(unknown).strip() == "":
        return "missing_unknown_quantity"
    step_plan = parse.get("step_plan")
    if not isinstance(step_plan, list) or not step_plan:
        return "empty_step_plan"
    ids = [step.get("step_id") for step in step_plan if isinstance(step, dict)]
    expected = [f"step_{index}" for index in range(1, len(step_plan) + 1)]
    if ids != expected or len(set(ids)) != len(ids):
        return "invalid_step_ids"
    last = step_plan[-1] if isinstance(step_plan[-1], dict) else {}
    if last.get("type") != "conclusion":
        return "last_step_not_conclusion"
    if str(unknown) not in _as_dict(last.get("output_var")):
        return "conclusion_missing_target"
    return "unknown_other"


MISSING_QUANTITY_PATTERNS = {
    "angle_degree": re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:°|degrees?\b|deg\b)", re.IGNORECASE),
    "scientific_notation": re.compile(
        r"[-+]?\d+(?:\.\d+)?\s*(?:×|x|\*)\s*10\s*(?:\^\s*[-+]?\d+|[⁻-]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)|[-+]?\d+(?:\.\d+)?e[-+]?\d+",
        re.IGNORECASE,
    ),
    "symbolic_ratio": re.compile(r"\b[A-Za-z]\w*\s*=\s*[-+]?\d+(?:\.\d+)?\s*[A-Za-z]\w*\b"),
    "word_ratio": re.compile(r"\b(?:twice|half|three times|one third|double|triple)\b", re.IGNORECASE),
    "percentage": re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE),
    "fraction": re.compile(r"\b\d+\s*/\s*\d+\b"),
    "range_or_pair": re.compile(r"\bfrom\s+[-+]?\d+(?:\.\d+)?\s*\w*\s+to\s+[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
    "coordinate": re.compile(r"\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)"),
}


def extract_missing_quantity_patterns(problem_text: str) -> Dict[str, List[str]]:
    """Find likely missed numeric expression types in problem text."""
    matches: Dict[str, List[str]] = {}
    for pattern_type, pattern in MISSING_QUANTITY_PATTERNS.items():
        found = [match.group(0) for match in pattern.finditer(problem_text)]
        if found:
            matches[pattern_type] = found[:10]
    return matches


def suggest_unknown_quantity(phrase: str, problem_text: str = "") -> Optional[str]:
    """Map a target phrase to a suggested unknown symbol."""
    combined = f"{phrase} {problem_text}".lower()
    if "energy stored" in combined and "capacitor" in combined:
        return "U_cap"
    for triggers, target in TARGET_MAPPING:
        if any(trigger in combined for trigger in triggers):
            return target
    return None


def analyze_missing_targets(records: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    """Analyze missing_target and target_mismatch records."""
    phrase_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    known_name_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    examples: List[Dict[str, Any]] = []

    for record in records:
        if not has_error(record, "missing_target", "target_mismatch"):
            continue
        parse = record["parse"]
        known = _as_dict(parse.get("known_quantities"))
        phrases = extract_question_phrases(record["problem_text"])
        phrase_counts.update(phrases or ["<no_phrase_found>"])
        known_name_counts.update(known.keys())
        unit_counts.update(str(quantity.get("unit_symbol")) for quantity in known.values() if isinstance(quantity, dict) and quantity.get("unit_symbol"))
        domain_counts.update(str(domain) for domain in _as_list(parse.get("domains")) or ["unknown"])
        if len(examples) < 20:
            examples.append(
                {
                    "problem_text": record["problem_text"],
                    "domains": parse.get("domains", []),
                    "sub_domains": parse.get("sub_domains", []),
                    "known_quantity_names": list(known.keys()),
                    "known_unit_symbols": [quantity.get("unit_symbol") for quantity in known.values() if isinstance(quantity, dict)],
                    "current_unknown_quantity": parse.get("unknown_quantity"),
                    "question_phrase_candidates": phrases,
                    "errors": record["errors"],
                }
            )

    return {
        "count": sum(1 for record in records if has_error(record, "missing_target", "target_mismatch")),
        "top_phrases": phrase_counts.most_common(top_k),
        "top_units": unit_counts.most_common(top_k),
        "top_known_quantity_names": known_name_counts.most_common(top_k),
        "domain_distribution": domain_counts.most_common(),
        "examples": examples,
    }


def analyze_invalid_final_steps(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze invalid_final_step records and classify causes."""
    cause_counts: Counter[str] = Counter()
    examples_by_cause: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in records:
        if not has_error(record, "invalid_final_step"):
            continue
        parse = record["parse"]
        cause = classify_invalid_final_step(parse)
        cause_counts[cause] += 1
        if len(examples_by_cause[cause]) < 10:
            examples_by_cause[cause].append(
                {
                    "problem_text": record["problem_text"],
                    "unknown_quantity": parse.get("unknown_quantity"),
                    "step_plan_summary": step_plan_summary(parse.get("step_plan")),
                    "verifier_errors": record["errors"],
                }
            )

    return {
        "count": sum(cause_counts.values()),
        "cause_counts": cause_counts.most_common(),
        "examples_by_cause": dict(examples_by_cause),
    }


def analyze_missing_quantities(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze missing_quantity records for missed numeric expression patterns."""
    pattern_counts: Counter[str] = Counter()
    examples_by_pattern: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in records:
        if not has_error(record, "missing_quantity"):
            continue
        matches = extract_missing_quantity_patterns(record["problem_text"])
        if not matches:
            pattern_counts["unclassified_numeric"] += 1
            if len(examples_by_pattern["unclassified_numeric"]) < 10:
                examples_by_pattern["unclassified_numeric"].append(
                    {
                        "problem_text": record["problem_text"],
                        "matches": [],
                        "known_quantity_names": list(_as_dict(record["parse"].get("known_quantities")).keys()),
                        "errors": record["errors"],
                    }
                )
            continue
        for pattern_type, found in matches.items():
            pattern_counts[pattern_type] += 1
            if len(examples_by_pattern[pattern_type]) < 10:
                examples_by_pattern[pattern_type].append(
                    {
                        "problem_text": record["problem_text"],
                        "matches": found,
                        "known_quantity_names": list(_as_dict(record["parse"].get("known_quantities")).keys()),
                        "errors": record["errors"],
                    }
                )

    return {
        "count": sum(1 for record in records if has_error(record, "missing_quantity")),
        "pattern_counts": pattern_counts.most_common(),
        "examples_by_pattern": dict(examples_by_pattern),
    }


def _contains_all(names: Iterable[str], required: Iterable[str]) -> bool:
    names_set = set(names)
    return all(name in names_set for name in required)


def generate_rule_suggestions(
    records: List[Dict[str, Any]],
    missing_target: Dict[str, Any],
    invalid_final_step: Dict[str, Any],
    missing_quantity: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    """Generate actionable rule, template, and extractor suggestions."""
    target_detector_suggestions = []
    for phrase, count in missing_target["top_phrases"]:
        if phrase == "<no_phrase_found>":
            continue
        suggested = suggest_unknown_quantity(phrase)
        if suggested:
            target_detector_suggestions.append(
                {
                    "phrase": phrase,
                    "suggested_unknown_quantity": suggested,
                    "count": count,
                    "reason": "Frequent missing_target phrase.",
                }
            )
        if len(target_detector_suggestions) >= top_k:
            break

    template_counter: Counter[str] = Counter()
    for record in records:
        if not has_error(record, "invalid_final_step", "low_confidence"):
            continue
        parse = record["parse"]
        known = _as_dict(parse.get("known_quantities"))
        known_names = set(known)
        target = parse.get("unknown_quantity")
        text = record["problem_text"].lower()
        if _contains_all(known_names, ["V", "R"]) and target == "I":
            template_counter["ohms_law_current"] += 1
        if _contains_all(known_names, ["I", "R"]) and target == "P":
            template_counter["power_from_current_resistance"] += 1
        if _contains_all(known_names, ["V", "I"]) and target == "P":
            template_counter["power_from_voltage_current"] += 1
        if _contains_all(known_names, ["C_cap", "V"]) and target == "Q":
            template_counter["capacitor_charge"] += 1
        if _contains_all(known_names, ["C_cap", "V"]) and target == "U_cap":
            template_counter["capacitor_energy"] += 1
        charge_names = [name for name, quantity in known.items() if isinstance(quantity, dict) and quantity.get("dimension") == "charge"]
        distance_names = [name for name, quantity in known.items() if isinstance(quantity, dict) and quantity.get("dimension") == "length"]
        if len(charge_names) >= 2 and distance_names and target == "F_e":
            template_counter["coulomb_force_scalar"] += 1
        if len(charge_names) >= 3 and len(distance_names) >= 2 and ("q3" in text or "resultant force" in text or "net force" in text):
            template_counter["coulomb_force_vector"] += 1
        if _contains_all(known_names, ["f", "lambda"]) and target in {"v", "v_wave"}:
            template_counter["wave_speed"] += 1
        if _contains_all(known_names, ["v_0", "a", "t"]) and target in {"v", "v_final"}:
            template_counter["kinematics_final_velocity"] += 1
        if _contains_all(known_names, ["v", "t"]) and target in {"d", "r"}:
            template_counter["constant_speed_distance"] += 1

    template_details = {
        "ohms_law_current": {
            "evidence": "Records contain V and R with target current.",
            "suggested_steps": ["I = V / R", "Report I"],
        },
        "power_from_current_resistance": {
            "evidence": "Records contain I and R with target power.",
            "suggested_steps": ["P = I^2 * R", "Report P"],
        },
        "power_from_voltage_current": {
            "evidence": "Records contain V and I with target power.",
            "suggested_steps": ["P = V * I", "Report P"],
        },
        "capacitor_charge": {
            "evidence": "Records contain C_cap and V with target charge.",
            "suggested_steps": ["Q = C * V", "Report Q"],
        },
        "capacitor_energy": {
            "evidence": "Records contain C_cap and V with target capacitor energy.",
            "suggested_steps": ["U = 0.5 * C * V^2", "Report U"],
        },
        "coulomb_force_scalar": {
            "evidence": "Records contain at least two charges and a distance with target electric force.",
            "suggested_steps": ["F = k * abs(q1*q2) / r^2", "Report F"],
        },
        "coulomb_force_vector": {
            "evidence": "Records contain q1/q2/q3 and multiple distances with target/resultant force.",
            "suggested_steps": ["Compute pairwise Coulomb forces", "Resolve vector components", "Report resultant force"],
        },
        "wave_speed": {
            "evidence": "Records contain f and lambda with target speed.",
            "suggested_steps": ["v = f * lambda", "Report v"],
        },
        "kinematics_final_velocity": {
            "evidence": "Records contain v_0, a, and t with target velocity.",
            "suggested_steps": ["v = v_0 + a*t", "Report v"],
        },
        "constant_speed_distance": {
            "evidence": "Records contain v and t with target distance.",
            "suggested_steps": ["d = v*t", "Report d"],
        },
    }
    template_fallback_suggestions = []
    for template_name, count in template_counter.most_common(top_k):
        detail = template_details[template_name]
        template_fallback_suggestions.append(
            {
                "template_name": template_name,
                "evidence": detail["evidence"],
                "count": count,
                "suggested_steps": detail["suggested_steps"],
            }
        )

    extractor_suggestions = []
    pattern_suggestion_text = {
        "angle_degree": "Add angle extraction with normalized radians.",
        "scientific_notation": "Add robust scientific notation extraction before plain number-unit matching.",
        "symbolic_ratio": "Add symbolic ratio extraction for forms like q1 = 4q2.",
        "word_ratio": "Add word-ratio extraction for twice/half/three times.",
        "percentage": "Add percentage extraction and normalized fraction value.",
        "fraction": "Add fraction extraction in quantity contexts.",
        "range_or_pair": "Add range extraction and preserve start/end quantities.",
        "coordinate": "Add coordinate tuple extraction for geometry/vector problems.",
        "unclassified_numeric": "Inspect unclassified missing numeric examples for new regex families.",
    }
    for pattern_type, count in missing_quantity["pattern_counts"]:
        examples = []
        for example in missing_quantity["examples_by_pattern"].get(pattern_type, [])[:2]:
            examples.extend(example.get("matches", []))
        extractor_suggestions.append(
            {
                "pattern_type": pattern_type,
                "examples": examples[:5],
                "count": count,
                "suggestion": pattern_suggestion_text.get(pattern_type, "Add or refine quantity extraction for this pattern."),
            }
        )

    final_step_suggestions = [
        {
            "cause": cause,
            "count": count,
            "suggestion": (
                "Add ensure_conclusion_step() and default setup/formula/conclusion skeleton when unknown_quantity exists."
                if cause == "empty_step_plan"
                else "Add or repair final conclusion step so output_var includes unknown_quantity."
                if cause == "conclusion_missing_target"
                else "Improve target detection before step planning."
                if cause == "missing_unknown_quantity"
                else "Normalize step ids sequentially before verification."
                if cause == "invalid_step_ids"
                else "Inspect examples and add a targeted step-plan repair."
            ),
        }
        for cause, count in invalid_final_step["cause_counts"]
    ]

    return {
        "target_detector_suggestions": target_detector_suggestions,
        "template_fallback_suggestions": template_fallback_suggestions,
        "quantity_extractor_suggestions": extractor_suggestions,
        "final_step_suggestions": final_step_suggestions,
    }


def _clean_display_text(text: str) -> str:
    """Clean common mojibake only in human-readable analysis reports."""
    replacements = {
        "Â°": "°",
        "Î¼": "μ",
        "Ã—": "×",
        "âˆ’": "−",
        "â€“": "−",
        "â‚": "₁",
        "â‚‚": "₂",
        "Ï€": "π",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def write_missing_target_report(path: Path, analysis: Dict[str, Any]) -> None:
    """Write a text report for missing target examples."""
    lines = ["Top Missing Target Phrases", ""]
    for phrase, count in analysis["top_phrases"]:
        lines.append(f"{count:5d}  {phrase}")
    lines.extend(["", "Representative Examples", ""])
    for index, example in enumerate(analysis["examples"], start=1):
        lines.append(f"{index}. {example['problem_text']}")
        lines.append(f"   domains: {example['domains']}")
        lines.append(f"   known: {example['known_quantity_names']}")
        lines.append(f"   phrases: {example['question_phrase_candidates']}")
        lines.append("")
    path.write_text(_clean_display_text("\n".join(lines)), encoding="utf-8")


def write_invalid_final_step_report(path: Path, analysis: Dict[str, Any]) -> None:
    """Write a text report for invalid final step examples."""
    lines = ["Invalid Final Step Cause Counts", ""]
    for cause, count in analysis["cause_counts"]:
        lines.append(f"{count:5d}  {cause}")
    for cause, examples in analysis["examples_by_cause"].items():
        lines.extend(["", f"Examples: {cause}", ""])
        for index, example in enumerate(examples, start=1):
            lines.append(f"{index}. {example['problem_text']}")
            lines.append(f"   unknown: {example['unknown_quantity']}")
            lines.append(f"   steps: {example['step_plan_summary']}")
            lines.append("")
    path.write_text(_clean_display_text("\n".join(lines)), encoding="utf-8")


def write_missing_quantity_report(path: Path, analysis: Dict[str, Any]) -> None:
    """Write a text report for missing quantity patterns."""
    lines = ["Missing Quantity Pattern Counts", ""]
    for pattern_type, count in analysis["pattern_counts"]:
        lines.append(f"{count:5d}  {pattern_type}")
    for pattern_type, examples in analysis["examples_by_pattern"].items():
        lines.extend(["", f"Examples: {pattern_type}", ""])
        for index, example in enumerate(examples, start=1):
            lines.append(f"{index}. {example['problem_text']}")
            lines.append(f"   matches: {example['matches']}")
            lines.append(f"   known: {example['known_quantity_names']}")
            lines.append("")
    path.write_text(_clean_display_text("\n".join(lines)), encoding="utf-8")


def analyze_failures(input_path: Path, output_dir: Path, top_k: int) -> Dict[str, Any]:
    """Analyze Stage 0 failure JSONL and write all output artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records, malformed = load_failure_records(input_path)

    error_counts: Counter[str] = Counter()
    for record in records:
        error_counts.update(error["error_type"] for error in record["errors"])

    missing_target = analyze_missing_targets(records, top_k)
    invalid_final_step = analyze_invalid_final_steps(records)
    missing_quantity = analyze_missing_quantities(records)
    suggestions = generate_rule_suggestions(records, missing_target, invalid_final_step, missing_quantity, top_k)

    failure_analysis = {
        "input_path": str(input_path),
        "total_records_analyzed": len(records),
        "malformed_lines_skipped": malformed,
        "error_type_counts": error_counts.most_common(),
        "missing_target_analysis": missing_target,
        "invalid_final_step_analysis": invalid_final_step,
        "missing_quantity_analysis": missing_quantity,
    }

    analysis_path = output_dir / "failure_analysis.json"
    suggestions_path = output_dir / "rule_suggestions.json"
    missing_target_path = output_dir / "missing_target_examples.txt"
    invalid_final_step_path = output_dir / "invalid_final_step_examples.txt"
    missing_quantity_path = output_dir / "missing_quantity_examples.txt"

    analysis_path.write_text(json.dumps(failure_analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    suggestions_path.write_text(json.dumps(suggestions, indent=2, ensure_ascii=False), encoding="utf-8")
    write_missing_target_report(missing_target_path, missing_target)
    write_invalid_final_step_report(invalid_final_step_path, invalid_final_step)
    write_missing_quantity_report(missing_quantity_path, missing_quantity)

    return {
        "summary": {
            "total_records_analyzed": len(records),
            "malformed_lines_skipped": malformed,
            "error_type_counts": error_counts.most_common(),
            "missing_target_top_10_phrases": missing_target["top_phrases"][:10],
            "invalid_final_step_cause_counts": invalid_final_step["cause_counts"],
            "missing_quantity_pattern_counts": missing_quantity["pattern_counts"],
        },
        "outputs": {
            "failure_analysis": str(analysis_path),
            "missing_target_examples": str(missing_target_path),
            "invalid_final_step_examples": str(invalid_final_step_path),
            "missing_quantity_examples": str(missing_quantity_path),
            "rule_suggestions": str(suggestions_path),
        },
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Analyze Stage 0 parser failure logs.")
    parser.add_argument("--input", default="outputs/stage0/stage0_failures.jsonl")
    parser.add_argument("--output-dir", default="outputs/stage0/analysis")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    result = analyze_failures(Path(args.input), Path(args.output_dir), args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
