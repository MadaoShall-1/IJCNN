"""Failure clustering and pattern mining for Stage 0 refinement."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple


CUES = [
    "find",
    "calculate",
    "determine",
    "compute",
    "what is",
    "what are",
    "how much",
    "how many",
    "evaluate",
    "derive",
    "compare",
    "show that",
]

PHYSICS_NOUNS = [
    "equivalent resistance",
    "equivalent capacitance",
    "electric field strength",
    "electric field",
    "electric force",
    "resultant force",
    "force acting on",
    "energy stored",
    "relative uncertainty",
    "percentage uncertainty",
    "equation of motion",
    "resonant frequency",
    "magnetic flux",
    "induced emf",
    "net force",
    "charge",
    "voltage",
    "current",
    "power",
    "resistance",
    "capacitance",
    "magnetic field",
    "velocity",
    "speed",
    "distance",
    "displacement",
    "direction",
    "angle",
    "uncertainty",
    "ratio",
]

TARGET_DIMENSIONS = {
    "I": "current",
    "I_rms": "current",
    "I_max": "current",
    "V": "voltage",
    "U_V": "voltage",
    "V_after": "voltage",
    "R": "resistance",
    "R_eq": "resistance",
    "C_cap": "capacitance",
    "C_eq": "capacitance",
    "C_after": "capacitance",
    "E": "electric_field",
    "F_e": "force",
    "F_net": "force",
    "F_on_q3": "force",
    "U_cap": "energy",
    "U_B": "energy",
    "U_E": "energy",
    "KE": "energy",
    "PE": "energy",
    "E_energy": "energy",
    "f": "frequency",
    "f_res": "frequency",
    "omega": "angular_frequency",
    "lambda": "length",
    "theta": "angle",
    "direction": "angle",
    "phi": "angle",
    "ratio": "dimensionless",
    "percent_error": "dimensionless",
    "rel_error": "dimensionless",
}


PATTERN_REGEXES = {
    "angle_degree": re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:°|Â°|degrees?\b|deg\b)", re.IGNORECASE),
    "scientific_notation": re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:e[-+]?\d+|[×x*·]\s*10\s*(?:\^?\{?[-+]?\d+\}?|[⁻-]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))", re.IGNORECASE),
    "fraction_numeric": re.compile(r"\b-?\d+\s*/\s*\d+\b"),
    "fraction_word": re.compile(r"\b(?:half|one half|one third|two thirds|quarter|three quarters)\b", re.IGNORECASE),
    "percentage": re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:%|percent\b|per cent\b)", re.IGNORECASE),
    "uncertainty": re.compile(r"(?:±|Â±|\+/-|plus or minus|uncertainty)", re.IGNORECASE),
    "function": re.compile(r"\b(?:I|U|V|q|Q|x|d|s|v|a|E|B)\s*\(\s*t\s*\)\s*=", re.IGNORECASE),
    "equation": re.compile(r"\b[A-Za-z][A-Za-z0-9_]*(?:\s*[+*/-]\s*[A-Za-z0-9_]+)+\s*=\s*[A-Za-z0-9_]+"),
    "ratio": re.compile(r"\b[A-Za-z]\w*\s*=\s*[-+]?\d+(?:\.\d+)?\s*[A-Za-z]\w*|\b(?:twice|three times|four times|half)\b", re.IGNORECASE),
    "range": re.compile(r"\bfrom\s+[-+]?\d+(?:\.\d+)?\s*\w*\s+to\s+[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
    "coordinate": re.compile(r"\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)"),
}


def infer_target_dimension(target: Any, unit: Any = None) -> str:
    if target is None:
        return "unknown"
    text = str(target)
    if re.match(r"q\d*$|Q(?:_after|_max)?$", text):
        return "charge"
    return TARGET_DIMENSIONS.get(text, "unknown")


def primary_error_type(record: Dict[str, Any]) -> str:
    errors = record.get("errors") or []
    if errors:
        return str(errors[0].get("error_type") or "unknown")
    return "unknown"


def known_dim_tuple(record: Dict[str, Any]) -> str:
    dims = [str(value.get("dimension", "unknown")) for value in (record.get("known_quantities") or {}).values() if isinstance(value, dict)]
    return "|".join(sorted(dims)) or "none"


def relation_type_tuple(record: Dict[str, Any]) -> str:
    types = sorted(str(relation.get("type", "unknown")) for relation in record.get("relations", []) if isinstance(relation, dict))
    return "|".join(types) or "none"


def cluster_key(record: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    parse = record.get("parse") or {}
    domains = parse.get("domains") or ["unknown"]
    sub_domains = parse.get("sub_domains") or ["unknown"]
    target = record.get("unknown_quantity")
    metadata = record.get("metadata") or {}
    fields = {
        "question_type": record.get("question_type") or "unknown",
        "primary_error_type": primary_error_type(record),
        "domain": str(domains[0] if domains else "unknown"),
        "sub_domain": str(sub_domains[0] if sub_domains else "unknown"),
        "known_dim_tuple": known_dim_tuple(record),
        "target_dim": infer_target_dimension(target, record.get("unknown_unit")),
        "target_quantity": str(target) if target is not None else "null",
        "relation_type_tuple": relation_type_tuple(record),
        "template_used": bool(metadata.get("used_template_fallback")),
    }
    cluster_id = (
        f"{fields['question_type']}::{fields['primary_error_type']}::"
        f"{fields['domain']}/{fields['sub_domain']}::{fields['known_dim_tuple']}->"
        f"{fields['target_dim']}:{fields['target_quantity']}::"
        f"relations={fields['relation_type_tuple']}::template={fields['template_used']}"
    )
    return fields, cluster_id


def extract_target_phrases(text: str) -> List[str]:
    lowered = text.lower()
    phrases: List[str] = []
    for cue in CUES:
        index = lowered.find(cue)
        if index >= 0:
            after = re.sub(r"[^a-zA-Z0-9_%/+\- ]+", " ", lowered[index + len(cue):])
            words = after.split()[:12]
            if words:
                phrases.append(" ".join(words))
    for noun in PHYSICS_NOUNS:
        if noun in lowered:
            phrases.append(noun)
    return phrases[:20]


def mine_patterns(text: str) -> Dict[str, int]:
    return {name: len(regex.findall(text)) for name, regex in PATTERN_REGEXES.items() if regex.search(text)}


def mine_relation_opportunities(text: str) -> Dict[str, int]:
    lowered = text.lower()
    found = mine_patterns(text)
    opportunities = {name: count for name, count in found.items() if name in {"function", "equation", "uncertainty", "ratio", "percentage"}}
    if any(word in lowered for word in ("meet after", "catch up", "overtake", "downstream", "upstream")):
        opportunities["motion_language"] = opportunities.get("motion_language", 0) + 1
    return opportunities


def infer_template_candidates(record: Dict[str, Any], phrases: Iterable[str]) -> List[str]:
    dims = known_dim_tuple(record)
    target = str(record.get("unknown_quantity") or "")
    text = " ".join(phrases).lower() + " " + record.get("problem_text", "").lower()
    candidates: List[str] = []
    if "voltage" in dims and "resistance" in dims or target in {"I", "V", "R"}:
        candidates.append("ohms_law")
    if "power" in text or target in {"P", "P_total"}:
        candidates.append("power")
    if "parallel" in text and "resistance" in text:
        candidates.append("parallel_resistance")
    if "series" in text and "resistance" in text:
        candidates.append("series_resistance")
    if "capacitance" in dims and "voltage" in dims:
        candidates.extend(["capacitor_charge", "capacitor_energy"])
    if "parallel plate" in text:
        candidates.append("parallel_plate_capacitance")
    if "charge" in dims and "length" in dims and target in {"F_e", "F_net", "F_on_q3"}:
        candidates.append("coulomb_scalar_force")
    if target in {"F_net", "F_on_q3"}:
        candidates.append("coulomb_vector_force")
    if target == "E":
        candidates.append("electric_field_superposition")
    if "frequency" in dims and "length" in dims:
        candidates.append("wave_speed")
    if target in {"d", "v", "t"}:
        candidates.append("kinematics_constant_speed")
    if "acceleration" in dims:
        candidates.append("kinematics_uniform_acceleration")
    if "toward each other" in text:
        candidates.append("relative_motion_meeting")
    if "downstream" in text or "upstream" in text:
        candidates.append("downstream_upstream")
    if "uncertainty" in text:
        candidates.append("measurement_uncertainty")
    if "function" in relation_type_tuple(record):
        candidates.append("function_integration_current_to_charge")
    if "equation" in relation_type_tuple(record):
        candidates.append("equation_system_solve")
    return sorted(set(candidates))


def _example(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_id": record.get("dataset_id"),
        "row_index": record.get("row_index"),
        "problem_text": record.get("problem_text", ""),
        "unknown_quantity": record.get("unknown_quantity"),
        "known_quantity_names": sorted((record.get("known_quantities") or {}).keys()),
        "conditions": record.get("conditions") or [],
        "relation_types": [relation.get("type") for relation in record.get("relations", [])],
        "errors": record.get("errors") or [],
    }


def cluster_failures(records: List[Dict[str, Any]], top_k: int = 50) -> List[Dict[str, Any]]:
    buckets: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    fields_by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        fields, cid = cluster_key(record)
        fields_by_id[cid] = fields
        buckets[cid].append(record)

    clusters: List[Dict[str, Any]] = []
    for cid, items in buckets.items():
        fields = fields_by_id[cid]
        error_counter: Counter[str] = Counter(error.get("error_type", "unknown") for item in items for error in item.get("errors", []))
        relation_counter: Counter[str] = Counter(relation.get("type", "unknown") for item in items for relation in item.get("relations", []))
        phrase_counter: Counter[str] = Counter()
        pattern_counter: Counter[str] = Counter()
        template_candidate_counter: Counter[str] = Counter()
        template_name_counter: Counter[str] = Counter()
        for item in items:
            phrases = extract_target_phrases(item.get("problem_text", ""))
            phrase_counter.update(phrases)
            pattern_counter.update(mine_patterns(item.get("problem_text", "")))
            template_candidate_counter.update(infer_template_candidates(item, phrases))
            template_name_counter.update(str(name) for name in (item.get("metadata") or {}).get("used_template_names", []) or [])
        clusters.append(
            {
                "cluster_id": cid,
                "count": len(items),
                **fields,
                "error_type_counts": dict(error_counter.most_common()),
                "target_counts": dict(Counter(str(item.get("unknown_quantity")) for item in items).most_common()),
                "unit_counts": dict(Counter(str(item.get("unknown_unit")) for item in items).most_common()),
                "relation_type_counts": dict(relation_counter.most_common()),
                "template_name_counts": dict(template_name_counter.most_common()),
                "top_missing_number_patterns": dict(pattern_counter.most_common()),
                "top_question_phrases": dict(phrase_counter.most_common()),
                "template_candidates": dict(template_candidate_counter.most_common()),
                "examples": [_example(item) for item in items[:20]],
            }
        )
    clusters.sort(key=lambda cluster: (-int(cluster["count"]), cluster["cluster_id"]))
    return clusters[:top_k] if top_k else clusters

