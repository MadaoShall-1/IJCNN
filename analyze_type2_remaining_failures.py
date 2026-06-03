#!/usr/bin/env python3
"""analyze_type2_remaining_failures.py

Step 8 - Remaining Failure Clustering + Targeted Fix Plan.

Diagnostics ONLY. This script reads the full-dataset validation summary JSON
(and, when available, the row-level JSONL) produced by
``validate_type2_full_dataset.py`` and groups the remaining unresolved /
symbolic / failed / risky rows into ranked, evidence-based clusters. It emits:

  * type2_remaining_failure_clusters.json   (machine-readable clusters)
  * type2_remaining_failure_clusters.md     (human-readable Step 9 repair plan)

It does NOT import any pipeline module, does NOT modify pipeline behaviour, does
NOT train anything, and adds no Transformer code or cache. Pure read-only
analysis over the validation artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Template / token vocabularies (kept local so this tool has no pipeline dep)  #
# --------------------------------------------------------------------------- #

COULOMB_TEMPLATES = {
    "force_resultant_coulomb",
    "coulomb_force_vector",
    "coulomb_right_angle_resultant",
    "coulomb_pairwise_vector_sum",
    "coulomb_force_scalar",
    "scalar_coulomb_single",
    "coulomb_collinear_opposite",
    "coulomb_equilateral_resultant",
}

OHM_TEMPLATES = {
    "ohms_current",
    "ohms_law_current",
    "ohms_voltage",
    "ohms_resistance",
    "power_from_voltage_resistance",
    "power_from_current_resistance",
}

CAP_TEMPLATES = {
    "capacitor_energy",
    "capacitance_from_energy_voltage",
    "voltage_from_capacitor_energy",
    "capacitance_definition",
    "capacitor_energy_charge_capacitance",
    "dielectric_battery_disconnected",
    "dielectric_battery_connected",
}

LC_RLC_TEMPLATES = {
    "lc_frequency_period",
    "inductance_from_energy_current",
    "current_from_inductor_energy",
    "rlc_omega_factor_for_resonance",
    "resonance_UL_calc",
    "ac_impedance_RLC",
    "ac_power_factor_from_R_Z",
    "sinusoidal_inductor_energy",
    "sinusoidal_capacitor_energy",
}

MEASUREMENT_TEMPLATES = {
    "mean_value",
    "average_absolute_error",
    "percent_error",
    "least_count_rel_error",
    "measurement_error",
    "absolute_error",
    "relative_error",
}

NON_NUMERIC_TEMPLATES = {"boolean_check_candidate", "conceptual_relation_target"}

SKELETON_TEMPLATE = "skeleton_placeholder"

# Unsupported-dispatch formula fragments that indicate a missing executor branch.
LC_UNSUPPORTED_FRAGMENTS = (
    "f = f_res",
    "omega = omega * sqrt",
    "z_2 = v / i2",
    "x_l =",
    "x_c =",
    "power_factor =",
    "k_ratio =",
    "r = z",
    "i = i1",
    "x = sqrt(r2",
    "k = sqrt(r2",
)
MAGNETIC_UNSUPPORTED_FRAGMENTS = (
    "b = mu_0",
    "phi_b = b * a",
    "emf = l * i / t",
    "n_turns_per_meter = n_turns",
)
MAGNETIC_TARGETS = {"B", "Phi_B", "emf", "n_turns_per_meter"}

# Variable vocabulary used for best-effort missing-var extraction from warnings.
KNOWN_VARS = [
    "C_cap", "U_cap", "U_B", "U_E", "epsilon_r", "Phi_B", "n_turns_per_meter",
    "X_L", "X_C", "Z_2", "I1", "I2", "R1", "R2", "q0", "q1", "q2", "q3",
    "r13", "r23", "emf", "V", "U", "R", "I", "L", "C", "Q", "B", "f", "r",
]

# Warning-text fragments (lower-cased) used by matchers.
COULOMB_GEOM_FRAGMENTS = (
    "role-aware coulomb scene extracted pair forces",
    "vector sum left symbolic",
    "missing distance between source point",
    "ambiguous vector geometry",
    "geometry is insufficient",
    "geometry is ambiguous",
    "coulomb scene",
)
WRITEBACK_FRAGMENT = "conclusion target not found in computed environment"
MEASUREMENT_FRAGMENTS = (
    "measurement array unavailable",
    "measured or true value unavailable",
)


# --------------------------------------------------------------------------- #
# Category -> ROI multiplier and fix-risk (Part 5 of the brief)               #
# --------------------------------------------------------------------------- #
# Explicit multipliers from the brief:
#   * unsupported dispatch / target writeback  -> 1.0 * row_count
#   * missing input alias                       -> 0.6 * row_count
#   * Coulomb ambiguous geometry                -> 0.4 * row_count
#   * parser failure / skeleton                 -> 0.3 * row_count
#   * conceptual / non-numeric                  -> 0.2 * row_count
# Two categories not enumerated in the brief use conservative values:
#   * measurement_array (extraction-bound)      -> 0.3
#   * sanity_risk (already numeric)             -> 0.2
CATEGORY_GAIN_MULT: Dict[str, float] = {
    "executor_dispatch": 1.0,
    "target_writeback": 1.0,
    "missing_input_alias": 0.6,
    "coulomb_geometry": 0.4,
    "parser_failure": 0.3,
    "non_numeric": 0.2,
    "measurement_array": 0.3,
    "sanity_risk": 0.2,
    "unknown": 0.3,
}
CATEGORY_RISK: Dict[str, str] = {
    "executor_dispatch": "low",
    "target_writeback": "medium",
    "missing_input_alias": "medium",
    "coulomb_geometry": "high",
    "parser_failure": "high",
    "non_numeric": "medium",
    "measurement_array": "high",
    "sanity_risk": "medium",
    "unknown": "medium",
}


# --------------------------------------------------------------------------- #
# IO helpers                                                                   #
# --------------------------------------------------------------------------- #

def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    # validator stores stats under "summary"; tolerate a bare dict too.
    return doc.get("summary", doc) if isinstance(doc, dict) else {}


def load_rows(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not path:
        return None
    try:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Row field accessors (defensive: every optional field may be absent)         #
# --------------------------------------------------------------------------- #

def _meta(row: Dict[str, Any]) -> Dict[str, Any]:
    m = row.get("metadata")
    return m if isinstance(m, dict) else {}


def _templates(row: Dict[str, Any]) -> List[str]:
    t = row.get("selected_templates")
    return [str(x) for x in t] if isinstance(t, list) else []


def _error_types(row: Dict[str, Any]) -> List[str]:
    e = row.get("execution_error_types")
    return [str(x) for x in e] if isinstance(e, list) else []


def _all_warnings(row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("execution_warnings", "pipeline_warnings"):
        v = row.get(key)
        if isinstance(v, list):
            out.extend(str(x) for x in v)
    return out


def _warnings_blob(row: Dict[str, Any]) -> str:
    return " \u2016 ".join(_all_warnings(row)).lower()


def _unsupported(row: Dict[str, Any]) -> List[str]:
    v = _meta(row).get("unsupported_dispatch_names")
    return [str(x) for x in v] if isinstance(v, list) else []


def _executed(row: Dict[str, Any]) -> List[str]:
    v = _meta(row).get("executed_dispatch_names")
    return [str(x) for x in v] if isinstance(v, list) else []


def _is_numeric(row: Dict[str, Any]) -> bool:
    return row.get("answer_type") == "numeric"


def _explicitly_risky(row: Dict[str, Any]) -> bool:
    return (
        row.get("sanity_risk_level") in {"HIGH", "CRITICAL"}
        or bool(row.get("sanity_downgraded"))
    )


def extract_missing_vars(row: Dict[str, Any]) -> List[str]:
    """Best-effort missing-variable extraction from warning text.

    The compact row does not carry an explicit missing-vars list, so we mine the
    warning strings: distance pairs, missing charge values, and any known
    variable name mentioned in a 'Missing ...' clause.
    """
    found: List[str] = []
    for w in _all_warnings(row):
        wl = w.lower()
        m = re.search(
            r"missing distance between source point\s+(\S+?)\s+and target point\s+(\S+?)[\.\s]",
            wl,
        )
        if m:
            found.append(f"dist({m.group(1)}->{m.group(2)})")
            continue
        m = re.search(r"missing charge value for source point\s+(\S+?)[\.\s]", wl)
        if m:
            found.append(f"q_missing:{m.group(1)}")
            continue
        if wl.startswith("missing"):
            for var in KNOWN_VARS:
                if re.search(rf"\b{re.escape(var.lower())}\b", wl):
                    found.append(var)
                    break
    return found


# --------------------------------------------------------------------------- #
# Candidate selection (Part 2)                                                 #
# --------------------------------------------------------------------------- #

def is_candidate(row: Dict[str, Any]) -> bool:
    """A row enters the failure universe if it matches any selection rule, except
    accepted LOW/MEDIUM numeric rows are excluded unless explicitly risky."""
    # Exclusion override: accepted, non-risky numeric answers are not failures.
    if _is_numeric(row) and not _explicitly_risky(row):
        return False

    et = _error_types(row)
    if row.get("answer_type") != "numeric":
        return True
    if row.get("pipeline_status") == "ERROR":
        return True
    if row.get("execution_status") in {"FAIL", "WARN"} and not _is_numeric(row):
        return True
    if "target_unresolved" in et:
        return True
    if "missing_inputs" in et:
        return True
    if _unsupported(row):
        return True
    if SKELETON_TEMPLATE in _templates(row):
        return True
    if row.get("parser_status") == "FAIL":
        return True
    if row.get("sanity_risk_level") in {"HIGH", "CRITICAL"}:
        return True
    if row.get("sanity_downgraded"):
        return True
    return False


# --------------------------------------------------------------------------- #
# Cluster assignment (Part 4). First match wins so row counts never overlap.   #
# Order encodes separation of concerns + ROI intent.                           #
# --------------------------------------------------------------------------- #

def _match_non_numeric_conceptual(row: Dict[str, Any]) -> bool:  # H
    tmpl = set(_templates(row))
    if tmpl & NON_NUMERIC_TEMPLATES:
        return True
    return (
        row.get("parser_status") == "PASS_NON_NUMERIC"
        and row.get("answer_type") in {"selected_step_plan", "symbolic_trace"}
    )


def _match_measurement(row: Dict[str, Any]) -> bool:  # F
    if _is_numeric(row):
        return False
    blob = _warnings_blob(row)
    has_meas_warn = any(frag in blob for frag in MEASUREMENT_FRAGMENTS)
    has_meas_tmpl = bool(set(_templates(row)) & MEASUREMENT_TEMPLATES)
    return has_meas_warn or has_meas_tmpl


def _match_magnetic(row: Dict[str, Any]) -> bool:  # E
    uns = " ".join(_unsupported(row)).lower()
    if any(frag in uns for frag in MAGNETIC_UNSUPPORTED_FRAGMENTS):
        return True
    return row.get("target") in MAGNETIC_TARGETS and not _is_numeric(row)


def _match_lc_rlc(row: Dict[str, Any]) -> bool:  # D
    tmpl = set(_templates(row))
    uns = " ".join(_unsupported(row)).lower()
    has_unsupported = any(frag in uns for frag in LC_UNSUPPORTED_FRAGMENTS)
    if has_unsupported:
        return True
    return bool(tmpl & LC_RLC_TEMPLATES) and not _is_numeric(row)


def _match_skeleton_parser(row: Dict[str, Any]) -> bool:  # G
    if _is_numeric(row):
        return False
    if SKELETON_TEMPLATE in _templates(row):
        return True
    return row.get("parser_status") == "FAIL"


def _match_coulomb_geometry(row: Dict[str, Any]) -> bool:  # A
    if _is_numeric(row):
        return False
    if not (set(_templates(row)) & COULOMB_TEMPLATES):
        return False
    blob = _warnings_blob(row)
    return any(frag in blob for frag in COULOMB_GEOM_FRAGMENTS)


def _match_ohm(row: Dict[str, Any]) -> bool:  # B
    if _is_numeric(row):
        return False
    return bool(set(_templates(row)) & OHM_TEMPLATES)


def _match_capacitor(row: Dict[str, Any]) -> bool:  # C
    if _is_numeric(row):
        return False
    return bool(set(_templates(row)) & CAP_TEMPLATES)


def _match_target_writeback(row: Dict[str, Any]) -> bool:  # I
    if _is_numeric(row):
        return False
    if _unsupported(row):
        return False
    if not _executed(row):
        return False
    if row.get("execution_status") not in {"PASS", "WARN"}:
        return False
    return WRITEBACK_FRAGMENT in _warnings_blob(row)


def _match_sanity_risk(row: Dict[str, Any]) -> bool:  # J
    return _explicitly_risky(row)


# (key, name, category, matcher) in precedence order. First match wins.
# J (sanity-risk) is checked first: the brief requires explicitly-risky rows to be
# isolated from the executor/parser failure buckets ("should not be mixed with
# symbolic executor failures"), even when they also carry an unsupported dispatch
# (e.g. numeric omega=0 from rlc_omega_factor_for_resonance) or a skeleton template.
CLUSTER_DEFS: List[Tuple[str, str, str, Any]] = [
    ("J", "Sanity downgrade / high-risk numeric", "sanity_risk", _match_sanity_risk),
    ("H", "Non-numeric conceptual / boolean", "non_numeric", _match_non_numeric_conceptual),
    ("G", "Skeleton placeholder / parser low confidence", "parser_failure", _match_skeleton_parser),
    ("F", "Measurement array unavailable", "measurement_array", _match_measurement),
    ("E", "Magnetic field / flux / EMF unsupported dispatch", "executor_dispatch", _match_magnetic),
    ("D", "LC / RLC unresolved", "executor_dispatch", _match_lc_rlc),
    ("A", "Coulomb force/vector geometry unresolved", "coulomb_geometry", _match_coulomb_geometry),
    ("B", "Ohm / power missing V or R", "missing_input_alias", _match_ohm),
    ("C", "Capacitor energy / inverse missing C_cap/U_cap/V", "missing_input_alias", _match_capacitor),
    ("I", "Target writeback unresolved despite executed dispatch", "target_writeback", _match_target_writeback),
]

# Static per-cluster guidance (root cause, fix, files) from the brief.
CLUSTER_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "A": {
        "likely_root_cause": "Target point not inferred; source/target distances "
        "missing; r13/r23 aliases incomplete; collinear/midpoint/opposite-side "
        "logic incomplete so the role-aware scene cannot safely compose the final vector.",
        "recommended_fix": "Complete distance inference for collinear / midpoint / "
        "opposite-side layouts and r13/r23 aliasing so the role-aware Coulomb scene "
        "can resolve the final vector magnitude instead of leaving it symbolic.",
        "files_to_modify": ["type2_numeric_executor.py", "type2_candidate_generator.py"],
    },
    "B": {
        "likely_root_cause": "Voltage alias U/V not resolved; resistance alias "
        "R/R_total/R1/R2 not resolved; quantity extracted with unit but wrong "
        "variable name; I/V/R target writeback failing.",
        "recommended_fix": "Add U<->V and R/R1/R2/R_total alias resolution plus "
        "I/V/R target writeback in the executor.",
        "files_to_modify": ["type2_numeric_executor.py", "unit_normalizer.py", "condition_extractor.py"],
    },
    "C": {
        "likely_root_cause": "U means voltage in some rows and energy in others; "
        "C means capacitance but collides with the Coulomb unit; U_cap/U_C/U_E "
        "target ambiguity; formula output alias not written back.",
        "recommended_fix": "Disambiguate U(voltage) vs U(energy) and C(capacitance) "
        "vs C(charge unit); resolve U_cap/U_C/U_E target aliases and write the "
        "computed capacitor quantity back to the requested target.",
        "files_to_modify": ["type2_numeric_executor.py", "type2_adapter.py"],
    },
    "D": {
        "likely_root_cause": "Missing executor dispatch for low-frequency RLC / "
        "resonance formulas; aliases L/L_ind, C/C_cap, f/f_res/f_osc, "
        "omega/omega_0 incomplete; resonance target writeback incomplete.",
        "recommended_fix": "Implement the missing RLC/resonance dispatches "
        "(f=f_res, omega scaling, Z_2=V/I2, X_L, X_C, power_factor, R=Z) and "
        "complete L/C/f/omega aliasing.",
        "files_to_modify": ["type2_numeric_executor.py"],
    },
    "E": {
        "likely_root_cause": "Formula dispatch not implemented for magnetic / flux "
        "/ EMF formulas; unit support for T, Wb, H, turns/m incomplete.",
        "recommended_fix": "Add dispatches for B=mu_0*n*I, Phi_B=B*A, emf=L*I/t, "
        "n_turns_per_meter=n_turns/L plus T/Wb/H/turns-per-m unit support.",
        "files_to_modify": ["type2_numeric_executor.py"],
    },
    "F": {
        "likely_root_cause": "Parser does not extract measurement lists, so the "
        "executor cannot compute mean / deviation / error arrays.",
        "recommended_fix": "Extract measurement arrays in the parser and add "
        "mean/abs-error/rel-error array execution, or accept these as symbolic.",
        "files_to_modify": ["condition_extractor.py", "rule_extractor.py", "type2_numeric_executor.py"],
    },
    "G": {
        "likely_root_cause": "Stage 0 parser / template coverage missing; weak "
        "target detection; formula template absent so a skeleton placeholder fired.",
        "recommended_fix": "Strengthen target detection and template coverage so "
        "real templates fire instead of the skeleton placeholder.",
        "files_to_modify": ["target_detector.py", "template_fallback.py", "condition_extractor.py", "type2_candidate_generator.py"],
    },
    "H": {
        "likely_root_cause": "The numeric executor is not meant to answer "
        "conceptual relation / boolean questions; a separate symbolic / conceptual "
        "handler is required.",
        "recommended_fix": "Route conceptual/boolean candidates to a dedicated "
        "symbolic/conceptual response handler instead of the numeric executor.",
        "files_to_modify": ["type2_candidate_pipeline.py", "type2_numeric_executor.py"],
    },
    "I": {
        "likely_root_cause": "Formula computed an intermediate value but did not "
        "write a compatible target alias; target aliases incomplete; conclusion "
        "resolution too strict.",
        "recommended_fix": "Map executed-dispatch outputs onto the requested "
        "target alias and relax conclusion resolution so computed values are "
        "accepted (highest ROI: dispatch already ran).",
        "files_to_modify": ["type2_numeric_executor.py"],
    },
    "J": {
        "likely_root_cause": "Mix of genuinely unsafe numeric answers and answers "
        "that could be accepted after better formula validation; should not be "
        "mixed with symbolic-executor failures.",
        "recommended_fix": "Re-check the flagged numerics; accept those validated "
        "by a correct formula, keep the rest downgraded.",
        "files_to_modify": ["type2_answer_sanity_checker.py", "type2_numeric_executor.py"],
    },
    "OTHER": {
        "likely_root_cause": "Selected as a failure but did not match a known "
        "cluster signature; inspect manually.",
        "recommended_fix": "Manual triage; refine cluster matchers if a pattern emerges.",
        "files_to_modify": [],
    },
}


def assign_cluster(row: Dict[str, Any]) -> str:
    for key, _name, _cat, matcher in CLUSTER_DEFS:
        if matcher(row):
            return key
    return "OTHER"


# --------------------------------------------------------------------------- #
# ROI scoring (Part 5)                                                         #
# --------------------------------------------------------------------------- #

def estimated_gain(category: str, row_count: int) -> float:
    return round(CATEGORY_GAIN_MULT.get(category, 0.3) * row_count, 1)


def priority_for(gain: float) -> str:
    if gain >= 30:
        return "P0"
    if gain >= 10:
        return "P1"
    if gain >= 3:
        return "P2"
    return "P3"


def should_fix_now(priority: str, risk: str) -> bool:
    if priority == "P0":
        return True
    if priority == "P1" and risk != "high":
        return True
    return False


# --------------------------------------------------------------------------- #
# Cluster building from rows                                                   #
# --------------------------------------------------------------------------- #

def _top(counter: Counter, k: int = 12) -> Dict[str, int]:
    return {str(name): int(cnt) for name, cnt in counter.most_common(k)}


def _sample_row(row: Dict[str, Any]) -> Dict[str, Any]:
    text = str(row.get("problem_text", ""))
    if len(text) > 240:
        text = text[:237] + "..."
    return {
        "row_index": row.get("row_index"),
        "problem_text": text,
        "parser_status": row.get("parser_status"),
        "pipeline_status": row.get("pipeline_status"),
        "target": row.get("target"),
        "target_unit": row.get("target_unit"),
        "answer_type": row.get("answer_type"),
        "selected_templates": _templates(row),
        "execution_status": row.get("execution_status"),
        "execution_warnings": row.get("execution_warnings") or [],
        "execution_error_types": _error_types(row),
        "missing_vars": extract_missing_vars(row),
        "unsupported_dispatch_names": _unsupported(row),
        "rank_margin": row.get("rank_margin"),
        "confidence": row.get("confidence"),
    }


def build_clusters_from_rows(
    rows: List[Dict[str, Any]], sample_per_cluster: int
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if not is_candidate(row):
            continue
        key = assign_cluster(row)
        buckets.setdefault(key, []).append(row)

    name_by_key = {k: n for k, n, _c, _m in CLUSTER_DEFS}
    cat_by_key = {k: c for k, _n, c, _m in CLUSTER_DEFS}
    name_by_key["OTHER"] = "Unclustered remaining failures"
    cat_by_key["OTHER"] = "unknown"

    clusters: List[Dict[str, Any]] = []
    for key, members in buckets.items():
        category = cat_by_key.get(key, "unknown")
        row_count = len(members)
        gain = estimated_gain(category, row_count)
        priority = priority_for(gain)
        risk = CATEGORY_RISK.get(category, "medium")

        templates: Counter = Counter()
        targets: Counter = Counter()
        missing_vars: Counter = Counter()
        error_types: Counter = Counter()
        warnings: Counter = Counter()
        unsupported: Counter = Counter()
        for r in members:
            templates.update(_templates(r))
            if r.get("target"):
                targets[str(r.get("target"))] += 1
            missing_vars.update(extract_missing_vars(r))
            error_types.update(_error_types(r))
            for w in (r.get("execution_warnings") or []):
                warnings[str(w)] += 1
            unsupported.update(_unsupported(r))

        guidance = CLUSTER_GUIDANCE.get(key, CLUSTER_GUIDANCE["OTHER"])
        # Stable, representative samples: lowest rank_margin first (most contested).
        ordered = sorted(
            members,
            key=lambda r: (r.get("rank_margin") if isinstance(r.get("rank_margin"), (int, float)) else 1.0),
        )
        samples = [_sample_row(r) for r in ordered[:sample_per_cluster]]

        clusters.append(
            {
                "cluster_id": key,
                "cluster_name": name_by_key.get(key, key),
                "priority": priority,
                "category": category,
                "row_count": row_count,
                "estimated_numeric_gain": gain,
                "risk_level": risk,
                "main_templates": _top(templates),
                "main_targets": _top(targets),
                "main_missing_vars": _top(missing_vars),
                "main_error_types": _top(error_types),
                "main_warnings": _top(warnings, 8),
                "unsupported_formulas": _top(unsupported),
                "likely_root_cause": guidance["likely_root_cause"],
                "recommended_fix": guidance["recommended_fix"],
                "files_to_modify": list(guidance["files_to_modify"]),
                "should_fix_now": should_fix_now(priority, risk),
                "sample_rows": samples,
            }
        )

    clusters.sort(key=lambda c: (-c["estimated_numeric_gain"], -c["row_count"]))
    return clusters


# --------------------------------------------------------------------------- #
# Cluster building from summary only (JSONL absent)                            #
# --------------------------------------------------------------------------- #

def build_clusters_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Approximate clusters from summary aggregates when row JSONL is missing."""
    sym = summary.get("symbolic_by_template_counts", {}) or {}
    unsupported = summary.get("unsupported_dispatch_by_formula", {}) or {}
    missing = summary.get("missing_input_by_variable_counts", {}) or {}

    def tmpl_sum(names) -> int:
        return int(sum(int(sym.get(n, 0)) for n in names))

    def make(key, names, extra_unsupported=()):  # noqa: ANN001
        category = next((c for k, _n, c, _m in CLUSTER_DEFS if k == key), "unknown")
        name = next((n for k, n, _c, _m in CLUSTER_DEFS if k == key), key)
        row_count = tmpl_sum(names)
        used = {n: int(sym[n]) for n in names if n in sym}
        unsup = {f: int(unsupported[f]) for f in unsupported if any(frag in f.lower() for frag in extra_unsupported)}
        if extra_unsupported and not row_count:
            row_count = int(sum(unsup.values()))
        gain = estimated_gain(category, row_count)
        priority = priority_for(gain)
        risk = CATEGORY_RISK.get(category, "medium")
        guidance = CLUSTER_GUIDANCE.get(key, CLUSTER_GUIDANCE["OTHER"])
        return {
            "cluster_id": key,
            "cluster_name": name,
            "priority": priority,
            "category": category,
            "row_count": row_count,
            "estimated_numeric_gain": gain,
            "risk_level": risk,
            "main_templates": used,
            "main_targets": {},
            "main_missing_vars": {k: int(v) for k, v in missing.items()} if key in {"B", "C"} else {},
            "main_error_types": {},
            "main_warnings": {},
            "unsupported_formulas": unsup,
            "likely_root_cause": guidance["likely_root_cause"],
            "recommended_fix": guidance["recommended_fix"],
            "files_to_modify": list(guidance["files_to_modify"]),
            "should_fix_now": should_fix_now(priority, risk),
            "sample_rows": [],
            "note": "summary-only estimate; row-level JSONL was not available",
        }

    clusters = [
        make("A", ["force_resultant_coulomb", "coulomb_force_vector", "coulomb_right_angle_resultant", "coulomb_pairwise_vector_sum", "scalar_coulomb_single"]),
        make("B", ["ohms_current", "ohms_voltage"]),
        make("C", ["capacitor_energy", "capacitance_from_energy_voltage", "voltage_from_capacitor_energy"]),
        make("D", ["lc_frequency_period", "inductance_from_energy_current", "current_from_inductor_energy"], LC_UNSUPPORTED_FRAGMENTS),
        make("E", [], MAGNETIC_UNSUPPORTED_FRAGMENTS),
        make("F", ["mean_value", "average_absolute_error", "least_count_rel_error"]),
        make("G", ["skeleton_placeholder"]),
    ]
    clusters = [c for c in clusters if c["row_count"] > 0]
    clusters.sort(key=lambda c: (-c["estimated_numeric_gain"], -c["row_count"]))
    return clusters


# --------------------------------------------------------------------------- #
# Markdown report (Part 6)                                                     #
# --------------------------------------------------------------------------- #

def _fmt_counts(d: Dict[str, int], limit: int = 6) -> str:
    if not d:
        return "-"
    items = list(d.items())[:limit]
    return ", ".join(f"`{k}`={v}" for k, v in items)


def render_markdown(summary: Dict[str, Any], clusters: List[Dict[str, Any]], have_rows: bool) -> str:
    evaluated = summary.get("evaluated", "?")
    numeric = summary.get("numeric_answer_count", "?")
    symbolic = summary.get("symbolic_trace_count", "?")
    error_ct = (summary.get("pipeline_status_counts", {}) or {}).get("ERROR", summary.get("failed_count", "?"))
    p0 = [c for c in clusters if c["priority"] == "P0"]

    lines: List[str] = []
    lines.append("# Type2 Remaining Failure Clusters")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Evaluated rows: **{evaluated}**")
    lines.append(f"- Current numeric answers: **{numeric}**")
    lines.append(f"- Current symbolic answers: **{symbolic}**")
    lines.append(f"- Current pipeline errors: **{error_ct}**")
    lines.append(f"- Failure rows clustered: **{sum(c['row_count'] for c in clusters)}** across **{len(clusters)}** clusters")
    if not have_rows:
        lines.append("- _Note: row-level JSONL was unavailable; clusters are summary-level estimates._")
    lines.append("")
    if p0:
        lines.append("**Top P0 clusters (highest ROI):**")
        lines.append("")
        for c in p0:
            lines.append(f"- {c['cluster_name']} — {c['row_count']} rows, est. gain {c['estimated_numeric_gain']}, fix risk {c['risk_level']}")
        lines.append("")
    lines.append("**Recommended next Step 9 fixes:** see the ordered plan at the end of this report.")
    lines.append("")

    # Cluster table
    lines.append("## Cluster Table")
    lines.append("")
    lines.append("| Priority | Cluster | Rows | Est. gain | Risk | Files | One-line fix |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in clusters:
        files = ", ".join(c["files_to_modify"]) or "-"
        one_line = c["recommended_fix"].split(". ")[0].rstrip(".")
        lines.append(
            f"| {c['priority']} | {c['cluster_name']} | {c['row_count']} | "
            f"{c['estimated_numeric_gain']} | {c['risk_level']} | {files} | {one_line} |"
        )
    lines.append("")

    # Detailed clusters
    lines.append("## Detailed Clusters")
    lines.append("")
    for c in clusters:
        lines.append(f"### [{c['priority']}] {c['cluster_id']} — {c['cluster_name']}")
        lines.append("")
        lines.append(f"- **Category:** {c['category']} | **Rows:** {c['row_count']} | "
                     f"**Est. numeric gain:** {c['estimated_numeric_gain']} | "
                     f"**Fix risk:** {c['risk_level']} | **Fix now:** {c['should_fix_now']}")
        lines.append(f"- **Main templates:** {_fmt_counts(c['main_templates'])}")
        lines.append(f"- **Main targets:** {_fmt_counts(c['main_targets'])}")
        if c["main_missing_vars"]:
            lines.append(f"- **Main missing vars:** {_fmt_counts(c['main_missing_vars'])}")
        if c["main_error_types"]:
            lines.append(f"- **Main error types:** {_fmt_counts(c['main_error_types'])}")
        if c["unsupported_formulas"]:
            lines.append(f"- **Unsupported formulas:** {_fmt_counts(c['unsupported_formulas'])}")
        if c["main_warnings"]:
            lines.append(f"- **Top warnings:** {_fmt_counts(c['main_warnings'], 4)}")
        lines.append(f"- **Likely root cause:** {c['likely_root_cause']}")
        lines.append(f"- **Recommended fix:** {c['recommended_fix']}")
        lines.append(f"- **Files to modify:** {', '.join(c['files_to_modify']) or '-'}")
        if c["sample_rows"]:
            lines.append("- **Sample rows:**")
            for s in c["sample_rows"][:5]:
                tmpl = ", ".join(s["selected_templates"]) or "-"
                lines.append(
                    f"    - row {s['row_index']} (target `{s['target']}` {s['target_unit']}, "
                    f"{s['answer_type']}, templates: {tmpl}): {s['problem_text']}"
                )
        lines.append("")

    # Step 9 plan, ordered by ROI then ascending fix-risk.
    lines.append("## Recommended Step 9 Plan")
    lines.append("")
    risk_rank = {"low": 0, "medium": 1, "high": 2}
    ordered = sorted(
        clusters,
        key=lambda c: (-c["estimated_numeric_gain"], risk_rank.get(c["risk_level"], 1)),
    )
    for i, c in enumerate(ordered, start=1):
        one_line = c["recommended_fix"].split(". ")[0].rstrip(".")
        lines.append(f"{i}. **{c['cluster_name']}** ({c['priority']}, ~{c['estimated_numeric_gain']} gain, "
                     f"{c['risk_level']} risk): {one_line}.")
    lines.append("")
    lines.append("_Diagnostics only. No pipeline behaviour was modified by this report._")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 8 diagnostics: cluster remaining Type2 failures (read-only)."
    )
    parser.add_argument("--summary-json", default="type2_full_dataset_validation_sanity_calibrated.json")
    parser.add_argument("--rows-jsonl", default="type2_full_dataset_rows_sanity_calibrated.jsonl")
    parser.add_argument("--output-json", default="type2_remaining_failure_clusters.json")
    parser.add_argument("--output-md", default="type2_remaining_failure_clusters.md")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--sample-per-cluster", type=int, default=10)
    args = parser.parse_args()

    try:
        summary = load_summary(args.summary_json)
    except FileNotFoundError:
        print(f"ERROR: summary JSON not found: {args.summary_json}")
        return 2
    except json.JSONDecodeError as exc:
        print(f"ERROR: could not parse summary JSON: {exc}")
        return 2

    rows = load_rows(args.rows_jsonl)
    have_rows = rows is not None

    if have_rows:
        clusters = build_clusters_from_rows(rows, args.sample_per_cluster)
        source = f"row-level JSONL ({len(rows)} rows)"
    else:
        clusters = build_clusters_from_summary(summary)
        source = "summary JSON only (row JSONL unavailable)"

    clusters = clusters[: args.top_k]

    out_doc = {
        "source": source,
        "summary_input": args.summary_json,
        "rows_input": args.rows_jsonl if have_rows else None,
        "evaluated": summary.get("evaluated"),
        "numeric_answer_count": summary.get("numeric_answer_count"),
        "symbolic_trace_count": summary.get("symbolic_trace_count"),
        "pipeline_status_counts": summary.get("pipeline_status_counts"),
        "sanity_risk_level_counts": summary.get("sanity_risk_level_counts"),
        "cluster_count": len(clusters),
        "clustered_row_total": sum(c["row_count"] for c in clusters),
        "clusters": clusters,
    }

    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(out_doc, fh, ensure_ascii=False, indent=2)

    md = render_markdown(summary, clusters, have_rows)
    with open(args.output_md, "w", encoding="utf-8") as fh:
        fh.write(md)

    print(f"Source: {source}")
    print(f"Wrote {len(clusters)} clusters -> {args.output_json}")
    print(f"Wrote report -> {args.output_md}")
    print("Top clusters by estimated numeric gain:")
    for c in clusters[:8]:
        print(f"  [{c['priority']}] {c['cluster_id']} {c['cluster_name']}: "
              f"rows={c['row_count']} gain={c['estimated_numeric_gain']} risk={c['risk_level']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())