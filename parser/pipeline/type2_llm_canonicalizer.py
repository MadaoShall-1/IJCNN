"""Step 10: deterministic canonicalization of LLM-fallback step plans.

The Stage-0 LLM fallback (local Qwen3) recovers parser failures by proposing a
loose, semantic step plan (formula names like ``electric_force`` or
``magnetic_field_solenoid``). Those names are NOT what the deterministic numeric
executor dispatches on, so they fall through to "unsupported dispatch" and the
row stays symbolic.

This module maps those loose names onto the executor's *canonical* formula
strings (the exact ``formula_name`` forms the executor already supports), using
a deterministic table plus input/target-driven variant selection. It never
executes anything, never uses eval, and never overwrites deterministic
quantities. If a loose name cannot be mapped safely it is left symbolic.

The candidate generator calls :func:`canonicalize_llm_fallback_candidate` and,
when it returns canonical formulas, adds an extra candidate with
``source="llm_fallback_canonicalized"`` for the verifier/ranker to evaluate like
any other candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from parser.pipeline.type2_adapter import Type2WorldModelInput


# --------------------------------------------------------------------------- #
# Dataclasses                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class Type2FormulaMapping:
    source_patterns: list[str]
    canonical_formula_name: str
    canonical_template_name: str
    output_var: dict[str, Any] = field(default_factory=dict)
    input_var_hints: dict[str, Any] = field(default_factory=dict)
    required_context: list[str] = field(default_factory=list)
    target_compatibility: list[str] = field(default_factory=list)
    unit_compatibility: list[str] = field(default_factory=list)
    risk_level: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Type2CanonicalizationResult:
    status: str  # PASS | WARN | FAIL
    original_template_names: list[str] = field(default_factory=list)
    original_formula_names: list[str] = field(default_factory=list)
    canonical_template_names: list[str] = field(default_factory=list)
    canonical_step_plan: list[dict[str, Any]] = field(default_factory=list)
    canonical_formula_names: list[str] = field(default_factory=list)
    mapping_log: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Quantity-presence helpers (read-only against world_input.known_quantities)   #
# --------------------------------------------------------------------------- #

_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "charge": ("q", "Q", "q1", "q2", "q3", "q0", "qo", "charge", "Q_max", "q_max"),
    "two_charges": ("q1", "q2"),
    "distance": ("r", "d", "r12", "r13", "r23", "distance"),
    "voltage": ("V", "U", "voltage", "V_rms", "U_C"),
    "capacitance": ("C_cap", "C", "capacitance"),
    "cap_energy": ("E_energy", "U_cap", "U_E", "energy", "stored_energy", "electric_field_energy"),
    "mag_energy": ("U_B", "E_energy", "energy", "magnetic_field_energy"),
    "current": ("I", "I_rms", "current", "I_max"),
    "current2": ("I2", "I_2"),
    "inductance": ("L_ind", "L", "inductance"),
    "resistance": ("R", "R1", "resistance"),
    "resistance2": ("R2",),
    "resistance3": ("R3",),
    "reactance_L": ("X_L",),
    "reactance_C": ("X_C",),
    "impedance": ("Z",),
    "omega": ("omega", "omega_0"),
    "freq": ("f", "f_res", "f_osc", "frequency"),
    "area": ("A", "area"),
    "n_turns": ("n_turns", "N", "turns"),
    "turn_density": ("n_turns_per_meter", "n"),
    "length": ("L_length", "length"),
    "magnetic_field": ("B", "magnetic_field"),
    "epsilon_0": ("epsilon_0",),
    "permittivity_area_gap": ("A", "d"),
    "time": ("t", "time"),
}


def _norm(name: Any) -> str:
    return str(name or "").strip()


def _kq_keys(world_input: Type2WorldModelInput) -> set[str]:
    kq = world_input.known_quantities or {}
    return {_norm(k) for k in kq.keys()}


def _present(world_input: Type2WorldModelInput, group: str) -> bool:
    keys = _kq_keys(world_input)
    return any(alias in keys for alias in _ALIAS_GROUPS.get(group, (group,)))


def _count_charges(world_input: Type2WorldModelInput) -> int:
    keys = _kq_keys(world_input)
    return sum(1 for c in ("q1", "q2", "q3", "q0", "qo") if c in keys)


def _conditions_blob(world_input: Type2WorldModelInput) -> str:
    return " ".join(
        [world_input.problem_text]
        + list(world_input.conditions or [])
        + list(world_input.domains or [])
        + list(world_input.sub_domains or [])
    ).lower()


# --------------------------------------------------------------------------- #
# Canonical mapping table (static patterns). Variant-bearing families are       #
# resolved in _resolve_canonical with input/target awareness.                  #
# --------------------------------------------------------------------------- #

FORMULA_MAPPINGS: list[Type2FormulaMapping] = [
    Type2FormulaMapping(
        ["electric_force", "coulomb_force", "force_between_charges", "electrostatic_force", "coulombs_law"],
        "F = k * abs(q1*q2) / r^2", "coulomb_force_scalar",
        {"F": "force"}, {"q1": "charge", "q2": "charge", "r": "distance"},
        ["charge", "distance"], ["F", "F_e", "F_net", "F_on_q3"], ["N"], "medium",
    ),
    Type2FormulaMapping(
        ["energy_in_capacitor", "energy_capacitor", "energy_stored_in_capacitor", "electric_energy_capacitor", "capacitor_energy_formula", "energy_stored"],
        "U_cap = 0.5 * C_cap * V^2", "capacitor_energy",
        {"U_cap": "energy"}, {"C_cap": "capacitance", "V": "voltage"},
        ["capacitance"], ["U_cap", "U_E", "U_total", "energy"], ["J"], "low",
    ),
    Type2FormulaMapping(
        ["capacitance_formula", "capacitance_of_parallel_plate", "parallel_plate_capacitance"],
        "C_cap = epsilon_0 * A / d", "parallel_plate_capacitance",
        {"C_cap": "capacitance"}, {"A": "area", "d": "distance"},
        ["area", "distance"], ["C_cap", "C"], ["F"], "low",
    ),
    Type2FormulaMapping(
        ["charge_on_capacitor", "charge_capacitor", "charge_and_voltage", "capacitor_charge"],
        "Q = C_cap * V", "capacitor_charge",
        {"Q": "charge"}, {"C_cap": "capacitance", "V": "voltage"},
        ["capacitance", "voltage"], ["Q", "q"], ["C"], "low",
    ),
    Type2FormulaMapping(
        ["electric_field", "electric_field_point_charge", "electric_field_of_charge_element", "field_point_charge"],
        "E = k * abs(q) / r^2", "electric_field_point_charge",
        {"E": "electric_field"}, {"q": "charge", "r": "distance"},
        ["charge", "distance"], ["E", "E_net", "E_total"], ["V/m", "N/C"], "low",
    ),
    Type2FormulaMapping(
        ["magnetic_field_solenoid", "magnetic_field", "solenoid_field"],
        "B = mu_0 * n_turns_per_meter * I", "solenoid_field_from_density",
        {"B": "magnetic_field"}, {"n_turns_per_meter": "turn_density", "I": "current"},
        ["current"], ["B"], ["T"], "low",
    ),
    Type2FormulaMapping(
        ["magnetic_flux_solenoid", "flux_linkage", "magnetic_flux_linkage", "magnetic_flux"],
        "Phi_B = B * A", "magnetic_flux_BA",
        {"Phi_B": "flux"}, {"B": "magnetic_field", "A": "area"},
        ["magnetic_field", "area"], ["Phi_B", "Phi"], ["Wb"], "low",
    ),
    Type2FormulaMapping(
        ["emf_from_flux_change", "faradays_law", "emf_self_inductance", "induced_emf"],
        "emf = L * I / t", "emf_from_di_dt",
        {"emf": "emf"}, {"L": "inductance", "I": "current", "t": "time"},
        ["inductance", "time"], ["emf"], ["V"], "low",
    ),
    Type2FormulaMapping(
        ["n_turns_per_meter", "turn_density"],
        "n_turns_per_meter = n_turns / L", "turn_density",
        {"n_turns_per_meter": "turn_density"}, {"n_turns": "n_turns", "L": "length"},
        ["n_turns", "length"], ["n_turns_per_meter", "n"], ["1/m"], "low",
    ),
    Type2FormulaMapping(
        ["impedance_series", "impedance_rlc", "impedance_of_series_rlc"],
        "Z = sqrt(R^2 + (X_L - X_C)^2)", "ac_impedance_RLC",
        {"Z": "impedance"}, {"R": "resistance", "X_L": "reactance_L", "X_C": "reactance_C"},
        ["resistance", "reactance_L", "reactance_C"], ["Z"], ["ohm"], "low",
    ),
    Type2FormulaMapping(
        ["current_rms", "current_in_ac_circuit", "rms_current"],
        "I_rms = V / Z", "ac_I_rms_from_V_Z",
        {"I_rms": "current"}, {"V": "voltage", "Z": "impedance"},
        ["voltage", "impedance"], ["I", "I_rms"], ["A"], "low",
    ),
    Type2FormulaMapping(
        ["inductive_reactance"],
        "X_L = omega * L_ind", "ac_X_L_from_L_omega",
        {"X_L": "reactance_L"}, {"omega": "omega", "L_ind": "inductance"},
        ["omega", "inductance"], ["X_L"], ["ohm"], "low",
    ),
    Type2FormulaMapping(
        ["capacitive_reactance"],
        "X_C = 1 / (omega * C_cap)", "ac_X_C_from_C_omega",
        {"X_C": "reactance_C"}, {"omega": "omega", "C_cap": "capacitance"},
        ["omega", "capacitance"], ["X_C"], ["ohm"], "low",
    ),
    Type2FormulaMapping(
        ["resonance_condition", "resonant_frequency", "resonance_frequency"],
        "f_res = 1 / (2*pi*sqrt(L_ind*C_cap))", "lc_resonance_frequency",
        {"f_res": "frequency"}, {"L_ind": "inductance", "C_cap": "capacitance"},
        ["inductance", "capacitance"], ["f", "f_res", "f_osc"], ["Hz"], "low",
    ),
    Type2FormulaMapping(
        ["angular_frequency", "extract_angular_frequency_from_voltage"],
        "omega = 2*pi*f", "ac_omega_from_f",
        {"omega": "omega"}, {"f": "freq"},
        ["freq"], ["omega", "omega_0"], ["rad/s"], "low",
    ),
    Type2FormulaMapping(
        ["power_factor_formula", "power_factor"],
        "power_factor = R / Z", "ac_power_factor_from_R_Z",
        {"power_factor": "power_factor"}, {"R": "resistance", "Z": "impedance"},
        ["resistance", "impedance"], ["power_factor"], [], "low",
    ),
    Type2FormulaMapping(
        ["voltage_across_resistor"],
        "U_R = I_rms * R", "voltage_across_resistor",
        {"U_R": "voltage"}, {"I_rms": "current", "R": "resistance"},
        ["current", "resistance"], ["U_R", "V_R"], ["V"], "low",
    ),
    Type2FormulaMapping(
        ["voltage_across_capacitor"],
        "U_C = I_rms * X_C", "voltage_across_capacitor",
        {"U_C": "voltage"}, {"I_rms": "current", "X_C": "reactance_C"},
        ["current", "reactance_C"], ["U_C", "V_C"], ["V"], "low",
    ),
]

# Fast lookup: source pattern -> mapping.
_PATTERN_INDEX: dict[str, Type2FormulaMapping] = {}
for _m in FORMULA_MAPPINGS:
    for _p in _m.source_patterns:
        _PATTERN_INDEX[_p.lower()] = _m


# --------------------------------------------------------------------------- #
# Variant-aware resolution                                                     #
# --------------------------------------------------------------------------- #

def _loose_name(formula_name: Any) -> str:
    """Normalize a step's formula_name to a loose lookup token.

    Real canonical formulas contain ``=``; loose semantic names do not. We only
    canonicalize loose names (and a few known unsupported expression strings).
    """
    text = _norm(formula_name).lower()
    if "=" in text:
        return ""
    return text.replace(" ", "_").replace("-", "_")


def _resolve_canonical(
    loose: str,
    world_input: Type2WorldModelInput,
    target: Optional[str],
) -> Optional[dict[str, Any]]:
    """Return {formula, template, output, risk, warning?} or None to stay symbolic."""
    mapping = _PATTERN_INDEX.get(loose)
    if mapping is None:
        return None

    tgt = _norm(target)

    # --- Family-specific variant selection / safety guards ---

    # A. Coulomb force: do not collapse to scalar for multi-charge vector force.
    if mapping.canonical_template_name == "coulomb_force_scalar":
        if _count_charges(world_input) >= 3 or tgt in {"F_net", "F_on_q3"}:
            return {"_skip": True, "warning": "Coulomb force is multi-charge/vector; left to geometry-aware candidates."}
        if not (_present(world_input, "two_charges") and _present(world_input, "distance")):
            return {"_skip": True, "warning": "Coulomb scalar needs q1,q2 and r; left symbolic."}
        return {"formula": "F = k * abs(q1*q2) / r^2", "template": "coulomb_force_scalar",
                "output": tgt or "F_e", "risk": "medium"}

    # B. Capacitor energy: pick variant by available inputs.
    if mapping.canonical_template_name == "capacitor_energy":
        if _present(world_input, "capacitance") and _present(world_input, "voltage"):
            return {"formula": "U_cap = 0.5 * C_cap * V^2", "template": "capacitor_energy", "output": tgt or "U_cap", "risk": "low"}
        if _present(world_input, "charge") and _present(world_input, "capacitance"):
            return {"formula": "U_cap = Q^2 / (2*C_cap)", "template": "capacitor_energy", "output": tgt or "U_cap", "risk": "low"}
        if _present(world_input, "charge") and _present(world_input, "voltage"):
            return {"formula": "U_cap = 0.5 * Q * V", "template": "capacitor_energy", "output": tgt or "U_cap", "risk": "low"}
        return {"_skip": True, "warning": "Capacitor energy needs (C,V) / (Q,C) / (Q,V); left symbolic."}

    # C. Capacitance: parallel-plate vs definition.
    if mapping.canonical_template_name == "parallel_plate_capacitance":
        if _present(world_input, "area") and _present(world_input, "distance"):
            return {"formula": "C_cap = epsilon_0 * A / d", "template": "parallel_plate_capacitance", "output": tgt or "C_cap", "risk": "low"}
        if _present(world_input, "charge") and _present(world_input, "voltage"):
            return {"formula": "C_cap = Q / V", "template": "capacitance_definition", "output": tgt or "C_cap", "risk": "low"}
        return {"_skip": True, "warning": "Capacitance needs (A,d) or (Q,V); left symbolic."}

    # E. Solenoid field: density form vs full N/L form.
    if mapping.canonical_template_name == "solenoid_field_from_density":
        if not _present(world_input, "current"):
            return {"_skip": True, "warning": "Solenoid field needs current; left symbolic."}
        if _present(world_input, "turn_density"):
            return {"formula": "B = mu_0 * n_turns_per_meter * I", "template": "solenoid_field_from_density", "output": tgt or "B", "risk": "low"}
        if _present(world_input, "n_turns") and _present(world_input, "length"):
            return {"formula": "B = mu_0 * n_turns * I / L_length", "template": "solenoid_field_full", "output": tgt or "B", "risk": "low"}
        return {"_skip": True, "warning": "Solenoid field needs turn density or (N,length); left symbolic."}

    # Generic: require declared context inputs to be present, else stay symbolic.
    for ctx in mapping.required_context:
        if not _present(world_input, ctx):
            return {"_skip": True, "warning": f"{loose}: missing required input '{ctx}'; left symbolic."}

    output_name = tgt or (next(iter(mapping.output_var.keys())) if mapping.output_var else None) or loose
    return {"formula": mapping.canonical_formula_name, "template": mapping.canonical_template_name,
            "output": output_name, "risk": mapping.risk_level}


# --------------------------------------------------------------------------- #
# Eligibility + main entry point                                               #
# --------------------------------------------------------------------------- #

def _candidate_is_llm_fallback(world_input: Type2WorldModelInput, candidate: Any) -> bool:
    if bool(world_input.metadata.get("used_llm_fallback")):
        return True
    source = _norm(getattr(candidate, "source", "")).lower()
    if "llm_fallback" in source or "skeleton" in source:
        return True
    templates = getattr(candidate, "template_names", None) or []
    if any("llm_fallback" in _norm(t).lower() or "skeleton" in _norm(t).lower() for t in templates):
        return True
    # Loose (non-"=") formula names also signal a semantic plan worth mapping.
    for step in getattr(candidate, "step_plan", None) or []:
        if isinstance(step, dict) and step.get("type") == "formula_application":
            if _loose_name(step.get("formula_name")):
                return True
    return False


def _formula_step(step_id: str, formula: str, output: str, template: str) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "goal": f"Apply canonical formula {formula}.",
        "type": "formula_application",
        "formula_name": formula,
        "template_name": template,
        "input_var": {},
        "output_var": {output: formula},
        "confidence": 0.7,
    }


def _conclusion_step(step_id: str, target: Optional[str], template: str) -> dict[str, Any]:
    out = _norm(target) or "target"
    return {
        "step_id": step_id,
        "goal": f"Report the final value of {out}.",
        "type": "conclusion",
        "template_name": template,
        "input_var": {out: out},
        "output_var": {out: out},
        "confidence": 0.7,
    }


def canonicalize_llm_fallback_candidate(
    world_input: Type2WorldModelInput,
    candidate: Any,
) -> Type2CanonicalizationResult:
    """Map an LLM-fallback/skeleton candidate's loose formulas to canonical ones.

    Returns a result with ``canonical_step_plan`` when at least one safe mapping
    is found (status PASS/WARN). Non-LLM deterministic candidates are never
    modified (status FAIL, empty plan).
    """
    orig_templates = [str(t) for t in (getattr(candidate, "template_names", None) or [])]
    orig_formulas: list[str] = []
    for step in getattr(candidate, "step_plan", None) or []:
        if isinstance(step, dict) and step.get("formula_name"):
            orig_formulas.append(str(step.get("formula_name")))

    result = Type2CanonicalizationResult(
        status="FAIL",
        original_template_names=orig_templates,
        original_formula_names=orig_formulas,
    )

    if not _candidate_is_llm_fallback(world_input, candidate):
        result.errors.append({"error_type": "not_llm_fallback", "message": "Candidate is not an LLM-fallback/skeleton candidate; not canonicalized."})
        return result

    target = world_input.target
    target_unit = world_input.target_unit

    canonical_steps: list[dict[str, Any]] = []
    canonical_formulas: list[str] = []
    canonical_templates: list[str] = []
    mapped = 0
    loose_total = 0

    # Collect loose names from the step plan; if none, fall back to template names.
    loose_names: list[str] = []
    for step in getattr(candidate, "step_plan", None) or []:
        if isinstance(step, dict) and step.get("type") == "formula_application":
            loose = _loose_name(step.get("formula_name"))
            if loose:
                loose_names.append(loose)
    if not loose_names:
        for t in orig_templates:
            lt = _loose_name(t)
            if lt and lt in _PATTERN_INDEX:
                loose_names.append(lt)

    for loose in loose_names:
        loose_total += 1
        resolved = _resolve_canonical(loose, world_input, target)
        if resolved is None:
            result.warnings.append(f"No canonical mapping for '{loose}'; left symbolic.")
            result.mapping_log.append({"source": loose, "status": "unmapped"})
            continue
        if resolved.get("_skip"):
            result.warnings.append(resolved.get("warning", f"'{loose}' not safely mappable; left symbolic."))
            result.mapping_log.append({"source": loose, "status": "skipped", "reason": resolved.get("warning")})
            continue
        step_id = f"canon_step_{len(canonical_steps) + 1}"
        canonical_steps.append(_formula_step(step_id, resolved["formula"], resolved["output"], resolved["template"]))
        canonical_formulas.append(resolved["formula"])
        if resolved["template"] not in canonical_templates:
            canonical_templates.append(resolved["template"])
        result.mapping_log.append({
            "source": loose, "status": "mapped",
            "canonical_formula": resolved["formula"], "canonical_template": resolved["template"],
            "risk": resolved.get("risk", "low"),
        })
        mapped += 1

    if mapped == 0:
        result.status = "FAIL" if loose_total == 0 else "WARN"
        result.errors.append({"error_type": "no_safe_mapping", "message": "No loose formula could be safely canonicalized."})
        return result

    # Append a conclusion step targeting the requested unknown.
    canonical_steps.append(_conclusion_step(f"canon_step_{len(canonical_steps) + 1}", target, canonical_templates[0]))

    result.status = "PASS" if mapped == loose_total else "WARN"
    result.canonical_step_plan = canonical_steps
    result.canonical_formula_names = canonical_formulas
    result.canonical_template_names = canonical_templates or ["llm_fallback_canonicalized"]
    result.metadata = {
        "loose_formula_count": loose_total,
        "mapped_count": mapped,
        "target": target,
        "target_unit": target_unit,
        "llm_original_candidate": {
            "source": _norm(getattr(candidate, "source", "")),
            "template_names": orig_templates,
            "formula_names": orig_formulas,
        },
    }
    return result