"""Type 2 Stage 2+3: Step-by-Step Generation and Calibration (design §5 Stage 2+3).

Stages 2 and 3 are merged into a single interleaved loop: generate one step,
verify it immediately, then advance.  This prevents wasted computation on steps
already known to be downstream of an error.

Always-available (no LLM dependency):
  init_vso(parse_obj)                     → Dict[str, VSOEntry]
  classify_checkable(type, ids, vso)      → bool
  sympy_verify_step(entry, vals, answer)  → (verdict, confidence)

DSPy-guarded (requires dspy-ai + configured LM):
  PhysicsSolverSignature
  StepVerificationSignature
  StepGeneratorModule
  StepVerifierLLMModule
  SolveTrace(dspy.Module)

Confidence values assigned by verifier (design §5 Stage 2+3):
  SymPy/Wolfram CORRECT                  → 1.0
  SymPy/Wolfram CORRECT after repair     → 0.6
  LLM verifier CORRECT                   → 0.8
  LLM verifier UNCERTAIN                 → 0.5
  not-checkable setup/conclusion         → 0.9
  not-checkable unit_conversion          → 0.85
"""

from __future__ import annotations

import json
import keyword
import logging
import math
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from parser.schemas import ProblemParseObject

from .schemas import FormulaEntry, FormulaSet, StepObject, TraceObject, VSOEntry
from .stage1 import canonicalize_variable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import dspy as _dspy
    _DSPY_AVAILABLE = True
except ModuleNotFoundError:
    _dspy = None  # type: ignore[assignment]
    _DSPY_AVAILABLE = False

try:
    import sympy as _sym
    _SYMPY_AVAILABLE = True
except ImportError:
    _sym = None  # type: ignore[assignment]
    _SYMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Physics constants (design §5 Stage 0 step 4 / §6 Variable State Management)
# ---------------------------------------------------------------------------

try:
    import scipy.constants as _sc
    PHYSICS_CONSTANTS: Dict[str, Dict] = {
        "g":         {"value": _sc.g,           "unit_symbol": "m/s^2",    "unit_name": "metres per second squared"},
        "c":         {"value": _sc.c,            "unit_symbol": "m/s",      "unit_name": "metres per second"},
        "h_planck":  {"value": _sc.h,            "unit_symbol": "J·s",      "unit_name": "joule seconds"},
        "e":         {"value": _sc.e,            "unit_symbol": "C",        "unit_name": "coulombs"},
        "m_e":       {"value": _sc.m_e,          "unit_symbol": "kg",       "unit_name": "kilograms"},
        "N_A":       {"value": _sc.N_A,          "unit_symbol": "mol^-1",   "unit_name": "per mole"},
        "k_B":       {"value": _sc.k,            "unit_symbol": "J/K",      "unit_name": "joules per kelvin"},
        "k_e":       {"value": 9.0e9,
                      "unit_symbol": "N·m²/C²", "unit_name": "newton metre squared per coulomb squared"},
        "epsilon_0": {"value": _sc.epsilon_0,    "unit_symbol": "F/m",      "unit_name": "farads per metre"},
        "mu_0":      {"value": _sc.mu_0,          "unit_symbol": "T*m/A",    "unit_name": "tesla metre per ampere"},
    }
except ImportError:
    # scipy not installed — use hand-coded CODATA values as fallback
    PHYSICS_CONSTANTS = {
        "g":         {"value": 9.80665,      "unit_symbol": "m/s^2",    "unit_name": "metres per second squared"},
        "c":         {"value": 299792458.0,  "unit_symbol": "m/s",      "unit_name": "metres per second"},
        "k_e":       {"value": 9.0e9, "unit_symbol": "N·m²/C²","unit_name": "newton metre squared per coulomb squared"},
        "epsilon_0": {"value": 8.8541878128e-12,"unit_symbol": "F/m",  "unit_name": "farads per metre"},
        "mu_0":      {"value": 4 * math.pi * 1e-7, "unit_symbol": "T*m/A", "unit_name": "tesla metre per ampere"},
    }

# ---------------------------------------------------------------------------
# VSO initialization
# ---------------------------------------------------------------------------

def init_vso(parse_obj: ProblemParseObject) -> Dict[str, VSOEntry]:
    """Build the initial Variable State Object for a problem.

    Populates from ``parse_obj.known_quantities`` (extracted by Stage 0) and
    adds universally known physics constants.  Constants only appear if not
    already overridden by the problem's own quantities.
    """
    vso: Dict[str, VSOEntry] = {}

    # Physics constants (lowest priority — problem quantities take precedence)
    for name, meta in PHYSICS_CONSTANTS.items():
        vso[name] = VSOEntry(
            value=meta["value"],
            unit_symbol=meta["unit_symbol"],
            unit_name=meta["unit_name"],
            defined_at="constants_table",
            updated_at="constants_table",
        )

    # Known quantities from Stage 0 (override constants where names collide)
    preserve_measurement_units = "measurement_error" in set(parse_obj.sub_domains or []) or "measurement_error" in set(parse_obj.domains or [])
    for var_name, qty in parse_obj.known_quantities.items():
        value = qty.get("value") if preserve_measurement_units else qty.get("normalized_value")
        source_text = str(qty.get("source_text", ""))
        if "±" in source_text or "+/-" in source_text:
            value = qty.get("value")
        if value is None:
            value = qty.get("value")
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        vso[var_name] = VSOEntry(
            value=value,
            unit_symbol=str(qty.get("unit_symbol", "") if preserve_measurement_units else (qty.get("normalized_unit_symbol") or qty.get("unit_symbol", ""))),
            unit_name=str(qty.get("unit_name", "")),
            defined_at="stage0",
            updated_at="stage0",
        )

    # Also seed from parse_obj.vso if already populated by Stage 0
    for var_name, entry in parse_obj.vso.items():
        if var_name in vso:
            continue   # don't overwrite known_quantities
        val = entry.get("normalized_value")
        source_text = str(entry.get("source_text", ""))
        if "±" in source_text or "+/-" in source_text:
            val = entry.get("value")
        if val is None:
            val = entry.get("value")
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        vso[var_name] = VSOEntry(
            value=val,
            unit_symbol=str(entry.get("normalized_unit_symbol") or entry.get("unit_symbol", "")),
            unit_name=str(entry.get("unit_name", "")),
            defined_at=entry.get("defined_at", "stage0"),
            updated_at=entry.get("updated_at", "stage0"),
        )

    return vso


# ---------------------------------------------------------------------------
# Checkable classification (design §5 Stage 2+3)
# ---------------------------------------------------------------------------

_CHECKABLE_STEP_TYPES = {"calculation", "formula_application", "unit_conversion"}


def classify_checkable(
    step_type: str,
    formula_ids: List[str],
    input_var_values: Dict[str, Optional[float]],
) -> bool:
    """Return True if this step should be verified by a deterministic tool.

    Rules (design §5):
    - step type must be calculation, formula_application, or unit_conversion
    - formula_ids must be non-empty
    - all input_var values must be present (not None) in the VSO
    """
    if step_type not in _CHECKABLE_STEP_TYPES:
        return False
    if not formula_ids:
        return False
    if not input_var_values:
        return False
    if any(v is None for v in input_var_values.values()):
        return False
    return True


# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(
    r"([+-]?\s*\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def _extract_numeric(text: str) -> Optional[float]:
    """Extract the first numeric value from a string like '5.0 V' or '-3.2e-4 C'.

    Returns None if no number is found.
    """
    text = text.strip()
    m = _NUMERIC_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ".").replace(" ", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# SymPy verification (design §5 Stage 2+3 — Immediate verification)
# ---------------------------------------------------------------------------

def _build_sympy_namespace(formula_entry: FormulaEntry) -> Optional[Dict]:
    """Create a SymPy symbol namespace for the formula's variables."""
    if not _SYMPY_AVAILABLE:
        return None
    sym_ns: Dict = {}
    for var_name in formula_entry.variables:
        sym_ns[var_name] = _sym.Symbol(var_name)
    # Pre-populate known physical constants as floats so they can be left
    # un-substituted and still evaluate numerically.
    sym_ns["k_e"] = _sym.Float(9.0e9)
    sym_ns["epsilon_0"] = _sym.Float(8.8541878128e-12)
    sym_ns["pi"] = _sym.pi
    return sym_ns


def sympy_verify_step(
    formula_entry: FormulaEntry,
    formula_var_values: Dict[str, float],
    intermediate_answer: str,
) -> Tuple[str, float]:
    """Verify a computation step using SymPy (design §5 Stage 3).

    Parameters
    ----------
    formula_entry:
        The formula library entry used in this step.
    formula_var_values:
        Mapping from formula variable names to known numeric values.
        The unknown variable (being solved for) must be absent from this dict.
    intermediate_answer:
        The LLM's claimed result, e.g. ``"5.0 V"`` or ``"25 J"``.

    Returns
    -------
    (verdict, confidence)
        verdict is one of ``"CORRECT"``, ``"INCORRECT"``, ``"UNCERTAIN"``.
        confidence is 1.0 / 0.0 / 0.5 respectively.
    """
    if not _SYMPY_AVAILABLE:
        return "UNCERTAIN", 0.5

    if formula_entry.tool_dispatch != "sympy" or not formula_entry.sympy_expr:
        return "UNCERTAIN", 0.5

    target_value = _extract_numeric(intermediate_answer)
    if target_value is None:
        return "UNCERTAIN", 0.5

    try:
        sym_ns = _build_sympy_namespace(formula_entry)
        if sym_ns is None:
            return "UNCERTAIN", 0.5

        # Eval the sympy_expr string ("Eq(V, I * R)") in the symbol namespace
        safe_globals = {k: getattr(_sym, k) for k in dir(_sym) if not k.startswith("_")}
        safe_globals.update(sym_ns)
        equation = eval(formula_entry.sympy_expr, {"__builtins__": {}}, safe_globals)  # noqa: S307

        # Substitute known values
        subs = {sym_ns[k]: _sym.Float(v) for k, v in formula_var_values.items() if k in sym_ns}
        eq_subs = equation.subs(subs)

        # Identify the unknown variable (not substituted)
        unknown_symbols = [
            sym_ns[k]
            for k in formula_entry.variables
            if k not in formula_var_values and k in sym_ns
        ]
        if len(unknown_symbols) != 1:
            return "UNCERTAIN", 0.5

        solutions = _sym.solve(eq_subs, unknown_symbols[0])
        if not solutions:
            return "UNCERTAIN", 0.5

        computed = float(_sym.N(solutions[0]))

        # Compare with relative + absolute tolerance
        abs_tol = max(abs(computed) * 1e-3, 1e-9)
        if abs(computed - target_value) <= abs_tol:
            return "CORRECT", 1.0
        else:
            return "INCORRECT", 0.0

    except Exception as exc:
        logger.debug("SymPy verification failed for %s: %s", formula_entry.id, exc)
        return "UNCERTAIN", 0.5


def _target_formula_var(
    formula_entry: FormulaEntry,
    plan_output_var: Dict[str, object],
    formula_var_values: Dict[str, float],
) -> Optional[str]:
    """Pick the formula variable that should be solved for."""
    for name in plan_output_var:
        if name in formula_entry.variables:
            return name

    for name in plan_output_var:
        target_canon = canonicalize_variable(name)
        if not target_canon:
            continue
        for formula_var in formula_entry.variables:
            if formula_var in formula_var_values:
                continue
            if canonicalize_variable(formula_var) == target_canon:
                return formula_var

    unknowns = [
        name
        for name in formula_entry.variables
        if name not in formula_var_values
    ]
    if len(unknowns) == 1:
        return unknowns[0]
    return None


def _solve_formula_var(
    formula_entry: FormulaEntry,
    formula_var_values: Dict[str, float],
    target_var: str,
) -> Optional[float]:
    """Solve one FormulaEntry sympy equation for target_var."""
    if not _SYMPY_AVAILABLE:
        return None
    if formula_entry.tool_dispatch != "sympy" or not formula_entry.sympy_expr:
        return None

    try:
        sym_ns = _build_sympy_namespace(formula_entry)
        if sym_ns is None or target_var not in sym_ns:
            return None

        safe_globals = {k: getattr(_sym, k) for k in dir(_sym) if not k.startswith("_")}
        safe_globals.update(sym_ns)
        equation = eval(formula_entry.sympy_expr, {"__builtins__": {}}, safe_globals)  # noqa: S307
        subs = {
            sym_ns[k]: _sym.Float(v)
            for k, v in formula_var_values.items()
            if k in sym_ns and k != target_var
        }
        equation = equation.subs(subs)
        solutions = _sym.solve(equation, sym_ns[target_var])
        numeric_solutions = [
            float(_sym.N(sol))
            for sol in solutions
            if getattr(_sym.N(sol), "is_real", False) is not False
        ]
        if not numeric_solutions:
            return None
        non_negative = [value for value in numeric_solutions if value >= 0]
        return non_negative[0] if non_negative else numeric_solutions[0]
    except Exception as exc:
        logger.debug("SymPy solve failed for %s: %s", formula_entry.id, exc)
        return None


def _try_sympy_solve_step(
    step: StepObject,
    formula_entry: Optional[FormulaEntry],
    plan_output_var: Dict[str, object],
    vso: Dict[str, VSOEntry],
) -> bool:
    """Populate a step via deterministic SymPy solve when possible."""
    if formula_entry is None:
        return False

    formula_var_values = map_formula_vars_to_vso(formula_entry, vso)
    target_var = _target_formula_var(formula_entry, plan_output_var, formula_var_values)
    if target_var is None:
        return False

    solved = _solve_formula_var(formula_entry, formula_var_values, target_var)
    if solved is None:
        return False

    output_names = list(plan_output_var.keys()) or [target_var]
    primary_output = target_var
    for output_name in output_names:
        if output_name == target_var:
            primary_output = output_name
            break
        out_canon = canonicalize_variable(output_name)
        target_canon = canonicalize_variable(target_var)
        if out_canon and target_canon and out_canon == target_canon:
            primary_output = output_name
            break

    unit_sym, _ = _infer_output_unit(primary_output, formula_entry, vso)
    step.step_input = (
        f"Solved {formula_entry.formula} for {target_var} using "
        f"{json.dumps(formula_var_values, ensure_ascii=False)}"
    )
    step.intermediate_answer = f"{solved:g} {unit_sym}".strip()
    step.output_var = {primary_output: solved}
    if primary_output != target_var:
        step.output_var[target_var] = solved
    step.confidence = 1.0
    step.status = "OK"
    step.verifier_notes = "Solved deterministically with SymPy fallback."
    return True


def _condition_violation_reason(
    step: StepObject,
    formula_entry: Optional[FormulaEntry],
    parse_obj: ProblemParseObject,
) -> Optional[str]:
    """Return a verifier failure reason when a formula violates conditions."""

    problem_text = str(parse_obj.problem_text or "")
    condition_text = " ".join(str(cond or "") for cond in (parse_obj.conditions or []))
    problem_context = f"{problem_text} {condition_text}".lower()
    step_context = " ".join(
        str(part or "")
        for part in (
            step.goal,
            step.step_input,
            step.intermediate_answer,
            step.verifier_notes,
            " ".join(step.formula_ids),
            formula_entry.id if formula_entry else "",
            formula_entry.text if formula_entry else "",
            formula_entry.formula if formula_entry else "",
            " ".join(formula_entry.conditions) if formula_entry else "",
        )
    ).lower()

    explicit_not_resonance = any(
        marker in problem_context
        for marker in ("not in resonance", "not at resonance", "not resonant")
    )
    resonance_shortcut = any(
        marker in step_context
        for marker in (
            "z = r",
            "impedance equals pure resistance",
            "pure resistance equals impedance",
            "at resonance",
            "x_l = x_c",
            "xl = xc",
        )
    )
    if explicit_not_resonance and resonance_shortcut:
        return (
            "Condition violation: problem explicitly states not in resonance, "
            "but this step used a resonance shortcut such as Z = R."
        )

    return None


def _apply_condition_validation(
    step: StepObject,
    formula_entry: Optional[FormulaEntry],
    parse_obj: ProblemParseObject,
) -> bool:
    """Mark a step WRONG if deterministic condition checks reject it."""

    reason = _condition_violation_reason(step, formula_entry, parse_obj)
    if not reason:
        return True
    step.status = "WRONG"
    step.confidence = 0.0
    step.verifier_notes = reason
    return False


def _format_direct_answer(var_name: str, value: float, vso: Dict[str, VSOEntry]) -> str:
    """Format deterministic template arithmetic output with a light unit guess."""
    if var_name in {"percent_error", "I_over_Imax_percent", "electric_energy_fraction_percent"}:
        return f"{value:g} %"
    if var_name == "B" and 0 < abs(value) < 0.1:
        return f"{value * 1e3:g} mT"
    if var_name == "L_ind" and 0 < abs(value) < 1:
        return f"{value * 1e3:g} mH"
    if var_name in {"Phi_B", "Phi_link"} and 0 < abs(value) < 1e-5:
        return f"{value * 1e6:g} μWb"
    if var_name in {"u_B"}:
        return f"{value:g} J/m^3"
    if var_name in {"U_B", "U_E", "U_total"}:
        return f"{value:g} J"
    if var_name in {"R", "R_eq", "Z", "X_L", "X_C"}:
        return f"{value:g} ohm"
    unit_sym = ""
    if var_name in {"abs_error", "mean_value"}:
        for candidate in vso.values():
            if candidate.defined_at != "constants_table" and candidate.unit_symbol:
                unit_sym = candidate.unit_symbol
                break
    return f"{value:g} {unit_sym}".strip()


def _measurement_unit(vso: Dict[str, VSOEntry]) -> str:
    for candidate in vso.values():
        if candidate.defined_at != "constants_table" and candidate.unit_symbol:
            return candidate.unit_symbol
    return ""


def _format_measurement_pair_answer(target_var: str, vso: Dict[str, VSOEntry]) -> Optional[str]:
    unit_sym = _measurement_unit(vso)
    if target_var == "mean_abs_error_pair":
        mean_entry = vso.get("mean_value")
        abs_entry = vso.get("abs_error")
        if mean_entry and abs_entry:
            return f"{mean_entry.value:g}; {abs_entry.value:g} {unit_sym}; {unit_sym}".strip()
    if target_var == "abs_rel_error_pair":
        abs_entry = vso.get("abs_error")
        rel_entry = vso.get("rel_error")
        if abs_entry and rel_entry:
            return f"{abs_entry.value:g}; {rel_entry.value * 100:g} {unit_sym}; %".strip()
    return None


def _format_capacitor_pair_answer(target_var: str, vso: Dict[str, VSOEntry]) -> Optional[str]:
    if target_var != "energy_charge_pair":
        return None
    energy_entry = vso.get("U_cap") or vso.get("U_E")
    charge_entry = vso.get("Q") or vso.get("q")
    if not energy_entry or not charge_entry:
        return None
    return f"{energy_entry.value * 1e6:g}; {charge_entry.value * 1e6:g} uJ; uC"


def _lookup_numeric_value(
    var_name: str,
    plan_item: Dict[str, object],
    vso: Dict[str, VSOEntry],
) -> Optional[float]:
    """Resolve a template formula variable against VSO with a few measurement aliases."""
    if var_name == "k":
        found = _lookup_vso_entry("k_e", vso)
        if found:
            return found[1].value

    found = _lookup_vso_entry(var_name, vso)
    if found:
        return found[1].value

    aliases = {
        "true_value": ["accepted_value", "actual_value"],
        "accepted_value": ["true_value", "actual_value"],
        "actual_value": ["true_value", "accepted_value"],
        "measured_value": ["measured_value"],
    }
    for alias in aliases.get(var_name, []):
        found = _lookup_vso_entry(alias, vso)
        if found:
            return found[1].value

    # Some measurement templates use a canonical placeholder in the expression
    # while the input list carries the extracted name.  Fall back to the first
    # non-measured input for true/accepted/actual values.
    if var_name in {"true_value", "accepted_value", "actual_value"}:
        for input_name in plan_item.get("input_var", {}):
            if str(input_name) == "measured_value":
                continue
            found = _lookup_vso_entry(str(input_name), vso)
            if found:
                return found[1].value

    input_names = [str(name) for name in plan_item.get("input_var", {})]

    def _inputs_by_dimension(dimension: str) -> List[VSOEntry]:
        entries: List[VSOEntry] = []
        for input_name in input_names:
            found = _lookup_vso_entry(input_name, vso)
            if found and canonicalize_variable(found[0]) == dimension:
                entries.append(found[1])
        return entries

    if re.fullmatch(r"F\d+", var_name):
        forces = _inputs_by_dimension("force")
        index = int(var_name[1:]) - 1
        if 0 <= index < len(forces):
            return forces[index].value
    if re.fullmatch(r"q\d+", var_name) or var_name == "q":
        charges = _inputs_by_dimension("charge")
        if not charges:
            for input_name in input_names:
                if not re.fullmatch(r"q\w*", input_name):
                    continue
                found = _lookup_vso_entry(input_name, vso)
                if found:
                    charges.append(found[1])
        index = int(var_name[1:]) - 1 if var_name != "q" else 0
        if 0 <= index < len(charges):
            return charges[index].value
        if charges:
            return charges[-1].value
    if var_name in {"r", "d", "r13", "r23"}:
        distances = _inputs_by_dimension("distance")
        if not distances:
            distances = _inputs_by_dimension("displacement")
        if var_name == "r23":
            index = 1
        else:
            index = 0
        if 0 <= index < len(distances):
            return distances[index].value
        if distances:
            return distances[-1].value
        numeric_inputs: List[float] = []
        for input_name in input_names:
            if input_name == "k" or re.fullmatch(r"[qF]\d*", input_name) or input_name in {"theta"}:
                continue
            found = _lookup_vso_entry(input_name, vso)
            if found:
                numeric_inputs.append(found[1].value)
        if 0 <= index < len(numeric_inputs):
            return numeric_inputs[index]
        if numeric_inputs:
            return numeric_inputs[-1]
    return None


def _measurement_input_values(
    plan_item: Dict[str, object],
    vso: Dict[str, VSOEntry],
    include_mean: bool = False,
) -> List[float]:
    values: List[float] = []
    for input_name in plan_item.get("input_var", {}):
        name = str(input_name)
        if name == "mean_value" and not include_mean:
            continue
        found = _lookup_vso_entry(name, vso)
        if found:
            values.append(found[1].value)
    return values


_TEMPLATE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")


def _try_template_arithmetic_step(
    step: StepObject,
    plan_item: Dict[str, object],
    vso: Dict[str, VSOEntry],
) -> bool:
    """Execute parser-template arithmetic directly before falling back to formula retrieval."""
    formula_text = str(plan_item.get("formula_name") or "")
    if "=" not in formula_text:
        return False

    output_names = list((plan_item.get("output_var") or {}).keys())
    lhs, rhs = formula_text.split("=", 1)
    output_name = output_names[0] if output_names else lhs.strip()
    rhs = rhs.strip().replace("^", "**")

    if str(plan_item.get("template_name")) == "square_field_cancellation":
        step.step_input = f"Evaluated symbolic template formula: {formula_text}"
        step.intermediate_answer = rhs
        step.output_var = {output_name: rhs}
        step.confidence = 1.0
        step.status = "OK"
        step.verifier_notes = "Solved by deterministic symbolic geometry template."
        return True

    if output_name in {"mean_abs_error_pair", "abs_rel_error_pair"}:
        answer = _format_measurement_pair_answer(output_name, vso)
        if not answer:
            return False
        step.step_input = f"Evaluated measurement pair template: {formula_text}"
        step.intermediate_answer = answer
        step.output_var = {}
        step.confidence = 1.0
        step.status = "OK"
        step.verifier_notes = "Formatted deterministic measurement pair answer."
        return True

    if output_name == "energy_charge_pair":
        answer = _format_capacitor_pair_answer(output_name, vso)
        if not answer:
            return False
        step.step_input = f"Evaluated capacitor pair template: {formula_text}"
        step.intermediate_answer = answer
        step.output_var = {output_name: answer}
        step.confidence = 1.0
        step.status = "OK"
        step.verifier_notes = "Formatted deterministic capacitor energy-charge pair answer."
        return True

    value: Optional[float] = None
    if "sum(measurements)" in rhs:
        measurements = _measurement_input_values(plan_item, vso)
        if measurements:
            value = sum(measurements) / len(measurements)
    elif "sum(abs(each - mean_value))" in rhs:
        mean_found = _lookup_vso_entry("mean_value", vso)
        measurements = _measurement_input_values(plan_item, vso)
        if mean_found and measurements:
            mean_value = mean_found[1].value
            value = sum(abs(x - mean_value) for x in measurements) / len(measurements)
    elif "abs(each measurement - mean_value)" in rhs:
        mean_found = _lookup_vso_entry("mean_value", vso)
        measurements = _measurement_input_values(plan_item, vso)
        if mean_found and measurements:
            mean_value = mean_found[1].value
            value = max(abs(x - mean_value) for x in measurements)
    elif "max(deviations)" in rhs:
        found = _lookup_vso_entry("deviations", vso)
        if found:
            value = found[1].value
    elif rhs.startswith("vector_sum(") and rhs.endswith(")"):
        arg_text = rhs[len("vector_sum("):-1]
        values: List[float] = []
        for arg in [part.strip() for part in arg_text.split(",")]:
            arg_value = _lookup_numeric_value(arg, plan_item, vso)
            if arg_value is None:
                return False
            values.append(abs(arg_value))
        if values:
            value = sum(values)
    elif "each" in rhs or "measurements" in rhs:
        return False
    else:
        rhs = re.sub(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*deg\b",
            lambda match: str(math.radians(float(match.group(1)))),
            rhs,
        )
        namespace = {
            "abs": abs,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "acos": math.acos,
            "round": round,
            "pi": math.pi,
        }
        safe_rhs = rhs
        identifiers = set(_TEMPLATE_IDENTIFIER_RE.findall(rhs))
        for token in identifiers:
            if token in namespace:
                continue
            token_value = _lookup_numeric_value(token, plan_item, vso)
            if token_value is None:
                return False
            safe_token = token if not keyword.iskeyword(token) else f"__var_{token}"
            namespace[safe_token] = token_value
            safe_rhs = re.sub(rf"\b{re.escape(token)}\b", safe_token, safe_rhs)
        try:
            raw_value = eval(safe_rhs, {"__builtins__": {}}, namespace)  # noqa: S307
            if isinstance(raw_value, (int, float)) and math.isfinite(float(raw_value)):
                value = float(raw_value)
        except Exception as exc:
            logger.debug("Template arithmetic failed for %s: %s", formula_text, exc)
            return False

    if value is None:
        return False

    step.step_input = f"Evaluated template formula: {formula_text}"
    step.intermediate_answer = _format_direct_answer(output_name, value, vso)
    step.output_var = {output_name: value}
    step.confidence = 1.0
    step.status = "OK"
    step.verifier_notes = "Solved deterministically from parser template arithmetic."
    return True


def _try_setup_outputs(
    step: StepObject,
    plan_item: Dict[str, object],
) -> bool:
    """Accept setup steps that already carry concrete numeric extracted outputs."""
    output_values: Dict[str, float] = {}
    for name, raw_value in (plan_item.get("output_var") or {}).items():
        value: Optional[float]
        if isinstance(raw_value, (int, float)):
            value = float(raw_value)
        else:
            raw_text = str(raw_value).strip()
            if not re.match(r"^[+-]?\s*\d", raw_text):
                value = None
            else:
                value = _extract_numeric(raw_text)
        if value is not None:
            output_values[str(name)] = value
    if not output_values:
        return False
    step.output_var = output_values
    step.intermediate_answer = "; ".join(f"{name}={value:g}" for name, value in output_values.items())
    step.status = "OK"
    step.confidence = 0.95
    step.verifier_notes = "Accepted numeric setup output from parser template."
    return True


def map_formula_vars_to_vso(
    formula_entry: FormulaEntry,
    vso: Dict[str, VSOEntry],
) -> Dict[str, float]:
    """Map formula variable names to numeric values using the VSO.

    Uses two strategies in priority order:
    1. Direct name match: formula variable name exists as-is in the VSO.
    2. Canonical match: formula variable and VSO variable share the same
       canonical physical quantity name (from ``stage1.canonicalize_variable``).

    Returns a dict of {formula_var_name: numeric_value} for all matches found.
    """
    result: Dict[str, float] = {}

    # Build canonical → vso_var_name mapping once
    canonical_to_vso: Dict[str, str] = {}
    for vso_var, entry in vso.items():
        if entry.defined_at == "constants_table":
            continue
        canon = canonicalize_variable(vso_var)
        if canon and canon not in canonical_to_vso:
            canonical_to_vso[canon] = vso_var

    used_vso_vars: set[str] = set()
    for formula_var in formula_entry.variables:
        # Strategy 1: direct name
        if formula_var in vso:
            result[formula_var] = vso[formula_var].value
            used_vso_vars.add(formula_var)
            continue
        # Strategy 2: canonical
        formula_canon = canonicalize_variable(formula_var)
        if formula_canon and formula_canon in canonical_to_vso:
            vso_var = canonical_to_vso[formula_canon]
            if vso_var not in used_vso_vars:
                result[formula_var] = vso[vso_var].value
                used_vso_vars.add(vso_var)

    return result


# ---------------------------------------------------------------------------
# Confidence values per verifier outcome (design §5 Stage 2+3)
# ---------------------------------------------------------------------------

def _verifier_confidence(
    step_type: str,
    verdict: str,
    was_repaired: bool = False,
    used_llm: bool = False,
) -> float:
    if not step_type.startswith(("calc", "formula", "unit")):
        # Not checkable
        if step_type == "unit_conversion":
            return 0.85
        return 0.9  # setup / conclusion
    if verdict == "CORRECT":
        if was_repaired:
            return 0.6
        if used_llm:
            return 0.8
        return 1.0
    if verdict == "UNCERTAIN":
        return 0.5
    return 0.0  # INCORRECT


# ---------------------------------------------------------------------------
# DSPy Signatures and Modules (guarded)
# ---------------------------------------------------------------------------

if _DSPY_AVAILABLE:

    class PhysicsSolverSignature(_dspy.Signature):
        """Generate one step of a physics solution trace.

        /no_think

        Return only the requested DSPy output fields.
        Do not include <think> or </think>.
        Do not include markdown code fences.
        Do not include extra natural language outside the fields.
        Do not reveal chain-of-thought.

        You must use the formula's variable names when writing step_input and
        output_values_json.  Do not invent new variable names.

        output_values_json must be a valid JSON object mapping formula variable
        names to their computed numeric values (floats only, no units in the JSON).
        """

        problem_text: str = _dspy.InputField(desc="Full physics problem statement")
        prior_steps_summary: str = _dspy.InputField(
            desc="Brief summary of completed steps and their results (empty if first step)"
        )
        step_goal: str = _dspy.InputField(desc="What this step must compute")
        step_type: str = _dspy.InputField(
            desc="Step type: setup | formula_application | calculation | unit_conversion | conclusion"
        )
        formula_text: str = _dspy.InputField(
            desc="Formula to apply, e.g. 'V = I * R'.  Empty string if no formula."
        )
        formula_variables: str = _dspy.InputField(
            desc="JSON mapping each formula variable to its name and unit, e.g. "
                 '{"V": {"name": "voltage", "unit": "V"}, ...}'
        )
        available_values: str = _dspy.InputField(
            desc="JSON mapping known variable names to their numeric values"
        )

        thought: str = _dspy.OutputField(
            desc="One short physical reason for this step. Do not include <think> or chain-of-thought."
        )
        step_input: str = _dspy.OutputField(
            desc="The mathematical expression or substitution performed, e.g. 'V = I*R = 0.5*10'"
        )
        intermediate_answer: str = _dspy.OutputField(
            desc="The numeric result with its unit, e.g. '5.0 V'"
        )
        output_values_json: str = _dspy.OutputField(
            desc='JSON object of computed values using formula variable names, e.g. {"V": 5.0}'
        )

    class StepVerificationSignature(_dspy.Signature):
        """Verify whether a physics calculation step is correct.

        /no_think

        Respond with verdict CORRECT, INCORRECT, or UNCERTAIN only.
        Do not include <think> or </think>.
        Do not include markdown code fences.
        If INCORRECT, explain the correction briefly in 'correction'.
        """

        step_goal: str = _dspy.InputField(desc="What the step was trying to compute")
        formula_text: str = _dspy.InputField(desc="Formula applied, or empty string")
        input_values: str = _dspy.InputField(desc="JSON of input variable values")
        step_input: str = _dspy.InputField(desc="Mathematical work shown by the solver")
        intermediate_answer: str = _dspy.InputField(desc="Result claimed by the solver")

        verdict: str = _dspy.OutputField(
            desc="Exactly one of: CORRECT, INCORRECT, UNCERTAIN"
        )
        correction: str = _dspy.OutputField(
            desc="If INCORRECT: the correct answer. Otherwise empty string."
        )

    class StepGeneratorModule(_dspy.Module):
        """Wraps PhysicsSolverSignature with Predict.

        Use Predict instead of ChainOfThought so Qwen3 does not spend the
        fallback budget on hidden reasoning or emit <think> blocks before the
        structured fields.
        """

        def __init__(self) -> None:
            self.generator = _dspy.Predict(PhysicsSolverSignature)

        def forward(self, **kwargs) -> "_dspy.Prediction":
            return self.generator(**kwargs)

    class StepVerifierLLMModule(_dspy.Module):
        """Wraps StepVerificationSignature with Predict."""

        def __init__(self) -> None:
            self.verifier = _dspy.Predict(StepVerificationSignature)

        def forward(self, **kwargs) -> "_dspy.Prediction":
            pred = self.verifier(**kwargs)
            # Normalize verdict to uppercase
            raw = str(pred.verdict).strip().upper()
            for v in ("CORRECT", "INCORRECT", "UNCERTAIN"):
                if v in raw:
                    pred.verdict = v
                    break
            else:
                pred.verdict = "UNCERTAIN"
            return pred

    # -----------------------------------------------------------------------
    # Main Stage 2+3 module
    # -----------------------------------------------------------------------

    class SolveTrace(_dspy.Module):
        """Stage 2+3: iterative step generation + immediate verification.

        For each step in the plan:
          1. Populate input values from the VSO.
          2. Generate the step via LLM (PhysicsSolverSignature).
          3. Classify checkable.
          4. If checkable: verify with SymPy first, LLM verifier as fallback.
          5. On INCORRECT: retry up to step_retry_limit; mark WRONG on exhaustion.
          6. On OK / UNCERTAIN / not-checkable: write outputs to VSO, snapshot, advance.

        Returns a TraceObject with trace_status PASS or FAIL.
        """

        def __init__(self) -> None:
            self.generator = StepGeneratorModule()
            self.llm_verifier = StepVerifierLLMModule()

        def forward(
            self,
            parse_obj: ProblemParseObject,
            formula_set: FormulaSet,
            problem_id: str = "unknown",
            step_retry_limit: int = 3,
            trace_budget: int = 10,
        ) -> "TraceObject":
            trace = TraceObject(
                problem_id=problem_id,
                formula_path_index=formula_set.path_index,
            )
            vso = init_vso(parse_obj)
            trace.vso = {k: asdict(v) for k, v in vso.items()}

            total_calls = 0

            for plan_item in parse_obj.step_plan:
                if not isinstance(plan_item, dict):
                    continue

                step_id = plan_item.get("step_id", "?")
                step_type = plan_item.get("type", "calculation")
                step_goal = plan_item.get("goal", "")
                formula_entry: Optional[FormulaEntry] = formula_set.formulas.get(step_id)

                step = StepObject(
                    step_id=step_id,
                    goal=step_goal,
                    type=step_type,
                    formula_ids=[formula_entry.id] if formula_entry else [],
                    input_var={},
                    output_var={},
                )

                # ── Populate input values from VSO ────────────────────────
                for var_name in plan_item.get("input_var", {}):
                    if var_name in vso:
                        step.input_var[var_name] = asdict(vso[var_name])
                    else:
                        step.input_var[var_name] = None

                input_numeric = {
                    k: (v["value"] if isinstance(v, dict) else None)
                    for k, v in step.input_var.items()
                }

                # ── Classify checkable ────────────────────────────────────
                if step_type == "conclusion":
                    target_var = next(iter(plan_item.get("input_var", {}) or plan_item.get("output_var", {}) or {}), None)
                    if target_var in {"mean_abs_error_pair", "abs_rel_error_pair", "energy_charge_pair"}:
                        answer = (
                            _format_capacitor_pair_answer(str(target_var), vso)
                            if target_var == "energy_charge_pair"
                            else _format_measurement_pair_answer(str(target_var), vso)
                        )
                        if answer:
                            step.intermediate_answer = answer
                            step.output_var = {}
                            step.status = "OK"
                            step.confidence = 0.95
                            step.verifier_notes = "Conclusion formatted from measurement pair VSO entries."
                            trace.vso = {k: asdict(v) for k, v in vso.items()}
                            trace.vso_snapshots[step_id] = {k: asdict(v) for k, v in vso.items()}
                            trace.steps.append(step)
                            continue
                    if target_var in vso:
                        target_entry = vso[target_var]
                        if target_var in {"percent_error", "I_over_Imax_percent", "electric_energy_fraction_percent", "B", "L_ind", "Phi_B", "Phi_link"}:
                            step.intermediate_answer = _format_direct_answer(target_var, target_entry.value, vso)
                        else:
                            step.intermediate_answer = f"{target_entry.value:g} {target_entry.unit_symbol}".strip()
                        step.output_var = {target_var: target_entry.value}
                        step.status = "OK"
                        step.confidence = 0.95
                        step.verifier_notes = "Conclusion copied from VSO."
                        trace.vso = {k: asdict(v) for k, v in vso.items()}
                        trace.vso_snapshots[step_id] = {k: asdict(v) for k, v in vso.items()}
                        trace.steps.append(step)
                        continue

                step.checkable = classify_checkable(step_type, step.formula_ids, input_numeric)

                # ── Build generation context ──────────────────────────────
                prior_summary = _build_prior_summary(trace.steps[-3:])
                available_vals = {
                    k: v["value"] if isinstance(v, dict) else None
                    for k, v in step.input_var.items()
                    if v is not None
                }

                formula_text = formula_entry.formula if formula_entry else ""
                formula_vars_json = (
                    json.dumps(formula_entry.variables, ensure_ascii=False)
                    if formula_entry else "{}"
                )

                # ── Generation + verification loop ────────────────────────
                if formula_entry is not None and _try_sympy_solve_step(
                    step=step,
                    formula_entry=formula_entry,
                    plan_output_var=plan_item.get("output_var", {}),
                    vso=vso,
                ):
                    if not _apply_condition_validation(step, formula_entry, parse_obj):
                        trace.steps.append(step)
                        record_first_wrong_step(trace, step, formula_entry, vso, plan_item, parse_obj)
                        trace.trace_status = "FAIL"
                        return trace
                    for out_var, out_val in step.output_var.items():
                        if isinstance(out_val, (int, float)) and out_val is not None:
                            unit_sym, unit_nm = _infer_output_unit(out_var, formula_entry, vso)
                            if out_var in vso:
                                entry = VSOEntry(
                                    value=float(out_val),
                                    unit_symbol=unit_sym,
                                    unit_name=unit_nm,
                                    defined_at=vso[out_var].defined_at,
                                    updated_at=step_id,
                                )
                            else:
                                entry = VSOEntry(
                                    value=float(out_val),
                                    unit_symbol=unit_sym,
                                    unit_name=unit_nm,
                                    defined_at=step_id,
                                    updated_at=step_id,
                                )
                            vso[out_var] = entry

                    trace.vso = {k: asdict(v) for k, v in vso.items()}
                    trace.vso_snapshots[step_id] = {k: asdict(v) for k, v in vso.items()}
                    trace.steps.append(step)
                    continue

                repaired = False
                verifier_feedback = ""

                for attempt in range(step_retry_limit):
                    total_calls += 1
                    if total_calls > trace_budget:
                        logger.warning("Trace budget exceeded at step %s.", step_id)
                        step.status = "WRONG"
                        step.verifier_notes = "Trace budget exceeded."
                        trace.steps.append(step)
                        record_first_wrong_step(trace, step, formula_entry, vso, plan_item, parse_obj)
                        trace.trace_status = "FAIL"
                        return trace

                    # Include verifier feedback in prompt on retries
                    goal_with_feedback = step_goal
                    if attempt > 0 and verifier_feedback:
                        goal_with_feedback = (
                            f"{step_goal}\n\n"
                            f"[Previous attempt was wrong: {verifier_feedback}. "
                            "Please correct your calculation.]"
                        )

                    try:
                        pred = self.generator(
                            problem_text=parse_obj.problem_text,
                            prior_steps_summary=prior_summary,
                            step_goal=goal_with_feedback,
                            step_type=step_type,
                            formula_text=formula_text,
                            formula_variables=formula_vars_json,
                            available_values=json.dumps(available_vals),
                        )
                    except Exception as exc:
                        logger.warning("Step generator failed at %s attempt %d: %s", step_id, attempt, exc)
                        if _try_sympy_solve_step(
                            step=step,
                            formula_entry=formula_entry,
                            plan_output_var=plan_item.get("output_var", {}),
                            vso=vso,
                        ):
                            break
                        step.status = "WRONG"
                        step.verifier_notes = f"Generator error: {exc}"
                        break

                    step.thought = getattr(pred, "thought", "")
                    step.step_input = getattr(pred, "step_input", "")
                    step.intermediate_answer = getattr(pred, "intermediate_answer", "")

                    # Parse output values from JSON
                    step.output_var = _parse_output_values(
                        getattr(pred, "output_values_json", "{}"),
                        plan_item.get("output_var", {}),
                    )

                    # ── Verification ──────────────────────────────────────
                    verdict, confidence = _verify_step(
                        step=step,
                        formula_entry=formula_entry,
                        vso=vso,
                        llm_verifier=self.llm_verifier,
                        was_repaired=(attempt > 0),
                    )

                    step.confidence = confidence

                    if verdict in ("CORRECT", "UNCERTAIN") or not step.checkable:
                        step.status = "OK" if verdict == "CORRECT" else "UNCERTAIN"
                        if attempt > 0:
                            step.status = "REPAIRED"
                        break

                    # INCORRECT: set up for retry
                    logger.info(
                        "Step %s attempt %d INCORRECT; retrying (%d left).",
                        step_id, attempt + 1, step_retry_limit - attempt - 1,
                    )
                    verifier_feedback = getattr(pred, "correction", "incorrect result")
                    repaired = True

                else:
                    # All retries exhausted
                    step.status = "WRONG"
                    step.verifier_notes = f"Failed after {step_retry_limit} attempts."
                    trace.steps.append(step)
                    record_first_wrong_step(trace, step, formula_entry, vso, plan_item, parse_obj)
                    trace.trace_status = "FAIL"
                    return trace

                # ── Write output vars to VSO ──────────────────────────────
                if step.status != "WRONG" and not _apply_condition_validation(step, formula_entry, parse_obj):
                    trace.steps.append(step)
                    record_first_wrong_step(trace, step, formula_entry, vso, plan_item, parse_obj)
                    trace.trace_status = "FAIL"
                    return trace

                if step.status == "WRONG":
                    trace.steps.append(step)
                    record_first_wrong_step(trace, step, formula_entry, vso, plan_item, parse_obj)
                    trace.trace_status = "FAIL"
                    return trace

                for out_var, out_val in step.output_var.items():
                    if isinstance(out_val, (int, float)) and out_val is not None:
                        # Infer unit from formula variables or carry existing unit
                        unit_sym, unit_nm = _infer_output_unit(out_var, formula_entry, vso)
                        if out_var in vso:
                            entry = VSOEntry(
                                value=float(out_val),
                                unit_symbol=unit_sym,
                                unit_name=unit_nm,
                                defined_at=vso[out_var].defined_at,
                                updated_at=step_id,
                            )
                        else:
                            entry = VSOEntry(
                                value=float(out_val),
                                unit_symbol=unit_sym,
                                unit_name=unit_nm,
                                defined_at=step_id,
                                updated_at=step_id,
                            )
                        vso[out_var] = entry

                trace.vso = {k: asdict(v) for k, v in vso.items()}
                trace.vso_snapshots[step_id] = {k: asdict(v) for k, v in vso.items()}
                trace.steps.append(step)

            # ── All steps completed ───────────────────────────────────────
            if trace.steps:
                trace.final_answer = trace.steps[-1].intermediate_answer

            if any(step.status == "WRONG" for step in trace.steps):
                trace.trace_status = "FAIL"
                wrong_step = next((step for step in trace.steps if step.status == "WRONG"), None)
                if wrong_step is not None:
                    record_first_wrong_step(trace, wrong_step, None, vso, None, parse_obj)
            elif not trace.final_answer.strip():
                trace.trace_status = "FAIL"
                if trace.steps:
                    record_first_wrong_step(trace, trace.steps[-1], None, vso, None, parse_obj)
            else:
                trace.trace_status = "PASS"
            return trace


# ---------------------------------------------------------------------------
# Internal helpers (used by SolveTrace but available regardless of DSPy)
# ---------------------------------------------------------------------------

def _verify_step(
    step: StepObject,
    formula_entry: Optional[FormulaEntry],
    vso: Dict[str, VSOEntry],
    llm_verifier=None,
    was_repaired: bool = False,
) -> Tuple[str, float]:
    """Run SymPy verification first; fall back to LLM verifier.

    Returns (verdict, confidence).
    """
    if not step.checkable:
        confidence = 0.85 if step.type == "unit_conversion" else 0.9
        return "UNCERTAIN", confidence

    # ── SymPy path ────────────────────────────────────────────────────────
    if formula_entry and formula_entry.tool_dispatch == "sympy" and _SYMPY_AVAILABLE:
        formula_var_values = map_formula_vars_to_vso(formula_entry, vso)
        verdict, confidence = sympy_verify_step(
            formula_entry, formula_var_values, step.intermediate_answer
        )
        if verdict != "UNCERTAIN":
            if verdict == "CORRECT" and was_repaired:
                confidence = 0.6
            return verdict, confidence

    # ── LLM verifier fallback ─────────────────────────────────────────────
    if llm_verifier is not None:
        input_vals_json = json.dumps(
            {k: (v.value if isinstance(v, VSOEntry) else v) for k, v in vso.items()
             if k in step.input_var},
            default=str,
        )
        try:
            pred = llm_verifier(
                step_goal=step.goal,
                formula_text=(formula_entry.formula if formula_entry else ""),
                input_values=input_vals_json,
                step_input=step.step_input,
                intermediate_answer=step.intermediate_answer,
            )
            verdict = pred.verdict
            used_llm = True
            confidence = _verifier_confidence(
                step.type, verdict, was_repaired=was_repaired, used_llm=True
            )
            step.verifier_notes = getattr(pred, "correction", "")
            return verdict, confidence
        except Exception as exc:
            logger.warning("LLM verifier failed for step %s: %s", step.step_id, exc)

    return "UNCERTAIN", 0.5


def _build_prior_summary(recent_steps: List[StepObject]) -> str:
    """Produce a compact multi-line summary of recently completed steps."""
    if not recent_steps:
        return ""
    lines = []
    for s in recent_steps:
        if s.intermediate_answer:
            lines.append(f"  {s.step_id} ({s.goal}): {s.intermediate_answer}")
    return "\n".join(lines)


def classify_first_wrong_error(
    step: StepObject,
    formula_entry: Optional[FormulaEntry],
    vso: Optional[Dict[str, VSOEntry]],
    plan_item: Optional[Dict] = None,
    parse_obj: Optional[ProblemParseObject] = None,
) -> Tuple[str, str]:
    """Classify a first wrong step using deterministic metadata only."""

    step_type = str(step.type or "").lower()
    goal = str(step.goal or "")
    notes = str(step.verifier_notes or "")
    answer = str(step.intermediate_answer or "")
    question = str(getattr(parse_obj, "problem_text", "") or "")
    target = str(getattr(parse_obj, "target", "") or "")
    domain = str(getattr(parse_obj, "domain", "") or "")
    subdomain = str(getattr(parse_obj, "subdomain", "") or "")
    formula_text = ""
    formula_vars = ""
    if formula_entry is not None:
        formula_text = " ".join(
            str(part or "")
            for part in (
                formula_entry.id,
                formula_entry.topic,
                formula_entry.subtopic,
                formula_entry.text,
                formula_entry.formula,
            )
        )
        formula_vars = " ".join(formula_entry.variables.keys())

    context = " ".join(
        [goal, notes, answer, question, target, domain, subdomain, formula_text, formula_vars]
    ).lower()

    if formula_entry is None and step_type in {
        "calculation",
        "formula_application",
        "unit_conversion",
    }:
        return (
            "missing_formula",
            "No formula was retrieved for this step. Check Stage1 formula retrieval and target-variable matching.",
        )

    if any(value is None for value in step.input_var.values()):
        return (
            "missing_input_value",
            "One or more required input variables are missing from VSO. Check Stage0 extraction, variable aliases, or previous step outputs.",
        )

    unit_markers = (
        " mt",
        " mh",
        " uj",
        " μj",
        " µj",
        " uc",
        " μc",
        " µc",
        " khz",
        " mhz",
        " cm",
        " mm",
    )
    mismatch_markers = ("unit", "scale", "conversion", "mismatch")
    if any(marker in f" {answer.lower()} " for marker in unit_markers) or (
        any(marker in notes.lower() for marker in mismatch_markers)
        and any(marker in f" {answer.lower()} " for marker in unit_markers)
    ):
        return (
            "unit_or_scale_error",
            "Likely unit conversion or display-scale error. Check SI normalization and final answer formatting.",
        )

    if (
        ("quality factor" in context or "q factor" in context or "resonance q" in context)
        and re.search(r"\b[qQ]\b", " ".join([goal, question, target, formula_vars]))
    ):
        return (
            "q_disambiguation",
            "Disambiguate Q as quality factor rather than electric charge.",
        )

    rlc_markers = (
        "rlc",
        "resonance",
        "impedance",
        "reactance",
        "power factor",
        "phase angle",
        "x_l",
        "xl",
        "x_c",
        "xc",
        "omega",
        "omega0",
        "ω",
    )
    if any(marker in context for marker in rlc_markers):
        return (
            "rlc_calculation",
            "Check resonance vs off-resonance logic. Only use Z=R when resonance is explicit; otherwise use Z=sqrt(R^2+(X_L-X_C)^2).",
        )

    vector_markers = (
        "coulomb",
        "electric field",
        "electric force",
        "resultant",
        "net force",
        "net field",
        "vector",
        "angle",
        "q3",
        "triangle",
        "right angle",
        "equilateral",
        "square",
    )
    if any(marker in context for marker in vector_markers):
        return (
            "vector_geometry",
            "Check Coulomb/electric-field vector geometry, distances, signs, and resultant formula.",
        )

    if formula_entry is not None and str(step.status or "").upper() == "WRONG" and any(
        marker in notes.lower()
        for marker in ("deterministic solver could not solve", "sympy", "selected formula", "could not be executed")
    ):
        return (
            "formula_execution_error",
            "Formula was selected but could not be executed. Check formula variables, SymPy expression, and VSO mapping.",
        )

    if answer and (
        str(step.status or "").upper() == "WRONG"
        or any(marker in notes.lower() for marker in ("incorrect", "rejected", "wrong"))
    ):
        return (
            "numeric_mismatch",
            "The step produced a numeric answer but verifier rejected it. Check arithmetic, unit scale, or formula substitution.",
        )

    output_values = list(step.output_var.values())
    if (
        not answer.strip()
        or any(isinstance(value, str) for value in output_values)
        or any(marker in notes.lower() for marker in ("json", "parse", "malformed", "format"))
    ):
        return (
            "symbolic_or_format_error",
            "Check symbolic answer propagation and final answer formatting.",
        )

    return (
        "unknown",
        "No specific deterministic error type matched. Inspect step input, formula, VSO, and verifier notes.",
    )


def record_first_wrong_step(
    trace: TraceObject,
    step: StepObject,
    formula_entry: Optional[FormulaEntry] = None,
    vso: Optional[Dict[str, VSOEntry]] = None,
    plan_item: Optional[Dict] = None,
    parse_obj: Optional[ProblemParseObject] = None,
) -> None:
    """Attach first-wrong-step diagnosis metadata without changing behavior."""

    if trace.first_wrong_step_id:
        return

    error_type, repair_hint = classify_first_wrong_error(
        step=step,
        formula_entry=formula_entry,
        vso=vso,
        plan_item=plan_item,
        parse_obj=parse_obj,
    )
    trace.first_wrong_step_id = step.step_id
    trace.first_wrong_step_goal = step.goal
    trace.first_wrong_step_type = step.type
    trace.first_wrong_feedback = step.verifier_notes
    trace.first_wrong_error_type = error_type
    trace.repair_hint = repair_hint
    if error_type and error_type not in trace.diagnostic_tags:
        trace.diagnostic_tags.append(error_type)


def replay_trace_deterministically(
    trace: TraceObject,
    parse_obj: ProblemParseObject,
    formula_set: FormulaSet,
) -> TraceObject:
    """Recompute an LLM-produced trace with deterministic formula execution.

    The replay is deliberately guarded: SymPy/template-arithmetic successes
    replace LLM numeric outputs, while setup/symbolic steps are preserved.  The
    final answer is then rebound from the replayed VSO target when available.
    """
    if not trace.steps:
        return trace

    original_answer = trace.final_answer or trace.steps[-1].intermediate_answer
    replayed = TraceObject(
        problem_id=trace.problem_id,
        formula_path_index=trace.formula_path_index,
    )
    replayed.trace_status = trace.trace_status
    replayed.deterministic_replay_attempted = True
    replayed.deterministic_replay_original_answer = original_answer

    vso = init_vso(parse_obj)
    plan_by_id = {
        str(item.get("step_id")): item
        for item in parse_obj.step_plan
        if isinstance(item, dict) and item.get("step_id")
    }
    recomputed = 0

    for original in trace.steps:
        plan_item = plan_by_id.get(original.step_id, {})
        formula_entry = formula_set.formulas.get(original.step_id)
        step = StepObject(
            step_id=original.step_id,
            goal=original.goal,
            type=original.type,
            formula_ids=list(original.formula_ids),
            input_var={},
            output_var=dict(original.output_var or {}),
            step_input=original.step_input,
            intermediate_answer=original.intermediate_answer,
            thought=original.thought,
            confidence=original.confidence,
            checkable=original.checkable,
            status=original.status,
            verifier_notes=original.verifier_notes,
            evaluator_response=list(original.evaluator_response),
            cot_consistent=original.cot_consistent,
        )

        input_names = list((plan_item.get("input_var") or {}).keys())
        if not input_names:
            input_names = list(step.input_var.keys())
        for var_name in input_names:
            found = _lookup_vso_entry(var_name, vso)
            step.input_var[var_name] = asdict(found[1]) if found else None

        recomputed_this_step = False
        if step.type in _CHECKABLE_STEP_TYPES:
            replay_step = StepObject(
                step_id=step.step_id,
                goal=step.goal,
                type=step.type,
                formula_ids=[formula_entry.id] if formula_entry else list(step.formula_ids),
                input_var=dict(step.input_var),
                output_var={},
            )
            replay_step.checkable = True
            if (
                _try_template_arithmetic_step(
                    step=replay_step,
                    plan_item=plan_item,
                    vso=vso,
                )
                or _try_sympy_solve_step(
                    step=replay_step,
                    formula_entry=formula_entry,
                    plan_output_var=plan_item.get("output_var", {}),
                    vso=vso,
                )
            ):
                if _apply_condition_validation(replay_step, formula_entry, parse_obj):
                    step.output_var = dict(replay_step.output_var)
                    step.intermediate_answer = replay_step.intermediate_answer
                    step.status = "OK"
                    step.confidence = max(float(step.confidence or 0.0), 0.9)
                    step.verifier_notes = "LLM step recomputed deterministically with SymPy/template arithmetic."
                    recomputed += 1
                    recomputed_this_step = True

        if step.type == "conclusion":
            target_names = list((plan_item.get("input_var") or {}).keys())
            target_names.extend((plan_item.get("output_var") or {}).keys())
            target = str(parse_obj.unknown_quantity or "")
            if target:
                target_names.insert(0, target)
            rebound = False
            for target_name in target_names:
                if not target_name:
                    continue
                found = _lookup_vso_entry(str(target_name), vso)
                if not found:
                    continue
                actual_name, entry = found
                answer_name = str(target_name)
                if answer_name in {"percent_error", "I_over_Imax_percent", "electric_energy_fraction_percent", "B", "L_ind", "Phi_B", "Phi_link"}:
                    step.intermediate_answer = _format_direct_answer(answer_name, entry.value, vso)
                else:
                    step.intermediate_answer = f"{entry.value:g} {entry.unit_symbol}".strip()
                step.output_var = {answer_name: entry.value}
                if actual_name != answer_name:
                    step.output_var[actual_name] = entry.value
                step.status = "OK"
                step.confidence = max(float(step.confidence or 0.0), 0.95)
                step.verifier_notes = "Final answer rebound from deterministic replay VSO."
                replayed.deterministic_replay_final_rebound = True
                rebound = True
                break
            if not rebound and recomputed_this_step:
                replayed.deterministic_replay_final_rebound = True

        for out_var, out_val in step.output_var.items():
            if not isinstance(out_val, (int, float)) or out_val is None:
                continue
            unit_sym, unit_nm = _infer_output_unit(out_var, formula_entry, vso)
            vso[out_var] = VSOEntry(
                value=float(out_val),
                unit_symbol=unit_sym,
                unit_name=unit_nm,
                defined_at=vso[out_var].defined_at if out_var in vso else step.step_id,
                updated_at=step.step_id,
            )

        replayed.vso = {k: asdict(v) for k, v in vso.items()}
        replayed.vso_snapshots[step.step_id] = {k: asdict(v) for k, v in vso.items()}
        replayed.steps.append(step)

    replayed.deterministic_replay_recomputed_steps = recomputed
    if replayed.steps:
        replayed.final_answer = replayed.steps[-1].intermediate_answer
    if recomputed > 0:
        replayed.trace_status = "PASS" if replayed.final_answer.strip() else trace.trace_status
        if "deterministic_replay" not in replayed.diagnostic_tags:
            replayed.diagnostic_tags.append("deterministic_replay")
    else:
        replayed.trace_status = trace.trace_status
        replayed.final_answer = trace.final_answer
    return replayed


def _parse_output_values(
    json_str: str,
    plan_output_var: Dict,
) -> Dict[str, object]:
    """Parse LLM-generated JSON of output values.

    Falls back gracefully if the JSON is malformed.
    """
    try:
        raw = json.loads(json_str)
        if isinstance(raw, dict):
            return {k: v for k, v in raw.items() if isinstance(v, (int, float))}
    except (json.JSONDecodeError, TypeError):
        pass
    # If parsing failed, return dict with plan keys but no values
    return {k: None for k in plan_output_var}


def _infer_output_unit(
    var_name: str,
    formula_entry: Optional[FormulaEntry],
    vso: Dict[str, VSOEntry],
) -> Tuple[str, str]:
    """Try to infer unit_symbol and unit_name for a new VSO entry.

    Priority:
    1. Exact match in formula entry variables.
    2. Canonical match in formula entry variables.
    3. Existing VSO entry (variable was pre-populated by Stage 0).
    4. Empty strings (unknown).
    """
    if formula_entry:
        if var_name in formula_entry.variables:
            v = formula_entry.variables[var_name]
            return v.get("unit_symbol", ""), v.get("unit_name", "")
        # Canonical match
        target_canon = canonicalize_variable(var_name)
        if target_canon:
            for fv, meta in formula_entry.variables.items():
                if canonicalize_variable(fv) == target_canon:
                    return meta.get("unit_symbol", ""), meta.get("unit_name", "")
    if var_name in vso:
        return vso[var_name].unit_symbol, vso[var_name].unit_name
    return "", ""


def _lookup_vso_entry(
    var_name: str,
    vso: Dict[str, VSOEntry],
) -> Optional[Tuple[str, VSOEntry]]:
    """Find a VSO entry by exact name, then by canonical quantity name."""
    if var_name in vso:
        return var_name, vso[var_name]

    target_canon = canonicalize_variable(var_name)
    if not target_canon:
        return None
    for candidate_name, entry in vso.items():
        if entry.defined_at == "constants_table":
            continue
        if canonicalize_variable(candidate_name) == target_canon:
            return candidate_name, entry
    return None


class DeterministicSolveTrace:
    """No-LLM Stage 2+3 solver using formula library SymPy expressions.

    This is the production fallback when DSPy/vLLM is unavailable.  It executes
    parser-provided formula_application steps, verifies them by construction,
    updates the VSO, and copies the final value in conclusion steps.
    """

    def forward(
        self,
        parse_obj: ProblemParseObject,
        formula_set: FormulaSet,
        problem_id: str = "unknown",
        step_retry_limit: int = 1,
        trace_budget: int = 10,
    ) -> TraceObject:
        trace = TraceObject(
            problem_id=problem_id,
            formula_path_index=formula_set.path_index,
        )
        vso = init_vso(parse_obj)
        trace.vso = {k: asdict(v) for k, v in vso.items()}

        for step_index, plan_item in enumerate(parse_obj.step_plan):
            if step_index >= trace_budget:
                trace.trace_status = "FAIL"
                if trace.steps:
                    record_first_wrong_step(trace, trace.steps[-1], None, vso, None, parse_obj)
                break
            if not isinstance(plan_item, dict):
                continue

            step_id = str(plan_item.get("step_id", f"step_{step_index + 1}"))
            step_type = str(plan_item.get("type", "calculation"))
            formula_entry = formula_set.formulas.get(step_id)
            step = StepObject(
                step_id=step_id,
                goal=str(plan_item.get("goal", "")),
                type=step_type,
                formula_ids=[formula_entry.id] if formula_entry else [],
                input_var={},
                output_var={},
            )

            for var_name in plan_item.get("input_var", {}):
                found = _lookup_vso_entry(var_name, vso)
                step.input_var[var_name] = asdict(found[1]) if found else None

            if step_type == "conclusion":
                target_names = list(plan_item.get("input_var", {}).keys())
                target_names.extend(plan_item.get("output_var", {}).keys())
                copied = False
                for target_name in target_names:
                    if target_name in {"mean_abs_error_pair", "abs_rel_error_pair", "energy_charge_pair"}:
                        answer = (
                            _format_capacitor_pair_answer(str(target_name), vso)
                            if target_name == "energy_charge_pair"
                            else _format_measurement_pair_answer(str(target_name), vso)
                        )
                        if answer:
                            step.intermediate_answer = answer
                            step.output_var = {}
                            step.status = "OK"
                            step.confidence = 0.95
                            step.verifier_notes = "Conclusion formatted from measurement pair VSO entries."
                            copied = True
                            break
                    found = _lookup_vso_entry(target_name, vso)
                    if not found:
                        continue
                    actual_name, entry = found
                    if target_name in {"percent_error", "I_over_Imax_percent", "electric_energy_fraction_percent", "B", "L_ind", "Phi_B", "Phi_link"}:
                        step.intermediate_answer = _format_direct_answer(target_name, entry.value, vso)
                    else:
                        step.intermediate_answer = f"{entry.value:g} {entry.unit_symbol}".strip()
                    step.output_var = {target_name: entry.value}
                    if actual_name != target_name:
                        step.output_var[actual_name] = entry.value
                    step.status = "OK"
                    step.confidence = 0.95
                    step.verifier_notes = "Conclusion copied from VSO."
                    copied = True
                    break
                if not copied:
                    for previous_step in reversed(trace.steps):
                        previous_value = previous_step.output_var.get(target_name)
                        if isinstance(previous_value, str) and previous_value.strip():
                            step.intermediate_answer = previous_value.strip()
                            step.output_var = {target_name: previous_value.strip()}
                            step.status = "OK"
                            step.confidence = 0.95
                            step.verifier_notes = "Conclusion copied symbolic output from prior step."
                            copied = True
                            break
                if not copied:
                    step.status = "WRONG"
                    step.confidence = 0.0
                    step.verifier_notes = "Conclusion target was not available in VSO."
            elif step_type in _CHECKABLE_STEP_TYPES:
                step.checkable = True
                if not (
                    _try_template_arithmetic_step(
                        step=step,
                        plan_item=plan_item,
                        vso=vso,
                    )
                    or _try_sympy_solve_step(
                    step=step,
                    formula_entry=formula_entry,
                    plan_output_var=plan_item.get("output_var", {}),
                    vso=vso,
                    )
                ):
                    step.status = "WRONG"
                    step.confidence = 0.0
                    step.verifier_notes = (
                        "Deterministic solver could not solve this step with the selected formula."
                    )
            else:
                if not _try_setup_outputs(step, plan_item):
                    step.status = "UNCERTAIN"
                    step.confidence = 0.9
                    step.verifier_notes = "Non-checkable step accepted by deterministic solver."

            if step.status != "WRONG":
                _apply_condition_validation(step, formula_entry, parse_obj)

            if step.status != "WRONG":
                for out_var, out_val in step.output_var.items():
                    if not isinstance(out_val, (int, float)) or out_val is None:
                        continue
                    unit_sym, unit_nm = _infer_output_unit(out_var, formula_entry, vso)
                    vso[out_var] = VSOEntry(
                        value=float(out_val),
                        unit_symbol=unit_sym,
                        unit_name=unit_nm,
                        defined_at=vso[out_var].defined_at if out_var in vso else step_id,
                        updated_at=step_id,
                    )

            trace.vso = {k: asdict(v) for k, v in vso.items()}
            trace.vso_snapshots[step_id] = {k: asdict(v) for k, v in vso.items()}
            trace.steps.append(step)

            if step.status == "WRONG":
                record_first_wrong_step(trace, step, formula_entry, vso, plan_item, parse_obj)
                trace.trace_status = "FAIL"
                return trace

        if trace.steps:
            trace.final_answer = trace.steps[-1].intermediate_answer
        trace.trace_status = "PASS" if trace.final_answer.strip() else "FAIL"
        if trace.trace_status == "FAIL" and trace.steps:
            record_first_wrong_step(trace, trace.steps[-1], None, vso, None, parse_obj)
        return trace
