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
import logging
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
        "k_e":       {"value": 1.0 / (4 * _sc.pi * _sc.epsilon_0),
                      "unit_symbol": "N·m²/C²", "unit_name": "newton metre squared per coulomb squared"},
        "epsilon_0": {"value": _sc.epsilon_0,    "unit_symbol": "F/m",      "unit_name": "farads per metre"},
    }
except ImportError:
    # scipy not installed — use hand-coded CODATA values as fallback
    PHYSICS_CONSTANTS = {
        "g":         {"value": 9.80665,      "unit_symbol": "m/s^2",    "unit_name": "metres per second squared"},
        "c":         {"value": 299792458.0,  "unit_symbol": "m/s",      "unit_name": "metres per second"},
        "k_e":       {"value": 8.9875517923e9, "unit_symbol": "N·m²/C²","unit_name": "newton metre squared per coulomb squared"},
        "epsilon_0": {"value": 8.8541878128e-12,"unit_symbol": "F/m",  "unit_name": "farads per metre"},
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
    for var_name, qty in parse_obj.known_quantities.items():
        value = qty.get("value")
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        vso[var_name] = VSOEntry(
            value=value,
            unit_symbol=str(qty.get("unit_symbol", "")),
            unit_name=str(qty.get("unit_name", "")),
            defined_at="stage0",
            updated_at="stage0",
        )

    # Also seed from parse_obj.vso if already populated by Stage 0
    for var_name, entry in parse_obj.vso.items():
        if var_name in vso:
            continue   # don't overwrite known_quantities
        val = entry.get("value")
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        vso[var_name] = VSOEntry(
            value=val,
            unit_symbol=str(entry.get("unit_symbol", "")),
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
    sym_ns["k_e"] = _sym.Float(8.9875517923e9)
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

    unit_sym, _ = _infer_output_unit(target_var, formula_entry, vso)
    step.step_input = (
        f"Solved {formula_entry.formula} for {target_var} using "
        f"{json.dumps(formula_var_values, ensure_ascii=False)}"
    )
    step.intermediate_answer = f"{solved:g} {unit_sym}".strip()
    step.output_var = {target_var: solved}
    step.confidence = 1.0
    step.status = "OK"
    step.verifier_notes = "Solved deterministically with SymPy fallback."
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
    for vso_var in vso:
        canon = canonicalize_variable(vso_var)
        if canon and canon not in canonical_to_vso:
            canonical_to_vso[canon] = vso_var

    for formula_var in formula_entry.variables:
        # Strategy 1: direct name
        if formula_var in vso:
            result[formula_var] = vso[formula_var].value
            continue
        # Strategy 2: canonical
        formula_canon = canonicalize_variable(formula_var)
        if formula_canon and formula_canon in canonical_to_vso:
            vso_var = canonical_to_vso[formula_canon]
            if vso_var not in result.values():  # avoid double-mapping
                result[formula_var] = vso[vso_var].value

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
            desc="Brief physical reasoning for this step (1–2 sentences)"
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

        Respond with verdict CORRECT, INCORRECT, or UNCERTAIN only.
        If INCORRECT, explain what the correct answer should be in 'correction'.
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
        """Wraps PhysicsSolverSignature with ChainOfThought."""

        def __init__(self) -> None:
            self.generator = _dspy.ChainOfThought(PhysicsSolverSignature)

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
                    if target_var in vso:
                        target_entry = vso[target_var]
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
                repaired = False
                verifier_feedback = ""

                for attempt in range(step_retry_limit):
                    total_calls += 1
                    if total_calls > trace_budget:
                        logger.warning("Trace budget exceeded at step %s.", step_id)
                        step.status = "WRONG"
                        step.verifier_notes = "Trace budget exceeded."
                        trace.steps.append(step)
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
                    trace.trace_status = "FAIL"
                    return trace

                # ── Write output vars to VSO ──────────────────────────────
                if step.status == "WRONG":
                    trace.steps.append(step)
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
            elif not trace.final_answer.strip():
                trace.trace_status = "FAIL"
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
