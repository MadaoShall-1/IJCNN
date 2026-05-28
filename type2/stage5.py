"""Type 2 Stage 5: FWS-Centered Repair (design §5 Stage 5).

Given a failed TraceObject and its DiagnosisObject, this stage attempts to
repair the solution by:
  1. Extracting the stable prefix (steps before the FWS that are OK/REPAIRED).
  2. Rolling back the VSO to the snapshot taken just before the FWS.
  3. Selecting an alternative formula for the FWS (if error is E1).
  4. Re-running the suffix (FWS and later steps) under the repaired configuration.

Always-available:
  extract_stable_prefix(trace, diagnosis) → List[StepObject]
  rollback_vso(trace, fws_step_id)        → Dict[str, VSOEntry]
  select_repair_formula(formula_set, fws_step_id, diagnosis) → Optional[FormulaEntry]
  repair_trace(trace, formula_set, parse_obj, diagnosis, solver) → TraceObject

DSPy-guarded:
  RepairSolveTrace(dspy.Module) — re-runs the suffix with a patched formula set
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from typing import Dict, List, Optional

from parser.schemas import ProblemParseObject

from .schemas import (
    DiagnosisObject,
    FormulaEntry,
    FormulaSet,
    StepObject,
    TraceObject,
    VSOEntry,
)

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

# ---------------------------------------------------------------------------
# Stable prefix extraction
# ---------------------------------------------------------------------------

def extract_stable_prefix(
    trace: TraceObject,
    diagnosis: DiagnosisObject,
) -> List[StepObject]:
    """Return a copy of the steps that precede the FWS and are verified OK.

    These steps form the stable prefix that does not need to be re-executed
    during repair.  Steps with status UNCERTAIN are included (they were accepted
    as-is in the original trace).
    """
    fws_index = diagnosis.fws_index
    if fws_index is None:
        # No FWS — entire trace is the stable prefix
        return copy.deepcopy(trace.steps)
    return copy.deepcopy(trace.steps[:fws_index])


def rollback_vso(
    trace: TraceObject,
    fws_step_id: str,
) -> Dict[str, VSOEntry]:
    """Restore the VSO to the snapshot taken just before the FWS.

    Returns a dict of {var_name: VSOEntry} representing the rolled-back VSO.
    Falls back to the initial VSO if no appropriate snapshot is found.
    """
    steps = trace.steps
    step_ids = [s.step_id for s in steps]

    fws_idx = next((i for i, s in enumerate(steps) if s.step_id == fws_step_id), None)
    if fws_idx is None or fws_idx == 0:
        # No preceding step — use snapshot at initial state (before any step)
        # Reconstruct from the snapshots by finding the earliest step before FWS
        # Fall back to empty dict (caller will re-init from parse_obj)
        return {}

    # Find the snapshot from the step immediately before the FWS
    predecessor_id = step_ids[fws_idx - 1]
    snapshot_dict = trace.vso_snapshots.get(predecessor_id, {})

    vso: Dict[str, VSOEntry] = {}
    for var_name, entry_dict in snapshot_dict.items():
        try:
            vso[var_name] = VSOEntry(
                value=float(entry_dict.get("value", 0.0)),
                unit_symbol=str(entry_dict.get("unit_symbol", "")),
                unit_name=str(entry_dict.get("unit_name", "")),
                defined_at=str(entry_dict.get("defined_at", "rollback")),
                updated_at=str(entry_dict.get("updated_at", "rollback")),
            )
        except (TypeError, ValueError):
            continue

    return vso


# ---------------------------------------------------------------------------
# Formula selection for repair
# ---------------------------------------------------------------------------

def select_repair_formula(
    formula_set: FormulaSet,
    fws_step_id: str,
    diagnosis: DiagnosisObject,
    all_formula_sets: Optional[List[FormulaSet]] = None,
) -> Optional[FormulaEntry]:
    """Select an alternative formula for the FWS during repair.

    Strategy:
    1. If the error type is E1 (formula selection) and alternative formula sets
       were provided, pick the next-best formula from those sets.
    2. Otherwise return None (no formula swap needed; just retry the arithmetic).

    Parameters
    ----------
    formula_set:
        The formula set used in the original (failed) trace.
    fws_step_id:
        The step_id of the First Wrong Step.
    diagnosis:
        The diagnosis produced by Stage 4.
    all_formula_sets:
        The full beam of formula sets returned by Stage 1 (optional).
    """
    if diagnosis.global_error_type != "E1":
        return None

    current_entry = formula_set.formulas.get(fws_step_id)
    current_id = current_entry.id if current_entry else None

    if all_formula_sets is None:
        return None

    for fs in all_formula_sets:
        candidate = fs.formulas.get(fws_step_id)
        if candidate is None:
            continue
        if candidate.id != current_id:
            return candidate

    return None


# ---------------------------------------------------------------------------
# Repair function (deterministic wrapper around solver)
# ---------------------------------------------------------------------------

def repair_trace(
    trace: TraceObject,
    formula_set: FormulaSet,
    parse_obj: ProblemParseObject,
    diagnosis: DiagnosisObject,
    solver,
    all_formula_sets: Optional[List[FormulaSet]] = None,
    step_retry_limit: int = 2,
) -> TraceObject:
    """Attempt to repair a failed trace using Stage 5 repair logic.

    Parameters
    ----------
    trace:
        The original failed trace.
    formula_set:
        The formula set used in the original trace.
    parse_obj:
        The original problem parse object.
    diagnosis:
        The DiagnosisObject from Stage 4.
    solver:
        An instance of SolveTrace (DSPy module) used to re-execute steps.
        If None (no DSPy), returns a copy of the trace with trace_status FAIL.
    all_formula_sets:
        Full beam from Stage 1, used for E1 formula swap.
    step_retry_limit:
        Retry limit for the repair solve attempt.

    Returns
    -------
    A new TraceObject with trace_status PASS or FAIL (REPAIRED if fixed).
    """
    if solver is None:
        logger.warning("repair_trace called without a solver; returning original trace.")
        repaired = copy.deepcopy(trace)
        repaired.trace_status = "FAIL"
        return repaired

    fws_index = diagnosis.fws_index
    if fws_index is None:
        # Already passing — nothing to repair
        repaired = copy.deepcopy(trace)
        repaired.trace_status = "PASS"
        return repaired

    fws_step_id = trace.steps[fws_index].step_id

    # Optionally swap formula for E1 errors
    repair_entry = select_repair_formula(
        formula_set, fws_step_id, diagnosis, all_formula_sets
    )

    # Build a patched formula set with the repaired formula
    patched_formulas = dict(formula_set.formulas)
    if repair_entry is not None:
        patched_formulas[fws_step_id] = repair_entry
    patched_set = FormulaSet(
        formulas=patched_formulas,
        retrieval_confidence=formula_set.retrieval_confidence,
        path_index=formula_set.path_index,
    )

    # Build a patched parse_obj that only contains the suffix steps
    suffix_plan = [
        item for item in parse_obj.step_plan
        if isinstance(item, dict)
        and item.get("step_id") in {s.step_id for s in trace.steps[fws_index:]}
    ]

    # Restore VSO to the pre-FWS snapshot
    rolled_back_vso = rollback_vso(trace, fws_step_id)

    # Inject rolled-back VSO values into a minimal parse_obj for re-solving
    patched_parse_obj = ProblemParseObject(
        problem_text=parse_obj.problem_text,
        domains=parse_obj.domains,
        sub_domains=parse_obj.sub_domains,
        known_quantities={
            var: {"value": entry.value, "unit_symbol": entry.unit_symbol,
                  "unit_name": entry.unit_name}
            for var, entry in rolled_back_vso.items()
        },
        step_plan=suffix_plan,
        vso={},
    )

    # Re-run the suffix via the solver
    suffix_trace = solver.forward(
        parse_obj=patched_parse_obj,
        formula_set=patched_set,
        problem_id=trace.problem_id + "_repair",
        step_retry_limit=step_retry_limit,
    )

    # Assemble the repaired trace: stable prefix + repaired suffix
    stable_steps = extract_stable_prefix(trace, diagnosis)
    repaired_steps = stable_steps + suffix_trace.steps

    repaired = TraceObject(
        problem_id=trace.problem_id,
        formula_path_index=formula_set.path_index,
    )
    repaired.steps = repaired_steps
    repaired.vso = suffix_trace.vso
    repaired.vso_snapshots = {**trace.vso_snapshots, **suffix_trace.vso_snapshots}
    repaired.final_answer = suffix_trace.final_answer or (
        repaired_steps[-1].intermediate_answer if repaired_steps else ""
    )
    repaired.trace_status = "REPAIRED" if suffix_trace.trace_status == "PASS" else "FAIL"

    return repaired


# ---------------------------------------------------------------------------
# DSPy module (thin wrapper, guarded)
# ---------------------------------------------------------------------------

if _DSPY_AVAILABLE:

    class RepairSolveTrace(_dspy.Module):
        """Stage 5 repair module: wraps repair_trace with a SolveTrace solver.

        Usage::

            repair_module = RepairSolveTrace()
            repaired_trace = repair_module.forward(
                trace, formula_set, parse_obj, diagnosis,
                all_formula_sets=beam_sets,
            )
        """

        def __init__(self) -> None:
            from .stage2 import SolveTrace
            self.solver = SolveTrace()

        def forward(
            self,
            trace: TraceObject,
            formula_set: FormulaSet,
            parse_obj: ProblemParseObject,
            diagnosis: DiagnosisObject,
            all_formula_sets: Optional[List[FormulaSet]] = None,
            step_retry_limit: int = 2,
        ) -> TraceObject:
            return repair_trace(
                trace=trace,
                formula_set=formula_set,
                parse_obj=parse_obj,
                diagnosis=diagnosis,
                solver=self.solver,
                all_formula_sets=all_formula_sets,
                step_retry_limit=step_retry_limit,
            )
