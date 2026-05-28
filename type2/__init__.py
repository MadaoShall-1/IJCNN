"""Type 2 (physics calculation) pipeline package.

Public API
----------
load_library(path=None)     → List[FormulaEntry]
FormulaRetriever(...)       → retriever instance
canonicalize_variable(name) → Optional[str]
detect_collisions(mapping)  → List[Tuple[str, str, str]]
init_vso(parse_obj)         → Dict[str, VSOEntry]
classify_checkable(...)     → bool
sympy_verify_step(...)      → (verdict, confidence)
map_formula_vars_to_vso(...)→ Dict[str, float]

Quick start::

    from type2 import FormulaRetriever, SolveTrace
    from parser.main import parse_problem          # Stage 0

    retriever = FormulaRetriever()
    parse_obj = parse_problem(problem_text)        # ProblemParseObject
    formula_sets = retriever.retrieve(parse_obj, beam_n=3)
    # formula_sets[0].formulas maps step_id → FormulaEntry
"""

from .schemas import (
    DiagnosisObject,
    CotIssue,
    FormulaEntry,
    FormulaSet,
    StepObject,
    TraceObject,
    VSOEntry,
)
from .stage1 import (
    CANONICAL_MAP,
    FormulaRetriever,
    canonicalize_variable,
    detect_collisions,
    load_library,
)
from .stage2 import (
    PHYSICS_CONSTANTS,
    classify_checkable,
    init_vso,
    map_formula_vars_to_vso,
    sympy_verify_step,
)
from .stage4 import diagnose_trace
from .stage5 import (
    extract_stable_prefix,
    repair_trace,
    rollback_vso,
    select_repair_formula,
)
from .stage6 import build_response, extract_final_answer

__all__ = [
    # Schemas
    "FormulaEntry",
    "VSOEntry",
    "StepObject",
    "TraceObject",
    "CotIssue",
    "DiagnosisObject",
    "FormulaSet",
    # Stage 1
    "CANONICAL_MAP",
    "canonicalize_variable",
    "detect_collisions",
    "load_library",
    "FormulaRetriever",
    # Stage 2+3
    "PHYSICS_CONSTANTS",
    "classify_checkable",
    "init_vso",
    "map_formula_vars_to_vso",
    "sympy_verify_step",
    # Stage 4
    "diagnose_trace",
    # Stage 5
    "extract_stable_prefix",
    "repair_trace",
    "rollback_vso",
    "select_repair_formula",
    # Stage 6
    "build_response",
    "extract_final_answer",
]
