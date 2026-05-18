"""Type 1 (logic-based educational query) pipeline package.

Public API
----------
parse_type1(payload)        → Type1ParseObject    (programmatic, no LLM)
run(payload, config)        → dict                (full pipeline; requires DSPy LM)
Type1Solver(generate_fol)   → dspy.Module         (requires DSPy LM configured)

Quick start::

    import dspy
    from type1 import parse_type1, Type1Solver

    dspy.configure(lm=dspy.LM(
        model="openai/qwen3-8b",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
    ))

    solver = Type1Solver()
    parse_obj = parse_type1(request_payload)
    responses = solver(parse_obj)          # list[Type1Response]
    api_output = responses[0].to_dict()    # competition submission format

DSPy is an optional dependency for the parser and schema layers.  Imports
that require DSPy are guarded so that ``from type1.parser import ...`` and
``from type1.schemas import ...`` always work regardless of whether
``dspy-ai`` is installed.
"""

# ── Always-available imports (no LLM dependency) ────────────────────────────
from .parser import detect_question_format, detect_solver_route, parse_type1
from .pipeline import run, run_all_questions
from .schemas import (
    QuestionFormat,
    SolverRoute,
    Type1ParseObject,
    Type1Question,
    Type1Response,
)

__all__ = [
    # Schemas
    "QuestionFormat",
    "SolverRoute",
    "Type1ParseObject",
    "Type1Question",
    "Type1Response",
    # Parser
    "parse_type1",
    "detect_question_format",
    "detect_solver_route",
    # Pipeline
    "run",
    "run_all_questions",
]

# ── DSPy-dependent exports (guarded) ────────────────────────────────────────
try:
    from .dspy_modules import (
        FOLFormalizer,
        MCQReasoner,
        MCQReasoning,
        OpenEndedReasoner,
        OpenEndedReasoning,
        PremiseFOLFormalization,
        Type1Solver,
        Type1Verifier,
        Type1Verification,
        YesNoReasoner,
        YesNoReasoning,
    )
    from .z3_solver import (
        Z3Formalization,
        Z3SolverModule,
        execute_z3_code,
        validate_z3_code,
    )

    __all__ += [
        "MCQReasoning",
        "YesNoReasoning",
        "OpenEndedReasoning",
        "PremiseFOLFormalization",
        "MCQReasoner",
        "YesNoReasoner",
        "OpenEndedReasoner",
        "FOLFormalizer",
        "Type1Solver",
        "Type1Verification",
        "Type1Verifier",
        "Z3Formalization",
        "Z3SolverModule",
        "validate_z3_code",
        "execute_z3_code",
    ]
except ModuleNotFoundError:
    pass  # dspy-ai not yet installed; install with: pip install dspy-ai
