"""Type 1 pipeline orchestrator.

Ties together the full Type 1 flow described in design §3.1:

    payload → parse_type1()
            → [Z3 formal solver | LLM reasoner]
            → [verifier pass]
            → response assembly
            → API dict

Timeout tier ladder (design §3.2):
    Tier 0 (< 12 s)  — full pipeline: Z3 + verifier pass + all optional fields.
    Tier 1 (≥ 12 s)  — skip verifier pass.
    Tier 2 (≥ 35 s)  — additionally skip optional field generation.
    Tier 3 (≥ 55 s)  — hard stop; emit best available answer immediately.

Phase 1 additions vs Phase 0:
    • SolverRoute.Z3 and SolverRoute.PROVER9 questions use Z3SolverModule
      (Prover9 is Phase 2; routed questions fall through to Z3 here).
    • On Z3 fallback (validation failure or UNKNOWN), the question is solved
      by the LLM path as before.
    • MCQ and open-ended answers go through Type1Verifier when
      type1_verify=True and Tier 1 has not been exceeded.
    • Confidence is populated: 1.0 for Z3, 0.9/0.4 for verified LLM.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from config import SolverConfig
from .parser import parse_type1
from .schemas import QuestionFormat, SolverRoute, Type1ParseObject, Type1Response

if TYPE_CHECKING:
    from .dspy_modules import Type1Solver  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    payload: Dict[str, Any],
    config: Optional[SolverConfig] = None,
    solver: Optional["Type1Solver"] = None,
) -> Dict[str, Any]:
    """Run the full Type 1 pipeline for a single API-request payload.

    Parameters
    ----------
    payload:
        Raw request dict.  Must contain ``premises-NL`` (or ``premises``) and
        either ``questions`` or ``question``.
    config:
        Runtime parameters.  Defaults to ``SolverConfig()`` (Phase 1 defaults).
    solver:
        Pre-instantiated ``Type1Solver``.  Pass a cached instance to avoid
        re-building DSPy modules on every call.

    Returns
    -------
    dict
        Competition API response matching the schema in contest_details.txt.
        Always contains ``answer`` and ``explanation``.  Optional fields
        (``fol``, ``cot``, ``premises``, ``confidence``) are included when
        time permits and config flags allow.
    """
    if config is None:
        config = SolverConfig()

    ctx = _PipelineContext(config=config, t_start=time.monotonic())

    # ── Stage 0: parse payload (deterministic, ~0 ms) ────────────────────
    try:
        parse_obj = parse_type1(payload)
    except Exception as exc:
        logger.error("Type 1 parse failed: %s", exc)
        return _error_response(f"Parse error: {exc}")

    if not parse_obj.questions:
        return _error_response("No questions found in payload.")

    for w in parse_obj.parse_warnings:
        logger.warning("Type 1 parse warning: %s", w)

    if ctx.hard_stopped():
        logger.warning("Hard timeout before reasoning stage.")
        return _timeout_response()

    # ── Select question ───────────────────────────────────────────────────
    question_idx = int(payload.get("_question_idx", 0))
    if question_idx >= len(parse_obj.questions):
        question_idx = 0
    question = parse_obj.questions[question_idx]

    # ── Formal solver path (Z3) ───────────────────────────────────────────
    response: Optional[Type1Response] = None

    if (
        config.type1_use_z3
        and question.format == QuestionFormat.YES_NO_UNCERTAIN
        and question.solver_route in (SolverRoute.Z3, SolverRoute.PROVER9)
        and not ctx.tier2_exceeded()
    ):
        response = _run_z3_path(parse_obj, question_idx, ctx)

    # ── LLM reasoning path (fallback or non-Yes/No questions) ────────────
    if response is None:
        if solver is None:
            from .dspy_modules import Type1Solver  # lazy — requires dspy-ai
            generate_fol = config.generate_fol and not ctx.tier2_exceeded()
            solver = Type1Solver(generate_fol=generate_fol)

        try:
            response = solver.solve_one(parse_obj, question_idx=question_idx)
        except Exception as exc:
            logger.error("Type 1 LLM solver error: %s", exc, exc_info=True)
            return _error_response(f"Solver error: {exc}")

    if ctx.hard_stopped():
        logger.warning("Hard timeout after reasoning; emitting partial response.")
        return _partial_response(response)

    # ── Verifier pass (MCQ + open-ended, Tier 0 only) ─────────────────────
    if (
        config.type1_verify
        and not ctx.tier1_exceeded()
        and question.format != QuestionFormat.YES_NO_UNCERTAIN
        and response.confidence is None  # skip if already set (e.g. by Z3)
    ):
        response = _run_verifier_pass(response, parse_obj, question_idx, ctx)

    if ctx.hard_stopped():
        return _partial_response(response)

    # ── Response assembly ─────────────────────────────────────────────────
    generate_optional = config.optional_fields_enabled(ctx.elapsed())
    return _assemble_response(response, config, generate_optional=generate_optional)


# ---------------------------------------------------------------------------
# Batch helper — process all questions in a Type1ParseObject
# ---------------------------------------------------------------------------

def run_all_questions(
    payload: Dict[str, Any],
    config: Optional[SolverConfig] = None,
    solver: Optional["Type1Solver"] = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper: solve every question in a Type 1 record.

    Useful for training-data evaluation where each record has multiple
    questions.  Returns one response dict per question in order.
    """
    if config is None:
        config = SolverConfig()

    parse_obj = parse_type1(payload)
    if not parse_obj.questions:
        return [_error_response("No questions found in payload.")]

    return [
        run(
            payload={**payload, "_question_idx": i},
            config=config,
            solver=solver,
        )
        for i in range(len(parse_obj.questions))
    ]


# ---------------------------------------------------------------------------
# Z3 path
# ---------------------------------------------------------------------------

def _run_z3_path(
    parse_obj: Type1ParseObject,
    question_idx: int,
    ctx: "_PipelineContext",
) -> Optional[Type1Response]:
    """Attempt Z3 formal solving.  Returns None to signal LLM fallback."""
    try:
        from .z3_solver import Z3SolverModule
    except ImportError:
        logger.warning("Z3SolverModule not available (DSPy or z3-solver not installed).")
        return None

    question = parse_obj.questions[question_idx]
    z3_module = Z3SolverModule()

    try:
        pred = z3_module(
            premises=parse_obj.premises_nl,
            question=question.text,
        )
    except Exception as exc:
        logger.warning("Z3SolverModule raised unexpectedly: %s", exc)
        return None

    if not pred.used_z3:
        # Validation failed or UNKNOWN — fall back to LLM
        return None

    # Z3 succeeded: the Z3 code IS the fol field (design §3.1 step v)
    fol = [pred.z3_code] if pred.z3_code else []
    return Type1Response(
        answer=pred.answer,
        explanation=pred.explanation,
        fol=fol,
        cot=[],
        premises=list(parse_obj.premises_nl),
        confidence=pred.confidence,
    )


# ---------------------------------------------------------------------------
# Verifier pass
# ---------------------------------------------------------------------------

def _run_verifier_pass(
    response: Type1Response,
    parse_obj: Type1ParseObject,
    question_idx: int,
    ctx: "_PipelineContext",
) -> Type1Response:
    """Run the Type1Verifier and update confidence + explanation."""
    try:
        from .dspy_modules import Type1Verifier
    except ModuleNotFoundError:
        logger.warning("DSPy not installed; skipping verifier pass.")
        return response

    question = parse_obj.questions[question_idx]
    verifier = Type1Verifier()

    try:
        vresult = verifier(
            premises=parse_obj.premises_nl,
            question=question.text,
            proposed_answer=response.answer,
            proposed_explanation=response.explanation,
        )
    except Exception as exc:
        logger.warning("Type1Verifier raised unexpectedly: %s", exc)
        return response

    updated_explanation = response.explanation + (vresult.note or "")
    return Type1Response(
        answer=response.answer,
        explanation=updated_explanation,
        fol=response.fol,
        cot=response.cot,
        premises=response.premises,
        confidence=vresult.confidence,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _PipelineContext:
    """Tracks wall-clock time for timeout-tier decisions."""

    def __init__(self, config: SolverConfig, t_start: float) -> None:
        self._config = config
        self._t_start = t_start

    def elapsed(self) -> float:
        return time.monotonic() - self._t_start

    def hard_stopped(self) -> bool:
        return self._config.tier(self.elapsed()) >= 3

    def tier1_exceeded(self) -> bool:
        return self._config.tier(self.elapsed()) >= 1

    def tier2_exceeded(self) -> bool:
        return self._config.tier(self.elapsed()) >= 2


def _assemble_response(
    response: Type1Response,
    config: SolverConfig,
    generate_optional: bool = True,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "answer": response.answer,
        "explanation": response.explanation,
    }
    if generate_optional:
        if config.generate_fol and response.fol:
            d["fol"] = response.fol
        if config.generate_cot and response.cot:
            d["cot"] = response.cot
        if config.generate_premises and response.premises:
            d["premises"] = response.premises
        if config.generate_confidence and response.confidence is not None:
            d["confidence"] = response.confidence
    return d


def _partial_response(response: Type1Response) -> Dict[str, Any]:
    return {
        "answer": response.answer or "",
        "explanation": (
            response.explanation
            or "Partial response: pipeline reached hard timeout after reasoning."
        ),
    }


def _timeout_response() -> Dict[str, Any]:
    return {
        "answer": "",
        "explanation": "Request exceeded latency budget before reasoning could begin.",
    }


def _error_response(message: str) -> Dict[str, Any]:
    return {
        "answer": "",
        "explanation": f"Pipeline error: {message}",
    }
