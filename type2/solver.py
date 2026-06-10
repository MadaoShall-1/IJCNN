"""Type 2 solver interface for the top-level dispatcher.

Orchestrates the full Type 2 physics pipeline (Stages 0-6) and
exposes a clean ``solve(query) -> dict`` contract.

This code was extracted from the former project-level api.py so that
the Type 2 package owns its own orchestration logic.

NOTE: This module expects the ``type2/`` directory to be on sys.path
so that bare imports like ``from parser.main import ...`` resolve.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import SolverConfig
from parser.main import parse_problem as _parse_stage0
from parser.schemas import ProblemParseObject

from type2.stage1 import FormulaRetriever
from type2.stage2 import DeterministicSolveTrace
from type2.stage4 import diagnose_trace
from type2.stage5 import repair_trace, select_repair_formula
from type2.stage6 import build_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import dspy as _dspy
    _DSPY_AVAILABLE = True
except ModuleNotFoundError:
    _DSPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level state (initialised once via ``init()``)
# ---------------------------------------------------------------------------

_retriever: Optional[FormulaRetriever] = None
_solve_trace = None
_repair_module = None
_config: SolverConfig = SolverConfig()
_solver_mode = "unloaded"
_stage0_cache: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
_stage0_cache_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Initialisation (called once at API startup)
# ---------------------------------------------------------------------------

def init(cfg: SolverConfig) -> None:
    """Load models and warm caches for the Type 2 pipeline."""
    global _retriever, _solve_trace, _repair_module, _config, _solver_mode

    _config = cfg

    logger.info("Loading FormulaRetriever...")
    _retriever = FormulaRetriever()
    _solve_trace = DeterministicSolveTrace()
    _repair_module = _solve_trace
    _solver_mode = "deterministic_sympy"

    if _DSPY_AVAILABLE and cfg.dspy_model:
        lm_kwargs = {
            "model": cfg.dspy_model,
            "api_key": cfg.dspy_api_key,
        }
        if cfg.dspy_api_base:
            lm_kwargs["api_base"] = cfg.dspy_api_base
        _dspy.configure(lm=_dspy.LM(**lm_kwargs))
        logger.info(
            "Configured DSPy LM model=%s api_base=%s",
            cfg.dspy_model,
            cfg.dspy_api_base or "<provider-default>",
        )
        from type2.stage2 import SolveTrace
        from type2.stage5 import RepairSolveTrace
        _solve_trace = SolveTrace()
        _repair_module = RepairSolveTrace()
        _solver_mode = "dspy_llm"
        logger.info("DSPy Type 2 modules loaded.")
    else:
        logger.info("Using deterministic SymPy Type 2 solver (no DSPy LM).")


def status() -> Dict[str, Any]:
    """Return diagnostic info for /health."""
    return {
        "type2_solver_mode": _solver_mode,
        "retriever_loaded": _retriever is not None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TYPE2_ROOT = Path(__file__).resolve().parent.parent


def _normalize_problem_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _resolve_cache_path(cfg: SolverConfig) -> Path:
    path = Path(cfg.stage0_cache_results_path)
    if path.is_absolute():
        return path
    return _TYPE2_ROOT / path


def _load_stage0_cache(cfg: SolverConfig) -> Dict[str, Dict[str, Dict[str, Any]]]:
    global _stage0_cache, _stage0_cache_path

    cache_path = str(_resolve_cache_path(cfg))
    if _stage0_cache is not None and _stage0_cache_path == cache_path:
        return _stage0_cache

    by_id: Dict[str, Dict[str, Any]] = {}
    by_text: Dict[str, Dict[str, Any]] = {}
    path = Path(cache_path)
    if not path.exists():
        logger.warning("Stage 0 cache file not found: %s", path)
        _stage0_cache = {"by_id": by_id, "by_text": by_text}
        _stage0_cache_path = cache_path
        return _stage0_cache

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid cache line %d: %s", line_no, exc)
                continue
            parse = record.get("parse")
            if not isinstance(parse, dict):
                continue
            dataset_id = record.get("dataset_id") or record.get("id")
            if dataset_id is not None:
                by_id[str(dataset_id)] = parse
            question = record.get("question") or parse.get("problem_text")
            if question:
                by_text[_normalize_problem_text(str(question))] = parse

    _stage0_cache = {"by_id": by_id, "by_text": by_text}
    _stage0_cache_path = cache_path
    logger.info("Loaded Stage 0 cache (%d ids, %d questions).", len(by_id), len(by_text))
    return _stage0_cache


def _dict_to_parse_obj(d: Dict[str, Any]) -> ProblemParseObject:
    return ProblemParseObject(
        problem_text=d.get("problem_text", ""),
        domains=d.get("domains", []),
        sub_domains=d.get("sub_domains", []),
        domain_confidence=d.get("domain_confidence", 0.0),
        known_quantities=d.get("known_quantities", {}),
        conditions=d.get("conditions", []),
        relations=d.get("relations", []),
        unknown_quantity=d.get("unknown_quantity"),
        unknown_unit=d.get("unknown_unit"),
        step_plan=d.get("step_plan", []),
        plan_confidence=d.get("plan_confidence", 0.0),
        parser_warnings=d.get("parser_warnings", []),
        vso=d.get("vso", {}),
        metadata=d.get("metadata", {}),
    )


def _get_stage0_parse(
    problem_text: str,
    problem_id: str,
    cfg: SolverConfig,
) -> Dict[str, Any]:
    use_cache = cfg.stage0_cache_enabled
    if use_cache:
        cache = _load_stage0_cache(cfg)
        parse = cache["by_id"].get(problem_id)
        if parse is None:
            parse = cache["by_text"].get(_normalize_problem_text(problem_text))
        if parse is not None:
            parse = dict(parse)
            metadata = dict(parse.get("metadata") or {})
            metadata["stage0_cache_hit"] = True
            parse["metadata"] = metadata
            return parse

    parse = _parse_stage0(
        problem_text,
        use_llm_fallback=cfg.stage0_use_llm_fallback,
    )
    metadata = dict(parse.get("metadata") or {})
    metadata["stage0_cache_hit"] = False
    parse["metadata"] = metadata
    return parse


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def solve(query: str, *, config: Optional[SolverConfig] = None) -> Dict[str, Any]:
    """Run the full Type 2 pipeline for a single physics query.

    Returns a dict with at least ``answer``, ``confidence``, and
    ``chain_of_thought``.  On failure returns ``answer=""`` with an
    ``error`` key.
    """
    cfg = config or _config
    t_start = time.monotonic()

    problem_text = query.strip()
    if not problem_text:
        return {"answer": "", "confidence": 0.0, "error": "empty problem text"}

    problem_id = "unknown"

    # Stage 0: parse
    try:
        parse_dict = _get_stage0_parse(problem_text, problem_id, cfg)
        parse_obj = _dict_to_parse_obj(parse_dict)
    except Exception as exc:
        logger.error("Stage 0 parse failed: %s", exc)
        return {"answer": "", "confidence": 0.0, "error": f"parse error: {exc}"}

    # Stage 1: formula retrieval
    elapsed = time.monotonic() - t_start
    beam_n = 1 if cfg.tier(elapsed) >= 1 else cfg.beam_n

    try:
        formula_sets = _retriever.retrieve(parse_obj, beam_n=beam_n)
    except Exception as exc:
        logger.error("Stage 1 retrieval failed: %s", exc)
        return {"answer": "", "confidence": 0.0, "error": f"retrieval error: {exc}"}

    if not formula_sets:
        return {"answer": "", "confidence": 0.0, "error": "no formula sets found"}

    # Stages 2-5: trace, diagnose, repair
    best_trace = None
    best_formula_set = formula_sets[0]
    best_diagnosis = None

    for fs in formula_sets:
        elapsed = time.monotonic() - t_start
        if cfg.tier(elapsed) >= 3:
            break

        if _solve_trace is None:
            from type2.schemas import TraceObject
            stub = TraceObject(problem_id=problem_id, formula_path_index=fs.path_index)
            stub.trace_status = "FAIL"
            best_trace = stub
            best_formula_set = fs
            break

        try:
            retry_limit = 1 if cfg.tier(elapsed) >= 1 else cfg.step_retry_limit
            trace = _solve_trace.forward(
                parse_obj=parse_obj,
                formula_set=fs,
                problem_id=problem_id,
                step_retry_limit=retry_limit,
                trace_budget=cfg.trace_budget,
            )
        except Exception as exc:
            logger.warning("SolveTrace failed for path %d: %s", fs.path_index, exc)
            continue

        try:
            diagnosis = diagnose_trace(trace, fs)
        except Exception as exc:
            logger.warning("Stage 4 failed: %s", exc)
            diagnosis = None

        if trace.trace_status == "PASS":
            best_trace = trace
            best_formula_set = fs
            best_diagnosis = diagnosis
            break

        elapsed = time.monotonic() - t_start
        if diagnosis is not None and cfg.tier(elapsed) < 1 and _repair_module is not None:
            try:
                repaired = repair_trace(
                    trace=trace,
                    formula_set=fs,
                    parse_obj=parse_obj,
                    diagnosis=diagnosis,
                    solver=_solve_trace,
                    all_formula_sets=formula_sets,
                    step_retry_limit=1,
                )
                if repaired.trace_status in ("PASS", "REPAIRED"):
                    best_trace = repaired
                    best_formula_set = fs
                    best_diagnosis = diagnosis
                    break
            except Exception as exc:
                logger.warning("Stage 5 repair failed: %s", exc)

        if best_trace is None or len(trace.steps) > len(best_trace.steps):
            best_trace = trace
            best_formula_set = fs
            best_diagnosis = diagnosis

    if best_trace is None:
        return {"answer": "", "confidence": 0.0, "error": "all formula paths failed"}

    # Stage 6: response assembly
    try:
        response = build_response(
            trace=best_trace,
            parse_obj=parse_obj,
            formula_set=best_formula_set,
            diagnosis=best_diagnosis,
        )
    except Exception as exc:
        logger.error("Stage 6 assembly failed: %s", exc)
        return {"answer": "", "confidence": 0.0, "error": f"assembly error: {exc}"}

    return response
