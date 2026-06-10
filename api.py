"""FastAPI prediction endpoint for the IJCNN EduQA competition.

Serves both Type 1 (logic-based) and Type 2 (physics calculation) pipelines
through a single ``POST /predict`` endpoint.  The request type is detected by
:func:`router.detect_query_type` and dispatched accordingly.

Type 2 pipeline flow per request:
  Stage 0 → Stage 1 (formula retrieval, beam_n paths)
    → for each path: Stage 2+3 (SolveTrace) → Stage 4 (diagnose) → Stage 5 (repair if FAIL)
    → pick best passing trace → Stage 6 (build_response)

Startup loads:
  - FormulaRetriever (warm formula library)
  - SolveTrace, DiagnosticReasonerModule, RepairSolveTrace (if DSPy available)
  - Type1Solver (if DSPy available)

Health endpoint:
  GET /health  → {"status": "ok", "dspy": bool, "sympy": bool}
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI (soft dependency)
# ---------------------------------------------------------------------------

try:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    Body = None  # type: ignore[assignment,misc]
    FastAPI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------

from config import SolverConfig
from router import detect_query_type
from parser.main import parse_problem as _parse_stage0
from parser.llm_fallback import get_model_status as _get_stage0_llm_status

import type1.pipeline as _type1_pipeline

from parser.schemas import ProblemParseObject
from type2.stage1 import FormulaRetriever
from type2.stage2 import DeterministicSolveTrace, replay_trace_deterministically
from type2.stage4 import diagnose_trace
from type2.stage5 import repair_trace, select_repair_formula
from type2.stage6 import build_response
from type2.special_cases import try_special_case


def _dict_to_parse_obj(d: Dict[str, Any]) -> ProblemParseObject:
    """Convert the dict returned by parser.main.parse_problem to ProblemParseObject."""
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

try:
    import dspy as _dspy
    _DSPY_AVAILABLE = True
except ModuleNotFoundError:
    _DSPY_AVAILABLE = False

try:
    import sympy as _sympy
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------

_retriever: Optional[FormulaRetriever] = None
_solve_trace = None          # SolveTrace instance (DSPy-guarded)
_repair_module = None        # RepairSolveTrace instance (DSPy-guarded)
_type1_solver = None         # Type1Solver instance (DSPy-guarded)
_config: SolverConfig = SolverConfig()
_stage0_cache: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
_stage0_cache_path: Optional[str] = None
_dspy_lm_configured = False
_type2_solver_mode = "unloaded"


def _normalize_problem_text(text: str) -> str:
    """Normalize problem text for stable cache lookup."""
    return " ".join(str(text).strip().split())


def _resolve_stage0_cache_path(cfg: SolverConfig) -> Path:
    path = Path(cfg.stage0_cache_results_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _load_stage0_cache(cfg: SolverConfig) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load Stage 0 JSONL artifacts and index by id and normalized question."""
    global _stage0_cache, _stage0_cache_path

    cache_path = str(_resolve_stage0_cache_path(cfg))
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
                logger.warning("Skipping invalid Stage 0 cache line %d: %s", line_no, exc)
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
    logger.info(
        "Loaded Stage 0 cache from %s (%d ids, %d questions).",
        path,
        len(by_id),
        len(by_text),
    )
    return _stage0_cache


def _payload_bool(payload: Dict[str, Any], key: str, default: bool) -> bool:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(raw)


_ANSWER_UNIT_RE = re.compile(
    r"^\s*([+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)(?:\s*([^\d\s,;].*?))?\s*$"
)


def _ascii_unit(unit: Any) -> str:
    """Normalize display units to the ASCII convention requested by EXACT."""
    text = str(unit or "").strip()
    if not text:
        return ""
    replacements = {
        "Ω": "ohm",
        "Ω": "ohm",
        "μ": "u",
        "µ": "u",
        "·": "*",
        "²": "^2",
        "³": "^3",
        "⁻": "-",
        "°": "deg",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.replace("Ohm", "ohm").replace("ohms", "ohm")
    text = text.strip(" .,:;")
    return re.sub(r"\s+", "", text)


def _split_answer_unit(answer: Any) -> tuple[str, str]:
    """Return numerical answer and unit separately for Type 2 submissions."""
    text = str(answer or "").strip()
    if not text:
        return "", ""
    match = _ANSWER_UNIT_RE.match(text)
    if not match:
        return text, ""
    value = match.group(1).replace(",", ".")
    unit = _ascii_unit(match.group(2) or "")
    return value, unit


def _reasoning_steps(result: Dict[str, Any], query_type: str) -> List[str]:
    if query_type == "type1":
        steps = result.get("fol") or result.get("cot") or []
        return [str(step) for step in steps if str(step).strip()]

    raw_steps = result.get("steps") or []
    steps: List[str] = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        parts = [
            step.get("goal"),
            ", ".join(str(fid) for fid in step.get("formula_ids") or []),
            step.get("intermediate_answer"),
        ]
        text = " | ".join(str(part) for part in parts if part)
        if text:
            steps.append(text)
    return steps


def _premises_used(payload: Dict[str, Any], result: Dict[str, Any], query_type: str) -> List[int]:
    if query_type != "type1":
        return []

    premises = [str(item) for item in payload.get("premises") or []]
    if not premises:
        return []

    used = result.get("premises") or []
    if not used:
        return list(range(len(premises)))

    indices: List[int] = []
    for premise in used:
        try:
            idx = premises.index(str(premise))
        except ValueError:
            continue
        if idx not in indices:
            indices.append(idx)
    return indices or list(range(len(premises)))


def _choice_answer(answer: str, options: Any) -> str:
    if not isinstance(options, list) or not options:
        return answer

    choices = [str(option) for option in options]
    if answer in choices:
        return answer

    if len(answer.strip()) == 1 and answer.strip().upper() in {"A", "B", "C", "D"}:
        idx = ord(answer.strip().upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]

    answer_norm = answer.strip().lower()
    for choice in choices:
        if choice.strip().lower() == answer_norm:
            return choice
    return answer


def _submission_result(
    payload: Dict[str, Any],
    result: Dict[str, Any],
    query_type: str,
) -> Dict[str, Any]:
    query_id = str(payload.get("query_id") or payload.get("id") or result.get("problem_id") or "")
    raw_answer = str(result.get("answer") or "").strip()

    if query_type == "type2":
        answer, unit = _split_answer_unit(raw_answer)
    else:
        answer = _choice_answer(raw_answer, payload.get("options"))
        unit = ""

    explanation = str(
        result.get("explanation")
        or result.get("chain_of_thought")
        or "Solved by the configured pipeline."
    ).strip()
    if not explanation:
        explanation = "Solved by the configured pipeline."

    steps = _reasoning_steps(result, query_type)
    reasoning = {"type": "fol" if query_type == "type1" else "cot", "steps": steps} if steps else None

    return {
        "query_id": query_id,
        "answer": answer,
        "unit": unit,
        "explanation": explanation,
        "premises_used": _premises_used(payload, result, query_type),
        "reasoning": reasoning,
    }


def _get_stage0_parse(
    problem_text: str,
    problem_id: str,
    payload: Dict[str, Any],
    cfg: SolverConfig,
) -> Dict[str, Any]:
    """Return a Stage 0 parse from cache when enabled, otherwise parse live."""
    use_cache = cfg.stage0_cache_enabled and _payload_bool(
        payload,
        "use_stage0_cache",
        True,
    )
    if use_cache:
        cache = _load_stage0_cache(cfg)
        parse = cache["by_id"].get(problem_id)
        cache_key = "id"
        if parse is None:
            parse = cache["by_text"].get(_normalize_problem_text(problem_text))
            cache_key = "question"
        if parse is not None:
            parse = dict(parse)
            metadata = dict(parse.get("metadata") or {})
            metadata["stage0_cache_hit"] = True
            metadata["stage0_cache_key"] = cache_key
            metadata["stage0_cache_path"] = str(_resolve_stage0_cache_path(cfg))
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


def _load_models(cfg: SolverConfig, load_type1: bool = True) -> None:
    global _retriever, _solve_trace, _repair_module, _type1_solver, _dspy_lm_configured, _type2_solver_mode

    if _retriever is None:
        logger.info("Loading FormulaRetriever...")
        _retriever = FormulaRetriever()
    _solve_trace = DeterministicSolveTrace()
    _repair_module = _solve_trace
    _type2_solver_mode = "deterministic_sympy"

    if _DSPY_AVAILABLE:
        if cfg.dspy_model:
            lm_kwargs = {
                "model": cfg.dspy_model,
                "api_key": cfg.dspy_api_key,
                "max_tokens": cfg.dspy_max_tokens,
                "temperature": cfg.dspy_temperature,
            }
            if cfg.dspy_api_base:
                lm_kwargs["api_base"] = cfg.dspy_api_base
            _dspy.configure(lm=_dspy.LM(**lm_kwargs))
            _dspy_lm_configured = True
            logger.info(
                "Configured DSPy LM model=%s api_base=%s",
                cfg.dspy_model,
                cfg.dspy_api_base or "<provider-default>",
            )
            from type2.stage2 import SolveTrace
            from type2.stage5 import RepairSolveTrace
            _solve_trace = SolveTrace()
            _repair_module = RepairSolveTrace()
            _type2_solver_mode = "dspy_llm"
            logger.info("DSPy Type 2 modules loaded.")
        else:
            _dspy_lm_configured = False
            logger.warning(
                "DSPy is installed but no LM is configured. Set DSPY_MODEL "
                "before starting uvicorn. Using deterministic SymPy Type 2 solver."
            )
    else:
        logger.warning("DSPy not available; using deterministic SymPy Type 2 solver.")

    # Type 1 solver
    if load_type1:
        try:
            from type1.dspy_modules import Type1Solver
            _type1_solver = Type1Solver()
            logger.info("Type1Solver loaded.")
        except Exception as exc:
            logger.warning("Type1Solver not available: %s", exc)
    else:
        _type1_solver = None


# ---------------------------------------------------------------------------
# Type 2 pipeline orchestrator
# ---------------------------------------------------------------------------

def _run_type2(
    payload: Dict[str, Any],
    cfg: SolverConfig,
    t_start: float,
) -> Dict[str, Any]:
    """Run the full Type 2 pipeline for a single request."""

    problem_text = str(
        payload.get("query") or payload.get("question") or payload.get("problem") or payload.get("text") or ""
    ).strip()
    problem_id = str(payload.get("query_id") or payload.get("id") or "unknown")

    if not problem_text:
        return {"answer": "", "confidence": 0.0, "error": "empty problem text"}

    # ── Stage 0: parse ───────────────────────────────────────────────────────
    try:
        parse_dict = _get_stage0_parse(
            problem_text,
            problem_id,
            payload,
            cfg,
        )
        parse_obj = _dict_to_parse_obj(parse_dict)
    except Exception as exc:
        logger.error("Stage 0 parse failed for %s: %s", problem_id, exc)
        return {"answer": "", "confidence": 0.0, "error": f"parse error: {exc}"}

    # ── Special-case pre-solve ────────────────────────────────────────────────
    try:
        special_trace = try_special_case(parse_obj, problem_id)
    except Exception as exc:
        logger.debug("Special-case check failed: %s", exc)
        special_trace = None

    if special_trace is not None and special_trace.trace_status == "PASS":
        elapsed = time.monotonic() - t_start
        from type2.schemas import FormulaSet
        empty_fs = FormulaSet(formulas={}, retrieval_confidence=0.0, path_index=0)
        response = build_response(
            trace=special_trace,
            parse_obj=parse_obj,
            formula_set=empty_fs,
            diagnosis=None,
        )
        response["latency_seconds"] = round(elapsed, 3)
        response["problem_id"] = problem_id
        response["hybrid_source"] = "deterministic"
        response["query_type"] = "type2"
        return response

    # ── Stage 1: formula retrieval ───────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    beam_n = 1 if cfg.tier(elapsed) >= 1 else cfg.beam_n

    try:
        formula_sets = _retriever.retrieve(parse_obj, beam_n=beam_n)
    except Exception as exc:
        logger.error("Stage 1 retrieval failed: %s", exc)
        return {"answer": "", "confidence": 0.0, "error": f"retrieval error: {exc}"}

    if not formula_sets:
        return {"answer": "", "confidence": 0.0, "error": "no formula sets found"}

    # ── Stages 2+3, 4, 5: trace generation, diagnosis, repair ───────────────
    best_trace = None
    best_formula_set = formula_sets[0]
    best_diagnosis = None

    for fs in formula_sets:
        elapsed = time.monotonic() - t_start
        if cfg.tier(elapsed) >= 3:
            logger.warning("Hard latency stop for %s", problem_id)
            break

        if _solve_trace is None:
            # No DSPy — return stub with retrieval metadata only
            from type2.schemas import TraceObject
            stub = TraceObject(problem_id=problem_id, formula_path_index=fs.path_index)
            stub.trace_status = "FAIL"
            best_trace = stub
            best_formula_set = fs
            break

        # Stage 2+3
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

        # Stage 4: diagnose
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

        # Stage 5: repair (skip under high latency)
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

        # Keep best-so-far (most steps completed)
        if best_trace is None or len(trace.steps) > len(best_trace.steps):
            best_trace = trace
            best_formula_set = fs
            best_diagnosis = diagnosis

    if best_trace is None:
        return {"answer": "", "confidence": 0.0, "error": "all formula paths failed"}

    if cfg.dspy_model and best_trace.trace_status in ("PASS", "REPAIRED"):
        try:
            best_trace = replay_trace_deterministically(
                trace=best_trace,
                parse_obj=parse_obj,
                formula_set=best_formula_set,
            )
        except Exception as exc:
            logger.warning("Deterministic replay of LLM trace failed: %s", exc)

    # ── Stage 6: response assembly ───────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    assembler = None  # LLM assembler only under Tier 0

    try:
        response = build_response(
            trace=best_trace,
            parse_obj=parse_obj,
            formula_set=best_formula_set,
            diagnosis=best_diagnosis,
            explanation_assembler=assembler,
        )
    except Exception as exc:
        logger.error("Stage 6 assembly failed: %s", exc)
        return {"answer": "", "confidence": 0.0, "error": f"assembly error: {exc}"}

    response["problem_id"] = problem_id
    response["stage0_cache_hit"] = bool(parse_obj.metadata.get("stage0_cache_hit"))
    if parse_obj.metadata.get("stage0_cache_key"):
        response["stage0_cache_key"] = parse_obj.metadata.get("stage0_cache_key")
    return response


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        global _config
        _load_models(_config)
        yield

    app = FastAPI(
        title="IJCNN EduQA Solver",
        description="Type 1 (logic) and Type 2 (physics) solver for the IJCNN competition.",
        version="1.0.0",
        lifespan=_lifespan,
    )

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "dspy": _DSPY_AVAILABLE,
            "dspy_lm_configured": _dspy_lm_configured,
            "dspy_model": _config.dspy_model,
            "dspy_api_base": _config.dspy_api_base,
            "sympy": _SYMPY_AVAILABLE,
            "type2_solver_mode": _type2_solver_mode,
            "retriever_loaded": _retriever is not None,
            "stage0_use_llm_fallback": _config.stage0_use_llm_fallback,
            "stage0_cache_enabled": _config.stage0_cache_enabled,
            "stage0_cache_results_path": str(_resolve_stage0_cache_path(_config)),
            "stage0_llm_fallback": _get_stage0_llm_status(),
        }

    @app.post("/predict")
    async def predict(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
        t_start = time.monotonic()
        query_type = detect_query_type(payload)

        try:
            if query_type == "type1":
                result = _type1_pipeline.run(
                    payload=payload,
                    config=_config,
                    solver=_type1_solver,
                )
            else:
                result = _run_type2(payload, _config, t_start)
        except Exception as exc:
            logger.exception("Pipeline error for %s: %s", query_type, exc)
            raise HTTPException(status_code=500, detail=str(exc))

        result["query_type"] = query_type
        result["latency_seconds"] = round(time.monotonic() - t_start, 3)
        return JSONResponse(content=[_submission_result(payload, result, query_type)])

    @app.post("/predict/type1")
    async def predict_type1(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
        t_start = time.monotonic()
        try:
            result = _type1_pipeline.run(
                payload=payload,
                config=_config,
                solver=_type1_solver,
            )
        except Exception as exc:
            logger.exception("Type 1 pipeline error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))
        result["query_type"] = "type1"
        result["latency_seconds"] = round(time.monotonic() - t_start, 3)
        return JSONResponse(content=result)

    @app.post("/predict/type2")
    async def predict_type2(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
        t_start = time.monotonic()
        try:
            result = _run_type2(payload, _config, t_start)
        except Exception as exc:
            logger.exception("Type 2 pipeline error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))
        result["query_type"] = "type2"
        result["latency_seconds"] = round(time.monotonic() - t_start, 3)
        return JSONResponse(content=result)

else:
    # FastAPI not installed — expose a no-op placeholder so imports don't crash
    logger.warning("FastAPI not installed; API server will not be available.")
    app = None  # type: ignore[assignment]
