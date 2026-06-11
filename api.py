"""Unified EXACT 2026 prediction endpoint.

This is THE single competition API. It accepts the unified EXACT schema on
``POST /predict`` and routes each query by its ``type`` field:

    POST /predict
      |
      +-- type == "type1"  ->  type1/IJCNN-Qiwei pipeline
      |                        (retained WM/SSM/Transformer classifier for
      |                         choice questions, vLLM reasoner for free-form)
      |
      +-- type == "type2"  ->  type2/ staged physics pipeline
                               (parse -> formula retrieval -> deterministic
                                symbolic solve -> diagnose/repair -> response)

Both pipelines share the single vLLM server (one 8B-class model loaded at
any moment, competition rule Q3). Serving:

    uvicorn api:app --host 0.0.0.0 --port 8080

Configuration is read from ``type1/IJCNN-Qiwei/.env`` (process environment
variables take precedence). The type1/ and type2/ directories contain pure
capability modules only — this file is the single HTTP server in the project.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_TYPE1_DIR = _ROOT / "type1" / "IJCNN-Qiwei"
_TYPE2_DIR = _ROOT / "type2"


# ---------------------------------------------------------------------------
# Environment: .env loader + DSPy propagation
# ---------------------------------------------------------------------------

def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines; existing process variables win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key and key not in os.environ:
            os.environ[key] = value


def _propagate_dspy_env() -> None:
    """Derive DSPY_* from VLLM_* so both pipelines share one vLLM server."""
    base = os.getenv("VLLM_BASE_URL", "").strip()
    model = os.getenv("VLLM_MODEL", "").strip()
    if base and not os.getenv("DSPY_API_BASE"):
        os.environ["DSPY_API_BASE"] = base
    if model and not os.getenv("DSPY_MODEL"):
        os.environ["DSPY_MODEL"] = model if model.startswith("openai/") else f"openai/{model}"
    if not os.getenv("DSPY_API_KEY"):
        os.environ["DSPY_API_KEY"] = os.getenv("VLLM_API_KEY", "EMPTY")


_load_env_file(_TYPE1_DIR / ".env")
_propagate_dspy_env()


# ---------------------------------------------------------------------------
# Type 2 pipeline (pure capability module type2/type2/pipeline.py)
# ---------------------------------------------------------------------------
# type2's code uses bare top-level imports (config, parser, type2), so its
# directory must lead sys.path. ``import type2.pipeline`` then resolves to
# type2/type2/pipeline.py.

_type2_str = str(_TYPE2_DIR)
while _type2_str in sys.path:
    sys.path.remove(_type2_str)
sys.path.insert(0, _type2_str)

import type2.pipeline as _t2  # noqa: E402


# ---------------------------------------------------------------------------
# Type 1 pipeline (pure capability module ijcnn_qiwei/type1_predictor.py)
# ---------------------------------------------------------------------------

_type1_str = str(_TYPE1_DIR)
if _type1_str not in sys.path:
    sys.path.append(_type1_str)

from ijcnn_qiwei.type1_predictor import Type1Predictor  # noqa: E402

_type1_predictor = Type1Predictor()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Route one unified-schema payload to its pipeline; returns a list."""
    query_type = str(payload.get("type") or "").strip().lower()
    if query_type == "type1":
        return _type1_predictor.predict_payload(payload)
    if query_type == "type2":
        t_start = time.monotonic()
        result = _t2._run_type2(payload, _t2._config, t_start)
        return [_t2._submission_result(payload, result, "type2")]
    raise ValueError("type must be 'type1' or 'type2'")


def _warm_pipelines() -> None:
    """Load both pipelines so the first graded query pays no warm-up cost."""
    logger.info("Warming Type 2 pipeline...")
    _t2._load_models(_t2._config, load_type1=False)
    logger.info("Warming Type 1 retained model...")
    _type1_predictor._get_type1_retained_predictor()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

try:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment,misc]

if _FASTAPI_AVAILABLE:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        _warm_pipelines()
        yield

    app = FastAPI(
        title="EXACT 2026 Unified Solver",
        description="Single /predict endpoint routing Type 1 (logic) and Type 2 (physics).",
        version="3.0.0",
        lifespan=_lifespan,
    )

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        retained = _type1_predictor._get_type1_retained_predictor()
        # The RAG memory silently degrades to tfidf/numpy when BGE cannot
        # load; surface the live backend so grading-day self-checks catch it
        # (expected: "bge" — anything else costs ~6 points of accuracy).
        rag_backend = ""
        if retained is not None:
            rag_backend = getattr(
                getattr(retained.trainer, "rag_memory", None), "vector_backend", ""
            )
        return {
            "status": "ok",
            "vllm_base_url": os.getenv("VLLM_BASE_URL", ""),
            "vllm_model": os.getenv("VLLM_MODEL", ""),
            "type1_retained_model_loaded": retained is not None,
            "type1_retained_model_error": _type1_predictor._type1_retained_error,
            "type1_rag_backend": rag_backend,
            "type2_solver_mode": _t2._type2_solver_mode,
            "type2_retriever_loaded": _t2._retriever is not None,
        }

    @app.post("/predict")
    async def predict(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
        try:
            return JSONResponse(content=dispatch(payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

else:
    logger.warning("FastAPI not installed; API server will not be available.")
    app = None  # type: ignore[assignment]
