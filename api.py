"""FastAPI prediction endpoint for the IJCNN EduQA competition.

Single endpoint: ``POST /predict``
Health check:    ``GET /health``

All routing is handled by :mod:`dispatcher` using ``payload["type"]``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Path setup: type2/ internal code uses bare imports like
# ``from parser.main import ...`` and ``from config import SolverConfig``.
# We add type2/ to sys.path so those resolve correctly.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
_TYPE2_DIR = str(_PROJECT_ROOT / "type2")
if _TYPE2_DIR not in sys.path:
    sys.path.insert(0, _TYPE2_DIR)
# Project root itself (for config.py, dispatcher.py, output_normalizer.py)
_ROOT_STR = str(_PROJECT_ROOT)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI (soft dependency)
# ---------------------------------------------------------------------------

try:
    from fastapi import Body, FastAPI
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------

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
# Startup
# ---------------------------------------------------------------------------

from config import SolverConfig
from dispatcher import dispatch

_config: SolverConfig = SolverConfig()


def _load_models(cfg: SolverConfig) -> None:
    """Initialise both pipelines at startup."""
    # Type 2 (type2/ is on sys.path, so ``type2`` = type2/type2/ package)
    from type2.solver import init as init_type2
    init_type2(cfg)

    # Type 1
    if _DSPY_AVAILABLE and cfg.dspy_model:
        try:
            from type1.type1.dspy_modules import Type1Solver
            from type1.type1.solver import set_solver
            set_solver(Type1Solver())
            logger.info("Type1Solver loaded.")
        except Exception as exc:
            logger.warning("Type1Solver not available: %s", exc)


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
        description="Type 1 (logic) and Type 2 (physics) solver.",
        version="2.0.0",
        lifespan=_lifespan,
    )

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        from type2.solver import status as type2_status
        info: Dict[str, Any] = {
            "status": "ok",
            "dspy": _DSPY_AVAILABLE,
            "sympy": _SYMPY_AVAILABLE,
        }
        info.update(type2_status())
        return info

    @app.post("/predict")
    async def predict(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
        t_start = time.monotonic()
        result_list = dispatch(payload)
        latency = round(time.monotonic() - t_start, 3)
        for r in result_list:
            r["latency_seconds"] = latency
        return JSONResponse(content=result_list)

else:
    logger.warning("FastAPI not installed; API server will not be available.")
    app = None  # type: ignore[assignment]
