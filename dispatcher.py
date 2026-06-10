"""Top-level request dispatcher.

Reads ``payload["type"]`` and routes to the correct pipeline.
This is the *only* routing logic -- no text classification.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from output_normalizer import (
    fallback_type1,
    fallback_type2,
    fallback_unknown,
    normalize_type1_output,
    normalize_type2_output,
)

logger = logging.getLogger(__name__)


def dispatch(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dispatch a prediction request and return the official response list.

    Always returns a single-element list. Never raises.
    """
    query_id = str(payload.get("query_id", "unknown"))
    qtype = str(payload.get("type", "")).strip().lower()
    query = str(payload.get("query", "")).strip()
    premises: list = payload.get("premises") or []
    options: list = payload.get("options") or []

    try:
        if qtype == "type1":
            raw = _solve_type1(query=query, premises=premises, options=options)
            result = normalize_type1_output(query_id, raw, options)

        elif qtype == "type2":
            raw = _solve_type2(query=query)
            result = normalize_type2_output(query_id, raw)

        else:
            logger.warning("Unknown query type %r for %s", qtype, query_id)
            result = fallback_unknown(query_id, qtype)

    except Exception as exc:
        logger.exception("Pipeline error for %s (type=%s): %s", query_id, qtype, exc)
        if qtype == "type1":
            result = fallback_type1(query_id, options)
        elif qtype == "type2":
            result = fallback_type2(query_id)
        else:
            result = fallback_unknown(query_id, qtype)

    return [result]


# ---------------------------------------------------------------------------
# Internal solver wrappers (isolate imports)
# ---------------------------------------------------------------------------

def _solve_type1(query: str, premises: list, options: list) -> Dict[str, Any]:
    from type1.type1.solver import solve
    return solve(query=query, premises=premises, options=options)


def _solve_type2(query: str) -> Dict[str, Any]:
    # type2/ is on sys.path so ``type2`` resolves to the type2/type2/ package
    from type2.solver import solve
    return solve(query=query)
