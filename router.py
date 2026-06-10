"""Top-level request router.

Determines whether an incoming API payload is a Type 1 (logic-based query)
or Type 2 (physics calculation) request, per design §3.1.

Detection priority
------------------
1. Explicit ``type`` or ``query_type`` field in the payload  (``"type1"`` or ``"type2"``).
2. Presence of a ``premises-NL`` or ``premises`` field → ``"type1"``.
3. Default → ``"type2"``.

This logic is intentionally kept trivial and dependency-free so that it can
run before any model is loaded.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

QueryType = Literal["type1", "type2"]

_VALID_TYPES = {"type1", "type2"}


def detect_query_type(payload: Dict[str, Any]) -> QueryType:
    """Return ``"type1"`` or ``"type2"`` for the given payload.

    Examples::

        detect_query_type({"query_type": "type1", ...})         # → "type1"
        detect_query_type({"premises-NL": [...], ...})          # → "type1"
        detect_query_type({"question": "Calculate ..."})        # → "type2"
        detect_query_type({})                                   # → "type2"
    """
    explicit: str = str(payload.get("type") or payload.get("query_type") or "").lower().strip()
    if explicit in _VALID_TYPES:
        return explicit  # type: ignore[return-value]

    if payload.get("premises-NL") or payload.get("premises"):
        return "type1"

    return "type2"
