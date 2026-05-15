"""JSONL error logging for failed Stage 0 verifier runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


DEFAULT_LOG_PATH = Path("parser_errors.jsonl")


def log_verifier_failure(
    problem_text: str,
    parse_object: Dict[str, object],
    verifier_result: Dict[str, object],
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append a failed verifier event to a JSONL log."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "problem_text": problem_text,
        "parse_object": parse_object,
        "verifier_result": verifier_result,
        "stage": "stage_0",
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

