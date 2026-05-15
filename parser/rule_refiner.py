"""Simple analyzer for parser error logs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List


def analyze_error_log(log_path: str = "parser_errors.jsonl") -> List[str]:
    """Group frequent verifier errors and emit human-readable rule suggestions."""
    path = Path(log_path)
    if not path.exists():
        return ["No error log found."]

    error_counts: Counter[str] = Counter()
    contexts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            text = str(entry.get("problem_text", "")).lower()
            for error in entry.get("verifier_result", {}).get("errors", []):
                error_type = error.get("error_type", "unknown")
                error_counts[error_type] += 1
                if "released from rest" in text or "from rest" in text:
                    contexts[(error_type, "released/from rest")] += 1
                if " c" in text or "°c" in text:
                    contexts[(error_type, "C unit ambiguity")] += 1
                if "electric field strength" in text:
                    contexts[(error_type, "electric field strength")] += 1

    suggestions = [f"{error_type}: {count} occurrence(s)" for error_type, count in error_counts.most_common()]
    if contexts[("missing_quantity", "released/from rest")]:
        suggestions.append("Many missing_quantity errors around 'released/from rest': consider adding or refining REST_PHRASES.")
    if contexts[("wrong_unit", "C unit ambiguity")]:
        suggestions.append("Many wrong_unit errors involving 'C': improve Celsius vs Coulomb disambiguation.")
    if contexts[("target_mismatch", "electric field strength")]:
        suggestions.append("Many target_mismatch errors involving 'electric field strength': add or refine target mapping to E.")
    return suggestions or ["No verifier errors found."]

