"""Interface-only LLM semantic fallback for future local model integration."""

from __future__ import annotations

from typing import Dict, List


LLM_FALLBACK_PROMPT_TEMPLATE = """
You are a semantic recovery module for a deterministic physics parser.
The deterministic parse is the source of truth.

Rules:
- Do not overwrite deterministic extracted quantities unless there is a clear unit/context error.
- Only fill missing fields: target, domain, subdomain, conditions, or step_plan.
- Preserve variable names whenever possible.
- Return JSON only.
- Do not solve the full problem.
- Do not hallucinate quantities not present or implied by common physics phrases.
- Explain proposed corrections in parser_warnings.

Problem:
{problem_text}

Partial parse:
{partial_parse}

Verifier errors:
{verifier_errors}
""".strip()


class LLMFallbackParser:
    """Stub interface for an open-source local LLM fallback parser."""

    def complete_parse(
        self,
        problem_text: str,
        partial_parse: Dict[str, object],
        verifier_errors: List[Dict[str, object]],
    ) -> Dict[str, object]:
        """Return a repaired parse object. The stub leaves the parse unchanged."""
        repaired = dict(partial_parse)
        warnings = list(repaired.get("parser_warnings", []))
        warnings.append("LLM fallback stub invoked; no semantic changes were made.")
        repaired["parser_warnings"] = warnings
        return repaired

