#!/usr/bin/env python3
"""EXACT 2026 HTTP interface.

The competition calls one POST endpoint with a unified schema for Type 1 and
Type 2 questions. The test input explicitly provides the type field, so this
module does not infer, merge, or fuse Type 1 and Type 2. It only performs hard
routing by payload["type"] and then calls the corresponding solver prompt.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
except Exception:  # pragma: no cover - lets the predictor run without FastAPI installed.
    FastAPI = None
    HTTPException = RuntimeError


ASCII_UNIT_ALIASES = {
    "Ω": "ohm",
    "\\Omega": "ohm",
    "Ω": "ohm",
    "μF": "uF",
    "\\mu F": "uF",
    "\\muF": "uF",
    "µF": "uF",
    "μC": "uC",
    "µC": "uC",
    "μA": "uA",
    "µA": "uA",
    "°": "degree",
}


class TextTools:
    @staticmethod
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(TextTools.clean(item) for item in value)
        return re.sub(r"\s+", " ", str(value)).strip()


@dataclass
class ExactQuery:
    query_id: str
    type: str
    query: str
    premises: list[str]
    options: list[str]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ExactQuery":
        missing = [key for key in ("query_id", "type", "query", "premises", "options") if key not in payload]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")
        premises = payload.get("premises")
        options = payload.get("options")
        if not isinstance(premises, list):
            raise ValueError("premises must be a list")
        if not isinstance(options, list):
            raise ValueError("options must be a list")
        return cls(
            query_id=TextTools.clean(payload.get("query_id")),
            type=TextTools.clean(payload.get("type")).lower(),
            query=TextTools.clean(payload.get("query")),
            premises=[TextTools.clean(item) for item in premises],
            options=[TextTools.clean(item) for item in options],
        )


class VLLMClient:
    """Small OpenAI-compatible client for a local or hosted vLLM server."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 45.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = model or os.getenv("VLLM_MODEL") or ""
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or "EMPTY"
        self.timeout = float(os.getenv("VLLM_TIMEOUT", str(timeout)))

    @property
    def enabled(self) -> bool:
        return bool(self.model)

    def chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("VLLM_MODEL is not set")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        text = self._post_json("/chat/completions", body)
        return self._extract_json(text)

    def _post_json(self, path: str, body: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = choices[0].get("text", "")
        return TextTools.clean(content)

    def _extract_json(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.S)
            if not match:
                raise
            return json.loads(match.group(0))


class ExactPredictor:
    def __init__(self, vllm_client: VLLMClient | None = None) -> None:
        self.vllm = vllm_client or VLLMClient()

    def predict_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        query = ExactQuery.from_payload(payload)
        started = time.time()
        # EXACT provides the type. Do not auto-detect or fuse Type 1/Type 2.
        if query.type == "type1":
            result = self.predict_type1(query)
        elif query.type == "type2":
            result = self.predict_type2(query)
        else:
            raise ValueError("type must be 'type1' or 'type2'")
        result["latency_seconds"] = round(time.time() - started, 4)
        # EXACT schema does not require latency; keep it internal only if explicitly enabled.
        if os.getenv("EXACT_INCLUDE_LATENCY", "0") != "1":
            result.pop("latency_seconds", None)
        return [self._normalize_result(query, result)]

    def predict_type1(self, query: ExactQuery) -> dict[str, Any]:
        if self.vllm.enabled:
            try:
                parsed = self._call_type1_vllm(query)
                return self._validate_type1(query, parsed)
            except Exception as exc:
                fallback = self._heuristic_type1(query)
                fallback["reasoning"]["steps"].insert(0, f"vLLM fallback reason: {type(exc).__name__}")
                return fallback
        return self._heuristic_type1(query)

    def predict_type2(self, query: ExactQuery) -> dict[str, Any]:
        if self.vllm.enabled:
            try:
                parsed = self._call_type2_vllm(query)
                return self._validate_type2(query, parsed)
            except Exception as exc:
                fallback = self._heuristic_type2(query)
                fallback["reasoning"]["steps"].insert(0, f"vLLM fallback reason: {type(exc).__name__}")
                return fallback
        return self._heuristic_type2(query)

    def _call_type1_vllm(self, query: ExactQuery) -> dict[str, Any]:
        system = (
            "You are the EXACT 2026 Type 1 logic solver. Return only valid JSON. "
            "If options is non-empty, answer must be exactly one option string. "
            "premises_used must contain only 0-based indices from the premises list."
        )
        user = json.dumps(
            {
                "task": "Solve a logic-based educational query.",
                "query_id": query.query_id,
                "query": query.query,
                "premises": query.premises,
                "options": query.options,
                "required_json": {
                    "answer": "string",
                    "explanation": "non-empty string",
                    "premises_used": "list[int]",
                    "reasoning": {"type": "fol", "steps": ["short structured steps"]},
                },
            },
            ensure_ascii=False,
        )
        return self.vllm.chat_json(system, user, max_tokens=700)

    def _call_type2_vllm(self, query: ExactQuery) -> dict[str, Any]:
        system = (
            "You are the EXACT 2026 Type 2 physics solver. Return only valid JSON. "
            "The answer field must contain the numerical value only; put the ASCII unit in unit. "
            "Use standard units such as A, V, ohm, V/m, J, W, uF, nC."
        )
        user = json.dumps(
            {
                "task": "Solve the physics problem.",
                "query_id": query.query_id,
                "problem": query.query,
                "required_json": {
                    "answer": "numeric value or short text without unit",
                    "unit": "ASCII unit string",
                    "explanation": "non-empty string",
                    "reasoning": {"type": "cot", "steps": ["calculation steps"]},
                },
            },
            ensure_ascii=False,
        )
        return self.vllm.chat_json(system, user, max_tokens=900)

    def _validate_type1(self, query: ExactQuery, parsed: dict[str, Any]) -> dict[str, Any]:
        answer = TextTools.clean(parsed.get("answer"))
        if query.options and answer not in query.options:
            answer = self._closest_option(answer, query.options)
        premises_used = self._clean_premise_indices(parsed.get("premises_used"), len(query.premises))
        explanation = TextTools.clean(parsed.get("explanation")) or f"The selected answer is {answer}."
        reasoning = self._clean_reasoning(parsed.get("reasoning"), default_type="fol")
        return {
            "query_id": query.query_id,
            "answer": answer,
            "unit": "",
            "explanation": explanation,
            "premises_used": premises_used,
            "reasoning": reasoning,
        }

    def _validate_type2(self, query: ExactQuery, parsed: dict[str, Any]) -> dict[str, Any]:
        answer = self._strip_unit(TextTools.clean(parsed.get("answer")))
        unit = self._ascii_unit(TextTools.clean(parsed.get("unit")))
        if not unit:
            answer, unit = self._split_answer_unit(answer)
        explanation = TextTools.clean(parsed.get("explanation")) or f"The computed answer is {answer} {unit}."
        reasoning = self._clean_reasoning(parsed.get("reasoning"), default_type="cot")
        return {
            "query_id": query.query_id,
            "answer": answer,
            "unit": unit,
            "explanation": explanation,
            "premises_used": [],
            "reasoning": reasoning,
        }

    def _heuristic_type1(self, query: ExactQuery) -> dict[str, Any]:
        used = self._select_relevant_premises(query.query, query.premises)
        context = " ".join([query.query, *query.premises]).lower()
        if query.options:
            yes_no_uncertain = {"Yes", "No", "Uncertain"}
            if set(query.options) >= yes_no_uncertain:
                answer = "Uncertain"
                threshold_match = re.search(r"at least\s+(\d+(?:\.\d+)?).{0,80}?(\d+(?:\.\d+)?)", context)
                below_threshold = bool(threshold_match and float(threshold_match.group(2)) < float(threshold_match.group(1)))
                if below_threshold or re.search(r"\bnot\b|\bno\b|\bnever\b|\bbelow\b|\bless than\b|\bfailed\b|\bineligible\b", context):
                    answer = "No"
                elif re.search(r"\btherefore\b|\beligible\b|\bsatisfies\b|\bcompleted\b|\bgreater than\b|\bat least\b", context):
                    answer = "Yes"
                if answer not in query.options:
                    answer = query.options[0]
            else:
                answer = self._best_overlap_option(query.query, query.premises, query.options)
        else:
            answer = self._short_free_form_answer(query.query, query.premises)
        return {
            "query_id": query.query_id,
            "answer": answer,
            "unit": "",
            "explanation": f"Selected {answer} using the query and the most relevant premises.",
            "premises_used": used,
            "reasoning": {
                "type": "proof",
                "steps": [
                    "Read the Type 1 query and premises.",
                    f"Selected premise indices: {used}.",
                    f"Returned answer: {answer}.",
                ],
            },
        }

    def _heuristic_type2(self, query: ExactQuery) -> dict[str, Any]:
        answer, unit, steps = self._simple_physics_solver(query.query)
        return {
            "query_id": query.query_id,
            "answer": answer,
            "unit": unit,
            "explanation": "Computed with the local physics fallback. Use vLLM for full competition coverage.",
            "premises_used": [],
            "reasoning": {"type": "cot", "steps": steps},
        }

    def _simple_physics_solver(self, text: str) -> tuple[str, str, list[str]]:
        lower = text.lower()
        numbers = [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d+)?", lower)]
        ohms = [float(item) for item in re.findall(r"([-+]?\d+(?:\.\d+)?)\s*(?:ohm|Ω|\\omega)\b", lower)]
        volts = [float(item) for item in re.findall(r"([-+]?\d+(?:\.\d+)?)\s*v\b", lower)]
        if "parallel" in lower and "resistor" in lower and len(ohms) >= 2 and volts:
            r1, r2, v = ohms[0], ohms[1], volts[0]
            req = 1.0 / (1.0 / r1 + 1.0 / r2)
            current = v / req
            return self._format_number(current), "A", [
                f"Parallel resistance: 1/Req = 1/{r1:g} + 1/{r2:g}.",
                f"Req = {req:g} ohm.",
                f"I = V/Req = {v:g}/{req:g} = {current:g} A.",
            ]
        if {"voltage", "current", "resistance"} & set(re.findall(r"[a-z]+", lower)) and len(numbers) >= 2:
            if "current" in lower and ("voltage" in lower or " v" in lower) and ("resistance" in lower or "ohm" in lower):
                value = numbers[0] / numbers[1]
                return self._format_number(value), "A", ["Applied Ohm's law I = V/R."]
        return "Uncertain", "", ["Local fallback could not confidently solve this physics problem."]

    def _normalize_result(self, query: ExactQuery, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "query_id": query.query_id,
            "answer": TextTools.clean(result.get("answer")) or "Uncertain",
            "unit": self._ascii_unit(TextTools.clean(result.get("unit"))),
            "explanation": TextTools.clean(result.get("explanation")) or "No explanation was generated.",
            "premises_used": self._clean_premise_indices(result.get("premises_used"), len(query.premises)),
            "reasoning": result.get("reasoning") if isinstance(result.get("reasoning"), dict) else None,
        }

    def _select_relevant_premises(self, query: str, premises: list[str]) -> list[int]:
        query_tokens = self._tokens(query)
        scored = []
        for idx, premise in enumerate(premises):
            premise_tokens = self._tokens(premise)
            overlap = len(query_tokens & premise_tokens)
            if overlap:
                scored.append((overlap, idx))
        if not scored:
            return list(range(len(premises)))
        scored.sort(reverse=True)
        return sorted(idx for _score, idx in scored[: min(3, len(scored))])

    def _best_overlap_option(self, query: str, premises: list[str], options: list[str]) -> str:
        context_tokens = self._tokens(" ".join([query, *premises]))
        best = max(options, key=lambda option: len(self._tokens(option) & context_tokens))
        return best

    def _short_free_form_answer(self, query: str, premises: list[str]) -> str:
        text = " ".join([query, *premises])
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]
        return "Uncertain"

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    def _closest_option(self, answer: str, options: list[str]) -> str:
        normalized = answer.lower().strip()
        for option in options:
            if normalized == option.lower().strip():
                return option
        for option in options:
            if re.search(rf"\b{re.escape(option.lower())}\b", normalized):
                return option
        return options[0] if options else answer

    def _clean_premise_indices(self, value: Any, premise_count: int) -> list[int]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            try:
                idx = int(item)
            except Exception:
                continue
            if 0 <= idx < premise_count and idx not in cleaned:
                cleaned.append(idx)
        return cleaned

    def _clean_reasoning(self, value: Any, default_type: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"type": default_type, "steps": []}
        steps = value.get("steps")
        if not isinstance(steps, list):
            steps = []
        return {
            "type": TextTools.clean(value.get("type")) or default_type,
            "steps": [TextTools.clean(step) for step in steps if TextTools.clean(step)][:8],
        }

    def _strip_unit(self, answer: str) -> str:
        answer, _unit = self._split_answer_unit(answer)
        return answer

    def _split_answer_unit(self, text: str) -> tuple[str, str]:
        match = re.match(r"^\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*([A-Za-z/]+|ohm|Ω|Ω)?\s*$", text, flags=re.I)
        if not match:
            return text, ""
        value, unit = match.groups()
        return value, self._ascii_unit(unit or "")

    def _ascii_unit(self, unit: str) -> str:
        cleaned = TextTools.clean(unit)
        for src, dst in ASCII_UNIT_ALIASES.items():
            cleaned = cleaned.replace(src, dst)
        cleaned = cleaned.replace(" ", "")
        return cleaned

    def _format_number(self, value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6g}"


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Run: pip install -r requirements.txt")
    app = FastAPI(title="IJCNN-Qiwei EXACT 2026 API", version="1.0")
    predictor = ExactPredictor()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "vllm_enabled": predictor.vllm.enabled, "vllm_base_url": predictor.vllm.base_url}

    @app.post("/predict")
    def predict(payload: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return predictor.predict_payload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except urllib.error.URLError as exc:
            raise HTTPException(status_code=502, detail=f"vLLM connection failed: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    return app


app = create_app() if FastAPI is not None else None
