#!/usr/bin/env python3
"""Type 1 prediction capability module (no HTTP server).

This module owns the full Type 1 answering chain used by the unified
root-level ``api.py``:

    retained WM/SSM/Transformer classifier  (choice questions)
      -> vLLM reasoner                      (free-form questions / fallback)
        -> deterministic heuristic          (last resort)

It contains no FastAPI code on purpose: the only HTTP endpoint in this
project is the root ``E:\\LLM-vllm\\api.py``. The logic here was extracted
verbatim from the retired ``exact_api.py`` so that answers are unchanged.
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


ASCII_UNIT_ALIASES = {
    "Ω": "ohm",
    "\\Omega": "ohm",
    "Ω": "ohm",
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


class Type1Predictor:
    """Full Type 1 answering chain: retained model -> vLLM -> heuristic."""

    def __init__(self, vllm_client: VLLMClient | None = None) -> None:
        self.vllm = vllm_client or VLLMClient()
        self._type1_retained = None
        self._type1_retained_error = ""

    def predict_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        query = ExactQuery.from_payload(payload)
        started = time.time()
        if query.type != "type1":
            raise ValueError("Type1Predictor only handles type == 'type1'")
        result = self.predict_type1(query)
        # premises_used drives P2 (50% of the Type 1 score). The dedicated
        # vLLM selector measured 40% exact-set / 0.768 Jaccard vs 3.3% / 0.356
        # for the token-overlap heuristic (92 gold-labelled dataset questions,
        # 2026-06-11), so it overrides whatever the answer path produced.
        selected = self._select_premises_vllm(query)
        if selected is not None:
            result["premises_used"] = selected
        result["latency_seconds"] = round(time.time() - started, 4)
        # EXACT schema does not require latency; keep it internal only if explicitly enabled.
        if os.getenv("EXACT_INCLUDE_LATENCY", "0") != "1":
            result.pop("latency_seconds", None)
        return [self._normalize_result(query, result)]

    def _select_premises_vllm(self, query: ExactQuery) -> list[int] | None:
        """Ask the vLLM which premises the answer actually depends on."""
        if not query.premises or not self.vllm.enabled:
            return None
        system = (
            "You select which premises are needed to answer a logic question. "
            'Return ONLY a JSON object like {"premises_used": [0, 2]} with '
            "0-based indices of the premises that are required to derive the "
            "answer. Be minimal but complete."
        )
        numbered = [f"[{i}] {p}" for i, p in enumerate(query.premises)]
        user = json.dumps(
            {"question": query.query[:600], "premises": numbered},
            ensure_ascii=False,
        ) + " /no_think"
        try:
            parsed = self.vllm.chat_json(system, user, max_tokens=300)
            raw = parsed.get("premises_used", [])
            selected = sorted({
                int(item) for item in raw
                if isinstance(item, (int, float)) and 0 <= int(item) < len(query.premises)
            })
            return selected or None
        except Exception:
            return None

    def predict_type1(self, query: ExactQuery) -> dict[str, Any]:
        # The retained model is a candidate classifier (Yes/No/Uncertain/A-D),
        # so it only applies to choice questions. Free-form number/text
        # questions (options == []) go to the vLLM reasoner instead.
        retained = self._get_type1_retained_predictor() if query.options else None
        if retained is not None:
            try:
                result = retained.predict(
                    query=query.query,
                    premises=query.premises,
                    options=query.options,
                )
                # Note (2026-06-11): a confidence fallback (Uncertain or
                # top_prob<0.62 -> vLLM retry) was evaluated and REJECTED —
                # live it cost 2 points on the official distribution (73/93
                # vs 75/93), roughly doubled latency on gated questions, and
                # showed no live gain on out-of-distribution phrasings.
                # Round1 gating below covers a newer OOD failure mode where
                # invalid MCQ labels were coerced to option A.
                fallback_reason = self._retained_fallback_reason(query, result)
                if fallback_reason:
                    return self._fallback_type1(
                        query,
                        fallback_reason=fallback_reason,
                        retained_result=result,
                    )
                return self._validate_type1(query, result)
            except Exception as exc:
                self._type1_retained_error = f"{type(exc).__name__}: {exc}"
        if self.vllm.enabled:
            return self._fallback_type1(query, fallback_reason="no_retained_path")
        fallback = self._heuristic_type1(query)
        if self._type1_retained_error:
            fallback["reasoning"]["steps"].insert(0, f"retained model fallback reason: {self._type1_retained_error}")
        return fallback

    def _call_type1_vllm(self, query: ExactQuery) -> dict[str, Any]:
        if query.options:
            return self._call_type1_vllm_entailment(query)

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

    def _call_type1_vllm_entailment(self, query: ExactQuery) -> dict[str, Any]:
        labels = self._option_labels(query.options)
        option_items = [
            {
                "label": labels[idx],
                "value": option,
                "text": self._option_text(query.query, labels[idx], option),
            }
            for idx, option in enumerate(query.options)
        ]
        ynu_options = {"Yes", "No", "Uncertain"}
        if set(query.options) >= ynu_options:
            task = (
                "Decide whether the queried conclusion follows from the premises. "
                "For questions asking whether the premises prove, establish, "
                "guarantee, or satisfy a requirement, answer Yes only if the "
                "full requested conclusion is entailed; answer No when any "
                "required condition is missing or contradicted. Use Uncertain "
                "only for a direct factual query whose truth is not determined "
                "by the premises, especially when a premise says no premise "
                "states that fact."
            )
            required = {
                "verdict": "entailed | not_proven_or_contradicted | unknown_fact",
                "answer": "exactly one of the provided option values",
                "premises_used": "list[int]",
                "reasoning": {"type": "fol", "steps": ["short proof steps"]},
            }
        else:
            task = (
                "Evaluate every multiple-choice option separately. Pick the one "
                "option whose statement is entailed by the premises. Do not choose "
                "an option merely because it is mentioned; it must be derivable."
            )
            required = {
                "option_verdicts": {
                    item["label"]: "entailed | contradicted | not_established"
                    for item in option_items
                },
                "answer": "exactly one of the provided option values",
                "premises_used": "list[int]",
                "reasoning": {"type": "fol", "steps": ["short proof steps"]},
            }

        system = (
            "You are the EXACT 2026 Type 1 proof checker. Return only valid JSON. "
            "Use 0-based premise indices. Keep premises_used minimal but complete. "
            "The answer must exactly match one provided option value."
        )
        user = json.dumps(
            {
                "task": task,
                "query_id": query.query_id,
                "query": query.query,
                "premises": [
                    {"index": idx, "text": premise}
                    for idx, premise in enumerate(query.premises)
                ],
                "options": option_items,
                "required_json": required,
            },
            ensure_ascii=False,
        ) + " /no_think"
        return self.vllm.chat_json(system, user, max_tokens=900)

    def _fallback_type1(
        self,
        query: ExactQuery,
        *,
        fallback_reason: str,
        retained_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.vllm.enabled:
            try:
                parsed = self._call_type1_vllm(query)
                result = self._validate_type1(query, parsed)
                result["reasoning"]["steps"].insert(0, f"fallback reason: {fallback_reason}")
                if retained_result is not None:
                    retained_answer = TextTools.clean(retained_result.get("answer"))
                    result["reasoning"]["steps"].insert(1, f"retained answer bypassed: {retained_answer}")
                return result
            except Exception as exc:
                fallback = self._heuristic_type1(query)
                fallback["reasoning"]["steps"].insert(0, f"vLLM fallback reason: {type(exc).__name__}")
                fallback["reasoning"]["steps"].insert(0, f"retained fallback reason: {fallback_reason}")
                if self._type1_retained_error:
                    fallback["reasoning"]["steps"].insert(0, f"retained model error: {self._type1_retained_error}")
                return fallback

        fallback = self._heuristic_type1(query)
        fallback["reasoning"]["steps"].insert(0, f"retained fallback reason: {fallback_reason}")
        return fallback

    def _retained_fallback_reason(self, query: ExactQuery, result: dict[str, Any]) -> str:
        if os.getenv("TYPE1_RETAINED_GATING", "1") == "0":
            return ""
        if not query.options:
            return ""

        answer = TextTools.clean(result.get("answer"))
        probabilities = result.get("candidate_probabilities")
        if not isinstance(probabilities, dict):
            probabilities = {}
        ranked = sorted(
            (
                (TextTools.clean(label), float(prob))
                for label, prob in probabilities.items()
                if isinstance(prob, (int, float))
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        top_label = ranked[0][0] if ranked else answer
        top_prob = ranked[0][1] if ranked else float(result.get("model_top_probability") or 0.0)
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_prob - second_prob

        normalized_options = {self._normalize_option_token(option) for option in query.options}
        normalized_answer = self._normalize_option_token(answer)
        normalized_top = self._normalize_option_token(top_label)
        ynu_options = {"yes", "no", "uncertain"}
        is_ynu = ynu_options <= normalized_options
        is_abcd = normalized_options <= {"a", "b", "c", "d"} and len(normalized_options) >= 2

        if normalized_answer not in normalized_options:
            return f"retained_answer_not_in_options:{answer or '<blank>'}"
        if is_abcd and normalized_top == "uncertain":
            return "retained_top_uncertain_for_mcq"
        if is_abcd and normalized_answer == "a" and top_prob < float(os.getenv("TYPE1_MCQ_A_CONFIDENCE_FLOOR", "0.70")):
            return f"weak_mcq_a_bias:{top_prob:.3f}"
        if is_abcd and top_prob < float(os.getenv("TYPE1_MCQ_CONFIDENCE_FLOOR", "0.40")):
            return f"low_mcq_confidence:{top_prob:.3f}"
        if is_abcd and margin < float(os.getenv("TYPE1_MCQ_MARGIN_FLOOR", "0.04")):
            return f"low_mcq_margin:{margin:.3f}"
        if (
            is_ynu
            and os.getenv("TYPE1_YNU_PROOF_FALLBACK", "1") != "0"
        ):
            return "ynu_entailment_proof_fallback"
        if is_ynu and top_prob < float(os.getenv("TYPE1_YNU_CONFIDENCE_FLOOR", "0.58")):
            return f"low_ynu_confidence:{top_prob:.3f}"
        return ""

    def _get_type1_retained_predictor(self) -> Any:
        if os.getenv("TYPE1_USE_RETAINED_MODEL", "1") == "0":
            return None
        if self._type1_retained is not None:
            return self._type1_retained
        try:
            from .type1_retained_predictor import Type1RetainedPredictor

            self._type1_retained = Type1RetainedPredictor()
            self._type1_retained_error = ""
            return self._type1_retained
        except Exception as exc:
            self._type1_retained_error = f"{type(exc).__name__}: {exc}"
            return None

    def _validate_type1(self, query: ExactQuery, parsed: dict[str, Any]) -> dict[str, Any]:
        answer = TextTools.clean(parsed.get("answer"))
        if query.options and answer not in query.options:
            answer = self._coerce_option_answer(answer, query.options)
        premises_used = self._clean_premise_indices(parsed.get("premises_used"), len(query.premises))
        explanation = TextTools.clean(parsed.get("explanation")) or f"The selected answer is {answer}."
        reasoning = self._clean_reasoning(parsed.get("reasoning"), default_type="fol")
        answer = self._adjust_ynu_answer(query, parsed, answer, reasoning)
        return {
            "query_id": query.query_id,
            "answer": answer,
            "unit": "",
            "explanation": explanation,
            "premises_used": premises_used,
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

    def _coerce_option_answer(self, answer: str, options: list[str]) -> str:
        labels = self._option_labels(options)
        normalized = self._normalize_option_token(answer)
        for idx, label in enumerate(labels):
            if normalized == label.lower() and idx < len(options):
                return options[idx]
        return self._closest_option(answer, options)

    def _adjust_ynu_answer(
        self,
        query: ExactQuery,
        parsed: dict[str, Any],
        answer: str,
        reasoning: dict[str, Any],
    ) -> str:
        normalized_options = {option.lower(): option for option in query.options}
        if not {"yes", "no", "uncertain"} <= set(normalized_options):
            return answer

        verdict = TextTools.clean(parsed.get("verdict")).lower()
        if verdict in {"entailed", "yes"}:
            adjusted = normalized_options["yes"]
        elif verdict in {"not_proven_or_contradicted", "contradicted", "not_established", "no"}:
            adjusted = normalized_options["no"]
        elif verdict in {"unknown_fact", "unknown", "uncertain"}:
            adjusted = normalized_options["uncertain"]
        else:
            adjusted = answer

        text = " ".join(str(step) for step in reasoning.get("steps") or []).lower()
        query_text = query.query.lower()
        no_premise_marker = "no premise states" in text
        missing_marker = bool(
            re.search(
                r"\b(no information about|do not have information|does not establish|"
                r"not established|not determined|missing|required condition|"
                r"but there is no|but we do not have)\b",
                text,
            )
        )
        proof_question = bool(
            re.search(
                r"\b(prove|establish|guarantee|satisfy every requirement|"
                r"does .* follow|do the premises)\b",
                query_text,
            )
        )

        if no_premise_marker and not proof_question:
            return normalized_options["uncertain"]
        if missing_marker and proof_question:
            return normalized_options["no"]
        if no_premise_marker:
            return normalized_options["uncertain"]
        return adjusted

    def _normalize_option_token(self, value: str) -> str:
        text = TextTools.clean(value).lower()
        match = re.match(r"^(?:option\s+)?([a-d])(?:[\s.)]|$)", text)
        if match:
            return match.group(1)
        return text

    def _option_labels(self, options: list[str]) -> list[str]:
        if all(self._normalize_option_token(option) in {"a", "b", "c", "d"} for option in options):
            return [self._normalize_option_token(option).upper() for option in options]
        return [chr(ord("A") + idx) for idx in range(len(options))]

    def _option_text(self, query_text: str, label: str, option: str) -> str:
        pattern = rf"(?:^|\n)\s*{re.escape(label)}\.\s*(.*?)(?=\n\s*[A-D]\.\s*|\Z)"
        match = re.search(pattern, query_text, flags=re.S)
        if match:
            return TextTools.clean(match.group(1))
        return TextTools.clean(option)

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

    def _ascii_unit(self, unit: str) -> str:
        cleaned = TextTools.clean(unit)
        for src, dst in ASCII_UNIT_ALIASES.items():
            cleaned = cleaned.replace(src, dst)
        cleaned = cleaned.replace(" ", "")
        return cleaned
