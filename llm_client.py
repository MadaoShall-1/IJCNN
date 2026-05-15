from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlparse

import requests


DEFAULT_TIMEOUT_SECONDS = 120
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
BLOCKED_MODEL_NAME_PARTS = ("gpt-", "claude", "gemini")


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = 1
    max_tokens: int = 1024
    disable_thinking: bool = False
    self_check: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 1,
        max_tokens: int = 1024,
        disable_thinking: bool = False,
        self_check: bool = False,
    ) -> "LLMConfig":
        return cls(
            base_url=(base_url if base_url is not None else os.getenv("LLM_BASE_URL", "")).strip(),
            model=(model if model is not None else os.getenv("LLM_MODEL", "")).strip(),
            api_key=(api_key if api_key is not None else os.getenv("LLM_API_KEY", "")).strip(),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
            self_check=self_check,
        )

    def validate(self) -> None:
        if not self.base_url:
            raise RuntimeError("LLM_BASE_URL is not set. Pass --base-url or set the environment variable.")
        if not self.model:
            raise RuntimeError("LLM_MODEL is not set. Pass --model or set the environment variable.")
        if self.timeout_seconds <= 0:
            raise RuntimeError("--timeout must be greater than 0.")
        if self.max_retries < 0:
            raise RuntimeError("--max-retries must be >= 0.")
        if self.max_tokens <= 0:
            raise RuntimeError("--max-tokens must be greater than 0.")
        _validate_local_base_url(self.base_url)
        _validate_model_name(self.model)


def solve_physics_problem(question: str, config: LLMConfig | None = None) -> dict[str, Any]:
    """Send a physics question to a local OpenAI-compatible chat-completions API."""

    llm_config = config or LLMConfig.from_env()
    llm_config.validate()

    thinking_control = "/no_think\n" if llm_config.disable_thinking else ""
    prompt = (
        thinking_control
        +
        "You are a physics problem solver for an evaluation benchmark.\n"
        "Solve the problem carefully.\n"
        "Return JSON only, with no markdown.\n\n"
        "Required JSON schema:\n"
        '{\n  "answer": "numeric or symbolic answer only",\n'
        '  "unit": "unit only",\n'
        '  "explanation": "short explanation of the solution"\n}\n\n'
        "Rules:\n"
        "- Do not include reasoning outside JSON.\n"
        "- The answer field should not include the unit.\n"
        "- The unit field should contain only the unit.\n"
        '- If the problem asks for a dimensionless value, use "dimensionless" as unit.\n'
        "- Prefer the unit requested by the question when it is clear; otherwise use SI units.\n"
        "- Be careful with metric prefixes: p = 1e-12, n = 1e-9, micro/μ/u = 1e-6, m = 1e-3, k = 1e3.\n"
        "- If you return a prefixed unit, scale the answer to that unit. If you return an SI unit, keep the SI scale.\n"
        "- Unit sanity examples: 0.0001 F = 100 μF, not 0.1 F; 0.00001125 J = 11.25 μJ.\n"
        "- For capacitors, convert capacitance to farads before using E = 0.5 * C * U^2.\n"
        "- Check arithmetic and unit conversions before returning the final JSON.\n"
        "- Use decimal notation when possible.\n\n"
        f"Question:\n{question}"
    )

    prediction = _request_prediction(prompt, llm_config)
    if prediction.get("parse_error") and not llm_config.disable_thinking:
        fallback_config = replace(llm_config, disable_thinking=True, max_retries=0, self_check=False)
        return solve_physics_problem(question, config=fallback_config)
    if not prediction.get("parse_error") and llm_config.self_check:
        return _self_check_prediction(question, prediction, llm_config)
    return prediction


def _request_prediction(prompt: str, llm_config: LLMConfig) -> dict[str, Any]:
    payload = {
        "model": llm_config.model,
        "messages": [
            {"role": "system", "content": "You are a careful physics solver."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": llm_config.max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if llm_config.api_key:
        headers["Authorization"] = f"Bearer {llm_config.api_key}"

    last_prediction: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(llm_config.max_retries + 1):
        try:
            response = requests.post(
                _chat_completions_url(llm_config.base_url),
                headers=headers,
                json=payload,
                timeout=llm_config.timeout_seconds,
            )
            response.raise_for_status()

            response_data = response.json()
            content = _extract_message_content(response_data)
            prediction = _parse_model_output(content)
            if not prediction.get("parse_error"):
                return prediction

            last_prediction = prediction
        except Exception as exc:
            last_error = exc

        if attempt < llm_config.max_retries:
            time.sleep(min(2**attempt, 8))

    if last_prediction is not None:
        return last_prediction

    raise RuntimeError(f"LLM request failed after {llm_config.max_retries + 1} attempt(s): {last_error}")


def _self_check_prediction(question: str, prediction: dict[str, Any], llm_config: LLMConfig) -> dict[str, Any]:
    audit_prompt = (
        "/no_think\n"
        "You are auditing a physics benchmark answer for arithmetic and unit-conversion mistakes.\n"
        "Return JSON only, with the same schema: answer, unit, explanation.\n"
        "If the draft answer is correct, return it unchanged. If it has a mistake, return a corrected JSON answer.\n\n"
        "Critical unit checks:\n"
        "- p = 1e-12, n = 1e-9, micro/μ/u = 1e-6, m = 1e-3, k = 1e3.\n"
        "- 0.0001 F = 100 μF, not 0.1 F.\n"
        "- 0.00001125 J = 11.25 μJ.\n"
        "- If answer and unit are changed, make sure their product represents the same physical quantity.\n\n"
        f"Question:\n{question}\n\n"
        f"Draft JSON:\n{json.dumps(prediction, ensure_ascii=False)}"
    )
    audit_config = replace(
        llm_config,
        disable_thinking=True,
        max_retries=0,
        max_tokens=min(llm_config.max_tokens, 512),
        self_check=False,
    )
    checked = _request_prediction(audit_prompt, audit_config)
    if checked.get("parse_error"):
        prediction["self_check_attempted"] = True
        prediction["self_check_parse_error"] = True
        return prediction
    checked["self_check_attempted"] = True
    checked["self_checked"] = True
    return checked


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _validate_local_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(f"LLM_BASE_URL must be an HTTP URL, got: {base_url}")
    if parsed.hostname not in LOCAL_HOSTS:
        raise RuntimeError(
            "LLM_BASE_URL must point to a local model server "
            f"({', '.join(sorted(LOCAL_HOSTS))}); got host: {parsed.hostname}"
        )


def _validate_model_name(model: str) -> None:
    normalized = model.casefold()
    if any(part in normalized for part in BLOCKED_MODEL_NAME_PARTS):
        raise RuntimeError(
            "Closed-source model names are blocked for this competition baseline. "
            "Use a local open-source model with 8B parameters or fewer."
        )


def _extract_message_content(response_data: dict[str, Any]) -> str:
    choices = response_data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content

    content = response_data.get("content")
    if isinstance(content, str):
        return content

    raise ValueError(f"Unexpected response shape: {response_data!r}")


def _parse_model_output(raw_output: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(raw_output).strip()
    parsed = _load_json_object(cleaned)
    if isinstance(parsed, dict):
        return {
            "answer": str(parsed.get("answer", "") or ""),
            "unit": str(parsed.get("unit", "") or ""),
            "explanation": str(parsed.get("explanation", "") or ""),
        }

    return {
        "answer": "",
        "unit": "",
        "explanation": raw_output,
        "parse_error": True,
    }


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def _load_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_fragment = _extract_first_json_object(text)
    if json_fragment is None:
        return None

    try:
        return json.loads(json_fragment)
    except json.JSONDecodeError:
        return None


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        character = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
