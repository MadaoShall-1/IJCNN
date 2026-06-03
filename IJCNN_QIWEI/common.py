"""Shared lightweight interfaces for IJCNN-Qiwei."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Stage0Input:
    question: str
    premises_nl: list[str] = field(default_factory=list)
    premises_fol: list[str] = field(default_factory=list)
    choices: Any = ""
    expected_answer: str = ""
    record_id: Any = None
    question_id: Any = None
    rag_context: str = ""


@dataclass
class LLMEndpointConfig:
    model: str
    api_base: str
    api_key: str = "EMPTY"
    request_timeout_seconds: float = 60.0


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        """Return raw model text for a chat-style request."""


class OpenAICompatibleLLMClient:
    """Minimal OpenAI-compatible client for vLLM or compatible local servers."""

    def __init__(self, config: LLMEndpointConfig) -> None:
        self.config = config
        load_env_file()
        self.ssl_context = self._build_ssl_context()

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        api_key = (
            self.config.api_key
            if self.config.api_key and self.config.api_key != "EMPTY"
            else os.environ.get("MINIGPT_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        )
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout_seconds,
                context=self.ssl_context,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response has no choices: {body}")
        message = choices[0].get("message") or {}
        return str(message.get("content", "")).strip()

    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext:
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()


def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency."""
    env_path = path or Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
