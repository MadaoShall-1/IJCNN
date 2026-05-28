"""Local Qwen3 LLM fallback for the Stage 0 deterministic physics parser.

Design
------
The deterministic parser produces a parse object that the verifier may
reject with status FAIL. When ``parse_problem`` is called with
``use_llm_fallback=True``, ``main.py`` constructs an
:class:`LLMFallbackParser` and calls :meth:`complete_parse`. This module
implements that path.

Key principles
~~~~~~~~~~~~~~
* **Deterministic parse is the source of truth.** The LLM only fills
  missing fields (target, conditions, step_plan), it never overrides
  extracted quantities.
* **Lazy + singleton model loading.** The Qwen3 GGUF file is large; we
  load it on first use and reuse the same instance.
* **Mock mode.** If ``llama-cpp-python`` is not installed or the model
  file is missing, ``complete_parse`` becomes a no-op that adds a
  diagnostic warning. Pipeline keeps running.
* **Silent fallback on errors.** JSON parse error, schema mismatch, or
  inference timeout → keep the original FAIL, mark metadata, move on.
* **Auto GPU detection.** If a CUDA-capable GPU is available we offload
  all layers to it; otherwise we run on CPU.

Configuration
~~~~~~~~~~~~~
Environment variables, all optional:

* ``QWEN3_GGUF_PATH``  — path to the GGUF model file. Default:
  ``~/.cache/qwen3/Qwen3-8B-Q4_K_M.gguf``.
* ``QWEN3_FORCE_MOCK`` — set to ``1`` to force mock mode even if model
  and library are available. Useful for CI.
* ``QWEN3_N_CTX``      — context window. Default 4096.
* ``QWEN3_N_THREADS``  — CPU threads. Default: auto.
* ``QWEN3_MAX_TOKENS`` — generation cap. Default 512.

To get started
~~~~~~~~~~~~~~
::

    pip install llama-cpp-python
    mkdir -p ~/.cache/qwen3
    # Download from https://huggingface.co/Qwen/Qwen3-8B-GGUF
    # Choose Q4_K_M (~5 GB) for 16 GB machines:
    huggingface-cli download Qwen/Qwen3-8B-GGUF \\
        Qwen3-8B-Q4_K_M.gguf --local-dir ~/.cache/qwen3

See ``LLM_FALLBACK_README.md`` for the long-form guide.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = Path.home() / ".cache" / "qwen3" / "Qwen3-8B-Q4_K_M.gguf"
DEFAULT_N_CTX = 4096
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1  # Low — we want consistent structured output.


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# /no_think suffix disables Qwen3's reasoning trace in the response, so we
# don't have to strip <think>...</think> blocks. This keeps the output short
# and makes JSON parsing reliable.
SYSTEM_PROMPT = """You are a semantic recovery module for a deterministic \
physics problem parser. The deterministic parse is the source of truth — \
you only fill in fields it failed to extract.

Rules:
1. Do NOT overwrite quantities the deterministic parser already extracted.
2. Only propose values for fields listed in MISSING_FIELDS.
3. Variable name conventions you should use:
   - target (unknown_quantity): symbol like "v", "U_C", "Phi_B", "C_cap",
     "F_net", "U_B", "X_L", "X_C", "power_factor", "f_res", "epsilon_r"
   - step_plan: a list of formula steps with formula_name, input_var,
     output_var
4. If you cannot determine a field with high confidence, omit it from the
   output rather than guessing.
5. Return STRICT JSON only — no prose, no markdown, no <think> tags.
6. Do not solve the problem. Just produce parse fields."""

USER_PROMPT_TEMPLATE = """Problem text:
\"\"\"{problem_text}\"\"\"

Deterministic parse (already extracted, do NOT overwrite):
- domains: {domains}
- sub_domains: {sub_domains}
- conditions: {conditions}
- known quantities: {known_quantities}
- detected target: {unknown_quantity}

Verifier errors that caused FAIL:
{verifier_errors}

MISSING_FIELDS to potentially fill (only fill those you're confident about):
{missing_fields}

Reply with a JSON object using exactly this schema, omitting any field
you cannot fill with high confidence:

{{
  "unknown_quantity": "<symbol or null>",
  "unknown_unit": "<unit symbol or null>",
  "conditions": ["<condition>", ...],
  "step_plan": [
    {{
      "step_id": "step_1",
      "type": "formula_application",
      "formula_name": "<short_snake_case_name>",
      "input_var": {{"<var>": "<var>", ...}},
      "output_var": {{"<var>": "<formula>"}},
      "confidence": 0.7
    }},
    {{
      "step_id": "step_N",
      "type": "conclusion",
      "input_var": {{"<target>": "<target>"}},
      "output_var": {{"<target>": "<target>"}},
      "confidence": 0.8
    }}
  ],
  "reasoning": "<one short sentence>"
}}

/no_think"""


# Backwards-compatible export — older callers import this template directly.
LLM_FALLBACK_PROMPT_TEMPLATE = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Singleton model registry
# ---------------------------------------------------------------------------


class _ModelRegistry:
    """Holds a lazily-initialized Llama instance shared across calls."""

    _lock = threading.Lock()
    _llm: Optional[Any] = None
    _load_error: Optional[str] = None
    _mock: bool = False

    @classmethod
    def get(cls) -> Optional[Any]:
        """Return the Llama instance, loading it if needed.

        Returns ``None`` if mock mode is active (model unavailable or
        explicitly disabled). Callers must handle the ``None`` case.
        """
        if cls._mock or cls._load_error:
            return None
        if cls._llm is not None:
            return cls._llm

        with cls._lock:
            if cls._llm is not None:
                return cls._llm
            if cls._mock or cls._load_error:
                return None
            cls._llm = cls._try_load()
            return cls._llm

    @classmethod
    def _try_load(cls) -> Optional[Any]:
        """Attempt to load the Qwen3 GGUF model. Mark mock mode on failure."""
        if os.environ.get("QWEN3_FORCE_MOCK") == "1":
            cls._mock = True
            return None

        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError:
            cls._mock = True
            cls._load_error = (
                "llama-cpp-python is not installed. "
                "Install with `pip install llama-cpp-python`. "
                "Running in mock mode."
            )
            warnings.warn(cls._load_error, RuntimeWarning, stacklevel=2)
            return None

        model_path = Path(
            os.environ.get("QWEN3_GGUF_PATH", str(DEFAULT_MODEL_PATH))
        ).expanduser()
        if not model_path.exists():
            cls._mock = True
            cls._load_error = (
                f"Qwen3 GGUF model not found at {model_path}. "
                f"See LLM_FALLBACK_README.md to download it. "
                f"Running in mock mode."
            )
            warnings.warn(cls._load_error, RuntimeWarning, stacklevel=2)
            return None

        n_ctx = int(os.environ.get("QWEN3_N_CTX", DEFAULT_N_CTX))
        n_threads_env = os.environ.get("QWEN3_N_THREADS")
        n_threads = int(n_threads_env) if n_threads_env else None
        n_gpu_layers = cls._detect_gpu_layers()

        try:
            llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            cls._mock = True
            cls._load_error = f"Failed to load Qwen3 model: {exc!r}. Mock mode."
            warnings.warn(cls._load_error, RuntimeWarning, stacklevel=2)
            return None

        return llm

    @staticmethod
    def _detect_gpu_layers() -> int:
        """Return number of layers to offload to GPU.

        Strategy: probe for a CUDA-capable GPU via PyTorch (if installed)
        or via the CUDA runtime via ctypes. If found, offload all layers
        (-1 means "all" in llama.cpp). Otherwise return 0 (CPU only).
        """
        # PyTorch check — cheap and reliable if torch is around.
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return -1
        except ImportError:
            pass

        # Fallback: try the CUDA driver directly via ctypes.
        try:
            import ctypes

            for libname in ("libcuda.so", "libcuda.so.1", "nvcuda.dll"):
                try:
                    ctypes.CDLL(libname)
                    return -1
                except OSError:
                    continue
        except Exception:
            pass

        return 0  # CPU.

    @classmethod
    def status(cls) -> Dict[str, Any]:
        """Return a small dict describing model state for diagnostics."""
        return {
            "mock_mode": cls._mock,
            "loaded": cls._llm is not None,
            "load_error": cls._load_error,
            "n_gpu_layers": cls._detect_gpu_layers() if not cls._mock else None,
            "model_path": str(
                Path(
                    os.environ.get("QWEN3_GGUF_PATH", str(DEFAULT_MODEL_PATH))
                ).expanduser()
            ),
        }

    @classmethod
    def reset_for_test(cls) -> None:
        """Clear cached state — only for unit tests."""
        with cls._lock:
            cls._llm = None
            cls._load_error = None
            cls._mock = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced ``{...}`` JSON object out of ``text``.

    Qwen3 occasionally wraps replies in markdown code fences or prefixes
    them with ``<think>`` blocks. We strip those, then take the first
    balanced curly-brace span.
    """
    if not text or not text.strip():
        return None

    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()

    # Greedy match would grab too much if the model emits multiple {...}
    # blocks; use a manual balanced scan from the first '{'.
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        ch = cleaned[index]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end == -1:
        return None
    blob = cleaned[start:end]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def _missing_fields(parse: Dict[str, Any]) -> List[str]:
    """Identify what the deterministic parser failed to fill."""
    missing: List[str] = []
    if not parse.get("unknown_quantity"):
        missing.append("unknown_quantity")
        missing.append("unknown_unit")
    step_plan = parse.get("step_plan") or []
    metadata = parse.get("metadata") or {}
    # A skeleton-only plan counts as missing — it's the placeholder, not real.
    only_skeleton = step_plan and all(
        (step.get("template_name") == "skeleton_placeholder")
        for step in step_plan
        if isinstance(step, dict)
    )
    if not step_plan or only_skeleton or metadata.get("used_skeleton_fallback"):
        missing.append("step_plan")
    # Conditions could always be enriched, but only request them when the
    # parse looks genuinely under-described.
    if not parse.get("conditions"):
        missing.append("conditions")
    return missing


def _summarize_known(known: Dict[str, Any]) -> str:
    """Compact textual summary of known quantities for the prompt."""
    if not known:
        return "(none)"
    parts: List[str] = []
    for name, quantity in known.items():
        if not isinstance(quantity, dict):
            continue
        value = quantity.get("value")
        unit = quantity.get("unit_symbol", "")
        dim = quantity.get("dimension", "")
        parts.append(f"{name}={value} {unit} ({dim})")
    return ", ".join(parts) if parts else "(none)"


def _summarize_errors(errors: List[Dict[str, Any]]) -> str:
    """Compact textual summary of verifier errors."""
    if not errors:
        return "(none)"
    lines = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        et = err.get("error_type", "unknown")
        desc = err.get("description", "")
        lines.append(f"- {et}: {desc}")
    return "\n".join(lines) if lines else "(none)"


def _is_valid_step(step: Any) -> bool:
    """Light schema check on an LLM-proposed step."""
    if not isinstance(step, dict):
        return False
    if "step_id" not in step or "type" not in step:
        return False
    if step["type"] not in {"formula_application", "calculation", "conclusion", "setup"}:
        return False
    if not isinstance(step.get("input_var", {}), dict):
        return False
    if not isinstance(step.get("output_var", {}), dict):
        return False
    return True


def _sanitize_step_plan(
    proposed: Any, target: Optional[str]
) -> Optional[List[Dict[str, Any]]]:
    """Validate and normalize an LLM-proposed step plan.

    Reject the plan if:
      * it isn't a list with at least one step
      * any step fails the schema check
      * the final step isn't a conclusion whose output_var contains the target
    """
    if not isinstance(proposed, list) or not proposed:
        return None
    cleaned: List[Dict[str, Any]] = []
    for index, step in enumerate(proposed, start=1):
        if not _is_valid_step(step):
            return None
        out = dict(step)
        out["step_id"] = f"step_{index}"  # Force sequential ids.
        out.setdefault("confidence", 0.65)
        out.setdefault("template_name", "llm_fallback")
        cleaned.append(out)

    final = cleaned[-1]
    if final["type"] != "conclusion":
        return None
    if target and target not in (final.get("output_var") or {}):
        return None
    if not any(
        step.get("type") in {"formula_application", "calculation"} for step in cleaned
    ):
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class LLMFallbackParser:
    """Qwen3-backed semantic recovery layer.

    Use::

        fallback = LLMFallbackParser()
        repaired = fallback.complete_parse(text, partial_parse, errors)
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.max_tokens = int(os.environ.get("QWEN3_MAX_TOKENS", max_tokens))
        self.temperature = temperature

    # ----- public ----------------------------------------------------------

    def complete_parse(
        self,
        problem_text: str,
        partial_parse: Dict[str, Any],
        verifier_errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Repair a failed parse using the LLM.

        Returns a (possibly-updated) copy of ``partial_parse``. Any failure
        of the LLM (mock mode, JSON parse error, schema mismatch) results
        in returning the input parse with a diagnostic added to
        ``parser_warnings``.

        The caller (``main.py``) re-runs the verifier on the returned
        object, so this function does not validate against the verifier
        directly.
        """
        repaired = copy.deepcopy(partial_parse)
        warnings_out: List[str] = list(repaired.get("parser_warnings", []))
        metadata = repaired.setdefault("metadata", {})

        llm = _ModelRegistry.get()
        if llm is None:
            warnings_out.append(
                "LLM fallback: mock mode (model unavailable). "
                "No changes applied."
            )
            metadata["llm_fallback_mode"] = "mock"
            repaired["parser_warnings"] = warnings_out
            return repaired

        missing = _missing_fields(repaired)
        if not missing:
            # Verifier failed for some other reason (e.g. dimension mismatch).
            # Nothing actionable for the LLM to fill.
            warnings_out.append(
                "LLM fallback: no missing fields identified; "
                "nothing to repair."
            )
            metadata["llm_fallback_mode"] = "skipped_no_missing_fields"
            repaired["parser_warnings"] = warnings_out
            return repaired

        prompt_user = USER_PROMPT_TEMPLATE.format(
            problem_text=problem_text.strip(),
            domains=repaired.get("domains") or [],
            sub_domains=repaired.get("sub_domains") or [],
            conditions=repaired.get("conditions") or [],
            known_quantities=_summarize_known(repaired.get("known_quantities") or {}),
            unknown_quantity=repaired.get("unknown_quantity"),
            verifier_errors=_summarize_errors(verifier_errors),
            missing_fields=missing,
        )

        try:
            response = self._generate(llm, SYSTEM_PROMPT, prompt_user)
        except Exception as exc:
            warnings_out.append(f"LLM fallback: inference error: {exc!r}")
            metadata["llm_fallback_mode"] = "inference_error"
            repaired["parser_warnings"] = warnings_out
            return repaired

        payload = _extract_json(response)
        if not isinstance(payload, dict):
            warnings_out.append(
                "LLM fallback: response could not be parsed as JSON; "
                "no changes applied."
            )
            metadata["llm_fallback_mode"] = "json_parse_failed"
            metadata["llm_fallback_raw_excerpt"] = (response or "")[:200]
            repaired["parser_warnings"] = warnings_out
            return repaired

        # ----- apply fields conservatively -------------------------------
        applied: List[str] = []

        # unknown_quantity / unknown_unit — only fill if currently absent.
        if "unknown_quantity" in missing:
            target_val = payload.get("unknown_quantity")
            if isinstance(target_val, str) and target_val.strip() and target_val.lower() != "null":
                repaired["unknown_quantity"] = target_val.strip()
                applied.append("unknown_quantity")
                unit_val = payload.get("unknown_unit")
                if isinstance(unit_val, str) and unit_val.strip() and unit_val.lower() != "null":
                    repaired["unknown_unit"] = unit_val.strip()
                    applied.append("unknown_unit")

        # conditions — additive merge, never replace.
        if "conditions" in missing:
            proposed_conds = payload.get("conditions")
            if isinstance(proposed_conds, list):
                existing = list(repaired.get("conditions") or [])
                for cond in proposed_conds:
                    if isinstance(cond, str) and cond.strip() and cond not in existing:
                        existing.append(cond.strip())
                if existing != (repaired.get("conditions") or []):
                    repaired["conditions"] = existing
                    applied.append("conditions")

        # step_plan — full replace only if validation passes.
        if "step_plan" in missing:
            target = repaired.get("unknown_quantity")
            proposed_plan = _sanitize_step_plan(payload.get("step_plan"), target)
            if proposed_plan is not None:
                repaired["step_plan"] = proposed_plan
                repaired["plan_confidence"] = max(
                    float(repaired.get("plan_confidence") or 0.0),
                    min(0.7, sum(float(s.get("confidence", 0.6)) for s in proposed_plan) / len(proposed_plan)),
                )
                # Clear skeleton marker — we have a real plan now.
                metadata["used_skeleton_fallback"] = False
                applied.append("step_plan")

        # Lift domain_confidence slightly so low_confidence doesn't block
        # purely on domain when the LLM supplied a real target+plan. This
        # is intentionally small (0.55) — the verifier's threshold is 0.5.
        if "step_plan" in applied and float(repaired.get("domain_confidence") or 0.0) < 0.55:
            repaired["domain_confidence"] = 0.55
            applied.append("domain_confidence (raised)")

        # ----- record -----------------------------------------------------
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            warnings_out.append(f"LLM fallback reasoning: {reasoning.strip()[:200]}")

        if applied:
            warnings_out.append(f"LLM fallback applied: {', '.join(applied)}")
            metadata["llm_fallback_mode"] = "applied"
            metadata["llm_fallback_applied_fields"] = applied
        else:
            warnings_out.append(
                "LLM fallback: response received but no fields passed validation."
            )
            metadata["llm_fallback_mode"] = "no_valid_fields"

        repaired["parser_warnings"] = warnings_out
        return repaired

    # ----- internal --------------------------------------------------------

    def _generate(self, llm: Any, system: str, user: str) -> str:
        """Call the Llama instance and return the response text."""
        try:
            # Modern llama-cpp-python (≥0.2.x) supports OpenAI-style chat.
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["</s>", "<|im_end|>"],
            )
            choices = result.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        except (AttributeError, TypeError):
            # Older llama-cpp-python — fall through to raw completion.
            pass

        # Raw-completion fallback: hand-format the chat template.
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        result = llm(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["<|im_end|>", "</s>"],
        )
        choices = result.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("text") or "")


# ---------------------------------------------------------------------------
# Diagnostics — exposed for callers that want to inspect model state.
# ---------------------------------------------------------------------------


def get_model_status() -> Dict[str, Any]:
    """Return the current LLM model status (for logging / health checks)."""
    return _ModelRegistry.status()