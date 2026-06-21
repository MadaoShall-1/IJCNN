"""Leg B of the hybrid Type 2 solver: LLM-derives-equation -> SymPy-computes.

For problems whose domain/extraction the deterministic Stage 0 + formula library
cannot handle (e.g. transformers, ideal gas, thin lens, latent heat, resistivity),
the LLM is asked only to MODEL the problem — produce a single governing equation,
the known values (in the units as given), and the target variable — and SymPy does
the arithmetic.  This plays to the LLM's strength (translating prose to an equation,
which the eval showed it does correctly) while removing its weakness (mental
arithmetic + free-text answer extraction that the pipeline used to corrupt).

The contract is intentionally tiny so it fits the 2048-token vLLM context.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import sympy as _sym
    _SYMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _sym = None  # type: ignore[assignment]
    _SYMPY_AVAILABLE = False


_SYSTEM_PROMPT = (
    "You are a physics solver. Reply with ONE compact JSON object and nothing else. "
    "No prose, no markdown, no <think>. Schema:\n"
    '{"equation": "<one algebraic equation, ASCII, ** for powers, e.g. V2 = U1*N2/N1>", '
    '"values": {"<var>": <number>}, "target": "<variable to solve for>", '
    '"unit": "<unit of the answer>"}\n'
    "Rules:\n"
    "- Use exactly one '=' sign. The equation may reference ONLY the variables "
    "listed in values plus the target; do NOT introduce undefined intermediate "
    "symbols. Inline any sub-formula, e.g. for series capacitors write "
    "Q = (C1*C2/(C1+C2))*V, not Q = Ceq*V.\n"
    "- Keep each value's numeric magnitude exactly as written: '3 uC' -> 3, "
    "'0.010 m^3' -> 0.010. Do NOT rescale to base SI.\n"
    "- Choose the target unit so it is dimensionally consistent with the as-written "
    "values, carrying any metric prefix: uC*V -> uJ, uF*V -> uC, mA*ohm -> mV. "
    "Unprefixed SI inputs give the plain SI unit.\n"
    "- Every variable in the equation except the target must appear in values. /no_think"
)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_think(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _call_llm(
    problem_text: str,
    *,
    api_base: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Optional[str]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": problem_text.strip()},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Leg B LLM call failed: %s", exc)
        return None


def _solve(equation: str, values: Dict[str, float], target: str) -> Optional[float]:
    """Solve an algebraic equation for ``target`` with SymPy.

    Handles two forms:
      * ``lhs = rhs``  -> solve algebraically for ``target`` (e.g. 1/f = 1/do + 1/di).
      * chained ``A = B = expr`` -> the model wrote equal quantities (e.g. the
        series-capacitor "Q1 = Q2 = Ceq*V"); evaluate the segment that is fully
        determined by ``values`` and return it.
    """
    if not _SYMPY_AVAILABLE or "=" not in equation:
        return None
    segments = [s.replace("^", "**").strip() for s in equation.split("=")]
    segments = [s for s in segments if s]
    if len(segments) < 2:
        return None
    try:
        # Only shadow actual variable names (from values + target) as symbols, so
        # that sqrt/pi/sin/cos/exp/log still resolve to SymPy functions/constants.
        # This also keeps physics names like I (current) and E (energy) as plain
        # symbols instead of SymPy's imaginary unit / Euler's number.
        var_names = {n for n in (set(values) | {target}) if n}
        local: Dict[str, Any] = {name: _sym.Symbol(name) for name in var_names}
        subs = {
            local[k]: _sym.Float(v)
            for k, v in values.items()
            if k in local and k != target and isinstance(v, (int, float))
        }

        def _real_nonneg(candidates: List[float]) -> Optional[float]:
            reals = [c for c in candidates if c is not None]
            if not reals:
                return None
            non_neg = [c for c in reals if c >= 0]
            return non_neg[0] if non_neg else reals[0]

        if len(segments) == 2:
            lhs = _sym.sympify(segments[0], locals=local)
            rhs = _sym.sympify(segments[1], locals=local)
            eq = _sym.Eq(lhs, rhs).subs(subs)
            if target in local:
                sols = _sym.solve(eq, local[target])
                vals = [float(_sym.N(s)) for s in sols
                        if getattr(_sym.N(s), "is_real", False) is not False]
                got = _real_nonneg(vals)
                if got is not None:
                    return got
            # Fall through: maybe one side is already a determined expression.

        # Chained equality, or 2-part where target isolation failed: evaluate the
        # first segment that becomes a finite number once values are substituted.
        for seg in segments:
            expr = _sym.sympify(seg, locals=local).subs(subs)
            val = _sym.N(expr)
            if getattr(val, "is_number", False) and getattr(val, "is_real", False):
                return float(val)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Leg B SymPy solve failed for %r: %s", equation, exc)
        return None


def _fmt_sig(value: float, sig: int = 3) -> str:
    import math
    if value == 0 or not math.isfinite(value):
        return "0"
    digits = sig - int(math.floor(math.log10(abs(value)))) - 1
    rounded = round(value, digits)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:g}"


def llm_sympy_solve(
    problem_text: str,
    problem_id: str,
    *,
    api_base: str,
    model: str,
    api_key: str = "EMPTY",
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: float = 20.0,
) -> Optional[Dict[str, Any]]:
    """Run the LLM->SymPy fallback. Returns a pipeline result dict or None."""
    if not api_base or not model:
        return None
    # One retry: the first call after a cold engine can time out (see vLLM warmup).
    spec = None
    for _attempt in range(2):
        content = _call_llm(
            problem_text,
            api_base=api_base,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if content:
            spec = _extract_json(content)
            if spec:
                break
            logger.warning("Leg B: could not parse JSON from LLM output: %.200s", content)
    if not spec:
        return None

    equation = str(spec.get("equation") or "")
    target = str(spec.get("target") or "")
    unit = str(spec.get("unit") or "").strip()
    raw_values = spec.get("values") or {}
    values: Dict[str, float] = {}
    for k, v in raw_values.items():
        try:
            values[str(k)] = float(v)
        except (TypeError, ValueError):
            continue

    solved = _solve(equation, values, target)
    if solved is None:
        return None

    answer = f"{_fmt_sig(solved)} {unit}".strip()
    return {
        "answer": answer,
        "confidence": 0.8,
        "chain_of_thought": (
            f"Modeled with {equation}; substituted {json.dumps(values, ensure_ascii=False)}; "
            f"solved for {target} = {_fmt_sig(solved)} {unit}."
        ),
        "trace_status": "PASS",
        "problem_id": problem_id,
        "hybrid_source": "llm_sympy",
        "steps": [
            {
                "step_id": "step_1",
                "goal": f"Model the problem as {equation} and solve for {target}.",
                "formula_ids": [],
                "intermediate_answer": answer,
                "status": "OK",
            }
        ],
    }
