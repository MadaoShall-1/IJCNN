r"""Smoke test for the DSPy/vLLM Type 2 path.

Requires an OpenAI-compatible vLLM server on http://localhost:8000/v1.
The Docker command used during development was:

docker run -d --name exact-vllm --gpus all -p 8000:8000 ^
  -v "%USERPROFILE%\.cache\huggingface:/root/.cache/huggingface" ^
  vllm/vllm-openai:latest ^
  --model Qwen/Qwen3-8B ^
  --served-model-name qwen3-8b ^
  --max-model-len 1024 ^
  --gpu-memory-utilization 0.65
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DSPY_MODEL", "openai/qwen3-8b")
os.environ.setdefault("DSPY_API_BASE", "http://localhost:8000/v1")
os.environ.setdefault("DSPY_API_KEY", "EMPTY")

import api  # noqa: E402
from config import SolverConfig  # noqa: E402


def require_vllm_server() -> None:
    url = os.environ["DSPY_API_BASE"].rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            models = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"vLLM server is not reachable at {url}: {exc}") from exc

    ids = [item.get("id") for item in models.get("data", [])]
    model_id = os.environ["DSPY_MODEL"].replace("openai/", "", 1)
    if model_id not in ids:
        raise SystemExit(f"Model {model_id!r} not served by vLLM. Available: {ids}")


def main() -> None:
    require_vllm_server()
    cfg = SolverConfig(stage0_cache_enabled=False, stage0_use_llm_fallback=False)
    api._load_models(cfg)
    result = api._run_type2(
        {
            "id": "vllm_smoke",
            "question": "A resistor of 10 ohm is connected to a 5V battery. Find the current.",
        },
        cfg,
        time.monotonic(),
    )

    summary = {
        "solver_mode": api._type2_solver_mode,
        "answer": result.get("answer"),
        "confidence": result.get("confidence"),
        "trace_status": result.get("trace_status"),
        "steps": result.get("steps"),
    }
    print(json.dumps(summary, indent=2))

    if api._type2_solver_mode != "dspy_llm":
        raise SystemExit("Expected solver_mode=dspy_llm")
    if result.get("trace_status") != "PASS":
        raise SystemExit("Expected trace_status=PASS")
    if "0.5" not in str(result.get("answer")):
        raise SystemExit("Expected answer to contain 0.5")


if __name__ == "__main__":
    main()
