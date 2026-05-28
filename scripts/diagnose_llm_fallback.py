"""Diagnose why LLM fallback isn't being applied.

Run from project root:
    python scripts/diagnose_llm_fallback.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print("=" * 70)
    print("LLM Fallback Diagnostic")
    print("=" * 70)

    # 1. Check llama-cpp-python install
    print("\n[1/5] llama-cpp-python installation")
    try:
        import llama_cpp  # noqa
        print(f"  ✓ llama-cpp-python installed (version: {llama_cpp.__version__})")
    except ImportError as exc:
        print(f"  ✗ NOT INSTALLED: {exc}")
        print("    Fix: pip install llama-cpp-python")
        return 1

    # 2. Check model file
    print("\n[2/5] Qwen3 GGUF model file")
    default_path = Path.home() / ".cache" / "qwen3" / "Qwen3-8B-Q4_K_M.gguf"
    model_path = Path(os.environ.get("QWEN3_GGUF_PATH", str(default_path))).expanduser()
    print(f"  Looking for: {model_path}")
    if model_path.exists():
        size_gb = model_path.stat().st_size / 1e9
        print(f"  ✓ Found ({size_gb:.2f} GB)")
    else:
        print(f"  ✗ NOT FOUND")
        print(f"    Fix: download via huggingface-cli or set QWEN3_GGUF_PATH")
        return 1

    # 3. Check environment variables
    print("\n[3/5] Environment variables")
    for var in ("QWEN3_FORCE_MOCK", "QWEN3_GGUF_PATH", "QWEN3_N_CTX", "QWEN3_MAX_TOKENS"):
        val = os.environ.get(var)
        print(f"  {var}: {val if val is not None else '(unset)'}")
    if os.environ.get("QWEN3_FORCE_MOCK") == "1":
        print("  ✗ QWEN3_FORCE_MOCK=1 is set — fallback will be in mock mode")
        print("    Fix: unset QWEN3_FORCE_MOCK")
        return 1

    # 4. Check model loads via the registry
    print("\n[4/5] Model registry status")
    from parser.llm_fallback import get_model_status, _ModelRegistry
    status = get_model_status()
    print(f"  Initial status (pre-load): {json.dumps(status, default=str)}")
    print("  Triggering load (may take 10-30 s)...")
    llm = _ModelRegistry.get()
    if llm is None:
        status_after = get_model_status()
        print(f"  ✗ Load returned None")
        print(f"  Final status: {json.dumps(status_after, default=str)}")
        if status_after.get("load_error"):
            print(f"  Load error: {status_after['load_error']}")
        return 1
    print(f"  ✓ Model loaded successfully")
    status_after = get_model_status()
    print(f"  Status after load: {json.dumps(status_after, default=str)}")

    # 5. Try generating
    print("\n[5/5] Generation smoke test")
    from parser.llm_fallback import LLMFallbackParser, SYSTEM_PROMPT
    parser = LLMFallbackParser()
    try:
        response = parser._generate(
            llm,
            SYSTEM_PROMPT,
            'Return exactly this JSON and nothing else: {"test": "ok"}',
        )
        print(f"  Raw response (first 300 chars):\n  {response[:300]!r}")
        if "ok" in response.lower() or "test" in response.lower():
            print("  ✓ Model can generate and follow simple instructions")
        else:
            print("  ⚠ Model generated but didn't follow instruction")
    except Exception as exc:
        print(f"  ✗ Generation failed: {exc!r}")
        return 1

    print("\n" + "=" * 70)
    print("All checks passed. LLM fallback should work.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())