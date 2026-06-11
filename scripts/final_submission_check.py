"""Print or run final submission stabilization checks.

The default mode is intentionally non-invasive: it prints the exact PowerShell
commands expected for the final reproducibility pass. Use ``--run-syntax`` to
execute only the syntax checks locally.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SYNTAX_COMMANDS = [
    ["python", "-m", "py_compile", ".\\type2\\pipeline.py"],
    ["python", "-m", "py_compile", ".\\router.py"],
    ["python", "-m", "py_compile", ".\\type2\\stage2.py"],
    ["python", "-m", "py_compile", ".\\type2\\schemas.py"],
    ["python", "-m", "py_compile", ".\\scripts\\run_api_on_dataset.py"],
    ["python", "-m", "py_compile", ".\\scripts\\analyze_api_results.py"],
]


POWERSHELL_COMMANDS = r"""
# 1. Syntax checks
python -m py_compile .\type2\pipeline.py
python -m py_compile .\router.py
python -m py_compile .\type2\stage2.py
python -m py_compile .\type2\schemas.py
python -m py_compile .\scripts\run_api_on_dataset.py
python -m py_compile .\scripts\analyze_api_results.py

# 2. Deterministic test run
python scripts\run_api_on_dataset.py `
  --dataset Dataset\Physics_Problems_Text_Only_test.csv `
  --output-dir outputs\final_test_deterministic `
  --query-type type2 `
  --type2-solver-mode deterministic

# 3. Hybrid test run
$env:DSPY_MODEL="openai/qwen3-8b-awq"
$env:DSPY_API_BASE="http://localhost:8002/v1"

python scripts\run_api_on_dataset.py `
  --dataset Dataset\Physics_Problems_Text_Only_test.csv `
  --output-dir outputs\final_test_hybrid `
  --query-type type2 `
  --type2-solver-mode hybrid

# 4. Full hybrid run
python scripts\run_api_on_dataset.py `
  --dataset Dataset\Physics_Problems_Text_Only.csv `
  --output-dir outputs\final_full_hybrid `
  --query-type type2 `
  --type2-solver-mode hybrid

# 5. Analysis
python scripts\analyze_api_results.py `
  --results outputs\final_full_hybrid\api_results.jsonl `
  --output-dir outputs\final_full_hybrid\error_analysis `
  --sample-limit 50

# Optional objective-answer evaluator
python scripts\evaluate_answers.py `
  --results outputs\final_full_hybrid\api_results.jsonl `
  --output-dir outputs\final_full_hybrid\answer_eval
""".strip()


def _run_syntax() -> int:
    for command in SYNTAX_COMMANDS:
        print("Running:", " ".join(command))
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            return completed.returncode
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-syntax", action="store_true", help="Run syntax checks instead of only printing commands.")
    args = parser.parse_args()

    if args.run_syntax:
        raise SystemExit(_run_syntax())

    print(POWERSHELL_COMMANDS)


if __name__ == "__main__":
    main()
