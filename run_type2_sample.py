"""Sample 50 Type2 problems and POST each to /predict."""

import csv
import io
import json
import random
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
TYPE2 = ROOT / "type2"
# The competition entrypoint is the unified root api.py, which routes
# type1/type2 to their pipelines internally.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api import app

DATASET = TYPE2 / "Dataset" / "Physics_Problems_Text_Only.csv"
OUTPUT = ROOT / "outputs" / "type2_sample50_results.jsonl"
SAMPLE_N = 50
SEED = 42
# False mimics the grading slot: unseen questions never hit the Stage 0
# cache, so every request exercises the live parser (+ vLLM LLM fallback).
USE_STAGE0_CACHE = False


def _format_eta(seconds: float) -> str:
    """Format seconds as Xs / XmYYs / XhYYm."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _print_progress(
    current: int,
    total: int,
    start_time: float,
    ok_count: int,
    empty_count: int,
    exact_count: int,
) -> None:
    """Print a one-line progress update — overwrites the previous line."""
    elapsed = time.monotonic() - start_time
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 else 0.0
    pct = 100.0 * current / total
    line = (
        f"  [{current:>2}/{total}] {pct:5.1f}% | "
        f"OK={ok_count} EMPTY={empty_count} EXACT={exact_count} | "
        f"elapsed {_format_eta(elapsed)} | ETA {_format_eta(remaining)} | "
        f"{rate:.2f} probs/s"
    )
    # \r returns to start of line, padding spaces clear any leftover chars
    print("\r" + line + " " * 6, end="", flush=True)


def _is_exact(pred: dict) -> bool:
    return (
        str(pred.get("answer", "")).strip()
        == str(pred.get("gold_answer", "")).strip()
    )


def load_dataset():
    rows = []
    with open(DATASET, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main():
    rows = load_dataset()
    print(f"Dataset: {len(rows)} rows, sampling {SAMPLE_N}")

    random.seed(SEED)
    sample = random.sample(rows, min(SAMPLE_N, len(rows)))

    results = []
    ok_count = empty_count = exact_count = 0
    t_all = time.monotonic()

    with TestClient(app) as client:
        print()
        for i, row in enumerate(sample):
            query_id = row.get("id", f"sample_{i}")
            question = row.get("question", "")
            gold_answer = row.get("answer", "")
            gold_unit = row.get("unit", "")

            payload = {
                "query_id": query_id,
                "type": "type2",
                "query": question,
                "premises": [],
                "options": [],
                "use_stage0_cache": USE_STAGE0_CACHE,
            }

            t0 = time.monotonic()
            resp = client.post("/predict", json=payload)
            elapsed = round(time.monotonic() - t0, 3)

            if resp.status_code == 200:
                data = resp.json()
                pred = data[0] if isinstance(data, list) and data else {}
            else:
                pred = {"query_id": query_id, "error": f"HTTP {resp.status_code}"}

            pred["gold_answer"] = gold_answer
            pred["gold_unit"] = gold_unit
            pred["elapsed_seconds"] = elapsed
            results.append(pred)

            if pred.get("answer"):
                ok_count += 1
            else:
                empty_count += 1
            if _is_exact(pred):
                exact_count += 1

            _print_progress(
                i + 1, len(sample), t_all, ok_count, empty_count, exact_count
            )
    # End the progress line with a newline so subsequent output starts clean
    print()

    total_time = round(time.monotonic() - t_all, 1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-problem result table
    print(f"\n{'#':>3} {'id':>8} {'status':>6} {'time':>7}  "
          f"{'predicted':<24} {'gold':<24}")
    print("-" * 80)
    for i, r in enumerate(results, start=1):
        status = "EXACT" if _is_exact(r) else ("OK" if r.get("answer") else "EMPTY")
        pred_s = f"{r.get('answer', '')} {r.get('unit', '')}".strip()[:24]
        gold_s = f"{r.get('gold_answer', '')} {r.get('gold_unit', '')}".strip()[:24]
        print(f"{i:>3} {r.get('query_id', ''):>8} {status:>6} "
              f"{r.get('elapsed_seconds', 0):>6.2f}s  {pred_s:<24} {gold_s:<24}")

    answered = sum(1 for r in results if r.get("answer"))
    exact = sum(1 for r in results if _is_exact(r))
    print(f"\n{'=' * 60}")
    print(f"Done in {_format_eta(total_time)}")
    print(f"Answered:    {answered}/{len(results)} ({100.0 * answered / len(results):.0f}%)")
    print(f"Exact match: {exact}/{len(results)} ({100.0 * exact / len(results):.0f}%)")
    print(f"Avg latency: {sum(r.get('elapsed_seconds', 0) for r in results) / len(results):.2f}s/prob")
    print(f"Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
