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
# TYPE2 must come AFTER ROOT so that root api.py / dispatcher.py are found first
if str(TYPE2) not in sys.path:
    sys.path.append(str(TYPE2))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api import app

DATASET = TYPE2 / "Dataset" / "Physics_Problems_Text_Only.csv"
OUTPUT = ROOT / "outputs" / "type2_sample50_results.jsonl"
SAMPLE_N = 50
SEED = 42


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
    t_all = time.monotonic()

    with TestClient(app) as client:
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

            status = "OK" if pred.get("answer") else "EMPTY"
            print(
                f"[{i+1:>2}/{len(sample)}] {query_id:>8}  {status:>5}  "
                f"{elapsed:>6.2f}s  pred={pred.get('answer','')!r} "
                f"unit={pred.get('unit','')!r}  "
                f"gold={gold_answer!r} {gold_unit}"
            )

    total_time = round(time.monotonic() - t_all, 1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    answered = sum(1 for r in results if r.get("answer"))
    exact = sum(
        1 for r in results
        if str(r.get("answer", "")).strip() == str(r.get("gold_answer", "")).strip()
    )
    print(f"\n{'='*60}")
    print(f"Done in {total_time}s")
    print(f"Answered: {answered}/{len(results)}")
    print(f"Exact match: {exact}/{len(results)}")
    print(f"Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
