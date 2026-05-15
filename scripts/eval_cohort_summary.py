"""Split Stage 0 results into eval cohort (non-QA) vs QA-only and report counts.

Usage:
  python scripts/eval_cohort_summary.py --results outputs/stage0_baseline/stage0_results.jsonl
  python scripts/eval_cohort_summary.py --results outputs/stage0_baseline/stage0_results.jsonl --json-out outputs/stage0_baseline/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _is_eval(record: dict) -> bool:
    rid = str(record.get("dataset_id") or "")
    return not rid.startswith("QA")


def summarize(results_path: Path) -> dict:
    eval_status: Counter = Counter()
    eval_errors: Counter = Counter()
    qa_status: Counter = Counter()
    qa_errors: Counter = Counter()
    full_status: Counter = Counter()
    full_errors: Counter = Counter()

    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            parse = rec.get("parse") or {}
            meta = parse.get("metadata") or {}
            status = str(meta.get("verifier_status", "FAIL"))
            errors = meta.get("verifier_errors") or []
            full_status[status] += 1
            for e in errors:
                full_errors[str(e.get("error_type", "unknown"))] += 1
            if _is_eval(rec):
                eval_status[status] += 1
                for e in errors:
                    eval_errors[str(e.get("error_type", "unknown"))] += 1
            else:
                qa_status[status] += 1
                for e in errors:
                    qa_errors[str(e.get("error_type", "unknown"))] += 1

    def _block(name: str, status: Counter, errors: Counter) -> dict:
        total = sum(status.values())
        passed = status.get("PASS", 0)
        pnn = status.get("PASS_NON_NUMERIC", 0)
        failed = status.get("FAIL", 0)
        rate = round((passed + pnn) / total, 4) if total else 0.0
        return {
            "subset": name,
            "total": total,
            "PASS": passed,
            "PASS_NON_NUMERIC": pnn,
            "FAIL": failed,
            "pass_pnn_rate": rate,
            "errors": dict(errors.most_common()),
        }

    return {
        "eval_cohort": _block("eval_cohort", eval_status, eval_errors),
        "qa_subset": _block("qa_subset", qa_status, qa_errors),
        "full_set": _block("full_set", full_status, full_errors),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    summary = summarize(Path(args.results))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
