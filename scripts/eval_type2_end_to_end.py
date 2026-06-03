"""Split and evaluate the full Type 2 pipeline on the physics dataset.

The evaluator intentionally calls ``api._run_type2`` from raw question text so
Stage 0 parser behavior is included in the measured result. Saved Stage 0
artifacts are not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api  # noqa: E402
from config import SolverConfig  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

try:
    from pint import UnitRegistry

    _UREG = UnitRegistry()
except Exception:  # noqa: BLE001
    _UREG = None


NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "question", "answer", "unit"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path} is missing required fields: {sorted(missing)}")
    return rows


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_dataset(
    rows: List[Dict[str, str]],
    *,
    train_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    indexed = [
        {
            "dataset_id": row.get("id") or f"row_{idx}",
            "row_index": idx,
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "unit": row.get("unit", ""),
            "cot": row.get("cot", ""),
        }
        for idx, row in enumerate(rows, start=1)
    ]
    rng = random.Random(seed)
    shuffled = list(indexed)
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    return shuffled[:train_count], shuffled[train_count:]


def _first_number(text: Any) -> Optional[float]:
    match = NUM_RE.search(str(text or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _unit_alias(unit: str) -> str:
    unit = (unit or "").strip()
    replacements = {
        "": "",
        "Ω": "ohm",
        "ohms": "ohm",
        "Ohm": "ohm",
        "μF": "microfarad",
        "ÂµF": "microfarad",
        "Î¼F": "microfarad",
        "uF": "microfarad",
        "µF": "microfarad",
        "μC": "microcoulomb",
        "ÂµC": "microcoulomb",
        "Î¼C": "microcoulomb",
        "uC": "microcoulomb",
        "µC": "microcoulomb",
        "mC": "millicoulomb",
        "%": "percent",
    }
    return replacements.get(unit, unit)


def _extract_pred_unit(answer: Any) -> str:
    text = str(answer or "").strip()
    match = NUM_RE.search(text.replace(",", ""))
    if not match:
        return ""
    unit = text[match.end() :].strip()
    unit = re.split(r"[\s,.;)\]]+", unit)[0] if unit else ""
    return unit


def _convert_value(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    if not from_unit or not to_unit or _unit_alias(from_unit) == _unit_alias(to_unit):
        return value
    if _UREG is None:
        return None
    try:
        quantity = value * _UREG(_unit_alias(from_unit))
        return float(quantity.to(_unit_alias(to_unit)).magnitude)
    except Exception:  # noqa: BLE001
        return None


def compare_answer(
    predicted: Any,
    gold_answer: Any,
    gold_unit: Any,
    *,
    rel_tol: float,
    abs_tol: float,
) -> Dict[str, Any]:
    pred_num = _first_number(predicted)
    gold_num = _first_number(gold_answer)
    pred_unit = _extract_pred_unit(predicted)
    expected_unit = str(gold_unit or "").strip()

    numeric_correct = False
    converted_pred = pred_num
    if pred_num is not None and gold_num is not None:
        converted = _convert_value(pred_num, pred_unit, expected_unit)
        if converted is not None:
            converted_pred = converted
        numeric_correct = math.isclose(
            float(converted_pred),
            float(gold_num),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    unit_correct = False
    if not expected_unit:
        unit_correct = True
    elif pred_unit:
        if _unit_alias(pred_unit) == _unit_alias(expected_unit):
            unit_correct = True
        elif _convert_value(1.0, pred_unit, expected_unit) is not None:
            unit_correct = True

    return {
        "gold_number": gold_num,
        "pred_number": pred_num,
        "converted_pred_number": converted_pred,
        "gold_unit": expected_unit,
        "pred_unit": pred_unit,
        "numeric_correct": numeric_correct,
        "unit_correct": unit_correct,
        "exact_correct": bool(numeric_correct and unit_correct),
    }


def _status_bucket(result: Dict[str, Any]) -> str:
    if result.get("error"):
        return "ERROR"
    return str(result.get("trace_status") or "UNKNOWN")


def _progress(current: int, total: int, start: float, exact: int, errors: int) -> None:
    elapsed = time.monotonic() - start
    rate = current / elapsed if elapsed > 0 else 0.0
    eta = (total - current) / rate if rate > 0 else 0.0
    print(
        f"\r[{current:>4}/{total}] exact={exact} errors={errors} "
        f"rate={rate:.2f}/s eta={eta:.0f}s",
        end="",
        flush=True,
    )


def evaluate(
    test_rows: List[Dict[str, Any]],
    *,
    output_dir: Path,
    rel_tol: float,
    abs_tol: float,
    limit: Optional[int],
    stage0_llm_fallback: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    cfg = SolverConfig(
        stage0_cache_enabled=False,
        stage0_use_llm_fallback=stage0_llm_fallback,
    )
    api._load_models(cfg)

    rows = test_rows[:limit] if limit is not None else test_rows
    counters: Counter[str] = Counter()
    by_trace: Counter[str] = Counter()
    by_unit: Dict[str, Counter[str]] = defaultdict(Counter)
    by_domain: Dict[str, Counter[str]] = defaultdict(Counter)
    latencies: List[float] = []
    exact_count = 0
    numeric_count = 0
    unit_count = 0
    diagnostics: Counter[str] = Counter()
    formula_top1: Counter[str] = Counter()
    required_vars_coverages: List[float] = []
    start = time.monotonic()

    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            case_start = time.monotonic()
            payload = {
                "id": row["dataset_id"],
                "question": row["question"],
            }
            try:
                result = api._run_type2(payload, cfg, case_start)
            except Exception as exc:  # noqa: BLE001
                result = {"answer": "", "confidence": 0.0, "error": repr(exc)}

            latency = time.monotonic() - case_start
            latencies.append(latency)
            comparison = compare_answer(
                result.get("answer", ""),
                row.get("answer", ""),
                row.get("unit", ""),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            if comparison["exact_correct"]:
                exact_count += 1
            if comparison["numeric_correct"]:
                numeric_count += 1
            if comparison["unit_correct"]:
                unit_count += 1

            status = _status_bucket(result)
            by_trace[status] += 1
            domain = "unknown"
            parse_meta = {}
            if isinstance(result, dict):
                parse_meta = result.get("parse_metadata") or {}
            domains = parse_meta.get("domains", []) or [domain]
            if not domains or all(str(d).lower() == "unknown" for d in domains):
                diagnostics["unknown_domain"] += 1
            for domain_name in parse_meta.get("domains", []) or [domain]:
                by_domain[str(domain_name)]["total"] += 1
                by_domain[str(domain_name)]["exact"] += int(comparison["exact_correct"])
            by_unit[str(row.get("unit", ""))]["total"] += 1
            by_unit[str(row.get("unit", ""))]["exact"] += int(comparison["exact_correct"])

            steps = result.get("steps") if isinstance(result, dict) else []
            formula_ids = []
            covered_inputs = 0
            total_inputs = 0
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    ids = step.get("formula_ids") or []
                    formula_ids.extend([str(item) for item in ids if item])
                    if step.get("status") == "WRONG":
                        diagnostics["wrong_steps"] += 1
                if formula_ids:
                    formula_top1[formula_ids[0]] += 1
                else:
                    diagnostics["no_formula"] += 1
            if not str(result.get("answer", "")).strip():
                diagnostics["empty_answer"] += 1
            if status == "PASS" and not comparison["exact_correct"]:
                diagnostics["pass_but_wrong"] += 1
            if comparison["unit_correct"]:
                diagnostics["target_dimension_match"] += 1

            if result.get("error"):
                counters["errors"] += 1
            counters["total"] += 1

            record = {
                "dataset_id": row["dataset_id"],
                "row_index": row["row_index"],
                "question": row["question"],
                "gold_answer": row.get("answer", ""),
                "gold_unit": row.get("unit", ""),
                "pred_answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "trace_status": result.get("trace_status"),
                "error": result.get("error"),
                "latency_seconds": round(latency, 4),
                "comparison": comparison,
                "result": result,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if index % 5 == 0 or index == len(rows):
                _progress(index, len(rows), start, exact_count, counters["errors"])

    print()

    total = counters["total"]
    sorted_latencies = sorted(latencies)
    p95 = sorted_latencies[int(0.95 * (len(sorted_latencies) - 1))] if sorted_latencies else 0.0

    def _rate(count: int) -> float:
        return round(count / total, 4) if total else 0.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "exact_match_count": exact_count,
        "exact_match_rate": _rate(exact_count),
        "numeric_match_count": numeric_count,
        "numeric_match_rate": _rate(numeric_count),
        "unit_match_count": unit_count,
        "unit_match_rate": _rate(unit_count),
        "error_count": counters["errors"],
        "error_rate": _rate(counters["errors"]),
        "trace_status_counts": dict(by_trace),
        "latency_seconds": {
            "mean": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
            "p95": round(p95, 4),
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "by_gold_unit": {
            unit: {
                "total": counts["total"],
                "exact": counts["exact"],
                "exact_rate": round(counts["exact"] / counts["total"], 4)
                if counts["total"]
                else 0.0,
            }
            for unit, counts in sorted(by_unit.items())
        },
        "by_domain": {
            domain: {
                "total": counts["total"],
                "exact": counts["exact"],
                "exact_rate": round(counts["exact"] / counts["total"], 4)
                if counts["total"]
                else 0.0,
            }
            for domain, counts in sorted(by_domain.items())
        },
        "outputs": {
            "predictions": str(predictions_path),
            "stage1_diagnostics": str(output_dir / "stage1_diagnostics.json"),
        },
        "config": {
            "rel_tol": rel_tol,
            "abs_tol": abs_tol,
            "stage0_cache_enabled": False,
            "stage0_use_llm_fallback": stage0_llm_fallback,
            "dspy_model": cfg.dspy_model,
            "dspy_api_base": cfg.dspy_api_base,
            "type2_solver_mode": api._type2_solver_mode,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    stage1_diagnostics = {
        "timestamp": summary["timestamp"],
        "total": total,
        "unknown_domain_count": diagnostics["unknown_domain"],
        "unknown_domain_rate": _rate(diagnostics["unknown_domain"]),
        "no_formula_count": diagnostics["no_formula"],
        "no_formula_rate": _rate(diagnostics["no_formula"]),
        "empty_answer_count": diagnostics["empty_answer"],
        "empty_answer_rate": _rate(diagnostics["empty_answer"]),
        "pass_but_wrong_count": diagnostics["pass_but_wrong"],
        "target_dimension_match_count": diagnostics["target_dimension_match"],
        "target_dimension_match_rate": _rate(diagnostics["target_dimension_match"]),
        "formula_top1_id_distribution": dict(formula_top1.most_common(50)),
        "trace_status_counts": dict(by_trace),
        "notes": "Template support is represented by formula subtopic/id in the existing FormulaSet interface.",
    }
    _write_json(output_dir / "stage1_diagnostics.json", stage1_diagnostics)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the full Type 2 pipeline end to end.")
    parser.add_argument("--dataset", default="Dataset/Physics_Problems_Text_Only.csv")
    parser.add_argument("--output-dir", default="outputs/type2_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=None, help="Optional test-set limit for smoke runs.")
    parser.add_argument("--rel-tol", type=float, default=1e-2)
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    parser.add_argument(
        "--eval-split",
        choices=("test", "train"),
        default="test",
        help="Which deterministic split to evaluate. Use train while tuning; reserve test for final evaluation.",
    )
    parser.add_argument(
        "--disable-stage0-llm-fallback",
        action="store_true",
        help="Disable parser LLM fallback during evaluation.",
    )
    parser.add_argument("--dspy-model", default=os.getenv("DSPY_MODEL", ""))
    parser.add_argument("--dspy-api-base", default=os.getenv("DSPY_API_BASE", ""))
    parser.add_argument("--dspy-api-key", default=os.getenv("DSPY_API_KEY", "EMPTY"))
    parser.add_argument("--split-only", action="store_true")
    args = parser.parse_args()

    if args.dspy_model:
        os.environ["DSPY_MODEL"] = args.dspy_model
    if args.dspy_api_base:
        os.environ["DSPY_API_BASE"] = args.dspy_api_base
    if args.dspy_api_key:
        os.environ["DSPY_API_KEY"] = args.dspy_api_key

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    rows = _read_csv(dataset_path)
    train_rows, test_rows = split_dataset(rows, train_ratio=args.train_ratio, seed=args.seed)

    split_dir = output_dir / "splits"
    _write_jsonl(split_dir / "train.jsonl", train_rows)
    _write_jsonl(split_dir / "test.jsonl", test_rows)
    split_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "total": len(rows),
        "train": len(train_rows),
        "test": len(test_rows),
        "outputs": {
            "train": str(split_dir / "train.jsonl"),
            "test": str(split_dir / "test.jsonl"),
        },
    }
    _write_json(output_dir / "split_summary.json", split_summary)

    print(json.dumps(split_summary, ensure_ascii=False, indent=2))
    if args.split_only:
        return

    eval_rows = train_rows if args.eval_split == "train" else test_rows
    summary = evaluate(
        eval_rows,
        output_dir=output_dir,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        limit=args.limit,
        stage0_llm_fallback=not args.disable_stage0_llm_fallback,
    )
    summary["evaluated_split"] = args.eval_split
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
