from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from evaluator import compare_answer
from llm_client import LLMConfig, solve_physics_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline physics benchmark.")
    parser.add_argument("--data", required=True, help="Path to Physics_Problems_Text_Only.xlsx")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of evaluated examples")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N filtered examples")
    parser.add_argument("--prefix", type=str, default=None, help="Filter rows by id prefix")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV, JSONL, and summary outputs")
    parser.add_argument("--base-url", default=None, help="Local OpenAI-compatible API base URL. Overrides LLM_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key. Overrides LLM_API_KEY.")
    parser.add_argument("--model", default=None, help="Model name. Overrides LLM_MODEL.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries per example after request or JSON failures")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum generated tokens per example")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Add /no_think for Qwen3-style models. Faster, but may reduce accuracy.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Ask the same local model to audit its JSON answer for arithmetic/unit mistakes.",
    )
    parser.add_argument("--save-every", type=int, default=25, help="Write partial outputs every N examples")
    return parser.parse_args()


def load_dataset(path: Path, prefix: str | None = None, limit: int | None = None) -> pd.DataFrame:
    dataframe = pd.read_excel(path)

    required_columns = ["id", "question", "answer", "unit"]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    dataframe = dataframe.dropna(subset=["question", "answer", "unit"])
    dataframe = dataframe[
        dataframe["question"].astype(str).str.strip().ne("")
        & dataframe["answer"].astype(str).str.strip().ne("")
        & dataframe["unit"].astype(str).str.strip().ne("")
    ]

    if prefix:
        prefix_text = prefix.strip().upper()
        dataframe = dataframe[dataframe["id"].astype(str).str.upper().str.startswith(prefix_text)]

    if limit is not None:
        dataframe = dataframe.head(limit)

    return dataframe.reset_index(drop=True)


def _id_prefix(value: Any) -> str:
    text = str(value).strip().upper()
    if not text:
        return "UNKNOWN"

    prefix = []
    for character in text:
        if character.isalpha():
            prefix.append(character)
        else:
            break

    return "".join(prefix) or text


def _progress_iterator(items: list[dict[str, Any]]):
    try:
        from tqdm import tqdm

        return tqdm(items, desc="Evaluating", unit="example", ascii=True)
    except Exception:
        return _plain_progress_iterator(items)


def _plain_progress_iterator(items: list[dict[str, Any]]):
    total = len(items)
    for index, item in enumerate(items, start=1):
        print(f"Evaluating {index}/{total}: {item['id']}")
        yield item


def _write_outputs(output_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)

    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def _build_summary(
    results: list[dict[str, Any]],
    prefix_totals: dict[str, int],
    prefix_correct: dict[str, int],
    parse_error_count: int,
    numeric_match_count: int,
    unit_match_count: int,
    correct_count: int,
) -> dict[str, Any]:
    total = len(results)
    accuracy = (correct_count / total) if total else 0.0
    accuracy_by_prefix = {
        prefix: (prefix_correct[prefix] / prefix_totals[prefix]) if prefix_totals[prefix] else 0.0
        for prefix in sorted(prefix_totals)
    }
    error_breakdown: dict[str, int] = defaultdict(int)
    for row in results:
        error_breakdown[str(row.get("error_type", "unknown"))] += 1

    return {
        "total_evaluated": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "parse_errors": parse_error_count,
        "numeric_matches": numeric_match_count,
        "unit_matches": unit_match_count,
        "accuracy_by_prefix": accuracy_by_prefix,
        "error_breakdown": dict(sorted(error_breakdown.items())),
    }


def _classify_error(parse_error: bool, numeric_match: bool, unit_match: bool, correct: bool) -> str:
    if correct:
        return "correct"
    if parse_error:
        return "parse_error"
    if not numeric_match and unit_match:
        return "numeric_mismatch"
    if numeric_match and not unit_match:
        return "unit_mismatch"
    if not numeric_match and not unit_match:
        return "numeric_and_unit_mismatch"
    return "unknown"


def _numeric_ratio(comparison: dict[str, Any]) -> float | None:
    pred_value = comparison.get("converted_pred_answer_num")
    gold_value = comparison.get("converted_gold_answer_num")
    if pred_value is None or gold_value is None:
        return None
    try:
        pred_number = float(pred_value)
        gold_number = float(gold_value)
    except (TypeError, ValueError):
        return None
    if gold_number == 0 or not math.isfinite(pred_number) or not math.isfinite(gold_number):
        return None
    return pred_number / gold_number


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataframe = load_dataset(data_path, prefix=args.prefix, limit=None)
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.start:
        dataframe = dataframe.iloc[args.start :].reset_index(drop=True)
    if args.limit is not None:
        dataframe = dataframe.head(args.limit).reset_index(drop=True)

    records = dataframe.to_dict(orient="records")
    llm_config = LLMConfig.from_env(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        disable_thinking=args.disable_thinking,
        self_check=args.self_check,
    )
    llm_config.validate()

    results: list[dict[str, Any]] = []
    prefix_totals = defaultdict(int)
    prefix_correct = defaultdict(int)
    parse_error_count = 0
    numeric_match_count = 0
    unit_match_count = 0
    correct_count = 0
    output_dir = Path(args.output_dir)

    print(f"Model: {llm_config.model}")
    print(f"Endpoint: {llm_config.base_url}")
    print(f"Examples: {len(records)}")
    print(f"Output dir: {output_dir}")
    print("")

    for record in _progress_iterator(records):
        question = str(record["question"])
        gold_answer = str(record["answer"])
        gold_unit = str(record["unit"])
        example_id = str(record["id"])
        prefix = _id_prefix(example_id)

        prediction: dict[str, Any]
        try:
            prediction = solve_physics_problem(question, config=llm_config)
        except Exception as exc:
            prediction = {
                "answer": "",
                "unit": "",
                "explanation": str(exc),
                "parse_error": True,
            }

        parse_error = bool(prediction.get("parse_error", False))
        if parse_error:
            parse_error_count += 1

        comparison = compare_answer(
            str(prediction.get("answer", "")),
            str(prediction.get("unit", "")),
            gold_answer,
            gold_unit,
        )

        numeric_match = bool(comparison["numeric_match"])
        unit_match = bool(comparison["unit_match"])
        correct = bool(comparison["correct"])
        error_type = _classify_error(parse_error, numeric_match, unit_match, correct)
        numeric_ratio = _numeric_ratio(comparison)

        numeric_match_count += int(numeric_match)
        unit_match_count += int(unit_match)
        correct_count += int(correct)

        prefix_totals[prefix] += 1
        prefix_correct[prefix] += int(correct)

        results.append(
            {
                "id": example_id,
                "question": question,
                "gold_answer": gold_answer,
                "gold_unit": gold_unit,
                "pred_answer": str(prediction.get("answer", "")),
                "pred_unit": str(prediction.get("unit", "")),
                "explanation": str(prediction.get("explanation", "")),
                "self_check_attempted": bool(prediction.get("self_check_attempted", False)),
                "self_checked": bool(prediction.get("self_checked", False)),
                "self_check_parse_error": bool(prediction.get("self_check_parse_error", False)),
                "exact_match": bool(comparison["exact_match"]),
                "numeric_match": numeric_match,
                "unit_match": unit_match,
                "correct": correct,
                "parse_error": parse_error,
                "error_type": error_type,
                "numeric_ratio": numeric_ratio,
                "pred_answer_num": comparison.get("pred_answer_num"),
                "gold_answer_num": comparison.get("gold_answer_num"),
                "converted_pred_answer_num": comparison.get("converted_pred_answer_num"),
                "converted_gold_answer_num": comparison.get("converted_gold_answer_num"),
                "normalized_pred_unit": comparison.get("normalized_pred_unit"),
                "normalized_gold_unit": comparison.get("normalized_gold_unit"),
            }
        )

        if args.save_every > 0 and len(results) % args.save_every == 0:
            partial_summary = _build_summary(
                results,
                prefix_totals,
                prefix_correct,
                parse_error_count,
                numeric_match_count,
                unit_match_count,
                correct_count,
            )
            _write_outputs(output_dir, results, partial_summary)

    summary = _build_summary(
        results,
        prefix_totals,
        prefix_correct,
        parse_error_count,
        numeric_match_count,
        unit_match_count,
        correct_count,
    )
    _write_outputs(output_dir, results, summary)

    total = summary["total_evaluated"]
    accuracy = summary["accuracy"]
    accuracy_by_prefix = summary["accuracy_by_prefix"]
    print(f"Evaluated: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Parse errors: {parse_error_count}")
    print(f"Numeric matches: {numeric_match_count}")
    print(f"Unit matches: {unit_match_count}")
    print("")
    print("Accuracy by prefix:")
    for prefix, prefix_accuracy in accuracy_by_prefix.items():
        print(f"{prefix}: {prefix_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
