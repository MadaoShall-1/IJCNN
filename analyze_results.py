from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze benchmark result files.")
    parser.add_argument("--results", required=True, help="Path to results.csv")
    parser.add_argument("--top", type=int, default=20, help="Number of failed examples to print")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    dataframe = pd.read_csv(results_path)
    if dataframe.empty:
        print("No rows found.")
        return

    total = len(dataframe)
    correct = int(dataframe["correct"].sum()) if "correct" in dataframe else 0
    parse_errors = int(dataframe["parse_error"].sum()) if "parse_error" in dataframe else 0
    accuracy = correct / total if total else 0.0

    print(f"Results: {results_path}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Parse errors: {parse_errors}")

    if "error_type" in dataframe:
        print("\nError breakdown:")
        print(dataframe["error_type"].value_counts(dropna=False).to_string())

    if "numeric_ratio" in dataframe:
        ratios = dataframe.loc[dataframe["numeric_ratio"].notna() & ~dataframe["correct"], "numeric_ratio"]
        if not ratios.empty:
            print("\nCommon numeric ratios for failed rows:")
            rounded = ratios.astype(float).round(6)
            print(rounded.value_counts().head(10).to_string())

    display_columns = [
        "id",
        "gold_answer",
        "gold_unit",
        "pred_answer",
        "pred_unit",
        "numeric_ratio",
        "error_type",
        "explanation",
    ]
    display_columns = [column for column in display_columns if column in dataframe.columns]
    failed = dataframe.loc[~dataframe["correct"].astype(bool), display_columns].head(args.top)
    if not failed.empty:
        print("\nFailed examples:")
        with pd.option_context("display.max_colwidth", 180):
            print(failed.to_string(index=False))


if __name__ == "__main__":
    main()
