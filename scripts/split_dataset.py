"""Split a CSV dataset into train and test files."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def _default_output_paths(dataset_path: Path) -> tuple[Path, Path]:
    stem = dataset_path.stem
    suffix = dataset_path.suffix or ".csv"
    return (
        dataset_path.with_name(f"{stem}_train{suffix}"),
        dataset_path.with_name(f"{stem}_test{suffix}"),
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_dataset(
    dataset_path: Path,
    train_path: Path,
    test_path: Path,
    train_ratio: float,
    seed: int,
) -> dict[str, object]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"No CSV header found in {dataset_path}")
        rows = list(reader)

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    train_count = int(round(len(shuffled) * train_ratio))
    train_rows = shuffled[:train_count]
    test_rows = shuffled[train_count:]

    _write_csv(train_path, fieldnames, train_rows)
    _write_csv(test_path, fieldnames, test_rows)

    return {
        "dataset": str(dataset_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "total": len(rows),
        "train": len(train_rows),
        "test": len(test_rows),
        "train_ratio": train_ratio,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a CSV dataset into train/test CSV files.")
    parser.add_argument("--dataset", default="Dataset/Physics_Problems_Text_Only.csv")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--test-output", default=None)
    args = parser.parse_args()

    dataset_path = _resolve_path(args.dataset)
    default_train_path, default_test_path = _default_output_paths(dataset_path)
    train_path = _resolve_path(args.train_output) if args.train_output else default_train_path
    test_path = _resolve_path(args.test_output) if args.test_output else default_test_path

    summary = split_dataset(
        dataset_path=dataset_path,
        train_path=train_path,
        test_path=test_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
