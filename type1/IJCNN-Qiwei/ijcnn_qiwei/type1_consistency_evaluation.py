#!/usr/bin/env python3
"""Evaluate explanation consistency for the retained Type 1 model."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .type1_backtracking_trace_training import (
    BacktrackingCandidate,
    BacktrackingTraceConfig,
    BacktrackingTraceTrainer,
    make_jsonable,
)


@dataclass
class ConsistencyConfig:
    input_path: Path = Path("../Logic_Based_Educational_Queries.json")
    retained_results_path: Path = Path("type1_backtracking_trace_best_eval_results.json")
    output_path: Path = Path("type1_consistency_eval_results.json")
    summary_output_path: Path = Path("type1_consistency_eval_summary.json")
    train_ratio: float = 0.8
    random_state: int = 42
    batch_size: int = 32
    max_trace_steps: int = 18
    propagation_top_k: int = 4
    grouped_batches: bool = True
    device: str = "auto"


class ConsistencyEvaluator:
    def __init__(self, config: ConsistencyConfig) -> None:
        self.config = config
        self.device = self._device()

    def run(self) -> dict[str, Any]:
        started = time.time()
        self._set_seed()
        train_groups, val_groups = self._load_groups()
        retained_predictions = self._load_retained_predictions()
        explanation_distributions = {
            group[0].key: self._trace_explanation_distribution(group)
            for group in val_groups
        }

        retained_metrics, retained_rows = self._metrics(retained_predictions, explanation_distributions, val_groups)
        summary = {
            "metric_definition": "consistency = 1 - total_variation_distance(answer_distribution, trace_explanation_distribution)",
            "train_questions": len(train_groups),
            "validation_questions": len(val_groups),
            "train_question_group_counts": self._question_group_counts(train_groups),
            "validation_question_group_counts": self._question_group_counts(val_groups),
            "retained_model": retained_metrics,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        payload = {
            "summary": summary,
            "retained_model_rows": retained_rows,
        }
        self._save_json(self.config.output_path, payload)
        self._save_json(self.config.summary_output_path, summary)
        self._print_summary(summary)
        return payload

    def _load_groups(self) -> tuple[list[list[BacktrackingCandidate]], list[list[BacktrackingCandidate]]]:
        trace_config = BacktrackingTraceConfig(
            input_path=self.config.input_path,
            train_ratio=self.config.train_ratio,
            random_state=self.config.random_state,
            batch_size=self.config.batch_size,
            max_trace_steps=self.config.max_trace_steps,
            propagation_top_k=self.config.propagation_top_k,
            device=self.config.device,
        )
        trainer = BacktrackingTraceTrainer(trace_config)
        records = json.loads(self.config.input_path.read_text(encoding="utf-8"))
        train_ids, val_ids = trainer._split_ids(len(records))
        train_groups = trainer._valid_groups(trainer._group_candidates(trainer._collect_candidates(records, train_ids, "consistency-train")))
        val_groups = trainer._valid_groups(trainer._group_candidates(trainer._collect_candidates(records, val_ids, "consistency-val")))
        return train_groups, val_groups

    def _load_retained_predictions(self) -> dict[str, dict[str, Any]]:
        payload = json.loads(self.config.retained_results_path.read_text(encoding="utf-8"))
        predictions = {}
        for item in payload.get("predictions", []):
            key = f"{item['record_id']}:{item['question_id']}"
            predictions[key] = {
                "expected": item["expected"],
                "question_group": item.get("question_group", ""),
                "predicted_answer": item["predicted_answer"],
                "correct": bool(item["correct"]),
                "candidate_probabilities": item["candidate_probabilities"],
            }
        return predictions

    def _trace_explanation_distribution(self, group: list[BacktrackingCandidate]) -> dict[str, float]:
        scores = []
        labels = []
        for candidate in group:
            labels.append(candidate.answer)
            scores.append(self._trace_explanation_score(candidate))
        probs = self._softmax(np.array(scores, dtype=np.float64))
        return {label: round(float(prob), 6) for label, prob in zip(labels, probs)}

    def _trace_explanation_score(self, candidate: BacktrackingCandidate) -> float:
        matrix = np.array(candidate.trace_features, dtype=np.float64)
        support = float(np.max(matrix[:, 2])) if matrix.size else 0.0
        mean_support = float(np.mean(matrix[:, 2])) if matrix.size else 0.0
        conflict = float(np.max(matrix[:, 3])) if matrix.size else 0.0
        backtrack = float(np.max(matrix[:, 9])) if matrix.size else 0.0
        rule_strength = float(np.max(matrix[:, 4])) if matrix.size else 0.0
        final_conflict = float(matrix[-1, 3]) if matrix.size else 0.0
        return 2.2 * support + 0.7 * mean_support + 0.25 * rule_strength - 1.2 * conflict - 0.35 * final_conflict - 0.25 * backtrack

    def _metrics(
        self,
        predictions: dict[str, dict[str, Any]],
        explanation_distributions: dict[str, dict[str, float]],
        groups: list[list[BacktrackingCandidate]],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        rows = []
        consistency_values = []
        top_agreements = []
        explanation_top_probs = []
        correct = 0
        for group in groups:
            key = group[0].key
            pred = predictions[key]
            model_dist = pred["candidate_probabilities"]
            explanation_dist = explanation_distributions[key]
            consistency = self._distribution_consistency(model_dist, explanation_dist)
            explanation_top = max(explanation_dist, key=explanation_dist.get)
            model_top = max(model_dist, key=model_dist.get)
            top_agreement = float(explanation_top == model_top)
            consistency_values.append(consistency)
            top_agreements.append(top_agreement)
            explanation_top_probs.append(float(model_dist.get(explanation_top, 0.0)))
            correct += int(pred["correct"])
            rows.append(
                {
                    "record_id": group[0].record_id,
                    "question_id": group[0].question_id,
                    "question_group": group[0].question_group,
                    "expected": pred["expected"],
                    "predicted_answer": pred["predicted_answer"],
                    "correct": pred["correct"],
                    "consistency": round(consistency, 6),
                    "model_top": model_top,
                    "explanation_top": explanation_top,
                    "top_explanation_agreement": bool(top_agreement),
                    "model_distribution": model_dist,
                    "trace_explanation_distribution": explanation_dist,
                }
            )
        metrics = {
            "accuracy": round(correct / max(1, len(groups)), 6),
            "consistency": round(float(np.mean(consistency_values)), 6),
            "top_explanation_agreement": round(float(np.mean(top_agreements)), 6),
            "mean_explanation_top_probability": round(float(np.mean(explanation_top_probs)), 6),
            "accuracy_by_group": self._accuracy_by_group(rows),
        }
        return metrics, rows

    def _distribution_consistency(self, left: dict[str, float], right: dict[str, float]) -> float:
        labels = sorted(set(left) | set(right))
        tvd = 0.5 * sum(abs(float(left.get(label, 0.0)) - float(right.get(label, 0.0))) for label in labels)
        return max(0.0, min(1.0, 1.0 - tvd))

    def _question_group_counts(self, groups: list[list[BacktrackingCandidate]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for group in groups:
            counts[group[0].question_group] = counts.get(group[0].question_group, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[0]))

    def _accuracy_by_group(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["question_group"]), []).append(row)
        return {
            key: {
                "questions": len(items),
                "accuracy": round(sum(int(bool(item["correct"])) for item in items) / max(1, len(items)), 6),
            }
            for key, items in sorted(grouped.items(), key=lambda item: item[0])
        }

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        values = values - np.max(values)
        exp = np.exp(values)
        return exp / max(float(exp.sum()), 1e-12)

    def _device(self) -> torch.device:
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _set_seed(self) -> None:
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print("\nType 1 Explanation Consistency Summary")
        print(f"- retained accuracy: {summary['retained_model']['accuracy']}")
        print(f"- retained consistency: {summary['retained_model']['consistency']}")
        print(f"- saved results: {self.config.output_path}")
        print(f"- saved summary: {self.config.summary_output_path}")

    @staticmethod
    def _save_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Type 1 explanation consistency.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--retained-results", type=Path, default=Path("type1_backtracking_trace_best_eval_results.json"))
    parser.add_argument("--output", type=Path, default=Path("type1_consistency_eval_results.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("type1_consistency_eval_summary.json"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-trace-steps", type=int, default=18)
    parser.add_argument("--propagation-top-k", type=int, default=4)
    parser.add_argument("--ungrouped-batches", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ConsistencyConfig:
    return ConsistencyConfig(
        input_path=args.input,
        retained_results_path=args.retained_results,
        output_path=args.output,
        summary_output_path=args.summary_output,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
        batch_size=args.batch_size,
        max_trace_steps=args.max_trace_steps,
        propagation_top_k=args.propagation_top_k,
        grouped_batches=not args.ungrouped_batches,
        device=args.device,
    )


def main() -> None:
    ConsistencyEvaluator(config_from_args(parse_args())).run()


if __name__ == "__main__":
    main()
