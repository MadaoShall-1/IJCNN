#!/usr/bin/env python3
"""Train and validate the local transformer brain evaluator.

This trains only the evaluator/readout over the retained architecture's
imagined candidate states. It does not add a separate random forest or external
fallback model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_hybrid_parser import SemanticHybridConfig, normalize_for_eval
from .type1_pipeline import Type1MultiStagePipeline, Type1PipelineConfig
from .type1_preprocessing import TextTools


@dataclass
class BrainTrainingConfig:
    input_path: Path = Path("../Logic_Based_Educational_Queries.json")
    output_path: Path = Path("type1_brain_train_eval_results.json")
    summary_output_path: Path = Path("type1_brain_train_eval_summary.json")
    model_output_path: Path = Path("type1_trained_brain_readout.json")
    train_ratio: float = 0.8
    random_state: int = 42
    epochs: int = 80
    learning_rate: float = 0.035
    l2: float = 0.0005
    pairwise_margin: float = 0.22
    pairwise_weight: float = 0.75
    hard_negative_weight: float = 0.55
    max_hard_negatives: int = 2
    class_balance: bool = False
    limit_records: int | None = None
    local_files_only: bool = False
    segmenter_model: str = "openai/Qwen2.5-1.5B-Instruct"
    segmenter_api_base: str = "http://localhost:8001/v1"
    api_key: str = "EMPTY"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    top_k: int = 8


@dataclass
class BrainCandidate:
    key: str
    record_id: int
    question_id: int
    answer: str
    expected: str
    features: list[float]


@dataclass
class BrainPrediction:
    record_id: int
    question_id: int
    expected: str
    predicted_answer: str
    correct: bool
    candidate_probabilities: dict[str, float]


class BrainReadoutTrainer:
    def __init__(self, config: BrainTrainingConfig) -> None:
        self.config = config
        stage0_config = SemanticHybridConfig(
            segmenter_model=config.segmenter_model,
            segmenter_api_base=config.segmenter_api_base,
            segmenter_api_key=config.api_key,
            embedding_model=config.embedding_model,
            top_k=config.top_k,
            local_files_only=config.local_files_only,
        )
        pipeline_config = Type1PipelineConfig(
            stage0=stage0_config,
            transformer_brain_allow_override=False,
        )
        self.pipeline = Type1MultiStagePipeline(pipeline_config)

    def run(self) -> dict[str, Any]:
        started = time.time()
        records = json.loads(self.config.input_path.read_text(encoding="utf-8"))
        if self.config.limit_records is not None:
            records = records[: self.config.limit_records]
        train_ids, val_ids = self._split_ids(len(records))

        train_candidates = self._collect_candidates(records, train_ids, "train")
        val_candidates = self._collect_candidates(records, val_ids, "val")
        train_groups = self._group_candidates(train_candidates)
        val_groups = self._group_candidates(val_candidates)

        train_groups = [group for group in train_groups if self._target_index(group) is not None]
        val_groups = [group for group in val_groups if self._target_index(group) is not None]
        if not train_groups:
            raise RuntimeError("No trainable questions found. Check candidate extraction and labels.")

        scaler = self._fit_scaler([candidate for group in train_groups for candidate in group])
        weights, history = self._train(train_groups, scaler)
        predictions = self._predict_groups(val_groups, scaler, weights)
        summary = self._summary(train_groups, val_groups, predictions, history, started)
        model_payload = {
            "architecture": "trained_readout_on_local_multihead_blockwise_ssm_transformer_world_model",
            "feature_count": len(weights) - 1,
            "weights": [round(float(value), 10) for value in weights.tolist()],
            "scaler_mean": [round(float(value), 10) for value in scaler["mean"].tolist()],
            "scaler_std": [round(float(value), 10) for value in scaler["std"].tolist()],
            "config": asdict(self.config),
        }
        payload = {
            "summary": summary,
            "predictions": [asdict(item) for item in predictions],
            "model": model_payload,
        }
        self._save_json(self.config.model_output_path, model_payload)
        self._save_json(self.config.output_path, payload)
        self._save_json(self.config.summary_output_path, summary)
        self._print_summary(summary)
        return payload

    def _collect_candidates(
        self,
        records: list[dict[str, Any]],
        record_ids: set[int],
        split_name: str,
    ) -> list[BrainCandidate]:
        candidates: list[BrainCandidate] = []
        processed = 0
        for record_id in sorted(record_ids):
            record = records[record_id]
            for question_id in range(len(record.get("questions", []) or [])):
                payload = dict(record)
                payload["_question_idx"] = question_id
                payload["record_id"] = record_id
                payload["question_id"] = question_id
                expected = normalize_for_eval(TextTools.safe_get(record.get("answers", []), question_id))
                result = self.pipeline.run(payload)
                brain = result.get("transformer_brain_world_model") or {}
                extracted = self._candidates_from_brain(
                    key=f"{record_id}:{question_id}",
                    record_id=record_id,
                    question_id=question_id,
                    expected=expected,
                    brain=brain,
                )
                candidates.extend(extracted)
                processed += 1
                print(
                    f"[{split_name} {processed}] r={record_id} q={question_id} "
                    f"expected={expected} candidates={len(extracted)}"
                )
        return candidates

    def _candidates_from_brain(
        self,
        *,
        key: str,
        record_id: int,
        question_id: int,
        expected: str,
        brain: dict[str, Any],
    ) -> list[BrainCandidate]:
        states = brain.get("external_imagination_states") or []
        latents = brain.get("candidate_latents") or {}
        logits = brain.get("evaluator_logits") or {}
        distribution = brain.get("answer_distribution") or {}
        confidence = self._float(brain.get("confidence"))
        margin = self._float(brain.get("margin"))
        output: list[BrainCandidate] = []
        for state in states:
            answer = normalize_for_eval(state.get("answer", ""))
            if not answer:
                continue
            base_features = [self._float(value) for value in state.get("feature_vector", [])]
            latent = [self._float(value) for value in latents.get(answer, [])]
            features = [
                self._float(state.get("prior_logit")),
                self._float(logits.get(answer)),
                self._float(distribution.get(answer)),
                confidence,
                margin,
                *base_features,
                *latent,
            ]
            output.append(
                BrainCandidate(
                    key=key,
                    record_id=record_id,
                    question_id=question_id,
                    answer=answer,
                    expected=expected,
                    features=features,
                )
            )
        return output

    def _train(
        self,
        groups: list[list[BrainCandidate]],
        scaler: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        feature_dim = len(groups[0][0].features)
        weights = np.zeros((feature_dim + 1,), dtype=np.float64)
        rng = random.Random(self.config.random_state)
        label_weights = self._label_weights(groups)
        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.epochs + 1):
            rng.shuffle(groups)
            total_loss = 0.0
            correct = 0
            count = 0
            for group in groups:
                target = self._target_index(group)
                if target is None:
                    continue
                x = self._matrix(group, scaler)
                scores = x @ weights
                probs = self._softmax(scores)
                target_weight = label_weights.get(group[target].expected, 1.0)
                loss = target_weight * -math.log(max(float(probs[target]), 1e-12))
                grad_scores = target_weight * probs
                grad_scores[target] -= target_weight
                pairwise_loss, pairwise_grad = self._pairwise_loss_and_grad(scores, target)
                loss += pairwise_loss
                grad_scores += pairwise_grad
                loss += self.config.l2 * float(weights[1:] @ weights[1:])
                grad = x.T @ grad_scores + np.r_[0.0, 2.0 * self.config.l2 * weights[1:]]
                weights -= self.config.learning_rate * grad
                total_loss += loss
                correct += int(int(np.argmax(scores)) == target)
                count += 1
            if epoch == 1 or epoch % 10 == 0 or epoch == self.config.epochs:
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": round(total_loss / max(1, count), 6),
                        "train_accuracy": round(correct / max(1, count), 6),
                    }
                )
        return weights, history

    def _pairwise_loss_and_grad(self, scores: np.ndarray, target: int) -> tuple[float, np.ndarray]:
        wrong_indices = [idx for idx in range(scores.shape[0]) if idx != target]
        if not wrong_indices:
            return 0.0, np.zeros_like(scores)
        wrong_indices.sort(key=lambda idx: float(scores[idx]), reverse=True)
        selected = wrong_indices[: max(1, self.config.max_hard_negatives)]
        grad = np.zeros_like(scores)
        total_loss = 0.0
        normalizer = float(len(selected))
        for rank, wrong_idx in enumerate(selected):
            weight = self.config.pairwise_weight
            if rank == 0:
                weight += self.config.hard_negative_weight
            weight /= normalizer
            gap = float(scores[target] - scores[wrong_idx])
            z = self.config.pairwise_margin - gap
            clipped_z = float(np.clip(z, -40.0, 40.0))
            sigmoid = 1.0 / (1.0 + math.exp(-clipped_z))
            total_loss += weight * math.log1p(math.exp(clipped_z))
            grad[target] -= weight * sigmoid
            grad[wrong_idx] += weight * sigmoid
        return total_loss, grad

    def _predict_groups(
        self,
        groups: list[list[BrainCandidate]],
        scaler: dict[str, np.ndarray],
        weights: np.ndarray,
    ) -> list[BrainPrediction]:
        predictions: list[BrainPrediction] = []
        for group in groups:
            x = self._matrix(group, scaler)
            scores = x @ weights
            probs = self._softmax(scores)
            best_idx = int(np.argmax(probs))
            best = group[best_idx]
            predictions.append(
                BrainPrediction(
                    record_id=best.record_id,
                    question_id=best.question_id,
                    expected=best.expected,
                    predicted_answer=best.answer,
                    correct=best.answer == best.expected,
                    candidate_probabilities={
                        item.answer: round(float(prob), 6)
                        for item, prob in sorted(zip(group, probs.tolist()), key=lambda pair: pair[1], reverse=True)
                    },
                )
            )
        return predictions

    def _summary(
        self,
        train_groups: list[list[BrainCandidate]],
        val_groups: list[list[BrainCandidate]],
        predictions: list[BrainPrediction],
        history: list[dict[str, float]],
        started: float,
    ) -> dict[str, Any]:
        total = len(predictions)
        correct = sum(item.correct for item in predictions)
        return {
            "train_ratio": self.config.train_ratio,
            "random_state": self.config.random_state,
            "train_questions": len(train_groups),
            "validation_questions": len(val_groups),
            "accuracy": round(correct / total, 6) if total else 0.0,
            "history": history,
            "elapsed_seconds": round(time.time() - started, 3),
        }

    def _split_ids(self, record_count: int) -> tuple[set[int], set[int]]:
        ids = list(range(record_count))
        rng = random.Random(self.config.random_state)
        rng.shuffle(ids)
        train_count = int(round(record_count * self.config.train_ratio))
        train_count = max(1, min(record_count - 1, train_count)) if record_count > 1 else record_count
        return set(ids[:train_count]), set(ids[train_count:])

    def _group_candidates(self, candidates: list[BrainCandidate]) -> list[list[BrainCandidate]]:
        grouped: dict[str, list[BrainCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.key, []).append(candidate)
        return list(grouped.values())

    def _target_index(self, group: list[BrainCandidate]) -> int | None:
        for idx, candidate in enumerate(group):
            if candidate.answer == candidate.expected:
                return idx
        return None

    def _label_weights(self, groups: list[list[BrainCandidate]]) -> dict[str, float]:
        if not self.config.class_balance:
            return {}
        counts: dict[str, int] = {}
        for group in groups:
            target = self._target_index(group)
            if target is not None:
                counts[group[target].expected] = counts.get(group[target].expected, 0) + 1
        if not counts:
            return {}
        total = sum(counts.values())
        class_count = len(counts)
        return {
            label: float(np.clip(total / max(1, class_count * count), 0.45, 2.8))
            for label, count in counts.items()
        }

    def _fit_scaler(self, candidates: list[BrainCandidate]) -> dict[str, np.ndarray]:
        matrix = np.array([item.features for item in candidates], dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return {"mean": mean, "std": std}

    def _matrix(self, group: list[BrainCandidate], scaler: dict[str, np.ndarray]) -> np.ndarray:
        matrix = np.array([item.features for item in group], dtype=np.float64)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = (matrix - scaler["mean"]) / scaler["std"]
        return np.c_[np.ones((matrix.shape[0],)), matrix]

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        exp_values = np.exp(np.clip(shifted, -40.0, 40.0))
        return exp_values / max(float(exp_values.sum()), 1e-12)

    def _float(self, value: Any) -> float:
        try:
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return 0.0
            return number
        except (TypeError, ValueError):
            return 0.0

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print("\nType 1 Trained Transformer Brain Summary")
        print(f"- train_questions: {summary['train_questions']}")
        print(f"- validation_questions: {summary['validation_questions']}")
        print(f"- accuracy: {summary['accuracy']}")
        print(f"- saved model: {self.config.model_output_path}")
        print(f"- saved results: {self.config.output_path}")
        print(f"- saved summary: {self.config.summary_output_path}")

    @staticmethod
    def _save_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the retained Type 1 transformer brain readout with an 8/2 split.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--output", type=Path, default=Path("type1_brain_train_eval_results.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("type1_brain_train_eval_summary.json"))
    parser.add_argument("--model-output", type=Path, default=Path("type1_trained_brain_readout.json"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--pairwise-margin", type=float, default=0.22)
    parser.add_argument("--pairwise-weight", type=float, default=0.75)
    parser.add_argument("--hard-negative-weight", type=float, default=0.55)
    parser.add_argument("--max-hard-negatives", type=int, default=2)
    parser.add_argument("--enable-class-balance", dest="class_balance", action="store_true", default=False)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--segmenter-model", default="openai/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--segmenter-api-base", default="http://localhost:8001/v1")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BrainTrainingConfig:
    return BrainTrainingConfig(
        input_path=args.input,
        output_path=args.output,
        summary_output_path=args.summary_output,
        model_output_path=args.model_output,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        pairwise_margin=args.pairwise_margin,
        pairwise_weight=args.pairwise_weight,
        hard_negative_weight=args.hard_negative_weight,
        max_hard_negatives=args.max_hard_negatives,
        class_balance=args.class_balance,
        limit_records=args.limit_records,
        local_files_only=args.local_files_only,
        api_key=args.api_key,
        segmenter_model=args.segmenter_model,
        segmenter_api_base=args.segmenter_api_base,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )


def main() -> None:
    BrainReadoutTrainer(config_from_args(parse_args())).run()


if __name__ == "__main__":
    main()
