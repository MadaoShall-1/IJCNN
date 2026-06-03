#!/usr/bin/env python3
"""Train a candidate-level multi-head Transformer evaluator for Type 1."""

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
import torch
from torch import nn

from .type1_brain_training import BrainCandidate, BrainPrediction, BrainReadoutTrainer, BrainTrainingConfig, make_jsonable


@dataclass
class CandidateTransformerConfig(BrainTrainingConfig):
    transformer_hidden_dim: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_dim: int = 128
    dropout: float = 0.1
    batch_size: int = 32
    patience: int = 18
    min_delta: float = 0.0005
    device: str = "auto"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.value = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = nn.functional.silu(self.gate(x)) * self.value(x)
        return self.out(self.dropout(hidden))


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("transformer_hidden_dim must be divisible by transformer_heads for RoPE attention.")
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even per-head dimension.")
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self._apply_rope(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(context)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        half_dim = self.head_dim // 2
        device = q.device
        dtype = q.dtype
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device, dtype=dtype) / half_dim))
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        angles = torch.outer(positions, inv_freq)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]
        q_rotated = torch.stack((q_even * cos - q_odd * sin, q_even * sin + q_odd * cos), dim=-1).flatten(-2)
        k_rotated = torch.stack((k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1).flatten(-2)
        return q_rotated, k_rotated


class CandidateTransformerBlock(nn.Module):
    def __init__(self, config: CandidateTransformerConfig) -> None:
        super().__init__()
        dim = config.transformer_hidden_dim
        self.attn_norm = RMSNorm(dim)
        self.attention = RoPEMultiHeadAttention(dim, config.transformer_heads, config.dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, config.transformer_ff_dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.attn_norm(x), mask))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class CandidateTransformer(nn.Module):
    def __init__(self, input_dim: int, config: CandidateTransformerConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(input_dim)
        self.projection = nn.Linear(input_dim, config.transformer_hidden_dim)
        self.blocks = nn.ModuleList(
            CandidateTransformerBlock(config)
            for _ in range(config.transformer_layers)
        )
        self.scorer = nn.Sequential(
            RMSNorm(config.transformer_hidden_dim),
            nn.Linear(config.transformer_hidden_dim, config.transformer_hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transformer_hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(self.input_norm(features))
        for block in self.blocks:
            hidden = block(hidden, mask)
        scores = self.scorer(hidden).squeeze(-1)
        return scores.masked_fill(~mask, -1e9)


class CandidateTransformerTrainer(BrainReadoutTrainer):
    def __init__(self, config: CandidateTransformerConfig) -> None:
        super().__init__(config)
        self.config: CandidateTransformerConfig = config
        self.device = self._device()

    def run(self) -> dict[str, Any]:
        self._set_seed()
        started = time.time()
        records = json.loads(self.config.input_path.read_text(encoding="utf-8"))
        if self.config.limit_records is not None:
            records = records[: self.config.limit_records]
        train_ids, val_ids = self._split_ids(len(records))

        train_candidates = self._collect_candidates(records, train_ids, "train")
        val_candidates = self._collect_candidates(records, val_ids, "val")
        train_groups = self._valid_groups(self._group_candidates(train_candidates))
        val_groups = self._valid_groups(self._group_candidates(val_candidates))
        if not train_groups:
            raise RuntimeError("No trainable questions found. Check candidate extraction and labels.")

        scaler = self._fit_scaler([candidate for group in train_groups for candidate in group])
        feature_dim = len(train_groups[0][0].features)
        model = CandidateTransformer(feature_dim, self.config).to(self.device)
        history, best_state = self._train_transformer(model, train_groups, val_groups, scaler)
        model.load_state_dict(best_state)
        predictions = self._predict_transformer(model, val_groups, scaler)
        summary = self._summary(train_groups, val_groups, predictions, history, started)
        model_payload = {
            "architecture": "trainable_candidate_multihead_transformer_world_model",
            "feature_count": feature_dim,
            "state_dict": {
                name: tensor.detach().cpu().tolist()
                for name, tensor in model.state_dict().items()
            },
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

    def _valid_groups(self, groups: list[list[BrainCandidate]]) -> list[list[BrainCandidate]]:
        return [group for group in groups if self._target_index(group) is not None]

    def _train_transformer(
        self,
        model: CandidateTransformer,
        train_groups: list[list[BrainCandidate]],
        val_groups: list[list[BrainCandidate]],
        scaler: dict[str, np.ndarray],
    ) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.l2)
        rng = random.Random(self.config.random_state)
        history: list[dict[str, float]] = []
        best_accuracy = -1.0
        best_loss = float("inf")
        best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        stale = 0
        for epoch in range(1, self.config.epochs + 1):
            model.train()
            shuffled = list(train_groups)
            rng.shuffle(shuffled)
            total_loss = 0.0
            correct = 0
            count = 0
            for batch in self._batches(shuffled, self.config.batch_size):
                features, mask, targets = self._tensor_batch(batch, scaler)
                optimizer.zero_grad(set_to_none=True)
                scores = model(features, mask)
                loss = self._loss(scores, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(batch)
                correct += int((torch.argmax(scores, dim=1) == targets).sum().detach().cpu())
                count += len(batch)
            val_loss, val_accuracy = self._evaluate_transformer(model, val_groups, scaler)
            train_accuracy = correct / max(1, count)
            train_loss = total_loss / max(1, count)
            if epoch == 1 or epoch % 10 == 0 or epoch == self.config.epochs:
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": round(train_loss, 6),
                        "train_accuracy": round(train_accuracy, 6),
                        "validation_loss": round(val_loss, 6),
                        "validation_accuracy": round(val_accuracy, 6),
                    }
                )
            improved = val_accuracy > best_accuracy + self.config.min_delta or (
                abs(val_accuracy - best_accuracy) <= self.config.min_delta and val_loss < best_loss
            )
            if improved:
                best_accuracy = val_accuracy
                best_loss = val_loss
                stale = 0
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.config.patience:
                    break
        return history, best_state

    def _loss(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(scores, targets)
        pairwise_terms: list[torch.Tensor] = []
        for row_idx in range(scores.shape[0]):
            target = int(targets[row_idx].detach().cpu())
            valid = scores[row_idx] > -1e8
            wrong = torch.where(valid)[0]
            wrong = wrong[wrong != target]
            if wrong.numel() == 0:
                continue
            wrong_scores = scores[row_idx, wrong]
            top_k = min(max(1, self.config.max_hard_negatives), wrong_scores.numel())
            hard_values, hard_indices = torch.topk(wrong_scores, k=top_k)
            selected = wrong[hard_indices]
            target_score = scores[row_idx, target]
            margins = self.config.pairwise_margin - (target_score - scores[row_idx, selected])
            weights = torch.full_like(margins, self.config.pairwise_weight / float(top_k))
            weights[0] += self.config.hard_negative_weight / float(top_k)
            pairwise_terms.append((weights * nn.functional.softplus(margins)).mean())
        if not pairwise_terms:
            return ce
        return ce + torch.stack(pairwise_terms).mean()

    def _evaluate_transformer(
        self,
        model: CandidateTransformer,
        groups: list[list[BrainCandidate]],
        scaler: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        count = 0
        with torch.no_grad():
            for batch in self._batches(groups, self.config.batch_size):
                features, mask, targets = self._tensor_batch(batch, scaler)
                scores = model(features, mask)
                loss = self._loss(scores, targets)
                total_loss += float(loss.detach().cpu()) * len(batch)
                correct += int((torch.argmax(scores, dim=1) == targets).sum().detach().cpu())
                count += len(batch)
        return total_loss / max(1, count), correct / max(1, count)

    def _predict_transformer(
        self,
        model: CandidateTransformer,
        groups: list[list[BrainCandidate]],
        scaler: dict[str, np.ndarray],
    ) -> list[BrainPrediction]:
        model.eval()
        predictions: list[BrainPrediction] = []
        with torch.no_grad():
            for group in groups:
                features, mask, _targets = self._tensor_batch([group], scaler)
                scores = model(features, mask)[0]
                valid_scores = scores[: len(group)]
                probs = torch.softmax(valid_scores, dim=0).detach().cpu().numpy().tolist()
                best_idx = int(torch.argmax(valid_scores).detach().cpu())
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
                            for item, prob in sorted(zip(group, probs), key=lambda pair: pair[1], reverse=True)
                        },
                    )
                )
        return predictions

    def _tensor_batch(
        self,
        groups: list[list[BrainCandidate]],
        scaler: dict[str, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_candidates = max(len(group) for group in groups)
        feature_dim = len(groups[0][0].features)
        features = np.zeros((len(groups), max_candidates, feature_dim), dtype=np.float32)
        mask = np.zeros((len(groups), max_candidates), dtype=bool)
        targets = np.zeros((len(groups),), dtype=np.int64)
        for row, group in enumerate(groups):
            matrix = np.array([item.features for item in group], dtype=np.float64)
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            matrix = (matrix - scaler["mean"]) / scaler["std"]
            features[row, : len(group), :] = matrix.astype(np.float32)
            mask[row, : len(group)] = True
            target = self._target_index(group)
            targets[row] = 0 if target is None else target
        return (
            torch.from_numpy(features).to(self.device),
            torch.from_numpy(mask).to(self.device),
            torch.from_numpy(targets).to(self.device),
        )

    def _batches(self, groups: list[list[BrainCandidate]], batch_size: int) -> list[list[list[BrainCandidate]]]:
        return [groups[idx : idx + batch_size] for idx in range(0, len(groups), batch_size)]

    def _device(self) -> torch.device:
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _set_seed(self) -> None:
        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print("\nType 1 Trainable Candidate Transformer Summary")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Type 1 candidate-level multi-head Transformer evaluator.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--output", type=Path, default=Path("type1_candidate_transformer_eval_results.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("type1_candidate_transformer_eval_summary.json"))
    parser.add_argument("--model-output", type=Path, default=Path("type1_candidate_transformer_model.json"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--pairwise-margin", type=float, default=0.22)
    parser.add_argument("--pairwise-weight", type=float, default=0.75)
    parser.add_argument("--hard-negative-weight", type=float, default=0.55)
    parser.add_argument("--max-hard-negatives", type=int, default=2)
    parser.add_argument("--enable-class-balance", dest="class_balance", action="store_true", default=False)
    parser.add_argument("--transformer-hidden-dim", type=int, default=64)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--segmenter-model", default="openai/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--segmenter-api-base", default="http://localhost:8001/v1")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> CandidateTransformerConfig:
    return CandidateTransformerConfig(
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
        transformer_hidden_dim=args.transformer_hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff_dim=args.transformer_ff_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        device=args.device,
        limit_records=args.limit_records,
        local_files_only=args.local_files_only,
        api_key=args.api_key,
        segmenter_model=args.segmenter_model,
        segmenter_api_base=args.segmenter_api_base,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )


def main() -> None:
    CandidateTransformerTrainer(config_from_args(parse_args())).run()


if __name__ == "__main__":
    main()
