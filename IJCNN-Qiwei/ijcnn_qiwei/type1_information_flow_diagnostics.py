#!/usr/bin/env python3
"""Diagnose whether trace/SSM information reaches the retained Type 1 model."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .type1_backtracking_trace_training import (
    BacktrackingCandidate,
    BacktrackingTraceConfig,
    BacktrackingTraceTrainer,
    make_jsonable,
)


PATH_FIELDS = {"input_path", "output_path", "summary_output_path", "model_output_path"}


def _config_from_model(payload: dict[str, Any], input_path: Path, device: str) -> BacktrackingTraceConfig:
    raw = dict(payload.get("config", {}))
    allowed = {item.name for item in fields(BacktrackingTraceConfig)}
    cleaned = {key: value for key, value in raw.items() if key in allowed}
    for key in PATH_FIELDS:
        if key in cleaned:
            cleaned[key] = Path(cleaned[key])
    config = BacktrackingTraceConfig(**cleaned)
    config.input_path = input_path
    config.device = device
    return config


class InformationFlowDiagnostics:
    def __init__(
        self,
        input_path: Path,
        model_path: Path,
        output_path: Path,
        device: str = "auto",
        limit_groups: int | None = None,
    ) -> None:
        self.input_path = input_path
        self.model_path = model_path
        self.output_path = output_path
        self.device_override = device
        self.limit_groups = limit_groups

    def run(self) -> dict[str, Any]:
        started = time.time()
        model_payload = json.loads(self.model_path.read_text(encoding="utf-8"))
        config = _config_from_model(model_payload, self.input_path, self.device_override)
        trainer = BacktrackingTraceTrainer(config)
        records = json.loads(config.input_path.read_text(encoding="utf-8"))
        _train_ids, val_ids = trainer._split_ids(len(records))
        val_groups = trainer._valid_groups(trainer._group_candidates(trainer._collect_candidates(records, val_ids, "diag")))
        if self.limit_groups is not None:
            val_groups = val_groups[: self.limit_groups]

        model = trainer._build_model(
            model_payload["candidate_feature_count"],
            model_payload["trace_feature_count"],
        ).to(trainer.device)
        state_dict = {
            name: torch.tensor(value, dtype=torch.float32, device=trainer.device)
            for name, value in model_payload["state_dict"].items()
        }
        model.load_state_dict(state_dict)
        scaler = model_payload["scaler"]

        normal = self._evaluate_mode(model, trainer, val_groups, scaler, mode="normal")
        mode_results = {"normal": normal}
        for mode in [
            "zero_trace_content",
            "shuffle_trace_between_candidates",
            "zero_candidate_features",
            "disable_causal_bias",
        ]:
            mode_result = self._evaluate_mode(model, trainer, val_groups, scaler, mode=mode)
            mode_result["mean_total_variation_vs_normal"] = round(
                self._mean_total_variation(normal["probabilities"], mode_result["probabilities"]), 6
            )
            mode_results[mode] = mode_result

        gradient = self._gradient_probe(model, trainer, val_groups, scaler)
        summary = {
            "model": str(self.model_path),
            "validation_questions": len(val_groups),
            "normal_accuracy": mode_results["normal"]["accuracy"],
            "zero_trace_accuracy": mode_results["zero_trace_content"]["accuracy"],
            "shuffle_trace_accuracy": mode_results["shuffle_trace_between_candidates"]["accuracy"],
            "zero_candidate_accuracy": mode_results["zero_candidate_features"]["accuracy"],
            "disable_causal_bias_accuracy": mode_results["disable_causal_bias"]["accuracy"],
            "zero_trace_tv_delta": mode_results["zero_trace_content"]["mean_total_variation_vs_normal"],
            "shuffle_trace_tv_delta": mode_results["shuffle_trace_between_candidates"]["mean_total_variation_vs_normal"],
            "zero_candidate_tv_delta": mode_results["zero_candidate_features"]["mean_total_variation_vs_normal"],
            "disable_causal_bias_tv_delta": mode_results["disable_causal_bias"]["mean_total_variation_vs_normal"],
            "candidate_grad_abs_mean": gradient["candidate_grad_abs_mean"],
            "trace_grad_abs_mean": gradient["trace_grad_abs_mean"],
            "trace_to_candidate_grad_ratio": gradient["trace_to_candidate_grad_ratio"],
            "interpretation": self._interpretation(mode_results, gradient),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        payload = {"summary": summary, "modes": mode_results, "gradient_probe": gradient}
        self.output_path.write_text(json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._print(summary)
        return payload

    def _evaluate_mode(
        self,
        model: nn.Module,
        trainer: BacktrackingTraceTrainer,
        groups: list[list[BacktrackingCandidate]],
        scaler: dict[str, list[float]],
        mode: str,
    ) -> dict[str, Any]:
        model.eval()
        correct = 0
        probabilities: list[dict[str, float]] = []
        previous_bias = getattr(model, "bias_strength", None)
        if mode == "disable_causal_bias" and previous_bias is not None:
            model.bias_strength = 0.0
        with torch.no_grad():
            for group in groups:
                tensors = trainer._tensor_batch([group], scaler)
                self._apply_mode(tensors, mode)
                scores, _action_logits, _conflict_logits = model(
                    tensors["candidate_features"],
                    tensors["trace_features"],
                    tensors["candidate_mask"],
                    tensors["trace_mask"],
                )
                valid_scores = scores[0, : len(group)]
                probs = torch.softmax(valid_scores, dim=0).detach().cpu().numpy()
                best_idx = int(np.argmax(probs))
                correct += int(group[best_idx].answer == group[best_idx].expected)
                probabilities.append({candidate.answer: float(prob) for candidate, prob in zip(group, probs)})
        if mode == "disable_causal_bias" and previous_bias is not None:
            model.bias_strength = previous_bias
        return {
            "accuracy": round(correct / max(1, len(groups)), 6),
            "probabilities": probabilities,
        }

    def _apply_mode(self, tensors: dict[str, torch.Tensor], mode: str) -> None:
        if mode == "zero_trace_content":
            tensors["trace_features"] = torch.zeros_like(tensors["trace_features"])
        elif mode == "shuffle_trace_between_candidates":
            tensors["trace_features"] = torch.flip(tensors["trace_features"], dims=[1])
            tensors["trace_mask"] = torch.flip(tensors["trace_mask"], dims=[1])
        elif mode == "zero_candidate_features":
            tensors["candidate_features"] = torch.zeros_like(tensors["candidate_features"])

    def _gradient_probe(
        self,
        model: nn.Module,
        trainer: BacktrackingTraceTrainer,
        groups: list[list[BacktrackingCandidate]],
        scaler: dict[str, list[float]],
    ) -> dict[str, float]:
        model.train()
        batch = groups[: max(1, min(len(groups), trainer.config.batch_size))]
        tensors = trainer._tensor_batch(batch, scaler)
        tensors["candidate_features"] = tensors["candidate_features"].detach().requires_grad_(True)
        tensors["trace_features"] = tensors["trace_features"].detach().requires_grad_(True)
        scores, _action_logits, _conflict_logits = model(
            tensors["candidate_features"],
            tensors["trace_features"],
            tensors["candidate_mask"],
            tensors["trace_mask"],
        )
        loss = nn.functional.cross_entropy(scores, tensors["targets"])
        model.zero_grad(set_to_none=True)
        loss.backward()
        candidate_grad = float(tensors["candidate_features"].grad.abs().mean().detach().cpu())
        trace_grad = float(tensors["trace_features"].grad.abs().mean().detach().cpu())
        model.eval()
        return {
            "candidate_grad_abs_mean": round(candidate_grad, 10),
            "trace_grad_abs_mean": round(trace_grad, 10),
            "trace_to_candidate_grad_ratio": round(trace_grad / max(candidate_grad, 1e-12), 6),
        }

    def _mean_total_variation(self, left: list[dict[str, float]], right: list[dict[str, float]]) -> float:
        distances = []
        for left_probs, right_probs in zip(left, right):
            labels = set(left_probs) | set(right_probs)
            distances.append(0.5 * sum(abs(left_probs.get(label, 0.0) - right_probs.get(label, 0.0)) for label in labels))
        return float(np.mean(distances)) if distances else 0.0

    def _interpretation(self, mode_results: dict[str, dict[str, Any]], gradient: dict[str, float]) -> str:
        normal_acc = mode_results["normal"]["accuracy"]
        zero_trace_drop = normal_acc - mode_results["zero_trace_content"]["accuracy"]
        shuffle_trace_drop = normal_acc - mode_results["shuffle_trace_between_candidates"]["accuracy"]
        bias_drop = normal_acc - mode_results["disable_causal_bias"]["accuracy"]
        trace_ratio = gradient["trace_to_candidate_grad_ratio"]
        weak_trace = zero_trace_drop <= 0.02 and shuffle_trace_drop <= 0.02 and trace_ratio < 0.2
        weak_bias = bias_drop <= 0.02
        if weak_trace and weak_bias:
            return "Trace/SSM information is weakly used; the retained model mostly relies on candidate lexical features."
        if weak_trace:
            return "Trace gradients or ablation impact are weak; SSM is present but not a dominant decision path."
        return "Trace/SSM information affects predictions."

    def _print(self, summary: dict[str, Any]) -> None:
        print("\nType 1 Trace/SSM Information Flow Diagnostics")
        for key in [
            "validation_questions",
            "normal_accuracy",
            "zero_trace_accuracy",
            "shuffle_trace_accuracy",
            "zero_candidate_accuracy",
            "disable_causal_bias_accuracy",
            "zero_trace_tv_delta",
            "shuffle_trace_tv_delta",
            "zero_candidate_tv_delta",
            "disable_causal_bias_tv_delta",
            "candidate_grad_abs_mean",
            "trace_grad_abs_mean",
            "trace_to_candidate_grad_ratio",
            "interpretation",
        ]:
            print(f"- {key}: {summary[key]}")
        print(f"- saved diagnostics: {self.output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose retained Type 1 trace/SSM information flow.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--model", type=Path, default=Path("type1_backtracking_trace_best_model.json"))
    parser.add_argument("--output", type=Path, default=Path("type1_information_flow_diagnostics.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-groups", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    InformationFlowDiagnostics(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        device=args.device,
        limit_groups=args.limit_groups,
    ).run()


if __name__ == "__main__":
    main()
