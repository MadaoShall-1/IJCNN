#!/usr/bin/env python3
"""Inference wrapper for the retained Type1 backtracking-trace model."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .type1_backtracking_trace_training import (
    BacktrackingCandidate,
    BacktrackingTraceGenerator,
    TextTools,
    TraceSSMOnlyReasoner,
    normalize_for_eval,
)
from .type1_modal_abductive_training import (
    Type1ModalAbductiveConfig,
    Type1ModalAbductiveTrainer,
    Type1SmallWorldAnalyzer,
)


class Type1RetainedPredictor:
    """Load ``type1_backtracking_trace_best_model.json`` and predict one query."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        self.model_path = Path(
            model_path
            or os.getenv("TYPE1_RETAINED_MODEL")
            or root / "type1_backtracking_trace_best_model.json"
        )
        payload = json.loads(self.model_path.read_text(encoding="utf-8"))
        config_data = dict(payload.get("config") or {})
        config_data["device"] = device or os.getenv("TYPE1_DEVICE") or "auto"
        config_data["llm_fallback"] = False
        config_data["rag_backend"] = os.getenv("TYPE1_RAG_BACKEND") or payload.get("rag_backend") or config_data.get("rag_backend", "auto")
        config_data["bge_local_files_only"] = os.getenv("TYPE1_ALLOW_BGE_DOWNLOAD", "0") != "1"
        config = Type1ModalAbductiveConfig(
            **{
                key: value
                for key, value in config_data.items()
                if key in Type1ModalAbductiveConfig.__dataclass_fields__
            }
        )
        self.config = config
        self.trainer = Type1ModalAbductiveTrainer(config)
        self.scaler = payload["scaler"]
        self.model = TraceSSMOnlyReasoner(
            int(payload["candidate_feature_count"]),
            int(payload["trace_feature_count"]),
            config,
        ).to(self.trainer.device)
        state_dict = {
            name: torch.tensor(value, dtype=torch.float32, device=self.trainer.device)
            for name, value in payload["state_dict"].items()
        }
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(
        self,
        *,
        query: str,
        premises: list[str],
        options: list[str] | None = None,
    ) -> dict[str, Any]:
        raw_question = self._question_with_options(query, options or [])
        group = self._build_group(raw_question, premises)
        with torch.no_grad():
            tensors = self.trainer._tensor_batch([group], self.scaler)
            scores, _action_logits, _conflict_logits = self.model(
                tensors["candidate_features"],
                tensors["trace_features"],
                tensors["candidate_mask"],
                tensors["trace_mask"],
            )
            valid_scores = scores[0, : len(group)]
            probs = torch.softmax(valid_scores, dim=0).detach().cpu().tolist()
            best_idx = int(torch.argmax(valid_scores).detach().cpu())
        best = group[best_idx]
        probability_map = {
            item.answer: round(float(prob), 6)
            for item, prob in sorted(zip(group, probs), key=lambda pair: pair[1], reverse=True)
        }
        answer = self._answer_for_api(best.answer, options or [])
        used = self._select_relevant_premises(query, premises)
        return {
            "answer": answer,
            "unit": "",
            "explanation": (
                f"Retained Type1 trace model selected {answer} "
                f"with top probability {float(probs[best_idx]):.4f}."
            ),
            "premises_used": used,
            "reasoning": {
                "type": "retained_trace_ssm",
                "steps": [
                    f"question_group={best.question_group}",
                    f"model_answer={best.answer}",
                    f"candidate_probabilities={probability_map}",
                ],
            },
            "model_top_probability": round(float(probs[best_idx]), 6),
            "candidate_probabilities": probability_map,
        }

    def _build_group(self, raw_question: str, premises: list[str]) -> list[BacktrackingCandidate]:
        clean_premises = [TextTools.clean(item) for item in premises]
        analyzer = Type1SmallWorldAnalyzer(
            clean_premises,
            self.trainer._tokens,
            self.trainer._overlap_ratio,
            self.trainer._jaccard,
        )
        stem, option_map = self.trainer._raw_options(raw_question)
        question_type = self.trainer.question_classifier.classify(
            raw_question,
            stem,
            option_map,
            clean_premises,
        )
        candidates: list[BacktrackingCandidate] = []
        for label, option_text in option_map.items():
            answer = normalize_for_eval(label)
            score = analyzer.score(
                answer,
                option_text,
                stem,
                len(option_map) > 3,
                question_type,
                options=option_map,
            )
            modal_features = score.features()
            steps = self.trainer._modal_trace_steps(answer, score)
            rag_features, rag_tags = self.trainer.rag_memory.retrieve(
                answer,
                option_text,
                stem,
                raw_question,
                clean_premises,
                option_map,
                question_type,
            )
            candidates.append(
                BacktrackingCandidate(
                    key="api:0",
                    record_id=0,
                    question_id=0,
                    answer=answer,
                    expected=answer,
                    candidate_features=self.trainer._candidate_features(
                        answer,
                        option_text,
                        stem,
                        clean_premises,
                        len(option_map) > 3,
                        question_type,
                    )
                    + modal_features,
                    trace_features=[step.features + rag_features + modal_features for step in steps],
                    trace_text=[
                        f"{step.description} | rag={','.join(rag_tags) if rag_tags else 'none'}"
                        for step in steps
                    ]
                    + score.transition_text,
                    question_group=question_type.group,
                    question_type_features=question_type.features,
                    raw_question=raw_question,
                    stem=stem,
                    premises=clean_premises,
                    options=option_map,
                )
            )
        return candidates

    def _question_with_options(self, query: str, options: list[str]) -> str:
        question = TextTools.clean(query)
        if re.search(r"(?:^|\n)\s*A\.\s+", question):
            return query
        normalized = {normalize_for_eval(option) for option in options}
        if normalized and normalized <= {"Yes", "No", "Uncertain"}:
            return query
        if not options:
            return query
        labels = ["A", "B", "C", "D"]
        rendered = "\n".join(
            f"{labels[idx]}. {TextTools.clean(option)}"
            for idx, option in enumerate(options[:4])
        )
        return f"{query}\n{rendered}"

    def _answer_for_api(self, answer: str, options: list[str]) -> str:
        if not options:
            return answer
        normalized_options = [normalize_for_eval(option) for option in options]
        if answer in options:
            return answer
        if answer in normalized_options:
            return options[normalized_options.index(answer)]
        if answer in {"A", "B", "C", "D"} and not all(option in {"A", "B", "C", "D"} for option in options):
            idx = ord(answer) - ord("A")
            if 0 <= idx < len(options):
                return options[idx]
        return answer

    def _select_relevant_premises(self, query: str, premises: list[str]) -> list[int]:
        query_tokens = set(self.trainer._tokens(query))
        scored = []
        for idx, premise in enumerate(premises):
            overlap = len(query_tokens & set(self.trainer._tokens(premise)))
            if overlap:
                scored.append((overlap, idx))
        if not scored:
            return list(range(min(3, len(premises))))
        scored.sort(reverse=True)
        return sorted(idx for _score, idx in scored[: min(3, len(scored))])
