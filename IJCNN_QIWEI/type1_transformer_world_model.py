#!/usr/bin/env python3
"""Local transformer-style world model for Type 1 reasoning.

The module implements the architecture requested by the user:

1. External imagination: collect possible answer states from the symbolic,
   semantic, causal, and global-option stages.
2. Internal brain imagination: run a local transformer-style self-attention
   model over imagined states, allowing candidates to compare against each
   other in a latent space.
3. Evaluation: produce an answer probability distribution and optionally
   override the pipeline answer only when the latent evaluator is confident.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .common import Stage0Input
from .type1_preprocessing import TextTools


@dataclass
class TransformerWorldModelConfig:
    hidden_dim: int = 32
    imagination_layers: int = 2
    attention_heads: int = 4
    frame_local_window: int = 4
    ssm_block_size: int = 4
    allow_override: bool = False
    temperature: float = 0.52
    override_margin: float = 0.15
    minimum_override_confidence: float = 0.45
    minimum_override_winner_margin: float = 0.08
    seed: int = 17


@dataclass
class ExternalImaginedState:
    answer: str
    text: str
    source: str
    feature_vector: list[float]
    feature_names: list[str]
    prior_logit: float


@dataclass
class InternalImaginationStep:
    layer: int
    attention: dict[str, dict[str, float]]
    head_attention: dict[str, dict[str, dict[str, float]]]
    ssm_state_norms: dict[str, float]
    scale_gates: dict[str, float]
    latent_norms: dict[str, float]


@dataclass
class TransformerWorldModelResult:
    architecture: str
    pipeline_answer: str
    raw_brain_answer: str
    final_answer: str
    should_override: bool
    confidence: float
    margin: float
    answer_distribution: dict[str, float]
    external_imagination_states: list[ExternalImaginedState]
    internal_imagination_steps: list[InternalImaginationStep]
    candidate_latents: dict[str, list[float]]
    evaluator_logits: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalTransformerWorldModel:
    """Small deterministic transformer brain over imagined answer states."""

    feature_names = [
        "support",
        "contradiction",
        "semantic",
        "causal_score",
        "global_prior",
        "premise_alignment",
        "posterior",
        "proof_depth_score",
        "stage2_confidence",
        "causal_yes",
        "causal_uncertain",
        "causal_no",
        "missing_condition",
        "counterfactual_gap",
        "is_pipeline_answer",
        "is_yes",
        "is_no",
        "is_uncertain",
        "is_mcq",
        "question_has_according",
        "question_has_statement",
        "question_has_requirement",
    ]

    def __init__(self, config: TransformerWorldModelConfig | None = None) -> None:
        self.config = config or TransformerWorldModelConfig()
        rng = np.random.default_rng(self.config.seed)
        feature_dim = len(self.feature_names)
        hidden = self.config.hidden_dim
        heads = max(1, self.config.attention_heads)
        if hidden % heads != 0:
            raise ValueError("transformer brain hidden_dim must be divisible by attention_heads.")
        self.head_dim = hidden // heads
        self.input_projection = rng.normal(0.0, 1.0 / math.sqrt(feature_dim), size=(feature_dim, hidden))
        self.condition_projection = rng.normal(0.0, 1.0 / math.sqrt(feature_dim), size=(feature_dim, hidden))
        self.adaln_gamma = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, hidden))
        self.adaln_beta = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, hidden))
        self.scale_projection = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, 3))
        self.ssm_in = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, hidden))
        self.ssm_state = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, hidden))
        self.ssm_out = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, hidden))
        self.wq = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(heads, hidden, self.head_dim))
        self.wk = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(heads, hidden, self.head_dim))
        self.wv = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(heads, hidden, self.head_dim))
        self.wo = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, hidden))
        self.ff1 = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden, hidden * 2))
        self.ff2 = rng.normal(0.0, 1.0 / math.sqrt(hidden * 2), size=(hidden * 2, hidden))
        self.readout = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=(hidden,))

    def run(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        graph_summary: dict[str, Any],
        stage2_candidate: dict[str, Any],
    ) -> TransformerWorldModelResult:
        states = self._external_imagine(stage_input, classification, graph_summary, stage2_candidate)
        if not states:
            states = [
                ExternalImaginedState(
                    answer="Uncertain",
                    text="No imagined states were available.",
                    source="empty_imagination",
                    feature_vector=[0.0] * len(self.feature_names),
                    feature_names=self.feature_names,
                    prior_logit=0.0,
                )
            ]
        features = np.array([state.feature_vector for state in states], dtype=np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.clip(features, -4.0, 4.0)
        latent = self._layer_norm(self._safe_matmul(features, self.input_projection))
        condition = self._condition_vector(features)
        steps: list[InternalImaginationStep] = []
        for layer in range(self.config.imagination_layers):
            latent, diagnostics = self._imagination_layer(latent, condition)
            steps.append(
                InternalImaginationStep(
                    layer=layer + 1,
                    attention=self._attention_dict(states, diagnostics["attention"]),
                    head_attention=self._head_attention_dict(states, diagnostics["head_attention"]),
                    ssm_state_norms={
                        state.answer: round(float(norm), 6)
                        for state, norm in zip(states, diagnostics["ssm_state_norms"].tolist())
                    },
                    scale_gates={
                        "ssm": round(float(diagnostics["scale_gates"][0]), 6),
                        "attention": round(float(diagnostics["scale_gates"][1]), 6),
                        "ffn": round(float(diagnostics["scale_gates"][2]), 6),
                    },
                    latent_norms={
                        state.answer: round(float(np.linalg.norm(vector)), 6)
                        for state, vector in zip(states, latent)
                    },
                )
            )

        logits = self._evaluate_logits(states, latent)
        distribution = self._softmax(logits, self.config.temperature)
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        raw_answer, confidence = ranked[0] if ranked else ("Uncertain", 0.0)
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = confidence - second
        pipeline_answer = TextTools.clean(stage2_candidate.get("answer", "")) or "Uncertain"
        pipeline_probability = distribution.get(pipeline_answer, 0.0)
        should_override = (
            self.config.allow_override
            and raw_answer != pipeline_answer
            and (confidence - pipeline_probability) >= self.config.override_margin
            and confidence >= self.config.minimum_override_confidence
            and margin >= self.config.minimum_override_winner_margin
        )
        final_answer = raw_answer if should_override else pipeline_answer
        return TransformerWorldModelResult(
            architecture="local_multihead_blockwise_ssm_transformer_world_model",
            pipeline_answer=pipeline_answer,
            raw_brain_answer=raw_answer,
            final_answer=final_answer,
            should_override=should_override,
            confidence=round(float(confidence), 6),
            margin=round(float(margin), 6),
            answer_distribution={label: round(float(prob), 6) for label, prob in distribution.items()},
            external_imagination_states=states,
            internal_imagination_steps=steps,
            candidate_latents={
                state.answer: [round(float(value), 6) for value in vector.tolist()]
                for state, vector in zip(states, latent)
            },
            evaluator_logits={label: round(float(value), 6) for label, value in logits.items()},
        )

    def _external_imagine(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        graph_summary: dict[str, Any],
        stage2_candidate: dict[str, Any],
    ) -> list[ExternalImaginedState]:
        pipeline_answer = TextTools.clean(stage2_candidate.get("answer", "")) or "Uncertain"
        option_scores = stage2_candidate.get("option_scores", []) or []
        causal = stage2_candidate.get("causal_inference", {}) or {}
        states: list[ExternalImaginedState] = []

        if classification.get("question_format") == "multiple_choice":
            labels = list((classification.get("mcq_options") or {}).keys())
            labels.extend(str(item.get("label", "")) for item in option_scores if isinstance(item, dict))
            labels.append("Uncertain")
        else:
            labels = ["Yes", "No", "Uncertain"]
        labels = [label for label in dict.fromkeys(TextTools.clean(item) for item in labels) if label]

        option_by_label = {
            TextTools.clean(item.get("label", "")): item
            for item in option_scores
            if isinstance(item, dict)
        }
        global_dist = causal.get("global_option_distribution", {}) if isinstance(causal, dict) else {}
        option_distribution = global_dist.get("option_distribution", {}) if isinstance(global_dist, dict) else {}

        for answer in labels:
            option_score = option_by_label.get(answer, {})
            trace = self._causal_trace(answer, causal)
            features = self._feature_vector(
                answer=answer,
                stage_input=stage_input,
                classification=classification,
                graph_summary=graph_summary,
                stage2_candidate=stage2_candidate,
                option_score=option_score,
                causal_trace=trace,
                global_probability=self._float(option_distribution.get(answer)),
                pipeline_answer=pipeline_answer,
            )
            prior_logit = self._prior_logit(features)
            states.append(
                ExternalImaginedState(
                    answer=answer,
                    text=self._state_text(answer, classification, option_score),
                    source="external_stage2_causal_semantic_imagination",
                    feature_vector=[round(value, 6) for value in features],
                    feature_names=self.feature_names,
                    prior_logit=round(prior_logit, 6),
                )
            )
        return states

    def _feature_vector(
        self,
        *,
        answer: str,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        graph_summary: dict[str, Any],
        stage2_candidate: dict[str, Any],
        option_score: dict[str, Any],
        causal_trace: dict[str, Any],
        global_probability: float,
        pipeline_answer: str,
    ) -> list[float]:
        answer_distribution = causal_trace.get("answer_distribution", {}) if isinstance(causal_trace, dict) else {}
        proof_depth = option_score.get("proof_depth")
        proof_depth_score = 0.0 if proof_depth is None else 1.0 / (1.0 + self._float(proof_depth))
        lower_question = stage_input.question.lower()
        return [
            self._float(option_score.get("support")),
            self._float(option_score.get("contradiction")),
            self._float(option_score.get("semantic")),
            self._float(option_score.get("causal_score")),
            max(global_probability, self._float(option_score.get("global_semantic_prior"))),
            self._float(option_score.get("premise_alignment_score")),
            self._float(option_score.get("posterior_probability")),
            proof_depth_score,
            self._float(stage2_candidate.get("confidence")),
            self._float(answer_distribution.get("Yes")),
            self._float(answer_distribution.get("Uncertain")),
            self._float(answer_distribution.get("No")),
            self._float(causal_trace.get("missing_condition_penalty")) if isinstance(causal_trace, dict) else 0.0,
            self._float(causal_trace.get("counterfactual_gap")) if isinstance(causal_trace, dict) else 0.0,
            float(answer == pipeline_answer),
            float(answer == "Yes"),
            float(answer == "No"),
            float(answer == "Uncertain"),
            float(classification.get("question_format") == "multiple_choice"),
            float("according to" in lower_question),
            float("statement:" in lower_question),
            float(any(cue in lower_question for cue in ["requirement", "eligible", "qualify", "can "])),
        ]

    def _prior_logit(self, features: list[float]) -> float:
        values = dict(zip(self.feature_names, features))
        return (
            values["support"] * 0.34
            - values["contradiction"] * 0.36
            + values["semantic"] * 0.14
            + values["global_prior"] * 0.25
            + values["posterior"] * 0.25
            + values["causal_yes"] * 0.2
            - values["causal_no"] * 0.18
            - values["missing_condition"] * 0.18
            + values["is_pipeline_answer"] * 0.08
        )

    def _imagination_layer(self, latent: np.ndarray, condition: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        gates = self._scale_gates(condition)

        ssm_input = self._adaln(latent, condition)
        ssm_output, ssm_norms = self._blockwise_ssm(ssm_input)
        latent = self._layer_norm(latent + gates[0] * ssm_output)

        attention_input = self._adaln(latent, condition)
        attention_output, head_attention = self._frame_local_multihead_attention(attention_input)
        latent = self._layer_norm(latent + gates[1] * attention_output)

        ffn_input = self._adaln(latent, condition)
        ffn_output = self._safe_matmul(self._gelu(self._safe_matmul(ffn_input, self.ff1)), self.ff2)
        ffn_output = np.nan_to_num(np.clip(ffn_output, -6.0, 6.0), nan=0.0, posinf=0.0, neginf=0.0)
        latent = self._layer_norm(latent + gates[2] * ffn_output)

        return latent, {
            "attention": head_attention.mean(axis=0),
            "head_attention": head_attention,
            "ssm_state_norms": ssm_norms,
            "scale_gates": gates,
        }

    def _blockwise_ssm(self, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        block_size = max(1, self.config.ssm_block_size)
        outputs = np.zeros_like(latent)
        state_norms = np.zeros((latent.shape[0],), dtype=np.float64)
        for start in range(0, latent.shape[0], block_size):
            end = min(latent.shape[0], start + block_size)
            state = np.zeros((latent.shape[1],), dtype=np.float64)
            for row in range(start, end):
                current = latent[row]
                state = 0.72 * state + self._safe_matmul(current, self.ssm_in)
                state = np.tanh(state + 0.18 * self._safe_matmul(current, self.ssm_state))
                outputs[row] = self._safe_matmul(state, self.ssm_out)
                state_norms[row] = np.linalg.norm(state)
        return np.nan_to_num(np.clip(outputs, -6.0, 6.0), nan=0.0, posinf=0.0, neginf=0.0), state_norms

    def _frame_local_multihead_attention(self, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        head_outputs: list[np.ndarray] = []
        head_attentions: list[np.ndarray] = []
        for head in range(max(1, self.config.attention_heads)):
            q = self._safe_matmul(latent, self.wq[head])
            k = self._safe_matmul(latent, self.wk[head])
            v = self._safe_matmul(latent, self.wv[head])
            scores = self._safe_matmul(q, k.T) / math.sqrt(max(1, self.head_dim))
            scores = self._apply_frame_local_mask(scores)
            attention = self._row_softmax(scores)
            head_outputs.append(self._safe_matmul(attention, v))
            head_attentions.append(attention)
        concatenated = np.concatenate(head_outputs, axis=1)
        projected = self._safe_matmul(concatenated, self.wo)
        return np.nan_to_num(np.clip(projected, -6.0, 6.0), nan=0.0, posinf=0.0, neginf=0.0), np.stack(head_attentions)

    def _apply_frame_local_mask(self, scores: np.ndarray) -> np.ndarray:
        window = max(1, self.config.frame_local_window)
        masked = np.full_like(scores, -1e4)
        for row in range(scores.shape[0]):
            start = max(0, row - window)
            end = min(scores.shape[1], row + window + 1)
            masked[row, start:end] = scores[row, start:end]
        return np.nan_to_num(np.clip(masked, -1e4, 12.0), nan=-1e4, posinf=12.0, neginf=-1e4)

    def _condition_vector(self, features: np.ndarray) -> np.ndarray:
        pooled = features.mean(axis=0)
        return np.tanh(self._safe_matmul(pooled, self.condition_projection))

    def _adaln(self, latent: np.ndarray, condition: np.ndarray) -> np.ndarray:
        normalized = self._layer_norm(latent)
        gamma = 0.1 * np.tanh(self._safe_matmul(condition, self.adaln_gamma))
        beta = 0.1 * np.tanh(self._safe_matmul(condition, self.adaln_beta))
        return normalized * (1.0 + gamma) + beta

    def _scale_gates(self, condition: np.ndarray) -> np.ndarray:
        raw = self._safe_matmul(condition, self.scale_projection)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -8.0, 8.0)))

    def _evaluate_logits(
        self,
        states: list[ExternalImaginedState],
        latent: np.ndarray,
    ) -> dict[str, float]:
        raw = self._safe_matmul(latent, self.readout)
        logits: dict[str, float] = {}
        for state, latent_logit in zip(states, raw.tolist()):
            feature_values = dict(zip(state.feature_names, state.feature_vector))
            uncertainty_bonus = 0.0
            if state.answer == "Uncertain":
                uncertainty_bonus = (
                    feature_values["causal_uncertain"] * 0.35
                    + feature_values["counterfactual_gap"] * 0.18
                    + (1.0 - feature_values["stage2_confidence"]) * 0.15
                )
            logits[state.answer] = (
                float(latent_logit) * 0.18
                + state.prior_logit
                + uncertainty_bonus
            )
        return logits

    def _attention_dict(
        self,
        states: list[ExternalImaginedState],
        attention: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for row_idx, source_state in enumerate(states):
            output[source_state.answer] = {
                target_state.answer: round(float(attention[row_idx, col_idx]), 6)
                for col_idx, target_state in enumerate(states)
            }
        return output

    def _head_attention_dict(
        self,
        states: list[ExternalImaginedState],
        head_attention: np.ndarray,
    ) -> dict[str, dict[str, dict[str, float]]]:
        output: dict[str, dict[str, dict[str, float]]] = {}
        for head_idx in range(head_attention.shape[0]):
            output[f"head_{head_idx}"] = self._attention_dict(states, head_attention[head_idx])
        return output

    def _state_text(self, answer: str, classification: dict[str, Any], option_score: dict[str, Any]) -> str:
        options = classification.get("mcq_options") or {}
        if answer in options:
            return TextTools.clean(options[answer])
        return TextTools.clean(option_score.get("text", "")) or answer

    def _causal_trace(self, answer: str, causal: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(causal, dict):
            return {}
        options = causal.get("options")
        if isinstance(options, dict) and isinstance(options.get(answer), dict):
            return options[answer]
        query = causal.get("query")
        if isinstance(query, dict):
            return query
        return {}

    def _row_softmax(self, matrix: np.ndarray) -> np.ndarray:
        shifted = matrix - matrix.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)

    def _softmax(self, logits: dict[str, float], temperature: float) -> dict[str, float]:
        if not logits:
            return {}
        temp = max(0.05, temperature)
        max_logit = max(logits.values())
        exp_values = {label: math.exp((value - max_logit) / temp) for label, value in logits.items()}
        total = sum(exp_values.values()) or 1.0
        return {label: value / total for label, value in exp_values.items()}

    def _layer_norm(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = np.clip(matrix, -8.0, 8.0)
        mean = matrix.mean(axis=1, keepdims=True)
        std = matrix.std(axis=1, keepdims=True)
        return (matrix - mean) / np.maximum(std, 1e-6)

    def _safe_matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left = np.nan_to_num(np.clip(left, -8.0, 8.0), nan=0.0, posinf=0.0, neginf=0.0)
        right = np.nan_to_num(np.clip(right, -8.0, 8.0), nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            output = np.matmul(left, right)
        return np.nan_to_num(np.clip(output, -8.0, 8.0), nan=0.0, posinf=0.0, neginf=0.0)

    def _gelu(self, values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(np.clip(values, -8.0, 8.0), nan=0.0, posinf=0.0, neginf=0.0)
        return 0.5 * values * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (values + 0.044715 * np.power(values, 3))))

    def _float(self, value: Any) -> float:
        try:
            if value is None:
                return 0.0
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return 0.0
            return number
        except (TypeError, ValueError):
            return 0.0
