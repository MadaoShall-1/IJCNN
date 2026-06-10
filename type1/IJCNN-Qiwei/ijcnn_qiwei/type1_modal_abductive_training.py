#!/usr/bin/env python3
"""Type 1 modal-abductive small-world trace experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .type1_backtracking_trace_training import (
    BacktrackingCandidate,
    BacktrackingTraceConfig,
    BacktrackingTraceGenerator,
    BacktrackingTraceTrainer,
    QuestionTypeProfile,
    TextTools,
    TraceStep,
    normalize_for_eval,
)


@dataclass
class SmallWorldScore:
    support: float
    contradiction: float
    unexplained: float
    simplicity: float
    accessibility: float
    rule_support: float
    direct_support: float
    negation_pressure: float
    uncertainty_pressure: float
    option_specificity: float
    world_prior: float
    modal_confidence: float
    proof_support: float
    counter_margin: float
    intermediate_support: float
    selected_premise_density: float
    option_contrast: float
    belief_reward: float
    evidence_count: float
    predicted_truth: str
    transition_text: list[str]

    def features(self) -> list[float]:
        return [
            self.support,
            self.contradiction,
            self.unexplained,
            self.simplicity,
            self.accessibility,
            self.rule_support,
            self.direct_support,
            self.negation_pressure,
            self.uncertainty_pressure,
            self.option_specificity,
            self.world_prior,
            self.modal_confidence,
            self.proof_support,
            self.counter_margin,
            self.intermediate_support,
            self.selected_premise_density,
            self.option_contrast,
            self.belief_reward,
            self.evidence_count,
            float(self.predicted_truth == "true"),
            float(self.predicted_truth == "false"),
            float(self.predicted_truth == "unknown"),
            1.0,
        ]


class Type1SmallWorldAnalyzer:
    """Builds candidate possible worlds from lexical premise evidence."""

    def __init__(self, premises: list[str], tokens_fn: Any, overlap_fn: Any, jaccard_fn: Any) -> None:
        self.premises = premises
        self.tokens_fn = tokens_fn
        self.overlap_fn = overlap_fn
        self.jaccard_fn = jaccard_fn
        self.neg_words = BacktrackingTraceGenerator.neg_words
        self.rule_words = BacktrackingTraceGenerator.rule_words
        self.modal_words = BacktrackingTraceGenerator.modal_words
        self.premise_tokens = [(idx, premise, set(tokens_fn(premise))) for idx, premise in enumerate(premises)]
        self.premise_union = set().union(*(tokens for _idx, _premise, tokens in self.premise_tokens)) if self.premise_tokens else set()

    def score(
        self,
        answer: str,
        option_text: str,
        stem: str,
        is_mcq: bool,
        profile: QuestionTypeProfile,
        options: dict[str, str] | None = None,
    ) -> SmallWorldScore:
        target_text = option_text if is_mcq else self._statement_text(stem)
        world_rows, target_tokens, option_tokens, question_tokens = self._world_rows(answer, target_text, stem)
        world_rows.sort(key=lambda item: item["support"], reverse=True)
        top_worlds = world_rows[:4]

        raw_support = max([row["support"] for row in top_worlds] or [0.0])
        support = raw_support
        direct_support = max([row["direct"] for row in top_worlds] or [0.0])
        contradiction = max([row["conflict"] for row in top_worlds] or [0.0])
        rule_support = max([row["rule"] for row in top_worlds] or [0.0])
        modal_support = max([row["modal"] for row in top_worlds] or [0.0])
        premise_density = sum(row["support"] for row in top_worlds) / max(1, len(top_worlds))
        evidence_count = min(1.0, sum(int(row["support"] > 0.12) for row in top_worlds) / 4.0)
        intermediate_support = max([row["support"] * (0.5 + 0.5 * row["rule"]) for row in top_worlds] or [0.0])
        best_other_support = self._best_other_option_support(answer, stem, options) if is_mcq and options else 0.0
        option_contrast = raw_support - best_other_support
        negation_pressure = self._negation_pressure(target_tokens | question_tokens)
        uncertainty_pressure = max(0.0, 1.0 - support) * (0.55 + 0.25 * float("uncertainty_gap" in profile.group))
        option_specificity = min(1.0, len(option_tokens) / 18.0)
        simplicity = 1.0 / (1.0 + max(0, len(top_worlds) - 1))

        if answer == "Yes":
            support = min(1.5, support + 0.12 * rule_support + 0.08 * modal_support)
            contradiction = min(1.5, contradiction + 0.12 * negation_pressure)
        elif answer == "No":
            support, contradiction = contradiction, max(0.0, support - 0.08 * rule_support)
        elif answer == "Uncertain":
            support = max(uncertainty_pressure, 0.25 * (1.0 - direct_support))
            contradiction = 0.35 * max(contradiction, direct_support)
            if is_mcq:
                support = max(0.05, support - 0.5 * best_other_support)
                contradiction = max(contradiction, 0.65 * best_other_support)
        elif is_mcq:
            support = min(1.5, support + 0.05 * option_specificity + 0.18 * max(0.0, option_contrast))
            contradiction = min(1.5, contradiction + 0.1 * negation_pressure + 0.12 * max(0.0, -option_contrast))

        unexplained = max(0.0, 1.0 - support)
        accessibility = max(0.0, support - contradiction + 0.22 * direct_support + 0.16 * rule_support + 0.12 * simplicity)
        world_prior = max(0.0, min(1.0, 0.44 * support + 0.27 * accessibility + 0.17 * simplicity - 0.28 * contradiction))
        modal_confidence = max(support, contradiction, support - unexplained + 0.5)
        proof_support = min(1.5, max(support, direct_support + 0.25 * rule_support + 0.25 * intermediate_support + 0.2 * max(0.0, option_contrast)))
        counter_margin = max(0.0, best_other_support - raw_support)
        belief_reward = max(-1.0, min(1.0, support - contradiction - 0.15 * unexplained + 0.2 * max(0.0, option_contrast) - 0.15 * counter_margin))
        if support >= contradiction + 0.18 and support >= 0.34:
            truth = "true"
        elif contradiction >= support + 0.18 and contradiction >= 0.34:
            truth = "false"
        else:
            truth = "unknown"

        transition_text = [
            f"observe target: {TextTools.clean(target_text)[:120]}",
            f"construct support worlds={len(top_worlds)}",
            f"best support={support:.3f} contradiction={contradiction:.3f}",
            f"proof support={proof_support:.3f} option contrast={option_contrast:.3f}",
            f"accessibility={accessibility:.3f} prior={world_prior:.3f}",
            f"rank truth={truth} answer={answer}",
        ]
        return SmallWorldScore(
            support=round(min(1.5, support), 6),
            contradiction=round(min(1.5, contradiction), 6),
            unexplained=round(min(1.5, unexplained), 6),
            simplicity=round(simplicity, 6),
            accessibility=round(min(1.5, accessibility), 6),
            rule_support=round(rule_support, 6),
            direct_support=round(direct_support, 6),
            negation_pressure=round(negation_pressure, 6),
            uncertainty_pressure=round(min(1.5, uncertainty_pressure), 6),
            option_specificity=round(option_specificity, 6),
            world_prior=round(world_prior, 6),
            modal_confidence=round(min(1.5, modal_confidence), 6),
            proof_support=round(proof_support, 6),
            counter_margin=round(counter_margin, 6),
            intermediate_support=round(min(1.5, intermediate_support), 6),
            selected_premise_density=round(min(1.5, premise_density), 6),
            option_contrast=round(max(-1.5, min(1.5, option_contrast)), 6),
            belief_reward=round(belief_reward, 6),
            evidence_count=round(evidence_count, 6),
            predicted_truth=truth,
            transition_text=transition_text,
        )

    def _world_rows(self, answer: str, target_text: str, stem: str) -> tuple[list[dict[str, float | int | str]], set[str], set[str], set[str]]:
        target_tokens = set(self.tokens_fn(f"{answer} {target_text} {stem}"))
        option_tokens = set(self.tokens_fn(target_text))
        question_tokens = set(self.tokens_fn(stem))
        rows: list[dict[str, float | int | str]] = []
        for idx, premise, tokens in self.premise_tokens:
            direct = self.overlap_fn(option_tokens, tokens)
            question = self.overlap_fn(question_tokens, tokens)
            joint = self.jaccard_fn(target_tokens | question_tokens, tokens)
            rule = min(1.0, len(tokens & self.rule_words) / 2.0)
            modal = min(1.0, len(tokens & self.modal_words) / 2.0)
            conflict = self._polarity_conflict(answer, target_tokens, question_tokens, tokens)
            support = 0.45 * direct + 0.25 * question + 0.22 * joint + 0.08 * rule
            rows.append(
                {
                    "idx": idx,
                    "premise": premise,
                    "support": support,
                    "conflict": conflict,
                    "rule": rule,
                    "modal": modal,
                    "direct": direct,
                }
            )
        return rows, target_tokens, option_tokens, question_tokens

    def _best_other_option_support(self, answer: str, stem: str, options: dict[str, str] | None) -> float:
        if not options:
            return 0.0
        scores = []
        for label, text in options.items():
            other = normalize_for_eval(label)
            if other == answer or other == "Uncertain":
                continue
            rows, _target_tokens, option_tokens, _question_tokens = self._world_rows(other, text, stem)
            support = max([float(row["support"]) for row in rows] or [0.0])
            specificity = min(1.0, len(option_tokens) / 18.0)
            scores.append(min(1.5, support + 0.05 * specificity))
        return max(scores or [0.0])

    def _statement_text(self, stem: str) -> str:
        marker = "Statement:"
        if marker.lower() in stem.lower():
            return stem.split(marker, 1)[-1].strip()
        return stem

    def _polarity_conflict(self, answer: str, target_tokens: set[str], question_tokens: set[str], premise_tokens: set[str]) -> float:
        target_negative = bool(target_tokens & self.neg_words) or answer == "No"
        question_negative = bool(question_tokens & self.neg_words)
        premise_negative = bool(premise_tokens & self.neg_words)
        polarity_conflict = float((target_negative or question_negative) != premise_negative)
        if answer == "Uncertain":
            return 0.2 * (1.0 - min(1.0, len((target_tokens | question_tokens) & premise_tokens) / 4.0))
        return 0.5 * polarity_conflict + 0.2 * float(answer == "Yes" and premise_negative) + 0.2 * float(answer == "No" and not premise_negative)

    def _negation_pressure(self, tokens: set[str]) -> float:
        return min(1.0, len(tokens & self.neg_words) / 3.0)


@dataclass
class Type1ModalAbductiveConfig(BacktrackingTraceConfig):
    output_path: Path = Path("type1_modal_abductive_eval_results.json")
    summary_output_path: Path = Path("type1_modal_abductive_eval_summary.json")
    model_output_path: Path = Path("type1_modal_abductive_model.json")


class Type1ModalAbductiveTrainer(BacktrackingTraceTrainer):
    def _collect_candidates(self, records: list[dict[str, Any]], record_ids: set[int], split_name: str) -> list[BacktrackingCandidate]:
        candidates: list[BacktrackingCandidate] = []
        processed = 0
        for record_id in sorted(record_ids):
            record = records[record_id]
            premises = [TextTools.clean(item) for item in record.get("premises-NL", []) or []]
            analyzer = Type1SmallWorldAnalyzer(premises, self._tokens, self._overlap_ratio, self._jaccard)
            questions = record.get("questions", []) or []
            for question_id in range(len(questions)):
                raw_question = str(TextTools.safe_get(questions, question_id))
                expected = normalize_for_eval(TextTools.safe_get(record.get("answers", []), question_id))
                stem, options = self._raw_options(raw_question)
                question_type = self.question_classifier.classify(raw_question, stem, options, premises)
                for label, option_text in options.items():
                    answer = normalize_for_eval(label)
                    score = analyzer.score(answer, option_text, stem, len(options) > 3, question_type, options=options)
                    modal_features = score.features()
                    steps = self._modal_trace_steps(answer, score)
                    rag_features, rag_tags = self.rag_memory.retrieve(answer, option_text, stem, raw_question, premises, options, question_type)
                    trace_text = [
                        f"{step.description} | rag={','.join(rag_tags) if rag_tags else 'none'}"
                        for step in steps
                    ]
                    candidates.append(
                        BacktrackingCandidate(
                            key=f"{record_id}:{question_id}",
                            record_id=record_id,
                            question_id=question_id,
                            answer=answer,
                            expected=expected,
                            candidate_features=self._candidate_features(answer, option_text, stem, premises, len(options) > 3, question_type)
                            + modal_features,
                            trace_features=[step.features + rag_features + modal_features for step in steps],
                            trace_text=trace_text + score.transition_text,
                            question_group=question_type.group,
                            question_type_features=question_type.features,
                            raw_question=raw_question,
                            stem=stem,
                            premises=premises,
                            options=options,
                        )
                    )
                processed += 1
                print(
                    f"[{split_name} {processed}] r={record_id} q={question_id} "
                    f"group={question_type.group} expected={expected} small_world_candidates={len(options)}"
                )
        return candidates

    def _modal_trace_steps(self, answer: str, score: SmallWorldScore) -> list[TraceStep]:
        raw_steps = [
            ("observe", 0.0, 0.0, score.support, score.contradiction, score.simplicity, "observe premises and answer candidate"),
            ("select_premise", 0.6, 1.0, score.selected_premise_density, score.counter_margin, score.evidence_count, "select strongest premise evidence"),
            ("propose_support_world", 1.0, 1.0, score.world_prior, score.contradiction, score.simplicity, f"propose support world for {answer}"),
            ("apply_rule", 1.35, 1.0, score.rule_support, score.contradiction, score.intermediate_support, "apply rule-like premise transition"),
            ("infer_fact", 1.65, 1.0, score.proof_support, score.counter_margin, score.intermediate_support, "infer intermediate fact state"),
            ("compare_option", 2.0, -1.0, max(0.0, score.option_contrast), max(0.0, -score.option_contrast), score.evidence_count, "compare candidate world against alternative worlds"),
            ("detect_conflict", 2.25, -1.0, score.unexplained, score.contradiction, score.negation_pressure, "detect contradiction or unsupported world"),
            ("access_world", 2.4, 1.0, score.accessibility, score.contradiction, score.direct_support, "move to accessible proof world"),
        ]
        if score.contradiction > 0.35 or score.predicted_truth == "unknown":
            raw_steps.extend(
                [
                    ("refute_or_suspend", 2.5, -1.0, score.support, score.contradiction, score.uncertainty_pressure, "refute or suspend weak world"),
                    ("backtrack", 1.0, -1.0, score.unexplained, score.contradiction, score.simplicity, "backtrack to adjacent possible world"),
                ]
            )
        raw_steps.extend(
            [
                ("update_belief", 0.4, 0.0, max(0.0, score.belief_reward), max(0.0, -score.belief_reward), score.modal_confidence, "update answer belief distribution"),
                ("rank_distribution", 0.0, 0.0, score.world_prior, score.contradiction, score.modal_confidence, "rank answer probability"),
            ]
        )
        return [
            self._proof_trace_step(
                action=action,
                depth=depth,
                branch=branch,
                support=support,
                conflict=conflict,
                rule_strength=rule_strength,
                answer=answer,
                score=score,
                description=description,
            )
            for action, depth, branch, support, conflict, rule_strength, description in raw_steps[: self.config.max_trace_steps]
        ]

    def _proof_trace_step(
        self,
        action: str,
        depth: float,
        branch: float,
        support: float,
        conflict: float,
        rule_strength: float,
        answer: str,
        score: SmallWorldScore,
        description: str,
    ) -> TraceStep:
        base = self.trace_generator._step(
            action=action,
            depth=depth,
            branch=branch,
            support=support,
            conflict=conflict,
            rule_strength=rule_strength,
            premise_position=0.0,
            answer=answer,
            description=description,
        )
        action_order = (
            "observe",
            "select_premise",
            "propose_support_world",
            "apply_rule",
            "infer_fact",
            "compare_option",
            "detect_conflict",
            "refute_or_suspend",
            "backtrack",
            "access_world",
            "update_belief",
            "rank_distribution",
        )
        proof_features = [
            *(float(action == name) for name in action_order),
            score.proof_support,
            score.counter_margin,
            score.intermediate_support,
            score.selected_premise_density,
            max(0.0, score.option_contrast),
            max(0.0, -score.option_contrast),
            score.belief_reward,
            score.evidence_count,
        ]
        return TraceStep(
            action=action,
            features=base.features + proof_features,
            conflict=base.conflict,
            description=description,
        )

    def _architecture_name(self) -> str:
        return "type1_modal_abductive_proof_state_possible_world_trace_ssm_bge_rag_transformer"

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print("\nType 1 Modal-Abductive Small-World WM Summary")
        print(f"- train_questions: {summary['train_questions']}")
        print(f"- validation_questions: {summary['validation_questions']}")
        print(f"- train_accuracy: {summary['train_accuracy']}")
        print(f"- validation_accuracy: {summary['validation_accuracy']}")
        print(f"- saved model: {self.config.model_output_path}")
        print(f"- saved results: {self.config.output_path}")
        print(f"- saved summary: {self.config.summary_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Type 1 modal-abductive small-world evaluator.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--output", type=Path, default=Path("type1_modal_abductive_eval_results.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("type1_modal_abductive_eval_summary.json"))
    parser.add_argument("--model-output", type=Path, default=Path("type1_modal_abductive_model.json"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--max-trace-steps", type=int, default=18)
    parser.add_argument("--ssm-block-size", type=int, default=8)
    parser.add_argument("--local-attention-window", type=int, default=8)
    parser.add_argument("--propagation-top-k", type=int, default=4)
    parser.add_argument("--implicit-bias-strength", type=float, default=0.18)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--ungrouped-batches", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--rag-backend", choices=["auto", "tfidf", "bge", "numpy"], default="auto")
    parser.add_argument("--bge-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--allow-bge-download", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Type1ModalAbductiveConfig:
    return Type1ModalAbductiveConfig(
        input_path=args.input,
        output_path=args.output,
        summary_output_path=args.summary_output,
        model_output_path=args.model_output,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff_dim=args.transformer_ff_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        max_trace_steps=args.max_trace_steps,
        ssm_block_size=args.ssm_block_size,
        local_attention_window=args.local_attention_window,
        propagation_top_k=args.propagation_top_k,
        implicit_bias_strength=args.implicit_bias_strength,
        consistency_loss_weight=args.consistency_loss_weight,
        grouped_batches=not args.ungrouped_batches,
        device=args.device,
        limit_records=args.limit_records,
        rag_backend=args.rag_backend,
        bge_model=args.bge_model,
        bge_local_files_only=not args.allow_bge_download,
    )


def main() -> None:
    Type1ModalAbductiveTrainer(config_from_args(parse_args())).run()


if __name__ == "__main__":
    main()
