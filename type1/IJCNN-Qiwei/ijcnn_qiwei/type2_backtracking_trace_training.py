#!/usr/bin/env python3
"""Train the retained WM+SSM+Transformer architecture on Type 2 symbolic logic questions."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .type1_backtracking_trace_training import (
    BacktrackingCandidate,
    BacktrackingTraceConfig,
    BacktrackingTraceTrainer,
    QuestionTypeProfile,
    TextTools,
    normalize_for_eval,
)


@dataclass
class Type2BacktrackingTraceConfig(BacktrackingTraceConfig):
    output_path: Path = Path("type2_backtracking_trace_eval_results.json")
    summary_output_path: Path = Path("type2_backtracking_trace_eval_summary.json")
    model_output_path: Path = Path("type2_backtracking_trace_model.json")
    include_nl_context: bool = True


class Type2BacktrackingTraceTrainer(BacktrackingTraceTrainer):
    """Type 2 adapter for formal/symbolic logic rows.

    Type 2 rows are identified by formal symbols or predicate-style expressions
    in the question/options. FOL premises are the primary premises; NL premises
    are appended as auxiliary context when enabled.
    """

    symbolic_question_pattern = re.compile(
        r"∀|∃|¬|→|->|ForAll|Exists|forall|exists|\b[A-Z][A-Za-z0-9_]*\s*\(\s*x\s*\)",
        re.I,
    )
    predicate_pattern = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\s*\(")
    symbol_replacements = {
        "∀": " forall ",
        "∃": " exists ",
        "¬": " not ",
        "→": " implies ",
        "->": " implies ",
        "∧": " and ",
        "∨": " or ",
        "⊥": " contradiction ",
    }

    def _collect_candidates(self, records: list[dict[str, Any]], record_ids: set[int], split_name: str) -> list[BacktrackingCandidate]:
        candidates: list[BacktrackingCandidate] = []
        processed = 0
        skipped = 0
        for record_id in sorted(record_ids):
            record = records[record_id]
            questions = record.get("questions", []) or []
            for question_id in range(len(questions)):
                raw_question = str(TextTools.safe_get(questions, question_id))
                if not self._is_type2_question(raw_question):
                    skipped += 1
                    continue
                premises = self._type2_premises(record)
                expected = normalize_for_eval(TextTools.safe_get(record.get("answers", []), question_id))
                stem, options = self._raw_options(raw_question)
                question_type = self._type2_profile(raw_question, stem, options, premises)
                for label, option_text in options.items():
                    answer = normalize_for_eval(label)
                    steps = self.trace_generator.build(answer, option_text, stem, premises, self._tokens, self._overlap_ratio, self._jaccard)
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
                            candidate_features=self._candidate_features(answer, option_text, stem, premises, len(options) > 3, question_type),
                            trace_features=[step.features + rag_features for step in steps],
                            trace_text=trace_text,
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
                    f"group={question_type.group} expected={expected} trace_candidates={len(options)}"
                )
        if split_name == "train":
            print(f"[type2] skipped_non_type2_questions={skipped}")
        return candidates

    def _is_type2_question(self, question: str) -> bool:
        return bool(self.symbolic_question_pattern.search(question))

    def _type2_premises(self, record: dict[str, Any]) -> list[str]:
        fol_items = [TextTools.clean(item) for item in record.get("premises-FOL", []) or []]
        nl_items = [TextTools.clean(item) for item in record.get("premises-NL", []) or []]
        premises = [f"FOL: {item}" for item in fol_items if item]
        if getattr(self.config, "include_nl_context", True):
            premises.extend(f"NL: {item}" for item in nl_items if item)
        return premises

    def _type2_profile(
        self,
        raw_question: str,
        stem: str,
        options: dict[str, str],
        premises: list[str],
    ) -> QuestionTypeProfile:
        base = self.question_classifier.classify(raw_question, stem, options, premises)
        lower = raw_question.lower()
        is_mcq = any(label in options for label in ("A", "B", "C", "D"))
        quantifier_count = len(re.findall(r"forall|exists|∀|∃|\ball\b|\bevery\b|\bthere exists\b", raw_question, flags=re.I))
        predicate_count = len(self.predicate_pattern.findall(raw_question))
        if is_mcq:
            task = "symbolic_inference_mcq"
        elif "statement" in lower:
            task = "symbolic_truth_statement"
        else:
            task = "symbolic_judgment"
        group = f"type2:{task}:{base.group}"
        features = base.features + [
            1.0,
            float(is_mcq),
            min(quantifier_count / 8.0, 2.0),
            min(predicate_count / 16.0, 2.0),
        ]
        return QuestionTypeProfile(group=group, features=features)

    def _tokens(self, text: str) -> list[str]:
        normalized = text
        for symbol, replacement in self.symbol_replacements.items():
            normalized = normalized.replace(symbol, replacement)
        predicates = [f"pred_{item.lower()}" for item in self.predicate_pattern.findall(normalized)]
        stop = {"the", "a", "an", "and", "or", "is", "are", "to", "of", "for", "in", "on", "that", "this", "it", "with", "from", "by", "above", "based", "does", "which"}
        tokens = [token for token in re.findall(r"[a-z0-9_']+", normalized.lower()) if token not in stop]
        return predicates + tokens

    def _architecture_name(self) -> str:
        return "type2_symbolic_trace_ssm_bge_rag_memory_blockwise_local_attention_transformer"

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print("\nType 2 Symbolic Backtracking Trace WM Summary")
        print(f"- train_questions: {summary['train_questions']}")
        print(f"- validation_questions: {summary['validation_questions']}")
        print(f"- train_accuracy: {summary['train_accuracy']}")
        print(f"- validation_accuracy: {summary['validation_accuracy']}")
        print(f"- saved model: {self.config.model_output_path}")
        print(f"- saved results: {self.config.output_path}")
        print(f"- saved summary: {self.config.summary_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Type 2 symbolic WM+SSM+Transformer evaluator.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--output", type=Path, default=Path("type2_backtracking_trace_eval_results.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("type2_backtracking_trace_eval_summary.json"))
    parser.add_argument("--model-output", type=Path, default=Path("type2_backtracking_trace_model.json"))
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
    parser.add_argument("--batch-size", type=int, default=24)
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
    parser.add_argument("--no-nl-context", action="store_true")
    parser.add_argument("--llm-fallback", action="store_true")
    parser.add_argument("--llm-fallback-model", default="minigpt")
    parser.add_argument("--llm-fallback-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-fallback-api-key-env", default="MINIGPT_API_KEY")
    parser.add_argument("--llm-fallback-threshold", type=float, default=0.62)
    parser.add_argument("--no-llm-fallback-on-uncertain", action="store_true")
    parser.add_argument("--llm-fallback-timeout", type=float, default=45.0)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Type2BacktrackingTraceConfig:
    return Type2BacktrackingTraceConfig(
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
        include_nl_context=not args.no_nl_context,
        llm_fallback=args.llm_fallback,
        llm_fallback_model=args.llm_fallback_model,
        llm_fallback_base_url=args.llm_fallback_base_url,
        llm_fallback_api_key_env=args.llm_fallback_api_key_env,
        llm_fallback_threshold=args.llm_fallback_threshold,
        llm_fallback_on_uncertain=not args.no_llm_fallback_on_uncertain,
        llm_fallback_timeout=args.llm_fallback_timeout,
    )


def main() -> None:
    Type2BacktrackingTraceTrainer(config_from_args(parse_args())).run()


if __name__ == "__main__":
    main()
