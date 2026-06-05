#!/usr/bin/env python3
"""Type 2 modal-abductive possible-world trace experiment."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .type1_backtracking_trace_training import BacktrackingCandidate, QuestionTypeProfile, TextTools, TraceStep, normalize_for_eval
from .type2_backtracking_trace_training import Type2BacktrackingTraceConfig, Type2BacktrackingTraceTrainer


@dataclass
class ModalWorldScore:
    support: float
    contradiction: float
    unexplained: float
    simplicity: float
    accessibility: float
    closure_support: float
    direct_support: float
    negation_pressure: float
    quantifier_match: float
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
            self.closure_support,
            self.direct_support,
            self.negation_pressure,
            self.quantifier_match,
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


class ModalAbductiveWorldAnalyzer:
    atom_pattern = re.compile(r"(¬)?\s*([A-Z][A-Za-z0-9_]*)\s*\(\s*x\s*\)")
    implication_pattern = re.compile(
        r"(?:∀x|forall\s+x)?\s*\(?\s*(¬)?\s*([A-Z][A-Za-z0-9_]*)\s*\(\s*x\s*\)\s*(?:→|->|implies)\s*(.+?)\s*\)?$",
        re.I,
    )

    def __init__(self, premises: list[str]) -> None:
        self.raw_premises = premises
        self.universal_atoms: set[str] = set()
        self.existential_atoms: set[str] = set()
        self.implications: dict[str, set[str]] = {}
        self.predicates: set[str] = set()
        self._parse_premises()
        self.closure = self._transitive_closure()

    def score(self, answer: str, option_text: str, stem: str, options: dict[str, str] | None = None) -> ModalWorldScore:
        target_text = self._target_text(answer, option_text, stem)
        formula = self._normalize(target_text)
        truth, support, contradiction, details = self._formula_truth(formula)
        raw_support = support
        if answer == "No":
            truth = "true" if truth == "false" else "false" if truth == "true" else "unknown"
            support, contradiction = contradiction, support
        elif answer == "Uncertain":
            unknown_score = 1.0 - max(support, contradiction)
            support, contradiction = max(0.0, unknown_score), max(support, contradiction) * 0.35
            truth = "true" if support >= 0.45 else "unknown"

        option_atoms = self._atoms(formula)
        closure_support = float(details.get("closure_support", 0.0))
        direct_hits = len(set(option_atoms) & (self.universal_atoms | self.existential_atoms))
        direct_support = direct_hits / max(1, len(option_atoms))
        quantifier_match = float(bool(re.search(r"forall|exists|∀|∃", formula, flags=re.I)))
        negation_pressure = min(1.0, formula.count("¬") / 3.0 + len(re.findall(r"\bnot\b", formula, flags=re.I)) / 3.0)
        option_specificity = min(1.0, len(set(option_atoms)) / 4.0)
        simplicity = 1.0 / (1.0 + max(0, len(set(option_atoms)) - 1))
        best_other_support = self._best_other_option_support(answer, stem, options) if answer in {"A", "B", "C", "D"} and options else 0.0
        option_contrast = raw_support - best_other_support
        if answer == "Uncertain" and options:
            support = max(0.05, support - 0.45 * best_other_support)
            contradiction = max(contradiction, 0.6 * best_other_support)
        elif answer in {"A", "B", "C", "D"}:
            support = min(1.5, support + 0.15 * max(0.0, option_contrast))
            contradiction = min(1.5, contradiction + 0.12 * max(0.0, -option_contrast))
        accessibility = max(0.0, support - contradiction + 0.25 * direct_support + 0.15 * simplicity)
        unexplained = max(0.0, 1.0 - support)
        world_prior = max(0.0, min(1.0, 0.42 * support + 0.28 * accessibility + 0.18 * simplicity - 0.25 * contradiction))
        modal_confidence = max(support, contradiction, 1.0 - unexplained)
        intermediate_support = max(closure_support, direct_support * (0.5 + 0.5 * quantifier_match))
        selected_premise_density = min(1.5, (support + direct_support + closure_support) / 3.0)
        proof_support = min(1.5, max(support, direct_support + 0.35 * closure_support + 0.2 * quantifier_match + 0.18 * max(0.0, option_contrast)))
        counter_margin = max(0.0, best_other_support - raw_support)
        belief_reward = max(-1.0, min(1.0, support - contradiction - 0.15 * unexplained + 0.2 * max(0.0, option_contrast) - 0.15 * counter_margin))
        evidence_count = min(1.0, len(set(option_atoms) & self.predicates) / max(1, len(set(option_atoms))))
        return ModalWorldScore(
            support=round(min(1.5, support), 6),
            contradiction=round(min(1.5, contradiction), 6),
            unexplained=round(min(1.5, unexplained), 6),
            simplicity=round(simplicity, 6),
            accessibility=round(min(1.5, accessibility), 6),
            closure_support=round(closure_support, 6),
            direct_support=round(direct_support, 6),
            negation_pressure=round(negation_pressure, 6),
            quantifier_match=round(quantifier_match, 6),
            option_specificity=round(option_specificity, 6),
            world_prior=round(world_prior, 6),
            modal_confidence=round(modal_confidence, 6),
            proof_support=round(proof_support, 6),
            counter_margin=round(counter_margin, 6),
            intermediate_support=round(intermediate_support, 6),
            selected_premise_density=round(selected_premise_density, 6),
            option_contrast=round(max(-1.5, min(1.5, option_contrast)), 6),
            belief_reward=round(belief_reward, 6),
            evidence_count=round(evidence_count, 6),
            predicted_truth=truth,
            transition_text=[
                f"observe target formula: {formula[:120]}",
                f"propose possible world for answer {answer}",
                f"test explanation support={support:.3f} contradiction={contradiction:.3f}",
                f"proof support={proof_support:.3f} option contrast={option_contrast:.3f}",
                f"access adjacent world score={accessibility:.3f}",
                f"rank explanation prior={world_prior:.3f} truth={truth}",
            ],
        )

    def _best_other_option_support(self, answer: str, stem: str, options: dict[str, str] | None) -> float:
        if not options:
            return 0.0
        scores = []
        for label, option_text in options.items():
            other = normalize_for_eval(label)
            if other == answer or other == "Uncertain":
                continue
            formula = self._normalize(self._target_text(other, option_text, stem))
            _truth, support, _contradiction, _details = self._formula_truth(formula)
            atoms = self._atoms(formula)
            specificity = min(1.0, len(set(atoms)) / 4.0)
            scores.append(min(1.5, support + 0.05 * specificity))
        return max(scores or [0.0])

    def _parse_premises(self) -> None:
        for premise in self.raw_premises:
            if not premise.startswith("FOL:"):
                continue
            text = self._normalize(premise[4:])
            if self._parse_implication(text):
                continue
            quantifier = "exists" if re.search(r"∃|exists", text, flags=re.I) else "forall" if re.search(r"∀|forall", text, flags=re.I) else ""
            for neg, pred in self.atom_pattern.findall(text):
                literal = self._literal(pred, bool(neg))
                self.predicates.add(pred.upper())
                if quantifier == "exists":
                    self.existential_atoms.add(literal)
                elif quantifier == "forall":
                    self.universal_atoms.add(literal)

    def _parse_implication(self, text: str) -> bool:
        match = self.implication_pattern.search(text)
        if not match:
            return False
        left_neg, left_pred, right = match.groups()
        left = self._literal(left_pred, bool(left_neg))
        consequents = self._atoms(right)
        if not consequents:
            return False
        self.predicates.add(left_pred.upper())
        for consequent in consequents:
            self.implications.setdefault(left, set()).add(consequent)
            self.predicates.add(consequent.replace("not_", "").upper())
            # Add contrapositive for common universal implication cases.
            self.implications.setdefault(self._negate(consequent), set()).add(self._negate(left))
        return True

    def _transitive_closure(self) -> dict[str, set[str]]:
        closure = {key: set(values) for key, values in self.implications.items()}
        changed = True
        while changed:
            changed = False
            for source, targets in list(closure.items()):
                expanded = set(targets)
                for target in targets:
                    expanded.update(closure.get(target, set()))
                if not expanded <= targets:
                    closure[source] = expanded
                    changed = True
        return closure

    def _formula_truth(self, formula: str) -> tuple[str, float, float, dict[str, float]]:
        atoms = self._atoms(formula)
        if not atoms:
            return "unknown", 0.15, 0.1, {"closure_support": 0.0}

        if self._is_implication(formula):
            antecedents, consequents = self._split_implication(formula)
            hits = 0
            misses = 0
            for left in antecedents:
                reachable = self.closure.get(left, set()) | self.implications.get(left, set())
                for right in consequents:
                    if right in reachable or right in self.universal_atoms:
                        hits += 1
                    elif self._negate(right) in reachable or self._negate(right) in self.universal_atoms:
                        misses += 1
            total = max(1, len(antecedents) * len(consequents))
            support = hits / total
            contradiction = misses / total
            if support >= 0.75 and contradiction <= 0.2:
                return "true", support, contradiction, {"closure_support": support}
            if contradiction >= 0.5:
                return "false", support, contradiction, {"closure_support": support}
            return "unknown", support, contradiction, {"closure_support": support}

        positive_hits = sum(int(atom in self.universal_atoms or atom in self.existential_atoms) for atom in atoms)
        negative_hits = sum(int(self._negate(atom) in self.universal_atoms or self._negate(atom) in self.existential_atoms) for atom in atoms)
        support = positive_hits / max(1, len(atoms))
        contradiction = negative_hits / max(1, len(atoms))
        if support >= 0.75 and contradiction == 0:
            return "true", support, contradiction, {"closure_support": 0.0}
        if contradiction >= 0.75:
            return "false", support, contradiction, {"closure_support": 0.0}
        return "unknown", support, contradiction, {"closure_support": 0.0}

    def _target_text(self, answer: str, option_text: str, stem: str) -> str:
        if answer in {"A", "B", "C", "D"}:
            return option_text
        statement = re.search(r"Statement:\s*(.+)$", stem, flags=re.I | re.S)
        return statement.group(1).strip() if statement else stem

    def _is_implication(self, formula: str) -> bool:
        return bool(re.search(r"→|->|implies", formula, flags=re.I))

    def _split_implication(self, formula: str) -> tuple[list[str], list[str]]:
        parts = re.split(r"→|->|implies", formula, maxsplit=1, flags=re.I)
        if len(parts) != 2:
            return self._atoms(formula), []
        return self._atoms(parts[0]), self._atoms(parts[1])

    def _atoms(self, text: str) -> list[str]:
        atoms = []
        for neg, pred in self.atom_pattern.findall(text):
            atoms.append(self._literal(pred, bool(neg)))
        return atoms

    def _literal(self, pred: str, negated: bool) -> str:
        clean = pred.upper()
        return f"not_{clean}" if negated else clean

    def _negate(self, literal: str) -> str:
        return literal[4:] if literal.startswith("not_") else f"not_{literal}"

    def _normalize(self, text: str) -> str:
        return (
            TextTools.clean(text)
            .replace("forall x", "∀x")
            .replace("ForAll", "∀")
            .replace("Exists", "∃")
            .replace("exists x", "∃x")
        )


@dataclass
class Type2ModalAbductiveConfig(Type2BacktrackingTraceConfig):
    output_path: Path = Path("type2_modal_abductive_eval_results.json")
    summary_output_path: Path = Path("type2_modal_abductive_eval_summary.json")
    model_output_path: Path = Path("type2_modal_abductive_model.json")


class Type2ModalAbductiveTrainer(Type2BacktrackingTraceTrainer):
    def _collect_candidates(self, records: list[dict[str, Any]], record_ids: set[int], split_name: str) -> list[BacktrackingCandidate]:
        candidates: list[BacktrackingCandidate] = []
        processed = 0
        skipped = 0
        for record_id in sorted(record_ids):
            record = records[record_id]
            questions = record.get("questions", []) or []
            premises = self._type2_premises(record)
            analyzer = ModalAbductiveWorldAnalyzer(premises)
            for question_id in range(len(questions)):
                raw_question = str(TextTools.safe_get(questions, question_id))
                if not self._is_type2_question(raw_question):
                    skipped += 1
                    continue
                expected = normalize_for_eval(TextTools.safe_get(record.get("answers", []), question_id))
                stem, options = self._raw_options(raw_question)
                question_type = self._type2_profile(raw_question, stem, options, premises)
                for label, option_text in options.items():
                    answer = normalize_for_eval(label)
                    world_score = analyzer.score(answer, option_text, stem, options=options)
                    modal_features = world_score.features()
                    steps = self._modal_trace_steps(answer, world_score)
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
                            trace_text=trace_text + world_score.transition_text,
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
                    f"group={question_type.group} expected={expected} modal_candidates={len(options)}"
                )
        if split_name == "train":
            print(f"[type2-modal] skipped_non_type2_questions={skipped}")
        return candidates

    def _modal_trace_steps(self, answer: str, score: ModalWorldScore) -> list[TraceStep]:
        raw_steps = [
            ("observe", 0.0, 0.0, score.support, score.contradiction, score.simplicity, "observe target and premises"),
            ("select_premise", 0.6, 1.0, score.selected_premise_density, score.counter_margin, score.evidence_count, "select strongest FOL premise evidence"),
            ("propose_support_world", 1.0, 1.0, score.world_prior, score.contradiction, score.simplicity, f"propose support world for {answer}"),
            ("apply_rule", 1.35, 1.0, score.closure_support, score.contradiction, score.intermediate_support, "apply symbolic implication closure"),
            ("infer_fact", 1.65, 1.0, score.proof_support, score.counter_margin, score.intermediate_support, "infer intermediate symbolic fact"),
            ("compare_option", 2.0, -1.0, max(0.0, score.option_contrast), max(0.0, -score.option_contrast), score.evidence_count, "compare candidate world against alternative worlds"),
            ("detect_conflict", 2.25, -1.0, score.unexplained, score.contradiction, score.negation_pressure, "detect contradiction or unsupported symbolic world"),
            ("access_world", 2.4, 1.0, score.accessibility, score.contradiction, score.direct_support, "move to accessible proof world"),
        ]
        if score.contradiction > 0.35 or score.predicted_truth == "unknown":
            raw_steps.extend(
                [
                    ("refute_or_suspend", 2.5, -1.0, score.support, score.contradiction, score.negation_pressure, "refute or suspend weak symbolic world"),
                    ("backtrack", 1.0, -1.0, score.unexplained, score.contradiction, score.simplicity, "backtrack to adjacent world"),
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
        score: ModalWorldScore,
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
        return "type2_modal_abductive_proof_state_possible_world_trace_ssm_bge_rag_transformer"

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print("\nType 2 Modal-Abductive Possible-World WM Summary")
        print(f"- train_questions: {summary['train_questions']}")
        print(f"- validation_questions: {summary['validation_questions']}")
        print(f"- train_accuracy: {summary['train_accuracy']}")
        print(f"- validation_accuracy: {summary['validation_accuracy']}")
        print(f"- saved model: {self.config.model_output_path}")
        print(f"- saved results: {self.config.output_path}")
        print(f"- saved summary: {self.config.summary_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Type 2 modal-abductive possible-world evaluator.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--output", type=Path, default=Path("type2_modal_abductive_eval_results.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("type2_modal_abductive_eval_summary.json"))
    parser.add_argument("--model-output", type=Path, default=Path("type2_modal_abductive_model.json"))
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
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Type2ModalAbductiveConfig:
    return Type2ModalAbductiveConfig(
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
    )


def main() -> None:
    Type2ModalAbductiveTrainer(config_from_args(parse_args())).run()


if __name__ == "__main__":
    main()
