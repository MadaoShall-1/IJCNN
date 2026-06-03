#!/usr/bin/env python3
"""Type 1 multi-stage reasoning pipeline.

The pipeline keeps Stage 0 as the semantic entry layer and adds Type 1-only
reasoning stages:

- Stage 1: build a lightweight evidence graph from FOL/NL premises.
- Stage 2: run a causal inference world model over action-effect rules.
- Stage 3: gate and fuse Stage 0/Stage 2 candidates with causal checks.

No Z3 dependency is used here. The symbolic pass is a small forward-chaining
engine designed for the Type 1 educational-query shapes in this project.
"""

from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .common import Stage0Input
from .semantic_hybrid_parser import (
    SemanticHybridCandidate,
    SemanticHybridConfig,
    SemanticHybridGate,
    Type1SemanticHybridParser,
    normalize_for_eval,
)
from .semantic_tree_rag import BGEVectorIndex, SemanticRAGConfig
from .type1_transformer_world_model import LocalTransformerWorldModel, TransformerWorldModelConfig
from .type1_preprocessing import TextTools, Type1QuestionClassifier


TOKEN_RE = re.compile(r"[a-z][a-z0-9_']*", re.IGNORECASE)
ATOM_RE = re.compile(
    r"(¬\s*)?[A-Za-z_][A-Za-z0-9_]*\s*\([^()]*\)"
    r"(?:\s*(?:=|>=|<=|>|<|≥|≤)\s*-?[A-Za-z0-9_.]+)?"
)
IMPLICATION_RE = re.compile(r"(?:->|→)")
VARIABLES = {"x", "y", "z", "d", "p", "s"}


@dataclass(frozen=True)
class LogicAtom:
    predicate: str
    args: tuple[str, ...] = ()
    negated: bool = False
    operator: str = ""
    value: str = ""

    def canonical(self) -> str:
        sign = "not_" if self.negated else ""
        args = "_".join(self.args)
        suffix = f"_{self.operator}_{self.value}" if self.operator else ""
        return f"{sign}{self.predicate}_{args}{suffix}".strip("_")

    def lexical_text(self) -> str:
        parts = [self.predicate.replace("_", " "), *[arg.replace("_", " ") for arg in self.args]]
        if self.operator:
            parts.extend([self.operator, self.value.replace("_", " ")])
        if self.negated:
            parts.insert(0, "not")
        return " ".join(part for part in parts if part)


@dataclass
class EvidenceFact:
    atom: LogicAtom
    source: str
    source_index: int | None = None
    depth: int = 0
    support_chain: list[str] = field(default_factory=list)


@dataclass
class EvidenceRule:
    rule_id: str
    antecedents: list[LogicAtom]
    consequents: list[LogicAtom]
    source: str
    source_index: int | None = None
    derived: bool = False


@dataclass
class EvidenceGraph:
    facts: list[EvidenceFact]
    rules: list[EvidenceRule]
    derived_facts: list[EvidenceFact]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_count": len(self.facts),
            "rule_count": len(self.rules),
            "derived_fact_count": len(self.derived_facts),
            "facts": [self._fact_to_dict(fact) for fact in self.facts],
            "rules": [self._rule_to_dict(rule) for rule in self.rules],
            "derived_facts": [self._fact_to_dict(fact) for fact in self.derived_facts],
        }

    def summary(self) -> dict[str, Any]:
        predicates = Counter(fact.atom.predicate for fact in self.facts + self.derived_facts)
        return {
            "fact_count": len(self.facts),
            "rule_count": len(self.rules),
            "derived_fact_count": len(self.derived_facts),
            "top_predicates": dict(predicates.most_common(12)),
        }

    @staticmethod
    def _fact_to_dict(fact: EvidenceFact) -> dict[str, Any]:
        return {
            "atom": asdict(fact.atom),
            "canonical": fact.atom.canonical(),
            "text": fact.atom.lexical_text(),
            "source": fact.source,
            "source_index": fact.source_index,
            "depth": fact.depth,
            "support_chain": fact.support_chain,
        }

    @staticmethod
    def _rule_to_dict(rule: EvidenceRule) -> dict[str, Any]:
        return {
            "rule_id": rule.rule_id,
            "antecedents": [asdict(atom) for atom in rule.antecedents],
            "consequents": [asdict(atom) for atom in rule.consequents],
            "source": rule.source,
            "source_index": rule.source_index,
            "derived": rule.derived,
        }


@dataclass
class Stage2OptionScore:
    label: str
    text: str
    support: float
    contradiction: float
    semantic: float
    causal_score: float
    causal_decision: str
    proof_depth: int | None
    evidence: list[str]
    decision: str
    global_semantic_prior: float = 0.0
    premise_alignment_score: float = 0.0
    posterior_probability: float = 0.0


@dataclass
class PremiseOptionJudgment:
    label: str
    option_text: str
    premise_index: int
    premise_text: str
    semantic_similarity: float
    lexical_overlap: float
    support_probability: float
    contradiction_probability: float
    relevance_probability: float
    contribution: float
    judge_source: str


@dataclass
class GlobalOptionDistribution:
    selected_option: str
    option_distribution: dict[str, float]
    option_logits: dict[str, float]
    margin: float
    entropy: float
    judgments: list[PremiseOptionJudgment]


@dataclass
class CausalSemanticMatch:
    semantic_text: str
    logic_text: str
    score: float
    polarity: str
    source_score: float


@dataclass
class PremiseCoupling:
    left: str
    right: str
    operation: str
    coupled_text: str
    semantic_logic_probability: float
    target_probability: float
    contradiction_probability: float
    answer_distribution: dict[str, float]


@dataclass
class Stage2Candidate:
    answer: str
    explanation: str
    source: str
    confidence: float
    option_scores: list[Stage2OptionScore]
    causal_inference: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalTransition:
    step: int
    rule_id: str
    action: str
    rule_source: str
    cause_variables: list[str]
    effect_variable: str
    produced: LogicAtom
    antecedents: list[str]
    target_match: float
    contradiction: float
    intervention_strength: float
    causal_effect_score: float
    semantic_logic_score: float
    semantic_matches: list[CausalSemanticMatch]
    answer_distribution: dict[str, float]
    premise_couplings: list[PremiseCoupling]


@dataclass
class CausalTrajectory:
    answer_signal: str
    causal_score: float
    answer_distribution: dict[str, float]
    target_match_score: float
    semantic_logic_score: float
    causal_effect_score: float
    rule_validity_score: float
    premise_coverage_score: float
    contradiction_penalty: float
    missing_condition_penalty: float
    counterfactual_gap: float
    proof_depth: int | None
    evidence: list[str]
    transitions: list[CausalTransition]
    premise_couplings: list[PremiseCoupling]


@dataclass
class CausalBeliefState:
    round_index: int
    selected_source: str
    selected_action: str
    answer_distribution: dict[str, float]
    confidence: float
    delta: float
    causal_score: float


@dataclass
class CausalInferenceResult:
    answer_signal: str
    causal_score: float
    answer_distribution: dict[str, float]
    target_match_score: float
    semantic_logic_score: float
    causal_effect_score: float
    rule_validity_score: float
    premise_coverage_score: float
    contradiction_penalty: float
    missing_condition_penalty: float
    counterfactual_gap: float
    proof_depth: int | None
    evidence: list[str]
    trajectories: list[CausalTrajectory]
    premise_couplings: list[PremiseCoupling]
    belief_states: list[CausalBeliefState]


@dataclass
class Type1PipelineConfig:
    stage0: SemanticHybridConfig = field(default_factory=SemanticHybridConfig)
    max_forward_steps: int = 8
    stage2_confidence_threshold: float = 0.58
    stage3_pass_threshold: float = 0.66
    include_evidence_graph: bool = False
    enable_causal_world_model: bool = True
    causal_max_steps: int = 6
    causal_top_k: int = 4
    causal_yes_threshold: float = 0.35
    causal_no_threshold: float = -0.35
    causal_yes_probability_threshold: float = 0.45
    causal_no_probability_threshold: float = 0.45
    premise_coupling_top_k: int = 8
    recurrent_planning_rounds: int = 4
    belief_confidence_threshold: float = 0.72
    belief_convergence_delta: float = 0.035
    enable_global_option_distribution: bool = True
    global_option_distribution_weight: float = 0.44
    option_distribution_temperature: float = 0.36
    premise_alignment_top_k: int = 6
    enable_transformer_brain_world_model: bool = True
    transformer_brain_hidden_dim: int = 32
    transformer_brain_imagination_layers: int = 2
    transformer_brain_attention_heads: int = 4
    transformer_brain_frame_local_window: int = 4
    transformer_brain_ssm_block_size: int = 4
    transformer_brain_allow_override: bool = False
    transformer_brain_temperature: float = 0.52
    transformer_brain_override_margin: float = 0.15
    transformer_brain_minimum_override_confidence: float = 0.45
    transformer_brain_minimum_override_winner_margin: float = 0.08


class FormulaParser:
    """Small parser for the project FOL strings."""

    def parse_premises(self, premises_fol: list[str], premises_nl: list[str]) -> tuple[list[EvidenceFact], list[EvidenceRule]]:
        facts: list[EvidenceFact] = []
        rules: list[EvidenceRule] = []
        for idx, raw in enumerate(premises_fol):
            formula = self._clean_formula(raw)
            source = TextTools.safe_get(premises_nl, idx, formula)
            if self._has_implication(formula):
                rule = self._parse_rule(formula, source, idx, f"r{len(rules)}")
                if rule is not None:
                    rules.append(rule)
                    contrapositive = self._contrapositive(rule, f"r{len(rules)}")
                    if contrapositive is not None:
                        rules.append(contrapositive)
                continue
            for atom in self._parse_atoms(formula):
                facts.append(EvidenceFact(atom=atom, source=source, source_index=idx, support_chain=[source]))
        return facts, rules

    def _clean_formula(self, value: Any) -> str:
        text = TextTools.clean(value).replace("�", "")
        text = text.replace("ForAll", "forall").replace("forall", "∀")
        text = text.replace("∧", " and ").replace("&", " and ")
        text = text.replace("¬", "¬ ")
        return text

    def _has_implication(self, formula: str) -> bool:
        return bool(IMPLICATION_RE.search(formula))

    def _parse_rule(self, formula: str, source: str, source_index: int, rule_id: str) -> EvidenceRule | None:
        inner = self._strip_quantifiers(formula)
        parts = IMPLICATION_RE.split(inner, maxsplit=1)
        if len(parts) != 2:
            return None
        antecedents = self._parse_atoms(parts[0])
        consequents = self._parse_atoms(parts[1])
        if not consequents:
            return None
        return EvidenceRule(rule_id=rule_id, antecedents=antecedents, consequents=consequents, source=source, source_index=source_index)

    def _strip_quantifiers(self, formula: str) -> str:
        text = formula.strip()
        while text.startswith("∀"):
            if text.startswith("∀("):
                open_idx = text.find("(")
                close_idx = self._matching_close(text, open_idx)
                if close_idx == len(text) - 1:
                    inner = text[open_idx + 1 : close_idx].strip()
                    comma = self._top_level_comma(inner)
                    if comma >= 0:
                        text = inner[comma + 1 :].strip()
                        continue
                break
            match = re.match(r"∀\s*[A-Za-z_][A-Za-z0-9_]*\s*\((.*)\)\s*$", text)
            if not match:
                break
            text = match.group(1).strip()
        return self._strip_outer_parens(text)

    def _top_level_comma(self, text: str) -> int:
        depth = 0
        for idx, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                return idx
        return -1

    def _strip_outer_parens(self, text: str) -> str:
        text = text.strip()
        while text.startswith("("):
            close_idx = self._matching_close(text, 0)
            if close_idx != len(text) - 1:
                break
            text = text[1:-1].strip()
        return text

    def _matching_close(self, text: str, open_idx: int) -> int:
        if open_idx < 0:
            return -1
        depth = 0
        for idx in range(open_idx, len(text)):
            if text[idx] == "(":
                depth += 1
            elif text[idx] == ")":
                depth -= 1
                if depth == 0:
                    return idx
        return -1

    def _parse_atoms(self, text: str) -> list[LogicAtom]:
        atoms: list[LogicAtom] = []
        for match in ATOM_RE.finditer(text):
            atom = self._parse_atom(match.group(0))
            if atom is not None:
                atoms.append(atom)
        return atoms

    def _parse_atom(self, text: str) -> LogicAtom | None:
        raw = TextTools.clean(text)
        negated = raw.startswith("¬") or raw.lower().startswith("not ")
        raw = raw.replace("¬", "").strip()
        if raw.lower().startswith("not "):
            raw = raw[4:].strip()
        pred_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)\s*(.*)$", raw)
        if not pred_match:
            return None
        predicate = self._norm_token(pred_match.group(1))
        args = tuple(self._norm_token(arg) for arg in pred_match.group(2).split(",") if arg.strip())
        tail = pred_match.group(3).strip()
        operator = ""
        value = ""
        if tail:
            op_match = re.match(r"(=|>=|<=|>|<|≥|≤)\s*(-?[A-Za-z0-9_.]+)", tail)
            if op_match:
                operator = op_match.group(1).replace("≥", ">=").replace("≤", "<=")
                value = self._norm_token(op_match.group(2))
        return LogicAtom(predicate=predicate, args=args, negated=negated, operator=operator, value=value)

    def _norm_token(self, token: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", token.strip().lower()).strip("_")

    def _contrapositive(self, rule: EvidenceRule, rule_id: str) -> EvidenceRule | None:
        if len(rule.antecedents) != 1 or len(rule.consequents) != 1:
            return None
        ant = rule.antecedents[0]
        cons = rule.consequents[0]
        if ant.operator or cons.operator:
            return None
        return EvidenceRule(
            rule_id=rule_id,
            antecedents=[LogicAtom(cons.predicate, cons.args, not cons.negated)],
            consequents=[LogicAtom(ant.predicate, ant.args, not ant.negated)],
            source=f"Contrapositive of: {rule.source}",
            source_index=rule.source_index,
            derived=True,
        )


class ForwardChainer:
    def __init__(self, max_steps: int = 8) -> None:
        self.max_steps = max_steps

    def run(self, facts: list[EvidenceFact], rules: list[EvidenceRule]) -> list[EvidenceFact]:
        known: dict[str, EvidenceFact] = {fact.atom.canonical(): fact for fact in facts}
        derived: list[EvidenceFact] = []
        for _ in range(self.max_steps):
            changed = False
            all_facts = list(known.values())
            for rule in rules:
                bindings = self._satisfy(rule.antecedents, all_facts)
                for binding, support in bindings:
                    depth = 1 + max((item.depth for item in support), default=0)
                    for consequent in rule.consequents:
                        atom = self._instantiate(consequent, binding)
                        key = atom.canonical()
                        if key in known:
                            continue
                        chain = [item.atom.lexical_text() for item in support]
                        chain.append(rule.source)
                        fact = EvidenceFact(atom=atom, source=rule.source, source_index=rule.source_index, depth=depth, support_chain=chain)
                        known[key] = fact
                        derived.append(fact)
                        changed = True
            if not changed:
                break
        return derived

    def _satisfy(self, antecedents: list[LogicAtom], facts: list[EvidenceFact]) -> list[tuple[dict[str, str], list[EvidenceFact]]]:
        states: list[tuple[dict[str, str], list[EvidenceFact]]] = [({}, [])]
        for antecedent in antecedents:
            next_states: list[tuple[dict[str, str], list[EvidenceFact]]] = []
            for binding, support in states:
                for fact in facts:
                    new_binding = self._match(antecedent, fact.atom, dict(binding))
                    if new_binding is not None:
                        next_states.append((new_binding, support + [fact]))
            states = next_states
            if not states:
                break
        return states

    def _match(self, pattern: LogicAtom, fact: LogicAtom, binding: dict[str, str]) -> dict[str, str] | None:
        if pattern.predicate != fact.predicate or pattern.negated != fact.negated:
            return None
        if pattern.operator:
            return self._match_numeric(pattern, fact, binding)
        if fact.operator:
            return None
        if len(pattern.args) != len(fact.args):
            return None
        for p_arg, f_arg in zip(pattern.args, fact.args):
            if p_arg in VARIABLES:
                existing = binding.get(p_arg)
                if existing is not None and existing != f_arg:
                    return None
                binding[p_arg] = f_arg
            elif f_arg in {"x", "*"}:
                continue
            elif p_arg != f_arg:
                return None
        return binding

    def _match_numeric(self, pattern: LogicAtom, fact: LogicAtom, binding: dict[str, str]) -> dict[str, str] | None:
        if pattern.predicate != fact.predicate or not fact.operator:
            return None
        if len(pattern.args) != len(fact.args):
            return None
        try:
            fact_value = float(fact.value)
            pattern_value = float(pattern.value)
        except ValueError:
            return None
        if not self._compare(fact_value, pattern.operator, pattern_value):
            return None
        for p_arg, f_arg in zip(pattern.args, fact.args):
            if p_arg in VARIABLES:
                binding[p_arg] = f_arg
            elif p_arg != f_arg:
                return None
        return binding

    def _compare(self, left: float, operator: str, right: float) -> bool:
        if operator == "=":
            return left == right
        if operator == ">=":
            return left >= right
        if operator == "<=":
            return left <= right
        if operator == ">":
            return left > right
        if operator == "<":
            return left < right
        return False

    def _instantiate(self, atom: LogicAtom, binding: dict[str, str]) -> LogicAtom:
        args = tuple(binding.get(arg, arg) for arg in atom.args)
        return LogicAtom(atom.predicate, args, atom.negated, atom.operator, atom.value)

    def _has_variables(self, atom: LogicAtom) -> bool:
        return any(arg in VARIABLES for arg in atom.args)


class Type1EvidenceGraphBuilder:
    def __init__(self, max_forward_steps: int = 8) -> None:
        self.parser = FormulaParser()
        self.chainer = ForwardChainer(max_forward_steps)

    def build(self, stage_input: Stage0Input) -> EvidenceGraph:
        facts, rules = self.parser.parse_premises(stage_input.premises_fol, stage_input.premises_nl)
        derived = self.chainer.run(facts, rules)
        return EvidenceGraph(facts=facts, rules=rules, derived_facts=derived)


class Type1CausalInferenceWorldModel:
    """Causal world model for action-effect inference and planning.

    The model treats facts as causal state variables and rules as text actions
    or interventions. Applying a rule corresponds to do(action), producing
    effect variables that can be compared with the query target.
    """

    def __init__(self, config: Type1PipelineConfig) -> None:
        self.config = config
        self.chainer = ForwardChainer(config.causal_max_steps)

    def infer(
        self,
        text: str,
        graph: EvidenceGraph,
        semantic_matches: list[dict[str, Any]] | None = None,
    ) -> CausalInferenceResult:
        query_tokens = self._tokens(text)
        if not query_tokens:
            return self._result("Uncertain")

        negative_query = self._is_negative(text)
        semantic_units = self._semantic_units(text, semantic_matches or [])
        known: dict[str, EvidenceFact] = {fact.atom.canonical(): fact for fact in graph.facts}
        premise_couplings = self._premise_couplings(query_tokens, negative_query, graph, semantic_units)
        trajectories = self._fact_seed_trajectories(query_tokens, negative_query, graph, semantic_units, premise_couplings)
        transitions_by_atom: dict[str, list[CausalTransition]] = {}

        for step in range(1, self.config.causal_max_steps + 1):
            changed = False
            all_facts = list(known.values())
            for rule in graph.rules:
                bindings = self.chainer._satisfy(rule.antecedents, all_facts)
                for binding, support in bindings:
                    for consequent in rule.consequents:
                        atom = self.chainer._instantiate(consequent, binding)
                        key = atom.canonical()
                        if key in known:
                            continue
                        support_text = [item.atom.lexical_text() for item in support]
                        chain = support_text + [rule.source]
                        fact = EvidenceFact(
                            atom=atom,
                            source=rule.source,
                            source_index=rule.source_index,
                            depth=step,
                            support_chain=chain,
                        )
                        known[key] = fact
                        changed = True
                        semantic_result = self._semantic_logic_score(
                            query_tokens=query_tokens,
                            logic_text=f"{self._action_text(rule)} {atom.lexical_text()}",
                            semantic_units=semantic_units,
                            negated=atom.negated,
                            negative_query=negative_query,
                        )
                        transition = CausalTransition(
                            step=step,
                            rule_id=rule.rule_id,
                            action=self._action_text(rule),
                            rule_source=rule.source,
                            cause_variables=[atom.lexical_text() for atom in rule.antecedents],
                            effect_variable=atom.lexical_text(),
                            produced=atom,
                            antecedents=support_text,
                            target_match=self._atom_target_match(query_tokens, atom),
                            contradiction=self._contradiction_score(query_tokens, negative_query, atom),
                            intervention_strength=self._intervention_strength(rule, support),
                            causal_effect_score=self._causal_effect_score(query_tokens, rule, atom, support),
                            semantic_logic_score=semantic_result["score"],
                            semantic_matches=semantic_result["matches"],
                            answer_distribution=self._answer_distribution(
                                target_match=self._atom_target_match(query_tokens, atom),
                                semantic_logic=semantic_result["score"],
                                causal_effect=self._causal_effect_score(query_tokens, rule, atom, support),
                                contradiction=self._contradiction_score(query_tokens, negative_query, atom),
                                missing_penalty=0.0,
                                counterfactual_gap=0.0,
                                premise_couplings=premise_couplings,
                            ),
                            premise_couplings=premise_couplings[: self.config.causal_top_k],
                        )
                        transitions_by_atom[key] = transitions_by_atom.get(key, []) + [transition]
                        trajectories.append(self._trajectory_from_fact(query_tokens, negative_query, semantic_units, premise_couplings, fact, transitions_by_atom[key]))
            if not changed:
                break

        rule_gap = self._missing_condition_penalty(query_tokens, list(known.values()), graph.rules)
        best = self._select_best_trajectory(trajectories, rule_gap)
        if best is None:
            distribution = self._distribution_from_logits(yes=0.0, uncertain=0.8, no=rule_gap)
            return self._result(
                "Uncertain",
                causal_score=round(distribution["Yes"] - distribution["No"], 4),
                missing_condition_penalty=rule_gap,
                answer_distribution=distribution,
                premise_couplings=premise_couplings,
                belief_states=[
                    CausalBeliefState(
                        round_index=0,
                        selected_source="no_trajectory",
                        selected_action="stop",
                        answer_distribution=distribution,
                        confidence=max(distribution.values()),
                        delta=0.0,
                        causal_score=round(distribution["Yes"] - distribution["No"], 4),
                    )
                ],
            )

        distribution, belief_states = self._recurrent_plan(trajectories, premise_couplings, rule_gap)
        best.answer_distribution = distribution
        best.causal_score = round(distribution["Yes"] - distribution["No"], 4)
        if distribution.get("Yes", 0.0) >= self.config.causal_yes_probability_threshold and best.missing_condition_penalty < 0.65:
            answer_signal = "Yes"
        elif distribution.get("No", 0.0) >= self.config.causal_no_probability_threshold or rule_gap >= 0.75:
            answer_signal = "No"
        else:
            answer_signal = "Uncertain"

        top = sorted(trajectories, key=lambda item: item.causal_score, reverse=True)[: self.config.causal_top_k]
        return CausalInferenceResult(
            answer_signal=answer_signal,
            causal_score=round(best.causal_score, 4),
            answer_distribution=distribution,
            target_match_score=round(best.target_match_score, 4),
            semantic_logic_score=round(best.semantic_logic_score, 4),
            causal_effect_score=round(best.causal_effect_score, 4),
            rule_validity_score=round(best.rule_validity_score, 4),
            premise_coverage_score=round(best.premise_coverage_score, 4),
            contradiction_penalty=round(best.contradiction_penalty, 4),
            missing_condition_penalty=round(max(best.missing_condition_penalty, rule_gap), 4),
            counterfactual_gap=round(best.counterfactual_gap, 4),
            proof_depth=best.proof_depth,
            evidence=best.evidence,
            trajectories=top,
            premise_couplings=premise_couplings,
            belief_states=belief_states,
        )

    def _fact_seed_trajectories(
        self,
        query_tokens: set[str],
        negative_query: bool,
        graph: EvidenceGraph,
        semantic_units: list[dict[str, Any]],
        premise_couplings: list[PremiseCoupling],
    ) -> list[CausalTrajectory]:
        trajectories: list[CausalTrajectory] = []
        for fact in graph.facts + graph.derived_facts:
            target_match = self._atom_target_match(query_tokens, fact.atom)
            contradiction = self._contradiction_score(query_tokens, negative_query, fact.atom)
            if max(target_match, contradiction) < 0.42:
                continue
            trajectories.append(self._trajectory_from_fact(query_tokens, negative_query, semantic_units, premise_couplings, fact, []))
        return trajectories

    def _trajectory_from_fact(
        self,
        query_tokens: set[str],
        negative_query: bool,
        semantic_units: list[dict[str, Any]],
        premise_couplings: list[PremiseCoupling],
        fact: EvidenceFact,
        transitions: list[CausalTransition],
    ) -> CausalTrajectory:
        target_match = self._atom_target_match(query_tokens, fact.atom)
        contradiction = self._contradiction_score(query_tokens, negative_query, fact.atom)
        semantic_result = self._semantic_logic_score(
            query_tokens=query_tokens,
            logic_text=fact.atom.lexical_text(),
            semantic_units=semantic_units,
            negated=fact.atom.negated,
            negative_query=negative_query,
        )
        semantic_logic = max(
            [semantic_result["score"], *[transition.semantic_logic_score for transition in transitions]],
            key=abs,
        )
        causal_effect = max((transition.causal_effect_score for transition in transitions), default=0.0)
        rule_validity = 0.92 if fact.depth > 0 else 0.72
        premise_coverage = min(1.0, 0.42 + 0.18 * len(fact.support_chain)) if fact.support_chain else 0.35
        missing_penalty = 0.0 if fact.depth > 0 else max(0.0, 0.28 - target_match * 0.2)
        counterfactual_gap = max(0.0, target_match - causal_effect)
        positive = (
            target_match * 0.28
            + causal_effect * 0.2
            + max(0.0, semantic_logic) * 0.3
            + rule_validity * 0.12
            + premise_coverage * 0.1
        )
        negative = (
            contradiction * 0.36
            + missing_penalty * 0.18
            + counterfactual_gap * 0.12
            + max(0.0, -semantic_logic) * 0.34
        )
        answer_distribution = self._answer_distribution(
            target_match=target_match,
            semantic_logic=semantic_logic,
            causal_effect=causal_effect,
            contradiction=contradiction,
            missing_penalty=missing_penalty,
            counterfactual_gap=counterfactual_gap,
            premise_couplings=premise_couplings,
        )
        causal_score = max(-1.0, min(1.0, answer_distribution["Yes"] - answer_distribution["No"]))
        if answer_distribution["Yes"] >= self.config.causal_yes_probability_threshold:
            answer_signal = "Yes"
        elif answer_distribution["No"] >= self.config.causal_no_probability_threshold:
            answer_signal = "No"
        else:
            answer_signal = "Uncertain"
        return CausalTrajectory(
            answer_signal=answer_signal,
            causal_score=round(causal_score, 4),
            answer_distribution=answer_distribution,
            target_match_score=round(target_match, 4),
            semantic_logic_score=round(semantic_logic, 4),
            causal_effect_score=round(causal_effect, 4),
            rule_validity_score=round(rule_validity, 4),
            premise_coverage_score=round(premise_coverage, 4),
            contradiction_penalty=round(contradiction, 4),
            missing_condition_penalty=round(missing_penalty, 4),
            counterfactual_gap=round(counterfactual_gap, 4),
            proof_depth=fact.depth,
            evidence=fact.support_chain[:5] or [fact.source],
            transitions=transitions[-self.config.causal_top_k :],
            premise_couplings=premise_couplings[: self.config.causal_top_k],
        )

    def _select_best_trajectory(
        self,
        trajectories: list[CausalTrajectory],
        rule_gap: float,
    ) -> CausalTrajectory | None:
        if not trajectories:
            return None
        ranked = sorted(
            trajectories,
            key=lambda item: (
                max(item.answer_distribution.values(), default=0.0),
                item.answer_distribution.get("Yes", 0.0) - item.answer_distribution.get("No", 0.0),
                item.target_match_score,
            ),
            reverse=True,
        )
        best = ranked[0]
        best.missing_condition_penalty = round(max(best.missing_condition_penalty, rule_gap), 4)
        if rule_gap:
            adjusted_no = min(1.0, best.answer_distribution["No"] + rule_gap * 0.16)
            adjusted_yes = max(0.0, best.answer_distribution["Yes"] - rule_gap * 0.12)
            best.answer_distribution = self._normalize_distribution(
                {"Yes": adjusted_yes, "Uncertain": best.answer_distribution["Uncertain"], "No": adjusted_no}
            )
        best.causal_score = round(best.answer_distribution["Yes"] - best.answer_distribution["No"], 4)
        return best

    def _recurrent_plan(
        self,
        trajectories: list[CausalTrajectory],
        premise_couplings: list[PremiseCoupling],
        rule_gap: float,
    ) -> tuple[dict[str, float], list[CausalBeliefState]]:
        candidates = self._planning_candidates(trajectories, premise_couplings)
        if not candidates:
            distribution = self._distribution_from_logits(yes=0.0, uncertain=0.8, no=rule_gap)
            return distribution, [
                CausalBeliefState(
                    round_index=0,
                    selected_source="empty_planning_space",
                    selected_action="stop",
                    answer_distribution=distribution,
                    confidence=max(distribution.values()),
                    delta=0.0,
                    causal_score=round(distribution["Yes"] - distribution["No"], 4),
                )
            ]

        current = {"Yes": 1 / 3, "Uncertain": 1 / 3, "No": 1 / 3}
        states: list[CausalBeliefState] = []
        used: set[str] = set()
        previous_confidence = max(current.values())

        for round_index in range(1, self.config.recurrent_planning_rounds + 1):
            candidate = self._choose_planning_candidate(candidates, used, current)
            if candidate is None:
                break
            used.add(candidate["id"])
            candidate_distribution = candidate["distribution"]
            confidence = max(candidate_distribution.values())
            alpha = min(0.68, 0.28 + confidence * 0.38)
            updated = self._blend_distributions(current, candidate_distribution, alpha)
            if rule_gap:
                updated = self._normalize_distribution(
                    {
                        "Yes": max(0.0, updated["Yes"] - rule_gap * 0.08),
                        "Uncertain": updated["Uncertain"],
                        "No": min(1.0, updated["No"] + rule_gap * 0.12),
                    }
                )
            delta = self._distribution_delta(current, updated)
            current = updated
            current_confidence = max(current.values())
            states.append(
                CausalBeliefState(
                    round_index=round_index,
                    selected_source=candidate["source"],
                    selected_action=candidate["action"],
                    answer_distribution=current,
                    confidence=round(current_confidence, 4),
                    delta=round(delta, 4),
                    causal_score=round(current["Yes"] - current["No"], 4),
                )
            )
            if current_confidence >= self.config.belief_confidence_threshold and delta <= self.config.belief_convergence_delta:
                break
            if current_confidence < previous_confidence and delta < self.config.belief_convergence_delta:
                break
            previous_confidence = current_confidence

        return current, states

    def _planning_candidates(
        self,
        trajectories: list[CausalTrajectory],
        premise_couplings: list[PremiseCoupling],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for idx, trajectory in enumerate(trajectories):
            confidence = max(trajectory.answer_distribution.values(), default=0.0)
            action = trajectory.transitions[-1].action if trajectory.transitions else "seed_fact"
            candidates.append(
                {
                    "id": f"trajectory:{idx}",
                    "source": "trajectory",
                    "action": action,
                    "distribution": trajectory.answer_distribution,
                    "confidence": confidence,
                    "target": trajectory.target_match_score,
                    "direction": trajectory.answer_distribution.get("Yes", 0.0) - trajectory.answer_distribution.get("No", 0.0),
                }
            )
        for idx, coupling in enumerate(premise_couplings):
            confidence = max(coupling.answer_distribution.values(), default=0.0)
            candidates.append(
                {
                    "id": f"coupling:{idx}",
                    "source": "premise_coupling",
                    "action": f"{coupling.operation}: {coupling.coupled_text[:120]}",
                    "distribution": coupling.answer_distribution,
                    "confidence": confidence,
                    "target": coupling.target_probability,
                    "direction": coupling.answer_distribution.get("Yes", 0.0) - coupling.answer_distribution.get("No", 0.0),
                }
            )
        candidates.sort(key=lambda item: (item["confidence"], item["target"], abs(item["direction"])), reverse=True)
        return candidates

    def _choose_planning_candidate(
        self,
        candidates: list[dict[str, Any]],
        used: set[str],
        current: dict[str, float],
    ) -> dict[str, Any] | None:
        current_direction = current.get("Yes", 0.0) - current.get("No", 0.0)
        current_uncertainty = current.get("Uncertain", 0.0)
        best: dict[str, Any] | None = None
        best_score = float("-inf")
        for candidate in candidates:
            if candidate["id"] in used:
                continue
            candidate_direction = candidate["direction"]
            explores = current_uncertainty * candidate["confidence"]
            refines = (1.0 - abs(current_direction - candidate_direction)) * 0.35
            target = candidate["target"] * 0.25
            score = explores + refines + target + abs(candidate_direction) * 0.2
            if score > best_score:
                best = candidate
                best_score = score
        return best

    def _blend_distributions(
        self,
        left: dict[str, float],
        right: dict[str, float],
        alpha: float,
    ) -> dict[str, float]:
        return self._normalize_distribution(
            {
                key: left.get(key, 0.0) * (1.0 - alpha) + right.get(key, 0.0) * alpha
                for key in ["Yes", "Uncertain", "No"]
            }
        )

    def _distribution_delta(self, left: dict[str, float], right: dict[str, float]) -> float:
        return sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in ["Yes", "Uncertain", "No"])

    def _missing_condition_penalty(
        self,
        query_tokens: set[str],
        facts: list[EvidenceFact],
        rules: list[EvidenceRule],
    ) -> float:
        best_gap = 0.0
        for rule in rules:
            target_overlap = max((self._atom_target_match(query_tokens, atom) for atom in rule.consequents), default=0.0)
            if target_overlap < 0.45:
                continue
            if not rule.antecedents:
                continue
            satisfied = sum(1 for antecedent in rule.antecedents if self._antecedent_satisfied(antecedent, facts))
            completeness = satisfied / len(rule.antecedents)
            best_gap = max(best_gap, (1.0 - completeness) * target_overlap)
        return round(best_gap, 4)

    def _antecedent_satisfied(self, antecedent: LogicAtom, facts: list[EvidenceFact]) -> bool:
        for fact in facts:
            if self.chainer._match(antecedent, fact.atom, {}) is not None:
                return True
        return False

    def _action_text(self, rule: EvidenceRule) -> str:
        causes = ", ".join(atom.lexical_text() for atom in rule.antecedents) or "state prior"
        effects = ", ".join(atom.lexical_text() for atom in rule.consequents) or "unknown effect"
        return f"do({causes}) -> {effects}"

    def _intervention_strength(self, rule: EvidenceRule, support: list[EvidenceFact]) -> float:
        if not rule.antecedents:
            return 1.0
        return round(min(1.0, len(support) / max(len(rule.antecedents), 1)), 4)

    def _causal_effect_score(
        self,
        query_tokens: set[str],
        rule: EvidenceRule,
        produced: LogicAtom,
        support: list[EvidenceFact],
    ) -> float:
        target_match = self._atom_target_match(query_tokens, produced)
        intervention_strength = self._intervention_strength(rule, support)
        specificity = 0.82 if produced.args and any(arg not in VARIABLES for arg in produced.args) else 0.62
        action_relevance = max((self._atom_target_match(query_tokens, atom) for atom in rule.antecedents), default=0.0)
        return round(min(1.0, target_match * 0.55 + intervention_strength * 0.25 + specificity * 0.12 + action_relevance * 0.08), 4)

    def _premise_couplings(
        self,
        query_tokens: set[str],
        negative_query: bool,
        graph: EvidenceGraph,
        semantic_units: list[dict[str, Any]],
    ) -> list[PremiseCoupling]:
        premise_texts: list[str] = []
        for fact in graph.facts + graph.derived_facts:
            source = TextTools.clean(fact.source)
            premise_texts.append(f"{fact.atom.lexical_text()} :: {source}" if source else fact.atom.lexical_text())
        for rule in graph.rules:
            left = " and ".join(atom.lexical_text() for atom in rule.antecedents)
            right = " and ".join(atom.lexical_text() for atom in rule.consequents)
            source = TextTools.clean(rule.source)
            symbolic = f"{left} implies {right}".strip()
            premise_texts.append(f"{symbolic} :: {source}" if source else symbolic)
        premise_texts = [text for text in dict.fromkeys(premise_texts) if text][:24]

        couplings: list[PremiseCoupling] = []
        for i, left in enumerate(premise_texts):
            for right in premise_texts[i + 1 :]:
                for operation, coupled_text in self._coupling_operations(left, right):
                    logic_tokens = self._tokens(coupled_text)
                    if not logic_tokens:
                        continue
                    semantic_probability = self._coupling_semantic_probability(logic_tokens, semantic_units)
                    target_probability = len(query_tokens & logic_tokens) / max(len(query_tokens | logic_tokens), 1)
                    contradiction_probability = self._coupling_contradiction_probability(
                        negative_query=negative_query,
                        left=left,
                        right=right,
                        target_probability=target_probability,
                    )
                    distribution = self._distribution_from_logits(
                        yes=target_probability * 1.2 + semantic_probability * 0.8,
                        uncertain=(1.0 - target_probability) * 0.55 + abs(semantic_probability - 0.5) * 0.15,
                        no=contradiction_probability * 1.25,
                    )
                    couplings.append(
                        PremiseCoupling(
                            left=left,
                            right=right,
                            operation=operation,
                            coupled_text=coupled_text,
                            semantic_logic_probability=round(semantic_probability, 4),
                            target_probability=round(target_probability, 4),
                            contradiction_probability=round(contradiction_probability, 4),
                            answer_distribution=distribution,
                        )
                    )
        couplings.sort(
            key=lambda item: (
                max(item.answer_distribution.values(), default=0.0),
                item.target_probability,
                item.semantic_logic_probability,
            ),
            reverse=True,
        )
        return couplings[: self.config.premise_coupling_top_k]

    def _coupling_operations(self, left: str, right: str) -> list[tuple[str, str]]:
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)
        shared = " ".join(sorted(left_tokens & right_tokens))
        union = " ".join(sorted(left_tokens | right_tokens))
        return [
            ("intersection", shared or f"{left} {right}"),
            ("union", union or f"{left} {right}"),
            ("ordered_pair", f"{left} => {right}"),
            ("reverse_pair", f"{right} => {left}"),
        ]

    def _coupling_semantic_probability(self, logic_tokens: set[str], semantic_units: list[dict[str, Any]]) -> float:
        best = 0.0
        for unit in semantic_units:
            semantic_tokens = self._tokens(str(unit.get("text", "")))
            if not semantic_tokens:
                continue
            overlap = len(semantic_tokens & logic_tokens) / max(len(semantic_tokens | logic_tokens), 1)
            try:
                source_score = float(unit.get("score", 1.0) or 0.0)
            except (TypeError, ValueError):
                source_score = 0.0
            best = max(best, overlap * max(0.0, min(1.0, source_score)))
        return max(0.0, min(1.0, best))

    def _coupling_contradiction_probability(
        self,
        *,
        negative_query: bool,
        left: str,
        right: str,
        target_probability: float,
    ) -> float:
        left_negative = self._is_negative(left)
        right_negative = self._is_negative(right)
        if left_negative != right_negative:
            return min(1.0, 0.35 + target_probability * 0.5)
        if negative_query != (left_negative or right_negative) and target_probability > 0.2:
            return min(1.0, target_probability * 0.65)
        return 0.0

    def _answer_distribution(
        self,
        *,
        target_match: float,
        semantic_logic: float,
        causal_effect: float,
        contradiction: float,
        missing_penalty: float,
        counterfactual_gap: float,
        premise_couplings: list[PremiseCoupling],
    ) -> dict[str, float]:
        coupling_yes = max((item.answer_distribution.get("Yes", 0.0) for item in premise_couplings), default=0.0)
        coupling_uncertain = max((item.answer_distribution.get("Uncertain", 0.0) for item in premise_couplings), default=0.0)
        coupling_no = max((item.answer_distribution.get("No", 0.0) for item in premise_couplings), default=0.0)
        return self._distribution_from_logits(
            yes=target_match * 1.25 + causal_effect * 0.9 + max(0.0, semantic_logic) * 1.1 + coupling_yes * 0.45,
            uncertain=0.45 + (1.0 - target_match) * 0.35 + missing_penalty * 0.45 + coupling_uncertain * 0.25,
            no=contradiction * 1.15 + max(0.0, -semantic_logic) * 1.05 + missing_penalty * 0.65 + counterfactual_gap * 0.45 + coupling_no * 0.45,
        )

    def _distribution_from_logits(self, *, yes: float, uncertain: float, no: float) -> dict[str, float]:
        logits = {"Yes": yes, "Uncertain": uncertain, "No": no}
        max_logit = max(logits.values())
        exps = {key: math.exp(value - max_logit) for key, value in logits.items()}
        total = sum(exps.values()) or 1.0
        return {key: round(value / total, 4) for key, value in exps.items()}

    def _normalize_distribution(self, distribution: dict[str, float]) -> dict[str, float]:
        clean = {key: max(0.0, float(distribution.get(key, 0.0) or 0.0)) for key in ["Yes", "Uncertain", "No"]}
        total = sum(clean.values()) or 1.0
        return {key: round(value / total, 4) for key, value in clean.items()}

    def _semantic_units(self, query_text: str, semantic_matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = [
            {"text": query_text, "score": 1.0, "source": "query"}
        ]
        for match in semantic_matches[:12]:
            text = TextTools.clean(match.get("text", ""))
            if not text:
                continue
            try:
                source_score = float(match.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                source_score = 0.0
            units.append({"text": text, "score": max(0.0, min(1.0, source_score)), "source": match.get("source", "semantic_match")})
        return units

    def _semantic_logic_score(
        self,
        *,
        query_tokens: set[str],
        logic_text: str,
        semantic_units: list[dict[str, Any]],
        negated: bool,
        negative_query: bool,
    ) -> dict[str, Any]:
        logic_tokens = self._tokens(logic_text)
        best: list[CausalSemanticMatch] = []
        for unit in semantic_units:
            semantic_text = str(unit.get("text", ""))
            semantic_tokens = self._tokens(semantic_text)
            if not semantic_tokens or not logic_tokens:
                continue
            logic_overlap = len(semantic_tokens & logic_tokens) / max(len(logic_tokens), 1)
            query_overlap = len(semantic_tokens & query_tokens) / max(len(query_tokens), 1)
            raw = (logic_overlap * 0.62 + query_overlap * 0.38) * float(unit.get("score", 1.0) or 0.0)
            semantic_negative = self._is_negative(semantic_text)
            polarity = "support"
            if semantic_negative != negated:
                raw *= -0.72
                polarity = "contradict"
            if negative_query != negated and query_overlap >= 0.18:
                raw *= -1.0
                polarity = "query_contradict"
            score = max(-1.0, min(1.0, raw))
            if abs(score) < 0.08:
                continue
            best.append(
                CausalSemanticMatch(
                    semantic_text=semantic_text,
                    logic_text=logic_text,
                    score=round(score, 4),
                    polarity=polarity,
                    source_score=round(float(unit.get("score", 1.0) or 0.0), 4),
                )
            )
        best.sort(key=lambda item: abs(item.score), reverse=True)
        selected = best[: self.config.causal_top_k]
        if not selected:
            return {"score": 0.0, "matches": []}
        return {
            "score": round(selected[0].score, 4),
            "matches": selected,
        }

    def _atom_target_match(self, query_tokens: set[str], atom: LogicAtom) -> float:
        atom_tokens = {token for token in self._tokens(atom.lexical_text()) if token not in VARIABLES}
        if not atom_tokens:
            return 0.0
        atom_coverage = len(query_tokens & atom_tokens) / len(atom_tokens)
        query_coverage = len(query_tokens & atom_tokens) / max(len(query_tokens), 1)
        predicate_bonus = 0.12 if query_tokens & set(atom.predicate.split("_")) else 0.0
        return min(1.0, atom_coverage * 0.68 + query_coverage * 0.2 + predicate_bonus)

    def _contradiction_score(self, query_tokens: set[str], negative_query: bool, atom: LogicAtom) -> float:
        if not self._predicate_mentions(query_tokens, atom):
            return 0.0
        if negative_query == atom.negated:
            return 0.0
        return min(1.0, self._atom_target_match(query_tokens, atom) * 0.82)

    def _predicate_mentions(self, query_tokens: set[str], atom: LogicAtom) -> bool:
        return bool(query_tokens & set(atom.predicate.split("_")))

    def _is_negative(self, text: str) -> bool:
        lower = text.lower()
        return any(cue in lower for cue in [" not ", "cannot", "can't", "without", "no ", "insufficient", "lacks", "lack"])

    def _tokens(self, text: str) -> set[str]:
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "be",
            "to",
            "for",
            "of",
            "and",
            "or",
            "does",
            "do",
            "did",
            "based",
            "according",
            "premises",
            "above",
            "which",
            "statement",
            "conclusion",
            "current",
        }
        return {token.lower().strip("'") for token in TOKEN_RE.findall(text.replace("_", " ")) if token.lower() not in stop}

    def _result(
        self,
        answer_signal: str,
        *,
        causal_score: float = 0.0,
        answer_distribution: dict[str, float] | None = None,
        target_match_score: float = 0.0,
        semantic_logic_score: float = 0.0,
        causal_effect_score: float = 0.0,
        rule_validity_score: float = 0.0,
        premise_coverage_score: float = 0.0,
        contradiction_penalty: float = 0.0,
        missing_condition_penalty: float = 0.0,
        counterfactual_gap: float = 0.0,
        proof_depth: int | None = None,
        evidence: list[str] | None = None,
        trajectories: list[CausalTrajectory] | None = None,
        premise_couplings: list[PremiseCoupling] | None = None,
        belief_states: list[CausalBeliefState] | None = None,
    ) -> CausalInferenceResult:
        distribution = answer_distribution or self._distribution_from_logits(
            yes=max(0.0, causal_score),
            uncertain=1.0 - min(1.0, abs(causal_score)),
            no=max(0.0, -causal_score),
        )
        return CausalInferenceResult(
            answer_signal=answer_signal,
            causal_score=round(causal_score, 4),
            answer_distribution=distribution,
            target_match_score=round(target_match_score, 4),
            semantic_logic_score=round(semantic_logic_score, 4),
            causal_effect_score=round(causal_effect_score, 4),
            rule_validity_score=round(rule_validity_score, 4),
            premise_coverage_score=round(premise_coverage_score, 4),
            contradiction_penalty=round(contradiction_penalty, 4),
            missing_condition_penalty=round(missing_condition_penalty, 4),
            counterfactual_gap=round(counterfactual_gap, 4),
            proof_depth=proof_depth,
            evidence=evidence or [],
            trajectories=trajectories or [],
            premise_couplings=premise_couplings or [],
            belief_states=belief_states or [],
        )


class GlobalOptionSemanticComparator:
    """Build a global option prior from premise-option semantic evidence.

    This module does not ask an LLM to choose the answer. It estimates how each
    premise supports or conflicts with each option, then converts those
    premise-level judgments into an option probability distribution.
    """

    def __init__(self, config: Type1PipelineConfig) -> None:
        self.config = config
        rag_config = SemanticRAGConfig(
            segmenter_model=config.stage0.segmenter_model,
            segmenter_api_base=config.stage0.segmenter_api_base,
            segmenter_api_key=config.stage0.segmenter_api_key,
            embedding_model=config.stage0.embedding_model,
            local_files_only=config.stage0.local_files_only,
            top_k=config.stage0.top_k,
        )
        self.vector_index = BGEVectorIndex(rag_config)

    def compare(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        graph: EvidenceGraph,
        stage0_result: dict[str, Any],
    ) -> GlobalOptionDistribution:
        options = classification.get("mcq_options") or {}
        if not options:
            return GlobalOptionDistribution("", {}, {}, 0.0, 0.0, [])

        premise_texts = self._premise_texts(stage_input, graph, stage0_result)
        if not premise_texts:
            premise_texts = [stage_input.question]

        judgments = self._semantic_judgments(options, premise_texts)
        logits = self._option_logits(options, judgments)
        distribution = self._softmax(logits, self.config.option_distribution_temperature)
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        selected = ranked[0][0] if ranked else ""
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else (ranked[0][1] if ranked else 0.0)
        entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in distribution.values())
        return GlobalOptionDistribution(
            selected_option=selected,
            option_distribution={label: round(prob, 6) for label, prob in distribution.items()},
            option_logits={label: round(value, 6) for label, value in logits.items()},
            margin=round(margin, 6),
            entropy=round(entropy, 6),
            judgments=judgments[: max(1, self.config.premise_alignment_top_k) * max(1, len(options))],
        )

    def _premise_texts(
        self,
        stage_input: Stage0Input,
        graph: EvidenceGraph,
        stage0_result: dict[str, Any],
    ) -> list[str]:
        texts: list[str] = []
        texts.extend(TextTools.clean(item) for item in stage_input.premises_nl if TextTools.clean(item))
        for fact in graph.derived_facts + graph.facts:
            source = TextTools.clean(fact.source)
            atom_text = fact.atom.lexical_text()
            texts.append(f"{atom_text}. {source}" if source else atom_text)
        for rule in graph.rules:
            symbolic = " ".join(
                [
                    " ".join(atom.lexical_text() for atom in rule.antecedents),
                    "implies",
                    " ".join(atom.lexical_text() for atom in rule.consequents),
                ]
            )
            source = TextTools.clean(rule.source)
            texts.append(f"{symbolic}. {source}" if source else symbolic)
        for match in stage0_result.get("semantic_matches", []) or []:
            text = TextTools.clean(match.get("text", ""))
            if text:
                texts.append(text)
        return [text for text in dict.fromkeys(texts) if text][:36]

    def _semantic_judgments(
        self,
        options: dict[str, str],
        premise_texts: list[str],
    ) -> list[PremiseOptionJudgment]:
        option_items = [(str(label), TextTools.clean(text)) for label, text in options.items()]
        texts = [text for _, text in option_items] + premise_texts
        vectors = self.vector_index.embed(texts)
        option_vectors = vectors[: len(option_items)]
        premise_vectors = vectors[len(option_items) :]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sim_matrix = np.nan_to_num(option_vectors @ premise_vectors.T, nan=0.0, posinf=0.0, neginf=0.0)

        judgments: list[PremiseOptionJudgment] = []
        for option_idx, (label, option_text) in enumerate(option_items):
            option_tokens = self._tokens(option_text)
            option_negative = self._is_negative(option_text)
            for premise_idx, premise_text in enumerate(premise_texts):
                premise_tokens = self._tokens(premise_text)
                semantic = max(0.0, min(1.0, float(sim_matrix[option_idx][premise_idx])))
                lexical = self._lexical_overlap(option_tokens, premise_tokens)
                relevance = min(1.0, semantic * 0.72 + lexical * 0.28)
                polarity_conflict = option_negative != self._is_negative(premise_text)
                contradiction = relevance * 0.72 if polarity_conflict and lexical >= 0.16 else relevance * 0.12
                support = max(0.0, relevance - contradiction * 0.58)
                if self._has_entailment_cue(premise_text):
                    support = min(1.0, support + 0.06)
                contribution = support - contradiction + relevance * 0.18
                judgments.append(
                    PremiseOptionJudgment(
                        label=label,
                        option_text=option_text,
                        premise_index=premise_idx,
                        premise_text=premise_text,
                        semantic_similarity=round(semantic, 6),
                        lexical_overlap=round(lexical, 6),
                        support_probability=round(support, 6),
                        contradiction_probability=round(contradiction, 6),
                        relevance_probability=round(relevance, 6),
                        contribution=round(contribution, 6),
                        judge_source="bge_global_semantic",
                    )
                )
        judgments.sort(key=lambda item: (item.label, item.contribution), reverse=True)
        return judgments

    def _option_logits(
        self,
        options: dict[str, str],
        judgments: list[PremiseOptionJudgment],
    ) -> dict[str, float]:
        logits = {str(label): 0.0 for label in options}
        grouped: dict[str, list[PremiseOptionJudgment]] = {str(label): [] for label in options}
        for judgment in judgments:
            grouped.setdefault(judgment.label, []).append(judgment)
        for label, items in grouped.items():
            top = sorted(items, key=lambda item: item.contribution, reverse=True)[: self.config.premise_alignment_top_k]
            if not top:
                continue
            support_mass = sum(item.support_probability * item.relevance_probability for item in top)
            contradiction_mass = sum(item.contradiction_probability * item.relevance_probability for item in top)
            coverage = len({item.premise_index for item in top if item.relevance_probability >= 0.32}) / max(1, self.config.premise_alignment_top_k)
            logits[label] = support_mass - contradiction_mass + coverage * 0.22
        return logits

    def _softmax(self, logits: dict[str, float], temperature: float) -> dict[str, float]:
        if not logits:
            return {}
        temp = max(0.05, temperature)
        max_logit = max(logits.values())
        exp_values = {label: math.exp((value - max_logit) / temp) for label, value in logits.items()}
        total = sum(exp_values.values()) or 1.0
        return {label: value / total for label, value in exp_values.items()}

    def _lexical_overlap(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, min(len(left), len(right)))

    def _has_entailment_cue(self, text: str) -> bool:
        lower = text.lower()
        return any(cue in lower for cue in ["if ", " then ", "implies", "requires", "must", "all ", "every "])

    def _is_negative(self, text: str) -> bool:
        lower = f" {TextTools.clean(text).lower()} "
        return any(cue in lower for cue in [" not ", " no ", " cannot", " can't", " without", " insufficient", " lacks", " lack "])

    def _tokens(self, text: str) -> set[str]:
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "be",
            "to",
            "for",
            "of",
            "and",
            "or",
            "does",
            "do",
            "did",
            "based",
            "according",
            "premises",
            "above",
            "which",
            "statement",
            "conclusion",
            "current",
            "option",
        }
        return {token.lower().strip("'") for token in TOKEN_RE.findall(TextTools.clean(text).replace("_", " ")) if token.lower() not in stop}

class Type1Stage2Reasoner:
    """Deterministic Type 1 candidate scorer over the Stage 1 evidence graph."""

    def __init__(self, config: Type1PipelineConfig) -> None:
        self.config = config
        self.world_model = Type1CausalInferenceWorldModel(config) if config.enable_causal_world_model else None
        self.option_comparator = (
            GlobalOptionSemanticComparator(config) if config.enable_global_option_distribution else None
        )

    def solve(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        graph: EvidenceGraph,
        stage0_result: dict[str, Any],
    ) -> Stage2Candidate:
        if classification.get("question_format") == "multiple_choice":
            return self._solve_mcq(stage_input, classification, graph, stage0_result)
        return self._solve_yes_no(stage_input, classification, graph, stage0_result)

    def _solve_mcq(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        graph: EvidenceGraph,
        stage0_result: dict[str, Any],
    ) -> Stage2Candidate:
        options = classification.get("mcq_options") or {}
        scores = [
            self._score_text(label, text, graph, stage0_result)
            for label, text in options.items()
        ]
        global_distribution = (
            self.option_comparator.compare(stage_input, classification, graph, stage0_result)
            if self.option_comparator is not None
            else GlobalOptionDistribution("", {}, {}, 0.0, 0.0, [])
        )
        self._merge_global_option_distribution(scores, global_distribution)
        causal_by_option: dict[str, dict[str, Any]] = {}
        if self.world_model is not None:
            for score in scores:
                causal_result = self.world_model.infer(
                    score.text,
                    graph,
                    self._semantic_matches(stage0_result),
                )
                causal_by_option[score.label] = asdict(causal_result)
                self._merge_causal_score(score, causal_result)
        self._refresh_option_posteriors(scores)
        if not scores:
            return Stage2Candidate("", "No options were available for Stage 2 scoring.", "stage2_no_options", 0.0, [])
        scores.sort(key=lambda score: (self._option_posterior_score(score), -(score.proof_depth or 99)), reverse=True)
        best = scores[0]
        margin = 0.0
        if len(scores) > 1:
            second = scores[1]
            margin = self._option_posterior_score(best) - self._option_posterior_score(second)
        if self._mcq_should_be_uncertain(best, margin):
            best.decision = "insufficient_unique_proof"
            explanation = self._explain_score(best, "not proven strongly enough for option selection")
            confidence = min(0.76, 0.38 + best.posterior_probability * 0.3 + best.semantic * 0.14 + max(0.0, margin) * 0.12)
            return Stage2Candidate(
                "Uncertain",
                explanation,
                "stage2_option_evidence_graph",
                round(confidence, 4),
                scores,
                {"options": causal_by_option, "global_option_distribution": asdict(global_distribution)},
            )
        confidence = min(
            0.94,
            0.28
            + best.support * 0.3
            + best.semantic * 0.12
            + best.posterior_probability * 0.34
            + max(0.0, margin) * 0.2,
        )
        explanation = self._explain_score(best, "best-supported option")
        return Stage2Candidate(
            best.label,
            explanation,
            "stage2_option_evidence_graph",
            round(confidence, 4),
            scores,
            {
                "selected_option": best.label,
                "options": causal_by_option,
                "global_option_distribution": asdict(global_distribution),
            },
        )

    def _mcq_should_be_uncertain(self, best: Stage2OptionScore, margin: float) -> bool:
        if best.posterior_probability >= 0.52 and margin >= 0.11:
            return False
        if best.support < 0.5 and best.global_semantic_prior < 0.44:
            return True
        if margin < 0.08:
            return True
        if (best.proof_depth is None or best.proof_depth == 0) and best.semantic < 0.18:
            return True
        if best.contradiction >= 0.55 and best.semantic < 0.35:
            return True
        return False

    def _merge_global_option_distribution(
        self,
        scores: list[Stage2OptionScore],
        distribution: GlobalOptionDistribution,
    ) -> None:
        alignments_by_label: dict[str, list[PremiseOptionJudgment]] = {}
        for judgment in distribution.judgments:
            alignments_by_label.setdefault(judgment.label, []).append(judgment)
        for score in scores:
            prior = float(distribution.option_distribution.get(score.label, 0.0) or 0.0)
            alignments = sorted(
                alignments_by_label.get(score.label, []),
                key=lambda item: item.contribution,
                reverse=True,
            )[: self.config.premise_alignment_top_k]
            alignment_score = sum(max(0.0, item.contribution) for item in alignments) / max(1, len(alignments))
            contradiction = sum(item.contradiction_probability for item in alignments) / max(1, len(alignments))
            score.global_semantic_prior = round(prior, 6)
            score.premise_alignment_score = round(alignment_score, 6)
            score.semantic = round(max(score.semantic, min(1.0, alignment_score)), 4)
            score.support = round(max(score.support, min(1.0, score.support * 0.72 + alignment_score * 0.42 + prior * 0.24)), 4)
            score.contradiction = round(max(score.contradiction, min(1.0, contradiction)), 4)
        self._refresh_option_posteriors(scores)

    def _refresh_option_posteriors(self, scores: list[Stage2OptionScore]) -> None:
        if not scores:
            return
        logits = {
            score.label: self._option_posterior_logit(score)
            for score in scores
        }
        max_logit = max(logits.values())
        exp_values = {label: math.exp(value - max_logit) for label, value in logits.items()}
        total = sum(exp_values.values()) or 1.0
        for score in scores:
            score.posterior_probability = round(exp_values[score.label] / total, 6)

    def _option_posterior_score(self, score: Stage2OptionScore) -> float:
        return score.posterior_probability + self._option_posterior_logit(score) * 0.18

    def _option_posterior_logit(self, score: Stage2OptionScore) -> float:
        causal_support = max(0.0, score.causal_score)
        causal_conflict = abs(min(0.0, score.causal_score))
        return (
            score.support * 0.34
            - score.contradiction * 0.31
            + score.semantic * 0.18
            + score.global_semantic_prior * self.config.global_option_distribution_weight
            + score.premise_alignment_score * 0.23
            + causal_support * 0.18
            - causal_conflict * 0.16
        )

    def _solve_yes_no(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        graph: EvidenceGraph,
        stage0_result: dict[str, Any],
    ) -> Stage2Candidate:
        score = self._score_text("query", stage_input.question, graph, stage0_result)
        lower_q = stage_input.question.lower()
        asks_requirement = any(word in lower_q for word in ["requirement", "requirements", "qualify", "eligible", "can ", "meet"])
        asks_follow = "follow" in lower_q or "according to the premises" in lower_q
        strict_entailment = asks_follow or any(
            cue in lower_q
            for cue in [
                "is it true",
                "statement:",
                "do the premises support",
                "does the premise support",
                "according to the premise",
                "based on the premises",
            ]
        )
        target = self._check_target_proof(stage_input.question, graph)
        causal_result = self.world_model.infer(
            stage_input.question,
            graph,
            self._semantic_matches(stage0_result),
        ) if self.world_model is not None else None
        if causal_result is not None:
            self._merge_causal_score(score, causal_result)

        if target["decision"] == "target_supported":
            answer = "Yes"
            decision = "target_supported"
            score.support = max(score.support, target["support"])
            score.contradiction = max(score.contradiction, target["contradiction"])
            score.proof_depth = target["proof_depth"]
            score.evidence = target["evidence"]
        elif target["decision"] == "target_contradicted":
            answer = "No"
            decision = "target_contradicted"
            score.support = max(score.support, target["support"])
            score.contradiction = max(score.contradiction, target["contradiction"])
            score.proof_depth = target["proof_depth"]
            score.evidence = target["evidence"]
        elif strict_entailment and target["decision"] in {"unknown", "target_rule_incomplete"}:
            answer = "No"
            decision = "strict_entailment_not_proven"
            score.support = min(score.support, target["support"])
            score.contradiction = max(score.contradiction, target["contradiction"], 0.52)
            score.proof_depth = target["proof_depth"]
            score.evidence = target["evidence"]
        elif causal_result is not None and causal_result.answer_signal == "No" and causal_result.causal_score <= self.config.causal_no_threshold:
            answer = "No"
            decision = "causal_intervention_contradicted"
            score.contradiction = max(score.contradiction, causal_result.contradiction_penalty)
            score.evidence = causal_result.evidence
            score.proof_depth = causal_result.proof_depth
        elif causal_result is not None and causal_result.answer_signal == "Yes" and causal_result.causal_score >= self.config.causal_yes_threshold:
            answer = "Yes"
            decision = "causal_intervention_reached_target"
            score.support = max(score.support, causal_result.target_match_score)
            score.evidence = causal_result.evidence
            score.proof_depth = causal_result.proof_depth
        elif asks_requirement and target["decision"] == "target_rule_incomplete":
            answer = "No"
            decision = "missing_required_support"
            score.support = min(score.support, target["support"])
            score.contradiction = max(score.contradiction, target["contradiction"])
            score.proof_depth = target["proof_depth"]
            score.evidence = target["evidence"]
        elif causal_result is not None and causal_result.missing_condition_penalty >= 0.75 and asks_requirement:
            answer = "No"
            decision = "causal_missing_required_support"
            score.contradiction = max(score.contradiction, causal_result.missing_condition_penalty)
            score.evidence = causal_result.evidence
            score.proof_depth = causal_result.proof_depth
        elif asks_requirement and (score.proof_depth is None or score.proof_depth == 0) and score.semantic < 0.5:
            answer = "No"
            decision = "missing_required_support"
        elif score.support >= 0.58 and score.support >= score.contradiction:
            answer = "Yes"
            decision = "supported"
        elif score.contradiction >= 0.55:
            answer = "No"
            decision = "contradicted"
        elif asks_requirement:
            answer = "No"
            decision = "missing_required_support"
        elif asks_follow:
            answer = "Uncertain"
            decision = "not_entailed"
        else:
            answer = "Uncertain"
            decision = "insufficient_evidence"
        score.decision = decision
        causal_bonus = abs(score.causal_score) * 0.1 if score.causal_decision in {"Yes", "No"} else 0.0
        confidence = min(0.9, 0.32 + max(score.support, score.contradiction) * 0.48 + score.semantic * 0.12 + causal_bonus)
        explanation = self._explain_score(score, decision)
        return Stage2Candidate(
            answer,
            explanation,
            "stage2_yes_no_evidence_graph",
            round(confidence, 4),
            [score],
            {"query": asdict(causal_result)} if causal_result is not None else {},
        )

    def _merge_causal_score(self, score: Stage2OptionScore, causal_result: CausalInferenceResult) -> None:
        score.causal_score = causal_result.causal_score
        score.causal_decision = causal_result.answer_signal
        if causal_result.answer_signal == "Yes":
            score.support = round(max(score.support, causal_result.target_match_score, max(0.0, causal_result.causal_score)), 4)
            if causal_result.proof_depth is not None:
                score.proof_depth = causal_result.proof_depth
            if causal_result.evidence:
                score.evidence = causal_result.evidence
        elif causal_result.answer_signal == "No":
            score.contradiction = round(max(score.contradiction, causal_result.contradiction_penalty, abs(min(0.0, causal_result.causal_score))), 4)
            if causal_result.evidence:
                score.evidence = causal_result.evidence

    def _semantic_matches(self, stage0_result: dict[str, Any]) -> list[dict[str, Any]]:
        matches = stage0_result.get("semantic_matches", []) or []
        return matches if isinstance(matches, list) else []

    def _check_target_proof(self, question: str, graph: EvidenceGraph) -> dict[str, Any]:
        query_tokens = self._tokens(question)
        if not query_tokens:
            return self._target_result("unknown")

        facts = graph.derived_facts + graph.facts
        best_fact: tuple[float, EvidenceFact] | None = None
        best_contradiction: tuple[float, EvidenceFact] | None = None
        negative_query = self._is_negative(question)
        for fact in facts:
            overlap = self._atom_overlap(query_tokens, fact.atom)
            if overlap < 0.45:
                continue
            if negative_query != fact.atom.negated and self._predicate_mentions(query_tokens, fact):
                if best_contradiction is None or overlap > best_contradiction[0]:
                    best_contradiction = (overlap, fact)
            elif best_fact is None or (overlap, fact.depth) > (best_fact[0], best_fact[1].depth):
                best_fact = (overlap, fact)

        if best_contradiction is not None and (best_fact is None or best_contradiction[0] >= best_fact[0]):
            overlap, fact = best_contradiction
            return self._target_result(
                "target_contradicted",
                support=0.0,
                contradiction=min(1.0, 0.55 + overlap * 0.35),
                proof_depth=fact.depth,
                evidence=fact.support_chain[:4] or [fact.source],
            )

        if best_fact is not None:
            overlap, fact = best_fact
            if fact.depth > 0 or overlap >= 0.68:
                return self._target_result(
                    "target_supported",
                    support=min(1.0, 0.58 + overlap * 0.38 + (0.08 if fact.depth > 0 else 0.0)),
                    contradiction=0.0,
                    proof_depth=fact.depth,
                    evidence=fact.support_chain[:4] or [fact.source],
                )

        best_rule: tuple[float, EvidenceRule] | None = None
        for rule in graph.rules:
            overlap = self._rule_target_overlap(query_tokens, rule)
            if overlap >= 0.45 and (best_rule is None or overlap > best_rule[0]):
                best_rule = (overlap, rule)

        if best_rule is not None:
            overlap, rule = best_rule
            completeness = self._rule_antecedent_completeness(rule, facts)
            if completeness >= 0.999:
                return self._target_result(
                    "target_supported",
                    support=min(0.92, 0.52 + overlap * 0.28),
                    contradiction=0.0,
                    proof_depth=1,
                    evidence=[rule.source],
                )
            return self._target_result(
                "target_rule_incomplete",
                support=max(0.0, 0.42 * completeness),
                contradiction=0.56 + (1.0 - completeness) * 0.24,
                proof_depth=None,
                evidence=[rule.source],
            )

        return self._target_result("unknown")

    def _target_result(
        self,
        decision: str,
        *,
        support: float = 0.0,
        contradiction: float = 0.0,
        proof_depth: int | None = None,
        evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "decision": decision,
            "support": round(support, 4),
            "contradiction": round(contradiction, 4),
            "proof_depth": proof_depth,
            "evidence": evidence or [],
        }

    def _atom_overlap(self, query_tokens: set[str], atom: LogicAtom) -> float:
        atom_tokens = self._tokens(atom.lexical_text())
        atom_tokens = {token for token in atom_tokens if token not in VARIABLES}
        if not atom_tokens:
            return 0.0
        return len(query_tokens & atom_tokens) / len(atom_tokens)

    def _rule_target_overlap(self, query_tokens: set[str], rule: EvidenceRule) -> float:
        overlaps = [self._atom_overlap(query_tokens, atom) for atom in rule.consequents]
        return max(overlaps, default=0.0)

    def _rule_antecedent_completeness(self, rule: EvidenceRule, facts: list[EvidenceFact]) -> float:
        if not rule.antecedents:
            return 1.0
        satisfied = sum(1 for antecedent in rule.antecedents if self._antecedent_satisfied(antecedent, facts))
        return satisfied / len(rule.antecedents)

    def _antecedent_satisfied(self, antecedent: LogicAtom, facts: list[EvidenceFact]) -> bool:
        for fact in facts:
            if fact.atom.predicate != antecedent.predicate or fact.atom.negated != antecedent.negated:
                continue
            if antecedent.operator:
                if fact.atom.operator and fact.atom.value == antecedent.value:
                    return True
                continue
            if len(antecedent.args) != len(fact.atom.args):
                continue
            matched = True
            for ant_arg, fact_arg in zip(antecedent.args, fact.atom.args):
                if ant_arg in VARIABLES or ant_arg == fact_arg:
                    continue
                matched = False
                break
            if matched:
                return True
        return False

    def _score_text(
        self,
        label: str,
        text: str,
        graph: EvidenceGraph,
        stage0_result: dict[str, Any],
    ) -> Stage2OptionScore:
        text_tokens = self._tokens(text)
        negated_query = self._is_negative(text)
        facts = graph.derived_facts + graph.facts
        best_support = 0.0
        best_contradiction = 0.0
        best_depth: int | None = None
        evidence: list[str] = []
        for fact in facts:
            fact_tokens = self._tokens(fact.atom.lexical_text())
            if not fact_tokens:
                continue
            overlap = len(text_tokens & fact_tokens) / max(len(fact_tokens), 1)
            if overlap < 0.18:
                continue
            support = min(1.0, overlap + (0.16 if fact.depth > 0 else 0.06))
            contradiction = 0.0
            if negated_query != fact.atom.negated:
                contradiction = support * 0.85 if self._predicate_mentions(text_tokens, fact) else 0.0
            else:
                previous_support = best_support
                best_support = max(best_support, support)
                if support >= previous_support or best_depth is None or fact.depth < best_depth:
                    best_depth = fact.depth
                    evidence = fact.support_chain[:4] or [fact.source]
            best_contradiction = max(best_contradiction, contradiction)

        for rule in graph.rules:
            rule_text = " ".join(
                [
                    rule.source,
                    " ".join(atom.lexical_text() for atom in rule.antecedents),
                    " ".join(atom.lexical_text() for atom in rule.consequents),
                ]
            )
            rule_tokens = self._tokens(rule_text)
            if not rule_tokens:
                continue
            overlap = len(text_tokens & rule_tokens) / max(len(text_tokens), 1)
            if overlap < 0.22:
                continue
            support = min(0.88, overlap + (0.1 if rule.derived else 0.18))
            if self._is_negative(text) and not any(atom.negated for atom in rule.consequents):
                best_contradiction = max(best_contradiction, support * 0.7)
            else:
                previous_support = best_support
                best_support = max(best_support, support)
                if support >= previous_support or best_depth is None or best_depth > 1:
                    best_depth = 1
                    evidence = [rule.source]

        semantic = self._stage0_semantic_score(text, stage0_result)
        decision = "supported" if best_support >= 0.58 else "weak_or_unmatched"
        return Stage2OptionScore(
            label=label,
            text=text,
            support=round(best_support, 4),
            contradiction=round(best_contradiction, 4),
            semantic=round(semantic, 4),
            causal_score=0.0,
            causal_decision="",
            proof_depth=best_depth,
            evidence=evidence,
            decision=decision,
        )

    def _stage0_semantic_score(self, text: str, stage0_result: dict[str, Any]) -> float:
        text_tokens = self._tokens(text)
        matches = stage0_result.get("semantic_matches", []) or []
        best = 0.0
        for match in matches[:8]:
            match_tokens = self._tokens(match.get("text", ""))
            if not match_tokens:
                continue
            overlap = len(text_tokens & match_tokens) / max(len(text_tokens), 1)
            best = max(best, overlap * float(match.get("score", 0.0) or 0.0))
        return min(best, 1.0)

    def _predicate_mentions(self, text_tokens: set[str], fact: EvidenceFact) -> bool:
        predicate_tokens = set(fact.atom.predicate.split("_"))
        return bool(text_tokens & predicate_tokens)

    def _is_negative(self, text: str) -> bool:
        lower = text.lower()
        return any(cue in lower for cue in [" not ", "cannot", "can't", "without", "no ", "insufficient", "lacks", "lack"])

    def _tokens(self, text: str) -> set[str]:
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "be",
            "to",
            "for",
            "of",
            "and",
            "or",
            "does",
            "do",
            "did",
            "based",
            "according",
            "premises",
            "above",
            "which",
            "statement",
            "conclusion",
            "current",
        }
        return {token.lower().strip("'") for token in TOKEN_RE.findall(text.replace("_", " ")) if token.lower() not in stop}

    def _explain_score(self, score: Stage2OptionScore, decision: str) -> str:
        evidence = "; ".join(score.evidence[:2]) if score.evidence else "no direct derived proof"
        return (
            f"Stage 2 selected {score.label} as {decision}. "
            f"support={score.support}, contradiction={score.contradiction}, semantic={score.semantic}, "
            f"global_prior={score.global_semantic_prior}, posterior={score.posterior_probability}. "
            f"Evidence: {evidence}."
        )


class Type1Stage3Controller:
    def __init__(self, config: Type1PipelineConfig) -> None:
        self.config = config
        self.gate = SemanticHybridGate(config.stage3_pass_threshold)

    def finalize(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        stage2_candidate: Stage2Candidate,
        graph: EvidenceGraph,
        stage0_result: dict[str, Any],
    ) -> dict[str, Any]:
        candidate = SemanticHybridCandidate(
            answer=stage2_candidate.answer,
            explanation=stage2_candidate.explanation,
            source=stage2_candidate.source,
            confidence=stage2_candidate.confidence,
            structured_hits=self._stage2_hits(stage2_candidate),
            text_stream=[],
        )
        matches = []
        gate = self.gate.evaluate(stage_input, classification, candidate, matches)
        gate = self._add_stage2_signal(gate, stage2_candidate)
        return {
            "candidate": asdict(candidate),
            "gate": asdict(gate),
            "fusion": {
                "stage0_source": (stage0_result.get("candidate") or {}).get("source", ""),
                "stage2_source": stage2_candidate.source,
                "stage2_confidence": stage2_candidate.confidence,
                "evidence_graph_summary": graph.summary(),
            },
        }

    def _stage2_hits(self, candidate: Stage2Candidate) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for score in candidate.option_scores:
            if score.support > 0 or score.semantic > 0:
                hits.append(asdict(score))
        return hits

    def _add_stage2_signal(
        self,
        gate: Any,
        stage2_candidate: Stage2Candidate,
    ) -> Any:
        score = gate.score
        reasons = list(gate.reasons)
        if stage2_candidate.confidence >= self.config.stage2_confidence_threshold:
            score += min(0.24, stage2_candidate.confidence * 0.24)
            reasons.append("stage2_confident")
        if any(item.evidence for item in stage2_candidate.option_scores):
            score += 0.08
            reasons.append("stage2_evidence_available")
        causal_trace = self._selected_causal_trace(stage2_candidate)
        if causal_trace:
            causal_score = float(causal_trace.get("causal_score", 0.0) or 0.0)
            answer_signal = str(causal_trace.get("answer_signal", ""))
            missing_penalty = float(causal_trace.get("missing_condition_penalty", 0.0) or 0.0)
            contradiction_penalty = float(causal_trace.get("contradiction_penalty", 0.0) or 0.0)
            agrees_with_yes = answer_signal == stage2_candidate.answer == "Yes" and causal_score >= self.config.causal_yes_threshold
            agrees_with_no = answer_signal == stage2_candidate.answer == "No" and causal_score <= self.config.causal_no_threshold
            if agrees_with_yes or agrees_with_no:
                score += 0.1
                reasons.append("causal_world_model_agrees")
            elif answer_signal in {"Yes", "No"} and answer_signal != stage2_candidate.answer:
                score -= 0.12
                reasons.append("causal_world_model_disagrees")
            if stage2_candidate.answer == "Yes" and missing_penalty >= 0.55:
                score -= min(0.18, missing_penalty * 0.2)
                reasons.append("causal_missing_condition_penalty")
            if stage2_candidate.answer == "Yes" and contradiction_penalty >= 0.55:
                score -= min(0.16, contradiction_penalty * 0.18)
                reasons.append("causal_contradiction_penalty")
        if stage2_candidate.answer == "Uncertain":
            score -= 0.04
            reasons.append("uncertain_answer_penalty")
        score = round(max(0.0, min(1.0, score)), 4)
        gate.score = score
        gate.reasons = reasons
        gate.passed = score >= self.config.stage3_pass_threshold
        return gate

    def _selected_causal_trace(self, stage2_candidate: Stage2Candidate) -> dict[str, Any]:
        causal_inference = stage2_candidate.causal_inference or {}
        query_trace = causal_inference.get("query")
        if isinstance(query_trace, dict):
            return query_trace
        selected = causal_inference.get("selected_option")
        options = causal_inference.get("options")
        if selected and isinstance(options, dict):
            option_trace = options.get(selected)
            if isinstance(option_trace, dict):
                return option_trace
        return {}


class Type1MultiStagePipeline:
    """Stage 0-3 Type 1-only reasoning pipeline."""

    def __init__(
        self,
        config: Type1PipelineConfig | None = None,
        stage0_parser: Type1SemanticHybridParser | None = None,
    ) -> None:
        self.config = config or Type1PipelineConfig()
        self.stage0_parser = stage0_parser or Type1SemanticHybridParser(self.config.stage0)
        self.classifier = Type1QuestionClassifier()
        self.stage1 = Type1EvidenceGraphBuilder(self.config.max_forward_steps)
        self.stage2 = Type1Stage2Reasoner(self.config)
        self.stage3 = Type1Stage3Controller(self.config)
        self.transformer_brain = (
            LocalTransformerWorldModel(
                TransformerWorldModelConfig(
                    hidden_dim=self.config.transformer_brain_hidden_dim,
                    imagination_layers=self.config.transformer_brain_imagination_layers,
                    attention_heads=self.config.transformer_brain_attention_heads,
                    frame_local_window=self.config.transformer_brain_frame_local_window,
                    ssm_block_size=self.config.transformer_brain_ssm_block_size,
                    allow_override=self.config.transformer_brain_allow_override,
                    temperature=self.config.transformer_brain_temperature,
                    override_margin=self.config.transformer_brain_override_margin,
                    minimum_override_confidence=self.config.transformer_brain_minimum_override_confidence,
                    minimum_override_winner_margin=self.config.transformer_brain_minimum_override_winner_margin,
                )
            )
            if self.config.enable_transformer_brain_world_model
            else None
        )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        stage0_result = self.stage0_parser.parse(payload)
        stage_input = self.stage0_parser._normalize_payload(payload)
        classification = stage0_result.get("classification") or self.classifier.classify(
            question=stage_input.question,
            answer="",
            premises_nl=stage_input.premises_nl,
            premises_fol=stage_input.premises_fol,
            choices=stage_input.choices,
        )
        graph = self.stage1.build(stage_input)
        stage2_candidate = self.stage2.solve(stage_input, classification, graph, stage0_result)
        transformer_brain_result: dict[str, Any] = {}
        if self.transformer_brain is not None:
            brain = self.transformer_brain.run(
                stage_input,
                classification,
                graph.summary(),
                asdict(stage2_candidate),
            )
            transformer_brain_result = brain.to_dict()
            stage2_candidate.causal_inference["transformer_brain_world_model"] = transformer_brain_result
            if brain.should_override:
                stage2_candidate.answer = brain.final_answer
                stage2_candidate.confidence = round(max(stage2_candidate.confidence, brain.confidence), 4)
                stage2_candidate.source = "stage2_transformer_brain_world_model"
                stage2_candidate.explanation = (
                    f"Transformer brain world model overrode Stage 2 after internal imagination. "
                    f"raw_brain_answer={brain.raw_brain_answer}, confidence={brain.confidence}, margin={brain.margin}. "
                    f"Previous explanation: {stage2_candidate.explanation}"
                )
        stage3_result = self.stage3.finalize(stage_input, classification, stage2_candidate, graph, stage0_result)
        final_candidate = stage3_result["candidate"]
        return {
            "query_type": "type1",
            "stage": "stage3",
            "method": "type1_multistage_semantic_evidence_pipeline",
            "normalized_input": asdict(stage_input),
            "classification": classification,
            "stage0": stage0_result,
            "stage1": {
                "evidence_graph_summary": graph.summary(),
                **({"evidence_graph": graph.to_dict()} if self.config.include_evidence_graph else {}),
            },
            "stage2": {
                "candidate": asdict(stage2_candidate),
                "option_scores": [asdict(score) for score in stage2_candidate.option_scores],
            },
            "transformer_brain_world_model": transformer_brain_result,
            "stage3": stage3_result,
            "candidate": final_candidate,
            "gate": stage3_result["gate"],
            "metadata": {
                "stages_completed": [
                    "stage0",
                    "stage1",
                    "stage2",
                    *(['transformer_brain_world_model'] if transformer_brain_result else []),
                    "stage3",
                ],
                "final_answer_normalized": normalize_for_eval(final_candidate.get("answer", "")),
            },
        }
