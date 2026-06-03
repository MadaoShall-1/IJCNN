#!/usr/bin/env python3
"""Type 1 Stage 0 parser based on semantic parsing and hybrid retrieval.

This is the Stage 0 path requested for Type 1:

1. Input text and premises.
2. Qwen-style semantic decomposition into fine-grained semantic units.
3. Structured attribute base initialization from decomposed units.
4. BGE-style vector mapping and semantic matching.
5. Branch output:
   - structured attribute hits for matched attributes
   - natural-language semantic stream for unmatched/free-form text
6. Gate checks retrieval quality and answer-shape reliability.

The parser is built around semantic decomposition, vector matching, and
structured attribute retrieval.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .common import LLMClient, Stage0Input
from .semantic_tree_rag import (
    BGEVectorIndex,
    SemanticRAGConfig,
    SemanticTree,
    SemanticTreeBuilder,
)
from .type1_preprocessing import AnswerNormalizer, TextTools, Type1QuestionClassifier


@dataclass
class SemanticHybridConfig:
    segmenter_model: str = "openai/Qwen2.5-1.5B-Instruct"
    segmenter_api_base: str = "http://localhost:8001/v1"
    segmenter_api_key: str = "EMPTY"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    local_files_only: bool = False
    top_k: int = 8
    attribute_hit_threshold: float = 0.48
    gate_pass_threshold: float = 0.62
    include_tree: bool = False
    enable_agent_fallback: bool = True
    agent_fallback_gate_threshold: float = 0.82


@dataclass
class StructuredAttribute:
    attr_id: str
    text: str
    attr_type: str
    entities: list[str] = field(default_factory=list)
    predicates: list[str] = field(default_factory=list)
    logic_cues: list[str] = field(default_factory=list)
    source_node_id: str = ""
    source_index: int | None = None


@dataclass
class AttributeMatch:
    attr_id: str
    text: str
    attr_type: str
    score: float
    entities: list[str]
    predicates: list[str]
    logic_cues: list[str]
    source_index: int | None


@dataclass
class SemanticHybridCandidate:
    answer: str
    explanation: str
    source: str
    confidence: float
    structured_hits: list[dict[str, Any]]
    text_stream: list[str]


@dataclass
class SemanticHybridGateResult:
    passed: bool
    score: float
    reasons: list[str]


@dataclass
class SemanticHybridParseResult:
    query_type: str
    stage: str
    method: str
    normalized_input: dict[str, Any]
    classification: dict[str, Any]
    semantic_tree_summary: dict[str, Any]
    structured_attribute_base: dict[str, Any]
    semantic_matches: list[dict[str, Any]]
    branch_output: dict[str, Any]
    candidate: dict[str, Any]
    gate: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StructuredAttributeBase:
    """Attribute base initialized from semantic tree nodes."""

    def __init__(self, attributes: list[StructuredAttribute]) -> None:
        self.attributes = attributes

    @classmethod
    def from_tree(cls, tree: SemanticTree) -> "StructuredAttributeBase":
        attributes: list[StructuredAttribute] = []
        for node in tree.nodes:
            if node.node_type in {"root", "question", "option"}:
                continue
            metadata = node.metadata or {}
            attributes.append(
                StructuredAttribute(
                    attr_id=f"attr:{len(attributes)}",
                    text=node.text,
                    attr_type=node.node_type,
                    entities=[str(item) for item in metadata.get("entities", [])],
                    predicates=[str(item) for item in metadata.get("predicates", [])],
                    logic_cues=[str(item) for item in metadata.get("logic_cues", [])],
                    source_node_id=node.node_id,
                    source_index=node.source_index,
                )
            )
        return cls(attributes)

    def summary(self) -> dict[str, Any]:
        return {
            "attribute_count": len(self.attributes),
            "attr_type_counts": dict(Counter(attr.attr_type for attr in self.attributes).most_common()),
            "logic_cue_counts": dict(
                Counter(cue for attr in self.attributes for cue in attr.logic_cues).most_common()
            ),
        }


class SemanticHybridMatcher:
    """Vector + structured-attribute matcher."""

    def __init__(self, vector_index: BGEVectorIndex, config: SemanticHybridConfig) -> None:
        self.vector_index = vector_index
        self.config = config

    def match(self, query: str, base: StructuredAttributeBase) -> list[AttributeMatch]:
        if not base.attributes:
            return []
        query_vector = self.vector_index.embed([query])[0]
        attr_vectors = self.vector_index.embed([attr.text for attr in base.attributes])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scores = np.nan_to_num(attr_vectors @ query_vector, nan=0.0, posinf=0.0, neginf=0.0)
        ranked: list[AttributeMatch] = []
        for attr, score in sorted(zip(base.attributes, scores.tolist()), key=lambda item: item[1], reverse=True):
            score_float = float(score)
            if score_float < self.config.attribute_hit_threshold and len(ranked) >= self.config.top_k:
                break
            ranked.append(
                AttributeMatch(
                    attr_id=attr.attr_id,
                    text=attr.text,
                    attr_type=attr.attr_type,
                    score=round(score_float, 6),
                    entities=attr.entities,
                    predicates=attr.predicates,
                    logic_cues=attr.logic_cues,
                    source_index=attr.source_index,
                )
            )
            if len(ranked) >= self.config.top_k:
                break
        return ranked


class SemanticHybridAnswerComposer:
    """Lightweight answer-shape composer from semantic matches.

    Stage 0 remains a parser/retriever, so this is intentionally conservative.
    It gives an answer candidate for automatic evaluation, but exposes the
    evidence and confidence so later stages can perform deeper reasoning.
    """

    def compose(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        matches: list[AttributeMatch],
    ) -> SemanticHybridCandidate:
        question_format = classification.get("question_format")
        if question_format == "multiple_choice":
            return self._compose_mcq(classification, matches)
        if question_format == "yes_no_judgment":
            return self._compose_yes_no(matches)
        return self._compose_open(matches)

    def _compose_mcq(
        self,
        classification: dict[str, Any],
        matches: list[AttributeMatch],
    ) -> SemanticHybridCandidate:
        options = classification.get("mcq_options") or {}
        if not options:
            return self._empty("No multiple-choice options were parsed.")
        evidence_text = " ".join(match.text.lower() for match in matches)
        option_scores: dict[str, float] = {}
        for label, option in options.items():
            option_tokens = set(option.lower().replace("-", " ").split())
            if not option_tokens:
                option_scores[label] = 0.0
                continue
            overlap = sum(1 for token in option_tokens if token in evidence_text)
            option_scores[label] = overlap / max(len(option_tokens), 1)
        answer, score = max(option_scores.items(), key=lambda item: item[1])
        confidence = 0.35 + min(score, 0.5)
        return SemanticHybridCandidate(
            answer=answer,
            explanation=f"Option {answer} has the strongest semantic overlap with retrieved structured evidence.",
            source="semantic_hybrid_mcq_overlap",
            confidence=round(confidence, 4),
            structured_hits=[asdict(match) for match in matches],
            text_stream=[match.text for match in matches if match.score < 0.48],
        )

    def _compose_yes_no(self, matches: list[AttributeMatch]) -> SemanticHybridCandidate:
        if not matches:
            return self._empty("No semantic evidence matched the query.")
        top = matches[0]
        cue_set = {cue for match in matches[:3] for cue in match.logic_cues}
        if top.score >= 0.62 and {"if_then", "universal"} & cue_set:
            answer = "Yes"
            explanation = "The query matched structured rule/fact evidence with sufficient semantic similarity."
            confidence = min(0.86, 0.45 + top.score * 0.45)
        elif top.score >= 0.5:
            answer = "Uncertain"
            explanation = "Relevant evidence was found, but the structured match is not strong enough for entailment."
            confidence = 0.52
        else:
            answer = "Uncertain"
            explanation = "No high-confidence structured attribute hit was found."
            confidence = 0.4
        return SemanticHybridCandidate(
            answer=answer,
            explanation=explanation,
            source="semantic_hybrid_attribute_match",
            confidence=round(confidence, 4),
            structured_hits=[asdict(match) for match in matches if match.score >= 0.48],
            text_stream=[match.text for match in matches if match.score < 0.48],
        )

    def _compose_open(self, matches: list[AttributeMatch]) -> SemanticHybridCandidate:
        if not matches:
            return self._empty("No semantic evidence matched the query.")
        top_text = matches[0].text
        return SemanticHybridCandidate(
            answer=top_text,
            explanation="The top semantic attribute match is returned as the structured parser candidate.",
            source="semantic_hybrid_open_match",
            confidence=round(min(0.75, 0.35 + matches[0].score * 0.4), 4),
            structured_hits=[asdict(match) for match in matches if match.score >= 0.48],
            text_stream=[match.text for match in matches if match.score < 0.48],
        )

    def _empty(self, explanation: str) -> SemanticHybridCandidate:
        return SemanticHybridCandidate(
            answer="",
            explanation=explanation,
            source="semantic_hybrid_no_match",
            confidence=0.0,
            structured_hits=[],
            text_stream=[],
        )


class SemanticStage0Agent:
    """Local Stage 0 agent for weak semantic candidates.

    The agent does not ask an LLM to return the answer. It loops over the
    semantic observations already produced by Stage 0, rescoring them as
    evidence for the current answer format.
    """

    stopwords = {
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
        "it",
        "that",
    }

    def solve(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        matches: list[AttributeMatch],
        prior_candidate: SemanticHybridCandidate,
        gate: SemanticHybridGateResult,
    ) -> SemanticHybridCandidate:
        observations = self._observe(stage_input, matches)
        if classification.get("question_format") == "multiple_choice":
            return self._solve_mcq(classification, observations, matches, gate)
        return self._solve_yes_no(stage_input, observations, matches, prior_candidate, gate)

    def _observe(self, stage_input: Stage0Input, matches: list[AttributeMatch]) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for match in matches:
            observations.append(
                {
                    "text": match.text,
                    "tokens": self._tokens(match.text),
                    "score": match.score,
                    "logic_cues": match.logic_cues,
                    "source": "semantic_match",
                }
            )
        for idx, premise in enumerate(stage_input.premises_nl[:12]):
            observations.append(
                {
                    "text": premise,
                    "tokens": self._tokens(premise),
                    "score": 0.42,
                    "logic_cues": [],
                    "source": f"premise:{idx}",
                }
            )
        return observations

    def _solve_mcq(
        self,
        classification: dict[str, Any],
        observations: list[dict[str, Any]],
        matches: list[AttributeMatch],
        gate: SemanticHybridGateResult,
    ) -> SemanticHybridCandidate:
        options = classification.get("mcq_options") or {}
        if not options:
            return self._empty("Stage 0 agent found no options to inspect.")
        option_scores: dict[str, float] = {}
        option_evidence: dict[str, list[str]] = {}
        for label, text in options.items():
            tokens = self._tokens(text)
            negative = self._negative(text)
            score = 0.0
            evidence: list[str] = []
            for obs in observations:
                overlap = len(tokens & obs["tokens"]) / max(len(tokens), 1)
                if overlap <= 0:
                    continue
                cue_bonus = 0.08 if obs["logic_cues"] else 0.0
                polarity_bonus = 0.05 if negative == self._negative(obs["text"]) else 0.0
                local = overlap * float(obs["score"]) + cue_bonus + polarity_bonus
                if local > 0.1:
                    evidence.append(obs["text"])
                score += local
            option_scores[label] = score
            option_evidence[label] = evidence[:3]
        answer, score = max(option_scores.items(), key=lambda item: item[1])
        sorted_scores = sorted(option_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        confidence = min(0.86, 0.38 + score * 0.2 + max(0.0, margin) * 0.18)
        return SemanticHybridCandidate(
            answer=answer,
            explanation=(
                f"Stage 0 agent selected option {answer} after evidence rescoring. "
                f"score={score:.3f}, margin={margin:.3f}, previous_gate={gate.score:.3f}."
            ),
            source="stage0_semantic_agent_mcq",
            confidence=round(confidence, 4),
            structured_hits=[asdict(match) for match in matches if match.score >= 0.42],
            text_stream=option_evidence.get(answer, []),
        )

    def _solve_yes_no(
        self,
        stage_input: Stage0Input,
        observations: list[dict[str, Any]],
        matches: list[AttributeMatch],
        prior_candidate: SemanticHybridCandidate,
        gate: SemanticHybridGateResult,
    ) -> SemanticHybridCandidate:
        query_tokens = self._tokens(stage_input.question)
        negative_query = self._negative(stage_input.question)
        best_support = 0.0
        best_conflict = 0.0
        evidence: list[str] = []
        for obs in observations:
            overlap = len(query_tokens & obs["tokens"]) / max(len(query_tokens), 1)
            if overlap <= 0:
                continue
            local = overlap * float(obs["score"])
            if negative_query != self._negative(obs["text"]) and overlap >= 0.2:
                best_conflict = max(best_conflict, local)
            else:
                best_support = max(best_support, local)
                if local > 0.08:
                    evidence.append(obs["text"])

        lower_q = stage_input.question.lower()
        asks_requirement = any(cue in lower_q for cue in ["requirement", "requirements", "qualify", "eligible", "meet all", "can "])
        asks_follow = "follow" in lower_q or "according to the premises" in lower_q
        if best_support >= 0.3 and best_support >= best_conflict:
            answer = "Yes"
            reason = "supporting evidence dominated the agent observations"
        elif best_conflict >= 0.24 or asks_requirement:
            answer = "No"
            reason = "required support was missing or conflicting evidence was stronger"
        elif asks_follow:
            answer = "Uncertain"
            reason = "the agent could not establish entailment from semantic observations"
        else:
            answer = normalize_for_eval(prior_candidate.answer) if prior_candidate.answer else "Uncertain"
            reason = "the agent retained the prior answer because evidence was weak"
        confidence = min(0.84, 0.36 + max(best_support, best_conflict) * 0.9)
        return SemanticHybridCandidate(
            answer=answer,
            explanation=(
                f"Stage 0 agent answered {answer}: {reason}. "
                f"support={best_support:.3f}, conflict={best_conflict:.3f}, previous_gate={gate.score:.3f}."
            ),
            source="stage0_semantic_agent_yes_no",
            confidence=round(confidence, 4),
            structured_hits=[asdict(match) for match in matches if match.score >= 0.42],
            text_stream=evidence[:4],
        )

    def _tokens(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_']*", TextTools.clean(text).replace("_", " "))
            if token.lower() not in self.stopwords
        }

    def _negative(self, text: str) -> bool:
        lower = f" {TextTools.clean(text).lower()} "
        return any(cue in lower for cue in [" not ", " no ", " cannot", " can't", " without", " insufficient", " lacks", " lack "])

    def _empty(self, explanation: str) -> SemanticHybridCandidate:
        return SemanticHybridCandidate(
            answer="",
            explanation=explanation,
            source="stage0_semantic_agent_no_decision",
            confidence=0.0,
            structured_hits=[],
            text_stream=[],
        )


class SemanticHybridGate:
    def __init__(self, pass_threshold: float) -> None:
        self.pass_threshold = pass_threshold

    def evaluate(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        candidate: SemanticHybridCandidate,
        matches: list[AttributeMatch],
    ) -> SemanticHybridGateResult:
        reasons: list[str] = []
        score = 0.0
        if classification.get("classification_confidence", 0.0) >= 0.75:
            score += 0.16
            reasons.append("classification_confident")
        if matches:
            score += min(0.34, matches[0].score * 0.34)
            reasons.append("semantic_match_available")
        if candidate.structured_hits:
            score += 0.22
            reasons.append("structured_attribute_hit")
        if candidate.source.startswith("stage0_semantic_agent_") and candidate.answer:
            score += 0.12
            reasons.append("agent_answer_available")
        if candidate.answer:
            score += 0.16
            reasons.append("candidate_answer_available")
        if candidate.text_stream:
            score += 0.06
            reasons.append("unmatched_text_stream_preserved")

        score = round(min(score, 1.0), 4)
        return SemanticHybridGateResult(
            passed=score >= self.pass_threshold,
            score=score,
            reasons=reasons,
        )


class Type1SemanticHybridParser:
    """Stage 0 parser using semantic tree + structured/vector retrieval only."""

    def __init__(
        self,
        config: SemanticHybridConfig | None = None,
        segmenter_client: LLMClient | None = None,
        classifier: Type1QuestionClassifier | None = None,
    ) -> None:
        self.config = config or SemanticHybridConfig()
        self.classifier = classifier or Type1QuestionClassifier()
        rag_config = SemanticRAGConfig(
            segmenter_model=self.config.segmenter_model,
            segmenter_api_base=self.config.segmenter_api_base,
            segmenter_api_key=self.config.segmenter_api_key,
            embedding_model=self.config.embedding_model,
            local_files_only=self.config.local_files_only,
            top_k=self.config.top_k,
            similarity_threshold=self.config.attribute_hit_threshold,
            enable_qwen_segmenter=True,
            enable_bge_embeddings=True,
        )
        self.vector_index = BGEVectorIndex(rag_config)
        self.tree_builder = SemanticTreeBuilder(config=rag_config)
        self.matcher = SemanticHybridMatcher(self.vector_index, self.config)
        self.composer = SemanticHybridAnswerComposer()
        self.agent = SemanticStage0Agent()
        self.gate = SemanticHybridGate(self.config.gate_pass_threshold)

        if segmenter_client is not None:
            from .semantic_tree_rag import QwenSemanticSegmenter

            self.tree_builder.segmenter = QwenSemanticSegmenter(segmenter_client, rag_config)

    def parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        stage_input = self._normalize_payload(payload)
        classification = self.classifier.classify(
            question=stage_input.question,
            answer="",
            premises_nl=stage_input.premises_nl,
            premises_fol=stage_input.premises_fol,
            choices=stage_input.choices,
        )
        tree = self.tree_builder.build(stage_input, classification)
        attr_base = StructuredAttributeBase.from_tree(tree)
        matches = self.matcher.match(stage_input.question, attr_base)
        candidate = self.composer.compose(stage_input, classification, matches)
        gate = self.gate.evaluate(stage_input, classification, candidate, matches)
        agent_metadata = self._maybe_apply_agent(
            stage_input,
            classification,
            matches,
            candidate,
            gate,
        )
        if agent_metadata.get("candidate") is not None:
            candidate = agent_metadata["candidate"]
            gate = self.gate.evaluate(stage_input, classification, candidate, matches)
            agent_metadata["gate_after_agent"] = asdict(gate)

        result = SemanticHybridParseResult(
            query_type="type1",
            stage="stage0",
            method="semantic_tree_hybrid_retrieval",
            normalized_input=asdict(stage_input),
            classification=classification,
            semantic_tree_summary=self._tree_summary(tree),
            structured_attribute_base=attr_base.summary(),
            semantic_matches=[asdict(match) for match in matches],
            branch_output={
                "structured_attribute_hits": candidate.structured_hits,
                "semantic_text_stream": candidate.text_stream,
            },
            candidate=asdict(candidate),
            gate=asdict(gate),
            metadata={
                "segmenter_model": self.config.segmenter_model,
                "embedding_model": self.config.embedding_model,
                "stage0_agent": self._jsonable_agent_metadata(agent_metadata),
                **({"semantic_tree": tree.to_dict()} if self.config.include_tree else {}),
            },
        )
        return result.to_dict()

    def _maybe_apply_agent(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
        matches: list[AttributeMatch],
        candidate: SemanticHybridCandidate,
        gate: SemanticHybridGateResult,
    ) -> dict[str, Any]:
        if not self.config.enable_agent_fallback:
            return {"enabled": False, "used": False}

        should_run_agent = (
            not gate.passed
            or gate.score < self.config.agent_fallback_gate_threshold
            or not candidate.answer
            or not candidate.structured_hits
            or (matches and matches[0].score < self.config.attribute_hit_threshold)
        )
        if not should_run_agent:
            return {
                "enabled": True,
                "used": False,
                "reason": "semantic_candidate_passed_gate",
            }

        agent_candidate = self.agent.solve(
            stage_input,
            classification,
            matches,
            candidate,
            gate,
        )
        return {
            "enabled": True,
            "used": True,
            "reason": "semantic_candidate_failed_gate_or_weak_match",
            "candidate": agent_candidate,
        }

    def _jsonable_agent_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(metadata)
        candidate = cleaned.get("candidate")
        if isinstance(candidate, SemanticHybridCandidate):
            cleaned["candidate"] = asdict(candidate)
        return cleaned

    def _normalize_payload(self, payload: dict[str, Any]) -> Stage0Input:
        premises_nl = payload.get("premises-NL") or payload.get("premises_nl") or payload.get("premises") or []
        premises_fol = payload.get("premises-FOL") or payload.get("premises_fol") or []
        questions = payload.get("questions")
        expected_answer = TextTools.clean(payload.get("answer") or payload.get("expected_answer") or "")
        if isinstance(questions, list) and questions:
            question_index = int(payload.get("_question_idx", payload.get("question_id", 0)) or 0)
            question_index = min(max(question_index, 0), len(questions) - 1)
            question = questions[question_index]
            answers = payload.get("answers", [])
            if isinstance(answers, list) and question_index < len(answers):
                expected_answer = TextTools.clean(answers[question_index])
        else:
            question_index = int(payload.get("question_id", 0) or 0)
            question = payload.get("question", "")

        if not isinstance(premises_nl, list):
            premises_nl = [premises_nl]
        if not isinstance(premises_fol, list):
            premises_fol = [premises_fol]

        return Stage0Input(
            question=TextTools.clean(question),
            premises_nl=[TextTools.clean(item) for item in premises_nl],
            premises_fol=[TextTools.clean(item) for item in premises_fol],
            choices=payload.get("choices", ""),
            expected_answer=expected_answer,
            record_id=payload.get("record_id") or payload.get("id"),
            question_id=payload.get("question_id", question_index),
        )

    def _tree_summary(self, tree: SemanticTree) -> dict[str, Any]:
        node_counts = Counter(node.node_type for node in tree.nodes)
        edge_counts = Counter(edge.edge_type for edge in tree.edges)
        return {
            "node_count": len(tree.nodes),
            "edge_count": len(tree.edges),
            "node_type_counts": dict(node_counts.most_common()),
            "edge_type_counts": dict(edge_counts.most_common()),
        }


def normalize_for_eval(answer: Any) -> str:
    normalized = AnswerNormalizer.normalize(answer)
    if normalized == "Unknown":
        return "Uncertain"
    if normalized.startswith("Option_"):
        return normalized.replace("Option_", "")
    if normalized == "True":
        return "Yes"
    if normalized == "False":
        return "No"
    return normalized
