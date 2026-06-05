#!/usr/bin/env python3
"""Train a Type 1 evaluator with a SAT-style backtracking trace world model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class TextTools:
    @staticmethod
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(TextTools.clean(item) for item in value)
        return re.sub(r"\s+", " ", str(value)).strip()

    @staticmethod
    def safe_get(values: Any, index: int, default: Any = "") -> Any:
        if isinstance(values, list) and 0 <= index < len(values):
            return values[index]
        return default


@dataclass
class QuestionTypeProfile:
    group: str
    features: list[float]


class Type1QuestionClassifier:
    """Deterministic Type 1 grouping used before retained-model ingestion."""

    qualification_words = {"eligible", "qualify", "qualified", "requirements", "meets", "receive", "graduation", "scholarship"}
    conclusion_words = {"conclusion", "follows", "correct", "strongest", "logically"}

    def __init__(self, tokens_fn: Any) -> None:
        self.tokens_fn = tokens_fn

    def classify(self, raw_question: str, stem: str, options: dict[str, str], premises: list[str]) -> QuestionTypeProfile:
        question_text = TextTools.clean(raw_question)
        stem_text = TextTools.clean(stem)
        lower = question_text.lower()
        stem_tokens = set(self.tokens_fn(stem_text))
        premise_tokens = set(self.tokens_fn(" ".join(premises)))
        is_mcq = any(label in options for label in ("A", "B", "C", "D"))
        is_judgment = not is_mcq
        explicit_option_text = question_text if is_mcq else ""
        option_text = " ".join(options.values())
        option_tokens = set(self.tokens_fn(option_text))
        has_strongest = "strongest conclusion" in lower or "fewest premises" in lower
        asks_correct_conclusion = "which conclusion" in lower or "what is the correct conclusion" in lower
        asks_follow = "does it follow" in lower or "logically follows" in lower or "according to the premises" in lower
        asks_qualification = bool((stem_tokens | option_tokens) & self.qualification_words)
        explicit_option_tokens = set(self.tokens_fn(explicit_option_text))
        has_negation = bool((stem_tokens | explicit_option_tokens | premise_tokens) & BacktrackingTraceGenerator.neg_words)
        has_uncertainty_language = any(phrase in lower for phrase in ("cannot be determined", "not provide enough", "unknown", "uncertain"))
        rule_hits = len(premise_tokens & BacktrackingTraceGenerator.rule_words)
        modal_hits = len((stem_tokens | option_tokens | premise_tokens) & BacktrackingTraceGenerator.modal_words)
        option_count = len(options)
        premise_count = len(premises)

        if is_mcq and has_strongest:
            task = "strongest_conclusion"
        elif is_mcq and asks_correct_conclusion:
            task = "conclusion_selection"
        elif is_judgment and asks_qualification:
            task = "qualification_judgment"
        elif is_judgment and asks_follow:
            task = "entailment_judgment"
        elif is_judgment:
            task = "truth_judgment"
        else:
            task = "choice_selection"

        logic_shape = "negation" if has_negation else "rule_chain" if rule_hits >= 2 else "fact_match"
        uncertainty_shape = "uncertainty_gap" if has_uncertainty_language else "closed"
        answer_format = "mcq" if is_mcq else "judgment"
        group = f"{answer_format}:{task}:{logic_shape}:{uncertainty_shape}"
        features = [
            float(is_mcq),
            float(is_judgment),
            float(has_strongest),
            float(asks_correct_conclusion),
            float(asks_follow),
            float(asks_qualification),
            float(has_negation),
            float(has_uncertainty_language),
            min(rule_hits / 6.0, 1.5),
            min(modal_hits / 8.0, 1.5),
            min(premise_count / 8.0, 2.0),
            min(option_count / 5.0, 1.5),
            min(len(stem_tokens) / 40.0, 2.0),
            min(len(option_tokens) / 80.0, 2.0),
            float(task in {"strongest_conclusion", "conclusion_selection"}),
            float(logic_shape == "rule_chain"),
            float(logic_shape == "negation"),
            1.0,
        ]
        return QuestionTypeProfile(group=group, features=features)


class Type1RAGMemory:
    """Small local RAG memory for recurring Type 1 failure modes.

    The retrieved vector is appended to every trace step, so the SSM consumes
    it as part of the latent trace rather than as an external graph.
    """

    prototypes = {
        "quantifier": "all every each exists there exists at least for all not every universal existential implication",
        "symbolic": "forall exists predicate arrow implication negation conjunction disjunction symbolic formula",
        "numeric": "greater less at least percent percentage credits score threshold ratio total more than minimum",
        "open_ended_invalid": "what which how why describe explain achieve qualifies open ended missing yes no options",
    }
    memory_items = [
        ("quantifier", "universal quantifier all every each student employee vehicle object statement"),
        ("quantifier", "existential quantifier exists there exists at least one not every cannot infer all from some"),
        ("quantifier", "universal conclusion requires closed implication chain from premise to answer"),
        ("symbolic", "ForAll Exists predicate logic symbolic formula arrow implication negation conjunction"),
        ("symbolic", "forall x predicate implication not exists formula option needs symbolic parser"),
        ("numeric", "numeric entailment compare score threshold percentage credits total more than at least"),
        ("numeric", "ratio computation 80 credits out of 120 greater than 65 percent"),
        ("open_ended_invalid", "open ended what which how describe explain question is not yes no uncertain"),
        ("open_ended_invalid", "definition question asks what a clause achieves or what qualifies a student"),
    ]
    quantifier_patterns = re.compile(r"\b(all|every|each|exists|there exists|at least|not every|for all)\b|∀|∃|ForAll|Exists", re.I)
    symbolic_patterns = re.compile(r"∀|∃|¬|→|->|ForAll|Exists|\b[A-Z][A-Za-z]*\s*\(")
    numeric_patterns = re.compile(r"\b\d+(?:\.\d+)?%?\b|\b(percent|percentage|credits?|score|threshold|greater|less|more than|at least|minimum)\b", re.I)
    open_ended_patterns = re.compile(r"^\s*(what|which|how|why|who|where|when|describe|explain)\b", re.I)
    yes_no_patterns = re.compile(r"^\s*(does|do|is|are|can|could|should|would|if)\b", re.I)

    def __init__(
        self,
        tokens_fn: Any,
        backend: str = "auto",
        bge_model: str = "BAAI/bge-small-en-v1.5",
        bge_local_files_only: bool = True,
    ) -> None:
        self.tokens_fn = tokens_fn
        self.prototype_tokens = {key: set(tokens_fn(value)) for key, value in self.prototypes.items()}
        self.memory_labels = [label for label, _text in self.memory_items]
        self.memory_texts = [text for _label, text in self.memory_items]
        self.vector_backend = "numpy"
        self.vectorizer = None
        self.vector_index = None
        self.memory_vectors = None
        self.bge_model = None
        self.embedding_cache: dict[str, np.ndarray] = {}
        if backend not in {"auto", "tfidf", "bge", "numpy"}:
            raise ValueError("rag_backend must be one of: auto, tfidf, bge, numpy")
        if backend in {"auto", "bge"} and self._try_init_bge(bge_model, bge_local_files_only):
            return
        if backend == "bge":
            print(f"[rag] BGE backend unavailable for {bge_model}; falling back to TF-IDF.")
        if backend in {"auto", "tfidf", "bge"}:
            self._try_init_tfidf()
        if self.vector_backend == "numpy":
            self.memory_vectors = [set(tokens_fn(text)) for text in self.memory_texts]

    def _try_init_bge(self, model_name: str, local_files_only: bool) -> bool:
        try:
            from sentence_transformers import SentenceTransformer

            self.bge_model = SentenceTransformer(model_name, local_files_only=local_files_only)
            self.memory_vectors = self.bge_model.encode(
                self.memory_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            self.vector_backend = "bge"
            return True
        except Exception as exc:
            print(f"[rag] BGE init failed: {type(exc).__name__}: {exc}")
            self.bge_model = None
            return False

    def _try_init_tfidf(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.neighbors import NearestNeighbors

            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            self.memory_vectors = self.vectorizer.fit_transform(self.memory_texts)
            self.vector_index = NearestNeighbors(n_neighbors=min(4, len(self.memory_texts)), metric="cosine")
            self.vector_index.fit(self.memory_vectors)
            self.vector_backend = "sklearn"
        except Exception:
            self.vector_backend = "numpy"

    def retrieve(
        self,
        answer: str,
        option_text: str,
        stem: str,
        raw_question: str,
        premises: list[str],
        options: dict[str, str],
        question_type: QuestionTypeProfile,
    ) -> tuple[list[float], list[str]]:
        context = TextTools.clean(" ".join([raw_question, option_text, " ".join(premises)]))
        stem_text = TextTools.clean(stem)
        has_mcq_options = any(label in options for label in ("A", "B", "C", "D"))
        context_tokens = set(self.tokens_fn(context))
        option_tokens = set(self.tokens_fn(option_text))
        premise_tokens = set(self.tokens_fn(" ".join(premises)))
        question_tokens = set(self.tokens_fn(stem_text))

        quantifier = float(bool(self.quantifier_patterns.search(context)))
        symbolic = float(bool(self.symbolic_patterns.search(context)))
        numeric = float(bool(self.numeric_patterns.search(context)))
        open_ended = float((not has_mcq_options) and bool(self.open_ended_patterns.search(raw_question)) and not bool(self.yes_no_patterns.search(raw_question)))
        similarities = [self._similarity(context_tokens, self.prototype_tokens[key]) for key in ("quantifier", "symbolic", "numeric", "open_ended_invalid")]
        vector_class_scores, vector_top_scores, retrieved_tags = self._vector_retrieve(context, context_tokens)
        numeric_truth = self._numeric_truth(context)
        implication_closure = self._implication_closure(option_tokens | question_tokens, premise_tokens, context)
        existential_query = float(bool(re.search(r"Exists|∃|\bexists\b|\bthere exists\b|\bat least one\b", raw_question, flags=re.I)))
        universal_query = float(bool(re.search(r"ForAll|∀|\ball\b|\bevery\b|\beach\b", raw_question, flags=re.I)))
        option_premise_overlap = self._overlap(option_tokens, premise_tokens)
        question_premise_overlap = self._overlap(question_tokens, premise_tokens)
        symbolic_option = float(bool(self.symbolic_patterns.search(option_text)))
        answer_yes = float(answer == "Yes")
        answer_no = float(answer == "No")
        answer_uncertain = float(answer == "Uncertain")
        answer_mcq = float(answer in {"A", "B", "C", "D"})
        unsupported_existential = existential_query * float("exists" not in premise_tokens and "there" not in premise_tokens)
        open_uncertain_support = open_ended * answer_uncertain
        numeric_yes_support = numeric * numeric_truth * answer_yes
        quantifier_uncertain_support = quantifier * answer_uncertain * max(0.0, 1.0 - implication_closure)
        positive_entailment_support = (answer_yes + answer_mcq) * max(implication_closure, option_premise_overlap)
        contradiction_support = answer_no * max(float(bool({"not", "no", "never", "cannot", "false"} & (option_tokens | question_tokens))), unsupported_existential)
        rag_confidence = max(similarities + [open_uncertain_support, numeric_yes_support, positive_entailment_support, contradiction_support])
        features = [
            quantifier,
            symbolic,
            numeric,
            open_ended,
            *similarities,
            *vector_class_scores,
            *vector_top_scores,
            float(self.vector_backend == "sklearn"),
            float(self.vector_backend == "bge"),
            existential_query,
            universal_query,
            symbolic_option,
            option_premise_overlap,
            question_premise_overlap,
            implication_closure,
            numeric_truth,
            unsupported_existential,
            open_uncertain_support,
            numeric_yes_support,
            quantifier_uncertain_support,
            positive_entailment_support,
            contradiction_support,
            answer_yes,
            answer_no,
            answer_uncertain,
            answer_mcq,
            rag_confidence,
            min(sum(question_type.features[:8]) / 8.0, 1.0),
            1.0,
        ]
        tags = [
            name
            for name, active in [
                ("quantifier", quantifier),
                ("symbolic", symbolic),
                ("numeric", numeric),
                ("open_ended_invalid", open_ended),
            ]
            if active
        ]
        tags.extend(retrieved_tags)
        tags = sorted(set(tags))
        return features, tags

    def _vector_retrieve(self, context: str, context_tokens: set[str]) -> tuple[list[float], list[float], list[str]]:
        class_order = ("quantifier", "symbolic", "numeric", "open_ended_invalid")
        class_scores = {key: 0.0 for key in class_order}
        top_scores = [0.0, 0.0, 0.0, 0.0]
        retrieved_tags: list[str] = []
        if self.vector_backend == "bge" and self.bge_model is not None and isinstance(self.memory_vectors, np.ndarray):
            query = self._bge_encode(context)
            scores = np.matmul(self.memory_vectors, query)
            top_indices = np.argsort(-scores)[:4]
            for rank, index in enumerate(top_indices):
                score = max(0.0, float(scores[int(index)]))
                label = self.memory_labels[int(index)]
                class_scores[label] = max(class_scores[label], score)
                top_scores[rank] = score
                if score > 0.25:
                    retrieved_tags.append(label)
        elif self.vector_backend == "sklearn" and self.vectorizer is not None and self.vector_index is not None:
            query = self.vectorizer.transform([context])
            distances, indices = self.vector_index.kneighbors(query, n_neighbors=min(4, len(self.memory_texts)))
            for rank, (distance, index) in enumerate(zip(distances[0], indices[0])):
                score = max(0.0, 1.0 - float(distance))
                label = self.memory_labels[int(index)]
                class_scores[label] = max(class_scores[label], score)
                top_scores[rank] = score
                if score > 0.05:
                    retrieved_tags.append(label)
        else:
            scored = []
            for idx, memory_tokens in enumerate(self.memory_vectors or []):
                score = self._similarity(context_tokens, memory_tokens)
                scored.append((score, idx))
            for rank, (score, index) in enumerate(sorted(scored, reverse=True)[:4]):
                label = self.memory_labels[index]
                class_scores[label] = max(class_scores[label], score)
                top_scores[rank] = score
                if score > 0.0:
                    retrieved_tags.append(label)
        return [class_scores[key] for key in class_order], top_scores, retrieved_tags

    def _bge_encode(self, text: str) -> np.ndarray:
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        if self.bge_model is None:
            raise RuntimeError("BGE model is not initialized.")
        vector = self.bge_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        self.embedding_cache[text] = vector
        return vector

    def _similarity(self, left: set[str], right: set[str]) -> float:
        return len(left & right) / max(1, len(left | right))

    def _overlap(self, left: set[str], right: set[str]) -> float:
        return len(left & right) / max(1, len(left))

    def _implication_closure(self, target_tokens: set[str], premise_tokens: set[str], context: str) -> float:
        rule_hits = len(premise_tokens & BacktrackingTraceGenerator.rule_words)
        overlap = self._overlap(target_tokens, premise_tokens)
        chain_markers = len(re.findall(r"\b(if|then|implies|therefore|because|ensures|leads to|qualifies|eligible|allowed)\b|→|->", context, flags=re.I))
        return min(1.0, 0.55 * overlap + 0.12 * rule_hits + 0.08 * chain_markers)

    def _numeric_truth(self, context: str) -> float:
        percentages = [float(item) / 100.0 for item in re.findall(r"(\d+(?:\.\d+)?)\s*%", context)]
        plain_numbers = [
            float(item)
            for item in re.findall(r"\b\d+(?:\.\d+)?\b", context)
            if f"{item}%" not in context
        ]
        if not percentages or len(plain_numbers) < 2:
            return 0.0
        threshold = max(percentages)
        ratios = []
        for numerator in plain_numbers:
            for denominator in plain_numbers:
                if denominator > numerator > 0:
                    ratios.append(numerator / denominator)
        if not ratios:
            return 0.0
        best_ratio = max(ratios)
        return float(best_ratio >= threshold)


def normalize_for_eval(answer: Any) -> str:
    raw = TextTools.clean(answer)
    lowered = raw.lower()
    if lowered in {"yes", "true"}:
        return "Yes"
    if lowered in {"no", "false"}:
        return "No"
    if lowered in {"uncertain", "unknown", "cannot be determined", "can't be determined"}:
        return "Uncertain"
    upper = raw.upper()
    if upper in {"A", "B", "C", "D"}:
        return upper
    return raw or "Uncertain"


@dataclass
class BrainPrediction:
    record_id: int
    question_id: int
    question_group: str
    expected: str
    predicted_answer: str
    correct: bool
    candidate_probabilities: dict[str, float]
    model_predicted_answer: str | None = None
    model_top_probability: float | None = None
    fallback_used: bool = False
    fallback_reason: str = ""
    fallback_error: str = ""


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    return value


@dataclass
class BacktrackingTraceConfig:
    input_path: Path = Path("../Logic_Based_Educational_Queries.json")
    output_path: Path = Path("type1_backtracking_trace_eval_results.json")
    summary_output_path: Path = Path("type1_backtracking_trace_eval_summary.json")
    model_output_path: Path = Path("type1_backtracking_trace_model.json")
    train_ratio: float = 0.8
    random_state: int = 42
    epochs: int = 120
    learning_rate: float = 0.0007
    l2: float = 0.001
    hidden_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_dim: int = 256
    dropout: float = 0.15
    batch_size: int = 32
    patience: int = 25
    min_delta: float = 0.0005
    max_trace_steps: int = 18
    ssm_block_size: int = 8
    local_attention_window: int = 8
    propagation_top_k: int = 4
    implicit_bias_strength: float = 0.18
    consistency_loss_weight: float = 0.0
    grouped_batches: bool = True
    device: str = "auto"
    limit_records: int | None = None
    rag_backend: str = "auto"
    bge_model: str = "BAAI/bge-small-en-v1.5"
    bge_local_files_only: bool = True
    llm_fallback: bool = False
    llm_fallback_model: str = "minigpt"
    llm_fallback_base_url: str = "https://api.openai.com/v1"
    llm_fallback_api_key_env: str = "MINIGPT_API_KEY"
    llm_fallback_threshold: float = 0.62
    llm_fallback_on_uncertain: bool = True
    llm_fallback_timeout: float = 45.0


@dataclass
class TraceStep:
    action: str
    features: list[float]
    conflict: float
    description: str


@dataclass
class BacktrackingCandidate:
    key: str
    record_id: int
    question_id: int
    answer: str
    expected: str
    candidate_features: list[float]
    trace_features: list[list[float]]
    trace_text: list[str]
    question_group: str
    question_type_features: list[float]
    raw_question: str = ""
    stem: str = ""
    premises: list[str] | None = None
    options: dict[str, str] | None = None


@dataclass
class LLMFallbackResult:
    answer: str
    raw_answer: str
    error: str = ""


class LLMFallbackClient:
    def __init__(self, config: BacktrackingTraceConfig) -> None:
        self.config = config
        self._load_dotenv(Path(".env"))
        self.api_key = os.getenv(config.llm_fallback_api_key_env, "")

    def available(self) -> bool:
        return bool(self.api_key)

    def choose_answer(self, group: list[BacktrackingCandidate]) -> LLMFallbackResult:
        if not self.api_key:
            return LLMFallbackResult(answer="", raw_answer="", error=f"missing env {self.config.llm_fallback_api_key_env}")
        prompt = self._prompt(group)
        payload = {
            "model": self.config.llm_fallback_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict logic solver. Answer only with one label from the allowed labels. "
                        "Use Uncertain when the premises do not entail or contradict the statement."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        url = self.config.llm_fallback_base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.llm_fallback_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            return LLMFallbackResult(answer="", raw_answer="", error=f"HTTP {exc.code}: {detail}")
        except Exception as exc:
            return LLMFallbackResult(answer="", raw_answer="", error=f"{type(exc).__name__}: {exc}")

        raw = TextTools.clean(body.get("choices", [{}])[0].get("message", {}).get("content", ""))
        allowed = [candidate.answer for candidate in group]
        answer = self._extract_answer(raw, allowed)
        if not answer:
            return LLMFallbackResult(answer="", raw_answer=raw, error="could not parse allowed answer")
        return LLMFallbackResult(answer=answer, raw_answer=raw)

    def _prompt(self, group: list[BacktrackingCandidate]) -> str:
        first = group[0]
        premises = first.premises or []
        options = first.options or {candidate.answer: candidate.answer for candidate in group}
        allowed = [candidate.answer for candidate in group]
        premise_text = "\n".join(f"{idx + 1}. {premise}" for idx, premise in enumerate(premises[:40]))
        option_text = "\n".join(f"{label}. {text}" for label, text in options.items())
        return (
            f"Premises:\n{premise_text}\n\n"
            f"Question:\n{first.raw_question or first.stem}\n\n"
            f"Allowed answer labels: {', '.join(allowed)}\n"
            f"Options/labels:\n{option_text}\n\n"
            "Return exactly one allowed answer label and nothing else."
        )

    def _extract_answer(self, raw: str, allowed: list[str]) -> str:
        normalized = normalize_for_eval(raw)
        if normalized in allowed:
            return normalized
        lowered = raw.lower()
        for answer in allowed:
            if re.search(rf"\b{re.escape(answer.lower())}\b", lowered):
                return answer
        if "unknown" in lowered or "uncertain" in lowered or "cannot be determined" in lowered:
            return "Uncertain" if "Uncertain" in allowed else ""
        if re.search(r"\byes\b|\btrue\b", lowered):
            return "Yes" if "Yes" in allowed else ""
        if re.search(r"\bno\b|\bfalse\b", lowered):
            return "No" if "No" in allowed else ""
        return ""

    def _load_dotenv(self, path: Path) -> None:
        if not path.exists():
            package_env = Path(__file__).resolve().parents[1] / ".env"
            path = package_env
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


class BacktrackingTraceGenerator:
    """Builds a compact CDCL-like trace from premise/answer lexical evidence.

    This is not a symbolic SAT solver. It gives the neural WM explicit search
    states: choose answer hypothesis, propagate premise evidence, detect/learn
    conflicts, backtrack, and finalize.
    """

    neg_words = {"not", "no", "never", "without", "cannot", "contradict", "false", "n't", "except"}
    modal_words = {"must", "may", "might", "can", "could", "should", "eligible", "qualify", "follows", "requires"}
    rule_words = {"if", "then", "all", "only", "unless", "every", "requires", "implies"}

    def __init__(self, max_steps: int, propagation_top_k: int) -> None:
        self.max_steps = max_steps
        self.propagation_top_k = propagation_top_k

    def build(
        self,
        answer: str,
        option_text: str,
        question: str,
        premises: list[str],
        tokens_fn: Any,
        overlap_fn: Any,
        jaccard_fn: Any,
    ) -> list[TraceStep]:
        answer_tokens = set(tokens_fn(f"{answer} {option_text} {question}"))
        question_tokens = set(tokens_fn(question))
        premise_items = []
        for idx, premise in enumerate(premises):
            premise_tokens = set(tokens_fn(premise))
            support = jaccard_fn(answer_tokens | question_tokens, premise_tokens)
            answer_support = overlap_fn(answer_tokens, premise_tokens)
            question_support = overlap_fn(question_tokens, premise_tokens)
            conflict = self._conflict_score(answer, answer_tokens, question_tokens, premise_tokens)
            rule_strength = min(1.0, len(premise_tokens & self.rule_words) / 2.0)
            premise_items.append((support + answer_support + question_support + rule_strength * 0.08, idx, premise, premise_tokens, conflict, rule_strength))
        premise_items.sort(reverse=True, key=lambda item: item[0])

        steps: list[TraceStep] = [
            self._step(
                "select_variable",
                depth=0,
                branch=1,
                support=0.0,
                conflict=0.0,
                rule_strength=0.0,
                premise_position=0.0,
                answer=answer,
                description=f"select answer variable {answer}",
            ),
            self._step(
                "try_assignment",
                depth=1,
                branch=1,
                support=0.0,
                conflict=0.0,
                rule_strength=0.0,
                premise_position=0.0,
                answer=answer,
                description=f"try {answer}=true",
            ),
        ]

        top = premise_items[: self.propagation_top_k]
        max_conflict = 0.0
        max_support = 0.0
        for rank, (score, idx, _premise, _tokens, conflict, rule_strength) in enumerate(top):
            max_conflict = max(max_conflict, conflict)
            max_support = max(max_support, score)
            steps.append(
                self._step(
                    "propagate",
                    depth=1 + rank / max(1, self.propagation_top_k),
                    branch=1,
                    support=score,
                    conflict=conflict,
                    rule_strength=rule_strength,
                    premise_position=idx / max(1, len(premises) - 1),
                    answer=answer,
                    description=f"propagate premise {idx} under {answer}=true",
                )
            )

        strong_conflict = max_conflict >= 0.42
        weak_support = max_support < 0.18
        if strong_conflict or (answer == "Uncertain" and weak_support):
            steps.extend(
                [
                    self._step(
                        "detect_conflict",
                        depth=2,
                        branch=1,
                        support=max_support,
                        conflict=max_conflict,
                        rule_strength=0.0,
                        premise_position=0.0,
                        answer=answer,
                        description=f"detect conflict for {answer}=true",
                    ),
                    self._step(
                        "learn_conflict",
                        depth=2,
                        branch=1,
                        support=max_support,
                        conflict=max_conflict,
                        rule_strength=0.5,
                        premise_position=0.0,
                        answer=answer,
                        description=f"learn reason for rejected {answer}=true branch",
                    ),
                    self._step(
                        "backtrack",
                        depth=1,
                        branch=-1,
                        support=max_support,
                        conflict=max_conflict,
                        rule_strength=0.5,
                        premise_position=0.0,
                        answer=answer,
                        description=f"backtrack and try {answer}=false",
                    ),
                ]
            )
            for rank, (score, idx, _premise, _tokens, conflict, rule_strength) in enumerate(top[:2]):
                false_conflict = max(0.0, score - conflict)
                steps.append(
                    self._step(
                        "propagate",
                        depth=1.5 + rank / 2.0,
                        branch=-1,
                        support=conflict,
                        conflict=false_conflict,
                        rule_strength=rule_strength,
                        premise_position=idx / max(1, len(premises) - 1),
                        answer=answer,
                        description=f"propagate premise {idx} under {answer}=false",
                    )
                )

        final_conflict = max_conflict if answer != "Uncertain" else max(0.0, 0.35 - max_support)
        steps.append(
            self._step(
                "finalize",
                depth=0,
                branch=0,
                support=max_support,
                conflict=final_conflict,
                rule_strength=float(len(top) > 0),
                premise_position=0.0,
                answer=answer,
                description=f"finalize search for {answer}",
            )
        )
        return steps[: self.max_steps]

    def _step(
        self,
        action: str,
        depth: float,
        branch: float,
        support: float,
        conflict: float,
        rule_strength: float,
        premise_position: float,
        answer: str,
        description: str,
    ) -> TraceStep:
        features = [
            min(depth / 4.0, 1.5),
            branch,
            min(max(support, 0.0), 1.5),
            min(max(conflict, 0.0), 1.5),
            min(max(rule_strength, 0.0), 1.0),
            min(max(premise_position, 0.0), 1.0),
            float(answer == "Yes"),
            float(answer == "No"),
            float(answer == "Uncertain"),
            float(action in {"detect_conflict", "learn_conflict", "backtrack"}),
            1.0,
        ]
        return TraceStep(action=action, features=features, conflict=float(conflict > 0.35), description=description)

    def _conflict_score(self, answer: str, answer_tokens: set[str], question_tokens: set[str], premise_tokens: set[str]) -> float:
        answer_negative = bool(answer_tokens & self.neg_words) or answer == "No"
        question_negative = bool(question_tokens & self.neg_words)
        premise_negative = bool(premise_tokens & self.neg_words)
        polarity_conflict = float((answer_negative or question_negative) != premise_negative)
        if answer == "Uncertain":
            return 0.25 * (1.0 - min(1.0, len((answer_tokens | question_tokens) & premise_tokens) / 4.0))
        return 0.55 * polarity_conflict + 0.25 * float(answer == "Yes" and premise_negative) + 0.25 * float(answer == "No" and not premise_negative)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.value = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.dropout(nn.functional.silu(self.gate(x)) * self.value(x)))


class ConditionedAdaLN(nn.Module):
    """AdaLN-style normalization conditioned on the current token block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.modulation = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + torch.tanh(scale[:, None, :])) + shift[:, None, :]


class ConditionedResidualScale(nn.Module):
    """Scale block outputs using the same condition path as the figure."""

    def __init__(self, dim: int, initial: float = 0.1) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.full((dim,), initial))
        self.modulation = nn.Linear(dim, dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = self.base[None, None, :] * (1.0 + torch.tanh(self.modulation(condition))[:, None, :])
        return x * scale


class BlockWiseSSM(nn.Module):
    """Independent bidirectional SSM scans inside fixed token blocks."""

    def __init__(self, dim: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.block_size = max(1, block_size)
        self.input_gate = nn.Linear(dim, dim)
        self.state_gate = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim * 2, dim)
        self.out = nn.Linear(dim * 3, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        chunks = []
        for start in range(0, x.shape[1], self.block_size):
            end = min(start + self.block_size, x.shape[1])
            block = x[:, start:end]
            block_mask = mask[:, start:end]
            forward = self._scan(block, block_mask)
            backward = torch.flip(self._scan(torch.flip(block, dims=[1]), torch.flip(block_mask, dims=[1])), dims=[1])
            chunks.append(self.out(torch.cat((block, forward, backward), dim=-1)))
        return self.dropout(torch.cat(chunks, dim=1)) * mask.unsqueeze(-1).to(x.dtype)

    def _scan(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        state = torch.zeros_like(x[:, 0])
        states: list[torch.Tensor] = []
        for idx in range(x.shape[1]):
            item = x[:, idx]
            valid = mask[:, idx].unsqueeze(-1).to(item.dtype)
            update = torch.sigmoid(self.input_gate(item) + self.state_gate(state))
            proposal = torch.tanh(self.value(torch.cat((item, state), dim=-1)))
            next_state = update * state + (1.0 - update) * proposal
            state = next_state * valid + state * (1.0 - valid)
            states.append(state * valid)
        return torch.stack(states, dim=1)


class ImplicitSATFlowWorldModel(nn.Module):
    """Latent SAT-search flow without explicit action/conflict readout."""

    def __init__(self, trace_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.trace_encoder = nn.Sequential(
            nn.LayerNorm(trace_dim),
            nn.Linear(trace_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.decision_gate = nn.Linear(hidden_dim, hidden_dim)
        self.propagation_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.conflict_gate = nn.Linear(hidden_dim, hidden_dim)
        self.backtrack_gate = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trace_features: torch.Tensor, trace_mask: torch.Tensor) -> torch.Tensor:
        sequence = self.forward_sequence(trace_features, trace_mask)
        counts = trace_mask.sum(dim=1, keepdim=True).clamp_min(1).to(sequence.dtype)
        mean_state = (sequence * trace_mask.unsqueeze(-1).to(sequence.dtype)).sum(dim=1) / counts
        max_state = sequence.masked_fill(~trace_mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_state = torch.where(torch.isfinite(max_state), max_state, torch.zeros_like(max_state))
        final_state = sequence.gather(
            1,
            (trace_mask.sum(dim=1).clamp_min(1) - 1).reshape(-1, 1, 1).expand(-1, 1, sequence.shape[-1]),
        ).squeeze(1)
        return self.dropout(self.out(torch.cat((final_state, mean_state, max_state), dim=-1)))

    def forward_sequence(self, trace_features: torch.Tensor, trace_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.trace_encoder(trace_features)
        state = torch.zeros_like(encoded[:, 0])
        conflict_memory = torch.zeros_like(encoded[:, 0])
        states: list[torch.Tensor] = []
        for idx in range(encoded.shape[1]):
            item = encoded[:, idx]
            valid = trace_mask[:, idx].unsqueeze(-1).to(item.dtype)
            decision = torch.sigmoid(self.decision_gate(item) + self.propagation_gate(state))
            conflict = torch.sigmoid(self.conflict_gate(item))
            backtrack = torch.sigmoid(self.backtrack_gate(item))
            proposal = torch.tanh(self.value(torch.cat((item, conflict_memory), dim=-1)))
            conflict_memory = (0.82 * conflict_memory + conflict * proposal) * valid
            state = (decision * state + (1.0 - decision) * proposal - 0.35 * backtrack * conflict_memory) * valid
            states.append(state)
        return self.dropout(torch.stack(states, dim=1))


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, local_window: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads.")
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even per-head dimension.")
        self.local_window = max(0, local_window)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self._rope(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_bias is not None:
            scores = scores + attention_bias[:, None, :, :]
        elif self.local_window > 0 and seq_len > self.local_window:
            positions = torch.arange(seq_len, device=x.device)
            local_mask = (positions[:, None] - positions[None, :]).abs() <= self.local_window
            scores = scores.masked_fill(~local_mask[None, None, :, :], -1e9)
        scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attention, v)
        return self.out(context.transpose(1, 2).contiguous().view(batch, seq_len, dim))

    def _rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        half_dim = self.head_dim // 2
        dtype = q.dtype
        device = q.device
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device, dtype=dtype) / half_dim))
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        angles = torch.outer(positions, inv_freq)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]
        q_rot = torch.stack((q_even * cos - q_odd * sin, q_even * sin + q_odd * cos), dim=-1).flatten(-2)
        k_rot = torch.stack((k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1).flatten(-2)
        return q_rot, k_rot


class TraceTransformerBlock(nn.Module):
    def __init__(self, config: BacktrackingTraceConfig) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.ssm_norm = ConditionedAdaLN(dim)
        self.ssm = BlockWiseSSM(dim, config.ssm_block_size, config.dropout)
        self.ssm_scale = ConditionedResidualScale(dim)
        self.attn_norm = ConditionedAdaLN(dim)
        self.attn = RoPEMultiHeadAttention(dim, config.transformer_heads, config.dropout, config.local_attention_window)
        self.attn_scale = ConditionedResidualScale(dim)
        self.ffn_norm = ConditionedAdaLN(dim)
        self.ffn = SwiGLU(dim, config.transformer_ff_dim, config.dropout)
        self.ffn_scale = ConditionedResidualScale(dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, attention_bias: torch.Tensor | None = None) -> torch.Tensor:
        condition = self._condition(x, mask)
        ssm_out = self.ssm(self.ssm_norm(x, condition), mask)
        x = x + self.ssm_scale(self.dropout(ssm_out), condition)
        condition = self._condition(x, mask)
        attn_out = self.attn(self.attn_norm(x, condition), mask, attention_bias=attention_bias)
        x = x + self.attn_scale(self.dropout(attn_out), condition)
        condition = self._condition(x, mask)
        ffn_out = self.ffn(self.ffn_norm(x, condition))
        x = x + self.ffn_scale(self.dropout(ffn_out), condition)
        return x

    def _condition(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
        return (x * mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / counts


class TraceSSMOnlyReasoner(nn.Module):
    """Trace/SSM-only reasoner without the global SAT graph path."""

    def __init__(self, candidate_dim: int, trace_dim: int, config: BacktrackingTraceConfig) -> None:
        super().__init__()
        dim = config.hidden_dim
        self.bias_strength = config.implicit_bias_strength
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(candidate_dim),
            nn.Linear(candidate_dim, dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.world_model = ImplicitSATFlowWorldModel(trace_dim, dim, config.dropout)
        self.trace_blocks = nn.ModuleList(TraceTransformerBlock(config) for _ in range(max(1, config.transformer_layers)))
        self.candidate_blocks = nn.ModuleList(TraceTransformerBlock(config) for _ in range(max(1, config.transformer_layers)))
        self.effect_projection = nn.Linear(dim, dim)
        self.cause_projection = nn.Linear(dim, dim)
        self.prior_scorer = nn.Linear(dim, 1)
        self.scorer = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        trace_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        trace_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        batch, candidates, steps, trace_dim = trace_features.shape
        candidate_state = self.candidate_encoder(candidate_features)
        flat_candidate_state = candidate_state.reshape(batch * candidates, -1)
        flat_trace = trace_features.reshape(batch * candidates, steps, trace_dim)
        flat_trace_mask = trace_mask.reshape(batch * candidates, steps)
        trace_sequence = self.world_model.forward_sequence(flat_trace, flat_trace_mask)
        local_sequence = torch.cat((flat_candidate_state[:, None, :], trace_sequence), dim=1)
        local_mask = torch.cat(
            (
                torch.ones((batch * candidates, 1), dtype=torch.bool, device=trace_mask.device),
                flat_trace_mask,
            ),
            dim=1,
        )
        for block in self.trace_blocks:
            local_sequence = block(local_sequence, local_mask)
        candidate_trace_state = local_sequence[:, 0].reshape(batch, candidates, -1)
        attention_bias = self._implicit_attention_bias(candidate_trace_state, candidate_mask)
        hidden = candidate_trace_state
        for block in self.candidate_blocks:
            hidden = block(hidden, candidate_mask, attention_bias=attention_bias)
        scores = self.scorer(hidden).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        return scores, None, None

    def _implicit_attention_bias(self, trace_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        effect = self.effect_projection(trace_state)
        cause = self.cause_projection(trace_state)
        causal_scores = torch.matmul(effect, cause.transpose(-2, -1)) / math.sqrt(trace_state.shape[-1])
        prior_logits = self.prior_scorer(trace_state).squeeze(-1).masked_fill(~mask, -1e9)
        prior = torch.softmax(prior_logits, dim=-1).clamp_min(1e-6).log()
        pair_mask = mask[:, :, None] & mask[:, None, :]
        bias = (causal_scores + prior[:, None, :]).masked_fill(~pair_mask, 0.0)
        return self.bias_strength * torch.tanh(bias)


class BacktrackingTraceTrainer:
    def __init__(self, config: BacktrackingTraceConfig) -> None:
        self.config = config
        self.device = self._device()
        self.trace_generator = BacktrackingTraceGenerator(config.max_trace_steps, config.propagation_top_k)
        self.question_classifier = Type1QuestionClassifier(self._tokens)
        self.rag_memory = Type1RAGMemory(
            self._tokens,
            backend=config.rag_backend,
            bge_model=config.bge_model,
            bge_local_files_only=config.bge_local_files_only,
        )
        self.llm_fallback = LLMFallbackClient(config) if config.llm_fallback else None

    def run(self) -> dict[str, Any]:
        self._set_seed()
        started = time.time()
        records = json.loads(self.config.input_path.read_text(encoding="utf-8"))
        if self.config.limit_records is not None:
            records = records[: self.config.limit_records]
        train_ids, val_ids = self._split_ids(len(records))
        train_groups = self._valid_groups(self._group_candidates(self._collect_candidates(records, train_ids, "train")))
        val_groups = self._valid_groups(self._group_candidates(self._collect_candidates(records, val_ids, "val")))
        if not train_groups or not val_groups:
            raise RuntimeError("No valid train/validation groups found.")

        scaler = self._fit_scaler(train_groups)
        candidate_dim = len(train_groups[0][0].candidate_features)
        trace_dim = len(train_groups[0][0].trace_features[0])
        model = self._build_model(candidate_dim, trace_dim).to(self.device)
        history, best_state = self._train(model, train_groups, val_groups, scaler)
        model.load_state_dict(best_state)
        train_loss, train_accuracy = self._evaluate(model, train_groups, scaler)
        val_loss, val_accuracy = self._evaluate(model, val_groups, scaler)
        predictions = self._predict(model, val_groups, scaler)
        final_val_accuracy = sum(int(item.correct) for item in predictions) / max(1, len(predictions))
        summary = {
            "architecture": self._architecture_name(),
            "train_ratio": self.config.train_ratio,
            "random_state": self.config.random_state,
            "train_questions": len(train_groups),
            "validation_questions": len(val_groups),
            "train_question_group_counts": self._question_group_counts(train_groups),
            "validation_question_group_counts": self._question_group_counts(val_groups),
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "validation_loss": round(val_loss, 6),
            "model_validation_accuracy": round(val_accuracy, 6),
            "validation_accuracy": round(final_val_accuracy, 6),
            "validation_accuracy_by_group": self._prediction_accuracy_by_group(predictions),
            "rag_backend": self.rag_memory.vector_backend,
            "bge_model": self.config.bge_model if self.rag_memory.vector_backend == "bge" else None,
            "llm_fallback_enabled": self.config.llm_fallback,
            "llm_fallback_model": self.config.llm_fallback_model if self.config.llm_fallback else None,
            "llm_fallback_used": sum(int(item.fallback_used) for item in predictions),
            "accuracy": round(final_val_accuracy, 6),
            "history": history,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        model_payload = {
            "architecture": summary["architecture"],
            "candidate_feature_count": candidate_dim,
            "trace_feature_count": trace_dim,
            "rag_backend": self.rag_memory.vector_backend,
            "bge_model": self.config.bge_model if self.rag_memory.vector_backend == "bge" else None,
            "scaler": scaler,
            "config": asdict(self.config),
            "state_dict": {name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()},
        }
        payload = {"summary": summary, "predictions": [asdict(item) for item in predictions], "model": model_payload}
        self._save_json(self.config.model_output_path, model_payload)
        self._save_json(self.config.output_path, payload)
        self._save_json(self.config.summary_output_path, summary)
        self._print_summary(summary)
        return payload

    def _collect_candidates(self, records: list[dict[str, Any]], record_ids: set[int], split_name: str) -> list[BacktrackingCandidate]:
        candidates: list[BacktrackingCandidate] = []
        processed = 0
        for record_id in sorted(record_ids):
            record = records[record_id]
            premises = [TextTools.clean(item) for item in record.get("premises-NL", []) or []]
            questions = record.get("questions", []) or []
            for question_id in range(len(questions)):
                raw_question = str(TextTools.safe_get(questions, question_id))
                expected = normalize_for_eval(TextTools.safe_get(record.get("answers", []), question_id))
                stem, options = self._raw_options(raw_question)
                question_type = self.question_classifier.classify(raw_question, stem, options, premises)
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
        return candidates

    def _candidate_features(
        self,
        label: str,
        option_text: str,
        question: str,
        premises: list[str],
        is_mcq: bool,
        question_type: QuestionTypeProfile,
    ) -> list[float]:
        premise_text = " ".join(premises)
        option_set = set(self._tokens(option_text))
        question_set = set(self._tokens(question))
        premise_set = set(self._tokens(premise_text))
        label_index = {"A": 0.0, "B": 0.25, "C": 0.5, "D": 0.75, "Yes": 0.15, "No": 0.5, "Uncertain": 0.85}.get(label, 0.0)
        neg_words = BacktrackingTraceGenerator.neg_words
        features = [
            float(label == "Yes"),
            float(label == "No"),
            float(label == "Uncertain"),
            float(is_mcq),
            label_index,
            self._overlap_ratio(option_set, premise_set),
            self._overlap_ratio(option_set, question_set),
            self._jaccard(option_set, premise_set),
            self._jaccard(option_set, question_set),
            self._jaccard(question_set, premise_set),
            min(len(option_set) / 24.0, 2.0),
            min(len(question_set) / 32.0, 2.0),
            min(len(premise_set) / 180.0, 3.0),
            float(bool(option_set & neg_words)),
            float(bool(question_set & neg_words)),
            float(bool(premise_set & neg_words)),
            1.0,
        ]
        return features + question_type.features

    def _raw_options(self, question: str) -> tuple[str, dict[str, str]]:
        matches = list(re.finditer(r"(?:^|\n)\s*([A-D])\.\s*(.+?)(?=\n\s*[A-D]\.\s*|\Z)", question, flags=re.S))
        if matches:
            stem = question[: matches[0].start()].strip()
            options = {match.group(1): TextTools.clean(match.group(2)) for match in matches}
            options.setdefault("Uncertain", "The premises do not provide enough information.")
            return stem, options
        return question, {
            "Yes": "The statement follows from the premises.",
            "No": "The statement contradicts the premises.",
            "Uncertain": "The premises do not provide enough information.",
        }

    def _build_model(self, candidate_dim: int, trace_dim: int) -> nn.Module:
        return TraceSSMOnlyReasoner(candidate_dim, trace_dim, self.config)

    def _architecture_name(self) -> str:
        return "trace_ssm_bge_rag_memory_blockwise_local_attention_transformer"

    def _train(
        self,
        model: nn.Module,
        train_groups: list[list[BacktrackingCandidate]],
        val_groups: list[list[BacktrackingCandidate]],
        scaler: dict[str, list[float]],
    ) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.l2)
        rng = random.Random(self.config.random_state)
        best_accuracy = -1.0
        best_loss = float("inf")
        best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        history: list[dict[str, float]] = []
        stale = 0
        for epoch in range(1, self.config.epochs + 1):
            model.train()
            shuffled = list(train_groups)
            rng.shuffle(shuffled)
            total_loss = 0.0
            correct = 0
            count = 0
            for batch in self._batches(shuffled):
                tensors = self._tensor_batch(batch, scaler)
                optimizer.zero_grad(set_to_none=True)
                scores, action_logits, conflict_logits = model(
                    tensors["candidate_features"],
                    tensors["trace_features"],
                    tensors["candidate_mask"],
                    tensors["trace_mask"],
                )
                loss = self._loss(scores, tensors)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(batch)
                correct += int((torch.argmax(scores, dim=1) == tensors["targets"]).sum().detach().cpu())
                count += len(batch)
            val_loss, val_accuracy = self._evaluate(model, val_groups, scaler)
            train_loss = total_loss / max(1, count)
            train_accuracy = correct / max(1, count)
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

    def _loss(self, scores: torch.Tensor, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        answer_loss = nn.functional.cross_entropy(scores, tensors["targets"])
        if self.config.consistency_loss_weight <= 0.0:
            return answer_loss
        log_probs = torch.log_softmax(scores, dim=-1)
        explanation_probs = tensors["explanation_probs"]
        explanation_ce = -(explanation_probs * log_probs).sum(dim=-1).mean()
        return answer_loss + self.config.consistency_loss_weight * explanation_ce

    def _evaluate(self, model: nn.Module, groups: list[list[BacktrackingCandidate]], scaler: dict[str, list[float]]) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        count = 0
        with torch.no_grad():
            for batch in self._batches(groups):
                tensors = self._tensor_batch(batch, scaler)
                scores, action_logits, conflict_logits = model(
                    tensors["candidate_features"],
                    tensors["trace_features"],
                    tensors["candidate_mask"],
                    tensors["trace_mask"],
                )
                loss = self._loss(scores, tensors)
                total_loss += float(loss.detach().cpu()) * len(batch)
                correct += int((torch.argmax(scores, dim=1) == tensors["targets"]).sum().detach().cpu())
                count += len(batch)
        return total_loss / max(1, count), correct / max(1, count)

    def _predict(self, model: nn.Module, groups: list[list[BacktrackingCandidate]], scaler: dict[str, list[float]]) -> list[BrainPrediction]:
        model.eval()
        predictions: list[BrainPrediction] = []
        with torch.no_grad():
            for group in groups:
                tensors = self._tensor_batch([group], scaler)
                scores, _action_logits, _conflict_logits = model(
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
                top_probability = float(probs[best_idx])
                final_answer = best.answer
                fallback_used = False
                fallback_reason = ""
                fallback_error = ""
                if self.llm_fallback is not None:
                    fallback_reason = self._fallback_reason(best.answer, top_probability)
                    if fallback_reason:
                        fallback_result = self.llm_fallback.choose_answer(group)
                        if fallback_result.answer:
                            final_answer = fallback_result.answer
                            fallback_used = True
                        fallback_error = fallback_result.error
                predictions.append(
                    BrainPrediction(
                        record_id=best.record_id,
                        question_id=best.question_id,
                        question_group=best.question_group,
                        expected=best.expected,
                        predicted_answer=final_answer,
                        correct=final_answer == best.expected,
                        candidate_probabilities=probability_map,
                        model_predicted_answer=best.answer,
                        model_top_probability=round(top_probability, 6),
                        fallback_used=fallback_used,
                        fallback_reason=fallback_reason,
                        fallback_error=fallback_error,
                    )
                )
        return predictions

    def _fallback_reason(self, predicted_answer: str, top_probability: float) -> str:
        reasons = []
        if top_probability < self.config.llm_fallback_threshold:
            reasons.append(f"low_confidence<{self.config.llm_fallback_threshold}")
        if self.config.llm_fallback_on_uncertain and predicted_answer == "Uncertain":
            reasons.append("predicted_uncertain")
        return ",".join(reasons)

    def _tensor_batch(self, groups: list[list[BacktrackingCandidate]], scaler: dict[str, list[float]]) -> dict[str, torch.Tensor]:
        batch_size = len(groups)
        candidate_count = max(len(group) for group in groups)
        trace_steps = self.config.max_trace_steps
        candidate_dim = len(groups[0][0].candidate_features)
        trace_dim = len(groups[0][0].trace_features[0])
        candidate_features = np.zeros((batch_size, candidate_count, candidate_dim), dtype=np.float32)
        trace_features = np.zeros((batch_size, candidate_count, trace_steps, trace_dim), dtype=np.float32)
        candidate_mask = np.zeros((batch_size, candidate_count), dtype=bool)
        trace_mask = np.zeros((batch_size, candidate_count, trace_steps), dtype=bool)
        explanation_probs = np.zeros((batch_size, candidate_count), dtype=np.float32)
        targets = np.zeros((batch_size,), dtype=np.int64)
        cand_mean = np.array(scaler["candidate_mean"], dtype=np.float64)
        cand_std = np.array(scaler["candidate_std"], dtype=np.float64)
        trace_mean = np.array(scaler["trace_mean"], dtype=np.float64)
        trace_std = np.array(scaler["trace_std"], dtype=np.float64)
        for row, group in enumerate(groups):
            targets[row] = self._target_index(group) or 0
            explanation_distribution = self._trace_explanation_distribution(group)
            for col, candidate in enumerate(group):
                candidate_mask[row, col] = True
                explanation_probs[row, col] = explanation_distribution.get(candidate.answer, 0.0)
                candidate_features[row, col] = ((np.array(candidate.candidate_features) - cand_mean) / cand_std).astype(np.float32)
                steps = min(trace_steps, len(candidate.trace_features))
                trace_matrix = (np.array(candidate.trace_features[:steps]) - trace_mean) / trace_std
                trace_features[row, col, :steps] = trace_matrix.astype(np.float32)
                trace_mask[row, col, :steps] = True
        return {
            "candidate_features": torch.from_numpy(candidate_features).to(self.device),
            "trace_features": torch.from_numpy(trace_features).to(self.device),
            "candidate_mask": torch.from_numpy(candidate_mask).to(self.device),
            "trace_mask": torch.from_numpy(trace_mask).to(self.device),
            "explanation_probs": torch.from_numpy(explanation_probs).to(self.device),
            "targets": torch.from_numpy(targets).to(self.device),
        }

    def _trace_explanation_distribution(self, group: list[BacktrackingCandidate]) -> dict[str, float]:
        labels = [candidate.answer for candidate in group]
        scores = np.array([self._trace_explanation_score(candidate) for candidate in group], dtype=np.float64)
        scores = scores - np.max(scores)
        exp = np.exp(scores)
        probs = exp / max(float(exp.sum()), 1e-12)
        return {label: float(prob) for label, prob in zip(labels, probs)}

    def _trace_explanation_score(self, candidate: BacktrackingCandidate) -> float:
        matrix = np.array(candidate.trace_features, dtype=np.float64)
        support = float(np.max(matrix[:, 2])) if matrix.size else 0.0
        mean_support = float(np.mean(matrix[:, 2])) if matrix.size else 0.0
        conflict = float(np.max(matrix[:, 3])) if matrix.size else 0.0
        backtrack = float(np.max(matrix[:, 9])) if matrix.size else 0.0
        rule_strength = float(np.max(matrix[:, 4])) if matrix.size else 0.0
        final_conflict = float(matrix[-1, 3]) if matrix.size else 0.0
        return 2.2 * support + 0.7 * mean_support + 0.25 * rule_strength - 1.2 * conflict - 0.35 * final_conflict - 0.25 * backtrack

    def _fit_scaler(self, groups: list[list[BacktrackingCandidate]]) -> dict[str, list[float]]:
        candidate_matrix = np.array([candidate.candidate_features for group in groups for candidate in group], dtype=np.float64)
        trace_matrix = np.array([step for group in groups for candidate in group for step in candidate.trace_features], dtype=np.float64)
        return {
            "candidate_mean": candidate_matrix.mean(axis=0).tolist(),
            "candidate_std": np.maximum(candidate_matrix.std(axis=0), 1e-6).tolist(),
            "trace_mean": trace_matrix.mean(axis=0).tolist(),
            "trace_std": np.maximum(trace_matrix.std(axis=0), 1e-6).tolist(),
        }

    def _group_candidates(self, candidates: list[BacktrackingCandidate]) -> list[list[BacktrackingCandidate]]:
        grouped: dict[str, list[BacktrackingCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.key, []).append(candidate)
        return list(grouped.values())

    def _valid_groups(self, groups: list[list[BacktrackingCandidate]]) -> list[list[BacktrackingCandidate]]:
        return [group for group in groups if self._target_index(group) is not None and group and group[0].trace_features]

    def _target_index(self, group: list[BacktrackingCandidate]) -> int | None:
        for idx, candidate in enumerate(group):
            if candidate.answer == candidate.expected:
                return idx
        return None

    def _batches(self, groups: list[list[BacktrackingCandidate]]) -> list[list[list[BacktrackingCandidate]]]:
        if not self.config.grouped_batches:
            return [groups[idx : idx + self.config.batch_size] for idx in range(0, len(groups), self.config.batch_size)]
        grouped: dict[str, list[list[BacktrackingCandidate]]] = {}
        ordered_keys: list[str] = []
        for group in groups:
            key = group[0].question_group
            if key not in grouped:
                ordered_keys.append(key)
                grouped[key] = []
            grouped[key].append(group)
        batches: list[list[list[BacktrackingCandidate]]] = []
        for key in ordered_keys:
            same_type = grouped[key]
            batches.extend(same_type[idx : idx + self.config.batch_size] for idx in range(0, len(same_type), self.config.batch_size))
        return batches

    def _question_group_counts(self, groups: list[list[BacktrackingCandidate]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for group in groups:
            counts[group[0].question_group] = counts.get(group[0].question_group, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[0]))

    def _prediction_accuracy_by_group(self, predictions: list[BrainPrediction]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[BrainPrediction]] = {}
        for prediction in predictions:
            grouped.setdefault(prediction.question_group, []).append(prediction)
        return {
            key: {
                "questions": len(items),
                "accuracy": round(sum(int(item.correct) for item in items) / max(1, len(items)), 6),
            }
            for key, items in sorted(grouped.items(), key=lambda item: item[0])
        }

    def _split_ids(self, record_count: int) -> tuple[set[int], set[int]]:
        ids = list(range(record_count))
        rng = random.Random(self.config.random_state)
        rng.shuffle(ids)
        train_count = int(round(record_count * self.config.train_ratio))
        train_count = max(1, min(record_count - 1, train_count)) if record_count > 1 else record_count
        return set(ids[:train_count]), set(ids[train_count:])

    def _tokens(self, text: str) -> list[str]:
        stop = {"the", "a", "an", "and", "or", "is", "are", "to", "of", "for", "in", "on", "that", "this", "it", "with", "from", "by", "above", "based", "does", "which"}
        return [token for token in re.findall(r"[a-z0-9_']+", text.lower()) if token not in stop]

    def _overlap_ratio(self, left: set[str], right: set[str]) -> float:
        return len(left & right) / max(1, len(left))

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        return len(left & right) / max(1, len(left | right))

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
        print("\nType 1 Backtracking Trace WM Summary")
        print(f"- train_questions: {summary['train_questions']}")
        print(f"- validation_questions: {summary['validation_questions']}")
        print(f"- train_accuracy: {summary['train_accuracy']}")
        print(f"- validation_accuracy: {summary['validation_accuracy']}")
        print(f"- saved model: {self.config.model_output_path}")
        print(f"- saved results: {self.config.output_path}")
        print(f"- saved summary: {self.config.summary_output_path}")

    @staticmethod
    def _save_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Type 1 SSM backtracking-trace world-model evaluator.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--output", type=Path, default=Path("type1_backtracking_trace_eval_results.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("type1_backtracking_trace_eval_summary.json"))
    parser.add_argument("--model-output", type=Path, default=Path("type1_backtracking_trace_model.json"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--max-trace-steps", type=int, default=18)
    parser.add_argument("--ssm-block-size", type=int, default=8)
    parser.add_argument("--local-attention-window", type=int, default=8)
    parser.add_argument("--propagation-top-k", type=int, default=4)
    parser.add_argument("--implicit-bias-strength", type=float, default=0.18)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.0)
    parser.add_argument("--ungrouped-batches", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--rag-backend", choices=["auto", "tfidf", "bge", "numpy"], default="auto")
    parser.add_argument("--bge-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--allow-bge-download", action="store_true")
    parser.add_argument("--llm-fallback", action="store_true")
    parser.add_argument("--llm-fallback-model", default="minigpt")
    parser.add_argument("--llm-fallback-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--llm-fallback-api-key-env", default="MINIGPT_API_KEY")
    parser.add_argument("--llm-fallback-threshold", type=float, default=0.62)
    parser.add_argument("--no-llm-fallback-on-uncertain", action="store_true")
    parser.add_argument("--llm-fallback-timeout", type=float, default=45.0)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BacktrackingTraceConfig:
    return BacktrackingTraceConfig(
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
        llm_fallback=args.llm_fallback,
        llm_fallback_model=args.llm_fallback_model,
        llm_fallback_base_url=args.llm_fallback_base_url,
        llm_fallback_api_key_env=args.llm_fallback_api_key_env,
        llm_fallback_threshold=args.llm_fallback_threshold,
        llm_fallback_on_uncertain=not args.no_llm_fallback_on_uncertain,
        llm_fallback_timeout=args.llm_fallback_timeout,
    )


def main() -> None:
    BacktrackingTraceTrainer(config_from_args(parse_args())).run()


if __name__ == "__main__":
    main()
