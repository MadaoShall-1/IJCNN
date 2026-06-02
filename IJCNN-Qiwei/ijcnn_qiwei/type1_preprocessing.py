#!/usr/bin/env python3
"""Object-oriented Type 1 preprocessing and classification pipeline.

This module keeps the user's original semantic preprocessing method:

- SentenceTransformer embeddings
- scikit-learn KMeans for coarse semantic clusters
- scikit-learn KNN for answer-class neighborhood labels
- TF-IDF keywords for cluster descriptions

It adds deterministic Type 1 routing/classification interfaces so each row can
be consumed by a later solver as a normalized Type 1 logic-query payload.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
warnings.filterwarnings(
    "ignore",
    message=r"Could not find the number of physical cores.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None  # type: ignore[assignment]


MULTISPACE_RE = re.compile(r"\s+")
MCQ_OPTION_RE = re.compile(
    r"(?:^|\n|\s)([A-D])[\.\)]\s+(.+?)(?=(?:\n|\s)[A-D][\.\)]\s+|$)",
    re.IGNORECASE | re.DOTALL,
)
YES_NO_START_RE = re.compile(
    r"^\s*(does|do|did|is|are|was|were|can|could|will|would|should|has|have|had)\b",
    re.IGNORECASE,
)


@dataclass
class LogicPreprocessingConfig:
    input_path: Path = Path("Logic_Based_Educational_Queries.json")
    output_path: Path = Path("processed_logic_queries_classified.json")
    stats_output_path: Path = Path("logic_query_classification_statistics.json")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_files_only: bool = False
    batch_size: int = 32
    n_clusters: int = 8
    k_neighbors: int = 5
    top_terms: int = 8
    random_state: int = 42


class TextTools:
    @staticmethod
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(TextTools.clean(item) for item in value)
        return MULTISPACE_RE.sub(" ", str(value).replace("\ufffd", " ")).strip()

    @staticmethod
    def safe_get(sequence: Any, index: int, default: Any = "") -> Any:
        if isinstance(sequence, list) and index < len(sequence):
            return sequence[index]
        return default


class AnswerNormalizer:
    @staticmethod
    def normalize(answer: Any) -> str:
        raw = TextTools.clean(answer)
        label = raw.strip().upper()
        if not label:
            return "missing"
        if label in {"YES", "NO", "UNKNOWN", "TRUE", "FALSE"}:
            return label.title() if label != "UNKNOWN" else "Unknown"
        if label in {"A", "B", "C", "D"}:
            return f"Option_{label}"
        return re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_") or "Other"


class Type1QuestionClassifier:
    """Deterministic Type 1 question classifier and API payload adapter."""

    entailment_patterns = [
        (re.compile(r"\bdoes\s+it\s+follow\b", re.IGNORECASE), "does_it_follow"),
        (re.compile(r"\bfollow(?:s)?\s+that\b", re.IGNORECASE), "follows_that"),
        (re.compile(r"\blogically\s+follows?\b", re.IGNORECASE), "logically_follows"),
        (re.compile(r"\baccording\s+to\s+the\s+premises\b", re.IGNORECASE), "according_to_premises"),
    ]
    conclusion_selection_patterns = [
        (re.compile(r"\bwhich\s+conclusion\b", re.IGNORECASE), "which_conclusion"),
        (re.compile(r"\bstrongest\s+conclusion\b", re.IGNORECASE), "strongest_conclusion"),
        (re.compile(r"\bcorrect\s+conclusion\b", re.IGNORECASE), "correct_conclusion"),
        (re.compile(r"\bwhat\s+can\s+we\s+conclude\b", re.IGNORECASE), "what_can_we_conclude"),
        (re.compile(r"\bfewest\s+premises\b", re.IGNORECASE), "fewest_premises"),
    ]
    requirement_patterns = [
        (re.compile(r"\bmeet(?:s)?\s+all\s+requirements\b", re.IGNORECASE), "meets_all_requirements"),
        (re.compile(r"\bqualif(?:y|ies|ied)\b", re.IGNORECASE), "qualification"),
        (re.compile(r"\beligib(?:le|ility)\b", re.IGNORECASE), "eligibility"),
        (re.compile(r"\bcan\s+(?:serve|teach|apply|access|cross|transport|propose)\b", re.IGNORECASE), "capability"),
    ]
    logic_structure_patterns = [
        ("conditional_chain", re.compile(r"(->|→|\bif\b.+\bthen\b|\bif\b)", re.IGNORECASE)),
        ("universal_rule", re.compile(r"(∀|forall|for all|\ball\b|\bevery\b|\bany\b)", re.IGNORECASE)),
        ("existential_claim", re.compile(r"(∃|exists|there exists|at least one)", re.IGNORECASE)),
        ("negation", re.compile(r"(¬|\bnot\b|\bcannot\b|\bdoes not\b|\bwithout\b)", re.IGNORECASE)),
        ("conjunction", re.compile(r"(\bif\b.+\band\b|\band\b.+\bthen\b|∧)", re.IGNORECASE)),
        ("contraposition", re.compile(r"\bcontraposition\b|\bif .+ not .+ then .+ not\b", re.IGNORECASE)),
        (
            "numeric_constraint",
            re.compile(
                r"([<>]=?|≤|≥)\s*\d|"
                r"\b(?:at least|at most|more than|less than|minimum of|maximum of|"
                r"no fewer than|no more than)\s+\d+|"
                r"\b\d+\s*(?:points?|credits?|percent|%)\b",
                re.IGNORECASE,
            ),
        ),
    ]

    def extract_mcq_options(self, question: str, choices: Any = "") -> dict[str, str]:
        normalized_question = (
            TextTools.clean(question)
            .replace(" B.", "\nB.")
            .replace(" C.", "\nC.")
            .replace(" D.", "\nD.")
        )
        options = {
            label.upper(): TextTools.clean(text)
            for label, text in MCQ_OPTION_RE.findall(normalized_question)
        }
        if options:
            return options

        if isinstance(choices, dict):
            return {
                str(key).strip().upper(): TextTools.clean(value)
                for key, value in choices.items()
                if str(key).strip().upper() in {"A", "B", "C", "D"}
            }

        if isinstance(choices, list):
            return {
                chr(ord("A") + index): TextTools.clean(value)
                for index, value in enumerate(choices[:4])
            }

        choice_text = TextTools.clean(choices)
        return {
            label.upper(): TextTools.clean(text)
            for label, text in MCQ_OPTION_RE.findall(choice_text)
        }

    def classify(
        self,
        *,
        question: str,
        answer: Any = "",
        premises_nl: Any = "",
        premises_fol: Any = "",
        explanation: Any = "",
        choices: Any = "",
    ) -> dict[str, Any]:
        clean_question = TextTools.clean(question)
        question_format, format_triggers = self._infer_question_format(clean_question, answer, choices)
        reasoning_task, task_triggers = self._infer_reasoning_task(clean_question, question_format)
        logic_structures = self._infer_logic_structures(clean_question, premises_nl, premises_fol, explanation)
        solver_route = self._infer_solver_route(question_format, reasoning_task, logic_structures)
        options = self.extract_mcq_options(clean_question, choices)

        confidence = 0.45
        if format_triggers:
            confidence += 0.2
        if task_triggers:
            confidence += 0.2
        if logic_structures and logic_structures != ["ordinary_rule_chain"]:
            confidence += 0.1
        if options and question_format == "multiple_choice":
            confidence += 0.05

        return {
            "dataset_type": "type1",
            "question_format": question_format,
            "reasoning_task": reasoning_task,
            "logic_structures": logic_structures,
            "solver_route": solver_route,
            "mcq_options": options,
            "classification_confidence": round(min(confidence, 0.98), 3),
            "classification_triggers": format_triggers + task_triggers,
        }

    def build_api_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "query_type": "type1",
            "record_id": row.get("record_id"),
            "question_id": row.get("question_id"),
            "premises-NL": row.get("premises_nl", []),
            "premises-FOL": row.get("premises_fol", []),
            "questions": [row.get("question", "")],
            "choices": row.get("type1_mcq_options", {}),
            "type1_classification": row.get("type1_classification", {}),
        }

    def _infer_question_format(self, question: str, answer: Any, choices: Any) -> tuple[str, list[str]]:
        triggers: list[str] = []
        options = self.extract_mcq_options(question, choices)
        answer_class = AnswerNormalizer.normalize(answer)

        if len(options) >= 2 or answer_class.startswith("Option_"):
            triggers.append("mcq_options")
            return "multiple_choice", triggers
        if YES_NO_START_RE.search(question) or answer_class in {"Yes", "No", "Unknown", "True", "False"}:
            triggers.append("yes_no_surface_form")
            return "yes_no_judgment", triggers
        if question.lower().startswith(("what ", "which ", "who ", "where ", "when ", "how ")):
            triggers.append("wh_question")
            return "open_ended", triggers
        return "logic_query", triggers

    def _infer_reasoning_task(self, question: str, question_format: str) -> tuple[str, list[str]]:
        conclusion_hits = self._pattern_hits(question, self.conclusion_selection_patterns)
        if question_format == "multiple_choice" and conclusion_hits:
            return "conclusion_selection", conclusion_hits

        entailment_hits = self._pattern_hits(question, self.entailment_patterns)
        if entailment_hits:
            return "entailment_judgment", entailment_hits

        requirement_hits = self._pattern_hits(question, self.requirement_patterns)
        if requirement_hits:
            return "requirement_satisfaction_judgment", requirement_hits

        if question_format == "yes_no_judgment":
            return "truth_judgment", ["yes_no_judgment_default"]
        if question_format == "multiple_choice":
            return "conclusion_selection", ["multiple_choice_default"]
        return "open_logic_answer", ["open_logic_default"]

    def _infer_logic_structures(
        self,
        question: str,
        premises_nl: Any,
        premises_fol: Any,
        explanation: Any,
    ) -> list[str]:
        combined = " ".join(
            part
            for part in [
                TextTools.clean(question),
                TextTools.clean(premises_nl),
                TextTools.clean(premises_fol),
                TextTools.clean(explanation),
            ]
            if part
        )
        structures = [
            name for name, pattern in self.logic_structure_patterns if pattern.search(combined)
        ]
        return structures or ["ordinary_rule_chain"]

    def _infer_solver_route(self, question_format: str, reasoning_task: str, logic_structures: list[str]) -> str:
        if question_format == "multiple_choice":
            return "semantic_mcq_retrieval"
        if "numeric_constraint" in logic_structures:
            return "semantic_numeric_constraint_retrieval"
        if reasoning_task in {"entailment_judgment", "truth_judgment", "requirement_satisfaction_judgment"}:
            return "semantic_entailment_retrieval"
        return "semantic_open_retrieval"

    @staticmethod
    def _pattern_hits(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> list[str]:
        return [name for pattern, name in patterns if pattern.search(text)]


class LogicDatasetFlattener:
    def __init__(self, classifier: Type1QuestionClassifier | None = None) -> None:
        self.classifier = classifier or Type1QuestionClassifier()

    def flatten(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record_id, item in enumerate(records):
            rows.extend(self._flatten_record(record_id, item))
        return rows

    def _flatten_record(self, record_id: int, item: dict[str, Any]) -> list[dict[str, Any]]:
        questions = item.get("questions", [])
        answers = item.get("answers", [])
        explanations = item.get("explanation", [])
        idx_values = item.get("idx", [])
        choices = item.get("choices", [])

        if not isinstance(questions, list):
            questions = [questions]
        row_count = max(len(questions), len(answers) if isinstance(answers, list) else 0, 1)

        premises_nl = item.get("premises-NL", [])
        premises_fol = item.get("premises-FOL", [])
        premises_nl_text = TextTools.clean(premises_nl)
        premises_fol_text = TextTools.clean(premises_fol)

        rows: list[dict[str, Any]] = []
        for question_id in range(row_count):
            question = TextTools.clean(TextTools.safe_get(questions, question_id))
            answer_raw = TextTools.safe_get(answers, question_id)
            explanation = TextTools.clean(TextTools.safe_get(explanations, question_id))
            support_idx = TextTools.safe_get(idx_values, question_id, [])
            choice_text = TextTools.clean(TextTools.safe_get(choices, question_id, choices))

            classification = self.classifier.classify(
                question=question,
                answer=answer_raw,
                premises_nl=premises_nl,
                premises_fol=premises_fol,
                explanation=explanation,
                choices=choice_text,
            )

            text_for_embedding = " ".join(
                part
                for part in [
                    "Premises:",
                    premises_nl_text,
                    "Formal logic:",
                    premises_fol_text,
                    "Question:",
                    question,
                    "Choices:",
                    choice_text,
                ]
                if part
            )

            row = {
                "record_id": record_id,
                "question_id": question_id,
                "premises_nl": premises_nl,
                "premises_fol": premises_fol,
                "question": question,
                "choices": choice_text,
                "answer": TextTools.clean(answer_raw),
                "answer_class": AnswerNormalizer.normalize(answer_raw),
                "explanation": explanation,
                "support_idx": support_idx,
                "premise_count": len(premises_nl) if isinstance(premises_nl, list) else 0,
                "fol_count": len(premises_fol) if isinstance(premises_fol, list) else 0,
                "support_count": len(support_idx) if isinstance(support_idx, list) else 0,
                "question_word_count": len(question.split()),
                "premise_word_count": len(premises_nl_text.split()),
                "has_multiple_choice": int(classification["question_format"] == "multiple_choice"),
                "type1_classification": classification,
                "type1_question_format": classification["question_format"],
                "type1_reasoning_task": classification["reasoning_task"],
                "type1_logic_structures": classification["logic_structures"],
                "type1_solver_route": classification["solver_route"],
                "type1_classification_confidence": classification["classification_confidence"],
                "type1_classification_triggers": classification["classification_triggers"],
                "type1_mcq_options": classification["mcq_options"],
                "type1_api_payload": {},
                "text_for_embedding": text_for_embedding,
            }
            row["type1_api_payload"] = self.classifier.build_api_payload(row)
            rows.append(row)

        return rows


class SemanticEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.local_files_only = local_files_only

    def encode(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if SentenceTransformer is None:
            raise ModuleNotFoundError(
                "sentence_transformers is required for embedding. Install requirements_semantic.txt "
                "or run only the deterministic Type1QuestionClassifier interfaces."
            )
        model = SentenceTransformer(self.model_name, local_files_only=self.local_files_only)
        texts = [row["text_for_embedding"] for row in rows]
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


class SemanticClusterer:
    def __init__(self, n_clusters: int = 8, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, embeddings: np.ndarray) -> tuple[np.ndarray, float]:
        n_clusters = max(1, min(self.n_clusters, len(embeddings)))
        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = model.fit_predict(embeddings)
        return labels.astype(int), float(model.inertia_)


class KNNFineClassifier:
    def __init__(self, k_neighbors: int = 5) -> None:
        self.k_neighbors = k_neighbors

    def classify(
        self,
        embeddings: np.ndarray,
        coarse_labels: np.ndarray,
        answer_labels: list[str],
    ) -> tuple[list[str], list[float]]:
        predicted: list[str] = []
        confidence: list[float] = []
        all_indices = np.arange(len(embeddings))

        for row_idx in range(len(embeddings)):
            same_cluster = all_indices[(coarse_labels == coarse_labels[row_idx]) & (all_indices != row_idx)]
            candidates = same_cluster if len(same_cluster) >= self.k_neighbors else all_indices[all_indices != row_idx]
            candidates = np.array(
                [idx for idx in candidates if answer_labels[int(idx)] != "missing"],
                dtype=int,
            )

            if len(candidates) == 0:
                predicted.append("missing")
                confidence.append(0.0)
                continue

            unique_labels = sorted({answer_labels[int(idx)] for idx in candidates})
            if len(unique_labels) == 1:
                predicted.append(unique_labels[0])
                confidence.append(1.0)
                continue

            effective_k = min(self.k_neighbors, len(candidates))
            classifier = KNeighborsClassifier(
                n_neighbors=effective_k,
                metric="cosine",
                algorithm="brute",
                weights="distance",
            )
            classifier.fit(embeddings[candidates], [answer_labels[int(idx)] for idx in candidates])
            prediction = classifier.predict(embeddings[row_idx : row_idx + 1])[0]
            probabilities = classifier.predict_proba(embeddings[row_idx : row_idx + 1])[0]
            predicted.append(str(prediction))
            confidence.append(float(probabilities.max()))

        return predicted, confidence


class ClusterAnalyzer:
    def __init__(self, top_terms: int = 8) -> None:
        self.top_terms = top_terms

    def top_terms_for_clusters(self, rows: list[dict[str, Any]], labels: np.ndarray) -> dict[int, list[str]]:
        question_texts = [row["question"] for row in rows]
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=3000,
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b",
        )
        tfidf = vectorizer.fit_transform(question_texts)
        terms = np.array(vectorizer.get_feature_names_out())

        cluster_terms: dict[int, list[str]] = {}
        for cluster_id in sorted(set(int(label) for label in labels)):
            member_mask = labels == cluster_id
            if not member_mask.any():
                cluster_terms[cluster_id] = []
                continue
            mean_scores = np.asarray(tfidf[member_mask].mean(axis=0)).ravel()
            top_indices = mean_scores.argsort()[::-1][: self.top_terms]
            cluster_terms[cluster_id] = [
                str(terms[idx]) for idx in top_indices if mean_scores[idx] > 0
            ]
        return cluster_terms

    def build_cluster_summaries(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            clusters[row["coarse_cluster"]].append(row)

        summaries: list[dict[str, Any]] = []
        for cluster_id, members in sorted(clusters.items()):
            answer_counts = Counter(row["answer_class"] for row in members)
            fine_counts = Counter(row["knn_fine_class"] for row in members)
            format_counts = Counter(row["type1_question_format"] for row in members)
            task_counts = Counter(row["type1_reasoning_task"] for row in members)
            route_counts = Counter(row["type1_solver_route"] for row in members)
            top_terms = members[0].get("cluster_top_terms", []) if members else []
            question_type = self._infer_question_type(members)
            domain_hint = self._infer_domain_hint(members)
            logic_style_hint = self._infer_logic_style(members)
            top_answer, top_answer_count = answer_counts.most_common(1)[0]
            top_answer_ratio = top_answer_count / len(members)
            avg_premises = sum(row["premise_count"] for row in members) / len(members)
            avg_question_words = sum(row["question_word_count"] for row in members) / len(members)

            if top_answer_ratio >= 0.7:
                answer_hint = f"answers are strongly concentrated in {top_answer}"
            elif top_answer_ratio >= 0.5:
                answer_hint = f"answers are mostly {top_answer}, with some mixture"
            else:
                answer_hint = "answers are relatively mixed"

            description = (
                f"{question_type}; primary scenario: {domain_hint}; "
                f"type1 task mix: {dict(task_counts.most_common(3))}; "
                f"top keywords: {', '.join(top_terms[:6]) or 'no clear keywords'}; "
                f"logical pattern: {logic_style_hint}; {answer_hint}; "
                f"average premise count is about {avg_premises:.1f}, "
                f"and average question length is about {avg_question_words:.1f} words."
            )

            summaries.append(
                {
                    "coarse_cluster": int(cluster_id),
                    "cluster_description": description,
                    "question_type_hint": question_type,
                    "domain_hint": domain_hint,
                    "logic_style_hint": logic_style_hint,
                    "count": len(members),
                    "top_terms": top_terms,
                    "type1_question_format_distribution": dict(format_counts.most_common()),
                    "type1_reasoning_task_distribution": dict(task_counts.most_common()),
                    "type1_solver_route_distribution": dict(route_counts.most_common()),
                    "answer_class_distribution": dict(answer_counts.most_common()),
                    "knn_fine_class_distribution": dict(fine_counts.most_common()),
                    "mean_premise_count": round(avg_premises, 3),
                    "mean_question_word_count": round(avg_question_words, 3),
                    "sample_questions": [row["question"] for row in members[:2]],
                }
            )
        return summaries

    def build_statistics(self, rows: list[dict[str, Any]], descriptions: dict[int, str]) -> pd.DataFrame:
        total_rows = len(rows)
        cluster_sizes = Counter(row["coarse_cluster"] for row in rows)
        groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[
                (
                    row["coarse_cluster"],
                    row["type1_question_format"],
                    row["type1_reasoning_task"],
                )
            ].append(row)

        records: list[dict[str, Any]] = []
        for (cluster_id, question_format, reasoning_task), members in sorted(groups.items()):
            answer_counts = Counter(row["answer_class"] for row in members)
            route_counts = Counter(row["type1_solver_route"] for row in members)
            records.append(
                {
                    "coarse_cluster": int(cluster_id),
                    "cluster_description": descriptions.get(int(cluster_id), ""),
                    "type1_question_format": question_format,
                    "type1_reasoning_task": reasoning_task,
                    "count": len(members),
                    "percent_total": round(len(members) / total_rows, 6) if total_rows else 0.0,
                    "percent_in_cluster": round(len(members) / cluster_sizes[cluster_id], 6)
                    if cluster_sizes[cluster_id]
                    else 0.0,
                    "mean_premise_count": round(sum(row["premise_count"] for row in members) / len(members), 3),
                    "mean_question_word_count": round(
                        sum(row["question_word_count"] for row in members) / len(members), 3
                    ),
                    "top_solver_route": route_counts.most_common(1)[0][0] if route_counts else "",
                    "top_answer_class": answer_counts.most_common(1)[0][0] if answer_counts else "missing",
                    "answer_class_distribution": dict(answer_counts.most_common()),
                    "solver_route_distribution": dict(route_counts.most_common()),
                }
            )
        return pd.DataFrame(records)

    def _infer_question_type(self, members: list[dict[str, Any]]) -> str:
        if not members:
            return "Unknown question type"
        formats = Counter(row["type1_question_format"] for row in members)
        tasks = Counter(row["type1_reasoning_task"] for row in members)
        top_format = formats.most_common(1)[0][0]
        top_task = tasks.most_common(1)[0][0]
        if top_format == "multiple_choice":
            return f"Multiple-choice {top_task.replace('_', ' ')}"
        if top_format == "yes_no_judgment":
            return f"Yes/no {top_task.replace('_', ' ')}"
        return f"Mixed Type 1 {top_task.replace('_', ' ')}"

    def _infer_domain_hint(self, members: list[dict[str, Any]]) -> str:
        domain_keywords = {
            "Python project logic scenario": {"python", "pep", "code", "project", "projects", "optimized"},
            "Student eligibility / learning condition scenario": {
                "student",
                "students",
                "course",
                "internship",
                "scholarship",
                "attendance",
                "lectures",
                "assignment",
                "assignments",
                "curriculum",
            },
            "Named-person eligibility scenario": {"david", "sophia", "john", "professor"},
        }
        domain_counts: Counter[str] = Counter()
        for row in members:
            text = f"{row.get('text_for_embedding', '')} {row.get('question', '')}".lower()
            tokens = set(re.findall(r"[a-z]+", text))
            for domain, keywords in domain_keywords.items():
                if tokens & keywords:
                    domain_counts[domain] += 1

        if not domain_counts:
            return "General educational logic scenario"
        total = len(members) or 1
        top_domains = domain_counts.most_common(2)
        top_domain, top_count = top_domains[0]
        if len(top_domains) > 1 and top_count / total < 0.55:
            return " / ".join(domain for domain, _ in top_domains)
        return top_domain

    def _infer_logic_style(self, members: list[dict[str, Any]]) -> str:
        structures = Counter(
            structure
            for row in members
            for structure in row.get("type1_logic_structures", [])
        )
        if not structures:
            return "mostly ordinary conditional-chain reasoning"
        return ", ".join(name for name, _ in structures.most_common(4))


class DatasetProfiler:
    def build(self, records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
        keysets = Counter(tuple(sorted(record.keys())) for record in records)
        answers = Counter(row["answer_class"] for row in rows)
        formats = Counter(row["type1_question_format"] for row in rows)
        tasks = Counter(row["type1_reasoning_task"] for row in rows)
        routes = Counter(row["type1_solver_route"] for row in rows)
        questions_per_record = Counter(
            len(record.get("questions", [])) if isinstance(record.get("questions", []), list) else 1
            for record in records
        )
        return {
            "source_records": len(records),
            "flattened_question_rows": len(rows),
            "record_keysets": {"|".join(keyset): count for keyset, count in keysets.most_common()},
            "questions_per_record": {str(key): value for key, value in questions_per_record.most_common()},
            "answer_class_distribution": dict(answers.most_common()),
            "type1_question_format_distribution": dict(formats.most_common()),
            "type1_reasoning_task_distribution": dict(tasks.most_common()),
            "type1_solver_route_distribution": dict(routes.most_common()),
        }


class JsonWriter:
    @staticmethod
    def make_jsonable(value: Any) -> Any:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): JsonWriter.make_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [JsonWriter.make_jsonable(item) for item in value]
        return value

    @staticmethod
    def save(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(JsonWriter.make_jsonable(payload), file, ensure_ascii=False, indent=2)


class LogicPreprocessingPipeline:
    def __init__(self, config: LogicPreprocessingConfig) -> None:
        self.config = config
        self.classifier = Type1QuestionClassifier()
        self.flattener = LogicDatasetFlattener(self.classifier)
        self.embedder = SemanticEmbedder(
            config.model_name,
            batch_size=config.batch_size,
            local_files_only=config.local_files_only,
        )
        self.clusterer = SemanticClusterer(config.n_clusters, config.random_state)
        self.knn = KNNFineClassifier(config.k_neighbors)
        self.analyzer = ClusterAnalyzer(config.top_terms)
        self.profiler = DatasetProfiler()

    def run(self) -> dict[str, Any]:
        records = self.load_records()
        rows = self.flattener.flatten(records)
        if not rows:
            raise ValueError("No question rows were produced from the input dataset.")

        embeddings = self.embedder.encode(rows)
        embeddings = normalize(embeddings, norm="l2")

        coarse_labels, inertia = self.clusterer.fit_predict(embeddings)
        fine_labels, confidence = self.knn.classify(
            embeddings,
            coarse_labels,
            [row["answer_class"] for row in rows],
        )
        cluster_top_terms = self.analyzer.top_terms_for_clusters(rows, coarse_labels)

        for row, coarse_label, fine_label, score in zip(rows, coarse_labels, fine_labels, confidence):
            row["coarse_cluster"] = int(coarse_label)
            row["knn_fine_class"] = fine_label
            row["knn_confidence"] = round(float(score), 6)
            row["cluster_top_terms"] = cluster_top_terms.get(int(coarse_label), [])

        cluster_summary_records = self.analyzer.build_cluster_summaries(rows)
        cluster_descriptions = {
            row["coarse_cluster"]: row["cluster_description"]
            for row in cluster_summary_records
        }
        for row in rows:
            row["cluster_description"] = cluster_descriptions.get(row["coarse_cluster"], "")

        statistics_df = self.analyzer.build_statistics(rows, cluster_descriptions)
        statistics_records = statistics_df.to_dict(orient="records")

        output_payload = {
            "metadata": {
                **self.profiler.build(records, rows),
                "embedding_model": self.config.model_name,
                "n_clusters": int(min(self.config.n_clusters, len(rows))),
                "k_neighbors": self.config.k_neighbors,
                "embedding_shape": list(embeddings.shape),
                "kmeans_inertia": round(inertia, 6),
                "notes": (
                    "Rows are question-level Type 1 samples. SentenceTransformer embeddings are used for "
                    "semantic features; KMeans creates coarse_cluster; KNN predicts knn_fine_class; "
                    "deterministic Type1QuestionClassifier provides question_format, reasoning_task, "
                    "logic_structures, solver_route, and API payload fields."
                ),
            },
            "cluster_summary_records": cluster_summary_records,
            "statistics_df_records": statistics_records,
            "processed_rows": rows,
        }

        JsonWriter.save(self.config.output_path, output_payload)
        JsonWriter.save(self.config.stats_output_path, statistics_records)
        self.print_summary(records, rows, embeddings, inertia, statistics_records, cluster_summary_records)
        return output_payload

    def load_records(self) -> list[dict[str, Any]]:
        with self.config.input_path.open("r", encoding="utf-8", errors="replace") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError("Expected the input JSON to be a list of records.")
        return records

    def print_summary(
        self,
        records: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        embeddings: np.ndarray,
        inertia: float,
        statistics_records: list[dict[str, Any]],
        cluster_summary_records: list[dict[str, Any]],
    ) -> None:
        print(f"Loaded records: {len(records)}")
        print(f"Flattened question rows: {len(rows)}")
        print(f"Embedding matrix shape: {embeddings.shape}")
        print(f"KMeans inertia: {inertia:.6f}")
        print(f"Statistics rows: {len(statistics_records)}")
        print(f"Saved full output: {self.config.output_path}")
        print(f"Saved statistics output: {self.config.stats_output_path}")
        print()
        print("Cluster feature descriptions:")
        for row in cluster_summary_records:
            print(f"- Cluster {row['coarse_cluster']} ({row['count']} rows): {row['cluster_description']}")
        print()
        self.print_statistics_preview(pd.DataFrame(statistics_records))

    @staticmethod
    def print_statistics_preview(statistics_df: pd.DataFrame, max_rows: int = 20) -> None:
        preview_columns = [
            "coarse_cluster",
            "type1_question_format",
            "type1_reasoning_task",
            "count",
            "percent_total",
            "percent_in_cluster",
            "top_solver_route",
            "top_answer_class",
        ]
        available_columns = [column for column in preview_columns if column in statistics_df.columns]
        print(statistics_df[available_columns].head(max_rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object-oriented Type 1 preprocessing pipeline.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--output", type=Path, default=Path("processed_logic_queries_classified.json"))
    parser.add_argument("--stats-output", type=Path, default=Path("logic_query_classification_statistics.json"))
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the embedding model only from the local Hugging Face cache or a local model path.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> LogicPreprocessingConfig:
    return LogicPreprocessingConfig(
        input_path=args.input,
        output_path=args.output,
        stats_output_path=args.stats_output,
        model_name=args.model_name,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
        n_clusters=args.n_clusters,
        k_neighbors=args.k_neighbors,
        top_terms=args.top_terms,
        random_state=args.random_state,
    )


def main() -> None:
    LogicPreprocessingPipeline(config_from_args(parse_args())).run()


if __name__ == "__main__":
    main()
