#!/usr/bin/env python3
"""Semantic-tree RAG support for Type 1 logic parsing.

The intended setup is:

- Qwen2.5-1.5B-Instruct: semantic decomposition / concept extraction.
- BGE embedding model: vector retrieval over semantic tree nodes.
- Stage 0 parser: receives structured evidence for semantic matching and
  downstream RAG-style reasoning.

All heavy components are optional. If the segmenter or embedding model is not
available, the module degrades to deterministic sentence/keyphrase splitting so
the parser remains usable.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .common import LLMClient, LLMEndpointConfig, OpenAICompatibleLLMClient, Stage0Input
from .type1_preprocessing import TextTools, Type1QuestionClassifier

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None  # type: ignore[assignment]


SPLIT_RE = re.compile(r"(?<=[.;?!])\s+|\n+")
KEYPHRASE_RE = re.compile(
    r"\b(?:if|then|all|every|any|not|does not|cannot|qualifies?|eligible|"
    r"requires?|must|can|according|premises?|conclusion|follows?)\b",
    re.IGNORECASE,
)


@dataclass
class SemanticRAGConfig:
    segmenter_model: str = "openai/Qwen2.5-1.5B-Instruct"
    segmenter_api_base: str = "http://localhost:8001/v1"
    segmenter_api_key: str = "EMPTY"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_files_only: bool = False
    similarity_threshold: float = 0.42
    top_k: int = 8
    max_tree_edges_per_node: int = 4
    enable_qwen_segmenter: bool = True
    enable_bge_embeddings: bool = True


@dataclass
class SemanticNode:
    node_id: str
    text: str
    node_type: str
    parent_id: str | None = None
    source_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticEdge:
    source_id: str
    target_id: str
    edge_type: str
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticTree:
    nodes: list[SemanticNode]
    edges: list[SemanticEdge]
    root_id: str = "root"

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }


@dataclass
class RetrievedSemanticContext:
    query: str
    retrieved_nodes: list[dict[str, Any]]
    context_text: str
    tree_summary: dict[str, Any]


class QwenSemanticSegmenter:
    """Semantic decomposition using Qwen, with deterministic fallback."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: SemanticRAGConfig | None = None,
    ) -> None:
        self.config = config or SemanticRAGConfig()
        self.llm_client = llm_client
        if self.llm_client is None and self.config.enable_qwen_segmenter:
            self.llm_client = OpenAICompatibleLLMClient(
                LLMEndpointConfig(
                    model=self.config.segmenter_model,
                    api_base=self.config.segmenter_api_base,
                    api_key=self.config.segmenter_api_key,
                )
            )

    def segment(self, text: str, *, source_type: str) -> list[dict[str, Any]]:
        cleaned = TextTools.clean(text)
        if not cleaned:
            return []
        if self.config.enable_qwen_segmenter and self.llm_client is not None:
            try:
                return self._segment_with_qwen(cleaned, source_type=source_type)
            except Exception:
                pass
        return self._segment_deterministically(cleaned, source_type=source_type)

    def _segment_with_qwen(self, text: str, *, source_type: str) -> list[dict[str, Any]]:
        response = self.llm_client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You split educational logic text into semantic units. "
                        "Return strict JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "source_type": source_type,
                            "text": text,
                            "schema": {
                                "units": [
                                    {
                                        "text": "atomic proposition or rule",
                                        "unit_type": "rule|fact|question|option|concept",
                                        "entities": ["entity names"],
                                        "predicates": ["predicate or relation names"],
                                        "logic_cues": ["if_then|negation|universal|existential|requirement|entailment"],
                                    }
                                ]
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.0,
        )
        payload = json.loads(self._extract_json(response))
        units = payload.get("units", [])
        if not isinstance(units, list):
            raise ValueError("Qwen segmenter returned non-list units.")
        normalized: list[dict[str, Any]] = []
        for unit in units:
            if not isinstance(unit, dict):
                continue
            unit_text = TextTools.clean(unit.get("text", ""))
            if not unit_text:
                continue
            normalized.append(
                {
                    "text": unit_text,
                    "unit_type": str(unit.get("unit_type") or source_type),
                    "entities": list(unit.get("entities") or []),
                    "predicates": list(unit.get("predicates") or []),
                    "logic_cues": list(unit.get("logic_cues") or []),
                    "segmenter": "qwen2.5-1.5b",
                }
            )
        if not normalized:
            raise ValueError("Qwen segmenter returned no usable units.")
        return normalized

    def _extract_json(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        if stripped.startswith("{"):
            return stripped
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return stripped[start : end + 1]
        return stripped

    def _segment_deterministically(self, text: str, *, source_type: str) -> list[dict[str, Any]]:
        parts = [TextTools.clean(part) for part in SPLIT_RE.split(text) if TextTools.clean(part)]
        if not parts:
            parts = [text]
        units: list[dict[str, Any]] = []
        for part in parts:
            units.append(
                {
                    "text": part,
                    "unit_type": self._guess_unit_type(part, source_type),
                    "entities": self._guess_entities(part),
                    "predicates": self._guess_predicates(part),
                    "logic_cues": self._guess_logic_cues(part),
                    "segmenter": "deterministic_fallback",
                }
            )
        return units

    def _guess_unit_type(self, text: str, source_type: str) -> str:
        lowered = text.lower()
        if source_type == "question":
            return "question"
        if re.search(r"^[A-D][\.\)]", text):
            return "option"
        if "if " in lowered or " then " in lowered:
            return "rule"
        return "fact"

    def _guess_entities(self, text: str) -> list[str]:
        return sorted(set(re.findall(r"\b[A-Z][a-zA-Z]+\b", text)))[:8]

    def _guess_predicates(self, text: str) -> list[str]:
        candidates = re.findall(r"\b(?:is|are|has|have|qualifies?|eligible|requires?|follows?|can|must)\b\s+[^.;,]+", text, re.IGNORECASE)
        return [TextTools.clean(item) for item in candidates[:5]]

    def _guess_logic_cues(self, text: str) -> list[str]:
        lowered = text.lower()
        cues: list[str] = []
        if "if " in lowered or " then " in lowered:
            cues.append("if_then")
        if "not" in lowered or "cannot" in lowered or "does not" in lowered:
            cues.append("negation")
        if "all " in lowered or "every " in lowered or "any " in lowered:
            cues.append("universal")
        if "there exists" in lowered or "at least one" in lowered:
            cues.append("existential")
        if "according to the premises" in lowered or "follow" in lowered:
            cues.append("entailment")
        if KEYPHRASE_RE.search(text):
            cues.append("logic_keyword")
        return sorted(set(cues))


class BGEVectorIndex:
    """Small in-memory vector index backed by BGE embeddings."""

    def __init__(self, config: SemanticRAGConfig | None = None) -> None:
        self.config = config or SemanticRAGConfig()
        self.model = None
        if self.config.enable_bge_embeddings:
            self.model = self._load_model()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.model is not None:
            vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return self._sanitize_vectors(np.asarray(vectors, dtype=np.float32))
        return self._hash_embed(texts)

    def retrieve(
        self,
        query: str,
        nodes: list[SemanticNode],
        *,
        top_k: int,
    ) -> list[tuple[SemanticNode, float]]:
        if not nodes:
            return []
        node_vectors = self.embed([node.text for node in nodes])
        query_vector = self.embed([query])[0]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scores = np.nan_to_num(node_vectors @ query_vector, nan=0.0, posinf=0.0, neginf=0.0)
        ranked = sorted(
            zip(nodes, scores.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(node, float(score)) for node, score in ranked[:top_k]]

    def pairwise_edges(
        self,
        nodes: list[SemanticNode],
        *,
        threshold: float,
        max_edges_per_node: int,
    ) -> list[SemanticEdge]:
        if len(nodes) < 2:
            return []
        vectors = self.embed([node.text for node in nodes])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sim = np.nan_to_num(vectors @ vectors.T, nan=0.0, posinf=0.0, neginf=0.0)
        edges: list[SemanticEdge] = []
        for i, node in enumerate(nodes):
            candidates: list[tuple[int, float]] = []
            for j in range(len(nodes)):
                if i == j:
                    continue
                score = float(sim[i, j])
                if score >= threshold:
                    candidates.append((j, score))
            candidates.sort(key=lambda item: item[1], reverse=True)
            for j, score in candidates[:max_edges_per_node]:
                edges.append(
                    SemanticEdge(
                        source_id=node.node_id,
                        target_id=nodes[j].node_id,
                        edge_type="semantic_similarity",
                        score=round(score, 6),
                    )
                )
        return edges

    def _load_model(self) -> Any:
        if SentenceTransformer is None:
            return None
        try:
            return SentenceTransformer(
                self.config.embedding_model,
                local_files_only=self.config.local_files_only,
            )
        except Exception:
            return None

    def _hash_embed(self, texts: list[str], dims: int = 256) -> np.ndarray:
        vectors = np.zeros((len(texts), dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-zA-Z_]+", text.lower()):
                idx = hash(token) % dims
                vectors[row, idx] += 1.0
        norms = np.linalg.norm(vectors, axis=1)
        for row, norm in enumerate(norms):
            if norm > 0:
                vectors[row] /= norm
        return vectors

    def _sanitize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        clean = np.clip(clean, -1_000.0, 1_000.0)
        norms = np.linalg.norm(clean, axis=1, keepdims=True)
        np.divide(clean, norms, out=clean, where=norms > 0)
        return clean


class SemanticTreeBuilder:
    def __init__(
        self,
        segmenter: QwenSemanticSegmenter | None = None,
        vector_index: BGEVectorIndex | None = None,
        config: SemanticRAGConfig | None = None,
    ) -> None:
        self.config = config or SemanticRAGConfig()
        self.segmenter = segmenter or QwenSemanticSegmenter(config=self.config)
        self.vector_index = vector_index or BGEVectorIndex(self.config)
        self.classifier = Type1QuestionClassifier()

    def build(self, stage_input: Stage0Input, classification: dict[str, Any] | None = None) -> SemanticTree:
        classification = classification or self.classifier.classify(
            question=stage_input.question,
            premises_nl=stage_input.premises_nl,
            premises_fol=stage_input.premises_fol,
            choices=stage_input.choices,
        )
        nodes: list[SemanticNode] = [
            SemanticNode(
                node_id="root",
                text="Type 1 semantic tree root",
                node_type="root",
                metadata={"classification": classification},
            )
        ]
        edges: list[SemanticEdge] = []

        for idx, premise in enumerate(stage_input.premises_nl):
            source_id = f"premise:{idx}"
            nodes.append(
                SemanticNode(
                    node_id=source_id,
                    text=premise,
                    node_type="premise",
                    parent_id="root",
                    source_index=idx,
                )
            )
            edges.append(SemanticEdge("root", source_id, "contains", 1.0))
            self._add_units(nodes, edges, source_id, premise, source_type="premise", source_index=idx)

        question_id = "question:0"
        nodes.append(
            SemanticNode(
                node_id=question_id,
                text=stage_input.question,
                node_type="question",
                parent_id="root",
                source_index=0,
            )
        )
        edges.append(SemanticEdge("root", question_id, "contains", 1.0))
        self._add_units(nodes, edges, question_id, stage_input.question, source_type="question", source_index=0)

        options = classification.get("mcq_options") or {}
        for label, option_text in options.items():
            option_id = f"option:{label}"
            nodes.append(
                SemanticNode(
                    node_id=option_id,
                    text=f"{label}. {option_text}",
                    node_type="option",
                    parent_id=question_id,
                    metadata={"label": label},
                )
            )
            edges.append(SemanticEdge(question_id, option_id, "has_option", 1.0))

        content_nodes = [node for node in nodes if node.node_id != "root"]
        edges.extend(
            self.vector_index.pairwise_edges(
                content_nodes,
                threshold=self.config.similarity_threshold,
                max_edges_per_node=self.config.max_tree_edges_per_node,
            )
        )
        return SemanticTree(nodes=nodes, edges=edges)

    def _add_units(
        self,
        nodes: list[SemanticNode],
        edges: list[SemanticEdge],
        parent_id: str,
        text: str,
        *,
        source_type: str,
        source_index: int,
    ) -> None:
        for unit_index, unit in enumerate(self.segmenter.segment(text, source_type=source_type)):
            node_id = f"{parent_id}:unit:{unit_index}"
            nodes.append(
                SemanticNode(
                    node_id=node_id,
                    text=unit["text"],
                    node_type=str(unit.get("unit_type") or "semantic_unit"),
                    parent_id=parent_id,
                    source_index=source_index,
                    metadata={
                        "entities": unit.get("entities", []),
                        "predicates": unit.get("predicates", []),
                        "logic_cues": unit.get("logic_cues", []),
                        "segmenter": unit.get("segmenter", ""),
                    },
                )
            )
            edges.append(SemanticEdge(parent_id, node_id, "decomposes_to", 1.0))


class SemanticTreeRetriever:
    def __init__(self, vector_index: BGEVectorIndex | None = None, config: SemanticRAGConfig | None = None) -> None:
        self.config = config or SemanticRAGConfig()
        self.vector_index = vector_index or BGEVectorIndex(self.config)

    def retrieve(self, tree: SemanticTree, query: str, *, top_k: int | None = None) -> RetrievedSemanticContext:
        k = top_k or self.config.top_k
        searchable = [
            node
            for node in tree.nodes
            if node.node_type not in {"root"} and TextTools.clean(node.text)
        ]
        ranked = self.vector_index.retrieve(query, searchable, top_k=k)
        retrieved_nodes = [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "text": node.text,
                "score": round(score, 6),
                "metadata": node.metadata,
            }
            for node, score in ranked
        ]
        return RetrievedSemanticContext(
            query=query,
            retrieved_nodes=retrieved_nodes,
            context_text=self._format_context(retrieved_nodes),
            tree_summary=self._tree_summary(tree),
        )

    def _format_context(self, retrieved_nodes: list[dict[str, Any]]) -> str:
        lines = ["Structured semantic evidence:"]
        for idx, node in enumerate(retrieved_nodes, start=1):
            cues = ", ".join(node.get("metadata", {}).get("logic_cues", []) or [])
            cue_text = f" cues=[{cues}]" if cues else ""
            lines.append(
                f"{idx}. ({node['node_type']}, score={node['score']}) {node['text']}{cue_text}"
            )
        return "\n".join(lines)

    def _tree_summary(self, tree: SemanticTree) -> dict[str, Any]:
        node_counts: dict[str, int] = {}
        edge_counts: dict[str, int] = {}
        for node in tree.nodes:
            node_counts[node.node_type] = node_counts.get(node.node_type, 0) + 1
        for edge in tree.edges:
            edge_counts[edge.edge_type] = edge_counts.get(edge.edge_type, 0) + 1
        return {
            "node_count": len(tree.nodes),
            "edge_count": len(tree.edges),
            "node_type_counts": node_counts,
            "edge_type_counts": edge_counts,
        }


class SemanticTreeRAG:
    """Facade that builds a semantic tree and returns structured RAG context."""

    def __init__(
        self,
        config: SemanticRAGConfig | None = None,
        segmenter_client: LLMClient | None = None,
    ) -> None:
        self.config = config or SemanticRAGConfig()
        self.segmenter = QwenSemanticSegmenter(segmenter_client, self.config)
        self.vector_index = BGEVectorIndex(self.config)
        self.builder = SemanticTreeBuilder(self.segmenter, self.vector_index, self.config)
        self.retriever = SemanticTreeRetriever(self.vector_index, self.config)

    def build_context(
        self,
        stage_input: Stage0Input,
        classification: dict[str, Any],
    ) -> tuple[SemanticTree, RetrievedSemanticContext]:
        tree = self.builder.build(stage_input, classification)
        context = self.retriever.retrieve(tree, stage_input.question, top_k=self.config.top_k)
        return tree, context
