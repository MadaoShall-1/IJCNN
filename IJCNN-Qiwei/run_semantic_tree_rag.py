#!/usr/bin/env python3
"""Build and print semantic-tree RAG context for a Type 1 dataset question."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ijcnn_qiwei.common import Stage0Input
from ijcnn_qiwei.semantic_tree_rag import SemanticRAGConfig, SemanticTreeRAG
from ijcnn_qiwei.type1_preprocessing import TextTools, Type1QuestionClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Type 1 semantic tree RAG context.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--segmenter-model", default="openai/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--segmenter-api-base", default="http://localhost:8001/v1")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_stage_input(path: Path, record_index: int, question_index: int) -> Stage0Input:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a Type 1 dataset JSON list.")
    record = data[record_index]
    questions = record.get("questions", [])
    answers = record.get("answers", [])
    return Stage0Input(
        question=TextTools.clean(TextTools.safe_get(questions, question_index)),
        premises_nl=[TextTools.clean(item) for item in record.get("premises-NL", [])],
        premises_fol=[TextTools.clean(item) for item in record.get("premises-FOL", [])],
        expected_answer=TextTools.clean(TextTools.safe_get(answers, question_index)),
        record_id=record_index,
        question_id=question_index,
    )


def main() -> None:
    args = parse_args()
    stage_input = load_stage_input(args.input, args.record_index, args.question_index)
    classifier = Type1QuestionClassifier()
    classification = classifier.classify(
        question=stage_input.question,
        answer=stage_input.expected_answer,
        premises_nl=stage_input.premises_nl,
        premises_fol=stage_input.premises_fol,
    )
    rag = SemanticTreeRAG(
        SemanticRAGConfig(
            segmenter_model=args.segmenter_model,
            segmenter_api_base=args.segmenter_api_base,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            local_files_only=args.local_files_only,
        )
    )
    tree, context = rag.build_context(stage_input, classification)
    payload = {
        "classification": classification,
        "context": context.__dict__,
        "tree": tree.to_dict(),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
