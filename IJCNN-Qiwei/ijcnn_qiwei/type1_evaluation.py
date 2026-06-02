#!/usr/bin/env python3
"""Batch evaluation for the Type 1 semantic-hybrid Stage 0 parser."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .semantic_hybrid_parser import SemanticHybridConfig, Type1SemanticHybridParser
from .type1_preprocessing import AnswerNormalizer, TextTools


@dataclass
class Type1EvaluationConfig:
    input_path: Path = Path("../Logic_Based_Educational_Queries.json")
    output_path: Path = Path("type1_stage0_eval_results.json")
    summary_output_path: Path = Path("type1_stage0_eval_summary.json")
    api_key: str = "EMPTY"
    segmenter_model: str = "openai/Qwen2.5-1.5B-Instruct"
    segmenter_api_base: str = "http://localhost:8001/v1"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    top_k: int = 8
    include_tree: bool = False
    local_files_only: bool = False
    enable_agent_fallback: bool = True
    agent_fallback_gate_threshold: float = 0.82
    limit: int | None = None
    record_start: int = 0
    question_index: int | None = None
    stop_on_error: bool = False


@dataclass
class Type1EvalRow:
    record_id: int
    question_id: int
    question: str
    expected_answer: str
    predicted_answer: str
    normalized_expected: str
    normalized_predicted: str
    correct: bool
    gate_passed: bool
    gate_score: float
    source: str
    structured_hit_count: int
    top_match_score: float
    question_format: str
    reasoning_task: str
    solver_route: str
    error: str = ""


class Type1Stage0Evaluator:
    """Run the semantic-hybrid Type 1 parser and summarize metrics."""

    def __init__(
        self,
        config: Type1EvaluationConfig,
        parser: Type1SemanticHybridParser | None = None,
    ) -> None:
        self.config = config
        self.parser = parser or Type1SemanticHybridParser(
            SemanticHybridConfig(
                segmenter_model=config.segmenter_model,
                segmenter_api_base=config.segmenter_api_base,
                segmenter_api_key=config.api_key,
                embedding_model=config.embedding_model,
                top_k=config.top_k,
                include_tree=config.include_tree,
                local_files_only=config.local_files_only,
                enable_agent_fallback=config.enable_agent_fallback,
                agent_fallback_gate_threshold=config.agent_fallback_gate_threshold,
            )
        )

    def run(self) -> dict[str, Any]:
        records = self._load_records()
        eval_rows: list[Type1EvalRow] = []
        full_results: list[dict[str, Any]] = []
        started = time.time()

        processed = 0
        for record_id, record in enumerate(records[self.config.record_start :], start=self.config.record_start):
            question_indices = self._question_indices(record)
            for question_id in question_indices:
                if self.config.limit is not None and processed >= self.config.limit:
                    return self._finalize(records, eval_rows, full_results, started)
                processed += 1
                row, result = self._evaluate_one(record_id, record, question_id)
                eval_rows.append(row)
                full_results.append(result)
                self._print_progress(processed, row)

        return self._finalize(records, eval_rows, full_results, started)

    def _evaluate_one(self, record_id: int, record: dict[str, Any], question_id: int) -> tuple[Type1EvalRow, dict[str, Any]]:
        payload = dict(record)
        payload["_question_idx"] = question_id
        payload["record_id"] = record_id
        payload["question_id"] = question_id

        question = TextTools.clean(TextTools.safe_get(record.get("questions", []), question_id))
        expected = TextTools.clean(TextTools.safe_get(record.get("answers", []), question_id))

        try:
            result = self.parser.parse(payload)
            candidate = result.get("candidate", {})
            gate = result.get("gate", {})
            classification = result.get("classification", {})
            predicted = TextTools.clean(candidate.get("answer", ""))
            matches = result.get("semantic_matches", []) or []
            structured_hits = result.get("branch_output", {}).get("structured_attribute_hits", []) or []
            normalized_expected = normalize_for_eval(expected)
            normalized_predicted = normalize_for_eval(predicted)
            row = Type1EvalRow(
                record_id=record_id,
                question_id=question_id,
                question=question,
                expected_answer=expected,
                predicted_answer=predicted,
                normalized_expected=normalized_expected,
                normalized_predicted=normalized_predicted,
                correct=normalized_expected == normalized_predicted,
                gate_passed=bool(gate.get("passed", False)),
                gate_score=float(gate.get("score", 0.0) or 0.0),
                source=str(candidate.get("source", "")),
                structured_hit_count=len(structured_hits),
                top_match_score=float(matches[0].get("score", 0.0)) if matches else 0.0,
                question_format=str(classification.get("question_format", "")),
                reasoning_task=str(classification.get("reasoning_task", "")),
                solver_route=str(classification.get("solver_route", "")),
            )
            return row, {"eval": asdict(row), "stage0_result": result}
        except Exception as exc:
            if self.config.stop_on_error:
                raise
            row = Type1EvalRow(
                record_id=record_id,
                question_id=question_id,
                question=question,
                expected_answer=expected,
                predicted_answer="",
                normalized_expected=normalize_for_eval(expected),
                normalized_predicted="",
                correct=False,
                gate_passed=False,
                gate_score=0.0,
                source="exception",
                structured_hit_count=0,
                top_match_score=0.0,
                question_format="unknown",
                reasoning_task="unknown",
                solver_route="unknown",
                error=str(exc),
            )
            return row, {"eval": asdict(row), "stage0_result": None}

    def _finalize(
        self,
        records: list[dict[str, Any]],
        eval_rows: list[Type1EvalRow],
        full_results: list[dict[str, Any]],
        started: float,
    ) -> dict[str, Any]:
        summary = self._summarize(records, eval_rows, full_results, started)
        payload = {
            "summary": summary,
            "results": full_results,
        }
        self._save_json(self.config.output_path, payload)
        self._save_json(self.config.summary_output_path, summary)
        self._print_summary(summary)
        return payload

    def _summarize(
        self,
        records: list[dict[str, Any]],
        eval_rows: list[Type1EvalRow],
        full_results: list[dict[str, Any]],
        started: float,
    ) -> dict[str, Any]:
        total = len(eval_rows)
        correct = sum(row.correct for row in eval_rows)
        gate_passed = sum(row.gate_passed for row in eval_rows)
        structured_hit_rows = sum(row.structured_hit_count > 0 for row in eval_rows)
        errors = sum(1 for row in eval_rows if row.error)
        elapsed = time.time() - started

        return {
            "source_records": len(records),
            "evaluated_questions": total,
            "correct": correct,
            "accuracy": round(correct / total, 6) if total else 0.0,
            "gate_passed": gate_passed,
            "gate_pass_rate": round(gate_passed / total, 6) if total else 0.0,
            "structured_hit_rows": structured_hit_rows,
            "structured_hit_rate": round(structured_hit_rows / total, 6) if total else 0.0,
            "mean_top_match_score": round(sum(row.top_match_score for row in eval_rows) / total, 6) if total else 0.0,
            "mean_structured_hit_count": round(sum(row.structured_hit_count for row in eval_rows) / total, 6) if total else 0.0,
            "errors": errors,
            "error_rate": round(errors / total, 6) if total else 0.0,
            "elapsed_seconds": round(elapsed, 3),
            "questions_per_second": round(total / elapsed, 6) if elapsed > 0 else 0.0,
            "by_question_format": self._group_metrics(eval_rows, "question_format"),
            "by_reasoning_task": self._group_metrics(eval_rows, "reasoning_task"),
            "by_solver_route": self._group_metrics(eval_rows, "solver_route"),
            "by_source": self._group_metrics(eval_rows, "source"),
            "stage0_agent": self._agent_summary(full_results),
            "answer_confusion": self._answer_confusion(eval_rows),
            "config": self._redacted_config(),
        }

    def _group_metrics(self, rows: list[Type1EvalRow], field_name: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[Type1EvalRow]] = defaultdict(list)
        for row in rows:
            groups[str(getattr(row, field_name))].append(row)

        metrics: dict[str, dict[str, Any]] = {}
        for key, members in sorted(groups.items()):
            total = len(members)
            correct = sum(row.correct for row in members)
            metrics[key] = {
                "count": total,
                "correct": correct,
                "accuracy": round(correct / total, 6) if total else 0.0,
                "gate_pass_rate": round(sum(row.gate_passed for row in members) / total, 6) if total else 0.0,
                "structured_hit_rate": round(sum(row.structured_hit_count > 0 for row in members) / total, 6) if total else 0.0,
                "mean_top_match_score": round(sum(row.top_match_score for row in members) / total, 6) if total else 0.0,
                "mean_structured_hit_count": round(sum(row.structured_hit_count for row in members) / total, 6) if total else 0.0,
                "mean_gate_score": round(sum(row.gate_score for row in members) / total, 6) if total else 0.0,
            }
        return metrics

    def _answer_confusion(self, rows: list[Type1EvalRow]) -> dict[str, dict[str, int]]:
        confusion: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            confusion[row.normalized_expected][row.normalized_predicted or "<blank>"] += 1
        return {expected: dict(counter.most_common()) for expected, counter in sorted(confusion.items())}

    def _redacted_config(self) -> dict[str, Any]:
        data = asdict(self.config)
        if data.get("api_key"):
            data["api_key"] = "<redacted>"
        data["input_path"] = str(self.config.input_path)
        data["output_path"] = str(self.config.output_path)
        data["summary_output_path"] = str(self.config.summary_output_path)
        return data

    def _agent_summary(self, full_results: list[dict[str, Any]]) -> dict[str, Any]:
        states: Counter[str] = Counter()
        for result in full_results:
            agent = (result.get("stage0_result") or {}).get("metadata", {}).get("stage0_agent", {})
            enabled = bool(agent.get("enabled", False))
            used = bool(agent.get("used", False))
            reason = str(agent.get("reason", "none"))
            states[f"enabled={enabled}|used={used}|reason={reason}"] += 1
        return {"state_counts": dict(states.most_common())}

    def _question_indices(self, record: dict[str, Any]) -> list[int]:
        questions = record.get("questions", [])
        if not isinstance(questions, list):
            questions = [questions]
        if self.config.question_index is not None:
            if 0 <= self.config.question_index < len(questions):
                return [self.config.question_index]
            return []
        return list(range(len(questions)))

    def _load_records(self) -> list[dict[str, Any]]:
        data = json.loads(self.config.input_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Expected Type 1 dataset JSON to be a list of records.")
        return data

    def _print_progress(self, processed: int, row: Type1EvalRow) -> None:
        status = "OK" if row.correct else "MISS"
        print(
            f"[{processed}] {status} r={row.record_id} q={row.question_id} "
            f"expected={row.normalized_expected or '<blank>'} "
            f"pred={row.normalized_predicted or '<blank>'} "
            f"source={row.source} match={row.top_match_score:.3f} hits={row.structured_hit_count} gate={row.gate_score:.3f}"
        )

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print()
        print("Type 1 Stage 0 Evaluation Summary")
        print(f"- evaluated_questions: {summary['evaluated_questions']}")
        print(f"- accuracy: {summary['accuracy']}")
        print(f"- gate_pass_rate: {summary['gate_pass_rate']}")
        print(f"- structured_hit_rate: {summary['structured_hit_rate']}")
        print(f"- mean_top_match_score: {summary['mean_top_match_score']}")
        print(f"- error_rate: {summary['error_rate']}")
        print(f"- saved results: {self.config.output_path}")
        print(f"- saved summary: {self.config.summary_output_path}")

    @staticmethod
    def _save_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_for_eval(answer: Any) -> str:
    normalized = AnswerNormalizer.normalize(answer)
    if normalized == "Unknown":
        return "Uncertain"
    if normalized in {"True"}:
        return "Yes"
    if normalized in {"False"}:
        return "No"
    if normalized.startswith("Option_"):
        return normalized.replace("Option_", "")
    return normalized


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Type 1 Stage 0 parser on labeled data.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--output", type=Path, default=Path("type1_stage0_eval_results.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("type1_stage0_eval_summary.json"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--segmenter-model", default="openai/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--segmenter-api-base", default="http://localhost:8001/v1")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--include-tree", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    agent_group = parser.add_mutually_exclusive_group()
    agent_group.add_argument("--enable-agent-fallback", dest="enable_agent_fallback", action="store_true", default=True)
    agent_group.add_argument("--disable-agent-fallback", dest="enable_agent_fallback", action="store_false")
    parser.add_argument("--agent-fallback-gate-threshold", type=float, default=0.82)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--question-index", type=int)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Type1EvaluationConfig:
    return Type1EvaluationConfig(
        input_path=args.input,
        output_path=args.output,
        summary_output_path=args.summary_output,
        api_key=args.api_key,
        segmenter_model=args.segmenter_model,
        segmenter_api_base=args.segmenter_api_base,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        include_tree=args.include_tree,
        local_files_only=args.local_files_only,
        enable_agent_fallback=args.enable_agent_fallback,
        agent_fallback_gate_threshold=args.agent_fallback_gate_threshold,
        limit=args.limit,
        record_start=args.record_start,
        question_index=args.question_index,
        stop_on_error=args.stop_on_error,
    )


def main() -> None:
    Type1Stage0Evaluator(config_from_args(parse_args())).run()


if __name__ == "__main__":
    main()
