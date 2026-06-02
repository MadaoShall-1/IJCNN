#!/usr/bin/env python3
"""Batch evaluation for the Type 1 Stage 0-3 pipeline."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .semantic_hybrid_parser import SemanticHybridConfig
from .type1_evaluation import make_jsonable, normalize_for_eval
from .type1_pipeline import Type1MultiStagePipeline, Type1PipelineConfig
from .type1_preprocessing import TextTools


@dataclass
class Type1PipelineEvaluationConfig:
    input_path: Path = Path("../Logic_Based_Educational_Queries.json")
    output_path: Path = Path("type1_pipeline_eval_results.json")
    summary_output_path: Path = Path("type1_pipeline_eval_summary.json")
    api_key: str = "EMPTY"
    segmenter_model: str = "openai/Qwen2.5-1.5B-Instruct"
    segmenter_api_base: str = "http://localhost:8001/v1"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    top_k: int = 8
    include_tree: bool = False
    include_evidence_graph: bool = False
    local_files_only: bool = False
    enable_agent_fallback: bool = True
    agent_fallback_gate_threshold: float = 0.82
    max_forward_steps: int = 8
    stage2_confidence_threshold: float = 0.58
    stage3_pass_threshold: float = 0.66
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
    limit: int | None = None
    record_start: int = 0
    question_index: int | None = None
    stop_on_error: bool = False


@dataclass
class Type1PipelineEvalRow:
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
    stage0_source: str
    stage2_source: str
    stage2_confidence: float
    fact_count: int
    rule_count: int
    derived_fact_count: int
    question_format: str
    reasoning_task: str
    solver_route: str
    error: str = ""


class Type1PipelineEvaluator:
    def __init__(
        self,
        config: Type1PipelineEvaluationConfig,
        pipeline: Type1MultiStagePipeline | None = None,
    ) -> None:
        self.config = config
        stage0_config = SemanticHybridConfig(
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
        pipeline_config = Type1PipelineConfig(
            stage0=stage0_config,
            max_forward_steps=config.max_forward_steps,
            stage2_confidence_threshold=config.stage2_confidence_threshold,
            stage3_pass_threshold=config.stage3_pass_threshold,
            include_evidence_graph=config.include_evidence_graph,
            enable_causal_world_model=config.enable_causal_world_model,
            causal_max_steps=config.causal_max_steps,
            causal_top_k=config.causal_top_k,
            causal_yes_threshold=config.causal_yes_threshold,
            causal_no_threshold=config.causal_no_threshold,
            causal_yes_probability_threshold=config.causal_yes_probability_threshold,
            causal_no_probability_threshold=config.causal_no_probability_threshold,
            premise_coupling_top_k=config.premise_coupling_top_k,
            recurrent_planning_rounds=config.recurrent_planning_rounds,
            belief_confidence_threshold=config.belief_confidence_threshold,
            belief_convergence_delta=config.belief_convergence_delta,
        )
        self.pipeline = pipeline or Type1MultiStagePipeline(pipeline_config)

    def run(self) -> dict[str, Any]:
        records = self._load_records()
        rows: list[Type1PipelineEvalRow] = []
        results: list[dict[str, Any]] = []
        started = time.time()
        processed = 0
        for record_id, record in enumerate(records[self.config.record_start :], start=self.config.record_start):
            for question_id in self._question_indices(record):
                if self.config.limit is not None and processed >= self.config.limit:
                    return self._finalize(records, rows, results, started)
                processed += 1
                row, result = self._evaluate_one(record_id, record, question_id)
                rows.append(row)
                results.append(result)
                self._print_progress(processed, row)
        return self._finalize(records, rows, results, started)

    def _evaluate_one(self, record_id: int, record: dict[str, Any], question_id: int) -> tuple[Type1PipelineEvalRow, dict[str, Any]]:
        payload = dict(record)
        payload["_question_idx"] = question_id
        payload["record_id"] = record_id
        payload["question_id"] = question_id
        question = TextTools.clean(TextTools.safe_get(record.get("questions", []), question_id))
        expected = TextTools.clean(TextTools.safe_get(record.get("answers", []), question_id))
        try:
            result = self.pipeline.run(payload)
            candidate = result.get("candidate", {})
            gate = result.get("gate", {})
            classification = result.get("classification", {})
            stage1_summary = result.get("stage1", {}).get("evidence_graph_summary", {})
            fusion = result.get("stage3", {}).get("fusion", {})
            stage2_candidate = result.get("stage2", {}).get("candidate", {})
            predicted = TextTools.clean(candidate.get("answer", ""))
            normalized_expected = normalize_for_eval(expected)
            normalized_predicted = normalize_for_eval(predicted)
            row = Type1PipelineEvalRow(
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
                stage0_source=str(fusion.get("stage0_source", "")),
                stage2_source=str(fusion.get("stage2_source", stage2_candidate.get("source", ""))),
                stage2_confidence=float(fusion.get("stage2_confidence", stage2_candidate.get("confidence", 0.0)) or 0.0),
                fact_count=int(stage1_summary.get("fact_count", 0) or 0),
                rule_count=int(stage1_summary.get("rule_count", 0) or 0),
                derived_fact_count=int(stage1_summary.get("derived_fact_count", 0) or 0),
                question_format=str(classification.get("question_format", "")),
                reasoning_task=str(classification.get("reasoning_task", "")),
                solver_route=str(classification.get("solver_route", "")),
            )
            return row, {"eval": asdict(row), "pipeline_result": result}
        except Exception as exc:
            if self.config.stop_on_error:
                raise
            row = Type1PipelineEvalRow(
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
                stage0_source="",
                stage2_source="",
                stage2_confidence=0.0,
                fact_count=0,
                rule_count=0,
                derived_fact_count=0,
                question_format="unknown",
                reasoning_task="unknown",
                solver_route="unknown",
                error=str(exc),
            )
            return row, {"eval": asdict(row), "pipeline_result": None}

    def _finalize(
        self,
        records: list[dict[str, Any]],
        rows: list[Type1PipelineEvalRow],
        results: list[dict[str, Any]],
        started: float,
    ) -> dict[str, Any]:
        summary = self._summarize(records, rows, results, started)
        payload = {"summary": summary, "results": results}
        self._save_json(self.config.output_path, payload)
        self._save_json(self.config.summary_output_path, summary)
        self._print_summary(summary)
        return payload

    def _summarize(
        self,
        records: list[dict[str, Any]],
        rows: list[Type1PipelineEvalRow],
        results: list[dict[str, Any]],
        started: float,
    ) -> dict[str, Any]:
        total = len(rows)
        correct = sum(row.correct for row in rows)
        elapsed = time.time() - started
        return {
            "source_records": len(records),
            "evaluated_questions": total,
            "correct": correct,
            "accuracy": round(correct / total, 6) if total else 0.0,
            "gate_pass_rate": round(sum(row.gate_passed for row in rows) / total, 6) if total else 0.0,
            "mean_gate_score": round(sum(row.gate_score for row in rows) / total, 6) if total else 0.0,
            "mean_stage2_confidence": round(sum(row.stage2_confidence for row in rows) / total, 6) if total else 0.0,
            "mean_derived_fact_count": round(sum(row.derived_fact_count for row in rows) / total, 6) if total else 0.0,
            "errors": sum(1 for row in rows if row.error),
            "error_rate": round(sum(1 for row in rows if row.error) / total, 6) if total else 0.0,
            "elapsed_seconds": round(elapsed, 3),
            "questions_per_second": round(total / elapsed, 6) if elapsed > 0 else 0.0,
            "by_source": self._group_metrics(rows, "source"),
            "by_stage0_source": self._group_metrics(rows, "stage0_source"),
            "by_stage2_source": self._group_metrics(rows, "stage2_source"),
            "by_question_format": self._group_metrics(rows, "question_format"),
            "by_reasoning_task": self._group_metrics(rows, "reasoning_task"),
            "by_solver_route": self._group_metrics(rows, "solver_route"),
            "stage0_agent": self._stage0_agent_summary(results),
            "answer_confusion": self._answer_confusion(rows),
            "config": self._redacted_config(),
        }

    def _group_metrics(self, rows: list[Type1PipelineEvalRow], field_name: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[Type1PipelineEvalRow]] = defaultdict(list)
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
                "mean_gate_score": round(sum(row.gate_score for row in members) / total, 6) if total else 0.0,
                "mean_stage2_confidence": round(sum(row.stage2_confidence for row in members) / total, 6) if total else 0.0,
                "mean_derived_fact_count": round(sum(row.derived_fact_count for row in members) / total, 6) if total else 0.0,
            }
        return metrics

    def _stage0_agent_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        states: Counter[str] = Counter()
        for result in results:
            agent = (
                (result.get("pipeline_result") or {})
                .get("stage0", {})
                .get("metadata", {})
                .get("stage0_agent", {})
            )
            states[f"enabled={bool(agent.get('enabled', False))}|used={bool(agent.get('used', False))}|reason={agent.get('reason', 'none')}"] += 1
        return {"state_counts": dict(states.most_common())}

    def _answer_confusion(self, rows: list[Type1PipelineEvalRow]) -> dict[str, dict[str, int]]:
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

    def _print_progress(self, processed: int, row: Type1PipelineEvalRow) -> None:
        status = "OK" if row.correct else "MISS"
        print(
            f"[{processed}] {status} r={row.record_id} q={row.question_id} "
            f"expected={row.normalized_expected or '<blank>'} pred={row.normalized_predicted or '<blank>'} "
            f"source={row.source} stage2={row.stage2_confidence:.3f} derived={row.derived_fact_count} gate={row.gate_score:.3f}"
        )

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print()
        print("Type 1 Stage 0-3 Pipeline Evaluation Summary")
        print(f"- evaluated_questions: {summary['evaluated_questions']}")
        print(f"- accuracy: {summary['accuracy']}")
        print(f"- gate_pass_rate: {summary['gate_pass_rate']}")
        print(f"- mean_stage2_confidence: {summary['mean_stage2_confidence']}")
        print(f"- mean_derived_fact_count: {summary['mean_derived_fact_count']}")
        print(f"- error_rate: {summary['error_rate']}")
        print(f"- saved results: {self.config.output_path}")
        print(f"- saved summary: {self.config.summary_output_path}")

    @staticmethod
    def _save_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Type 1 Stage 0-3 pipeline on labeled data.")
    parser.add_argument("--input", type=Path, default=Path("../Logic_Based_Educational_Queries.json"))
    parser.add_argument("--output", type=Path, default=Path("type1_pipeline_eval_results.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("type1_pipeline_eval_summary.json"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--segmenter-model", default="openai/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--segmenter-api-base", default="http://localhost:8001/v1")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--include-tree", action="store_true")
    parser.add_argument("--include-evidence-graph", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    agent_group = parser.add_mutually_exclusive_group()
    agent_group.add_argument("--enable-agent-fallback", dest="enable_agent_fallback", action="store_true", default=True)
    agent_group.add_argument("--disable-agent-fallback", dest="enable_agent_fallback", action="store_false")
    parser.add_argument("--agent-fallback-gate-threshold", type=float, default=0.82)
    parser.add_argument("--max-forward-steps", type=int, default=8)
    parser.add_argument("--stage2-confidence-threshold", type=float, default=0.58)
    parser.add_argument("--stage3-pass-threshold", type=float, default=0.66)
    causal_group = parser.add_mutually_exclusive_group()
    causal_group.add_argument("--enable-causal-world-model", dest="enable_causal_world_model", action="store_true", default=True)
    causal_group.add_argument("--disable-causal-world-model", dest="enable_causal_world_model", action="store_false")
    parser.add_argument("--causal-max-steps", type=int, default=6)
    parser.add_argument("--causal-top-k", type=int, default=4)
    parser.add_argument("--causal-yes-threshold", type=float, default=0.35)
    parser.add_argument("--causal-no-threshold", type=float, default=-0.35)
    parser.add_argument("--causal-yes-probability-threshold", type=float, default=0.45)
    parser.add_argument("--causal-no-probability-threshold", type=float, default=0.45)
    parser.add_argument("--premise-coupling-top-k", type=int, default=8)
    parser.add_argument("--recurrent-planning-rounds", type=int, default=4)
    parser.add_argument("--belief-confidence-threshold", type=float, default=0.72)
    parser.add_argument("--belief-convergence-delta", type=float, default=0.035)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--question-index", type=int)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Type1PipelineEvaluationConfig:
    return Type1PipelineEvaluationConfig(
        input_path=args.input,
        output_path=args.output,
        summary_output_path=args.summary_output,
        api_key=args.api_key,
        segmenter_model=args.segmenter_model,
        segmenter_api_base=args.segmenter_api_base,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        include_tree=args.include_tree,
        include_evidence_graph=args.include_evidence_graph,
        local_files_only=args.local_files_only,
        enable_agent_fallback=args.enable_agent_fallback,
        agent_fallback_gate_threshold=args.agent_fallback_gate_threshold,
        max_forward_steps=args.max_forward_steps,
        stage2_confidence_threshold=args.stage2_confidence_threshold,
        stage3_pass_threshold=args.stage3_pass_threshold,
        enable_causal_world_model=args.enable_causal_world_model,
        causal_max_steps=args.causal_max_steps,
        causal_top_k=args.causal_top_k,
        causal_yes_threshold=args.causal_yes_threshold,
        causal_no_threshold=args.causal_no_threshold,
        causal_yes_probability_threshold=args.causal_yes_probability_threshold,
        causal_no_probability_threshold=args.causal_no_probability_threshold,
        premise_coupling_top_k=args.premise_coupling_top_k,
        recurrent_planning_rounds=args.recurrent_planning_rounds,
        belief_confidence_threshold=args.belief_confidence_threshold,
        belief_convergence_delta=args.belief_convergence_delta,
        limit=args.limit,
        record_start=args.record_start,
        question_index=args.question_index,
        stop_on_error=args.stop_on_error,
    )


def main() -> None:
    Type1PipelineEvaluator(config_from_args(parse_args())).run()


if __name__ == "__main__":
    main()
